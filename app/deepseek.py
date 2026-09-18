"""DeepSeek cleanup: raw OCR markdown -> a clean, question-only paper.

The OCR text of a photographed exam paper contains both the printed questions
and the student's handwritten answers, which the model happily reads as if they
were part of the text (``则 $x+y=\\underline{5}$`` where the 5 was written by
hand). This module asks DeepSeek to strip the handwriting and return the blank
paper.

Figures are the fragile part: an LLM asked to re-emit an ``<img src=...>`` tag
or a long ImgBB URL may quietly corrupt it. So every figure is swapped for an
opaque marker (``[[FIG3-Q13]]`` — "figure 3 belongs to question 13") before the
call, and swapped back afterwards. Missing markers are re-attached rather than
lost.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import requests

from . import config

# Any figure reference, HTML or markdown, local path or remote URL.
_ANY_IMG_RE = re.compile(r"<img\s[^>]*?>|!\[[^\]]*\]\([^)]*\)", re.IGNORECASE)
# A "*[Q13 附图]*" label sitting immediately before a figure.
_LABEL_BEFORE_RE = re.compile(r"\*\[Q(\d+)\s*附图\]\*\s*\Z")
# Our marker, matched leniently in case the model adds whitespace.
_TOKEN_RE = re.compile(r"\[\[\s*FIG\s*(\d+)\s*(?:-\s*Q\s*(\d+)\s*)?\]\]", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"<!--\s*/?page\s*\d+\s*-->\s*", re.IGNORECASE)
_EMPTY_DIV_RE = re.compile(r"<div[^>]*?>\s*</div>", re.IGNORECASE)

SYSTEM_PROMPT = """\
You clean up OCR output from photographed exam papers.

Each photo contains PRINTED exam text and a student's HANDWRITTEN answers,
working and notes. Produce the clean, blank question paper: the questions only.

REMOVE (this is handwritten, not part of the paper):
- answers written into blanks or after \\underline{...}
- any number, letter, word or symbol filling an answer slot
- circled or ticked options, solution steps, working, margin notes, scribbles
- cross-outs, teacher marks and scores

KEEP (this is printed):
- question stems, printed options (A/B/C/D), instructions, headings, marks
- printed values that are genuinely part of the question

Where an answer was written in, restore an empty blank: \\underline{\\hspace{2em}}
  e.g.  "则 $x+y=\\underline{5}$"      ->  "则 $x+y=\\underline{\\hspace{2em}}$"
  e.g.  "圆心角是 120 度"                ->  "圆心角是 \\underline{\\hspace{2em}} 度"

RULES
1. Copy printed text EXACTLY. Never rephrase, translate, summarise, reorder or
   renumber the questions.
2. Reproduce all LaTeX unchanged: $...$, \\(...\\), \\[...\\], \\begin{...}...\\end{...}.
3. Figures appear as markers such as [[FIG3-Q13]], meaning figure 3 belongs to
   question 13. Move each marker onto its own line immediately after the
   question it names. Copy the marker text character-for-character.
   Every marker you are given must appear exactly once: never invent, delete,
   duplicate, reorder or renumber one.
4. Drop OCR noise: stray single characters, orphaned punctuation, fragments like
   "n", "~", "nm", and empty <div> wrappers.
5. Remove page markers such as <!-- page 2 -->.
6. Output GitHub-flavoured Markdown only. No preamble, no commentary, no code
   fences, no answers, no explanations.
"""


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API cannot be used or returned nothing usable."""


def log(message: str) -> None:
    print(f"[deepseek] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Figure markers
# --------------------------------------------------------------------------


def protect_figures(markdown: str) -> tuple[str, dict[str, dict[str, str]]]:
    """Swap every figure (and its label) for an opaque marker.

    Returns the rewritten markdown and ``{marker: {"label": ..., "ref": ...}}``.
    """
    regions: list[tuple[int, int, str, str, str]] = []

    for index, match in enumerate(_ANY_IMG_RE.finditer(markdown), start=1):
        ref = match.group(0)
        start, end = match.start(), match.end()
        label = ""

        label_match = _LABEL_BEFORE_RE.search(markdown[:start])
        if label_match:
            start = label_match.start()
            label = label_match.group(0).strip()

        question = label_match.group(1) if label_match else None
        marker = f"[[FIG{index}-Q{question}]]" if question else f"[[FIG{index}]]"
        regions.append((start, end, marker, label, ref))

    if not regions:
        return markdown, {}

    mapping: dict[str, dict[str, str]] = {}
    out = markdown
    for start, end, marker, label, ref in reversed(regions):
        out = out[:start] + marker + out[end:]
        mapping[marker] = {"label": label, "ref": ref}

    return _EMPTY_DIV_RE.sub("", out), mapping


def restore_figures(
    markdown: str, mapping: dict[str, dict[str, str]]
) -> tuple[str, list[str]]:
    """Put the real figures back. Returns the markdown and any markers the model dropped."""
    if not mapping:
        return markdown, []

    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        marker = f"[[FIG{match.group(1)}" + (f"-Q{match.group(2)}" if match.group(2) else "") + "]]"
        # The model may have rewritten the question number; match on figure id.
        resolved = marker if marker in mapping else _by_figure_id(mapping, match.group(1))
        if resolved is None:
            return ""
        seen.add(resolved)
        entry = mapping[resolved]
        if entry["label"]:
            return f"{entry['label']}\n\n{entry['ref']}"
        return entry["ref"]

    out = _TOKEN_RE.sub(replace, markdown)

    missing = [marker for marker in mapping if marker not in seen]
    if missing:
        out = out.rstrip() + "\n\n"
        for marker in missing:
            entry = mapping[marker]
            prefix = f"{entry['label']}\n\n" if entry["label"] else ""
            out += f"{prefix}{entry['ref']}\n\n"

    return _EMPTY_DIV_RE.sub("", out).strip() + "\n", missing


def _by_figure_id(mapping: dict[str, dict[str, str]], figure_id: str) -> str | None:
    for marker in mapping:
        if marker.startswith(f"[[FIG{figure_id}-") or marker == f"[[FIG{figure_id}]]":
            return marker
    return None


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def split_into_chunks(markdown: str, limit: int | None = None) -> list[str]:
    """Split on page markers, then greedily group, then hard-wrap if needed."""
    limit = limit or config.DEEPSEEK_MAX_CHARS

    if len(markdown) <= limit:
        return [markdown]

    pieces = [piece for piece in _PAGE_MARKER_RE.split(markdown) if piece.strip()]
    if len(pieces) <= 1:
        pieces = [piece for piece in re.split(r"\n\s*\n", markdown) if piece.strip()]

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > limit:
            chunks.append(current.strip())
            current = ""
        if len(piece) > limit:  # single oversized piece: hard-wrap it
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(piece), limit):
                chunks.append(piece[start : start + limit].strip())
            continue
        current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        chunks.append(current.strip())

    return chunks or [markdown]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }


def list_models() -> list[str]:
    """Model ids available to this account."""
    if not config.DEEPSEEK_API_KEY:
        raise DeepSeekError("DEEPSEEK_API_KEY is not set")
    try:
        response = requests.get(
            f"{config.DEEPSEEK_BASE_URL}/models", headers=_headers(), timeout=30
        )
    except requests.RequestException as exc:
        raise DeepSeekError(f"could not reach DeepSeek: {exc}") from exc
    if response.status_code == 401:
        raise DeepSeekError("DeepSeek rejected the API key (401 Unauthorized)")
    if not response.ok:
        raise DeepSeekError(f"DeepSeek returned HTTP {response.status_code}: {response.text[:200]}")
    try:
        return [item["id"] for item in response.json().get("data", [])]
    except (ValueError, KeyError, TypeError) as exc:
        raise DeepSeekError(f"unexpected /models response: {exc}") from exc


_model_cache: str | None = None


def resolve_model(refresh: bool = False) -> str:
    """Pick the model to use, preferring DEEPSEEK_MODEL when the account has it."""
    global _model_cache
    if _model_cache and not refresh:
        return _model_cache

    wanted = config.DEEPSEEK_MODEL
    try:
        available = list_models()
    except DeepSeekError as exc:
        log(f"could not list models ({exc}); trying {wanted!r} anyway")
        _model_cache = wanted
        return wanted

    if wanted in available:
        _model_cache = wanted
    elif config.DEEPSEEK_FALLBACK_MODEL in available:
        log(f"{wanted!r} is not available on this account; using {config.DEEPSEEK_FALLBACK_MODEL!r}")
        _model_cache = config.DEEPSEEK_FALLBACK_MODEL
    else:
        _model_cache = available[0] if available else config.DEEPSEEK_FALLBACK_MODEL
        log(f"{wanted!r} is not available; using {_model_cache!r}")

    log(f"available models: {', '.join(available) or '(none reported)'}")
    return _model_cache


def _chat(markdown: str, model: str) -> tuple[str, dict[str, int]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        "temperature": 0,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{config.DEEPSEEK_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=config.DEEPSEEK_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise DeepSeekError(f"DeepSeek request failed: {exc}") from exc

    if response.status_code == 401:
        raise DeepSeekError("DeepSeek rejected the API key (401 Unauthorized)")
    if not response.ok:
        raise DeepSeekError(f"DeepSeek returned HTTP {response.status_code}: {response.text[:300]}")

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise DeepSeekError(f"unexpected response from DeepSeek: {exc}") from exc

    if not content or not content.strip():
        raise DeepSeekError("DeepSeek returned an empty result")

    return content.strip(), body.get("usage", {}) or {}


# --------------------------------------------------------------------------
# The operation
# --------------------------------------------------------------------------


def polish_markdown(markdown: str) -> tuple[str, dict[str, Any]]:
    """Clean ``markdown`` into a question-only paper.

    Returns ``(cleaned_markdown, info)`` where info carries the model used, token
    usage and any figures the model dropped.
    """
    if not config.DEEPSEEK_API_KEY:
        raise DeepSeekError(
            "DEEPSEEK_API_KEY is not set — add it to .env and restart the app"
        )
    if not markdown.strip():
        raise DeepSeekError("there is nothing to clean up yet")

    model = resolve_model()
    chunks = split_into_chunks(markdown)
    log(f"cleaning {len(markdown):,} chars in {len(chunks)} chunk(s) with {model}")

    cleaned_parts: list[str] = []
    dropped: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for number, chunk in enumerate(chunks, start=1):
        protected, mapping = protect_figures(chunk)
        answer, chunk_usage = _chat(protected, model)
        restored, missing = restore_figures(answer, mapping)
        restored = _PAGE_MARKER_RE.sub("", restored).strip()
        cleaned_parts.append(restored)
        dropped.extend(missing)
        for key in usage:
            usage[key] += int(chunk_usage.get(key) or 0)
        log(f"chunk {number}/{len(chunks)}: {len(mapping)} figure(s)"
            + (f", {len(missing)} recovered after the model dropped them" if missing else ""))

    info: dict[str, Any] = {
        "model": model,
        "chunks": len(chunks),
        "usage": usage,
        "figures_total": sum(len(protect_figures(c)[1]) for c in chunks),
        "figures_recovered": len(dropped),
    }
    return "\n\n".join(part for part in cleaned_parts if part).strip() + "\n", info
