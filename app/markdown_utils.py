"""Markdown helpers.

Everything that understands the shape of OCR output lives here: normalising raw
PaddleOCR-VL markdown, finding figure references, moving figures underneath the
question they belong to, and merging per-page output into one document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "gif", "webp", "bmp")
_EXT_ALT = "|".join(IMAGE_EXTENSIONS)

# --- regexes -----------------------------------------------------------------

# `<img ... />` -> `<img ...>`; stray `</img>` is dropped separately.
_IMG_SELF_CLOSE_RE = re.compile(r"<img\s([^>]*?)\s*/?>", re.IGNORECASE)
_IMG_CLOSE_RE = re.compile(r"</img\s*>", re.IGNORECASE)
# A lone backslash surrounded by whitespace is an OCR'd LaTeX line break.
_LATEX_LINEBREAK_RE = re.compile(r"(?<=\s)\\(?=\s)")
# Any figure reference, either HTML or markdown form, in document order.
_ANY_IMG_RE = re.compile(r"<img\s[^>]*?>|!\[[^\]]*\]\([^)]*\)", re.IGNORECASE)
_IMG_SRC_RE = re.compile(r"""<img[^>]*?\ssrc=["']([^"']+)["']""", re.IGNORECASE)
_MD_SRC_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# Rewrite the src of an existing tag while keeping its other attributes.
_TAG_SRC_RE = re.compile(r"""(<img[^>]*?\ssrc=["'])([^"']+)(["'])""", re.IGNORECASE)
_MD_IMG_SRC_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
# A bare image filename anywhere in the text.
_FILENAME_RE = re.compile(rf"""([^/\\"'\s()]+\.(?:{_EXT_ALT}))""", re.IGNORECASE)
# The label we insert in front of a figure.
# The old form was a standalone line:  *[Q13 附图]*
# The current form carries the label in the image's alt text:  ![Q13 附图](url)
# and numbers them when a question has more than one figure.
_LABEL_LINE_RE = re.compile(r"^[ \t]*\*\[Q\d+\s*附图\d*\]\*[ \t]*\n?", re.MULTILINE)
ALT_QUESTION_RE = re.compile(r"!\[\s*Q(\d+)\s*附图", re.IGNORECASE)
_ALT_LABEL_RE = re.compile(r"!\[\s*Q\d+\s*附图\d*\s*\]\(", re.IGNORECASE)
# A standalone label immediately followed by the figure it belongs to.
_BAKED_LABEL_RE = re.compile(
    r"\*\[Q(\d+)\s*附图\d*\]\*\s*\n*\s*(<img\s[^>]*?>|!\[[^\]]*\]\([^)]*\))",
    re.IGNORECASE,
)
# "13." / "13、" / "13．" / "13)" / "13," starting a line.
_QUESTION_RE = re.compile(r"^[ \t]*(\d{1,2})[ \t]*[.、,．)]", re.MULTILINE)
_REMOTE_REF_RE = re.compile(r"^(?:https?:|data:)", re.IGNORECASE)


# --- normalisation -----------------------------------------------------------


def normalize_img_tags(md: str) -> str:
    """Self-closing `<img/>` becomes `<img>`, stray `</img>` is removed."""
    md = _IMG_SELF_CLOSE_RE.sub(r"<img \1>", md)
    return _IMG_CLOSE_RE.sub("", md)


def normalize_latex(md: str) -> str:
    """Turn a lone backslash between whitespace into a LaTeX line break."""
    return _LATEX_LINEBREAK_RE.sub(r"\\\\", md)


def normalize_image_refs(md: str) -> str:
    """Point every figure reference at ``images/<basename>``.

    PaddleOCR-VL writes things like ``imgs/xxx.jpg`` or an absolute scratch
    path; the web app always serves figures from ``images/``.
    """

    def _tag(match: re.Match[str]) -> str:
        return f"{match.group(1)}images/{Path(match.group(2)).name}{match.group(3)}"

    def _md_img(match: re.Match[str]) -> str:
        return f"{match.group(1)}images/{Path(match.group(2)).name}{match.group(3)}"

    md = _TAG_SRC_RE.sub(_tag, md)
    return _MD_IMG_SRC_RE.sub(_md_img, md)


def normalize_markdown(md: str) -> str:
    """Apply every markdown fix-up, in order."""
    if not md:
        return ""
    return normalize_image_refs(normalize_latex(normalize_img_tags(md)))


# --- figure references -------------------------------------------------------


def iter_image_refs(md: str):
    """Yield ``(filename, full_reference_text)`` in document order."""
    for match in _ANY_IMG_RE.finditer(md):
        text = match.group(0)
        src = _IMG_SRC_RE.search(text) or _MD_SRC_RE.search(text)
        if not src:
            continue
        ref = src.group(1).strip()
        if _REMOTE_REF_RE.match(ref):
            continue
        yield Path(ref).name, text


def extract_filenames(md: str) -> list[str]:
    """Ordered, de-duplicated basenames of every local figure in ``md``."""
    names: list[str] = []
    seen: set[str] = set()
    for name, _ in iter_image_refs(md):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def find_all_image_refs(md: str) -> list[str]:
    """Raw reference strings (local ones only), de-duplicated, in order."""
    refs: list[str] = []
    seen: set[str] = set()
    for match in _ANY_IMG_RE.finditer(md):
        text = match.group(0)
        src = _IMG_SRC_RE.search(text) or _MD_SRC_RE.search(text)
        if not src:
            continue
        ref = src.group(1).strip()
        if not _REMOTE_REF_RE.match(ref) and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def filenames_in_text(md: str) -> list[str]:
    """Every image-looking filename mentioned anywhere in ``md``."""
    names: list[str] = []
    seen: set[str] = set()
    for match in _FILENAME_RE.finditer(md):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


# --- questions ---------------------------------------------------------------


def detect_questions(md: str) -> list[str]:
    """Question numbers found at the start of a line, in order, de-duplicated."""
    numbers: list[str] = []
    seen: set[str] = set()
    for match in _QUESTION_RE.finditer(md):
        number = str(int(match.group(1)))
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def normalize_question(value: object) -> str:
    """``"Q13"`` / ``"13."`` / ``13`` -> ``"13"``; anything unusable -> ``""``."""
    if value is None:
        return ""
    text = str(value).strip().lstrip("Qq").strip().rstrip(".、,．)")
    return str(int(text)) if text.isdigit() else ""


# --- labels ------------------------------------------------------------------


def strip_labels(md: str) -> str:
    """Remove every figure label, leaving the image itself in place.

    Handles both the old standalone ``*[Q13 附图]*`` line and the current form
    where the label sits in the image's alt text (``![Q13 附图](url)`` becomes
    ``![](url)``). Stripping is what lets a mapping be re-applied cleanly.
    """
    md = _LABEL_LINE_RE.sub("", md)
    return _ALT_LABEL_RE.sub("![](", md)


def image_ref_url(text: str) -> str:
    """Pull the URL/path out of an ``<img>`` tag or a markdown image."""
    match = _IMG_SRC_RE.search(text) or _MD_SRC_RE.search(text)
    return match.group(1).strip() if match else ""


def figure_label(question: str, index: int, total: int) -> str:
    """``Q13 附图`` for a lone figure, ``Q13 附图1`` / ``附图2`` when several."""
    return f"Q{question} 附图{index}" if total > 1 else f"Q{question} 附图"


def _remove_images(md: str, only: set[str]) -> str:
    """Delete the references whose filename is in ``only``, keep the rest."""

    def repl(match: re.Match[str]) -> str:
        text = match.group(0)
        src = _IMG_SRC_RE.search(text) or _MD_SRC_RE.search(text)
        if src and Path(src.group(1)).name in only:
            return ""
        return text

    md = _ANY_IMG_RE.sub(repl, md)
    # Drop wrappers the model leaves behind once their image is gone.
    return re.sub(r"<div[^>]*?>\s*</div>", "", md, flags=re.IGNORECASE)


def _label_in_place(md: str, mapping: dict[str, str]) -> str:
    """Tag figures without moving them (used when no question headers exist)."""
    totals: dict[str, int] = {}
    for question in mapping.values():
        totals[question] = totals.get(question, 0) + 1
    seen: dict[str, int] = {}

    def repl(match: re.Match[str]) -> str:
        text = match.group(0)
        url = image_ref_url(text)
        if not url:
            return text
        question = mapping.get(Path(url).name)
        if not question:
            return text
        seen[question] = seen.get(question, 0) + 1
        label = figure_label(question, seen[question], totals.get(question, 1))
        return f"![{label}]({url})"

    return _ANY_IMG_RE.sub(repl, md)


# Mapping values that mean "delete this figure" rather than "put it under Qn".
DROP_VALUES = {"drop", "remove", "delete", "hide", "x", "-", "✕", "✗", "❌"}


def extract_baked_labels(md: str) -> tuple[str, dict[str, str]]:
    """Pull figure labels that an engine baked into stored markdown back out.

    Older runs wrote the label into the page text instead of storing it as a
    mapping. This recovers ``{filename: question}`` so the label can be applied
    at render time instead, and returns the markdown with the labels removed.
    Handles both the standalone ``*[Q13 附图]*`` line and the alt-text form.
    """
    mapping: dict[str, str] = {}

    for match in _BAKED_LABEL_RE.finditer(md):
        url = image_ref_url(match.group(2))
        if url:
            mapping[Path(url).name] = match.group(1)

    for match in _ANY_IMG_RE.finditer(md):
        alt = ALT_QUESTION_RE.match(match.group(0))
        if alt:
            url = image_ref_url(match.group(0))
            if url:
                mapping[Path(url).name] = alt.group(1)

    return strip_labels(md), mapping


def _figure_block(names: list[str], question: str, texts: dict[str, str]) -> str:
    """The markdown for every figure belonging to one question.

    Emitted as standard markdown images whose alt text carries the label, so the
    figure and its question stay together in any markdown renderer:

        ![Q17 附图](https://.../a.jpg)
        ![Q17 附图2](https://.../b.jpg)     <- second figure of the same question
    """
    total = len(names)
    if not total:
        return ""
    out = ["\n\n"]
    for index, name in enumerate(names, start=1):
        url = image_ref_url(texts[name])
        if not url:
            continue
        out.append(f"![{figure_label(question, index, total)}]({url})\n")
    out.append("\n")
    return "".join(out)


def apply_labels_mapping(md: str, mapping: dict[str, str]) -> str:
    """Attach figures to their questions, and delete the ones marked to drop.

    A mapping value of ``"13"`` moves that figure under question 13 and tags it
    ``*[Q13 附图]*``. A value in :data:`DROP_VALUES` removes the figure from the
    document entirely — which is how a region the OCR mistook for a figure, such
    as a student's handwritten working, gets erased.

    Figures with no mapping at all stay exactly where OCR put them, so a partial
    mapping never shuffles the document around.
    """
    md = strip_labels(md or "")
    if not md.strip():
        return md

    available = set(extract_filenames(md))
    resolved: dict[str, str] = {}
    dropped: set[str] = set()

    for name, value in (mapping or {}).items():
        if name not in available:
            continue
        if str(value).strip().lower() in DROP_VALUES:
            dropped.add(name)
            continue
        question = normalize_question(value)
        if question:
            resolved[name] = question

    if not resolved and not dropped:
        return md

    # Deletions only — there is no question to move anything under.
    if not resolved:
        return _remove_images(md, dropped).strip() + "\n"

    grouped: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    for name, text in iter_image_refs(md):
        if name in resolved:
            grouped.setdefault(resolved[name], []).append(name)
            texts.setdefault(name, text)

    body = _remove_images(md, set(resolved) | dropped)
    headers = [(m.start(), str(int(m.group(1)))) for m in _QUESTION_RE.finditer(body)]

    if not headers:
        return _label_in_place(_remove_images(md, dropped), resolved)

    parts: list[str] = []
    cursor = 0
    for index, (start, number) in enumerate(headers):
        end = headers[index + 1][0] if index + 1 < len(headers) else len(body)
        parts.append(body[cursor:end])
        parts.append(_figure_block(grouped.pop(number, []), number, texts))
        cursor = end
    parts.append(body[cursor:])

    # Figures whose question number never showed up in the text.
    for number in sorted(grouped, key=int):
        parts.append(_figure_block(grouped[number], number, texts))

    return re.sub(r"\n{4,}", "\n\n\n", "".join(parts))


# --- merging -----------------------------------------------------------------


def page_markdown(md: str, label_mapping: str | None) -> str:
    """Apply a stored ``{filename: "QN"}`` JSON mapping to a page's markdown."""
    if not md:
        return ""
    mapping: dict[str, str] = {}
    if label_mapping:
        try:
            parsed = json.loads(label_mapping)
            if isinstance(parsed, dict):
                mapping = parsed
        except (json.JSONDecodeError, TypeError):
            mapping = {}
    return apply_labels_mapping(md, mapping) if mapping else md


def merge_pages(pages: list[str]) -> str:
    """Concatenate page markdowns with ``<!-- page N -->`` wrappers."""
    parts: list[str] = []
    for index, md in enumerate(pages, start=1):
        parts.append(f"<!-- page {index} -->\n\n{(md or '').strip()}\n\n<!-- /page {index} -->\n\n")
    return "".join(parts).strip() + "\n"
