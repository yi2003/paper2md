"""Markdown helpers.

Everything that understands the shape of OCR output lives here: normalising raw
PaddleOCR-VL markdown, finding figure references, moving figures underneath the
question they belong to, laying several figures out on one line, and merging
per-page output into one document.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "gif", "webp", "bmp")
_EXT_ALT = "|".join(IMAGE_EXTENSIONS)

# The class the app puts on a row it built, so its own rows stay recognisable
# after the user has hand-edited the inline style around them.
ROW_CLASS = "p2md-row"
# Layout defaults for a generated row. `align-items:flex-end` lines the figures
# up along their baselines, which is what a side-by-side diagram wants.
ROW_GAP_PX = 12
ROW_MAX_HEIGHT_PX = 200
# Where a moved figure was, while the text around it is being rewritten. NUL
# cannot appear in markdown text, so it can never collide with real content.
_ROW_ANCHOR = "\x00ph0\x00"
_PLACEHOLDER_RE = re.compile(r"\x00ph\d+\x00")

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
# `<div …>` or `</div>`, innermost blocks close first.
_DIV_TAG_RE = re.compile(r"<div\b[^>]*>|</div\s*>", re.IGNORECASE)
# A wrapper div left behind once its figure is gone.
_EMPTY_DIV_RE = re.compile(r"<div[^>]*?>\s*</div>", re.IGNORECASE)
# `display:flex` / `inline-flex` / `display: grid` — the one-line layout idiom.
_FLEX_DISPLAY_RE = re.compile(r"display\s*:\s*(?:inline-)?(?:flex|grid)", re.IGNORECASE)
# Attributes of an existing `<img>` tag, and the alt text of a markdown image.
_TAG_ATTR_RE = re.compile(r"""([\w:-]+)\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_MD_ALT_RE = re.compile(r"!\[\s*([^\]]*?)\s*\]")
# Three or more blank lines collapse to one blank line.
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{4,}")


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


# --- LaTeX that fell outside math mode ----------------------------------------

# $$...$$, \[...\], \(...\) and $...$ — the four ways this app writes math.
_MATH_SPAN_RE = re.compile(
    r"\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|\$[^$\n]+?\$", re.DOTALL
)

# A fill-in-the-blank: the one LaTeX fragment an exam paper is full of, and the
# one a model is most likely to leave sitting in the prose.
_BLANK_RE = re.compile(r"\\underline\{\\hspace\{[^{}]*\}\}")


def wrap_bare_latex(md: str) -> str:
    """Put a blank back inside math mode, wherever the model wrote it.

    KaTeX only renders LaTeX between delimiters, so a blank written as
    ``\\underline{\\hspace{2em}}`` in the middle of a Chinese sentence arrives
    as literal backslashes — the reader sees ``\\underline{\\hspace{2em}}``
    instead of a ruled space. Models do this constantly, because a blank
    standing on its own between two words looks like it belongs to the prose.

    Only fragments *outside* every math span are touched, so a blank already
    written as ``$\\underline{\\hspace{2em}}$`` — or inside a ``$$...$$`` block —
    is left exactly as it is. ``\\(...\\)`` is the wrapper rather than ``$...$``
    because a line can hold several blanks in a row, and ``$a$$b$`` is
    ambiguous to read back.
    """
    if "\\underline{\\hspace" not in md:
        return md

    def wrap(fragment: str) -> str:
        return _BLANK_RE.sub(lambda m: "\\(" + m.group(0) + "\\)", fragment)

    parts: list[str] = []
    cursor = 0
    for match in _MATH_SPAN_RE.finditer(md):
        parts.append(wrap(md[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(wrap(md[cursor:]))
    return "".join(parts)


# --- figure references -------------------------------------------------------


def strip_figures(md: str) -> str:
    """Remove every figure reference from a page of text.

    Used when OCR is not allowed to find figures: both models emit
    ``![](images/...)`` on their own initiative, and a reference with no file
    behind it is a broken image in the finished paper.
    """
    without = _EXCESS_BLANK_LINES_RE.sub("\n\n", _ANY_IMG_RE.sub("", md)).strip()
    return without + "\n" if without else ""


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
    return _EMPTY_DIV_RE.sub("", md)


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


# --- one-line figure rows ----------------------------------------------------
#
# Two small figures of the same question read far better side by side than
# stacked with a page break between them. Markdown has no syntax for that, so a
# row is a plain ``<div>`` laid out with flexbox, holding the figures as direct
# children — which is also why the figures have to become ``<img>`` tags:
# inside an HTML block a markdown image is not parsed, it prints as text.


def iter_div_blocks(md: str) -> Iterator[tuple[int, int, str, str]]:
    """Yield ``(start, end, open_tag, inner)`` for every ``<div>…</div>``.

    ``start``/``end`` span the whole block including its tags. Nesting is
    handled by matching tags, so an inner ``</div>`` never closes an outer one,
    and inner blocks are yielded before the block that contains them.
    """
    open_blocks: list[tuple[int, str]] = []
    for match in _DIV_TAG_RE.finditer(md):
        tag = match.group(0)
        if tag.startswith("</"):
            if not open_blocks:
                continue
            start, open_tag = open_blocks.pop()
            yield start, match.end(), open_tag, md[start + len(open_tag) : match.start()]
        else:
            open_blocks.append((match.start(), tag))


def is_figure_row(open_tag: str, inner: str) -> bool:
    """Is this div a row that lays figures out side by side?

    True for a div the app generated (it carries :data:`ROW_CLASS`) and for one
    the user wrote by hand with flexbox or grid — which is what makes a row
    survive a hand edit and, later, the DeepSeek cleanup.
    """
    if not _ANY_IMG_RE.search(inner):
        return False
    return ROW_CLASS in open_tag or bool(_FLEX_DISPLAY_RE.search(open_tag))


def figures_in_rows(md: str) -> list[str]:
    """Filenames of the figures that already sit inside a one-line row."""
    names: list[str] = []
    for _start, _end, open_tag, inner in iter_div_blocks(md):
        if not is_figure_row(open_tag, inner):
            continue
        for name, _text in iter_image_refs(inner):
            if name not in names:
                names.append(name)
    return names


def _attr_escape(value: str) -> str:
    """Make a value safe inside a double-quoted attribute.

    ``&`` is left alone on purpose: the figure URLs are rewritten by plain
    string replacement later, and escaping them here would break that.
    """
    return value.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _figure_tag(
    ref: str, *, max_width: int = 0, max_height: int = 0, width: int | None = None
) -> str:
    """One figure as an ``<img>`` tag, keeping its label and any size it has.

    A markdown image becomes ``<img src alt>``; an existing tag keeps the
    attributes it had. A cap is only added when the tag does not already carry
    one, so a hand-written size — or a row's ``max-height`` — always wins.
    """
    url = image_ref_url(ref)
    if not url:
        return ""

    if _IMG_SRC_RE.search(ref):
        attrs = dict(_TAG_ATTR_RE.findall(ref))
    else:
        alt = _MD_ALT_RE.match(ref.strip())
        attrs = {"src": url, "alt": alt.group(1) if alt else "Image"}

    if width and "width" not in attrs:
        # A plain attribute, for viewers that do not honour inline CSS.
        attrs["width"] = str(width)

    style = attrs.get("style", "").strip().rstrip(";")
    declarations = [style] if style else []
    if max_width and "max-width" not in style:
        declarations.append(f"max-width:{max_width}px")
    if max_height and "max-height" not in style:
        declarations.append(f"max-height:{max_height}px")
    if declarations:
        attrs["style"] = "; ".join(declarations)

    rendered = " ".join(f'{key}="{_attr_escape(value)}"' for key, value in attrs.items())
    return f"<img {rendered}>"


def _img_tag(ref: str, max_height: int = ROW_MAX_HEIGHT_PX) -> str:
    """One figure for a row, bounded so the row stays on one line."""
    return _figure_tag(ref, max_height=max_height)


def style_figures(
    md: str,
    max_width: int = 0,
    max_height: int = 0,
    size_for: Callable[[str], tuple[int | None, int | None]] | None = None,
) -> str:
    """Give every figure the size of the rectangle it was cut from.

    A figure is sized by how much of the page it covered: a diagram drawn across
    half the page is half the page wide in the paper. ``size_for`` works that
    out — it knows the rectangle and the page — and returns ``(width, height)``
    in pixels, or ``(None, None)`` for a figure nothing is known about, which
    then gets the ceiling. ``max_width``/``max_height`` are that ceiling, so a
    rectangle covering the whole page cannot swallow the text around it; 0
    lifts them.

    A crop keeps the full resolution of the photo it came from — right for
    zooming into, wrong for reading: a diagram cut from a 12-megapixel phone
    photo is 3000 px wide and sprawls across the page. This sizes the display
    only; the file on disk is untouched, as is the text you saved.

    The size is written twice, because Markdown viewers disagree:

    * ``style="max-width:…px; max-height:…px"`` — CSS-aware renderers (GitHub,
      VS Code, Obsidian).
    * ``width="…"`` — a plain HTML attribute, honoured by everything else,
      including viewers and exporters that strip inline styles.
    """
    if not (max_width or max_height or size_for):
        return md

    def replace(match: re.Match[str]) -> str:
        ref = match.group(0)
        if _has_size(ref):
            # It already says how big it should be — that always wins.
            return _figure_tag(ref)

        width, height = size_for(image_ref_url(ref) or "") if size_for else (None, None)
        width = _capped(width, max_width)
        height = _capped(height, max_height)
        return _figure_tag(
            ref, width=width, max_width=width or max_width, max_height=height or max_height
        )

    return _ANY_IMG_RE.sub(replace, md)


def _capped(size: int | None, cap: int) -> int:
    """``size`` within ``cap``; 0 means no limit in either direction."""
    if not size:
        return cap
    return min(size, cap) if cap else size


def _has_size(ref: str) -> bool:
    """Does this reference already say how big it should be?"""
    return "max-width" in ref or bool(re.search(r"""\swidth\s*=""", ref, re.IGNORECASE))


def figure_row(
    refs: Iterable[str], max_height: int = ROW_MAX_HEIGHT_PX, gap: int = ROW_GAP_PX
) -> str:
    """The one-line row div for ``refs``, in the order given.

    ``refs`` are figure references — ``<img …>`` tags or ``![…](…)`` — not bare
    filenames. The figures are direct flex children, so they sit on one line
    and wrap to the next only if the line runs out of room.
    """
    items = [tag for tag in (_img_tag(ref, max_height) for ref in refs) if tag]
    if not items:
        return ""
    style = (
        f"display:flex; flex-wrap:wrap; align-items:flex-end; "
        f"justify-content:center; gap:{gap}px; margin:0.6em 0;"
    )
    body = "\n".join(items)
    return f'<div class="{ROW_CLASS}" style="{style}">\n{body}\n</div>'


def _is_single_row(inner: str) -> bool:
    """Is this div's content exactly one row div and nothing else?"""
    text = inner.strip()
    for start, end, open_tag, body in iter_div_blocks(text):
        if start == 0 and end == len(text):
            return is_figure_row(open_tag, body)
    return False


def _unwrap_row_containers(md: str) -> str:
    """Drop a wrapper div the OCR left around the row.

    OCR wraps a figure in ``<div style="text-align: center;">``. The row is cut
    out of the middle of that wrapper, so the wrapper would end up holding
    nothing but the row; leaving it would nest the row inside a redundant box.
    """
    for _attempt in range(4):  # only a wrapper-in-a-wrapper needs unwinding twice
        found = next(
            (
                (start, end, inner.strip())
                for start, end, _tag, inner in iter_div_blocks(md)
                if _is_single_row(inner)
            ),
            None,
        )
        if found is None:
            return md
        start, end, row = found
        md = md[:start] + row + md[end:]
    return md


def _row_containing(md: str, offset: int) -> tuple[int, int] | None:
    """The outermost figure row that encloses ``offset``, as a span."""
    enclosing: tuple[int, int] | None = None
    # iter_div_blocks yields inner blocks first, so the last match is the outer one.
    for start, end, open_tag, inner in iter_div_blocks(md):
        if start <= offset < end and is_figure_row(open_tag, inner):
            enclosing = (start, end)
    return enclosing


def wrap_figures_in_row(
    md: str, names: Iterable[str], max_height: int = ROW_MAX_HEIGHT_PX
) -> str:
    """Move ``names`` into one row so they end up on a single line.

    The row is placed where the first of the figures used to be — or just after
    the row it was pulled out of, so rows never end up nested inside each other.
    The wrappers the OCR left around those figures are cleaned away. Figures that
    are not in ``md`` are ignored.

    Raises:
        ValueError: fewer than two of ``names`` are figures in ``md`` — a row of
            one is just the figure on its own line.
    """
    wanted = {Path(str(name)).name for name in names if str(name).strip()}
    if not wanted:
        raise ValueError("no figures were given")

    # The first occurrence of each figure, in document order.
    found: dict[str, tuple[tuple[int, int], str]] = {}
    for match in _ANY_IMG_RE.finditer(md):
        ref = match.group(0)
        url = image_ref_url(ref)
        if url and Path(url).name in wanted:
            found.setdefault(Path(url).name, (match.span(), ref))

    if len(found) < 2:
        raise ValueError(
            f"only {len(found)} of those figures are on this page — "
            "a one-line row needs at least two"
        )

    cuts = sorted(found.items(), key=lambda item: item[1][0][0])
    row = figure_row((ref for _name, (_span, ref) in cuts), max_height=max_height)

    # Cut the figures out, leaving a placeholder where each one was. The
    # placeholder is what the row is placed at afterwards, so its position
    # survives the text around it being rewritten.
    parts: list[str] = []
    cursor = 0
    for index, (_name, ((start, end), _ref)) in enumerate(cuts):
        parts.append(md[cursor:start])
        parts.append(f"\x00ph{index}\x00")
        cursor = end
    parts.append(md[cursor:])

    out = _EMPTY_DIV_RE.sub("", _unwrap_row_containers("".join(parts)))

    # The row goes where the first figure was — see `_PLACEHOLDER_RE`.
    insert_at = out.index(_ROW_ANCHOR)
    enclosing = _row_containing(out, insert_at)
    if enclosing is not None:
        insert_at = enclosing[1]  # beside the row it came out of, not inside it
    out = out[:insert_at] + f"\n{row}\n" + out[insert_at:]

    out = _PLACEHOLDER_RE.sub("", out)
    out = _unwrap_row_containers(_EMPTY_DIV_RE.sub("", out))
    return _EXCESS_BLANK_LINES_RE.sub("\n\n\n", out).strip() + "\n"


def stale_mapping_names(mapping: dict[str, str], filenames: Iterable[str]) -> list[str]:
    """Entries of ``mapping`` that name a figure the page no longer has.

    Figure filenames carry their crop coordinates, so a page read again may
    number its figures differently — and the numbers set for the old filenames
    quietly stop applying. Listing them is the only way the user finds out.
    """
    present = {Path(name).name for name in filenames}
    return [name for name in mapping if Path(name).name not in present]


def manual_crop_reference(name: str, max_width: int = 0, max_height: int = 0) -> str:
    """The markdown reference for a figure the user outlined by hand."""
    reference = f"![Image](images/{name})"
    if not (max_width or max_height):
        return reference
    return _figure_tag(reference, max_width=max_width, max_height=max_height)


def append_crops(
    md: str, names: Iterable[str], max_width: int = 0, max_height: int = 0
) -> str:
    """Put hand-outlined figures at the end of a page's text.

    They go on *before* the figure mapping is applied, so a crop given a
    question number is moved under that question exactly like an OCR figure,
    and one with no number simply stays at the end of the page.

    A figure whose reference is already in the text — it was put into a
    hand-made one-line row — is left alone rather than appearing twice.

    ``max_width``/``max_height`` are for the one case ``style_figures`` cannot
    cover: hand-edited text, which is used exactly as saved and so must carry
    the size itself.
    """
    references = [
        manual_crop_reference(name, max_width, max_height)
        for name in names
        if str(name).strip() and f"images/{name}" not in md
    ]
    if not references:
        return md
    body = "\n\n".join(references)
    if not md.strip():
        return body + "\n"
    return md.rstrip() + "\n\n" + body + "\n"


def strip_figure(md: str, name: str) -> str:
    """Remove every reference to one figure, in either syntax.

    Used when an outlined figure is deleted but its reference is already in the
    page's hand-edited text — inside a one-line row, say — so the paper cannot
    keep pointing at a file that is gone.
    """
    md = re.sub(rf"<img[^>]*?\ssrc=[\"']images/{re.escape(name)}[\"'][^>]*?>\s*", "", md)
    md = re.sub(rf"!\[[^\]]*\]\(images/{re.escape(name)}\)\s*", "", md)
    return _EMPTY_DIV_RE.sub("", md).strip() + "\n" if md.strip() else md


def page_markdown(md: str, label_mapping: str | None, edited: str | None = None) -> str:
    """The markdown a page contributes to the paper.

    ``edited`` is the user's own version of the page, saved from the review
    screen's markdown editor; when there is one it *is* the page, so the
    mapping-derived version is left alone as the fallback to revert to.
    """
    if edited and edited.strip():
        return edited
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
