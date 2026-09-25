"""PDF ingestion: turn a PDF into page images the app already understands.

The rest of Paper2MD knows one trick well — take a photograph of a page, read it
with PaddleOCR-VL or DeepSeek, crop the figures out of it, let the user attach
them to questions, and emit one clean Markdown file. A PDF is therefore not a
second pipeline: it is a **page source**. This module renders each PDF page to
exactly the JPEG the app would have received from a phone camera, writes it where
:func:`app.config.page_image_path` expects it, and reports which page index each
one landed on. From there the existing worker, OCR engine, figure cropper,
review screen, DeepSeek cleanup and download are all unchanged.

Rendering uses `pypdfium2 <https://pypi.org/project/pypdfium2/>`_, a
self-contained wheel with no system poppler/Ghostscript/Java dependency — the
same library the PaddleOCR stack already pulls in. It is imported lazily so the
app still boots (and still serves photos) when it is missing.

Two things matter for quality:

* **Resolution.** 200 DPI is the default: enough for small subscripts and
  decimal points, which is exactly what the OCR gets wrong at screen DPI, while
  staying close to a phone photo in size.
* **Blank pages.** Exam PDFs are full of them (back sides, separator sheets). By
  default a page that is essentially pure white is skipped instead of spending
  ~30 s of OCR on it. The test is deliberately conservative — only a page with
  almost no ink at all qualifies — and the skipped pages are reported rather
  than silently vanishing.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config

# The smallest useful render; anything below this is a thumbnail, not a page.
MIN_DPI = 36
# Past ~600 DPI the page is larger than any phone photo and OCR gains nothing.
MAX_DPI = 600


class PdfError(RuntimeError):
    """Raised when a PDF cannot be opened, read, or rendered."""


def log(message: str) -> None:
    print(f"[pdf] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedPage:
    """One PDF page that was rendered to a JPEG on disk."""

    page_index: int  # position within the task (0-based, after blank skipping)
    page_number: int  # position within the PDF (1-based, never changes)
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class SkippedPage:
    """One PDF page that was deliberately not queued for OCR."""

    page_number: int
    reason: str


@dataclass
class PdfRenderResult:
    """What :func:`render_pages` produced, and what it chose to leave out."""

    source: str
    total_pages: int  # pages in the PDF
    first_page: int  # first page considered (1-based, inclusive)
    last_page: int  # last page considered (1-based, inclusive)
    dpi: int
    pages: list[RenderedPage] = field(default_factory=list)
    skipped: list[SkippedPage] = field(default_factory=list)
    # True when ``max_pages`` cut the range short — the rest of the PDF was not
    # looked at, and the caller should say so rather than implying completeness.
    truncated: bool = False

    @property
    def rendered(self) -> int:
        return len(self.pages)

    @property
    def page_count(self) -> int:
        """Number of pages actually considered (rendered + skipped)."""
        return len(self.pages) + len(self.skipped)

    @property
    def blank(self) -> list[int]:
        return [page.page_number for page in self.skipped]


# --------------------------------------------------------------------------
# Opening a document
# --------------------------------------------------------------------------


def is_available() -> bool:
    """True when the renderer can be imported (the app works without it)."""
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return False
    return True


def _read_bytes(source: str | Path | bytes | bytearray) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        try:
            data = Path(source).read_bytes()
        except OSError as exc:
            raise PdfError(f"could not read the PDF: {exc}") from exc
    if not data:
        raise PdfError("the PDF is empty")
    # A PDF need not start with "%PDF-" at byte 0 (some tools prepend junk), but
    # it must appear in the header region. Catching it early gives a much better
    # message than pdfium's "Failed to load document".
    if b"%PDF-" not in data[:1024]:
        raise PdfError("this file is not a PDF (no %PDF- header)")
    return data


def _open(source: str | Path | bytes | bytearray, password: str = ""):
    """Open a PDF and return ``(pdfium_module, document)``.

    The caller owns the document and must close it.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PdfError(
            "PDF support is not installed — run 'pip install pypdfium2' "
            "(it is in requirements.txt)"
        ) from exc

    data = _read_bytes(source)
    try:
        if password:
            document = pdfium.PdfDocument(data, password=password)
        else:
            document = pdfium.PdfDocument(data)
    except Exception as exc:  # noqa: BLE001 - pdfium raises a single error type
        message = str(exc) or type(exc).__name__
        if "password" in message.lower() or "encrypt" in message.lower():
            raise PdfError(
                "this PDF is password-protected; remove the password and try again"
            ) from exc
        raise PdfError(f"could not open the PDF: {message}") from exc
    return pdfium, document


def page_count(source: str | Path | bytes | bytearray) -> int:
    """Number of pages in a PDF, without rendering anything."""
    _, document = _open(source)
    try:
        return len(document)
    finally:
        document.close()


# --------------------------------------------------------------------------
# Deciding what a page is
# --------------------------------------------------------------------------


def _dark_ratio(image) -> float:
    """Fraction of pixels that carry ink.

    Uses a luminance histogram so it costs one pass over the image. "Dark" is
    generous (< 200/255) so a faint pencil line still counts as content.
    """
    gray = image.convert("L")
    histogram = gray.histogram()
    dark = sum(histogram[:200])
    return dark / max(1, gray.width * gray.height)


def _is_blank(image, threshold: float) -> bool:
    return _dark_ratio(image) < threshold


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_pages(
    source: str | Path | bytes | bytearray,
    destination: Path,
    *,
    first_index: int = 0,
    first_page: int = 1,
    last_page: int | None = None,
    max_pages: int | None = None,
    dpi: int | None = None,
    quality: int | None = None,
    skip_blank: bool | None = None,
    blank_ratio: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> PdfRenderResult:
    """Render the pages of a PDF into ``destination`` as ``page_N.jpg``.

    The filenames match :func:`app.config.page_image_path`, so the caller can
    hand the result straight to the existing worker.

    Args:
        source: a path, or the PDF bytes.
        destination: directory for the page JPEGs (usually the task's ``pages``).
        first_index: page index the first rendered page should take. Continue
            from :func:`app.store.next_page_index` to append a PDF to a task
            that already has pages. Blank pages do not consume an index.
        first_page / last_page: 1-based, inclusive page range within the PDF.
            ``last_page=None`` means "to the end".
        max_pages: hard cap on pages considered. Defaults to
            ``config.PDF_MAX_PAGES``; 0 disables the cap.
        dpi: render resolution; defaults to ``config.PDF_DPI``.
        quality: JPEG quality; defaults to ``config.PDF_JPEG_QUALITY``.
        skip_blank: drop essentially-white pages. Defaults to
            ``config.PDF_SKIP_BLANK``.
        blank_ratio: ink fraction below which a page counts as blank. Defaults
            to ``config.PDF_BLANK_RATIO``.
        on_progress: optional ``on_progress(done, total)`` callback.

    Returns:
        A :class:`PdfRenderResult`. The written files are ``.pages[*].path``.

    Raises:
        PdfError: the file is not a PDF, is encrypted, or has no pages in range.
    """
    pdfium, document = _open(source)
    try:
        total_pages = len(document)
        if total_pages == 0:
            raise PdfError("the PDF has no pages")

        first = max(1, first_page or 1)
        last = total_pages if not last_page or last_page < 1 else min(last_page, total_pages)
        if first > last:
            raise PdfError(
                f"page range {first}-{last} is empty — the PDF has {total_pages} page(s)"
            )

        wanted = last - first + 1
        cap = config.PDF_MAX_PAGES if max_pages is None else max_pages
        truncated = False
        if cap and wanted > cap:
            truncated = True
            wanted = cap
            last = first + wanted - 1

        render_dpi = max(MIN_DPI, min(MAX_DPI, dpi or config.PDF_DPI))
        jpeg_quality = quality if quality is not None else config.PDF_JPEG_QUALITY
        blank = config.PDF_SKIP_BLANK if skip_blank is None else skip_blank
        threshold = config.PDF_BLANK_RATIO if blank_ratio is None else blank_ratio
        scale = render_dpi / 72.0

        destination.mkdir(parents=True, exist_ok=True)
        result = PdfRenderResult(
            source=getattr(source, "name", None) or "pdf",
            total_pages=total_pages,
            first_page=first,
            last_page=last,
            dpi=render_dpi,
            truncated=truncated,
        )

        index = first_index
        for offset in range(wanted):
            number = first + offset
            page = document[number - 1]
            try:
                image = _render_one(page, scale)
            finally:
                page.close()

            try:
                if blank and _is_blank(image, threshold):
                    result.skipped.append(SkippedPage(number, "page is blank"))
                else:
                    path = destination / config.page_image_name(index)
                    image.save(path, format="JPEG", quality=jpeg_quality)
                    result.pages.append(
                        RenderedPage(
                            page_index=index,
                            page_number=number,
                            path=path,
                            width=image.width,
                            height=image.height,
                        )
                    )
                    index += 1
            finally:
                image.close()

            if on_progress:
                on_progress(offset + 1, wanted)

        log(
            f"{result.source}: {total_pages} page(s) in the PDF, "
            f"rendered {result.rendered} at {render_dpi} DPI, "
            f"skipped {len(result.skipped)} blank"
            + (" (capped)" if truncated else "")
        )
        return result
    finally:
        document.close()


def _render_one(page, scale: float):
    """Render one pdfium page to a PIL image, clamping absurd page sizes."""
    width_pt, height_pt = page.get_size()
    max_pixels = config.PDF_MAX_PIXELS
    if max_pixels and width_pt > 0 and height_pt > 0:
        projected = (width_pt * scale) * (height_pt * scale)
        if projected > max_pixels:
            scale = math.sqrt(max_pixels / (width_pt * height_pt))
            log(f"page is {projected / 1e6:.0f} MPx at {scale:.2f} scale — clamped")
    bitmap = page.render(scale=scale)
    try:
        return bitmap.to_pil()
    finally:
        bitmap.close()
