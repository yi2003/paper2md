"""PaddleOCR-VL wrapper.

Turns one photo of an exam page into markdown plus a folder of extracted
figures. The model is heavy, so instances are created lazily and handed out
through a small pool (one per worker thread).
"""

from __future__ import annotations

import re
import shutil
import sys
import threading
from pathlib import Path

from . import config, deepseek, layout
from .deepseek import DeepSeekError
from .layout import LayoutError
from .markdown_utils import apply_labels_mapping, iter_image_refs, normalize_markdown

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


class OcrError(RuntimeError):
    """Raised when the OCR pipeline cannot produce a result."""


def log(message: str) -> None:
    print(f"[ocr] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
#
# One PaddleOCR-VL instance is shared by every OCR thread. Inference only reads
# the weights, so concurrent predict() calls run in parallel instead of
# duplicating the model in RAM.
#
# Measured on a 16-core CPU (Intel Core Ultra 9 285H), one page per thread:
#
#     threads   wall (3 pages)   throughput     peak RSS
#        1         266.8 s       12.3 chars/s    9.2 GB
#        3         115.3 s       28.5 chars/s    9.2 GB
#        6         171.0 s       38.5 chars/s    9.2 GB
#
# A pool of separate instances (the obvious alternative) would need ~5 GB per
# worker and cannot fit more than two of them in 16 GB.

_pipeline = None
_pipeline_lock = threading.Lock()


def _build_pipeline():
    from paddleocr import PaddleOCRVL

    log(f"loading PaddleOCR-VL {config.OCR_PIPELINE_VERSION} (first run downloads models)…")
    try:
        pipeline = PaddleOCRVL(
            pipeline_version=config.OCR_PIPELINE_VERSION,
            enable_mkldnn=config.OCR_ENABLE_MKLDNN,
        )
    except Exception as exc:  # noqa: BLE001 - fall back rather than die
        if not config.OCR_ENABLE_MKLDNN:
            raise
        log(f"enable_mkldnn=True failed ({exc}); retrying without oneDNN")
        pipeline = PaddleOCRVL(pipeline_version=config.OCR_PIPELINE_VERSION)
    log("model ready")
    return pipeline


def get_pipeline():
    """Return the shared pipeline, loading it on first use (thread-safe)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = _build_pipeline()
    return _pipeline


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------


def prepare_input(src: Path, dst: Path, max_dim: int | None = None) -> Path:
    """Normalise any input photo into a reasonably sized RGB JPEG.

    Phone photos are usually rotated via EXIF and far larger than the model
    needs; both are handled here.
    """
    from PIL import Image, ImageOps

    dst.parent.mkdir(parents=True, exist_ok=True)
    limit = config.OCR_MAX_DIM if max_dim is None else max_dim
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        width, height = image.size
        if limit and max(width, height) > limit:
            scale = limit / max(width, height)
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.LANCZOS,
            )
            log(f"downsampled {width}x{height} -> {image.size[0]}x{image.size[1]}")
        image.save(dst, format="JPEG", quality=92)
    return dst


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def _newest_markdown(work_dir: Path) -> Path | None:
    candidates = sorted(work_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _replace_ref(md: str, old: str, new: str) -> str:
    return re.sub(
        rf"(?<![\w/])images/{re.escape(old)}(?![\w.])",
        f"images/{new}",
        md,
    )


def run_ocr(
    image_path: Path,
    work_dir: Path,
    images_dir: Path,
    prefix: str = "",
) -> tuple[str, list[str]]:
    """OCR a single page using the configured engine.

    ``paddle``   — PaddleOCR-VL end to end.
    ``deepseek`` — local layout detection for the figure boxes, DeepSeek for the
                   text (see :func:`run_ocr_hybrid`).
    ``auto``     — DeepSeek, falling back to Paddle if it fails.

    Returns ``(markdown, figure_filenames)``; markdown references figures as
    ``images/<prefixed filename>``.
    """
    engine = config.OCR_ENGINE

    if engine == "deepseek":
        return run_ocr_hybrid(image_path, work_dir, images_dir, prefix)

    if engine == "auto":
        try:
            return run_ocr_hybrid(image_path, work_dir, images_dir, prefix)
        except (DeepSeekError, LayoutError) as exc:
            log(f"deepseek engine failed ({exc}); falling back to PaddleOCR-VL")

    return run_ocr_paddle(image_path, work_dir, images_dir, prefix)


def run_ocr_paddle(
    image_path: Path,
    work_dir: Path,
    images_dir: Path,
    prefix: str = "",
) -> tuple[str, list[str]]:
    """OCR a single page with PaddleOCR-VL.

    Args:
        image_path: the uploaded photo.
        work_dir: scratch directory (removed by the caller).
        images_dir: where extracted figures are copied.
        prefix: prepended to every figure filename so pages never collide.

    Returns:
        ``(markdown, figure_filenames)``. Markdown references figures as
        ``images/<prefixed filename>``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_input(image_path, work_dir / "input.jpg")

    pipeline = get_pipeline()
    results = pipeline.predict(
        str(prepared),
        use_layout_detection=True,
        use_chart_recognition=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        format_block_content=True,
    )
    for result in results:
        result.save_to_markdown(save_path=str(work_dir))

    markdown_path = _newest_markdown(work_dir)
    if markdown_path is None:
        raise OcrError("PaddleOCR-VL produced no Markdown output")
    markdown = normalize_markdown(markdown_path.read_text(encoding="utf-8"))

    # Every image the model wrote out, keyed by basename.
    extracted: dict[str, Path] = {}
    for path in sorted(work_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.name == prepared.name:
                continue
            extracted.setdefault(path.name, path)

    copied: list[str] = []
    for basename, _ in list(iter_image_refs(markdown)):
        source = extracted.get(basename)
        if source is None:
            log(f"referenced figure not found on disk: {basename}")
            continue
        target_name = f"{prefix}{basename}"
        shutil.copy2(source, images_dir / target_name)
        markdown = _replace_ref(markdown, basename, target_name)
        if target_name not in copied:
            copied.append(target_name)

    log(f"markdown {len(markdown):,} chars, {len(copied)} figure(s)")
    return markdown, copied


# --------------------------------------------------------------------------
# Hybrid engine: local layout boxes + DeepSeek text
# --------------------------------------------------------------------------
#
# Why the split: DeepSeek reads the page better than PaddleOCR-VL (cleaner
# LaTeX, and it drops the handwriting in the same pass) but its bounding boxes
# are only approximate, and a bad crop clips the diagram or keeps the very
# handwriting we are trying to remove.
#
# PP-DocLayoutV3 gives pixel-accurate boxes for ~3.5 s locally. So the boxes
# come from there, the text from DeepSeek, and DeepSeek is asked only which
# question each figure belongs to — a judgement it is well suited to.


def _crop_figure(
    prepared: Path,
    block: "layout.Block",
    images_dir: Path,
    prefix: str,
) -> str | None:
    """Crop one layout block out of the page and save it as a figure."""
    from PIL import Image

    try:
        with Image.open(prepared) as page:
            width, height = page.size
            # Judge the raw block, not the padded one: padding would turn a
            # 2x2 speck into a 14x14 "figure".
            if block.width < 8 or block.height < 8:
                return None
            pad = config.LAYOUT_PADDING
            x1 = max(0, block.x1 - pad)
            y1 = max(0, block.y1 - pad)
            x2 = min(width, block.x2 + pad)
            y2 = min(height, block.y2 + pad)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return None
            crop = page.crop((x1, y1, x2, y2))
            if crop.mode != "RGB":
                crop = crop.convert("RGB")
            # Same naming convention as PaddleOCR-VL, so the rest of the app
            # (figure serving, box parsing, auto-fill) works unchanged.
            name = f"{prefix}img_in_image_box_{x1}_{y1}_{x2}_{y2}.jpg"
            crop.save(images_dir / name, format="JPEG", quality=92)
            return name
    except Exception as exc:  # noqa: BLE001 - one bad crop must not kill the page
        log(f"could not crop {block.label} at ({block.x1},{block.y1}): {exc}")
        return None


def _attach_questions(
    figures: list["layout.Block"],
    hints: list[dict],
    page_size: tuple[int, int],
) -> list[str]:
    """Work out which question each layout figure belongs to.

    DeepSeek reports boxes in 0-1000 space; we scale those onto the page and
    match each to the layout block it overlaps most. Question numbers come from
    the hint, never from the coordinates.
    """
    width, height = page_size
    scaled = []
    for hint in hints:
        question = str(hint.get("question") or "").strip()
        if not question:
            continue
        scaled.append(
            {
                "x1": hint["x1"] / 1000 * width,
                "y1": hint["y1"] / 1000 * height,
                "x2": hint["x2"] / 1000 * width,
                "y2": hint["y2"] / 1000 * height,
                "question": question,
            }
        )

    questions: list[str] = []
    for block in figures:
        best_question, best_score = "", 0.0
        for hint in scaled:
            ix1, iy1 = max(block.x1, hint["x1"]), max(block.y1, hint["y1"])
            ix2, iy2 = min(block.x2, hint["x2"]), min(block.y2, hint["y2"])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            intersection = (ix2 - ix1) * (iy2 - iy1)
            smaller = min(
                (block.x2 - block.x1) * (block.y2 - block.y1),
                (hint["x2"] - hint["x1"]) * (hint["y2"] - hint["y1"]),
            )
            score = intersection / smaller if smaller > 0 else 0.0
            if score > best_score:
                best_question, best_score = hint["question"], score

        # Below this the two boxes barely overlap; better to leave it unplaced
        # and let the review screen assign it than to guess wrong.
        questions.append(best_question if best_score >= 0.3 else "")
        if best_question and best_score < 0.3:
            log(f"weak figure match ({best_score:.2f}); leaving it unplaced")

    return questions


def run_ocr_hybrid(
    image_path: Path,
    work_dir: Path,
    images_dir: Path,
    prefix: str = "",
) -> tuple[str, list[str]]:
    """Local layout detection for the boxes, DeepSeek for the text."""
    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_input(image_path, work_dir / "input.jpg")
    from PIL import Image

    with Image.open(prepared) as image:
        page_size = image.size

    # 1. Where is everything? Fast, local, pixel-accurate.
    blocks = layout.detect(prepared)
    figure_blocks = layout.figure_blocks(blocks)
    log(f"layout: {len(figure_blocks)} figure(s) to crop from {len(blocks)} block(s)")

    # 2. Read the page. This also removes the handwriting, so no cleanup pass.
    markdown, hints = deepseek.read_page(prepared)

    # 3. Crop the figures from the precise layout boxes.
    names: list[str] = []
    kept: list["layout.Block"] = []
    for block in figure_blocks:
        name = _crop_figure(prepared, block, images_dir, prefix)
        if name:
            names.append(name)
            kept.append(block)

    if not names:
        log(f"markdown {len(markdown):,} chars, no figures")
        return markdown, []

    # 4. Which question does each figure belong to?
    questions = _attach_questions(kept, hints, page_size)
    mapping = {name: q for name, q in zip(names, questions) if q}

    # 5. Let the shared labelling code move them under their questions.
    markdown = markdown.rstrip() + "\n\n" + "\n\n".join(
        f'<img src="images/{name}">' for name in names
    ) + "\n"
    if mapping:
        markdown = apply_labels_mapping(markdown, mapping)

    placed = sum(1 for q in questions if q)
    log(
        f"markdown {len(markdown):,} chars, {len(names)} figure(s), "
        f"{placed} placed under a question"
    )
    return markdown, names
