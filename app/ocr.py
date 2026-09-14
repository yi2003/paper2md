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

from . import config
from .markdown_utils import iter_image_refs, normalize_markdown

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
    """OCR a single page.

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
