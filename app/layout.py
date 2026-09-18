"""Local layout detection with PP-DocLayoutV3.

This is the cheap half of PaddleOCR-VL: a detection CNN that finds *where* the
blocks are — text, figures, tables — in about 3.5 seconds, without the 0.9B
recognition model that takes another 90-130 seconds.

In the hybrid engine the boxes come from here (precise, local, free) and the
text comes from DeepSeek (better LaTeX, and it drops the handwriting in the same
pass). PaddleOCR-VL itself is still available as a fallback.

Coordinates are ``[x1, y1, x2, y2]`` in the pixel space of the image passed in.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from . import config

# Labels whose region is a figure we should crop out and embed.
FIGURE_LABELS = {"image", "figure", "chart"}
# Labels that are printed text — used to reason about reading order.
TEXT_LABELS = {
    "text",
    "paragraph_title",
    "doc_title",
    "figure_title",
    "table_title",
    "abstract",
    "content",
    "reference",
}

_model = None
_model_lock = threading.Lock()
# Detection is serialised: it is only ~3.5 s, and sharing one predictor across
# the OCR worker threads is safer than assuming it is reentrant.
_predict_lock = threading.Lock()


class LayoutError(RuntimeError):
    """Raised when layout detection cannot run."""


def log(message: str) -> None:
    print(f"[layout] {message}", flush=True)


@dataclass
class Block:
    """One detected region of the page."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    order: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def is_figure(self) -> bool:
        return self.label in FIGURE_LABELS

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def overlap(self, other: "Block") -> float:
        """Intersection-over-union with another block, 0..1."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.width * self.height + other.width * other.height - intersection
        return intersection / union if union > 0 else 0.0


def get_model():
    """Load PP-DocLayoutV3 once (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from paddlex import create_model

                log(f"loading layout model {config.LAYOUT_MODEL_NAME}…")
                _model = create_model(config.LAYOUT_MODEL_NAME)
                log("layout model ready")
    return _model


def detect(image_path: Path) -> list[Block]:
    """Detect blocks in ``image_path``, in reading order.

    Raises:
        LayoutError: if the model cannot be loaded or returns nothing usable.
    """
    try:
        model = get_model()
        with _predict_lock:
            results = list(model.predict(str(image_path)))
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        raise LayoutError(f"layout detection failed: {type(exc).__name__}: {exc}") from exc

    blocks: list[Block] = []
    for result in results:
        for index, raw in enumerate(result.get("boxes") or [], start=1):
            coordinate = raw.get("coordinate") or []
            if len(coordinate) != 4:
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in coordinate)
            blocks.append(
                Block(
                    label=str(raw.get("label") or "unknown"),
                    x1=min(x1, x2),
                    y1=min(y1, y2),
                    x2=max(x1, x2),
                    y2=max(y1, y2),
                    score=float(raw.get("score") or 0.0),
                    order=int(raw.get("order") or index),
                )
            )

    if not blocks:
        raise LayoutError("layout detection found no blocks")

    blocks.sort(key=lambda block: block.order)
    figures = sum(1 for block in blocks if block.is_figure)
    log(f"{len(blocks)} block(s), {figures} figure(s)")
    return blocks


def figure_blocks(blocks: list[Block], min_score: float | None = None) -> list[Block]:
    """The figure regions worth cropping, in reading order."""
    threshold = config.LAYOUT_MIN_SCORE if min_score is None else min_score
    figures = [
        block
        for block in blocks
        if block.is_figure
        and block.score >= threshold
        and block.width >= config.LAYOUT_MIN_PIXELS
        and block.height >= config.LAYOUT_MIN_PIXELS
    ]
    figures.sort(key=lambda block: (block.y1, block.x1))
    return figures


def contains(outer: Block, inner: Block) -> bool:
    """Is ``inner``'s centre inside ``outer``? Used to match two box lists."""
    cx, cy = inner.center()
    return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2
