"""Tests for the hybrid engine: local layout boxes + DeepSeek text.

The layout model and the DeepSeek call are both stubbed, so this runs offline
and fast. What is exercised for real is the cropping, the box matching and the
figure placement — the parts that are ours.
"""

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import _env  # noqa: E402,F401  (isolate the data dir before importing app)

from PIL import Image  # noqa: E402

from app import ocr  # noqa: E402
from app.layout import Block  # noqa: E402


def _page(width=1000, height=1000) -> Path:
    path = Path(tempfile.mkdtemp(prefix="hybrid-")) / "page.jpg"
    Image.new("RGB", (width, height), "white").save(path, format="JPEG")
    return path


# --- _attach_questions -------------------------------------------------------


def test_questions_are_taken_from_the_overlapping_hint():
    figures = [Block("image", 100, 100, 300, 200, 0.9, 1)]
    hints = [{"x1": 110, "y1": 110, "x2": 290, "y2": 190, "question": "13"}]
    assert ocr._attach_questions(figures, hints, (1000, 1000)) == ["13"]


def test_a_barely_overlapping_hint_is_not_trusted():
    """A weak match must leave the figure unplaced rather than guess."""
    figures = [Block("image", 100, 100, 300, 200, 0.9, 1)]
    hints = [{"x1": 295, "y1": 195, "x2": 400, "y2": 300, "question": "13"}]
    assert ocr._attach_questions(figures, hints, (1000, 1000)) == [""]


def test_hints_carrying_no_question_are_ignored():
    figures = [Block("image", 100, 100, 300, 200, 0.9, 1)]
    hints = [{"x1": 100, "y1": 100, "x2": 300, "y2": 200, "question": ""}]
    assert ocr._attach_questions(figures, hints, (1000, 1000)) == [""]


def test_each_figure_takes_its_own_question():
    figures = [
        Block("image", 0, 0, 200, 200, 0.9, 1),
        Block("image", 400, 400, 600, 600, 0.9, 2),
    ]
    hints = [
        {"x1": 0, "y1": 0, "x2": 200, "y2": 200, "question": "9"},
        {"x1": 400, "y1": 400, "x2": 600, "y2": 600, "question": "16"},
    ]
    assert ocr._attach_questions(figures, hints, (1000, 1000)) == ["9", "16"]


def test_normalised_hints_are_scaled_onto_the_page():
    """Hints arrive in 0-1000 space whatever size the page is."""
    # On a 2000x1000 page this hint covers x 200..600, y 100..200.
    figures = [Block("image", 150, 50, 650, 250, 0.9, 1)]
    hints = [{"x1": 100, "y1": 100, "x2": 300, "y2": 200, "question": "14"}]
    assert ocr._attach_questions(figures, hints, (2000, 1000)) == ["14"]


def test_no_hints_means_no_placements():
    figures = [Block("image", 10, 10, 100, 100, 0.9, 1)]
    assert ocr._attach_questions(figures, [], (1000, 1000)) == [""]


# --- _crop_figure ------------------------------------------------------------


def test_crop_writes_a_figure_with_the_expected_name():
    page = _page()
    images = page.parent / "images"
    images.mkdir()
    name = ocr._crop_figure(page, Block("image", 100, 200, 400, 500, 0.9, 1), images, "p1_")
    assert name is not None
    assert name.startswith("p1_img_in_image_box_")
    assert (images / name).is_file()
    with Image.open(images / name) as crop:
        # Padding is added, so it is slightly bigger than the raw box.
        assert crop.size[0] >= 300 and crop.size[1] >= 300


def test_crop_is_clamped_to_the_page_and_padded():
    page = _page(300, 300)
    images = page.parent / "images"
    images.mkdir()
    # A box touching the top-left corner: padding must not go negative.
    name = ocr._crop_figure(page, Block("image", 0, 0, 100, 100, 0.9, 1), images, "p1_")
    assert name is not None
    assert name.startswith("p1_img_in_image_box_0_0_")
    with Image.open(images / name) as crop:
        assert crop.size == (106, 106)  # 100 + 6px padding on the far edges


def test_crop_refuses_a_sliver():
    page = _page()
    images = page.parent / "images"
    images.mkdir()
    assert ocr._crop_figure(page, Block("image", 10, 10, 12, 12, 0.9, 1), images, "p1_") is None


# --- run_ocr_hybrid ----------------------------------------------------------


PAGE_MARKDOWN = """9. Simplify $x+y$.

*[Q9 附图]*

<img src="images/p1_img_in_image_box_100_100_300_200.jpg">

16. Explain the result.
"""


def test_hybrid_places_figures_under_their_questions():
    page = _page()
    work = page.parent / "work"
    images = page.parent / "images"
    images.mkdir()

    saved_layout = ocr.layout.detect
    saved_blocks = ocr.layout.figure_blocks
    saved_read = ocr.deepseek.read_page
    try:
        ocr.layout.detect = lambda path: [
            Block("text", 0, 0, 900, 90, 0.9, 1),
            Block("image", 100, 100, 300, 200, 0.9, 2),
        ]
        ocr.layout.figure_blocks = lambda blocks, min_score=None: [blocks[1]]
        ocr.deepseek.read_page = lambda path, model=None: (
            "9. Simplify $x+y$.\n\n16. Explain the result.\n",
            [{"x1": 100, "y1": 100, "x2": 300, "y2": 200, "question": "9"}],
        )
        markdown, figures = ocr.run_ocr_hybrid(page, work, images, "p1_")
    finally:
        ocr.layout.detect = saved_layout
        ocr.layout.figure_blocks = saved_blocks
        ocr.deepseek.read_page = saved_read

    assert len(figures) == 1
    assert "images/" + figures[0] in markdown
    # The shared labelling code moved it under question 9 and tagged it.
    assert "*[Q9 附图]*" in markdown
    assert markdown.index("*[Q9 附图]*") > markdown.index("9. Simplify")
    assert markdown.index("*[Q9 附图]*") < markdown.index("16. Explain")


def test_hybrid_keeps_unplaced_figures_rather_than_dropping_them():
    page = _page()
    work = page.parent / "work"
    images = page.parent / "images"
    images.mkdir()

    saved_layout = ocr.layout.detect
    saved_blocks = ocr.layout.figure_blocks
    saved_read = ocr.deepseek.read_page
    try:
        ocr.layout.detect = lambda path: [Block("image", 100, 100, 300, 200, 0.9, 1)]
        ocr.layout.figure_blocks = lambda blocks, min_score=None: list(blocks)
        # No hints at all — DeepSeek gave no figure information.
        ocr.deepseek.read_page = lambda path, model=None: ("9. Simplify $x+y$.\n", [])
        markdown, figures = ocr.run_ocr_hybrid(page, work, images, "p1_")
    finally:
        ocr.layout.detect = saved_layout
        ocr.layout.figure_blocks = saved_blocks
        ocr.deepseek.read_page = saved_read

    assert len(figures) == 1
    assert "images/" + figures[0] in markdown, "the figure must still be in the paper"


def test_hybrid_returns_cleanly_when_there_are_no_figures():
    page = _page()
    work = page.parent / "work"
    images = page.parent / "images"
    images.mkdir()

    saved_layout = ocr.layout.detect
    saved_blocks = ocr.layout.figure_blocks
    saved_read = ocr.deepseek.read_page
    try:
        ocr.layout.detect = lambda path: [Block("text", 0, 0, 900, 90, 0.9, 1)]
        ocr.layout.figure_blocks = lambda blocks, min_score=None: []
        ocr.deepseek.read_page = lambda path, model=None: ("9. Simplify $x+y$.\n", [])
        markdown, figures = ocr.run_ocr_hybrid(page, work, images, "p1_")
    finally:
        ocr.layout.detect = saved_layout
        ocr.layout.figure_blocks = saved_blocks
        ocr.deepseek.read_page = saved_read

    assert figures == []
    assert "9. Simplify" in markdown
    assert "<img" not in markdown


# --- engine dispatch ---------------------------------------------------------


def test_engine_selection():
    from app import config

    saved_engine = config.OCR_ENGINE
    saved_hybrid = ocr.run_ocr_hybrid
    saved_paddle = ocr.run_ocr_paddle
    page = _page()
    try:
        ocr.run_ocr_paddle = lambda *a, **k: ("paddle", [])
        ocr.run_ocr_hybrid = lambda *a, **k: ("hybrid", [])

        config.OCR_ENGINE = "paddle"
        assert ocr.run_ocr(page, page.parent, page.parent, "")[0] == "paddle"

        config.OCR_ENGINE = "deepseek"
        assert ocr.run_ocr(page, page.parent, page.parent, "")[0] == "hybrid"

        # auto falls back to Paddle when DeepSeek blows up.
        def boom(*a, **k):
            raise ocr.DeepSeekError("no api today")

        ocr.run_ocr_hybrid = boom
        config.OCR_ENGINE = "auto"
        assert ocr.run_ocr(page, page.parent, page.parent, "")[0] == "paddle"

        # ...and uses DeepSeek when it works.
        ocr.run_ocr_hybrid = lambda *a, **k: ("hybrid", [])
        assert ocr.run_ocr(page, page.parent, page.parent, "")[0] == "hybrid"
    finally:
        config.OCR_ENGINE = saved_engine
        ocr.run_ocr_hybrid = saved_hybrid
        ocr.run_ocr_paddle = saved_paddle


def test_auto_falls_back_when_layout_fails():
    from app import config

    saved_engine = config.OCR_ENGINE
    saved_hybrid = ocr.run_ocr_hybrid
    saved_paddle = ocr.run_ocr_paddle
    page = _page()
    try:
        def broken(*a, **k):
            raise ocr.LayoutError("model missing")

        ocr.run_ocr_hybrid = broken
        ocr.run_ocr_paddle = lambda *a, **k: ("paddle", [])
        config.OCR_ENGINE = "auto"
        assert ocr.run_ocr(page, page.parent, page.parent, "")[0] == "paddle"
    finally:
        config.OCR_ENGINE = saved_engine
        ocr.run_ocr_hybrid = saved_hybrid
        ocr.run_ocr_paddle = saved_paddle
