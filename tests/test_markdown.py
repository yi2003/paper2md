"""Unit tests for the markdown plumbing (no OCR, no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.markdown_utils import (  # noqa: E402
    apply_labels_mapping,
    detect_questions,
    extract_filenames,
    merge_pages,
    normalize_markdown,
)

PAGE = """1. Simplify the expression below.

<img src="imgs/img_in_image_box_100_50_300_200.jpg" />

2. Solve for x.

<img src="imgs/img_in_image_box_400_60_600_210.jpg" />

3. Prove the identity.
"""


def test_normalize_rewrites_refs_and_img_tags():
    md = normalize_markdown('<img src="imgs/a.jpg" /> and ![x](imgs/b.png)')
    assert 'src="images/a.jpg"' in md
    assert "(images/b.png)" in md
    assert "/>" not in md


def test_normalize_fixes_latex_linebreak():
    assert "\\\\" in normalize_markdown("a \\ b")


def test_extract_filenames_is_ordered_and_deduped():
    names = extract_filenames(normalize_markdown(PAGE))
    assert names == [
        "img_in_image_box_100_50_300_200.jpg",
        "img_in_image_box_400_60_600_210.jpg",
    ]


def test_extract_filenames_ignores_remote_refs():
    md = "![a](https://example.com/a.jpg) ![b](images/b.jpg)"
    assert extract_filenames(md) == ["b.jpg"]


def test_detect_questions():
    assert detect_questions(PAGE) == ["1", "2", "3"]
    assert detect_questions("13、求值\n14．证明") == ["13", "14"]


def test_mapping_moves_figure_under_its_question():
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(
        md, {"img_in_image_box_100_50_300_200.jpg": "2"}
    )
    # The figure moved out of question 1 and now sits under question 2.
    assert out.index("*[Q2 附图]*") > out.index("2. Solve for x.")
    assert out.index("*[Q2 附图]*") < out.index("3. Prove the identity.")
    first_image = out.index("img_in_image_box_100_50_300_200.jpg")
    assert first_image > out.index("2. Solve for x.")


def test_unmapped_figures_stay_where_they_were():
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(md, {"img_in_image_box_400_60_600_210.jpg": "13"})
    # The figure the user did not touch keeps its original position, before "2.".
    assert out.index("img_in_image_box_100_50_300_200.jpg") < out.index("2. Solve for x.")


def test_labels_are_idempotent():
    md = normalize_markdown(PAGE)
    mapping = {"img_in_image_box_100_50_300_200.jpg": "Q1"}
    once = apply_labels_mapping(md, mapping)
    twice = apply_labels_mapping(once, mapping)
    assert once.count("*[Q1 附图]*") == 1
    assert twice.count("*[Q1 附图]*") == 1


def test_mapping_accepts_q_prefix_and_trailing_punctuation():
    md = normalize_markdown(PAGE)
    for value in ("Q2", "2", "2.", "2．"):
        out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": value})
        assert "*[Q2 附图]*" in out, value


def test_mapping_without_question_headers_falls_back_to_inline_labels():
    md = 'Intro text.\n\n<img src="images/a.jpg">\n\nMore text.'
    out = apply_labels_mapping(md, {"a.jpg": "7"})
    assert "*[Q7 附图]*" in out
    assert out.index("*[Q7 附图]*") < out.index("More text.")


def test_merge_pages_wraps_every_page():
    merged = merge_pages(["one", "two"])
    assert "<!-- page 1 -->" in merged
    assert "<!-- /page 2 -->" in merged
    assert merged.index("one") < merged.index("two")


# --- dropping figures (handwriting the OCR mistook for a diagram) ------------


def test_drop_removes_the_figure_entirely():
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": "drop"})
    assert "img_in_image_box_100_50_300_200.jpg" not in out
    # The other figure is untouched.
    assert "img_in_image_box_400_60_600_210.jpg" in out
    assert "1. Simplify the expression below." in out


def test_drop_works_with_no_question_headers():
    md = 'Intro.\n\n<img src="images/a.jpg">\n\nOutro.'
    out = apply_labels_mapping(md, {"a.jpg": "DROP"})
    assert "a.jpg" not in out
    assert "Intro." in out and "Outro." in out


def test_drop_and_assign_at_the_same_time():
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(
        md,
        {
            "img_in_image_box_100_50_300_200.jpg": "drop",
            "img_in_image_box_400_60_600_210.jpg": "2",
        },
    )
    assert "img_in_image_box_100_50_300_200.jpg" not in out
    assert "*[Q2 附图]*" in out
    assert out.index("*[Q2 附图]*") > out.index("2. Solve for x.")


def test_drop_aliases_are_accepted():
    md = normalize_markdown(PAGE)
    for value in ("drop", "DROP", " remove ", "x", "✕", "-"):
        out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": value})
        assert "img_in_image_box_100_50_300_200.jpg" not in out, value


def test_a_question_number_is_never_treated_as_a_drop():
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": "1"})
    assert "img_in_image_box_100_50_300_200.jpg" in out
    assert "*[Q1 附图]*" in out


def test_dropping_every_figure_leaves_clean_question_text():
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(
        md,
        {
            "img_in_image_box_100_50_300_200.jpg": "drop",
            "img_in_image_box_400_60_600_210.jpg": "drop",
        },
    )
    assert "<img" not in out
    assert "附图" not in out
    assert "1. Simplify the expression below." in out
    assert "3. Prove the identity." in out


def test_merge_pages_keeps_numbering_when_a_page_is_empty():
    merged = merge_pages(["one", "", "three"])
    assert "<!-- page 3 -->" in merged
    assert merged.index("one") < merged.index("three")
