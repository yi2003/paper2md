"""Unit tests for the markdown plumbing (no OCR, no network)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.markdown_utils import (  # noqa: E402
    apply_labels_mapping,
    append_crops,
    extract_baked_labels,
    detect_questions,
    extract_filenames,
    figures_in_rows,
    is_figure_row,
    iter_div_blocks,
    merge_pages,
    normalize_markdown,
    page_markdown,
    strip_figure,
    strip_figures,
    style_figures,
    wrap_bare_latex,
    wrap_figures_in_row,
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
    assert out.index("![Q2 附图](") > out.index("2. Solve for x.")
    assert out.index("![Q2 附图](") < out.index("3. Prove the identity.")
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
    assert once.count("![Q1 附图](") == 1
    assert twice.count("![Q1 附图](") == 1


def test_mapping_accepts_q_prefix_and_trailing_punctuation():
    md = normalize_markdown(PAGE)
    for value in ("Q2", "2", "2.", "2．"):
        out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": value})
        assert "![Q2 附图](" in out, value


def test_mapping_without_question_headers_falls_back_to_inline_labels():
    md = 'Intro text.\n\n<img src="images/a.jpg">\n\nMore text.'
    out = apply_labels_mapping(md, {"a.jpg": "7"})
    assert "![Q7 附图](" in out
    assert out.index("![Q7 附图](") < out.index("More text.")


def test_baked_labels_are_recovered_from_old_markdown():
    """Engines used to write the label into the text; recover it as a mapping."""
    md = (
        "13. Look.\n\n*[Q13 附图]*\n\n"
        '<img src="images/a.jpg">\n\n14. And.\n\n*[Q14 附图]*\n\n'
        '<img src="images/b.jpg">\n'
    )
    cleaned, mapping = extract_baked_labels(md)
    assert mapping == {"a.jpg": "13", "b.jpg": "14"}
    assert "附图" not in cleaned
    assert "13. Look." in cleaned and "14. And." in cleaned
    # The images themselves survive, ready to be re-labelled at render time.
    assert "images/a.jpg" in cleaned and "images/b.jpg" in cleaned


def test_baked_labels_are_recovered_from_the_alt_text_form():
    md = (
        "13. Look.\n\n![Q13 附图](images/a.jpg)\n\n"
        "14. And.\n\n![Q14 附图1](images/b.jpg)\n![Q14 附图2](images/c.jpg)\n"
    )
    cleaned, mapping = extract_baked_labels(md)
    assert mapping == {"a.jpg": "13", "b.jpg": "14", "c.jpg": "14"}
    assert "附图" not in cleaned
    assert "images/a.jpg" in cleaned and "images/c.jpg" in cleaned


def test_baked_label_extraction_is_a_no_op_without_labels():
    md = "13. Look.\n\n<img src=\"images/a.jpg\">\n"
    cleaned, mapping = extract_baked_labels(md)
    assert mapping == {}
    assert cleaned == md


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
    assert "![Q2 附图](" in out
    assert out.index("![Q2 附图](") > out.index("2. Solve for x.")


def test_drop_aliases_are_accepted():
    md = normalize_markdown(PAGE)
    for value in ("drop", "DROP", " remove ", "x", "✕", "-"):
        out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": value})
        assert "img_in_image_box_100_50_300_200.jpg" not in out, value


def test_a_question_number_is_never_treated_as_a_drop():
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": "1"})
    assert "img_in_image_box_100_50_300_200.jpg" in out
    assert "![Q1 附图](" in out


# --- the output format of a labelled figure ----------------------------------


def test_labelled_figure_uses_markdown_image_syntax():
    """The label belongs in the alt text, not on a separate line."""
    md = normalize_markdown(PAGE)
    out = apply_labels_mapping(md, {"img_in_image_box_100_50_300_200.jpg": "1"})
    assert "![Q1 附图](images/img_in_image_box_100_50_300_200.jpg)" in out
    assert "*[Q1 附图]*" not in out


def test_multiple_figures_for_one_question_are_numbered():
    md = normalize_markdown(
        '13. First.\n\n<img src="images/a.jpg">\n\n<img src="images/b.jpg">\n'
    )
    out = apply_labels_mapping(md, {"a.jpg": "13", "b.jpg": "13"})
    assert "![Q13 附图1](images/a.jpg)" in out
    assert "![Q13 附图2](images/b.jpg)" in out
    # A lone figure is not numbered.
    out_one = apply_labels_mapping(
        normalize_markdown('13. First.\n\n<img src="images/a.jpg">\n'), {"a.jpg": "13"}
    )
    assert "![Q13 附图](images/a.jpg)" in out_one
    assert "附图1" not in out_one


def test_formatting_is_idempotent():
    """Re-applying must not stack labels or renumber differently."""
    md = normalize_markdown(
        '13. First.\n\n<img src="images/a.jpg">\n\n<img src="images/b.jpg">\n'
    )
    once = apply_labels_mapping(md, {"a.jpg": "13", "b.jpg": "13"})
    twice = apply_labels_mapping(once, {"a.jpg": "13", "b.jpg": "13"})
    assert once.count("![Q13 附图1](") == 1
    assert twice.count("![Q13 附图1](") == 1
    assert twice.count("![Q13 附图2](") == 1
    assert "附图1 附图" not in twice


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


# --- one-line figure rows ----------------------------------------------------
#
# Two figures of one question read far better side by side, so they are moved
# into a flexbox div: markdown has no way to say "same line", and inside an HTML
# block a markdown image would print as text rather than render.

TWO_FIGURES = """16. Look at the two figures.

<div style="text-align: center;"><img src="images/a.jpg" alt="Image" width="10%" /></div>

<div style="text-align: center;">甲</div>

<div style="text-align: center;"><img src="images/b.jpg" alt="Image" width="5%" /></div>

17. Next question.
"""


def test_row_puts_the_figures_in_one_flex_div():
    out = wrap_figures_in_row(normalize_markdown(TWO_FIGURES), ["a.jpg", "b.jpg"])
    assert out.count("display:flex") == 1
    # Both figures are direct children of that one div, so they share a line.
    row = out[out.index("<div class=") : out.index("</div>")]
    assert row.count("<img") == 2
    assert 'src="images/a.jpg"' in row and 'src="images/b.jpg"' in row


def test_row_keeps_the_labels_and_the_surrounding_text():
    md = normalize_markdown("13. Q.\n\n![Q13 附图](images/a.jpg)\n\n![Q13 附图2](images/b.jpg)\n")
    out = wrap_figures_in_row(md, ["a.jpg", "b.jpg"])
    assert 'alt="Q13 附图"' in out and 'alt="Q13 附图2"' in out
    assert "13. Q." in out


def test_row_drops_the_wrappers_the_ocr_left_behind():
    out = wrap_figures_in_row(normalize_markdown(TWO_FIGURES), ["a.jpg", "b.jpg"])
    # No empty wrapper divs survive, and the row is not nested in the
    # text-align wrapper the OCR put around the first figure.
    assert "text-align: center;\">\n<div" not in out
    assert out.count("<div") == out.count("</div>")
    assert "甲" in out  # the caption between the figures is still there


def test_row_takes_the_figures_in_document_order_not_the_order_asked_for():
    out = wrap_figures_in_row(normalize_markdown(TWO_FIGURES), ["b.jpg", "a.jpg"])
    assert out.index("images/a.jpg") < out.index("images/b.jpg")


def test_row_lands_where_the_first_figure_was():
    md = normalize_markdown(TWO_FIGURES)
    out = wrap_figures_in_row(md, ["a.jpg", "b.jpg"])
    assert out.index("<div class=") < out.index("17. Next question.")
    assert out.index("16. Look at the two figures.") < out.index("<div class=")


def test_a_row_is_never_nested_inside_another_row():
    """Moving figures out of a row puts the new row beside it, not inside it."""
    md = normalize_markdown(
        "13. Q.\n\n![Q13 附图](images/a.jpg)\n\n"
        "![Q13 附图2](images/b.jpg)\n\n![Q13 附图3](images/c.jpg)\n"
    )
    first = wrap_figures_in_row(md, ["a.jpg", "b.jpg"])
    second = wrap_figures_in_row(first, ["b.jpg", "c.jpg"])

    assert second.count("display:flex") == 2
    rows = [open_tag for _s, _e, open_tag, _i in iter_div_blocks(second)]
    assert len(rows) == 2, "two sibling rows, neither inside the other"
    assert "\x00" not in second, "no leftover placeholder"

    # Moving every figure out again collapses it to a single row.
    third = wrap_figures_in_row(second, ["a.jpg", "b.jpg", "c.jpg"])
    assert third.count("display:flex") == 1


def test_row_keeps_the_ocr_caption_between_the_figures():
    out = wrap_figures_in_row(normalize_markdown(TWO_FIGURES), ["a.jpg", "b.jpg"])
    assert out.count("<div") == out.count("</div>") == 2  # the row and 甲's caption
    assert "甲" in out


def test_row_needs_two_figures():
    md = normalize_markdown(TWO_FIGURES)
    for figures in (["a.jpg"], ["a.jpg", "gone.jpg"]):
        try:
            wrap_figures_in_row(md, figures)
        except ValueError as exc:
            assert "two" in str(exc), exc
        else:
            raise AssertionError(f"expected a ValueError for {figures}")
    try:
        wrap_figures_in_row(md, [])
    except ValueError as exc:
        assert "no figures" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an empty selection")


def test_row_is_idempotent():
    """Re-running it must not wrap the row inside another row."""
    md = normalize_markdown(TWO_FIGURES)
    once = wrap_figures_in_row(md, ["a.jpg", "b.jpg"])
    twice = wrap_figures_in_row(once, ["a.jpg", "b.jpg"])
    assert once == twice
    assert figures_in_rows(twice) == ["a.jpg", "b.jpg"]


def test_figures_in_rows_finds_a_hand_written_row():
    """A row the user typed by hand counts, even without our class name."""
    hand = (
        '13. Q.\n\n<div style="display: flex; gap: 8px;">\n'
        '<img src="images/x.jpg">\n<img src="images/y.jpg">\n</div>\n'
    )
    assert figures_in_rows(hand) == ["x.jpg", "y.jpg"]
    assert figures_in_rows(normalize_markdown(TWO_FIGURES)) == []


def test_a_plain_wrapper_div_is_not_a_row():
    plain = '<div style="text-align: center;"><img src="images/a.jpg"></div>'
    assert is_figure_row('<div style="text-align: center;">', '<img src="images/a.jpg">') is False


def test_a_row_of_one_is_still_a_row():
    """Nothing wrong with it — the user may be spacing a lone figure out."""
    single = '<div style="display:flex;"><img src="images/a.jpg"></div>'
    assert figures_in_rows(single) == ["a.jpg"]


# --- a hand-edited page ------------------------------------------------------


def test_page_markdown_prefers_the_hand_edited_text():
    mapping = json.dumps({"img_in_image_box_100_50_300_200.jpg": "2"})
    edited = "1. As I typed it.\n"
    assert page_markdown(PAGE, mapping, edited) == edited


# --- figures outlined by hand ------------------------------------------------


def test_stripping_figures_removes_every_reference():
    text = (
        "13. Find the area.\n\n"
        '<img src="images/a.jpg" />\n\n'
        "![Q13 附图](images/b.jpg)\n\n"
        "14. Next.\n"
    )
    out = strip_figures(text)
    assert "a.jpg" not in out and "b.jpg" not in out
    assert "13. Find the area." in out and "14. Next." in out
    assert "\n\n\n" not in out, "the gaps left behind are tidied up"


def test_stripping_figures_from_a_page_with_none_leaves_it_alone():
    assert strip_figures("13. Text only.\n") == "13. Text only.\n"
    assert strip_figures("") == ""


def test_stripping_figures_is_not_fooled_by_an_empty_alt_text():
    """`![](...)` with nothing between the brackets is still a figure."""
    assert "images" not in strip_figures("![](images/c.jpg)\n")


def test_a_crop_reference_is_an_ordinary_markdown_image():
    assert append_crops("13. Q.\n", ["p1_manual_box_1_2_3_4.jpg"]) == (
        "13. Q.\n\n![Image](images/p1_manual_box_1_2_3_4.jpg)\n"
    )


# --- how large a figure looks in the paper ------------------------------------
#
# A crop keeps the photo's full resolution, which is right for zooming into and
# wrong for reading. These bound the display only; the file is untouched.


def test_nothing_known_about_a_figure_leaves_it_at_the_ceiling():
    out = style_figures("![Q13 附图](images/a.jpg)\n", 800, 1100)
    assert out == (
        '<img src="images/a.jpg" alt="Q13 附图" width="800" '
        'style="max-width:800px; max-height:1100px">\n'
    )


def test_a_size_the_user_wrote_themselves_is_left_alone():
    out = style_figures(
        '<img src="images/a.jpg" style="max-width:200px">',
        800,
        1100,
        size_for=lambda ref: (400, 300),
    )
    assert "max-width:200px" in out
    assert "width=" not in out, "a size already written is never overridden"


def test_a_rows_own_height_cap_survives():
    rowed = '<div class="p2md-row"><img src="images/a.jpg" style="max-height:200px"></div>'
    out = style_figures(rowed, 800, 1100, size_for=lambda ref: (400, 300))
    assert "max-height:200px" in out, "the row must stay one line"
    assert "max-height:1100px" not in out


def test_a_figure_that_carries_a_percentage_width_keeps_it():
    out = style_figures(
        '<img src="images/a.jpg" width="10%">', 800, 1100, size_for=lambda ref: (400, 300)
    )
    assert 'width="10%"' in out


def test_sizing_the_display_does_not_change_the_text_around_it():
    text = "13. Q.\n\n![Q13 附图](images/a.jpg)\n\n14. Next.\n"
    out = style_figures(text, 800, 1100, size_for=lambda ref: (400, 300))
    assert out.startswith("13. Q.\n")
    assert "14. Next." in out


def test_a_blank_written_outside_math_is_put_back_inside():
    """Models leave the fill-in blank sitting in the prose, where KaTeX never
    sees it and the reader gets literal backslashes."""
    out = wrap_bare_latex("1. 下列说法正确的是（\\underline{\\hspace{2em}}）\n")
    assert out == "1. 下列说法正确的是（\\(\\underline{\\hspace{2em}}\\)）\n"


def test_several_blanks_on_one_line_do_not_run_into_each_other():
    out = wrap_bare_latex("班级：\\underline{\\hspace{2em}} 姓名：\\underline{\\hspace{2em}}\n")
    assert out.count("\\(") == 2 and out.count("\\)") == 2
    assert "$" not in out, "$$ would be ambiguous to read back"


def test_a_blank_already_in_math_is_left_alone():
    for text in (
        "计算：$a^9 = \\underline{\\hspace{2em}}$.\n",
        "$$ x = \\underline{\\hspace{2em}} $$\n",
        "则 $x+y=\\underline{\\hspace{3em}}$\n",
    ):
        assert wrap_bare_latex(text) == text


def test_wider_blanks_are_wrapped_too():
    out = wrap_bare_latex("系数是 \\underline{\\hspace{3.5em}}.\n")
    assert "\\(\\underline{\\hspace{3.5em}}\\)" in out


def test_text_with_no_bare_latex_is_untouched():
    text = "整式 $\\frac{3xy}{2}$ 的系数是 3.\n"
    assert wrap_bare_latex(text) == text


def test_no_cap_is_no_change():
    text = '<img src="images/a.jpg" alt="Image">'
    assert style_figures(text, 0, 0) == text


def test_a_plain_width_attribute_is_written_as_well_as_the_style():
    """Some Markdown viewers strip inline CSS, so the size is stated twice."""
    out = style_figures(
        "![Q13 附图](images/a.jpg)\n", 800, 1100, size_for=lambda ref: (400, 300)
    )
    assert 'width="400"' in out
    assert "max-width:400px" in out


def test_a_small_figure_is_not_blown_up_to_the_cap():
    out = style_figures(
        "![Q13 附图](images/a.jpg)\n", 800, 1100, size_for=lambda ref: (200, 150)
    )
    assert 'width="200"' in out, "the size it was drawn at, not the ceiling"


def test_a_sizer_that_knows_nothing_falls_back_to_the_ceiling():
    out = style_figures(
        "![Q13 附图](images/a.jpg)\n", 800, 1100, size_for=lambda ref: (None, None)
    )
    assert 'width="800"' in out
    assert "max-width:800px" in out


def test_the_sizer_is_asked_for_the_reference_it_was_given():
    seen = []

    def size_for(ref):
        seen.append(ref)
        return 300, 200

    style_figures('<img src="images/a.jpg">', 800, 1100, size_for=size_for)
    style_figures("![x](https://i.ibb.co/abc123.jpg)", 800, 1100, size_for=size_for)
    assert seen == ["images/a.jpg", "https://i.ibb.co/abc123.jpg"]


def test_the_ceiling_clips_a_figure_bigger_than_a_whole_page():
    out = style_figures(
        '<img src="images/a.jpg">', 800, 1100, size_for=lambda ref: (2400, 3000)
    )
    assert 'width="800"' in out


def test_a_crop_can_carry_its_own_size_for_hand_edited_text():
    """Hand-edited text is used exactly as saved, so the size goes in the tag."""
    out = append_crops("16. Mine.\n", ["a.jpg"], 480, 640)
    assert out == (
        "16. Mine.\n\n"
        '<img src="images/a.jpg" alt="Image" style="max-width:480px; max-height:640px">\n'
    )
    # ...and with no cap it stays plain markdown.
    assert "![Image](images/a.jpg)" in append_crops("16. Mine.\n", ["a.jpg"])


def test_appending_crops_to_nothing_still_works():
    """A page whose OCR found no text at all can still carry a drawn figure."""
    assert append_crops("", ["a.jpg"]) == "![Image](images/a.jpg)\n"
    assert append_crops("13. Q.\n", []) == "13. Q.\n"


def test_several_crops_keep_their_order():
    out = append_crops("13. Q.\n", ["a.jpg", "b.jpg"])
    assert out.index("a.jpg") < out.index("b.jpg")
    assert out.count("![Image]") == 2


def test_a_crop_already_in_the_text_is_not_appended_twice():
    """It went into a one-line row, so the row is where it belongs."""
    rowed = '<div class="p2md-row" style="display:flex;">\n<img src="images/a.jpg">\n</div>\n'
    assert append_crops(rowed, ["a.jpg", "b.jpg"]).count("images/a.jpg") == 1
    assert append_crops(rowed, ["a.jpg"]).count("images/a.jpg") == 1
    # ...while a crop that is not there yet still gets added.
    assert "images/b.jpg" in append_crops(rowed, ["a.jpg", "b.jpg"])


def test_stripping_a_figure_removes_it_in_either_syntax():
    text = (
        '<div class="p2md-row" style="display:flex;">\n'
        '<img src="images/a.jpg" alt="Q13 附图">\n<img src="images/b.jpg">\n</div>\n'
    )
    out = strip_figure(text, "a.jpg")
    assert "a.jpg" not in out and "b.jpg" in out

    markdown = "13. Q.\n\n![Q13 附图](images/a.jpg)\n\n![Q14 附图](images/b.jpg)\n"
    out = strip_figure(markdown, "a.jpg")
    assert "a.jpg" not in out and "images/b.jpg" in out
    assert "13. Q." in out

    # A figure nobody mentions changes nothing.
    assert strip_figure(markdown, "z.jpg") == markdown


def test_page_markdown_falls_back_to_the_mapping_when_there_is_no_edit():
    mapping = json.dumps({"img_in_image_box_100_50_300_200.jpg": "2"})
    md = normalize_markdown(PAGE)
    assert page_markdown(md, mapping) == apply_labels_mapping(md, json.loads(mapping))
    # Blank edits are no edits.
    assert page_markdown(md, mapping, "   \n") == apply_labels_mapping(md, json.loads(mapping))
    assert page_markdown(md, mapping, None) == apply_labels_mapping(md, json.loads(mapping))
