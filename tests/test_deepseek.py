"""Tests for the DeepSeek cleanup helpers — no network calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import deepseek  # noqa: E402

PAGE = """1. Solve for $x$.

*[Q13 附图]*

<img src="https://i.ibb.co/Kxx4GM8Z/fig.jpg" alt="Image" width="41%">

14. Explain the result.

<img src="https://i.ibb.co/abcd1234/second.jpg">
"""

# A page with a single, labelled figure — used when a test should not have to
# account for the second figure being "dropped" by the model.
ONE_FIG = """1. Solve for $x$.

*[Q13 附图]*

<img src="https://i.ibb.co/Kxx4GM8Z/fig.jpg" alt="Image" width="41%">
"""


def test_protect_replaces_every_figure_with_a_marker():
    protected, mapping = deepseek.protect_figures(PAGE)
    assert "i.ibb.co" not in protected
    assert len(mapping) == 2
    assert "[[FIG1-Q13]]" in protected
    assert "[[FIG2]]" in protected


def test_protect_swallows_the_label_into_the_marker():
    protected, mapping = deepseek.protect_figures(PAGE)
    # The "*[Q13 附图]*" label becomes part of the marker, so the model cannot
    # reattach it to the wrong question.
    assert "附图" not in protected
    assert mapping["[[FIG1-Q13]]"]["label"] == "*[Q13 附图]*"


def test_round_trip_is_lossless():
    protected, mapping = deepseek.protect_figures(PAGE)
    restored, missing = deepseek.restore_figures(protected, mapping)
    assert missing == []
    assert "i.ibb.co/Kxx4GM8Z/fig.jpg" in restored
    assert "i.ibb.co/abcd1234/second.jpg" in restored
    assert "*[Q13 附图]*" in restored


def test_model_move_of_a_marker_is_respected():
    """The model relocating a marker is the whole point — the figure follows it."""
    protected, mapping = deepseek.protect_figures(PAGE)
    moved = "1. Solve for $x$.\n\n[[FIG1-Q13]]\n\n14. Explain the result.\n\n[[FIG2]]\n"
    restored, missing = deepseek.restore_figures(moved, mapping)
    assert missing == []
    # Figure 1 now sits above question 14, where the model put it.
    assert restored.index("Kxx4GM8Z") < restored.index("14. Explain")


def test_dropped_markers_are_recovered_not_lost():
    protected, mapping = deepseek.protect_figures(PAGE)
    # The model silently dropped both markers.
    restored, missing = deepseek.restore_figures("1. Solve for $x$.\n", mapping)
    assert len(missing) == 2
    assert "i.ibb.co/Kxx4GM8Z/fig.jpg" in restored
    assert "i.ibb.co/abcd1234/second.jpg" in restored


def test_marker_with_rewritten_question_number_still_resolves():
    """Model changed Q13 -> Q12; we match on figure id, so the real label wins."""
    protected, mapping = deepseek.protect_figures(ONE_FIG)
    restored, missing = deepseek.restore_figures("text\n\n[[FIG1-Q12]]\n", mapping)
    assert missing == []
    assert "*[Q13 附图]*" in restored


def test_unknown_marker_is_stripped():
    protected, mapping = deepseek.protect_figures(PAGE)
    restored, _ = deepseek.restore_figures("text\n\n[[FIG99-Q1]]\n", mapping)
    assert "FIG99" not in restored


def test_lenient_marker_matching_tolerates_whitespace():
    protected, mapping = deepseek.protect_figures(ONE_FIG)
    restored, missing = deepseek.restore_figures("x\n\n[[ FIG 1 - Q13 ]]\n", mapping)
    assert missing == []
    assert "Kxx4GM8Z" in restored


def test_no_figures_is_a_no_op():
    protected, mapping = deepseek.protect_figures("Just text, no figures.")
    assert protected == "Just text, no figures."
    assert mapping == {}
    restored, missing = deepseek.restore_figures(protected, mapping)
    assert restored == protected and missing == []


def test_split_keeps_short_documents_in_one_chunk():
    assert deepseek.split_into_chunks("small", limit=1000) == ["small"]


def test_split_prefers_page_boundaries():
    doc = "<!-- page 1 -->\n" + ("a" * 40) + "\n<!-- page 2 -->\n" + ("b" * 40)
    chunks = deepseek.split_into_chunks(doc, limit=60)
    assert len(chunks) >= 2
    # Page markers are stripped by the splitter.
    assert all("<!-- page" not in chunk for chunk in chunks)
    assert "".join(chunks).count("a") == 40
    assert "".join(chunks).count("b") == 40


def test_split_hard_wraps_an_oversized_page():
    doc = "x" * 250
    chunks = deepseek.split_into_chunks(doc, limit=100)
    assert len(chunks) >= 3
    assert sum(len(chunk) for chunk in chunks) == 250


def test_empty_key_raises_a_clear_error():
    saved = deepseek.config.DEEPSEEK_API_KEY
    deepseek.config.DEEPSEEK_API_KEY = ""
    try:
        deepseek.polish_markdown("some text")
    except deepseek.DeepSeekError as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:
        raise AssertionError("expected DeepSeekError")
    finally:
        deepseek.config.DEEPSEEK_API_KEY = saved
