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


# --- streaming transport (mocked; no network) --------------------------------


class _FakeStream:
    """Stands in for the streamed response from requests.post."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.ok = status_code == 200
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


def _patch_post(monkeypatch_lines, blocking_body=None, calls=None):
    def fake_post(url, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        if kwargs.get("stream"):
            return _FakeStream(monkeypatch_lines)
        return _FakeBlocking(blocking_body)

    deepseek.requests.post = fake_post


class _FakeBlocking:
    status_code = 200
    ok = True
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def test_chat_streams_and_reports_character_progress():
    saved = deepseek.requests.post
    seen: list[int] = []
    _patch_post(
        [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
            "data: [DONE]",
        ]
    )
    try:
        content, usage = deepseek._chat("text", "model", on_chars=seen.append)
        assert content == "Hello"
        assert usage["total_tokens"] == 7
        assert seen == [3, 5]
    finally:
        deepseek.requests.post = saved


def test_chat_ignores_non_data_and_malformed_lines():
    saved = deepseek.requests.post
    _patch_post(
        [
            "",
            ": keep-alive",
            "data: not json",
            'data: {"choices":[{"delta":{"content":"x"}}]}',
            "data: [DONE]",
        ]
    )
    try:
        content, _ = deepseek._chat("text", "model")
        assert content == "x"
    finally:
        deepseek.requests.post = saved


def test_chat_falls_back_to_blocking_when_stream_is_empty():
    saved = deepseek.requests.post
    calls: list[dict] = []
    _patch_post(
        ["data: [DONE]"],
        blocking_body={"choices": [{"message": {"content": "from fallback"}}], "usage": {"total_tokens": 3}},
        calls=calls,
    )
    try:
        content, usage = deepseek._chat("text", "model")
        assert content == "from fallback"
        assert usage["total_tokens"] == 3
        assert len(calls) == 2, "expected a streaming attempt then a blocking retry"
        assert calls[0].get("stream") is True
        assert not calls[1].get("stream"), "the retry must not stream"
    finally:
        deepseek.requests.post = saved


def test_chat_counts_reasoning_content_towards_progress():
    """deepseek-flash thinks for a long time before emitting answer text."""
    saved = deepseek.requests.post
    seen: list[int] = []
    _patch_post(
        [
            'data: {"choices":[{"delta":{"reasoning_content":"thinking hard"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"some more"}}]}',
            'data: {"choices":[{"delta":{"content":"Answer"}}]}',
            "data: [DONE]",
        ]
    )
    try:
        content, _ = deepseek._chat("text", "model", on_chars=seen.append)
        assert content == "Answer"
        # Progress advanced during the reasoning phase, not only at the end.
        assert len(seen) == 3
        assert seen[0] == len("thinking hard")
        assert seen[1] == len("thinking hard") + len("some more")
        assert seen[2] == seen[1] + len("Answer")
    finally:
        deepseek.requests.post = saved


def test_soft_fraction_is_monotonic_and_never_completes():
    fractions = [deepseek._soft_fraction(n, 1000) for n in range(0, 6000, 250)]
    assert fractions[0] == 0.0
    assert fractions == sorted(fractions), "must never go backwards"
    assert all(f < 1.0 for f in fractions), "must never claim completion on its own"
    assert fractions[-1] > 0.9, "should end up near the end"
    assert deepseek._soft_fraction(0, 0) == 0.0


def test_chat_reports_bad_key_clearly():
    saved = deepseek.requests.post

    def fake_post(url, **kwargs):
        return _FakeStream([], status_code=401)

    deepseek.requests.post = fake_post
    try:
        deepseek._chat("text", "model")
    except deepseek.DeepSeekError as exc:
        assert "401" in str(exc)
    else:
        raise AssertionError("expected DeepSeekError")
    finally:
        deepseek.requests.post = saved
