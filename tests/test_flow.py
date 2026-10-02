"""End-to-end test of the web flow with OCR stubbed out.

Runs the real FastAPI app, real SQLite store, real worker thread and real
markdown pipeline — only the PaddleOCR-VL call is replaced, so this test works
on a machine where Paddle is not installed yet.
"""

import io
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# Isolate the data directory *before* `app.config` is imported.
import _env  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app import config, ocr  # noqa: E402
from app.main import app  # noqa: E402

# Guard against this suite ever running on the real database again.
assert "paper2md-test-" in str(config.DATA_DIR), (
    f"tests must use a throwaway DATA_DIR, got {config.DATA_DIR}"
)

PAGE_MARKDOWN = """1. Simplify the expression.

<img src="images/fig1.jpg" />

2. Solve for $x$.
"""


def _jpeg(label: str) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (300, 400), "white")
    ImageDraw.Draw(image).text((20, 20), label, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _fake_run_ocr(image_path, work_dir, images_dir, prefix=""):
    """Stand-in for PaddleOCR-VL: writes one figure and returns markdown.

    Mirrors the real contract — the figure on disk is prefixed and the markdown
    references that prefixed name.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}fig1.jpg"
    (images_dir / name).write_bytes(_jpeg("figure"))
    markdown = PAGE_MARKDOWN.replace("images/fig1.jpg", f"images/{name}")
    return markdown, [name], {}


ocr.run_ocr = _fake_run_ocr


def test_suite_uses_a_throwaway_data_directory():
    """Guards against the suite ever being pointed at the real database."""
    assert "paper2md-test-" in str(config.DATA_DIR), config.DATA_DIR
    assert config.DATA_DIR.is_dir()


def _wait(client, task_id, timeout=20.0):
    deadline = time.time() + timeout
    data = {}
    while time.time() < deadline:
        data = client.get(f"/api/tasks/{task_id}").json()
        if data["status"] in ("done", "partial", "failed"):
            return data
        time.sleep(0.1)
    raise AssertionError(f"task never settled: {data}")


def test_full_flow():
    with TestClient(app) as client:
        # --- create ---
        response = client.post("/api/tasks", json={"title": "Midterm 2024"})
        assert response.status_code == 201
        task_id = response.json()["task_id"]

        # --- upload two pages in one request ---
        response = client.post(
            f"/api/tasks/{task_id}/pages",
            files=[
                ("files", ("page1.jpg", _jpeg("page one"), "image/jpeg")),
                ("files", ("page2.jpg", _jpeg("page two"), "image/jpeg")),
            ],
        )
        assert response.status_code == 201, response.text
        assert len(response.json()["accepted"]) == 2

        # --- wait for OCR ---
        status = _wait(client, task_id)
        assert status["status"] == "done", status
        assert [page["status"] for page in status["pages"]] == ["done", "done"]

        # --- a page's figures and questions are visible ---
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["figures"] == ["p1_fig1.jpg"]
        assert page["questions"] == ["1", "2"]

        # --- figure files are served ---
        image = client.get(f"/tasks/{task_id}/images/p1_fig1.jpg")
        assert image.status_code == 200
        raw = client.get(f"/tasks/{task_id}/pages/0/raw")
        assert raw.status_code == 200

        # --- map a figure to question 2 and check it moved ---
        response = client.put(
            f"/api/tasks/{task_id}/pages/0/mapping", json={"p1_fig1.jpg": "2"}
        )
        assert response.status_code == 200
        labelled = response.json()["markdown"]
        preview = response.json()["preview"]
        tag = figure_tag("p1_fig1.jpg", "2")
        # The text keeps the figure as a markdown image; the paper shows it with
        # a display size, in the same place.
        assert "![Q2 附图](images/p1_fig1.jpg)" in labelled
        assert tag in preview
        assert preview.index(tag) > preview.index("2. Solve for $x$.")

        # --- download: page wrappers present, figure inlined as base64 ---
        markdown = client.get(f"/tasks/{task_id}/download").text
        assert "<!-- page 1 -->" in markdown
        assert "<!-- page 2 -->" in markdown
        assert "data:image/jpeg;base64," in markdown
        assert "images/p1_fig1.jpg" not in markdown

        # --- HTML pages render ---
        assert client.get("/").status_code == 200
        assert client.get(f"/tasks/{task_id}").status_code == 200
        edit = client.get(f"/tasks/{task_id}/edit?page=0")
        assert edit.status_code == 200
        assert "Figures on this page" in edit.text

        # --- health ---
        assert client.get("/api/health").json()["status"] == "ok"

        # --- retry rejects a healthy page ---
        assert client.post(f"/api/tasks/{task_id}/pages/0/retry").status_code == 409


def test_engine_suggested_mapping_is_stored_and_rendered():
    """The hybrid engine's question numbers must land in the page mapping."""
    original = ocr.run_ocr

    def with_suggestion(image_path, work_dir, images_dir, prefix=""):
        markdown, figures, _ = original(image_path, work_dir, images_dir, prefix)
        return markdown, figures, {figures[0]: "16"} if figures else {}

    ocr.run_ocr = with_suggestion
    try:
        with TestClient(app) as client:
            task_id = _make_ready_task(client, "suggested")
            page = client.get(f"/api/tasks/{task_id}/pages/0").json()
            # Stored as a mapping, not baked into the text.
            assert page["mapping"] == {"p1_fig1.jpg": "16"}
            assert "附图" not in page["markdown"]
            # ...and the renderer places it in the new markdown image format.
            assert figure_tag("p1_fig1.jpg", "16") in page["preview"]
    finally:
        ocr.run_ocr = original


def test_rejects_unsupported_and_unreadable_files():
    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "bad"}).json()["task_id"]
        response = client.post(
            f"/api/tasks/{task_id}/pages",
            files=[
                ("files", ("notes.txt", b"hello", "text/plain")),
                ("files", ("broken.jpg", b"not an image", "image/jpeg")),
            ],
        )
        assert response.status_code == 400
        body = response.json()
        assert body["accepted"] == []
        assert len(body["skipped"]) == 2


def test_failed_page_can_be_retried():
    calls = {"count": 0}
    original = ocr.run_ocr

    def flaky(image_path, work_dir, images_dir, prefix=""):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("model exploded")
        return original(image_path, work_dir, images_dir, prefix)

    ocr.run_ocr = flaky
    try:
        with TestClient(app) as client:
            task_id = client.post("/api/tasks", json={"title": "flaky"}).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
            )
            status = _wait(client, task_id)
            assert status["status"] == "failed"
            assert "model exploded" in status["pages"][0]["error"]

            assert client.post(f"/api/tasks/{task_id}/pages/0/retry").status_code == 200
            status = _wait(client, task_id)
            assert status["status"] == "done", status
    finally:
        ocr.run_ocr = original


def test_delete_task_removes_rows_and_files():
    from app import config

    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "to delete"}).json()["task_id"]
        client.post(
            f"/api/tasks/{task_id}/pages",
            files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
        )
        _wait(client, task_id)
        task_dir = config.task_dir(task_id)
        assert task_dir.is_dir()

        assert client.delete(f"/api/tasks/{task_id}").status_code == 200
        assert client.get(f"/api/tasks/{task_id}").status_code == 404
        assert client.get(f"/api/tasks/{task_id}/edit?page=0").status_code == 404
        assert not task_dir.exists()
        assert all(task["id"] != task_id for task in client.get("/api/tasks").json()["tasks"])


def test_page_detail_reports_labels_and_questions():
    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "labels"}).json()["task_id"]
        client.post(
            f"/api/tasks/{task_id}/pages",
            files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
        )
        _wait(client, task_id)

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["mapping"] == {}
        assert page["questions"] == ["1", "2"]
        # Nothing mapped yet, so the figure stays put and carries no label.
        assert "附图" not in page["labelled_markdown"]

        client.put(f"/api/tasks/{task_id}/pages/0/mapping", json={"p1_fig1.jpg": "2"})
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["mapping"] == {"p1_fig1.jpg": "2"}
        assert figure_tag("p1_fig1.jpg", "2") in page["preview"]

        # Blank values are dropped rather than stored.
        client.put(f"/api/tasks/{task_id}/pages/0/mapping", json={"p1_fig1.jpg": "  "})
        assert client.get(f"/api/tasks/{task_id}/pages/0").json()["mapping"] == {}


# --- hand-editing the markdown and putting figures on one line ----------------

TWO_FIGURE_MARKDOWN = """16. Look at both figures.

<img src="images/figA.jpg" alt="Image" width="10%" />

<img src="images/figB.jpg" alt="Image" width="5%" />

17. Next question.
"""


def figure_tag(name: str, question: str) -> str:
    """A mapped figure renders as an img tag carrying its question.

    An ``<img>`` rather than markdown image syntax, because that is what lets a
    figure carry a display size (FIGURE_MAX_WIDTH/HEIGHT) — and it is what a
    one-line row has always used.
    """
    return f'<img src="images/{name}" alt="Q{question} 附图"'

def _two_figure_task(client, title="layout"):
    """A ready task whose single page has two figures to lay out."""
    original = ocr.run_ocr

    def two_figures(image_path, work_dir, images_dir, prefix=""):
        images_dir.mkdir(parents=True, exist_ok=True)
        names = []
        for tag in ("A", "B"):
            name = f"{prefix}fig{tag}.jpg"
            (images_dir / name).write_bytes(_jpeg(tag))
            names.append(name)
        markdown = TWO_FIGURE_MARKDOWN.replace("images/figA.jpg", f"images/{names[0]}")
        markdown = markdown.replace("images/figB.jpg", f"images/{names[1]}")
        return markdown, names, {}

    # What the fake OCR names its crops: page 1 is "p1_".
    names = ["p1_figA.jpg", "p1_figB.jpg"]

    ocr.run_ocr = two_figures
    try:
        task_id = client.post("/api/tasks", json={"title": title}).json()["task_id"]
        client.post(
            f"/api/tasks/{task_id}/pages",
            files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
        )
        _wait(client, task_id)
        assert client.get(f"/api/tasks/{task_id}/pages/0").json()["figures"] == names
        return task_id, names
    finally:
        ocr.run_ocr = original


def test_hand_edited_markdown_becomes_the_page():
    with TestClient(app) as client:
        task_id, _names = _two_figure_task(client)

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["edited"] is False
        assert page["labelled_markdown"] == page["generated_markdown"]

        edited = "16. As I typed it.\n\n![note](images/figA.jpg)\n"
        response = client.put(
            f"/api/tasks/{task_id}/pages/0/markdown", json={"markdown": edited}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["edited"] is True
        assert body["markdown"] == edited

        # The page, the page download and the whole paper all use it.
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["labelled_markdown"] == edited
        assert page["generated_markdown"] != edited
        assert "As I typed it." in client.get(f"/tasks/{task_id}/pages/0/download").text
        assert "As I typed it." in client.get(f"/tasks/{task_id}/download").text

        # Reverting brings the generated text back.
        response = client.put(f"/api/tasks/{task_id}/pages/0/markdown", json={"reset": True})
        assert response.json()["edited"] is False
        assert "As I typed it." not in client.get(f"/tasks/{task_id}/download").text


def test_editing_normalises_the_text_on_the_way_in():
    with TestClient(app) as client:
        task_id, _names = _two_figure_task(client)
        client.put(
            f"/api/tasks/{task_id}/pages/0/markdown",
            json={"markdown": '16. Q.\n\n<img src="imgs/x.jpg" />\n'},
        )
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert 'src="images/x.jpg"' in page["labelled_markdown"]
        assert "/>" not in page["labelled_markdown"]


def test_an_empty_edit_reverts_instead_of_blanking_the_page():
    with TestClient(app) as client:
        task_id, _names = _two_figure_task(client)
        client.put(
            f"/api/tasks/{task_id}/pages/0/markdown", json={"markdown": "hand written\n"}
        )
        body = client.put(
            f"/api/tasks/{task_id}/pages/0/markdown", json={"markdown": "   \n"}
        ).json()
        assert body["edited"] is False
        assert "Look at both figures." in body["markdown"]


def test_figure_row_puts_two_figures_on_one_line_in_the_download():
    with TestClient(app) as client:
        task_id, names = _two_figure_task(client)

        response = client.post(
            f"/api/tasks/{task_id}/pages/0/figure-row", json={"figures": names}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["row_figures"] == names
        # One flex div holding both figures as direct children.
        assert body["markdown"].count("display:flex") == 1
        assert 'src="images/%s"' % names[0] in body["markdown"]
        assert body["markdown"].index(names[0]) < body["markdown"].index(names[1])

        # ...and that is what comes out of the download.
        markdown = client.get(f"/tasks/{task_id}/download").text
        assert "display:flex" in markdown
        assert markdown.count("data:image/jpeg;base64,") == 2

        # Reverting drops the row again.
        client.put(f"/api/tasks/{task_id}/pages/0/markdown", json={"reset": True})
        assert "display:flex" not in client.get(f"/tasks/{task_id}/download").text


def test_figure_row_needs_two_figures_and_reports_why():
    with TestClient(app) as client:
        task_id, names = _two_figure_task(client)
        path = f"/api/tasks/{task_id}/pages/0/figure-row"

        assert client.post(path, json={"figures": []}).status_code == 400
        response = client.post(path, json={"figures": [names[0]]})
        assert response.status_code == 400
        assert "two" in response.json()["detail"]
        assert client.post(path, json={"figures": [names[0], "gone.jpg"]}).status_code == 400


def test_a_mapping_change_does_not_disturb_a_hand_edited_page():
    """The hand-written text wins until it is reverted — and is then restored."""
    with TestClient(app) as client:
        task_id, names = _two_figure_task(client)
        client.put(f"/api/tasks/{task_id}/pages/0/markdown", json={"markdown": "1. Mine.\n"})

        client.put(f"/api/tasks/{task_id}/pages/0/mapping", json={names[0]: "16"})
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["mapping"] == {names[0]: "16"}
        assert page["labelled_markdown"] == "1. Mine.\n"

        # The numbers were kept, so reverting brings them back.
        reverted = client.put(
            f"/api/tasks/{task_id}/pages/0/markdown", json={"reset": True}
        ).json()
        assert figure_tag(names[0], "16") in reverted["preview"]


def test_a_reread_page_drops_its_hand_edited_markdown():
    """A fresh read replaces the OCR text an edit was written against.

    There is no route that re-reads a page that already succeeded, so this is
    asserted at the store: whatever gets the page there, the stale edit must not
    survive it.
    """
    from app import store

    with TestClient(app):
        task_id = store.create_task("reread")
        store.add_page(task_id, 0, "p.jpg")
        store.mark_page_done(task_id, 0, "1. Read.\n", 0)
        store.set_page_markdown(task_id, 0, "1. Mine.\n")
        assert store.get_page(task_id, 0)["edited_markdown"] == "1. Mine.\n"

        store.mark_page_done(task_id, 0, "1. Read again.\n", 0)
        page = store.get_page(task_id, 0)
        assert page["edited_markdown"] is None
        assert page["markdown"] == "1. Read again.\n"


def test_migration_gives_existing_databases_the_edited_markdown_column():
    from app import store

    with TestClient(app):
        columns = {row[1] for row in store._conn.execute("PRAGMA table_info(pages)")}
        assert "edited_markdown" in columns
        # Running it again on an up-to-date database changes nothing.
        store._migrate(store._conn)
        columns_again = {row[1] for row in store._conn.execute("PRAGMA table_info(pages)")}
        assert columns_again == columns


# --- reading one page again ---------------------------------------------------

SECOND_READ_MARKDOWN = "1. Simplify the expression.\n\n<img src=\"images/{name}\">\n\n9. Nine.\n"


def _recording_ocr():
    """A stub OCR that records every call and writes a figure each time."""
    calls = []

    def run(image_path, work_dir, images_dir, prefix="", engine=None):
        calls.append(engine)
        images_dir.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}fig1.jpg"
        (images_dir / name).write_bytes(_jpeg(f"read {len(calls)}"))
        return SECOND_READ_MARKDOWN.format(name=name), [name], {}

    return run, calls


def test_a_finished_page_can_be_read_again():
    run, calls = _recording_ocr()
    original = ocr.run_ocr
    ocr.run_ocr = run
    try:
        with TestClient(app) as client:
            task_id = client.post("/api/tasks", json={"title": "again"}).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
            )
            _wait(client, task_id)
            # `retry` stays for failed pages only.
            assert client.post(f"/api/tasks/{task_id}/pages/0/retry").status_code == 409

            response = client.post(f"/api/tasks/{task_id}/pages/0/reread")
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "queued"
            assert _wait(client, task_id)["status"] == "done"
    finally:
        ocr.run_ocr = original

    assert len(calls) == 2, "the page really was read twice"
    with TestClient(app) as client:
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["status"] == "done"
        assert "9. Nine." in page["markdown"], "the second read replaced the text"


def test_re_reading_uses_the_engine_the_page_was_pinned_to():
    with TestClient(app) as client:
        run, calls = _recording_ocr()
        original = ocr.run_ocr
        ocr.run_ocr = run
        try:
            task_id = client.post("/api/tasks", json={"title": "engine"}).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
            )
            _wait(client, task_id)
            # No pin: the stub is called the old four-argument way, so `calls`
            # records None rather than an engine name.
            assert calls == [None]

            response = client.post(
                f"/api/tasks/{task_id}/pages/0/reread", json={"engine": "paddle"}
            )
            assert response.status_code == 200, response.text
            assert response.json()["engine"] == "paddle"
            _wait(client, task_id)
            assert calls == [None, "paddle"]
        finally:
            ocr.run_ocr = original


def test_a_pinned_engine_is_kept_for_the_next_read():
    with TestClient(app) as client:
        run, calls = _recording_ocr()
        original = ocr.run_ocr
        ocr.run_ocr = run
        try:
            task_id = client.post("/api/tasks", json={"title": "pinned"}).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
            )
            _wait(client, task_id)

            # A read with no engine in the body leaves the pin alone.
            client.post(f"/api/tasks/{task_id}/pages/0/reread", json={"engine": "deepseek"})
            _wait(client, task_id)
            client.post(f"/api/tasks/{task_id}/pages/0/reread")
            _wait(client, task_id)
            assert calls[-3:] == [None, "deepseek", "deepseek"]
        finally:
            ocr.run_ocr = original


def test_an_unknown_engine_is_refused():
    with TestClient(app) as client:
        task_id, _names = _two_figure_task(client)
        response = client.post(
            f"/api/tasks/{task_id}/pages/0/reread", json={"engine": "tesseract"}
        )
        assert response.status_code == 400
        assert "engine" in response.json()["detail"]
        # A refused request must not have queued anything.
        assert client.get(f"/api/tasks/{task_id}/pages/0").json()["status"] == "done"


def test_a_page_being_read_cannot_be_queued_again():
    from app import store

    with TestClient(app) as client:
        task_id, _names = _two_figure_task(client)
        store.mark_page_processing(task_id, 0)
        response = client.post(f"/api/tasks/{task_id}/pages/0/reread")
        assert response.status_code == 409
        assert "processing" in response.json()["detail"]
        # Leave it as we found it: a page left "processing" is re-queued by the
        # next app start, which would run OCR behind this test's back.
        store._execute(
            "UPDATE pages SET status = 'done' WHERE task_id = ? AND page_index = 0",
            (task_id,),
        )


def test_a_missing_page_photo_is_reported():
    from app import config as app_config, store

    with TestClient(app) as client:
        task_id, _names = _two_figure_task(client)
        app_config.page_image_path(task_id, 0).unlink()

        response = client.post(f"/api/tasks/{task_id}/pages/0/reread")
        assert response.status_code == 400
        assert "upload it again" in response.json()["detail"]

        # `retry` only ever runs a failed page, and says so first.
        assert client.post(f"/api/tasks/{task_id}/pages/0/retry").status_code == 409
        store.mark_page_failed(task_id, 0, "boom")
        response = client.post(f"/api/tasks/{task_id}/pages/0/retry")
        assert response.status_code == 400
        assert "upload it again" in response.json()["detail"]


def test_re_reading_reports_figure_numbers_the_new_crops_orphaned():
    """Filenames carry crop coordinates, so a new read can rename every figure."""
    with TestClient(app) as client:
        run, calls = _recording_ocr()
        original = ocr.run_ocr
        ocr.run_ocr = run
        try:
            task_id = client.post("/api/tasks", json={"title": "orphan"}).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
            )
            _wait(client, task_id)

            client.put(f"/api/tasks/{task_id}/pages/0/mapping", json={"p1_fig1.jpg": "3"})
            page = client.get(f"/api/tasks/{task_id}/pages/0").json()
            assert page["stale_mapping"] == []

            # A second read whose layout names the figure differently.
            def renamed(image_path, work_dir, images_dir, prefix="", engine=None):
                markdown, figures, _ = run(
                    image_path, work_dir, images_dir, prefix, engine=engine
                )
                images_dir.mkdir(parents=True, exist_ok=True)
                (images_dir / "p1_fig9.jpg").write_bytes(_jpeg("renamed"))
                markdown = markdown.replace(figures[0], "p1_fig9.jpg")
                return markdown, ["p1_fig9.jpg"], {}

            ocr.run_ocr = renamed
            client.post(f"/api/tasks/{task_id}/pages/0/reread")
            _wait(client, task_id)
        finally:
            ocr.run_ocr = original

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        # The number survived, but nothing on the page answers to that name.
        assert page["mapping"] == {"p1_fig1.jpg": "3"}
        assert page["stale_mapping"] == ["p1_fig1.jpg"]


def test_re_reading_clears_a_hand_edited_page_and_the_cleaned_paper():
    from app import store

    with TestClient(app) as client:
        run, calls = _recording_ocr()
        original = ocr.run_ocr
        ocr.run_ocr = run
        try:
            task_id = client.post("/api/tasks", json={"title": "stale"}).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
            )
            _wait(client, task_id)
            client.put(
                f"/api/tasks/{task_id}/pages/0/markdown", json={"markdown": "1. Mine.\n"}
            )
            store.set_polished_markdown(task_id, "cleaned", {"model": "fake"})

            client.post(f"/api/tasks/{task_id}/pages/0/reread")
            _wait(client, task_id)
        finally:
            ocr.run_ocr = original

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["edited"] is False
        assert "1. Mine." not in page["labelled_markdown"]
        # The cleaned paper described the old text.
        assert store.get_polished_markdown(task_id) == (None, None)
        assert client.get(f"/tasks/{task_id}/download?variant=polished").status_code == 400


# --- outlining a figure the OCR missed ---------------------------------------

# A photo with three distinguishable bands, so a crop can be proved correct by
# looking at the pixels that came back.
def _banded_page() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 600, 300), fill=(255, 0, 0))     # top: red
    draw.rectangle((0, 300, 600, 600), fill=(0, 255, 0))   # middle: green
    draw.rectangle((0, 600, 600, 900), fill=(0, 0, 255))   # bottom: blue
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=98)
    return buffer.getvalue()


def _crop_task(client, title="outlined"):
    task_id = client.post("/api/tasks", json={"title": title}).json()["task_id"]
    client.post(
        f"/api/tasks/{task_id}/pages",
        files=[("files", ("page.jpg", _banded_page(), "image/jpeg"))],
    )
    _wait(client, task_id)
    return task_id


def _photo_page(size=(3072, 4096)):
    """A page photo the size a phone actually takes, so it gets downscaled."""
    from PIL import Image

    image = Image.new("RGB", size, (250, 250, 245))
    for y in range(0, size[1], 64):
        for x in range(0, size[0], 64):
            image.paste((40, 90, 160), (x, y, x + 40, y + 40))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=70)
    return out.getvalue()


def _engine_figure_task(client, title="engine+hand", photo=None):
    """A page with one figure the engine found, ready to draw another on.

    This is what ``ocr.run_ocr_hybrid`` really does: prepare a downscaled copy
    for the model, find a box on *that* copy, and crop the figure out of it —
    which is why the box in the name is in prepared pixels, not page-photo
    pixels.
    """
    from PIL import Image

    original = ocr.run_ocr

    def engine_one(image_path, work_dir, images_dir, prefix=""):
        images_dir.mkdir(parents=True, exist_ok=True)
        prepared = ocr.prepare_input(image_path, work_dir / "input.jpg")
        with Image.open(prepared) as small:
            width, height = small.size
        box = (
            round(width * 0.25),
            round(height * 0.25),
            round(width * 0.58),
            round(height * 0.55),
        )
        with Image.open(prepared) as page:
            crop = page.crop(box)
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        name = f"{prefix}img_in_image_box_{box[0]}_{box[1]}_{box[2]}_{box[3]}.jpg"
        crop.save(images_dir / name, format="JPEG", quality=92)
        markdown = f'13. Draw this.\n\n<img src="images/{name}">\n'
        return markdown, [name], {}

    ocr.run_ocr = engine_one
    try:
        task_id = client.post("/api/tasks", json={"title": title}).json()["task_id"]
        client.post(
            f"/api/tasks/{task_id}/pages",
            files=[("files", ("page.jpg", photo or _photo_page(), "image/jpeg"))],
        )
        _wait(client, task_id)
    finally:
        ocr.run_ocr = original
    return task_id


def test_a_figure_the_engine_found_is_sized_by_the_box_it_was_cut_from():
    """Auto crops are cut from the downscaled copy, but the size of the paper
    must be the size of the rectangle either way."""
    from PIL import Image

    from app import store

    with TestClient(app) as client:
        task_id = _engine_figure_task(client)
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        name = page["figures"][0]
        assert name, page["figures"]

        with Image.open(config.page_image_path(task_id, 0)) as photo:
            photo_size = photo.size
        assert max(photo_size) > config.OCR_MAX_DIM, "a real photo, so it was downscaled"

        with Image.open(config.images_dir(task_id) / name) as crop:
            natural = crop.size[0]

        box = store.box_from_name(name)
        prepared_width = round(photo_size[0] * config.OCR_MAX_DIM / max(photo_size))
        expected = round((box[2] - box[0]) / prepared_width * config.FIGURE_PAGE_WIDTH)
        line = [l for l in page["preview"].splitlines() if name in l][0]
        assert f'width="{expected}"' in line, line
        assert expected < natural, "the cut is what the model saw, the paper is smaller"


def test_a_hand_drawn_figure_sits_alongside_the_ones_the_engine_found():
    """Manual is the fallback, not a replacement: both kinds end up in the
    paper, and the drawn one is the only one that is removed outright."""
    with TestClient(app) as client:
        task_id = _engine_figure_task(client)
        before = client.get(f"/api/tasks/{task_id}/pages/0").json()
        engine_figure = before["figures"][0]

        drawn = client.post(
            f"/api/tasks/{task_id}/pages/0/crop",
            json={"x": 0.6, "y": 0.6, "w": 0.3, "h": 0.2, "question": "13"},
        ).json()["filename"]

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert sorted(page["figures"]) == sorted([engine_figure, drawn])
        assert page["crops"] == [drawn], "only the drawn one is hand-made"
        assert page["mapping"] == {drawn: "13"}
        # The engine's figure is still placed under the question too.
        assert engine_figure in page["preview"]
        assert drawn in page["preview"]

        # And both are sized by their own rectangle.
        for figure in page["figures"]:
            line = [l for l in page["preview"].splitlines() if figure in l][0]
            assert 'width="' in line, line

        # Dropping the drawn one leaves the engine's figure alone.
        client.delete(f"/api/tasks/{task_id}/pages/0/crop/{drawn}")
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["figures"] == [engine_figure]


def test_a_blank_left_outside_math_reaches_the_paper_inside_it():
    """DeepSeek regularly writes the fill-in blank into the prose rather than
    into math mode. The preview and the download have to carry the fix, while
    the text the page stores keeps what the model said."""
    with TestClient(app) as client:
        original = ocr.run_ocr

        def with_bare_blanks(image_path, work_dir, images_dir, prefix=""):
            return (
                "1. 下列说法正确的是（\\underline{\\hspace{2em}}）\n"
                "2. 计算：$a^9 = \\underline{\\hspace{2em}}$.\n"
                "班级：\\underline{\\hspace{2em}} 姓名：\\underline{\\hspace{2em}}\n",
                [],
                {},
            )

        ocr.run_ocr = with_bare_blanks
        try:
            task_id = client.post("/api/tasks", json={"title": "blanks"}).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("page.jpg", _banded_page(), "image/jpeg"))],
            )
            _wait(client, task_id)
        finally:
            ocr.run_ocr = original

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        # The stored text is exactly what came back from the model.
        assert page["markdown"].count("（\\underline{\\hspace{2em}}）") == 1
        # The paper shows all four blanks as math.
        paper = client.get(f"/tasks/{task_id}/download").text
        # Four blanks on the page: the three in the prose get wrapped, the one
        # that was already inside math is left alone.
        assert paper.count("\\underline{\\hspace{2em}}") == 4, paper
        assert paper.count("\\(\\underline{\\hspace{2em}}\\)") == 3, paper
        assert page["preview"].count("\\(\\underline{\\hspace{2em}}\\)") == 3
        # ...and the one already inside math is untouched.
        assert "$a^9 = \\underline{\\hspace{2em}}$" in paper


def _centre_colour(task_id, filename):
    from PIL import Image

    path = config.images_dir(task_id) / filename
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixel = rgb.getpixel((width // 2, height // 2))
    return pixel


def test_an_outlined_box_becomes_a_figure():
    with TestClient(app) as client:
        task_id = _crop_task(client)

        response = client.post(
            f"/api/tasks/{task_id}/pages/0/crop",
            json={"x": 0.25, "y": 0.4, "w": 0.5, "h": 0.2, "question": "13"},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        # Named after its box, like an OCR crop, and cut from the original photo.
        name = body["filename"]
        assert name.startswith("p1_manual_box_")
        assert body["photo_size"] == [600, 900]
        assert (config.images_dir(task_id) / name).exists()

        # The middle band is green: the crop really is where it was drawn.
        red, green, blue = _centre_colour(task_id, name)
        assert green > 200 and red < 90 and blue < 90, (red, green, blue)

        # It is a figure like any other: listed, numbered and placed.
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert name in page["figures"]
        assert page["crops"] == [name]
        assert page["mapping"] == {name: "13"}
        assert figure_tag(name, "13") in page["preview"]

        # The test host inlines the figure, so the tag keeps its alt text.
        markdown = client.get(f"/tasks/{task_id}/download").text
        assert 'alt="Q13 附图"' in markdown
        # Half the page across (300 px of 600, plus the 6 px padding either
        # side), shown at that share of FIGURE_PAGE_WIDTH.
        assert 'width="416"' in markdown
        assert "max-width:416px" in markdown


def test_a_crop_without_a_question_number_just_lands_on_the_page():
    with TestClient(app) as client:
        task_id = _crop_task(client)
        body = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.2}
        ).json()

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["mapping"] == {}
        assert body["filename"] in page["labelled_markdown"]


def test_a_crop_can_be_given_a_question_later_like_any_figure():
    with TestClient(app) as client:
        task_id = _crop_task(client)
        name = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.2}
        ).json()["filename"]

        client.put(f"/api/tasks/{task_id}/pages/0/mapping", json={name: "7"})
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert figure_tag(name, "7") in page["preview"]
        assert page["stale_mapping"] == []


def test_outlining_the_same_box_twice_does_not_duplicate_it():
    with TestClient(app) as client:
        task_id = _crop_task(client)
        body = {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}
        first = client.post(f"/api/tasks/{task_id}/pages/0/crop", json=body).json()
        second = client.post(f"/api/tasks/{task_id}/pages/0/crop", json=body).json()

        assert first["filename"] == second["filename"]
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["figures"].count(first["filename"]) == 1


def test_an_outlined_figure_survives_reading_the_page_again():
    """It lives beside the OCR text, so a fresh read cannot lose it."""
    run, calls = _recording_ocr()
    original = ocr.run_ocr
    ocr.run_ocr = run
    try:
        with TestClient(app) as client:
            task_id = _crop_task(client, "crop-then-reread")
            name = client.post(
                f"/api/tasks/{task_id}/pages/0/crop",
                json={"x": 0.1, "y": 0.4, "w": 0.4, "h": 0.2, "question": "13"},
            ).json()["filename"]

            client.post(f"/api/tasks/{task_id}/pages/0/reread")
            _wait(client, task_id)
    finally:
        ocr.run_ocr = original

    with TestClient(app) as client:
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert name in page["figures"]
        assert page["crops"] == [name]
        assert page["mapping"] == {name: "13"}
        assert (config.images_dir(task_id) / name).exists()
        assert figure_tag(name, "13") in page["preview"]


def test_an_outlined_figure_goes_on_a_hand_edited_page_too():
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-on-edit")
        client.put(
            f"/api/tasks/{task_id}/pages/0/markdown",
            json={"markdown": "13. My own text.\n"},
        )
        name = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.2}
        ).json()["filename"]

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert "13. My own text." in page["labelled_markdown"]
        assert f"images/{name}" in page["labelled_markdown"]
        # Reverting removes it along with the hand-written text it sat in.
        reverted = client.put(
            f"/api/tasks/{task_id}/pages/0/markdown", json={"reset": True}
        ).json()
        assert name in reverted["markdown"]


def page_preview(client, task_id: int) -> str:
    """The page's markdown as the paper shows it."""
    return client.get(f"/api/tasks/{task_id}/pages/0").json()["preview"]


def test_a_figure_is_shown_at_the_size_of_the_rectangle_it_was_cut_from():
    """The rectangle *is* the size: a diagram drawn across half the page is
    half the page wide in the paper, even though the cut-out file is the full
    resolution of the photo."""
    from PIL import Image

    from app import config, store

    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-sized")
        page_width = Image.open(config.page_image_path(task_id, 0)).size[0]

        def outlined(box: dict) -> str:
            return client.post(
                f"/api/tasks/{task_id}/pages/0/crop", json={**box, "question": "13"}
            ).json()["filename"]

        def shown_as(ref: str) -> str:
            line = [l for l in page_preview(client, task_id).splitlines() if ref in l]
            assert line, page_preview(client, task_id)
            return line[0]

        # Half the page across, a fifth of it down.
        half = outlined({"x": 0.25, "y": 0.4, "w": 0.5, "h": 0.2})
        with Image.open(config.images_dir(task_id) / half) as crop:
            natural = crop.size[0]
        assert natural >= 300, "the file is the crop, at full resolution"

        def box_of(ref: str) -> list[int]:
            page = store.get_task(task_id)["pages"][0]
            return next(c for c in store.page_crops(page) if c["name"] == ref)["box"]

        def expected_width(ref: str) -> int:
            x1, _, x2, _ = box_of(ref)
            return round((x2 - x1) / page_width * config.FIGURE_PAGE_WIDTH)

        assert f'width="{expected_width(half)}"' in shown_as(half), shown_as(half)
        assert expected_width(half) != natural, "shown smaller than the file it came from"

        # A quarter of the page across comes out a quarter of the width: the
        # size follows the rectangle, not one flat number for every figure.
        quarter = outlined({"x": 0.1, "y": 0.7, "w": 0.25, "h": 0.15})
        assert f'width="{expected_width(quarter)}"' in shown_as(quarter)
        assert expected_width(quarter) < expected_width(half)

        # And the height is bounded by the rectangle too.
        tall = outlined({"x": 0.6, "y": 0.05, "w": 0.3, "h": 0.5})
        # Half the page down is half the page tall on a page of that width, so
        # the figure cannot come out taller than it was drawn.
        box = box_of(tall)
        tallest = round((box[3] - box[1]) / page_width * config.FIGURE_PAGE_WIDTH)
        assert f"max-height:{min(tallest, config.FIGURE_MAX_HEIGHT)}px" in shown_as(tall)

        # The ceiling still clips a rectangle that covers the whole page.
        full = outlined({"x": 0.0, "y": 0.95, "w": 1.0, "h": 0.04})
        assert f'width="{config.FIGURE_MAX_WIDTH}"' in shown_as(full)

        # Written as a plain attribute too, and never into the stored text.
        paper = client.get(f"/tasks/{task_id}/download").text
        assert f'width="{expected_width(half)}"' in paper
        assert f'max-width:{config.FIGURE_MAX_WIDTH}px' in shown_as(full)
        assert "max-width" not in client.get(f"/api/tasks/{task_id}/pages/0").json()[
            "labelled_markdown"
        ]

        # FIGURE_PAGE_WIDTH = 0 is the off switch: no sizes at all, and the
        # figures come out at whatever the viewer does with the file.
        saved = config.FIGURE_PAGE_WIDTH
        try:
            config.FIGURE_PAGE_WIDTH = 0
            assert "max-width" not in page_preview(client, task_id)
        finally:
            config.FIGURE_PAGE_WIDTH = saved


def test_a_hand_edited_page_is_capped_without_being_rewritten():
    """The cap is a rendering concern, so it applies to edited text as well —
    but what you saved stays exactly as you saved it."""
    hand_typed = '13. Mine.\n\n<img src="images/own.jpg" style="max-width:120px">\n'
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-capped-edited")
        client.put(f"/api/tasks/{task_id}/pages/0/markdown", json={"markdown": hand_typed})
        name = client.post(
            f"/api/tasks/{task_id}/pages/0/crop",
            json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.2},
        ).json()["filename"]

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        # Stored, and shown in the editor, untouched (the API trims the edge).
        assert page["edited_markdown"] == hand_typed.strip()
        assert page["labelled_markdown"].startswith(hand_typed.rstrip("\n"))

        # The paper: your own size wins, the figure we added gets the cap.
        preview = page["preview"]
        assert "max-width:120px" in preview, "your own size, as written"
        added = [line for line in preview.splitlines() if f"images/{name}" in line][0]
        # A bit under half the page across (240 px of 600, plus the padding)
        # out of FIGURE_PAGE_WIDTH — the rectangle, not a flat number.
        assert 'width="336"' in added, added

        # Change the setting and the edited page follows it like any other.
        saved = config.FIGURE_PAGE_WIDTH
        try:
            config.FIGURE_PAGE_WIDTH = 1400
            page = client.get(f"/api/tasks/{task_id}/pages/0").json()
            assert "max-width:588px" in page["preview"], "a wider page means wider figures"
            assert page["edited_markdown"] == hand_typed.strip()
        finally:
            config.FIGURE_PAGE_WIDTH = saved


def test_a_one_line_row_keeps_following_the_setting():
    """The row's own height cap is stored with it; the width cap is not."""
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-row-capped")
        first = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.05, "w": 0.4, "h": 0.2}
        ).json()["filename"]
        second = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.45, "w": 0.4, "h": 0.2}
        ).json()["filename"]
        body = client.post(
            f"/api/tasks/{task_id}/pages/0/figure-row", json={"figures": [first, second]}
        ).json()

        # Stored with the row's height cap and nothing else...
        assert "max-height:200px" in body["markdown"]
        assert "max-width" not in body["markdown"]
        # ...so the paper adds the width, sized from the rectangle.
        rowed = [line for line in body["preview"].splitlines() if f"images/{first}" in line]
        assert rowed, body["preview"]
        assert "max-height:200px" in rowed[0]
        assert "1100px" not in rowed[0], "the row keeps its own height"
        assert re.search(r'width="\d+"', rowed[0]), rowed[0]


def test_an_outlined_figure_can_be_removed():
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-removed")
        name = client.post(
            f"/api/tasks/{task_id}/pages/0/crop",
            json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.2, "question": "13"},
        ).json()["filename"]
        path = config.images_dir(task_id) / name
        assert path.exists()

        response = client.delete(f"/api/tasks/{task_id}/pages/0/crop/{name}")
        assert response.status_code == 200, response.text

        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert name not in page["figures"]
        assert page["crops"] == []
        assert page["mapping"] == {}, "its number goes with it"
        assert name not in client.get(f"/tasks/{task_id}/download").text
        assert not path.exists(), "the cut-out file is cleaned up"

        # Removing it twice is a 404, not a silent second unlink.
        assert client.delete(f"/api/tasks/{task_id}/pages/0/crop/{name}").status_code == 404


def test_an_outlined_figure_joins_a_one_line_row():
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-row")
        first = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.05, "w": 0.4, "h": 0.2}
        ).json()["filename"]
        second = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.45, "w": 0.4, "h": 0.2}
        ).json()["filename"]

        body = client.post(
            f"/api/tasks/{task_id}/pages/0/figure-row", json={"figures": [first, second]}
        ).json()
        assert body["row_figures"] == [first, second]
        assert body["markdown"].count("display:flex") == 1
        # Outlining is untouched by the row: it is still removable by hand.
        assert client.get(f"/api/tasks/{task_id}/pages/0").json()["crops"] == [first, second]


def test_a_bad_box_is_refused_with_a_reason():
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-bad")
        path = f"/api/tasks/{task_id}/pages/0/crop"

        # Nothing drawn, a box outside the photo, and a speck.
        assert client.post(path, json={}).status_code == 400
        assert client.post(path, json={"x": 0.5, "y": 0.5, "w": 2, "h": 0.2}).status_code == 400
        tiny = client.post(path, json={"x": 0.5, "y": 0.5, "w": 0.001, "h": 0.001})
        assert tiny.status_code == 400
        assert "too small" in tiny.json()["detail"]

        # Nothing was stored by any of them.
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["crops"] == []
        assert "manual_box" not in page["labelled_markdown"]


def test_outlining_a_missing_photo_is_refused():
    from app import config as app_config

    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-nophoto")
        app_config.page_image_path(task_id, 0).unlink()
        response = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.2}
        )
        assert response.status_code == 400
        assert "photo is gone" in response.json()["detail"]


def test_a_box_can_be_described_by_its_corners_too():
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-corners")
        corners = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x1": 0.2, "y1": 0.4, "x2": 0.7, "y2": 0.6}
        ).json()
        sized = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.2, "y": 0.4, "w": 0.5, "h": 0.2}
        ).json()
        # Both descriptions cut the same rectangle, so they collide by name.
        assert corners["box"] == sized["box"] == corners["box"]
        assert len(client.get(f"/api/tasks/{task_id}/pages/0").json()["crops"]) == 1


def test_an_outlined_figure_put_in_a_row_is_still_only_referenced_once():
    """The row holds the figure, so the page must not append it again."""
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-rowed")
        first = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.05, "w": 0.4, "h": 0.2}
        ).json()["filename"]
        second = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.45, "w": 0.4, "h": 0.2}
        ).json()["filename"]

        body = client.post(
            f"/api/tasks/{task_id}/pages/0/figure-row", json={"figures": [first, second]}
        ).json()
        assert body["markdown"].count(first) == 1, body["markdown"]
        assert body["markdown"].count(second) == 1

        # Three figures on the page — the OCR one and the two outlined — and
        # the test host inlines them, so a duplicate would show up as a fourth.
        markdown = client.get(f"/tasks/{task_id}/download").text
        assert markdown.count("data:image/jpeg;base64,") == 3
        assert "display:flex" in markdown


def test_removing_an_outlined_figure_that_was_put_in_a_row():
    """The row referenced it, so that reference has to go with the file."""
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-rowed-removed")
        first = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.05, "w": 0.4, "h": 0.2}
        ).json()["filename"]
        second = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.05, "y": 0.45, "w": 0.4, "h": 0.2}
        ).json()["filename"]
        client.post(
            f"/api/tasks/{task_id}/pages/0/figure-row", json={"figures": [first, second]}
        )

        assert client.delete(f"/api/tasks/{task_id}/pages/0/crop/{first}").status_code == 200
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert first not in page["labelled_markdown"]
        assert second in page["labelled_markdown"], "the row keeps the other figure"
        assert first not in client.get(f"/tasks/{task_id}/download").text


def test_a_click_with_no_drag_is_refused():
    """Padding must not turn a stray click into a 12x12 'figure'."""
    with TestClient(app) as client:
        task_id = _crop_task(client, "crop-speck")
        response = client.post(
            f"/api/tasks/{task_id}/pages/0/crop", json={"x": 0.5, "y": 0.5, "w": 0, "h": 0}
        )
        assert response.status_code == 400
        assert "too small" in response.json()["detail"]
        assert client.get(f"/api/tasks/{task_id}/pages/0").json()["crops"] == []


def test_a_page_read_without_figure_detection_ends_up_with_no_figures():
    """The point of OCR_EXTRACT_FIGURES=0, through the real engine.

    This module replaces ``ocr.run_ocr`` wholesale, so the real engine is called
    directly here. Only the DeepSeek call is stubbed: the layout skip, the
    reference-stripping, the store and the API are all the real code, and if any
    of them were still reached this would load the layout model or leave a
    broken image behind.
    """
    from app import config

    saved_extract = config.OCR_EXTRACT_FIGURES
    saved_read = ocr.deepseek.read_page
    seen: dict = {}

    def read(image_path, model=None, figures=True):
        seen["figures"] = figures
        # A model with ideas of its own, whatever the prompt says.
        return (
            "13. Find the area.\n\n<img src=\"images/made_up.jpg\">\n\n14. Next.\n",
            [],
        )

    def run(image_path, work_dir, images_dir, prefix="", engine=None):
        return ocr.run_ocr_hybrid(
            image_path, work_dir, images_dir, prefix, extract=config.OCR_EXTRACT_FIGURES
        )

    original = ocr.run_ocr
    try:
        ocr.run_ocr = run
        ocr.deepseek.read_page = read
        config.OCR_EXTRACT_FIGURES = False
        with TestClient(app) as client:
            task_id = client.post(
                "/api/tasks", json={"title": "no-auto-figures"}
            ).json()["task_id"]
            client.post(
                f"/api/tasks/{task_id}/pages",
                files=[("files", ("page.jpg", _banded_page(), "image/jpeg"))],
            )
            _wait(client, task_id)
    finally:
        ocr.run_ocr = original
        ocr.deepseek.read_page = saved_read
        config.OCR_EXTRACT_FIGURES = saved_extract

    assert seen["figures"] is False, "DeepSeek was never asked about figures"

    with TestClient(app) as client:
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["figures"] == [], "no figure, detected or invented"
        assert page["crops"] == []
        assert "made_up" not in page["markdown"], "no broken image left behind"
        assert "13. Find the area." in page["markdown"]
        assert "<img" not in client.get(f"/tasks/{task_id}/download").text

        # Drawing one is then the only way a figure gets in.
        name = client.post(
            f"/api/tasks/{task_id}/pages/0/crop",
            json={"x": 0.25, "y": 0.4, "w": 0.5, "h": 0.2, "question": "13"},
        ).json()["filename"]
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["figures"] == [name]
        assert page["mapping"] == {name: "13"}
        assert figure_tag(name, "13") in page["preview"]


def test_missing_resources_return_404():
    with TestClient(app) as client:
        assert client.get("/api/tasks/t-nope").status_code == 404
        assert client.get("/tasks/t-nope").status_code == 404
        assert client.get("/tasks/t-nope/images/a.jpg").status_code == 404
        assert client.get("/tasks/t-nope/pages/0/raw").status_code == 404
        assert client.delete("/api/tasks/t-nope").status_code == 404
        assert client.post("/api/tasks/t-nope/pages/0/retry").status_code == 404
        assert client.post("/api/tasks/t-nope/pages/0/reread").status_code == 404
        assert client.put("/api/tasks/t-nope/pages/0/markdown", json={"markdown": "x"}).status_code == 404
        assert client.post("/api/tasks/t-nope/pages/0/figure-row", json={"figures": []}).status_code == 404


def test_edit_page_requires_a_finished_page():
    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "empty"}).json()["task_id"]
        # No pages at all.
        assert client.get(f"/tasks/{task_id}/edit?page=0").status_code == 400
        # A page index that does not exist.
        client.post(
            f"/api/tasks/{task_id}/pages",
            files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
        )
        assert client.get(f"/tasks/{task_id}/edit?page=5").status_code == 404


def test_local_and_base64_host_modes():
    from app import image_host

    markdown = '<img src="images/p1_a.jpg">'
    local = image_host.local_urls(markdown, "t-abc")
    assert local == '<img src="http://localhost:8000/tasks/t-abc/images/p1_a.jpg">'


# --- DeepSeek cleanup: background job + progress -----------------------------


def _make_ready_task(client, title="polish"):
    task_id = client.post("/api/tasks", json={"title": title}).json()["task_id"]
    client.post(
        f"/api/tasks/{task_id}/pages",
        files=[("files", ("p.jpg", _jpeg("p"), "image/jpeg"))],
    )
    _wait(client, task_id)
    return task_id


def test_polish_state_round_trip():
    from app import store

    # TestClient runs the lifespan, which is what opens the database.
    with TestClient(app):
        task_id = store.create_task("state")
        assert store.get_polish_state(task_id)["status"] == store.POLISH_IDLE

        store.set_polish_state(task_id, store.POLISH_RUNNING, stage="cleaning", done=2, total=5)
        state = store.get_polish_state(task_id)
        assert state == {"status": "running", "stage": "cleaning", "done": 2, "total": 5, "error": None}

        store.set_polish_state(task_id, store.POLISH_FAILED, error="boom")
        state = store.get_polish_state(task_id)
        assert state["status"] == "failed" and state["error"] == "boom"

        store.set_polished_markdown(task_id, "cleaned", {"model": "fake"})
        state = store.get_polish_state(task_id)
        assert state["status"] == "done" and state["error"] is None
        assert store.get_polished_markdown(task_id)[0] == "cleaned"

        store.clear_polished_markdown(task_id)
        assert store.get_polish_state(task_id)["status"] == store.POLISH_IDLE
        assert store.get_polished_markdown(task_id) == (None, None)


def test_migration_marks_already_cleaned_tasks_as_done():
    """Rows cleaned before polish_status existed must not report 'idle'."""
    from app import store

    with TestClient(app):
        task_id = store.create_task("legacy")
        # Simulate the pre-migration row: cleaned paper, default status.
        store.set_polished_markdown(task_id, "cleaned earlier", {"model": "old"})
        store._execute(
            "UPDATE tasks SET polish_status = ? WHERE id = ?", (store.POLISH_IDLE, task_id)
        )
        assert store.get_polish_state(task_id)["status"] == store.POLISH_IDLE

        store._migrate(store._conn)
        assert store.get_polish_state(task_id)["status"] == store.POLISH_DONE
        # The cleaned paper itself is untouched.
        assert store.get_polished_markdown(task_id)[0] == "cleaned earlier"


def test_stale_running_cleanup_is_reset_on_startup():
    from app import store

    with TestClient(app):
        task_id = store.create_task("stale")
        store.set_polish_state(task_id, store.POLISH_RUNNING, stage="cleaning", done=1, total=3)
        assert store.reset_stale_polish_states() >= 1
        assert store.get_polish_state(task_id)["status"] == store.POLISH_FAILED


def test_polish_endpoint_runs_in_background_and_reports_progress():
    from app import config as app_config
    from app import deepseek as ds

    original_polish = ds.polish_markdown
    saved_key = app_config.DEEPSEEK_API_KEY
    app_config.DEEPSEEK_API_KEY = "test-key"

    release = threading.Event()
    observed = {}

    def fake_polish(markdown, progress=None):
        if progress:
            progress("cleaning", 0, 2)
        observed["started"] = True
        release.wait(timeout=10)
        if progress:
            progress("cleaning", 2, 2)
        return "# Cleaned paper\n\n1. A question.\n", {
            "model": "fake",
            "chunks": 2,
            "figures_total": 0,
            "figures_recovered": 0,
            "usage": {},
        }

    ds.polish_markdown = fake_polish
    try:
        with TestClient(app) as client:
            task_id = _make_ready_task(client)

            # Starts immediately with 202 rather than blocking.
            response = client.post(f"/api/tasks/{task_id}/polish")
            assert response.status_code == 202, response.text

            # Progress becomes visible while the job is still running.
            deadline = time.time() + 10
            state = {}
            while time.time() < deadline:
                state = client.get(f"/api/tasks/{task_id}/polish").json()
                if state["status"] == "running" and state["polish"]["total"] == 2:
                    break
                time.sleep(0.05)
            assert state["status"] == "running", state
            assert state["polish"]["stage"] == "cleaning"
            assert state["polish"]["total"] == 2

            # A second concurrent run is refused.
            assert client.post(f"/api/tasks/{task_id}/polish").status_code == 409

            release.set()

            deadline = time.time() + 15
            while time.time() < deadline:
                state = client.get(f"/api/tasks/{task_id}/polish").json()
                if state["status"] in ("done", "failed"):
                    break
                time.sleep(0.05)
            assert state["status"] == "done", state
            assert "Cleaned paper" in state["markdown"]
            assert state["info"]["model"] == "fake"

            # The task status endpoint carries the same state.
            assert client.get(f"/api/tasks/{task_id}").json()["polish"]["status"] == "done"

            # And the cleaned paper is downloadable.
            downloaded = client.get(f"/tasks/{task_id}/download?variant=polished")
            assert downloaded.status_code == 200
            assert "Cleaned paper" in downloaded.text
            assert "_cleaned.md" in downloaded.headers["content-disposition"]
    finally:
        ds.polish_markdown = original_polish
        app_config.DEEPSEEK_API_KEY = saved_key


def test_polish_requires_a_key():
    from app import config as app_config

    saved = app_config.DEEPSEEK_API_KEY
    app_config.DEEPSEEK_API_KEY = ""
    try:
        with TestClient(app) as client:
            task_id = _make_ready_task(client, "no-key")
            assert client.post(f"/api/tasks/{task_id}/polish").status_code == 400
    finally:
        app_config.DEEPSEEK_API_KEY = saved
