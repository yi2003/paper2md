"""End-to-end test of the web flow with OCR stubbed out.

Runs the real FastAPI app, real SQLite store, real worker thread and real
markdown pipeline — only the PaddleOCR-VL call is replaced, so this test works
on a machine where Paddle is not installed yet.
"""

import io
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
    return markdown, [name]


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
        assert "*[Q2 附图]*" in labelled
        assert labelled.index("*[Q2 附图]*") > labelled.index("2. Solve for $x$.")

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
        assert "*[Q2 附图]*" in page["labelled_markdown"]

        # Blank values are dropped rather than stored.
        client.put(f"/api/tasks/{task_id}/pages/0/mapping", json={"p1_fig1.jpg": "  "})
        assert client.get(f"/api/tasks/{task_id}/pages/0").json()["mapping"] == {}


def test_missing_resources_return_404():
    with TestClient(app) as client:
        assert client.get("/api/tasks/t-nope").status_code == 404
        assert client.get("/tasks/t-nope").status_code == 404
        assert client.get("/tasks/t-nope/images/a.jpg").status_code == 404
        assert client.get("/tasks/t-nope/pages/0/raw").status_code == 404
        assert client.delete("/api/tasks/t-nope").status_code == 404
        assert client.post("/api/tasks/t-nope/pages/0/retry").status_code == 404


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
