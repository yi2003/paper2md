"""End-to-end test of the web flow with OCR stubbed out.

Runs the real FastAPI app, real SQLite store, real worker thread and real
markdown pipeline — only the PaddleOCR-VL call is replaced, so this test works
on a machine where Paddle is not installed yet.
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Configure an isolated data directory *before* importing the app.
_TMP = tempfile.mkdtemp(prefix="paper2md-test-")
os.environ["DATA_DIR"] = _TMP
os.environ["IMAGE_HOST"] = "base64"
os.environ["IMGBB_API_KEY"] = ""
os.environ["OCR_WORKERS"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app import ocr  # noqa: E402
from app.main import app  # noqa: E402

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
