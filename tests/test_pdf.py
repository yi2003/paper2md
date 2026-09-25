"""PDF input: a PDF is rendered into page images and read like photos.

Runs the real FastAPI app, real SQLite store, real worker and the real PDF
renderer (pypdfium2 on a PDF built on the fly with Pillow) — only the OCR call
is stubbed, exactly like ``test_flow``. That means the test proves the whole
PDF -> pages -> markdown -> download path works, not just the renderer.
"""

import io
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# Isolate the data directory *before* `app.config` is imported.
import _env  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app import config, ocr, pdf  # noqa: E402
from app.main import app  # noqa: E402

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


def _pdf_bytes(pages: list[str]) -> bytes:
    """Build a small PDF. An empty string produces a genuinely blank page."""
    from PIL import Image, ImageDraw

    images = []
    for text in pages:
        image = Image.new("RGB", (700, 900), "white")
        if text:
            draw = ImageDraw.Draw(image)
            # The box guarantees plenty of ink, so blank detection (which is
            # deliberately conservative) never mistakes a content page for one.
            draw.rectangle((40, 40, 660, 300), outline="black", width=4)
            draw.text((60, 330), text, fill="black")
        images.append(image)

    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def _fake_run_ocr(image_path, work_dir, images_dir, prefix=""):
    """Stand-in for the OCR engine: one figure, page markdown, no suggestions."""
    images_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}fig1.jpg"
    (images_dir / name).write_bytes(_jpeg("figure"))
    markdown = PAGE_MARKDOWN.replace("images/fig1.jpg", f"images/{name}")
    return markdown, [name], {}


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


def _task_count(client) -> int:
    return len(client.get("/api/tasks").json()["tasks"])


def _tmpdir() -> Path:
    """A throwaway directory. Plain tempfile, so `tests/run.py` works too."""
    return Path(tempfile.mkdtemp(prefix="paper2md-pdf-"))


# --- the renderer itself -----------------------------------------------------


def test_render_pages_writes_images_and_skips_blanks():
    tmp_path = _tmpdir()
    data = _pdf_bytes(["Q1", "", "Q2"])
    result = pdf.render_pages(data, tmp_path)

    assert result.total_pages == 3
    assert result.rendered == 2
    assert result.blank == [2]
    # Blank pages do not consume a page index.
    assert [page.page_index for page in result.pages] == [0, 1]
    # Provenance: the PDF page number is kept, independent of the task index.
    assert [page.page_number for page in result.pages] == [1, 3]
    assert (tmp_path / "page_1.jpg").is_file()
    assert (tmp_path / "page_2.jpg").is_file()
    assert not (tmp_path / "page_3.jpg").exists()
    for page in result.pages:
        assert page.width > 0 and page.height > 0


def test_render_pages_can_be_told_not_to_skip_blanks():
    tmp_path = _tmpdir()
    result = pdf.render_pages(_pdf_bytes(["Q1", ""]), tmp_path, skip_blank=False)
    assert result.rendered == 2
    assert result.blank == []


def test_render_pages_reports_progress_even_for_blank_pages():
    """The bar must reach the total even when the last pages are blank."""
    seen = []
    result = pdf.render_pages(
        _pdf_bytes(["a", "", "c"]), _tmpdir(), on_progress=lambda done, total: seen.append((done, total))
    )
    assert result.rendered == 2
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_render_pages_honours_range_and_cap():
    tmp_path = _tmpdir()
    data = _pdf_bytes(["a", "b", "c", "d"])
    result = pdf.render_pages(
        data, tmp_path, first_page=2, last_page=4, max_pages=2, skip_blank=False
    )
    assert result.first_page == 2 and result.last_page == 3
    assert result.truncated is True
    assert [page.page_number for page in result.pages] == [2, 3]


def test_render_pages_continues_an_existing_page_index():
    tmp_path = _tmpdir()
    result = pdf.render_pages(_pdf_bytes(["a", "b"]), tmp_path, first_index=5)
    assert [page.page_index for page in result.pages] == [5, 6]
    assert (tmp_path / "page_6.jpg").is_file()
    assert (tmp_path / "page_7.jpg").is_file()


def test_render_pages_rejects_non_pdf_and_empty_input():
    tmp_path = _tmpdir()
    for bad, fragment in ((b"hello world", "not a PDF"), (b"", "empty")):
        try:
            pdf.render_pages(bad, tmp_path)
        except pdf.PdfError as exc:
            assert fragment in str(exc), exc
        else:
            raise AssertionError("expected PdfError")


# --- the HTTP surface --------------------------------------------------------


def test_add_pdf_to_a_task_renders_queues_and_downloads():
    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "pdf task"}).json()["task_id"]

        response = client.post(
            f"/api/tasks/{task_id}/pdf",
            files={"file": ("paper.pdf", _pdf_bytes(["Q1", "", "Q2"]), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["pdf_pages"] == 3
        assert body["queued_pages"] == 2
        assert body["blank_pages"] == [2]
        assert [page["page_index"] for page in body["accepted"]] == [0, 1]
        assert [page["pdf_page"] for page in body["accepted"]] == [1, 3]

        status = _wait(client, task_id)
        assert status["status"] == "done", status
        # The stored filename ties each page back to its PDF and page number.
        assert "paper.pdf" in status["pages"][0]["filename"]
        assert "p1" in status["pages"][0]["filename"]

        # The rendered page is served like any uploaded photo.
        assert client.get(f"/tasks/{task_id}/pages/0/raw").status_code == 200

        # And the figures came out of the normal pipeline.
        page = client.get(f"/api/tasks/{task_id}/pages/0").json()
        assert page["figures"] == ["p1_fig1.jpg"]

        markdown = client.get(f"/tasks/{task_id}/download").text
        assert "<!-- page 1 -->" in markdown
        assert "<!-- page 2 -->" in markdown
        assert "data:image/jpeg;base64," in markdown


def test_pdf_pages_continue_after_uploaded_photos():
    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "mixed"}).json()["task_id"]
        client.post(
            f"/api/tasks/{task_id}/pages",
            files=[("files", ("photo.jpg", _jpeg("photo"), "image/jpeg"))],
        )
        response = client.post(
            f"/api/tasks/{task_id}/pdf",
            files={"file": ("second.pdf", _pdf_bytes(["Q2", "Q3"]), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        # The PDF continues the task's page order rather than restarting at 0.
        assert [page["page_index"] for page in response.json()["accepted"]] == [1, 2]

        status = _wait(client, task_id)
        assert status["status"] == "done"
        filenames = [page["filename"] for page in status["pages"]]
        assert filenames[0] == "photo.jpg"
        assert filenames[1] == "second.pdf · p1"
        assert filenames[2] == "second.pdf · p2"
        download = client.get(f"/tasks/{task_id}/download").text
        assert download.index("<!-- page 1 -->") < download.index("<!-- page 2 -->")


def test_convert_pdf_one_shot_creates_and_fills_a_task():
    with TestClient(app) as client:
        response = client.post(
            "/api/convert/pdf",
            files={"file": ("midterm.pdf", _pdf_bytes(["Q1", "Q2"]), "application/pdf")},
            data={"title": "One-shot midterm"},
        )
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        assert response.json()["queued_pages"] == 2

        task = client.get(f"/api/tasks/{task_id}").json()
        assert task["title"] == "One-shot midterm"
        assert len(task["pages"]) == 2

        status = _wait(client, task_id)
        assert status["status"] == "done", status
        markdown = client.get(f"/api/tasks/{task_id}/markdown").json()["markdown"]
        assert "page 1" in markdown and "page 2" in markdown


def test_convert_pdf_defaults_the_title_to_the_filename():
    with TestClient(app) as client:
        response = client.post(
            "/api/convert/pdf",
            files={"file": ("algebra-2024.pdf", _pdf_bytes(["Q1"]), "application/pdf")},
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        assert client.get(f"/api/tasks/{task_id}").json()["title"] == "algebra-2024"


def test_pdf_upload_rejects_the_wrong_type_and_broken_files():
    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "bad pdf"}).json()["task_id"]

        wrong = client.post(
            f"/api/tasks/{task_id}/pdf",
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )
        assert wrong.status_code == 400
        assert "not a PDF" in wrong.json()["detail"]

        broken = client.post(
            f"/api/tasks/{task_id}/pdf",
            files={"file": ("broken.pdf", b"this is not really a pdf", "application/pdf")},
        )
        assert broken.status_code == 400
        assert "%PDF-" in broken.json()["detail"]

        # Nothing was queued, so the task still has no pages.
        assert client.get(f"/api/tasks/{task_id}").json()["pages"] == []


def test_a_blank_only_pdf_is_rejected_with_a_useful_message():
    with TestClient(app) as client:
        task_id = client.post("/api/tasks", json={"title": "blank"}).json()["task_id"]
        response = client.post(
            f"/api/tasks/{task_id}/pdf",
            files={"file": ("blank.pdf", _pdf_bytes(["", ""]), "application/pdf")},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "blank" in detail and "PDF_SKIP_BLANK" in detail


def test_failed_one_shot_conversion_leaves_no_empty_task_behind():
    with TestClient(app) as client:
        before = _task_count(client)
        response = client.post(
            "/api/convert/pdf",
            files={"file": ("broken.pdf", b"still not a pdf", "application/pdf")},
        )
        assert response.status_code == 400
        assert _task_count(client) == before


def test_pdf_settings_drive_the_renderer():
    """PDF_FIRST_PAGE / PDF_LAST_PAGE are honoured end to end."""
    saved = (config.PDF_FIRST_PAGE, config.PDF_LAST_PAGE)
    config.PDF_FIRST_PAGE, config.PDF_LAST_PAGE = 2, 3
    try:
        with TestClient(app) as client:
            task_id = client.post("/api/tasks", json={"title": "range"}).json()["task_id"]
            response = client.post(
                f"/api/tasks/{task_id}/pdf",
                files={"file": ("range.pdf", _pdf_bytes(["a", "b", "c", "d"]), "application/pdf")},
            )
            assert response.status_code == 201, response.text
            assert [page["pdf_page"] for page in response.json()["accepted"]] == [2, 3]
            _wait(client, task_id)
    finally:
        config.PDF_FIRST_PAGE, config.PDF_LAST_PAGE = saved


def test_health_reports_pdf_support():
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["pdf_ready"] is True
        assert health["pdf_dpi"] == config.PDF_DPI


# --- automatic cleanup for PDF tasks -----------------------------------------
#
# A PDF is expected to come out as a *clean* paper, so its DeepSeek pass starts
# by itself once every page has been read. The DeepSeek call is stubbed here;
# the real PolishRunner, store and worker are exercised.


def _fake_polish(markdown, progress=None):
    if progress:
        progress("cleaning", 0, 1)
        progress("cleaning", 1, 1)
    return "# Cleaned paper\n\n1. A question.\n", {
        "model": "fake",
        "chunks": 1,
        "figures_total": 0,
        "figures_recovered": 0,
        "usage": {},
    }


def _wait_for_polish(client, task_id, timeout=15.0):
    deadline = time.time() + timeout
    state = {}
    while time.time() < deadline:
        state = client.get(f"/api/tasks/{task_id}/polish").json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError(f"cleanup never settled: {state}")


def test_pdf_task_is_auto_cleaned_without_pressing_the_button():
    from app import config as app_config
    from app import deepseek as ds

    original = ds.polish_markdown
    saved = (app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH)
    ds.polish_markdown = _fake_polish
    try:
        # The key and the flag are set *inside* the lifespan, so the startup
        # sweep cannot pick up tasks left behind by other tests.
        with TestClient(app) as client:
            app_config.DEEPSEEK_API_KEY = "test-key"
            app_config.PDF_AUTO_POLISH = True

            response = client.post(
                "/api/convert/pdf",
                files={"file": ("auto.pdf", _pdf_bytes(["Q1", "Q2"]), "application/pdf")},
                data={"title": "auto clean"},
            )
            assert response.status_code == 202, response.text
            task_id = response.json()["task_id"]

            status = _wait(client, task_id)
            assert status["status"] == "done", status

            # Nobody pressed ✨ Clean up, and nobody called POST /polish.
            state = _wait_for_polish(client, task_id)
            assert state["status"] == "done", state
            assert "Cleaned paper" in state["markdown"]
            assert state["info"]["model"] == "fake"

            downloaded = client.get(f"/tasks/{task_id}/download?variant=polished")
            assert downloaded.status_code == 200
            assert "Cleaned paper" in downloaded.text
            assert "_cleaned.md" in downloaded.headers["content-disposition"]
    finally:
        ds.polish_markdown = original
        app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH = saved


def test_pdf_auto_clean_reports_itself_to_the_status_page():
    from app import config as app_config
    from app import deepseek as ds

    original = ds.polish_markdown
    saved = (app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH)
    ds.polish_markdown = _fake_polish
    try:
        with TestClient(app) as client:
            app_config.DEEPSEEK_API_KEY = "test-key"
            app_config.PDF_AUTO_POLISH = True

            task_id = client.post("/api/tasks", json={"title": "flag"}).json()["task_id"]
            # A photo task is not auto-cleaned...
            assert client.get(f"/api/tasks/{task_id}").json()["auto_polish_ready"] is False
            # ...but ingesting a PDF turns it on.
            client.post(
                f"/api/tasks/{task_id}/pdf",
                files={"file": ("flag.pdf", _pdf_bytes(["Q1"]), "application/pdf")},
            )
            assert client.get(f"/api/tasks/{task_id}").json()["auto_polish_ready"] is True
            _wait(client, task_id)
            assert _wait_for_polish(client, task_id)["status"] == "done"
    finally:
        ds.polish_markdown = original
        app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH = saved


def test_pdf_auto_clean_can_be_switched_off():
    from app import config as app_config

    saved = (app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH)
    try:
        with TestClient(app) as client:
            # A key is present, so the only reason not to clean is the flag.
            app_config.DEEPSEEK_API_KEY = "test-key"
            app_config.PDF_AUTO_POLISH = False

            response = client.post(
                "/api/convert/pdf",
                files={"file": ("off.pdf", _pdf_bytes(["Q1"]), "application/pdf")},
            )
            assert response.status_code == 202, response.text
            task_id = response.json()["task_id"]
            _wait(client, task_id)
            time.sleep(0.4)

            task = client.get(f"/api/tasks/{task_id}").json()
            assert task["auto_polish_ready"] is False
            assert task["polish"]["status"] == "idle"
    finally:
        app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH = saved


def test_startup_sweep_resumes_an_uncleaned_auto_polish_task():
    """A cleanup cut short by a restart must be picked up again, not lost."""
    from app import config as app_config
    from app import deepseek as ds
    from app import store, worker

    original = ds.polish_markdown
    saved = (app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH)
    ds.polish_markdown = _fake_polish
    try:
        with TestClient(app) as client:
            app_config.DEEPSEEK_API_KEY = "test-key"
            app_config.PDF_AUTO_POLISH = True

            # A task that finished reading but never got cleaned.
            task_id = store.create_task("resume")
            store.add_page(task_id, 0, "p.jpg")
            store.mark_page_done(task_id, 0, "1. A question.\n", 0)
            store.set_auto_polish(task_id, True)
            store.rebuild_task(task_id)
            assert store.get_polished_markdown(task_id) == (None, None)

            assert worker.auto_polish_pending() >= 1

            deadline = time.time() + 15
            while time.time() < deadline:
                if store.get_polish_state(task_id)["status"] == store.POLISH_DONE:
                    break
                time.sleep(0.05)
            assert store.get_polish_state(task_id)["status"] == store.POLISH_DONE
            assert store.get_polished_markdown(task_id)[0] == "# Cleaned paper\n\n1. A question.\n"
            # Already clean, so the sweep no longer considers it.
            assert task_id not in store.tasks_awaiting_auto_polish()
    finally:
        ds.polish_markdown = original
        app_config.DEEPSEEK_API_KEY, app_config.PDF_AUTO_POLISH = saved
