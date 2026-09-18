"""Paper2MD — upload exam paper photos, get markdown.

Run with:  python -m app    (or ./run.sh)
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, deepseek, image_host, store
from .markdown_utils import (
    detect_questions,
    extract_filenames,
    page_markdown,
    strip_labels,
)
from .worker import polish_runner, worker

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _format_timestamp(unix_seconds: float | None) -> str:
    if not unix_seconds:
        return ""
    import datetime

    return datetime.datetime.fromtimestamp(unix_seconds).strftime("%Y-%m-%d %H:%M")


TEMPLATES.env.filters["ts"] = _format_timestamp


def log(message: str) -> None:
    print(f"[app] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    store.init()
    store.reset_stale_polish_states()
    worker.start()
    worker.requeue_interrupted()
    log(f"ready on http://localhost:{config.PORT} (image host: {config.IMAGE_HOST})")
    try:
        yield
    finally:
        worker.stop()
        store.close()


app = FastAPI(title="Paper2MD", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response = await call_next(request)
    if "text/html" in response.headers.get("content-type", ""):
        response.headers["Cache-Control"] = "no-store"
    return response


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _task_or_404(task_id: str) -> dict:
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(404, f"No such task: {task_id}")
    return task


def _page_or_404(task_id: str, page_index: int) -> dict:
    page = store.get_page(task_id, page_index)
    if page is None:
        raise HTTPException(404, f"No such page: {page_index}")
    return page


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE).strip() or "paper"
    return cleaned[:80]


def _attachment(filename: str) -> dict[str, str]:
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "paper2md.md"
    return {
        "Content-Disposition": (
            f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(filename)}'
        )
    }


def _to_jpeg(raw: bytes) -> bytes:
    """Re-encode any accepted image as a sane RGB JPEG."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def _page_status_payload(page: dict) -> dict:
    return {
        "page_index": page["page_index"],
        "filename": page["filename"],
        "status": page["status"],
        "error": page["error"],
        "image_count": page["image_count"],
    }


def _mapping_of(page: dict) -> dict[str, str]:
    try:
        parsed = json.loads(page.get("label_mapping") or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------
# HTML pages
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "tasks": store.list_tasks(),
            "image_host": config.IMAGE_HOST,
            "imgbb_ready": bool(config.IMGBB_API_KEY),
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "ocr_workers": config.OCR_WORKERS,
        },
    )


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
async def page_task(request: Request, task_id: str):
    task = _task_or_404(task_id)
    store.rebuild_task(task_id)
    task = _task_or_404(task_id)
    _, polish_info = store.get_polished_markdown(task_id)
    return TEMPLATES.TemplateResponse(
        request,
        "task.html",
        {
            "task": task,
            "deepseek_ready": config.deepseek_ready(),
            "deepseek_model": config.DEEPSEEK_MODEL,
            "polish_info": polish_info,
            "all_pages_done": bool(task["pages"]) and all(p["status"] == store.DONE for p in task["pages"]),
        },
    )


@app.get("/tasks/{task_id}/edit", response_class=HTMLResponse)
async def page_edit(request: Request, task_id: str, page: int = 0):
    task = _task_or_404(task_id)
    if not task["pages"]:
        raise HTTPException(400, "This task has no pages yet")
    if page < 0 or page >= len(task["pages"]):
        raise HTTPException(404, f"Task has {len(task['pages'])} page(s)")

    current = task["pages"][page]
    if current["status"] != store.DONE:
        raise HTTPException(400, f"Page {page + 1} is not ready yet ({current['status']})")

    _, polish_info = store.get_polished_markdown(task_id)
    return TEMPLATES.TemplateResponse(
        request,
        "edit.html",
        {
            "task": task,
            "page_index": page,
            "page_count": len(task["pages"]),
            "image_host": config.IMAGE_HOST,
            "deepseek_ready": config.deepseek_ready(),
            "deepseek_model": config.DEEPSEEK_MODEL,
            "polish_info": polish_info,
            "all_pages_done": all(p["status"] == store.DONE for p in task["pages"]),
        },
    )


# --------------------------------------------------------------------------
# Task API
# --------------------------------------------------------------------------


@app.post("/api/tasks")
async def api_create_task(payload: dict | None = Body(default=None)):
    title = ""
    if isinstance(payload, dict):
        title = str(payload.get("title") or "").strip()
    return JSONResponse({"task_id": store.create_task(title)}, status_code=201)


@app.get("/api/tasks")
async def api_list_tasks():
    return {"tasks": store.list_tasks()}


@app.get("/api/tasks/{task_id}")
async def api_task(task_id: str):
    store.rebuild_task(task_id)
    task = _task_or_404(task_id)
    return {
        "task_id": task["id"],
        "title": task["title"],
        "status": task["status"],
        "pages": [_page_status_payload(page) for page in task["pages"]],
        "queue": worker.pending,
        "polish": store.get_polish_state(task_id),
    }


@app.delete("/api/tasks/{task_id}")
async def api_delete_task(task_id: str):
    _task_or_404(task_id)
    store.delete_task(task_id)
    import shutil

    shutil.rmtree(config.task_dir(task_id), ignore_errors=True)
    return {"deleted": task_id}


@app.post("/api/tasks/{task_id}/pages")
async def api_upload_pages(task_id: str, files: list[UploadFile] = File(...)):
    """Upload one or more page photos. Upload order defines page order."""
    _task_or_404(task_id)

    if not files:
        raise HTTPException(400, "No files were uploaded")
    if len(files) > config.MAX_PAGES_PER_UPLOAD:
        raise HTTPException(
            400, f"At most {config.MAX_PAGES_PER_UPLOAD} pages per upload"
        )

    limit_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    index = store.next_page_index(task_id)
    accepted: list[dict] = []
    skipped: list[dict] = []

    for upload in files:
        name = upload.filename or "page.jpg"
        suffix = Path(name).suffix.lower()

        if suffix not in config.ALLOWED_SUFFIXES:
            skipped.append({"filename": name, "reason": f"unsupported type {suffix or '(none)'}"})
            continue

        raw = await upload.read()
        if not raw:
            skipped.append({"filename": name, "reason": "empty file"})
            continue
        if len(raw) > limit_bytes:
            skipped.append(
                {"filename": name, "reason": f"larger than {config.MAX_UPLOAD_MB} MB"}
            )
            continue

        try:
            jpeg = await asyncio.to_thread(_to_jpeg, raw)
        except Exception as exc:  # noqa: BLE001 - report and carry on
            skipped.append({"filename": name, "reason": f"not a readable image ({exc})"})
            continue

        destination = config.page_image_path(task_id, index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(jpeg)

        store.add_page(task_id, index, name)
        worker.submit(task_id, index, destination)
        accepted.append({"page_index": index, "filename": name, "status": store.QUEUED})
        index += 1

    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)  # new pages invalidate the cleaned paper
    return JSONResponse(
        {
            "task_id": task_id,
            "accepted": accepted,
            "skipped": skipped,
            "queued_pages": len(accepted),
        },
        status_code=201 if accepted else 400,
    )


@app.post("/api/tasks/{task_id}/pages/{page_index}/retry")
async def api_retry_page(task_id: str, page_index: int):
    page = _page_or_404(task_id, page_index)
    if page["status"] != store.FAILED:
        raise HTTPException(409, f"Page {page_index + 1} is {page['status']}, not failed")

    image_path = config.page_image_path(task_id, page_index)
    if not image_path.exists():
        raise HTTPException(400, "The original page image is gone — upload it again")

    store.reset_page_to_queued(task_id, page_index)
    worker.submit(task_id, page_index, image_path)
    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)
    return {"page_index": page_index, "status": store.QUEUED}


# --------------------------------------------------------------------------
# Page markdown & figure → question mapping
# --------------------------------------------------------------------------


@app.get("/api/tasks/{task_id}/pages/{page_index}")
async def api_page(task_id: str, page_index: int):
    page = _page_or_404(task_id, page_index)
    markdown = page["markdown"] or ""
    mapping = _mapping_of(page)
    return {
        "page_index": page_index,
        "status": page["status"],
        "error": page["error"],
        "markdown": markdown,
        "labelled_markdown": page_markdown(markdown, page["label_mapping"]),
        "mapping": mapping,
        "figures": extract_filenames(markdown),
        "questions": detect_questions(strip_labels(markdown)),
        "image_count": page["image_count"],
    }


@app.put("/api/tasks/{task_id}/pages/{page_index}/mapping")
async def api_save_mapping(
    task_id: str, page_index: int, payload: dict | None = Body(default=None)
):
    """Persist the ``{filename: "Q13"}`` mapping and return the relabelled page."""
    page = _page_or_404(task_id, page_index)
    _task_or_404(task_id)

    raw = payload if isinstance(payload, dict) else {}
    mapping = {str(key): str(value) for key, value in raw.items() if str(value).strip()}

    store.set_page_mapping(task_id, page_index, mapping)
    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)  # the cleaned paper is now stale

    return {
        "page_index": page_index,
        "mapping": mapping,
        "markdown": page_markdown(page["markdown"] or "", json.dumps(mapping)),
    }


# --------------------------------------------------------------------------
# Figure classification (handwriting vs printed diagram)
# --------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/pages/{page_index}/classify-figures")
async def api_classify_figures(task_id: str, page_index: int):
    """Ask DeepSeek which of this page's figures are handwriting, not diagrams.

    PaddleOCR-VL sometimes cuts a region of handwriting out as a "figure", so it
    ends up embedded in the finished paper. This flags those; the caller decides
    what to do with the answer.
    """
    page = _page_or_404(task_id, page_index)
    if page["status"] != store.DONE:
        raise HTTPException(400, f"Page {page_index + 1} is not ready yet")
    if not config.deepseek_ready():
        raise HTTPException(
            400, "DEEPSEEK_API_KEY is not set — add it to .env and restart the app"
        )

    names = extract_filenames(page["markdown"] or "")
    if not names:
        return {"task_id": task_id, "page_index": page_index, "kinds": {}, "handwritten": []}

    images_dir = config.images_dir(task_id)
    paths = [images_dir / name for name in names if (images_dir / name).is_file()]
    if not paths:
        raise HTTPException(404, "No figure files found on disk for this page")

    log(f"[{task_id}] classifying {len(paths)} figure(s) on page {page_index + 1}")
    kinds = await asyncio.to_thread(deepseek.classify_figures, paths)
    handwritten = [name for name, kind in kinds.items() if kind == deepseek.HANDWRITTEN]
    log(f"[{task_id}] page {page_index + 1}: {len(handwritten)} look handwritten")

    return {
        "task_id": task_id,
        "page_index": page_index,
        "kinds": kinds,
        "handwritten": handwritten,
    }


# --------------------------------------------------------------------------
# Image hosting & download
# --------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/host-images")
async def api_host_images(task_id: str):
    """Upload the figures to the configured host and return hosted markdown."""
    task = store.rebuild_task(task_id)
    if task is None:
        raise HTTPException(404, "No such task")

    markdown, urls = await asyncio.to_thread(
        image_host.host_markdown, task["merged_markdown"] or "", task_id
    )
    return {"markdown": markdown, "urls": urls, "mode": config.IMAGE_HOST}


@app.get("/api/tasks/{task_id}/markdown")
async def api_task_markdown(task_id: str, host: bool = False, variant: str = "original"):
    task = store.rebuild_task(task_id)
    if task is None:
        raise HTTPException(404, "No such task")

    if variant == "polished":
        polished, info = store.get_polished_markdown(task_id)
        if polished is None:
            raise HTTPException(404, "This task has not been cleaned up with DeepSeek yet")
        return {"markdown": polished, "info": info, "variant": "polished"}

    markdown = task["merged_markdown"] or ""
    if host:
        markdown, _ = await asyncio.to_thread(image_host.host_markdown, markdown, task_id)
    return {"markdown": markdown, "variant": "original"}


@app.post("/api/tasks/{task_id}/polish")
async def api_polish(task_id: str):
    """Start the DeepSeek cleanup in the background.

    Returns immediately with 202. Poll ``GET /api/tasks/{id}/polish`` (or the
    task status endpoint) for progress — the job reports each figure it uploads
    and each chunk it cleans.
    """
    task = store.rebuild_task(task_id)
    if task is None:
        raise HTTPException(404, "No such task")
    if not config.deepseek_ready():
        raise HTTPException(
            400, "DEEPSEEK_API_KEY is not set — add it to .env and restart the app"
        )
    if not (task["merged_markdown"] or "").strip():
        raise HTTPException(400, "There is nothing to clean up yet")
    if not polish_runner.start(task_id):
        raise HTTPException(409, "A cleanup is already running for this task")

    return JSONResponse(
        {"task_id": task_id, "status": store.POLISH_RUNNING, "polish": store.get_polish_state(task_id)},
        status_code=202,
    )


@app.get("/api/tasks/{task_id}/polish")
async def api_polish_status(task_id: str):
    """Progress of the DeepSeek cleanup — poll this while it runs."""
    _task_or_404(task_id)
    state = store.get_polish_state(task_id)
    if state["status"] == store.POLISH_RUNNING and not polish_runner.is_running(task_id):
        # The worker thread died without writing a state; do not hang the UI.
        store.set_polish_state(task_id, store.POLISH_FAILED, error="cleanup stopped unexpectedly")
        state = store.get_polish_state(task_id)
    if state["status"] == store.POLISH_DONE:
        markdown, info = store.get_polished_markdown(task_id)
        return {"status": state["status"], "polish": state, "info": info, "markdown": markdown}
    return {"status": state["status"], "polish": state}


@app.get("/tasks/{task_id}/download")
async def download_task(task_id: str, inline: bool = False, variant: str = "original"):
    """Download the merged markdown, with figures hosted per IMAGE_HOST.

    ``variant=polished`` returns the DeepSeek-cleaned paper instead.
    """
    task = store.rebuild_task(task_id)
    if task is None:
        raise HTTPException(404, "No such task")

    if variant == "polished":
        markdown, _ = store.get_polished_markdown(task_id)
        if markdown is None:
            raise HTTPException(
                400, "This task has not been cleaned up yet — run 'Clean up with DeepSeek' first"
            )
        suffix = "_cleaned"
    else:
        markdown, _ = await asyncio.to_thread(
            image_host.host_markdown, task["merged_markdown"] or "", task_id
        )
        suffix = ""

    if inline:
        return PlainTextResponse(markdown, media_type="text/plain; charset=utf-8")

    name = _safe_filename(task["title"] or f"exam_{task_id}")
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers=_attachment(f"{name}{suffix}.md"),
    )


@app.get("/tasks/{task_id}/pages/{page_index}/download")
async def download_page(task_id: str, page_index: int, inline: bool = False):
    page = _page_or_404(task_id, page_index)
    if page["status"] != store.DONE:
        raise HTTPException(400, f"Page {page_index + 1} is not ready yet")

    markdown = page_markdown(page["markdown"] or "", page["label_mapping"])
    markdown, _ = await asyncio.to_thread(image_host.host_markdown, markdown, task_id)

    if inline:
        return PlainTextResponse(markdown, media_type="text/plain; charset=utf-8")

    task = _task_or_404(task_id)
    name = _safe_filename(task["title"] or f"exam_{task_id}")
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers=_attachment(f"{name}_page{page_index + 1}.md"),
    )


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


@app.get("/tasks/{task_id}/images/{filename}")
async def serve_figure(task_id: str, filename: str):
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid filename")
    path = config.images_dir(task_id) / filename
    if not path.is_file():
        raise HTTPException(404, f"No such image: {filename}")
    return FileResponse(path)


@app.get("/tasks/{task_id}/pages/{page_index}/raw")
async def serve_raw_page(task_id: str, page_index: int):
    path = config.page_image_path(task_id, page_index)
    if not path.is_file():
        raise HTTPException(404, "Original page image not found")
    return FileResponse(path)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "queue": worker.pending,
        "ocr_engine": config.OCR_ENGINE,
        "ocr_workers": config.OCR_WORKERS,
        "image_host": config.IMAGE_HOST,
        "imgbb_configured": bool(config.IMGBB_API_KEY),
        "deepseek_configured": config.deepseek_ready(),
        "deepseek_model": config.DEEPSEEK_MODEL,
    }
