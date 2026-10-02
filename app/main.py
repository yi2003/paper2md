"""Paper2MD — upload exam paper photos, get markdown.

Run with:  python -m app    (or ./run.sh)
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, deepseek, image_host, pdf, store
from .markdown_utils import (
    ROW_MAX_HEIGHT_PX,
    detect_questions,
    extract_filenames,
    figures_in_rows,
    normalize_markdown,
    normalize_question,
    page_markdown,
    stale_mapping_names,
    strip_labels,
    strip_figure,
    wrap_figures_in_row,
)
from .worker import auto_polish_pending, polish_runner, worker

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
    auto_polish_pending()  # resume PDF cleanups a restart cut short
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
        # Which engine this page is read with, pinned or configured.
        "engine": store.page_engine(page),
    }


def _mapping_of(page: dict) -> dict[str, str]:
    try:
        parsed = json.loads(page.get("label_mapping") or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


async def _ingest_pdf(task_id: str, upload: UploadFile) -> dict:
    """Render a PDF into page images and queue them for OCR.

    Shared by ``POST /api/tasks/{id}/pdf`` and the one-shot
    ``POST /api/convert/pdf``. Rendering is synchronous but fast (pdfium), so it
    runs in a thread; the OCR itself is queued and reported through the normal
    per-page status. Nothing here is PDF-specific once the pages are on disk —
    the pages are byte-for-byte what a photo upload would have produced.
    """
    if not pdf.is_available():
        raise HTTPException(
            400, "PDF support is not installed — run 'pip install pypdfium2' and restart"
        )

    name = upload.filename or "document.pdf"
    if Path(name).suffix.lower() not in config.PDF_SUFFIXES:
        raise HTTPException(400, f"{name} is not a PDF")

    raw = await upload.read()
    if not raw:
        raise HTTPException(400, f"{name} is empty")
    limit_bytes = config.MAX_PDF_MB * 1024 * 1024
    if len(raw) > limit_bytes:
        raise HTTPException(400, f"{name} is larger than {config.MAX_PDF_MB} MB")

    # Keep the original alongside the task, so a page can always be traced back
    # to the PDF (and page number) it came from.
    source = config.pdf_dir(task_id) / f"{uuid.uuid4().hex[:8]}_{_safe_filename(name)}"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(raw)

    first_index = store.next_page_index(task_id)
    try:
        result = await asyncio.to_thread(
            pdf.render_pages,
            source,
            config.pages_dir(task_id),
            first_index=first_index,
            first_page=config.PDF_FIRST_PAGE,
            last_page=config.PDF_LAST_PAGE or None,
        )
    except pdf.PdfError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - report, do not leak a traceback
        source.unlink(missing_ok=True)
        log(f"[{task_id}] could not render {name}: {exc}")
        raise HTTPException(500, f"could not render the PDF: {exc}") from exc

    if not result.pages:
        # Either every page in range was blank, or the range was empty. Keep the
        # message specific so the user knows which knob to turn.
        if result.skipped:
            raise HTTPException(
                400,
                f"all {len(result.skipped)} page(s) of {name} look blank — "
                "set PDF_SKIP_BLANK=0 to read them anyway",
            )
        raise HTTPException(400, f"{name} produced no pages")

    accepted: list[dict] = []
    for page in result.pages:
        label = f"{name} · p{page.page_number}"
        store.add_page(task_id, page.page_index, label)
        worker.submit(task_id, page.page_index, page.path)
        accepted.append(
            {
                "page_index": page.page_index,
                "pdf_page": page.page_number,
                "filename": label,
                "width": page.width,
                "height": page.height,
            }
        )

    store.set_auto_polish(task_id, config.PDF_AUTO_POLISH)
    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)  # the cleaned paper is now stale
    log(
        f"[{task_id}] {name}: queued {len(accepted)} page(s), "
        f"skipped {len(result.skipped)} blank"
    )

    return {
        "task_id": task_id,
        "source": name,
        "pdf_pages": result.total_pages,
        "page_range": [result.first_page, result.last_page],
        "dpi": result.dpi,
        "truncated": result.truncated,
        "accepted": accepted,
        "queued_pages": len(accepted),
        "blank_pages": result.blank,
    }


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
            "max_pdf_mb": config.MAX_PDF_MB,
            "pdf_ready": pdf.is_available(),
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
            "pdf_ready": pdf.is_available(),
            "ocr_engine": config.OCR_ENGINE,
            "ocr_engines": list(config.OCR_ENGINES),
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
            "page_engine": store.page_engine(current),
            "image_host": config.IMAGE_HOST,
            "ocr_engines": list(config.OCR_ENGINES),
            "extract_figures": config.OCR_EXTRACT_FIGURES,
            "figure_page_width": config.FIGURE_PAGE_WIDTH,
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
        # True when this task will clean itself as soon as it is read, which the
        # status page uses to keep polling for the cleanup it has not seen yet.
        "auto_polish_ready": bool(task.get("auto_polish"))
        and config.PDF_AUTO_POLISH
        and config.deepseek_ready(),
    }


@app.delete("/api/tasks/{task_id}")
async def api_delete_task(task_id: str):
    _task_or_404(task_id)
    store.delete_task(task_id)
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


def _queue_reread(task_id: str, page_index: int, *, engine: str | None = None) -> dict:
    """Put a page back on the queue to be read again from its original photo.

    Everything the page is *about* survives — the photo, its filename, the
    figure numbers you set — while the text produced by the last read is
    replaced when the new one lands. A page's hand-edited markdown does not
    survive: it was written against the text this run throws away.

    Raises:
        HTTPException: the page is already queued or processing (409), or its
            original photo is gone (400).
    """
    page = _page_or_404(task_id, page_index)
    if page["status"] in (store.QUEUED, store.PROCESSING):
        raise HTTPException(409, f"Page {page_index + 1} is already {page['status']}")

    image_path = config.page_image_path(task_id, page_index)
    if not image_path.exists():
        raise HTTPException(400, "The original page image is gone — upload it again")

    if engine is not None:
        store.set_page_engine(task_id, page_index, engine)

    store.reset_page_to_queued(task_id, page_index)
    worker.submit(task_id, page_index, image_path)
    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)  # the cleaned paper is now stale
    return {
        "page_index": page_index,
        "status": store.QUEUED,
        "engine": store.page_engine(store.get_page(task_id, page_index) or {}),
    }


@app.post("/api/tasks/{task_id}/pages/{page_index}/retry")
async def api_retry_page(task_id: str, page_index: int):
    """Re-run a page that failed."""
    page = _page_or_404(task_id, page_index)
    if page["status"] != store.FAILED:
        raise HTTPException(409, f"Page {page_index + 1} is {page['status']}, not failed")
    return _queue_reread(task_id, page_index)


@app.post("/api/tasks/{task_id}/pages/{page_index}/reread")
async def api_reread_page(
    task_id: str, page_index: int, payload: dict | None = Body(default=None)
):
    """Read one page again — the fix for a page the engine got wrong.

    Unlike ``retry`` this also works on a page that read successfully, so a bad
    transcription, a missed figure or a mangled formula can be thrown away and
    done again without touching the rest of the paper. The body may carry
    ``{"engine": "paddle"}`` to read this page with the other engine, which is
    usually the point: DeepSeek reads at temperature 0, so the same engine twice
    tends to give the same answer.

    The figure numbers you set are kept. If the new read crops its figures at
    different coordinates their filenames change, and those numbers stop matching
    — the review screen reports which ones.
    """
    raw = payload if isinstance(payload, dict) else {}
    engine = raw.get("engine")
    engine = str(engine).strip().lower() if engine else None
    if engine is not None and engine not in config.OCR_ENGINES:
        raise HTTPException(
            400,
            f"Unknown engine {engine!r} — use one of {', '.join(sorted(config.OCR_ENGINES))}",
        )
    return _queue_reread(task_id, page_index, engine=engine)


# --------------------------------------------------------------------------
# PDF input
# --------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/pdf")
async def api_upload_pdf(task_id: str, file: UploadFile = File(...)):
    """Render a PDF's pages and queue them, continuing the task's page order.

    One PDF becomes N page images, each read exactly like an uploaded photo. The
    response reports which pages were queued and which blank ones were skipped.
    """
    _task_or_404(task_id)
    payload = await _ingest_pdf(task_id, file)
    return JSONResponse(payload, status_code=201)


@app.post("/api/convert/pdf")
async def api_convert_pdf(file: UploadFile = File(...), title: str = Form("")):
    """Convert a PDF in one call: create the task, render, and start reading.

    Returns ``202`` with a task id as soon as the pages are rendered — OCR runs
    in the background, so poll ``GET /api/tasks/{id}`` for progress and then
    ``GET /api/tasks/{id}/markdown`` (or ``?variant=polished``) for the result.
    """
    if not pdf.is_available():
        raise HTTPException(
            400, "PDF support is not installed — run 'pip install pypdfium2' and restart"
        )

    name = file.filename or "document.pdf"
    task_id = store.create_task((title or "").strip() or Path(name).stem or "paper")
    try:
        payload = await _ingest_pdf(task_id, file)
    except Exception:
        # Do not leave a half-created, pageless task behind on a bad upload.
        store.delete_task(task_id)
        shutil.rmtree(config.task_dir(task_id), ignore_errors=True)
        raise

    return JSONResponse(payload, status_code=202)


# --------------------------------------------------------------------------
# Page markdown & figure → question mapping
# --------------------------------------------------------------------------


@app.get("/api/tasks/{task_id}/pages/{page_index}")
async def api_page(task_id: str, page_index: int):
    page = _page_or_404(task_id, page_index)
    markdown = page["markdown"] or ""
    mapping = _mapping_of(page)
    edited = (page.get("edited_markdown") or "").strip()
    # What the page would be from the OCR output plus the mapping — always
    # available, and what "revert" goes back to.
    generated = store.page_generated_markdown(page)
    # What the page actually is: the hand-edited text when there is one, plus
    # any figure the user outlined on the photo.
    effective = store.page_plain_markdown(page)
    preview = store.page_render_markdown(page)
    crops = store.page_crops(page)
    figures = extract_filenames(effective)
    return {
        "page_index": page_index,
        "status": page["status"],
        "error": page["error"],
        "markdown": markdown,
        # The text as it would be stored, and the same text as the paper shows
        # it — the difference is the figure display sizes.
        "labelled_markdown": effective,
        "preview": preview,
        "generated_markdown": generated,
        "edited": bool(edited),
        "edited_markdown": edited,
        "mapping": mapping,
        "figures": figures,
        # Figures outlined by hand, which are removed rather than dropped.
        "crops": [crop["name"] for crop in crops],
        "row_figures": figures_in_rows(effective),
        # Figure numbers left pointing at figures this page no longer has.
        "stale_mapping": stale_mapping_names(mapping, figures),
        "engine": store.page_engine(page),
        "questions": detect_questions(strip_labels(effective)),
        "image_count": page["image_count"],
    }


def _page_markdown_payload(page: dict) -> dict:
    """The versions of a page's markdown, as the editor needs them.

    ``markdown`` is the text — what the editor shows and what saving stores.
    ``preview`` is the same text with the figure display sizes applied, which is
    what the paper and every download use.
    """
    generated = store.page_generated_markdown(page)
    edited = (page.get("edited_markdown") or "").strip()
    effective = store.page_plain_markdown(page)
    preview = store.page_render_markdown(page)
    return {
        "page_index": page["page_index"],
        "markdown": effective,
        "preview": preview,
        "generated_markdown": generated,
        "edited": bool(edited),
        "edited_markdown": edited,
        "row_figures": figures_in_rows(preview),
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
        # A hand-edited page is its own markdown, so the mapping only shows up
        # here once the page has been reverted to the generated version.
        **_page_markdown_payload(_page_or_404(task_id, page_index)),
    }


@app.put("/api/tasks/{task_id}/pages/{page_index}/markdown")
async def api_save_markdown(
    task_id: str, page_index: int, payload: dict | None = Body(default=None)
):
    """Save the hand-edited markdown of a page, or revert to the generated one.

    The body is ``{"markdown": "…"}`` to save, ``{"reset": true}`` to go back to
    what the OCR output plus the figure mapping produce. Empty text is treated
    as a revert rather than as an empty page.
    """
    page = _page_or_404(task_id, page_index)
    _task_or_404(task_id)

    raw = payload if isinstance(payload, dict) else {}
    text = raw.get("markdown")
    reset = bool(raw.get("reset")) or not isinstance(text, str) or not text.strip()

    # Normalise on the way in, exactly as OCR text is, so `<img/>`, a stray
    # `</img>` and `imgs/x.jpg` all end up as the rest of the pipeline expects.
    store.set_page_markdown(task_id, page_index, None if reset else normalize_markdown(text))
    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)  # the cleaned paper is now stale

    return _page_markdown_payload(_page_or_404(task_id, page_index))


@app.post("/api/tasks/{task_id}/pages/{page_index}/figure-row")
async def api_figure_row(
    task_id: str, page_index: int, payload: dict | None = Body(default=None)
):
    """Put several figures on one line by moving them into a flexbox row.

    The body is ``{"figures": ["a.jpg", "b.jpg"]}``, optionally with a
    ``max_height`` in pixels. This edits the page's markdown — the user's own
    version when there is one, the generated one otherwise — so the row can
    afterwards be tweaked by hand in the markdown editor.
    """
    page = _page_or_404(task_id, page_index)
    _task_or_404(task_id)
    if page["status"] != store.DONE:
        raise HTTPException(400, f"Page {page_index + 1} is not ready yet")

    raw = payload if isinstance(payload, dict) else {}
    figures = raw.get("figures")
    if not isinstance(figures, list) or not figures:
        raise HTTPException(400, "Pick at least two figures to put on one line")

    try:
        max_height = int(raw.get("max_height") or ROW_MAX_HEIGHT_PX)
    except (TypeError, ValueError):
        max_height = ROW_MAX_HEIGHT_PX
    max_height = max(60, min(800, max_height))

    current = store.page_plain_markdown(page)
    try:
        updated = wrap_figures_in_row(current, [str(name) for name in figures], max_height)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    store.set_page_markdown(task_id, page_index, updated)
    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)  # the cleaned paper is now stale

    return _page_markdown_payload(_page_or_404(task_id, page_index))


# --------------------------------------------------------------------------
# Outlining a figure by hand
# --------------------------------------------------------------------------
#
# The layout model misses things: a small diagram inside a paragraph, a figure
# with no border, a table it reads as prose. The fix is to draw the box
# yourself. The rectangle is sent as fractions of the photo, so it means the
# same thing whatever size the browser showed it at, and the crop is taken from
# the original file rather than the downscaled copy the model sees — a
# hand-drawn figure is as sharp as the photo.

CROP_MIN_PIXELS = 12


def _normalized_box(payload: dict) -> tuple[float, float, float, float]:
    """``(x, y, w, h)`` as fractions of the photo, from either form.

    Accepts ``{x, y, w, h}`` or ``{x1, y1, x2, y2}``, because a rectangle drawn
    on a photo is naturally described both ways.
    """
    def number(*keys: str) -> float | None:
        for key in keys:
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    raise HTTPException(400, f"{key!r} must be a number") from None
        return None

    if all(key in payload for key in ("x1", "y1", "x2", "y2")):
        x1, y1 = number("x1"), number("y1")
        x2, y2 = number("x2"), number("y2")
        return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)

    x, y, w, h = number("x"), number("y"), number("w"), number("h")
    if None in (x, y, w, h):
        raise HTTPException(
            400, "Draw a box on the photo first — {x, y, w, h} or {x1, y1, x2, y2}"
        )
    return x, y, w, h


def _crop_pixels(
    photo: Path, box: tuple[float, float, float, float]
) -> tuple[tuple[int, int, int, int], int, int]:
    """Turn fractions of the photo into a padded pixel rectangle."""
    x, y, width, height = box
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        raise HTTPException(400, "The box must lie inside the photo (0 to 1)")

    from PIL import Image

    with Image.open(photo) as image:
        page_w, page_h = image.size

    x1, y1 = int(x * page_w), int(y * page_h)
    x2, y2 = int((x + width) * page_w), int((y + height) * page_h)
    # Judge the box the user drew, not the padded crop: padding would otherwise
    # turn a stray click into a 12x12 "figure".
    if x2 - x1 < CROP_MIN_PIXELS or y2 - y1 < CROP_MIN_PIXELS:
        raise HTTPException(400, "That box is too small to be a figure")

    # A little margin, so the outline of the figure is not cut off.
    pad = config.LAYOUT_PADDING
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(page_w, x2 + pad)
    y2 = min(page_h, y2 + pad)
    return (x1, y1, x2, y2), page_w, page_h


@app.post("/api/tasks/{task_id}/pages/{page_index}/crop")
async def api_manual_crop(
    task_id: str, page_index: int, payload: dict | None = Body(default=None)
):
    """Crop the rectangle the user drew on the page photo and add it as a figure.

    The body is ``{"x": .1, "y": .2, "w": .3, "h": .25}`` — fractions of the
    photo — plus an optional ``"question": "13"`` to put it under that question,
    exactly as typing the number on a figure card would.
    """
    page = _page_or_404(task_id, page_index)
    _task_or_404(task_id)
    if page["status"] != store.DONE:
        raise HTTPException(400, f"Page {page_index + 1} is not ready yet")

    photo = config.page_image_path(task_id, page_index)
    if not photo.is_file():
        raise HTTPException(400, "The page photo is gone — upload it again")

    raw = payload if isinstance(payload, dict) else {}
    (x1, y1, x2, y2), page_w, page_h = _crop_pixels(photo, _normalized_box(raw))

    from PIL import Image, ImageOps

    images_dir = config.images_dir(task_id)
    images_dir.mkdir(parents=True, exist_ok=True)
    # The same box-in-the-name convention the OCR crops use, so figure serving
    # and the auto-fill sort work on these too.
    name = f"p{page_index + 1}_manual_box_{x1}_{y1}_{x2}_{y2}.jpg"

    with Image.open(photo) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        crop = image.crop((x1, y1, x2, y2))
        crop.save(images_dir / name, format="JPEG", quality=95)

    crops = store.page_crops(page)
    crops = [crop for crop in crops if crop["name"] != name]
    crops.append({"name": name, "box": [x1, y1, x2, y2]})
    store.set_page_crops(task_id, page_index, crops)

    question = normalize_question(raw.get("question") or "")
    if question:
        mapping = _mapping_of(page)
        mapping[name] = question
        store.set_page_mapping(task_id, page_index, mapping)

    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)  # the cleaned paper is now stale
    log(f"[{task_id}] page {page_index + 1}: outlined {name} at ({x1},{y1})-({x2},{y2})")

    fresh = _page_or_404(task_id, page_index)
    return {
        "filename": name,
        "url": f"/tasks/{task_id}/images/{name}",
        "box": [x1, y1, x2, y2],
        "photo_size": [page_w, page_h],
        "question": question,
        **_page_markdown_payload(fresh),
    }


@app.delete("/api/tasks/{task_id}/pages/{page_index}/crop/{filename}")
async def api_delete_crop(task_id: str, page_index: int, filename: str):
    """Remove a figure the user outlined, and its number."""
    page = _page_or_404(task_id, page_index)
    name = Path(filename).name
    crops = [crop for crop in store.page_crops(page) if crop["name"] != name]
    if len(crops) == len(store.page_crops(page)):
        raise HTTPException(404, f"This page has no outlined figure called {name}")

    store.set_page_crops(task_id, page_index, crops)

    mapping = _mapping_of(page)
    if name in mapping:
        mapping.pop(name)
        store.set_page_mapping(task_id, page_index, mapping)

    # The figure may also be inside the page's hand-edited text — put in a
    # one-line row by hand — so the reference has to go from there too.
    edited = page.get("edited_markdown") or ""
    if edited and f"images/{name}" in edited:
        store.set_page_markdown(task_id, page_index, strip_figure(edited, name))

    store.rebuild_task(task_id)
    store.clear_polished_markdown(task_id)
    # Only ever delete a file this feature wrote.
    if "manual_box_" in name:
        (config.images_dir(task_id) / name).unlink(missing_ok=True)

    return _page_markdown_payload(_page_or_404(task_id, page_index))


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

    markdown = store.page_render_markdown(page)
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
        "pdf_ready": pdf.is_available(),
        "pdf_dpi": config.PDF_DPI,
    }
