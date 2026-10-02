"""SQLite persistence and task aggregation.

One small local database, accessed from both the web routes and the OCR worker
threads. Every call is short and local, so a plain ``sqlite3`` connection
guarded by a re-entrant lock is simpler (and fast enough) than an async driver.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config
from .markdown_utils import (
    append_crops,
    extract_baked_labels,
    merge_pages,
    page_markdown,
    style_figures,
    wrap_bare_latex,
)

# Page statuses
QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"

# Task statuses
TASK_QUEUED = "queued"
TASK_PROCESSING = "processing"
TASK_DONE = "done"
TASK_PARTIAL = "partial"
TASK_FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'queued',
    merged_markdown   TEXT,
    polished_markdown TEXT,
    polish_info       TEXT,
    polish_status     TEXT NOT NULL DEFAULT 'idle',
    polish_progress   TEXT,
    polish_error      TEXT,
    auto_polish       INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    task_id         TEXT NOT NULL,
    page_index      INTEGER NOT NULL,
    filename        TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'queued',
    error           TEXT,
    markdown        TEXT,
    label_mapping   TEXT,
    edited_markdown TEXT,
    manual_crops    TEXT,
    ocr_engine      TEXT,
    image_count     INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    PRIMARY KEY (task_id, page_index)
);

CREATE TABLE IF NOT EXISTS image_urls (
    task_id    TEXT NOT NULL,
    filename   TEXT NOT NULL,
    url        TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (task_id, filename)
);

CREATE INDEX IF NOT EXISTS idx_pages_task ON pages (task_id, page_index);
"""

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


# --- plumbing ----------------------------------------------------------------


def init() -> None:
    """Open the database (creating directories and schema as needed)."""
    global _conn
    config.ensure_dirs()
    with _lock:
        if _conn is not None:
            return
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(SCHEMA)
        _migrate(_conn)
        _conn.commit()


def _migrate_page_labels(connection: sqlite3.Connection) -> None:
    """Move figure labels baked into page text into the page mapping.

    Engines used to write "*[Q13 附图]*" into the stored markdown. Labels are
    applied at render time now, so the baked ones are stripped and recovered as
    a mapping. An existing mapping is never overwritten — the user's own numbers
    win over whatever the engine guessed.
    """
    rows = connection.execute(
        "SELECT task_id, page_index, markdown, label_mapping FROM pages "
        "WHERE markdown IS NOT NULL AND markdown LIKE '%附图%'"
    ).fetchall()

    migrated = 0
    for task_id, page_index, markdown, mapping_json in rows:
        cleaned, found = extract_baked_labels(markdown)
        if not found or cleaned == markdown:
            continue
        try:
            existing = json.loads(mapping_json) if mapping_json else {}
        except json.JSONDecodeError:
            existing = {}
        merged = {**found, **existing}
        connection.execute(
            "UPDATE pages SET markdown = ?, label_mapping = ? "
            "WHERE task_id = ? AND page_index = ?",
            (cleaned, json.dumps(merged, ensure_ascii=False), task_id, page_index),
        )
        migrated += 1

    if migrated:
        print(f"[store] moved baked figure labels to mappings on {migrated} page(s)")


def _migrate(connection: sqlite3.Connection) -> None:
    """Add columns that were introduced after a database was first created."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
    for name, ddl in (
        ("polished_markdown", "ALTER TABLE tasks ADD COLUMN polished_markdown TEXT"),
        ("polish_info", "ALTER TABLE tasks ADD COLUMN polish_info TEXT"),
        ("polish_status", "ALTER TABLE tasks ADD COLUMN polish_status TEXT NOT NULL DEFAULT 'idle'"),
        ("polish_progress", "ALTER TABLE tasks ADD COLUMN polish_progress TEXT"),
        ("polish_error", "ALTER TABLE tasks ADD COLUMN polish_error TEXT"),
        (
            "auto_polish",
            "ALTER TABLE tasks ADD COLUMN auto_polish INTEGER NOT NULL DEFAULT 0",
        ),
    ):
        if name not in columns:
            connection.execute(ddl)

    # The hand-edited markdown of a page: what the markdown editor saved, used
    # in place of the mapping-derived text until it is reverted.
    # ``manual_crops`` holds the figures the user outlined on the page photo
    # because the layout model missed them.
    # ``ocr_engine`` pins one page to an engine, so it can be re-read with a
    # second opinion without changing the setting for the whole app.
    page_columns = {row[1] for row in connection.execute("PRAGMA table_info(pages)")}
    for name, ddl in (
        ("edited_markdown", "ALTER TABLE pages ADD COLUMN edited_markdown TEXT"),
        ("manual_crops", "ALTER TABLE pages ADD COLUMN manual_crops TEXT"),
        ("ocr_engine", "ALTER TABLE pages ADD COLUMN ocr_engine TEXT"),
    ):
        if name not in page_columns:
            connection.execute(ddl)

    _migrate_page_labels(connection)

    # Tasks cleaned before polish_status existed took the column default
    # ('idle'), which understates reality: a stored cleaned paper means done.
    connection.execute(
        "UPDATE tasks SET polish_status = ? "
        "WHERE polished_markdown IS NOT NULL AND polish_status = ?",
        (POLISH_DONE, POLISH_IDLE),
    )


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def _execute(sql: str, params: tuple = ()) -> None:
    with _lock:
        _conn.execute(sql, params)  # type: ignore[union-attr]
        _conn.commit()  # type: ignore[union-attr]


def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with _lock:
        rows = _conn.execute(sql, params).fetchall()  # type: ignore[union-attr]
    return [dict(row) for row in rows]


def _query_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    rows = _query(sql, params)
    return rows[0] if rows else None


# --- tasks -------------------------------------------------------------------


def create_task(title: str = "") -> str:
    task_id = f"t-{uuid.uuid4().hex[:12]}"
    now = time.time()
    _execute(
        "INSERT INTO tasks (id, title, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, title.strip(), TASK_QUEUED, now, now),
    )
    return task_id


def list_tasks() -> list[dict[str, Any]]:
    tasks = _query("SELECT * FROM tasks ORDER BY created_at DESC")
    for task in tasks:
        task["pages"] = _query(
            "SELECT * FROM pages WHERE task_id = ? ORDER BY page_index", (task["id"],)
        )
        task["page_count"] = len(task["pages"])
        task["done_count"] = sum(1 for p in task["pages"] if p["status"] == DONE)
        task["image_count"] = sum(p["image_count"] or 0 for p in task["pages"])
    return tasks


def get_task(task_id: str) -> dict[str, Any] | None:
    task = _query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
    if task is None:
        return None
    task["pages"] = _query(
        "SELECT * FROM pages WHERE task_id = ? ORDER BY page_index", (task_id,)
    )
    return task


def get_page(task_id: str, page_index: int) -> dict[str, Any] | None:
    return _query_one(
        "SELECT * FROM pages WHERE task_id = ? AND page_index = ?", (task_id, page_index)
    )


def pages_with_status(status: str) -> list[dict[str, Any]]:
    return _query(
        "SELECT * FROM pages WHERE status = ? ORDER BY created_at", (status,)
    )


def reset_page_to_queued(task_id: str, page_index: int) -> None:
    _execute(
        "UPDATE pages SET status = ?, error = NULL, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (QUEUED, time.time(), task_id, page_index),
    )


def next_page_index(task_id: str) -> int:
    row = _query_one(
        "SELECT COALESCE(MAX(page_index), -1) + 1 AS next FROM pages WHERE task_id = ?",
        (task_id,),
    )
    return int(row["next"]) if row else 0


def add_page(task_id: str, page_index: int, filename: str) -> None:
    now = time.time()
    _execute(
        "INSERT OR REPLACE INTO pages "
        "(task_id, page_index, filename, status, error, markdown, label_mapping, "
        " edited_markdown, manual_crops, ocr_engine, image_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, 0, ?, ?)",
        (task_id, page_index, filename, QUEUED, now, now),
    )


def mark_page_processing(task_id: str, page_index: int) -> None:
    _execute(
        "UPDATE pages SET status = ?, error = NULL, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (PROCESSING, time.time(), task_id, page_index),
    )


def mark_page_done(task_id: str, page_index: int, markdown: str, image_count: int) -> None:
    """Record a finished read.

    Any hand-edited markdown is dropped: it was written against the previous
    OCR output, which this call has just replaced wholesale.
    """
    _execute(
        "UPDATE pages SET status = ?, markdown = ?, image_count = ?, edited_markdown = NULL, "
        "error = NULL, updated_at = ? WHERE task_id = ? AND page_index = ?",
        (DONE, markdown, image_count, time.time(), task_id, page_index),
    )


def mark_page_failed(task_id: str, page_index: int, error: str) -> None:
    _execute(
        "UPDATE pages SET status = ?, error = ?, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (FAILED, error[:2000], time.time(), task_id, page_index),
    )


def set_page_mapping(task_id: str, page_index: int, mapping: dict[str, str]) -> None:
    _execute(
        "UPDATE pages SET label_mapping = ?, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (json.dumps(mapping, ensure_ascii=False), time.time(), task_id, page_index),
    )


def set_page_markdown(task_id: str, page_index: int, markdown: str | None) -> None:
    """Save the hand-edited markdown of a page; ``None`` reverts to generated."""
    _execute(
        "UPDATE pages SET edited_markdown = ?, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (markdown, time.time(), task_id, page_index),
    )


def set_page_engine(task_id: str, page_index: int, engine: str | None) -> None:
    """Pin a page to one OCR engine; ``None`` goes back to the configured one."""
    _execute(
        "UPDATE pages SET ocr_engine = ?, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (engine, time.time(), task_id, page_index),
    )


def page_engine(page: dict[str, Any]) -> str:
    """The engine a page is read with: its own pin, else the configured one."""
    pinned = (page.get("ocr_engine") or "").strip()
    return pinned or config.OCR_ENGINE


def page_crops(page: dict[str, Any]) -> list[dict[str, Any]]:
    """The figures the user outlined on this page's photo, in order.

    Each entry is ``{"name": filename, "box": [x1, y1, x2, y2]}`` with the box in
    pixels of the original photo. Kept apart from the OCR text so that reading
    the page again cannot throw a hand-drawn figure away.
    """
    raw = page.get("manual_crops") or ""
    try:
        parsed = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        {"name": str(item.get("name") or ""), "box": item.get("box")}
        for item in parsed
        if isinstance(item, dict) and item.get("name")
    ]


def set_page_crops(task_id: str, page_index: int, crops: list[dict[str, Any]]) -> None:
    _execute(
        "UPDATE pages SET manual_crops = ?, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (json.dumps(crops, ensure_ascii=False), time.time(), task_id, page_index),
    )


# Image sizes, by file and mtime: reading the header is cheap, but a rebuild
# re-renders every page on every keystroke in the figure mapping, so it is worth
# remembering. Keyed on the stat so a re-crop is never served a stale size.
_image_size_cache: dict[tuple[str, int, int], tuple[int, int] | None] = {}


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    key = (str(path), info.st_mtime_ns, info.st_size)
    if key not in _image_size_cache:
        if len(_image_size_cache) > 4096:
            _image_size_cache.clear()
        size = None
        try:
            from PIL import Image

            with Image.open(path) as image:
                size = (int(image.size[0]), int(image.size[1]))
        except Exception:  # noqa: BLE001 - an unreadable file just gets no size
            size = None
        _image_size_cache[key] = size
    return _image_size_cache[key]


_BOX_IN_NAME_RE = re.compile(r"_box_(\d+)_(\d+)_(\d+)_(\d+)\.")


def box_from_name(name: str) -> list[int] | None:
    """The rectangle an engine cropped, read back out of the figure's name.

    Both engines name a crop after the box it came from —
    ``p2_img_in_image_box_475_2365_1335_2938.jpg`` — so the figure's own name
    says how much of the page it covered, which is what it should be shown at.
    """
    match = _BOX_IN_NAME_RE.search(name)
    if not match:
        return None
    return [int(value) for value in match.groups()]


def figure_sizer(task_id: str, page_index: int, boxes: dict[str, list]):
    """A ``size_for(ref) -> (width, height)`` for one page's figures.

    The size a figure should *be* is the size of the rectangle that made it: a
    diagram drawn across a third of the page is a third of the page wide in the
    paper. So the rectangle is measured against the page, and the full width of
    the page counts as ``FIGURE_PAGE_WIDTH`` pixels.

    The two kinds of rectangle are in different coordinate spaces, which does
    not matter because only the *share* of the page matters: a figure the user
    outlined carries its box in page-photo pixels, while an engine crop carries
    its box in the pixels of the downscaled copy the model saw. Dividing by the
    width of the image that box was measured in gives the same fraction either
    way.

    The figures' own files are the fallback for anything with no box at all, so
    nothing is ever enlarged to fill the space.
    """
    page_size = _image_size(config.page_image_path(task_id, page_index))
    page_width = config.FIGURE_PAGE_WIDTH
    prepared_width = _prepared_width(page_size)

    def size_for(ref: str) -> tuple[int | None, int | None]:
        name = Path(ref.split("?", 1)[0]).name
        if not name:
            return None, None

        box = boxes.get(name)
        if box and len(box) == 4 and page_size:
            return _size_of_box(box, page_size[0], page_width)
        if page_size and prepared_width:
            engine_box = box_from_name(name)
            if engine_box:
                return _size_of_box(engine_box, prepared_width, page_width)

        natural = _image_size(config.images_dir(task_id) / name)
        return (natural[0], natural[1]) if natural else (None, None)

    return size_for


def _prepared_width(page_size: tuple[int, int] | None) -> int:
    """The width of the downscaled copy the model reads, from the page size.

    ``ocr.prepare_input`` only ever shrinks the long edge to ``OCR_MAX_DIM``, so
    the width of the copy is derivable without keeping the file around.
    """
    if not page_size or not page_size[0]:
        return 0
    limit = config.OCR_MAX_DIM
    longest = max(page_size)
    if not limit or longest <= limit:
        return page_size[0]
    return max(1, round(page_size[0] * limit / longest))


def _size_of_box(box: list[int], page_width: int, target_width: int):
    """The width and height a rectangle of the page should be shown at."""
    if not page_width or len(box) != 4:
        return None, None
    x1, y1, x2, y2 = box
    width = round(max(0, x2 - x1) / page_width * target_width)
    # The height the same rectangle would have on a page that many pixels wide,
    # so a figure can never come out taller than it was on the page.
    height = round(max(0, y2 - y1) / page_width * target_width)
    return width or None, height or None


def page_generated_markdown(page: dict[str, Any]) -> str:
    """What the page would be without a hand edit — exactly what revert restores.

    Unstyled on purpose: this is the text, and it is what the markdown editor
    shows when you press **Revert to generated**.
    """
    return page_markdown(page["markdown"] or "", page["label_mapping"])


def page_plain_markdown(page: dict[str, Any]) -> str:
    """The page's text as it would be stored: no display sizes.

    Hand-outlined figures are appended *before* the mapping runs, so a crop with
    a question number is placed like any other figure. A hand-edited page keeps
    its own text, so they are appended to that instead — also unsized, because
    hand-edited text is used exactly as saved.
    """
    raw = page["markdown"] or ""
    crops = page_crops(page)
    edited = page.get("edited_markdown") or ""
    names = [crop["name"] for crop in crops]

    if edited.strip():
        return append_crops(page_markdown(raw, page["label_mapping"], edited), names)
    return page_markdown(append_crops(raw, names), page["label_mapping"], edited)


def page_render_markdown(page: dict[str, Any]) -> str:
    """What the paper shows: the text with the figure display sizes applied.

    Presentation, decided here rather than baked into what is stored — so
    changing ``FIGURE_PAGE_WIDTH`` changes every page, including the hand-edited
    ones, and the text itself is never rewritten.
    """
    if not config.FIGURE_PAGE_WIDTH:
        return page_plain_markdown(page)  # sizing switched off

    boxes = {crop["name"]: crop.get("box") or [] for crop in page_crops(page)}
    return style_figures(
        wrap_bare_latex(page_plain_markdown(page)),
        config.FIGURE_MAX_WIDTH,
        config.FIGURE_MAX_HEIGHT,
        size_for=figure_sizer(page.get("task_id") or "", page["page_index"], boxes),
    )


def delete_task(task_id: str) -> None:
    _execute("DELETE FROM pages WHERE task_id = ?", (task_id,))
    _execute("DELETE FROM image_urls WHERE task_id = ?", (task_id,))
    _execute("DELETE FROM tasks WHERE id = ?", (task_id,))


# --- aggregation -------------------------------------------------------------


def _derive_status(pages: list[dict[str, Any]]) -> str:
    if not pages:
        return TASK_QUEUED
    statuses = [page["status"] for page in pages]
    if all(status == DONE for status in statuses):
        return TASK_DONE
    if all(status == FAILED for status in statuses):
        return TASK_FAILED
    if any(status == PROCESSING for status in statuses):
        return TASK_PROCESSING
    if any(status == DONE for status in statuses):
        # Some finished, some still queued or failed.
        return TASK_PARTIAL if all(status in (DONE, FAILED) for status in statuses) else TASK_PROCESSING
    if any(status == QUEUED for status in statuses):
        return TASK_QUEUED
    return TASK_PARTIAL


def rebuild_task(task_id: str) -> dict[str, Any] | None:
    """Recompute the task's merged markdown and status from its pages."""
    task = get_task(task_id)
    if task is None:
        return None

    pages = task["pages"]
    page_texts: list[str] = []
    for page in pages:
        if page["status"] == DONE:
            page_texts.append(page_render_markdown(page))
        elif page["status"] == FAILED:
            page_texts.append(
                f"<!-- page {page['page_index'] + 1} FAILED: {page['error'] or 'unknown error'} -->"
            )
        else:
            page_texts.append("")

    merged = merge_pages(page_texts) if pages else ""
    status = _derive_status(pages)
    _execute(
        "UPDATE tasks SET status = ?, merged_markdown = ?, updated_at = ? WHERE id = ?",
        (status, merged, time.time(), task_id),
    )
    task["status"] = status
    task["merged_markdown"] = merged
    return task


# --- DeepSeek cleanup result -------------------------------------------------

POLISH_IDLE = "idle"
POLISH_RUNNING = "running"
POLISH_DONE = "done"
POLISH_FAILED = "failed"


def set_polished_markdown(
    task_id: str, markdown: str, info: dict[str, Any] | None = None
) -> None:
    _execute(
        "UPDATE tasks SET polished_markdown = ?, polish_info = ?, polish_status = ?, "
        "polish_progress = NULL, polish_error = NULL, updated_at = ? WHERE id = ?",
        (
            markdown,
            json.dumps(info, ensure_ascii=False) if info else None,
            POLISH_DONE,
            time.time(),
            task_id,
        ),
    )


def set_polish_state(
    task_id: str,
    status: str,
    *,
    stage: str | None = None,
    done: int = 0,
    total: int = 0,
    error: str | None = None,
) -> None:
    """Record cleanup progress so the UI can poll it."""
    progress = (
        json.dumps({"stage": stage, "done": done, "total": total}) if stage else None
    )
    _execute(
        "UPDATE tasks SET polish_status = ?, polish_progress = ?, polish_error = ?, "
        "updated_at = ? WHERE id = ?",
        (status, progress, error[:2000] if error else None, time.time(), task_id),
    )


def get_polish_state(task_id: str) -> dict[str, Any]:
    """Current cleanup state: status, stage, done/total and any error."""
    row = _query_one(
        "SELECT polish_status, polish_progress, polish_error FROM tasks WHERE id = ?",
        (task_id,),
    )
    if row is None:
        return {"status": POLISH_IDLE, "stage": None, "done": 0, "total": 0, "error": None}

    progress: dict[str, Any] = {}
    if row["polish_progress"]:
        try:
            progress = json.loads(row["polish_progress"])
        except json.JSONDecodeError:
            progress = {}

    return {
        "status": row["polish_status"] or POLISH_IDLE,
        "stage": progress.get("stage"),
        # `done` can be fractional: a chunk in progress counts as a fraction.
        "done": round(float(progress.get("done") or 0), 3),
        "total": int(progress.get("total") or 0),
        "error": row["polish_error"],
    }


def reset_stale_polish_states() -> int:
    """Mark cleanups interrupted by a restart as failed, so they can be retried."""
    stale = _query("SELECT id FROM tasks WHERE polish_status = ?", (POLISH_RUNNING,))
    for row in stale:
        set_polish_state(
            row["id"], POLISH_FAILED, error="interrupted by an app restart — try again"
        )
    return len(stale)


def get_polished_markdown(task_id: str) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(cleaned_markdown, info)``; both None when it has not been run."""
    row = _query_one(
        "SELECT polished_markdown, polish_info FROM tasks WHERE id = ?", (task_id,)
    )
    if row is None or not row["polished_markdown"]:
        return None, None
    info: dict[str, Any] | None = None
    if row["polish_info"]:
        try:
            info = json.loads(row["polish_info"])
        except json.JSONDecodeError:
            info = None
    return row["polished_markdown"], info


def clear_polished_markdown(task_id: str) -> None:
    """Drop the cleaned version — the pages or the mapping have changed."""
    _execute(
        "UPDATE tasks SET polished_markdown = NULL, polish_info = NULL, "
        "polish_status = ?, polish_progress = NULL, polish_error = NULL, "
        "updated_at = ? WHERE id = ?",
        (POLISH_IDLE, time.time(), task_id),
    )


# --- automatic cleanup for PDF tasks -----------------------------------------


def set_auto_polish(task_id: str, enabled: bool = True) -> None:
    """Record that this task should be cleaned by itself once it is read.

    Set when a PDF is ingested: a PDF is expected to come out as a clean paper
    without anyone pressing ✨ Clean up.
    """
    _execute(
        "UPDATE tasks SET auto_polish = ?, updated_at = ? WHERE id = ?",
        (1 if enabled else 0, time.time(), task_id),
    )


def auto_polish_enabled(task_id: str) -> bool:
    row = _query_one("SELECT auto_polish FROM tasks WHERE id = ?", (task_id,))
    return bool(row and row["auto_polish"])


def tasks_awaiting_auto_polish() -> list[str]:
    """Ids of fully-read auto-clean tasks that still have no cleaned paper.

    Drives the startup sweep, so a cleanup cut short by a restart is picked up
    again rather than sitting un-cleaned forever.
    """
    rows = _query(
        "SELECT id FROM tasks WHERE auto_polish = 1 AND status = ? "
        "AND merged_markdown IS NOT NULL AND merged_markdown != '' "
        "AND (polished_markdown IS NULL OR polished_markdown = '') "
        "AND polish_status != ?",
        (TASK_DONE, POLISH_RUNNING),
    )
    return [row["id"] for row in rows]


# --- image hosting cache -----------------------------------------------------


def get_image_url(task_id: str, filename: str) -> str | None:
    row = _query_one(
        "SELECT url FROM image_urls WHERE task_id = ? AND filename = ?", (task_id, filename)
    )
    return row["url"] if row else None


def get_image_urls(task_id: str) -> dict[str, str]:
    rows = _query("SELECT filename, url FROM image_urls WHERE task_id = ?", (task_id,))
    return {row["filename"]: row["url"] for row in rows}


def save_image_url(task_id: str, filename: str, url: str) -> None:
    _execute(
        "INSERT OR REPLACE INTO image_urls (task_id, filename, url, created_at) VALUES (?, ?, ?, ?)",
        (task_id, filename, url, time.time()),
    )


def forget_image_urls(task_id: str) -> None:
    _execute("DELETE FROM image_urls WHERE task_id = ?", (task_id,))
