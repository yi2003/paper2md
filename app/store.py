"""SQLite persistence and task aggregation.

One small local database, accessed from both the web routes and the OCR worker
threads. Every call is short and local, so a plain ``sqlite3`` connection
guarded by a re-entrant lock is simpler (and fast enough) than an async driver.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from . import config
from .markdown_utils import merge_pages, page_markdown

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
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    task_id        TEXT NOT NULL,
    page_index     INTEGER NOT NULL,
    filename       TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'queued',
    error          TEXT,
    markdown       TEXT,
    label_mapping  TEXT,
    image_count    INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
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


def _migrate(connection: sqlite3.Connection) -> None:
    """Add columns that were introduced after a database was first created."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
    for name, ddl in (
        ("polished_markdown", "ALTER TABLE tasks ADD COLUMN polished_markdown TEXT"),
        ("polish_info", "ALTER TABLE tasks ADD COLUMN polish_info TEXT"),
        ("polish_status", "ALTER TABLE tasks ADD COLUMN polish_status TEXT NOT NULL DEFAULT 'idle'"),
        ("polish_progress", "ALTER TABLE tasks ADD COLUMN polish_progress TEXT"),
        ("polish_error", "ALTER TABLE tasks ADD COLUMN polish_error TEXT"),
    ):
        if name not in columns:
            connection.execute(ddl)


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
        " image_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, NULL, NULL, NULL, 0, ?, ?)",
        (task_id, page_index, filename, QUEUED, now, now),
    )


def mark_page_processing(task_id: str, page_index: int) -> None:
    _execute(
        "UPDATE pages SET status = ?, error = NULL, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
        (PROCESSING, time.time(), task_id, page_index),
    )


def mark_page_done(task_id: str, page_index: int, markdown: str, image_count: int) -> None:
    _execute(
        "UPDATE pages SET status = ?, markdown = ?, image_count = ?, error = NULL, updated_at = ? "
        "WHERE task_id = ? AND page_index = ?",
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
            page_texts.append(page_markdown(page["markdown"] or "", page["label_mapping"]))
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
