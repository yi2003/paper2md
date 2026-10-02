"""Background jobs.

Pages are queued for OCR as soon as they are uploaded; a small pool of threads
drains the queue. The DeepSeek cleanup runs the same way. Every state change is
written straight to SQLite so the web UI can poll cheaply, show progress, and
survive a restart.
"""

from __future__ import annotations

import queue
import shutil
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

from . import config, deepseek, image_host, ocr, store


@dataclass
class Job:
    task_id: str
    page_index: int
    image_path: Path


_STOP = None


class OcrWorker:
    """A FIFO queue plus N worker threads."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self, workers: int | None = None) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            count = max(1, workers or config.OCR_WORKERS)
            for index in range(count):
                thread = threading.Thread(
                    target=self._loop, name=f"ocr-worker-{index}", daemon=True
                )
                thread.start()
                self._threads.append(thread)
        ocr.log(f"{count} OCR worker(s) started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        for _ in self._threads:
            self._queue.put(_STOP)
        for thread in self._threads:
            thread.join(timeout=10)
        self._threads.clear()

    # -- queueing ----------------------------------------------------------

    def submit(self, task_id: str, page_index: int, image_path: Path) -> None:
        self._queue.put(Job(task_id, page_index, Path(image_path)))

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def requeue_interrupted(self) -> int:
        """Re-queue pages that were mid-flight when the process last stopped."""
        count = 0
        for page in store.pages_with_status(store.PROCESSING):
            image_path = config.page_image_path(page["task_id"], page["page_index"])
            if not image_path.exists():
                store.mark_page_failed(
                    page["task_id"], page["page_index"], "source image missing after restart"
                )
                continue
            store.reset_page_to_queued(page["task_id"], page["page_index"])
            self.submit(page["task_id"], page["page_index"], image_path)
            count += 1
        if count:
            ocr.log(f"re-queued {count} interrupted page(s)")
        return count

    # -- internals ---------------------------------------------------------

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is _STOP:
                    return
                self._process(job)
            except Exception:  # noqa: BLE001 - a bad job must not kill the worker
                ocr.log("worker loop error:\n" + traceback.format_exc())
            finally:
                self._queue.task_done()

    def _process(self, job: Job) -> None:
        label = f"[{job.task_id}] page {job.page_index + 1}"
        images_dir = config.images_dir(job.task_id)
        work_dir = config.work_dir(job.task_id, job.page_index)
        prefix = f"p{job.page_index + 1}_"

        store.mark_page_processing(job.task_id, job.page_index)
        store.rebuild_task(job.task_id)
        ocr.log(f"{label} started")

        try:
            page = store.get_page(job.task_id, job.page_index) or {}
            # A page re-read with a different engine keeps that pin, so the
            # choice survives a restart and applies to any later re-read.
            engine = (page.get("ocr_engine") or "").strip()

            if engine:
                markdown, figures, suggested = ocr.run_ocr(
                    job.image_path, work_dir, images_dir, prefix, engine=engine
                )
            else:
                # The plain call keeps the usual four-argument contract, which
                # anything stubbing the OCR still honours.
                markdown, figures, suggested = ocr.run_ocr(
                    job.image_path, work_dir, images_dir, prefix
                )

            # An engine may work out which question each figure belongs to.
            # Store it as the page's mapping — never baked into the text — so the
            # renderer applies it and the user can adjust it in the review screen.
            if suggested:
                existing = store.get_page(job.task_id, job.page_index) or {}
                if not (existing.get("label_mapping") or "").strip():
                    store.set_page_mapping(job.task_id, job.page_index, suggested)
                    ocr.log(f"{label} matched {len(suggested)} figure(s) to questions")
            store.mark_page_done(job.task_id, job.page_index, markdown, len(figures))
            ocr.log(
                f"{label} done ({len(figures)} figure(s), engine {engine or config.OCR_ENGINE})"
            )
        except Exception as exc:  # noqa: BLE001 - recorded and shown in the UI
            if config.OCR_DEBUG:
                ocr.log(traceback.format_exc())
            store.mark_page_failed(
                job.task_id, job.page_index, f"{type(exc).__name__}: {exc}"
            )
            ocr.log(f"{label} failed: {exc}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            store.rebuild_task(job.task_id)
            maybe_auto_polish(job.task_id)


worker = OcrWorker()


class PolishRunner:
    """Runs the DeepSeek cleanup for a task on a background thread.

    At most one cleanup per task. Progress is written to SQLite after every
    figure and every chunk, so the web UI can poll it exactly like OCR progress
    instead of staring at a blocked request.
    """

    def __init__(self) -> None:
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._active

    def start(self, task_id: str) -> bool:
        """Start a cleanup. Returns False if one is already running."""
        with self._lock:
            if task_id in self._active:
                return False
            self._active.add(task_id)
        threading.Thread(
            target=self._run, args=(task_id,), name=f"polish-{task_id}", daemon=True
        ).start()
        return True

    def _run(self, task_id: str) -> None:
        try:
            task = store.rebuild_task(task_id)
            markdown = (task or {}).get("merged_markdown") or ""
            if not markdown.strip():
                raise deepseek.DeepSeekError("there is nothing to clean up yet")

            store.set_polish_state(
                task_id, store.POLISH_RUNNING, stage="hosting", done=0, total=0
            )

            def on_figures(done: int, total: int) -> None:
                store.set_polish_state(
                    task_id, store.POLISH_RUNNING, stage="hosting", done=done, total=total
                )

            # Host figures first, so the cleaned paper references permanent URLs.
            markdown, _ = image_host.host_markdown(markdown, task_id, progress=on_figures)

            def on_chunks(stage: str, done: int, total: int) -> None:
                store.set_polish_state(
                    task_id, store.POLISH_RUNNING, stage=stage, done=done, total=total
                )

            cleaned, info = deepseek.polish_markdown(markdown, progress=on_chunks)
            store.set_polished_markdown(task_id, cleaned, info)
            ocr.log(
                f"[{task_id}] cleaned with {info.get('model')}: "
                f"{len(markdown):,} -> {len(cleaned):,} chars, "
                f"{info.get('figures_total', 0)} figure(s)"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            if config.OCR_DEBUG:
                ocr.log(traceback.format_exc())
            store.set_polish_state(task_id, store.POLISH_FAILED, error=str(exc))
            ocr.log(f"[{task_id}] cleanup failed: {exc}")
        finally:
            with self._lock:
                self._active.discard(task_id)


polish_runner = PolishRunner()


# --------------------------------------------------------------------------
# Automatic cleanup for PDF tasks
# --------------------------------------------------------------------------
#
# A PDF is supposed to come out as a *clean* paper, so its cleanup does not wait
# for someone to press ✨ Clean up: once every page has been read successfully,
# the DeepSeek pass starts on its own. The flag lives on the task (set when a PDF
# is ingested) rather than on the process, so it survives a restart and photo
# tasks are unaffected.


def maybe_auto_polish(task_id: str) -> bool:
    """Start the cleanup if this fully-read task asked for it. Idempotent."""
    if not config.PDF_AUTO_POLISH or not config.deepseek_ready():
        return False
    if not store.auto_polish_enabled(task_id):
        return False

    task = store.get_task(task_id)
    if task is None or task["status"] != store.TASK_DONE:
        # Still reading, or a page failed — a half-read paper must not be cleaned.
        return False
    if not (task.get("merged_markdown") or "").strip():
        return False
    if polish_runner.is_running(task_id):
        return False

    started = polish_runner.start(task_id)
    if started:
        ocr.log(f"[{task_id}] all pages read — auto-cleaning with DeepSeek")
    return started


def auto_polish_pending() -> int:
    """Re-arm auto-clean for tasks left un-cleaned by a restart.

    Returns how many were started. Safe to call at startup: tasks already
    cleaned, still reading, or currently cleaning are ignored.
    """
    if not config.PDF_AUTO_POLISH or not config.deepseek_ready():
        return 0
    started = 0
    for task_id in store.tasks_awaiting_auto_polish():
        if polish_runner.start(task_id):
            ocr.log(f"[{task_id}] auto-clean resumed after restart")
            started += 1
    return started
