"""Background OCR queue.

Uploaded pages are queued immediately; a small pool of threads drains the queue
one page at a time. Every state change is written straight to SQLite so the web
UI can poll cheaply and survive a restart.
"""

from __future__ import annotations

import queue
import shutil
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

from . import config, ocr, store


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
            markdown, figures = ocr.run_ocr(job.image_path, work_dir, images_dir, prefix)
            store.mark_page_done(job.task_id, job.page_index, markdown, len(figures))
            ocr.log(f"{label} done ({len(figures)} figure(s))")
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


worker = OcrWorker()
