#!/usr/bin/env python
"""Measure OCR throughput when several pages are processed at once.

    .venv/bin/python tools/concurrency.py --image samples/crop.jpg --jobs 3 --mode sequential
    .venv/bin/python tools/concurrency.py --image samples/crop.jpg --jobs 3 --mode shared
    .venv/bin/python tools/concurrency.py --image samples/crop.jpg --jobs 3 --mode pool

Modes:
  sequential - one model, pages one after another (today's default)
  shared     - one model, N threads calling predict() on it concurrently
  pool       - N separate model instances, one per thread (memory heavy)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def rss_mb() -> int:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def peak_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def load_pipeline(mkldnn: bool):
    from paddleocr import PaddleOCRVL

    return PaddleOCRVL(pipeline_version="v1.6", enable_mkldnn=mkldnn)


def run_one(pipeline, image: Path, work: Path, chart: bool) -> int:
    """OCR one prepared image with an already-loaded pipeline; returns chars."""
    work.mkdir(parents=True, exist_ok=True)
    results = pipeline.predict(
        str(image),
        use_layout_detection=True,
        use_chart_recognition=chart,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        format_block_content=True,
    )
    for result in results:
        result.save_to_markdown(save_path=str(work))
    markdowns = list(work.glob("*.md"))
    return len(markdowns[0].read_text(encoding="utf-8")) if markdowns else 0


def prepare(source: Path, dest: Path, max_dim: int) -> Path:
    from PIL import Image, ImageOps

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        if max_dim and max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
            )
        image.save(dest, format="JPEG", quality=92)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--mode", choices=["sequential", "shared", "pool"], default="sequential")
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--chart", type=int, default=1)
    parser.add_argument("--mkldnn", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="threads per job (0 = let Paddle decide)")
    args = parser.parse_args()

    # Thread count must be set before Paddle is imported to take effect.
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    root = Path(tempfile.mkdtemp(prefix="p2md-conc-"))
    prepared = prepare(Path(args.image), root / "in.jpg", args.max_dim)

    load_start = time.time()
    if args.mode == "pool":
        pipelines = [load_pipeline(bool(args.mkldnn)) for _ in range(args.jobs)]
    else:
        pipelines = [load_pipeline(bool(args.mkldnn))]
    load_s = time.time() - load_start
    print(f"loaded {len(pipelines)} pipeline(s) in {load_s:.1f}s, rss={rss_mb()}MB", file=sys.stderr)

    chart = bool(args.chart)
    results: list[tuple[int, float, int]] = []
    lock = threading.Lock()

    def job(index: int) -> None:
        pipeline = pipelines[index % len(pipelines)]
        start = time.time()
        chars = run_one(pipeline, prepared, root / f"job{index}", chart)
        elapsed = time.time() - start
        with lock:
            results.append((index, elapsed, chars))
            print(f"  job {index} finished in {elapsed:.0f}s ({chars} chars), rss={rss_mb()}MB",
                  file=sys.stderr)

    wall_start = time.time()
    if args.mode == "sequential":
        for index in range(args.jobs):
            job(index)
    else:
        threads = [threading.Thread(target=job, args=(i,)) for i in range(args.jobs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    wall_s = time.time() - wall_start

    total_chars = sum(chars for _, _, chars in results)
    print(json.dumps({
        "mode": args.mode,
        "jobs": args.jobs,
        "jobs_completed": len(results),
        "max_dim": args.max_dim,
        "chart": args.chart,
        "threads_per_job": args.threads,
        "wall_s": round(wall_s, 1),
        "avg_job_s": round(sum(e for _, e, _ in results) / max(len(results), 1), 1),
        "total_chars": total_chars,
        "chars_per_sec": round(total_chars / max(wall_s, 1), 1),
        "rss_mb": rss_mb(),
        "peak_mb": peak_mb(),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
