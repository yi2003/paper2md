#!/usr/bin/env python
"""Benchmark one PaddleOCR-VL configuration on one image.

    .venv/bin/python tools/bench.py --image samples/crop.jpg --chart 0

Prints a single JSON line so a sweep can be collected and compared.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-dim", type=int, default=1536)
    parser.add_argument("--mkldnn", type=int, default=0)
    parser.add_argument("--chart", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    work = Path(tempfile.mkdtemp(prefix="p2md-bench-"))

    # Exactly the preparation the app does, so numbers are comparable.
    from PIL import Image, ImageOps

    with Image.open(args.image) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        if args.max_dim and max(width, height) > args.max_dim:
            scale = args.max_dim / max(width, height)
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
            )
        prepared = work / "in.jpg"
        image.save(prepared, format="JPEG", quality=92)
        size = image.size

    import paddle  # noqa: F401  (imported so the thread env vars apply)

    load_start = time.time()
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(pipeline_version="v1.6", enable_mkldnn=bool(args.mkldnn))
    load_s = time.time() - load_start
    print(f"[{args.label}] model loaded in {load_s:.1f}s, rss={rss_mb()}MB", file=sys.stderr)

    infer_start = time.time()
    try:
        results = pipeline.predict(
            str(prepared),
            use_layout_detection=True,
            use_chart_recognition=bool(args.chart),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            format_block_content=True,
        )
        for result in results:
            result.save_to_markdown(save_path=str(work))
        error = None
    except Exception as exc:  # noqa: BLE001 - reported in the JSON
        error = f"{type(exc).__name__}: {exc}"
    infer_s = time.time() - infer_start

    markdowns = list(work.glob("*.md"))
    chars = len(markdowns[0].read_text(encoding="utf-8")) if markdowns else 0

    print(
        json.dumps(
            {
                "label": args.label,
                "size": list(size),
                "max_dim": args.max_dim,
                "mkldnn": args.mkldnn,
                "chart": args.chart,
                "threads": args.threads,
                "load_s": round(load_s, 1),
                "infer_s": round(infer_s, 1),
                "chars": chars,
                "sec_per_1k_chars": round(infer_s / max(chars, 1) * 1000, 1),
                "rss_mb": rss_mb(),
                "peak_mb": peak_mb(),
                "error": error,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
