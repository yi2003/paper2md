"""Could DeepSeek replace PaddleOCR-VL entirely?

Sends one page photo to DeepSeek and asks it to do the whole job: read the
paper into markdown AND report where each figure is, so the figures can be
cropped locally. Compares the boxes it returns against the ones PaddleOCR-VL
actually found.

    .venv/bin/python tools/deepseek_ocr_probe.py <task_id> <page_index>
"""

import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, store  # noqa: E402
from app.markdown_utils import iter_image_refs  # noqa: E402

TASK_ID = sys.argv[1]
PAGE_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_DIM = 1536  # what PaddleOCR-VL sees

PROMPT = """\
This is a photographed exam paper page.

Do two things.

1. Transcribe the PRINTED text into GitHub-flavoured Markdown, with mathematics
   as LaTeX ($...$ or $$...$$). Remove anything handwritten by the student.
   Keep the question numbers.

2. List every FIGURE (diagram, chart, graph, geometric drawing, table image,
   picture). For each, give a bounding box as four integers 0-1000, scaled to
   the image width and height: {"x1": left, "y1": top, "x2": right, "y2": bottom}.
   Do NOT include plain printed text, and do NOT include handwriting.

Reply with JSON only, no code fences:

{"markdown": "...", "figures": [{"x1": 0, "y1": 0, "x2": 0, "y2": 0, "caption": "..."}]}
"""


def load_page_image() -> tuple[str, tuple[int, int]]:
    from PIL import Image, ImageOps

    with Image.open(config.page_image_path(TASK_ID, PAGE_INDEX)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        if max(width, height) > MAX_DIM:
            scale = MAX_DIM / max(width, height)
            image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        size = image.size
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), size


def paddle_boxes(markdown: str, size: tuple[int, int]) -> list[dict]:
    """PaddleOCR's boxes, normalised to 0-1000 to compare with DeepSeek's.

    Paddle reports them in the coordinate space of the image it processed, which
    is the same downscaled one we are sending.
    """
    width, height = size
    boxes = []
    for name, _ in iter_image_refs(markdown):
        match = re.search(r"box_(\d+)_(\d+)_(\d+)_(\d+)", name)
        if not match:
            continue
        y1, x1, y2, x2 = (int(v) for v in match.groups())
        boxes.append({
            "name": name,
            "x1": round(x1 / width * 1000),
            "y1": round(y1 / height * 1000),
            "x2": round(x2 / width * 1000),
            "y2": round(y2 / height * 1000),
        })
    return boxes


def overlaps(a: dict, b: dict) -> bool:
    """Do two normalised boxes overlap at all?"""
    return not (a["x2"] < b["x1"] or b["x2"] < a["x1"]
                or a["y2"] < b["y1"] or b["y2"] < a["y1"])


def main() -> int:
    store.init()
    task = store.rebuild_task(TASK_ID)
    if task is None:
        print(f"no such task {TASK_ID}")
        return 1
    page = next((p for p in task["pages"] if p["page_index"] == PAGE_INDEX), None)
    if page is None:
        print("no such page")
        return 1

    image_uri, size = load_page_image()
    truth = paddle_boxes(page["markdown"] or "", size)
    print(f"page {PAGE_INDEX + 1}, image sent at {size[0]}x{size[1]}")
    print(f"PaddleOCR-VL found {len(truth)} figure(s):")
    for b in truth:
        print(f"  x{b['x1']:>4}..{b['x2']:<4} y{b['y1']:>4}..{b['y2']:<4}")
    print()

    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_uri}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 32000,
    }

    print("asking DeepSeek to do the whole page …")
    started = time.time()
    response = requests.post(
        f"{config.DEEPSEEK_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"},
        json=payload, timeout=1800,
    )
    elapsed = time.time() - started
    if not response.ok:
        print(f"  HTTP {response.status_code}: {response.text[:300]}")
        return 1

    body = response.json()
    choice = body["choices"][0]
    text = choice["message"].get("content") or ""
    usage = body.get("usage", {})
    print(f"  {elapsed:.0f}s  finish={choice.get('finish_reason')}  "
          f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
    print()

    # Parse the JSON reply (tolerate stray prose around it).
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        print("no JSON in the reply. First 500 chars:")
        print(text[:500])
        return 1
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        print(f"reply is not valid JSON: {exc}")
        print(text[:500])
        return 1

    markdown = parsed.get("markdown") or ""
    figures = parsed.get("figures") or []
    print(f"DeepSeek returned {len(markdown):,} chars of markdown and {len(figures)} figure(s):")
    for f in figures:
        print(f"  x{f.get('x1'):>4}..{f.get('x2'):<4} y{f.get('y1'):>4}..{f.get('y2'):<4}  "
              f"{str(f.get('caption'))[:40]}")
    print()

    matched = sum(1 for t in truth if any(overlaps(t, f) for f in figures))
    phantom = sum(1 for f in figures if not any(overlaps(t, f) for t in truth))
    print("=== bounding-box accuracy ===")
    print(f"  Paddle boxes also found by DeepSeek : {matched}/{len(truth)}")
    print(f"  DeepSeek boxes overlapping nothing  : {phantom}/{len(figures)}")

    Path("/tmp/deepseek_page.md").write_text(markdown, encoding="utf-8")
    Path("/tmp/paddle_page.md").write_text(page["markdown"] or "", encoding="utf-8")
    print("\n  wrote /tmp/deepseek_page.md and /tmp/paddle_page.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
