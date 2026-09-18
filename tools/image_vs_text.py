"""Experiment: does sending the page photo make the cleanup more accurate?

Runs the same page through the cleanup twice — text-only, then text+image — and
checks which one better removes the student's handwritten answers.

    .venv/bin/python tools/image_vs_text.py <task_id> <page_index>
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, deepseek, store  # noqa: E402

TASK_ID = sys.argv[1]
PAGE_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_DIM = 1400

VISION_SYSTEM = deepseek.SYSTEM_PROMPT + """
You are also given a PHOTOGRAPH of the original page.

Use the photo as the source of truth for one thing above all: telling PRINTED
text from HANDWRITTEN writing. Printed text is typeset and uniform; handwritten
answers, working and annotations are not. Anything handwritten must be removed,
even if it sits in the middle of a printed sentence, and even if it looks
plausible as part of the question.

Use the photo to double-check figure placement: confirm which question each
figure actually belongs to from where it sits on the page.
"""


def prepare_image(task_id: str, page_index: int) -> tuple[str, tuple[int, int]]:
    """Load the stored page photo, downscaled, as a JPEG data URI."""
    from PIL import Image, ImageOps

    path = config.page_image_path(task_id, page_index)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        if max(width, height) > MAX_DIM:
            scale = MAX_DIM / max(width, height)
            image = image.resize(
                (int(width * scale), int(height * scale)), Image.LANCZOS
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        size = image.size
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", size


def run(markdown: str, model: str, image_uri: str | None) -> tuple[str, dict, float]:
    """One cleanup pass, optionally with the page photo attached."""
    protected, mapping = deepseek.protect_figures(markdown)

    content: list[dict] = [{"type": "text", "text": protected}]
    if image_uri:
        content.append({"type": "image_url", "image_url": {"url": image_uri}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM if image_uri else deepseek.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 32000,
    }
    started = time.time()
    response = requests.post(
        f"{config.DEEPSEEK_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"},
        json=payload,
        timeout=1200,
    )
    elapsed = time.time() - started
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    body = response.json()
    answer = body["choices"][0]["message"].get("content") or ""
    restored, _ = deepseek.restore_figures(answer, mapping)
    return restored, body.get("usage", {}), elapsed


def measure(markdown: str) -> dict:
    """Objective counts, so the two passes can be compared without opinion."""
    import re

    return {
        "chars": len(markdown),
        "blanks": len(re.findall(r"\\underline\{\\hspace\{2em\}\}", markdown)),
        # A blank that still holds a value means handwriting was left behind.
        "filled_blanks": len(re.findall(r"\\underline\{(?!\\hspace)[^}]{1,30}\}", markdown)),
        "figures": len(re.findall(r"<img\s", markdown)) + len(re.findall(r"!\[", markdown)),
        "labels": len(re.findall(r"\*\[Q\d+ 附图\]\*", markdown)),
        "display_math": len(re.findall(r"\$\$", markdown)) // 2,
    }


def show_diff(a: str, b: str) -> None:
    """Show what the image pass changed, line by line."""
    import difflib

    left = [line for line in a.splitlines() if line.strip()]
    right = [line for line in b.splitlines() if line.strip()]
    print("=== lines that DIFFER (text-only vs text+image) ===")
    shown = 0
    for line in difflib.unified_diff(left, right, lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        marker = "  -" if line.startswith("-") else "  +"
        print(f"{marker} {line[1:].strip()[:150]}")
        shown += 1
        if shown > 30:
            print("  … (truncated)")
            break
    if shown == 0:
        print("  (identical)")


def main() -> int:
    store.init()
    task = store.rebuild_task(TASK_ID)
    if task is None:
        print(f"no such task: {TASK_ID}")
        return 1

    page = None
    for candidate in task["pages"]:
        if candidate["page_index"] == PAGE_INDEX:
            page = candidate
    if page is None:
        print(f"no page {PAGE_INDEX}")
        return 1

    from app.markdown_utils import page_markdown

    markdown = page_markdown(page["markdown"] or "", page["label_mapping"])
    print(f"task {TASK_ID} page {PAGE_INDEX + 1}: {len(markdown):,} chars")
    print()

    model = deepseek.resolve_model()
    image_uri, size = prepare_image(TASK_ID, PAGE_INDEX)
    print(f"page photo downscaled to {size[0]}x{size[1]}, "
          f"{len(image_uri) // 1024} KB as base64")
    print()

    print("running TEXT-ONLY pass …")
    text_out, text_usage, text_s = run(markdown, model, None)
    text_m = measure(text_out)
    print(f"  {text_s:.0f}s  prompt={text_usage.get('prompt_tokens')} "
          f"completion={text_usage.get('completion_tokens')}")
    print(f"  {text_m}")
    print()

    print("running TEXT+IMAGE pass …")
    image_out, image_usage, image_s = run(markdown, model, image_uri)
    image_m = measure(image_out)
    print(f"  {image_s:.0f}s  prompt={image_usage.get('prompt_tokens')} "
          f"completion={image_usage.get('completion_tokens')}")
    print(f"  {image_m}")
    print()

    print("=== comparison ===")
    print(f"  blanks created (higher = more handwriting removed): "
          f"{text_m['blanks']} -> {image_m['blanks']}")
    print(f"  blanks still holding a value (lower = better):      "
          f"{text_m['filled_blanks']} -> {image_m['filled_blanks']}")
    print(f"  figures kept:                                       "
          f"{text_m['figures']} -> {image_m['figures']}")
    print(f"  prompt tokens (cost):                               "
          f"{text_usage.get('prompt_tokens')} -> {image_usage.get('prompt_tokens')}")
    print(f"  seconds:                                            {text_s:.0f} -> {image_s:.0f}")
    print()

    show_diff(text_out, image_out)

    Path("/tmp/cleanup_text_only.md").write_text(text_out, encoding="utf-8")
    Path("/tmp/cleanup_with_image.md").write_text(image_out, encoding="utf-8")
    print("\n  wrote /tmp/cleanup_text_only.md and /tmp/cleanup_with_image.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
