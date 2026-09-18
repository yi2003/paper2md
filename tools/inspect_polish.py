"""Inspect a task's markdown before/after the DeepSeek cleanup."""
import json
import re
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"
task_id = sys.argv[1]

QUESTION_RE = re.compile(r"^[ \t]*\d{1,2}[ \t]*[.、,．)]", re.MULTILINE)
FIG_RE = re.compile(r"<img\b|!\[[^\]]*\]\(", re.IGNORECASE)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=600) as response:
        return json.load(response)


def describe(label, markdown):
    print(f"  chars          : {len(markdown):,}")
    print(f"  figures        : {len(FIG_RE.findall(markdown))}")
    print(f"  Q-labels (附图) : {markdown.count('附图')}")
    print(f"  questions      : {len(QUESTION_RE.findall(markdown))}")
    print(f"  hosted URLs    : {len(re.findall(r'https?://', markdown))}")
    print(f"  base64 inlines : {markdown.count('base64,')}")
    print(f"  page markers   : {len(re.findall(r'<!--\\s*/?page', markdown))}")
    print(f"  LaTeX ($...$)  : {markdown.count('$')}")


data = get(f"/api/tasks/{task_id}/markdown")
markdown = data["markdown"]
print(f"=== {task_id} BEFORE ===")
describe("before", markdown)
print()
print("--- first 700 chars ---")
print(markdown[:700])
print()
print("=== calling DeepSeek cleanup ===")

request = urllib.request.Request(
    f"{BASE}/api/tasks/{task_id}/polish", method="POST", data=b"{}",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=1800) as response:
    result = json.load(response)

cleaned = result["markdown"]
info = result["info"]
print(f"  model    : {info['model']}")
print(f"  chunks   : {info['chunks']}")
print(f"  figures  : {info['figures_total']} total, {info['figures_recovered']} recovered")
print(f"  usage    : {info['usage']}")
print()
print(f"=== AFTER ===")
describe("after", cleaned)
print()
print("--- cleaned output (first 900 chars) ---")
print(cleaned[:900])
print()
print("--- figure check: every original URL still present? ---")
urls_before = set(re.findall(r"https?://[^\s\"')>]+", markdown))
urls_after = set(re.findall(r"https?://[^\s\"')>]+", cleaned))
missing = urls_before - urls_after
print(f"  URLs before: {len(urls_before)}, after: {len(urls_after)}, lost: {len(missing)}")
for url in sorted(missing):
    print("    LOST:", url)
