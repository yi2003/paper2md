"""Compare a task's raw and DeepSeek-cleaned markdown.

Checks that the cleanup did not drop questions or figures, and lists the
question numbers so gaps are obvious.

    .venv/bin/python tools/compare_polish.py <task_id>
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
QUESTION_RE = re.compile(r"^[ \t]*(\d{1,2})[ \t]*[.、,．)]", re.MULTILINE)
FIG_RE = re.compile(r"<img\s[^>]*?src=[\"']([^\"']+)[\"']|!\[[^\]]*\]\(([^)]+)\)", re.IGNORECASE)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=1800) as response:
        return json.load(response)


def questions(markdown):
    seen, ordered = set(), []
    for match in QUESTION_RE.finditer(markdown):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            ordered.append(number)
    return ordered


def figures(markdown):
    names = []
    for match in FIG_RE.finditer(markdown):
        ref = match.group(1) or match.group(2) or ""
        names.append(Path(ref).name or ref)
    return names


task_id = sys.argv[1]
raw = get(f"/api/tasks/{task_id}/markdown")["markdown"]
cleaned = get(f"/api/tasks/{task_id}/markdown?variant=polished")["markdown"]

raw_q, clean_q = questions(raw), questions(cleaned)
raw_f, clean_f = figures(raw), figures(cleaned)

print(f"task {task_id}")
print()
print(f"  raw questions      : {len(raw_q)}  {sorted(raw_q)}")
print(f"  cleaned questions  : {len(clean_q)}  {sorted(clean_q)}")
lost_q = sorted(set(raw_q) - set(clean_q))
print(f"  QUESTIONS LOST     : {lost_q if lost_q else 'none ✓'}")
print()
print(f"  raw figures        : {len(raw_f)}")
print(f"  cleaned figures    : {len(clean_f)}")
# Names legitimately change: hosting renames each figure. Only the count matters,
# and the cleaned run should have at least as many as the raw one.
if len(clean_f) >= len(raw_f):
    print("  FIGURES LOST       : none ✓")
else:
    print(f"  FIGURES LOST       : {len(raw_f) - len(clean_f)} (raw {len(raw_f)} -> cleaned {len(clean_f)})")
print()
print(f"  raw chars          : {len(raw):,}")
print(f"  cleaned chars      : {len(cleaned):,}  ({100 - round(len(cleaned) / max(len(raw), 1) * 100)}% shorter)")

out = Path("/tmp") / f"{task_id}_cleaned.md"
out.write_text(cleaned, encoding="utf-8")
print(f"\n  cleaned markdown written to: {out}")
