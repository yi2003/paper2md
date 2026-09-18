"""Start a DeepSeek cleanup and watch its progress the way the UI does.

    .venv/bin/python tools/watch_polish.py <task_id>

Proves two things: the POST returns immediately instead of blocking, and
progress is observable while the job runs.
"""

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
task_id = sys.argv[1]


def post(path):
    request = urllib.request.Request(
        BASE + path, method="POST", data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    return body, time.time() - started


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return json.load(response)


print(f"task {task_id}")
print("POST /polish …")
body, elapsed = post(f"/api/tasks/{task_id}/polish")
print(f"  -> HTTP response in {elapsed:.2f}s  status={body.get('status')}")
if elapsed > 5:
    print("  !! the request blocked; progress cannot be shown")
print()

started = time.time()
last = None
while time.time() - started < 900:
    state = get(f"/api/tasks/{task_id}/polish")
    polish = state.get("polish", {})
    snapshot = (state["status"], polish.get("stage"), polish.get("done"), polish.get("total"))
    if snapshot != last:
        bars = ""
        total = polish.get("total") or 0
        if total > 1:
            filled = int(round(10 * (polish.get("done") or 0) / total))
            bars = "  [" + "#" * filled + "." * (10 - filled) + "]"
        print(
            f"  t+{time.time() - started:5.1f}s  status={state['status']:<8} "
            f"stage={str(polish.get('stage')):<9} "
            f"{polish.get('done')}/{polish.get('total')}{bars}"
        )
        last = snapshot

    if state["status"] in ("done", "failed"):
        if state["status"] == "failed":
            print(f"\n  FAILED: {polish.get('error')}")
            sys.exit(1)
        info = state.get("info") or {}
        print(f"\n  finished in {time.time() - started:.1f}s")
        print(f"  model   : {info.get('model')}")
        print(f"  chunks  : {info.get('chunks')}")
        print(f"  figures : {info.get('figures_total')}")
        print(f"  usage   : {info.get('usage')}")
        print(f"  output  : {len(state.get('markdown') or ''):,} chars")
        break
    time.sleep(0.4)
else:
    print("\n  timed out waiting for the cleanup")
    sys.exit(1)
