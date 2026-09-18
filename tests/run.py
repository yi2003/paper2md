"""Minimal test runner so the suite runs before pytest is installed.

    python tests/run.py

Once pytest is available, ``.venv/bin/pytest tests -q`` works too.
"""

import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# Must happen before any test module imports `app.config`.
import _env  # noqa: E402

MODULES = ["test_markdown", "test_deepseek", "test_hybrid", "test_flow"]


def main() -> int:
    passed = 0
    failures: list[str] = []

    for module_name in MODULES:
        module = importlib.import_module(module_name)
        print(f"\n{module_name}")
        for attribute in sorted(dir(module)):
            if not attribute.startswith("test_"):
                continue
            function = getattr(module, attribute)
            if not callable(function):
                continue
            try:
                function()
            except Exception:  # noqa: BLE001 - report and keep going
                failures.append(f"{module_name}.{attribute}")
                print(f"  FAIL  {attribute}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"  ok    {attribute}")

    total = passed + len(failures)
    print(f"\n{passed}/{total} passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
