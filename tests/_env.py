"""Point the test suite at a throwaway data directory.

This module must be imported *before* anything under ``app`` — ``app.config``
reads its settings once, at import time. If a test module imports the app first,
``DATA_DIR`` is already fixed and the suite silently runs against the real
database.

Imported by tests/conftest.py (pytest), by tests/run.py, and by any test module
that needs the environment ready on its own. Idempotent.
"""

from __future__ import annotations

import os
import tempfile

_MARKER = "PAPER2MD_TEST_ENV"


def prepare() -> str:
    """Configure an isolated environment; returns the temp data directory."""
    if os.environ.get(_MARKER) == "1":
        return os.environ["DATA_DIR"]

    directory = tempfile.mkdtemp(prefix="paper2md-test-")
    os.environ["DATA_DIR"] = directory
    # Keep the suite offline and free of side effects.
    os.environ["IMAGE_HOST"] = "base64"
    os.environ["IMGBB_API_KEY"] = ""
    os.environ["DEEPSEEK_API_KEY"] = ""
    os.environ["OCR_WORKERS"] = "1"
    os.environ[_MARKER] = "1"
    return directory


TEST_DATA_DIR = prepare()
