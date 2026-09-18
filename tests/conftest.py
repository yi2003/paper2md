"""pytest configuration.

Loaded by pytest before any test module, which is what makes the isolation in
``tests/_env.py`` take effect in time.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import _env  # noqa: F401,E402  (imported for its side effect)
