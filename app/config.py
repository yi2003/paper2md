"""Paper2MD configuration.

Everything is environment driven (read from `.env` at the project root, then
from the real environment). The defaults are tuned for a single-user local
install on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# `.env` at the project root; real environment variables always win.
load_dotenv(BASE_DIR / ".env")


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


# Pin the BLAS/OpenMP thread count before Paddle is ever imported (it is
# imported lazily, deep inside app.ocr). Parallelism comes from running several
# pages at once, not from giving each page every core.
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OCR_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("OCR_THREADS", "1"))


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------

HOST = _str("HOST", "0.0.0.0")
PORT = _int("PORT", 8000)
# Used when IMAGE_HOST=local so the generated markdown points somewhere the
# reader can actually reach. Set this to your machine's LAN IP if you want to
# share the markdown with someone on another device.
PUBLIC_BASE_URL = _str("PUBLIC_BASE_URL", f"http://localhost:{PORT}").rstrip("/")

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

DATA_DIR = Path(_str("DATA_DIR", str(BASE_DIR / "data"))).expanduser().resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "paper2md.db"

# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

# Longest edge of the image handed to the model. 0 disables resizing.
OCR_MAX_DIM = _int("OCR_MAX_DIM", 1536)
# Pages OCR'd at the same time. All of them share one copy of the model, so
# this costs almost no extra RAM — raise it until the CPU saturates. Measured
# on a 16-core CPU: 1 -> 12.3 chars/s, 3 -> 28.5, 6 -> 38.5.
OCR_WORKERS = max(1, _int("OCR_WORKERS", 4))
# Native threads each inference may use. Single-threaded is fastest here: the
# work is already parallelised across pages, and letting every page also spawn
# 16 threads just causes contention.
OCR_THREADS = max(1, _int("OCR_THREADS", 1))
# oneDNN acceleration. Enable if your Paddle build supports it; it is off by
# default because some PaddleOCR-VL builds crash during load with it on.
OCR_ENABLE_MKLDNN = _bool("OCR_ENABLE_MKLDNN", False)
# Retained for parity with the reference project; biggest lever on accuracy.
OCR_PIPELINE_VERSION = _str("OCR_PIPELINE_VERSION", "v1.6")

# When true the OCR job prints the full pipeline traceback to the server log.
OCR_DEBUG = _bool("OCR_DEBUG", False)

# --------------------------------------------------------------------------
# Which engine reads a page
# --------------------------------------------------------------------------
#
#   paddle   - PaddleOCR-VL end to end (local, free, offline; ~90-130 s a page)
#   deepseek - local layout detection for the figure boxes (~3.5 s) plus
#              DeepSeek vision for the text (~60 s). Better LaTeX, and the
#              handwriting goes in the same pass, but it needs the API.
#   auto     - try DeepSeek, fall back to Paddle if it fails.

OCR_ENGINE = _str("OCR_ENGINE", "paddle").lower()

# PP-DocLayoutV3 is the layout half of PaddleOCR-VL, used by the deepseek engine.
LAYOUT_MODEL_NAME = _str("LAYOUT_MODEL_NAME", "PP-DocLayoutV3")
LAYOUT_MIN_SCORE = float(_str("LAYOUT_MIN_SCORE", "0.5"))
# Ignore slivers: a "figure" smaller than this is almost certainly a stray mark.
LAYOUT_MIN_PIXELS = _int("LAYOUT_MIN_PIXELS", 32)
# Margin added around each cropped figure so nothing gets clipped.
LAYOUT_PADDING = _int("LAYOUT_PADDING", 6)

# --------------------------------------------------------------------------
# DeepSeek — turns raw OCR into a clean, question-only paper
# --------------------------------------------------------------------------

DEEPSEEK_API_KEY = _str("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _str("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
# Verified against GET /models on first use; if the account does not offer it we
# fall back to DEEPSEEK_FALLBACK_MODEL rather than failing.
DEEPSEEK_MODEL = _str("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_FALLBACK_MODEL = _str("DEEPSEEK_FALLBACK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = _int("DEEPSEEK_TIMEOUT", 600)
# Anything longer than this is cleaned page by page rather than in one call.
DEEPSEEK_MAX_CHARS = _int("DEEPSEEK_MAX_CHARS", 24000)


def deepseek_ready() -> bool:
    """True when an API key is configured."""
    return bool(DEEPSEEK_API_KEY)

# --------------------------------------------------------------------------
# Image hosting
# --------------------------------------------------------------------------

# imgbb  -> upload every extracted figure to imgbb.com and reference the URL
# local  -> reference the app's own /tasks/<id>/images/<file> URL (offline)
# base64 -> inline every figure as a data URI (fully self-contained file)
IMAGE_HOST = _str("IMAGE_HOST", "imgbb").lower()
IMGBB_API_KEY = _str("IMGBB_API_KEY", "")
# Seconds until ImgBB expires the image. 0 means "never".
IMGBB_EXPIRATION = _int("IMGBB_EXPIRATION", 0)

# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------

MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 25)
MAX_PAGES_PER_UPLOAD = _int("MAX_PAGES_PER_UPLOAD", 50)
# Anything not already JPEG is re-encoded as JPEG before it reaches the model.
JPEG_SUFFIXES = {".jpg", ".jpeg"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def ensure_dirs() -> None:
    """Create the runtime directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Paths inside a task's upload folder
# --------------------------------------------------------------------------
#
#   data/uploads/<task_id>/
#       pages/page_1.jpg     original upload (1-based name, 0-based index)
#       images/p1_xxx.jpg    figures extracted from page 1
#       work/page_1/         scratch space, deleted once OCR finishes


def task_dir(task_id: str) -> Path:
    return UPLOADS_DIR / task_id


def page_image_path(task_id: str, page_index: int) -> Path:
    return task_dir(task_id) / "pages" / f"page_{page_index + 1}.jpg"


def images_dir(task_id: str) -> Path:
    return task_dir(task_id) / "images"


def work_dir(task_id: str, page_index: int) -> Path:
    return task_dir(task_id) / "work" / f"page_{page_index + 1}"
