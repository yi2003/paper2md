"""Hosting the extracted figures.

Three modes, chosen with ``IMAGE_HOST`` in ``.env``:

``imgbb``   upload every figure to imgbb.com and reference the returned URL.
            URLs are cached per task so a download never uploads twice.
            Any figure that fails to upload falls back to an inline data URI.
``local``   reference this app's own image endpoint, e.g.
            ``http://localhost:8000/tasks/<id>/images/p1_fig.jpg``. Nothing
            leaves the machine, but the link only works while the app runs.
``base64``  inline every figure into the markdown as a data URI, producing a
            single fully self-contained file.
"""

from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path

import requests

from . import config, store
from .markdown_utils import find_all_image_refs

IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"


class ImageHostError(RuntimeError):
    """Raised when a figure could not be uploaded."""


def log(message: str) -> None:
    print(f"[images] {message}", file=sys.stderr, flush=True)


def _mime_for(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime if mime and mime.startswith("image/") else "image/jpeg"


def upload_to_imgbb(path: Path, api_key: str | None = None) -> str:
    """Upload one image to ImgBB and return its direct URL."""
    key = api_key or config.IMGBB_API_KEY
    if not key:
        raise ImageHostError("IMGBB_API_KEY is not set")

    payload: dict[str, str] = {
        "key": key,
        "image": base64.b64encode(path.read_bytes()).decode("ascii"),
    }
    if config.IMGBB_EXPIRATION > 0:
        payload["expiration"] = str(config.IMGBB_EXPIRATION)

    response = requests.post(IMGBB_UPLOAD_URL, data=payload, timeout=90)
    response.raise_for_status()
    body = response.json()
    if not body.get("success"):
        error = body.get("error", {})
        raise ImageHostError(error.get("message") or str(body)[:200])
    return body["data"]["url"]


def _data_uri(path: Path) -> str:
    return f"data:{_mime_for(path.name)};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _replace(md: str, ref: str, replacement: str) -> str:
    return md.replace(ref, replacement)


def inline_base64(md: str, task_id: str) -> str:
    """Replace every local figure reference with an inline data URI."""
    images_dir = config.images_dir(task_id)
    for ref in find_all_image_refs(md):
        path = images_dir / Path(ref).name
        if not path.exists():
            log(f"missing figure, left as-is: {ref}")
            continue
        md = _replace(md, ref, _data_uri(path))
    return md


def local_urls(md: str, task_id: str) -> str:
    """Replace every local figure reference with this app's own URL."""
    for ref in find_all_image_refs(md):
        url = f"{config.PUBLIC_BASE_URL}/tasks/{task_id}/images/{Path(ref).name}"
        md = _replace(md, ref, url)
    return md


def host_markdown(md: str, task_id: str) -> tuple[str, dict[str, str]]:
    """Rewrite figure references for the configured host.

    Returns the rewritten markdown and the ``{filename: url}`` map that was used
    (empty for ``local`` and ``base64`` modes).
    """
    if not md:
        return md, {}

    mode = config.IMAGE_HOST

    if mode == "base64":
        return inline_base64(md, task_id), {}

    if mode == "local":
        return local_urls(md, task_id), {}

    if mode != "imgbb":
        log(f"unknown IMAGE_HOST={mode!r}, falling back to imgbb")

    if not config.IMGBB_API_KEY:
        log("IMGBB_API_KEY is not set — inlining figures as base64 instead")
        return inline_base64(md, task_id), {}

    images_dir = config.images_dir(task_id)
    cached = store.get_image_urls(task_id)
    used: dict[str, str] = {}

    for ref in find_all_image_refs(md):
        name = Path(ref).name

        url = cached.get(name)
        if url:
            md = _replace(md, ref, url)
            used[name] = url
            continue

        path = images_dir / name
        if not path.exists():
            log(f"missing figure, left as-is: {ref}")
            continue

        try:
            url = upload_to_imgbb(path)
        except (ImageHostError, requests.RequestException, ValueError) as exc:
            log(f"imgbb upload failed for {name} ({exc}); inlining it instead")
            md = _replace(md, ref, _data_uri(path))
            continue

        store.save_image_url(task_id, name, url)
        used[name] = url
        md = _replace(md, ref, url)
        log(f"uploaded {name} -> {url}")

    return md, used
