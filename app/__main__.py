"""Entry point: ``python -m app``."""

from __future__ import annotations

import uvicorn

from . import config


def main() -> None:
    print(f"Paper2MD starting on http://localhost:{config.PORT}")
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
