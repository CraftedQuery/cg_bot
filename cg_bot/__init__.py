"""cg_bot package shim.

The codebase modules are currently stored at the repository root. This package
provides import compatibility (e.g. `import cg_bot.main`) without requiring a
physical move of all files.

Use `cg_bot.get_app()` to access the FastAPI application.
"""

from __future__ import annotations

__all__ = ["get_app", "__version__"]

__version__ = "7.0"


def get_app():
    from .main import app

    return app
