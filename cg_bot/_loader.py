"""Internal helper to execute the existing repo-root modules under the cg_bot package.

This repository's Python modules currently live at the repo root (e.g. `main.py`,
`routers/chat_routes.py`). The test suite and CLI expect an importable `cg_bot`
package. These helpers allow us to keep the existing layout while making
`import cg_bot.*` work reliably.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exec_from_root(module_globals: dict[str, Any], rel_path: str) -> None:
    """Execute a repo-root file into the current module namespace."""

    path = repo_root() / rel_path
    code = path.read_text(encoding="utf-8")

    # Ensure relative imports resolve as if this module were the real one.
    module_globals.setdefault("__file__", str(path))

    exec(compile(code, str(path), "exec"), module_globals)
