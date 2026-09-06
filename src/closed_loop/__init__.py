"""Expose the root Frozen STEP14 module without duplicating its code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_NAME = "src._frozen_closed_loop"
_MODULE_PATH = Path(__file__).resolve().parents[2] / "closed_loop.py"
if _MODULE_NAME in sys.modules:
    frozen_closed_loop = sys.modules[_MODULE_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Unable to load Frozen STEP14 module: {_MODULE_PATH}")
    frozen_closed_loop = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = frozen_closed_loop
    _spec.loader.exec_module(frozen_closed_loop)

for _name in dir(frozen_closed_loop):
    if not _name.startswith("_"):
        globals()[_name] = getattr(frozen_closed_loop, _name)

__all__ = [name for name in dir(frozen_closed_loop) if not name.startswith("_")]
