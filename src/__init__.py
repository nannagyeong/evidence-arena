"""Expose the project's reusable modules through the local ``src`` package."""

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in __path__:
    __path__.append(str(_PROJECT_ROOT))
