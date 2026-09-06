"""Public STEP15 service API with a lazy orchestrator import."""

from __future__ import annotations


def run_evidence_arena(stock_code: str) -> dict:
    from .orchestrator import run_evidence_arena as _run

    return _run(stock_code)


__all__ = ["run_evidence_arena"]
