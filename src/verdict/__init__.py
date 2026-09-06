"""Frozen STEP13 Verdict Engine export."""

from src import verdict_engine

synthesize_claim_verdict = verdict_engine.synthesize_claim_verdict

__all__ = ["verdict_engine", "synthesize_claim_verdict"]

