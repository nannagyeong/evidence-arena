"""Bull/Bear structured-claim agents with deterministic fallback."""

from .bull_agent import generate_bull_claims
from .bear_agent import generate_bear_claims

__all__ = ["generate_bull_claims", "generate_bear_claims"]

