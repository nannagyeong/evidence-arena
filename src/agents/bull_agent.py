"""Bull Agent entry point."""

from __future__ import annotations

from src.config.settings import Settings

from .base import generate_claims


def generate_bull_claims(fact_room: dict, settings: Settings, telemetry: dict) -> list[dict]:
    return generate_claims("bull", fact_room, settings, telemetry)
