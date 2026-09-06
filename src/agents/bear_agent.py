"""Bear Agent entry point."""

from __future__ import annotations

from src.config.settings import Settings

from .base import generate_claims


def generate_bear_claims(fact_room: dict, settings: Settings, telemetry: dict) -> list[dict]:
    return generate_claims("bear", fact_room, settings, telemetry)
