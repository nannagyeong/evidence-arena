"""PIT-safe Fact Room adapter around the Frozen STEP14 implementation."""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from src import closed_loop

from .date_resolver import resolve_analysis_as_of_date


def build_service_fact_room(stock_code: str) -> dict:
    as_of = resolve_analysis_as_of_date(stock_code)
    room = closed_loop.build_fact_room(stock_code, as_of)
    if room["as_of_date"] != as_of.isoformat():
        raise AssertionError("Fact Room date differs from the automatic resolver")
    _assert_fact_room_dates(room)
    return room


def _walk_dates(value: Any, parent_key: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"effective_date", "source_date", "data_date"} and child:
                yield parent_key + "." + key, child
            yield from _walk_dates(child, parent_key + "." + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_dates(child, f"{parent_key}[{index}]")


def _assert_fact_room_dates(room: dict) -> None:
    as_of = pd.Timestamp(room["as_of_date"])
    violations = [path for path, value in _walk_dates(room) if pd.Timestamp(value) > as_of]
    if violations:
        raise AssertionError("PIT violation in Fact Room: " + ", ".join(violations))


def fact_room_summary(room: dict) -> dict:
    fundamental = {
        name: copy.deepcopy(payload)
        for name, payload in room["fundamental_snapshot"].items()
        if payload.get("value") is not None
    }
    macro = {
        name: copy.deepcopy(payload)
        for name, payload in room["macro_context"].items()
        if payload.get("value") is not None
    }
    market = {
        name: copy.deepcopy(payload)
        for name, payload in room["market_context"].items()
        if payload.get("value") is not None
    }
    return {
        "fact_room_id": room["fact_room_id"],
        "fact_room_hash": room["fact_room_hash"],
        "company_name": room["company_name"],
        "market": room["market"],
        "industry": room["industry"],
        "latest_market_date": room["data_cutoff_trading_date"],
        "fundamental": fundamental,
        "market_context": market,
        "macro_context": macro,
        "recent_disclosures": copy.deepcopy(room["recent_disclosures"]),
        "pit_guards": copy.deepcopy(room["pit_guards"]),
    }

