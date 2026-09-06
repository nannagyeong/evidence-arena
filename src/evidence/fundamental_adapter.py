"""Convert the shared Fact Room into the STEP13 structured-fundamental input."""

from __future__ import annotations


def fact_room_fundamental_packet(fact_room: dict, claim: dict) -> dict:
    requested = claim.get("feature")
    signal = "operating_margin_change_yoy" if requested == "operating_margin" else requested
    snapshot = fact_room.get("fundamental_snapshot", {})
    payload = snapshot.get(signal, {})
    available = payload.get("value") is not None
    return {
        "fundamental_status": "success" if available else "unavailable",
        "stock_code": fact_room["stock_code"],
        "as_of_date": fact_room["as_of_date"],
        "features": {
            name: {"value": item.get("value"), "unit": item.get("unit")}
            for name, item in snapshot.items()
        },
        "financial_source": {
            "effective_date": payload.get("effective_date"),
            "rcept_no": payload.get("rcept_no"),
        },
        "pit_safe": bool(fact_room.get("pit_guards", {}).get("all_source_dates_lte_as_of")),
    }

