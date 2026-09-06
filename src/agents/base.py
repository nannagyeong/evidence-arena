"""Shared Bull/Bear generation and fallback policy."""

from __future__ import annotations

import copy

from src import closed_loop
from src.config.settings import Settings

from .llm_client import GoogleStructuredClaimClient, LLMUnavailable


def _validated_candidates(role: str, candidates: list[dict], fact_room: dict) -> list[dict]:
    valid: list[dict] = []
    prefix = "BULL" if role == "bull" else "BEAR"
    for candidate in candidates:
        normalized = copy.deepcopy(candidate)
        normalized["claim_id"] = f"{prefix}-{fact_room['stock_code']}-{len(valid) + 1:03d}"
        normalized["parent_claim_id"] = None
        normalized["source_agent"] = role
        normalized["stock_code"] = fact_room["stock_code"]
        normalized["as_of_date"] = fact_room["as_of_date"]
        try:
            closed_loop.validate_agent_claim(normalized, fact_room)
        except Exception:
            continue
        if closed_loop.claim_signature(normalized) in {closed_loop.claim_signature(item) for item in valid}:
            continue
        valid.append(normalized)
        if len(valid) == closed_loop.MAX_INITIAL_CLAIMS_PER_AGENT:
            break
    closed_loop.validate_claim_batch(valid, role, fact_room)
    return valid


def deterministic_claims(role: str, fact_room: dict) -> list[dict]:
    frozen_agent = closed_loop.StructuredClaimAgent(role)
    view = closed_loop.initial_agent_view(fact_room)
    return _validated_candidates(role, frozen_agent._candidate_pool(view), fact_room)


def generate_claims(
    role: str,
    fact_room: dict,
    settings: Settings,
    telemetry: dict,
    *,
    force_deterministic: bool = False,
) -> list[dict]:
    stage = f"{role}_initial"
    if force_deterministic or not settings.llm_configured:
        telemetry["fallback_used"] = True
        if not force_deterministic:
            telemetry["error_codes"].append("LLM_NOT_CONFIGURED")
        telemetry["llm_stage_status"][stage] = {
            "mode": "deterministic_fallback",
            "attempted": False,
            "error_code": "FORCED_DETERMINISTIC" if force_deterministic else "LLM_NOT_CONFIGURED",
        }
        return deterministic_claims(role, fact_room)

    client = GoogleStructuredClaimClient(settings, telemetry)
    try:
        blueprints = deterministic_claims(role, fact_room)
        candidates = client.generate_claims(
            role,
            closed_loop.initial_agent_view(fact_room),
            closed_loop.AGENT_CLAIM_SCHEMA,
            blueprints,
        )
        validated = _validated_candidates(role, candidates, fact_room)
        telemetry["llm_stage_status"][stage] = {
            "mode": "actual_llm",
            "attempted": True,
            "error_code": None,
        }
        return validated
    except (LLMUnavailable, ValueError) as error:
        telemetry["fallback_used"] = True
        detail = getattr(error, "error_code", type(error).__name__)
        telemetry["error_codes"].append(f"{role.upper()}_LLM_FALLBACK:{detail}")
        telemetry["llm_stage_status"][stage] = {
            "mode": "deterministic_fallback",
            "attempted": True,
            "error_code": detail,
        }
        return deterministic_claims(role, fact_room)
