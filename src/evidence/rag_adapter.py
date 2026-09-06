"""Thin STEP15 wrapper over the Frozen STEP11 retrieval function."""

from __future__ import annotations

from src import evidence_adapters


def retrieve_claim_evidence(claim: dict) -> dict:
    return evidence_adapters.retrieve_production_disclosure_evidence(
        {
            "stock_code": claim["stock_code"],
            "as_of_date": claim["as_of_date"],
            "query": claim["atomic_claim_text"],
            "evidence_type": "qualitative_disclosure",
            "report_types": [],
            "top_k": 5,
        }
    )


def frozen_policy_metadata() -> dict:
    return {
        "policy_hash": evidence_adapters.FROZEN_POLICY_HASH,
        "weights": dict(evidence_adapters.HYBRID_SCORE_WEIGHTS),
        "relevance_floor": evidence_adapters.INITIAL_VECTOR_RELEVANCE_FLOOR,
    }

