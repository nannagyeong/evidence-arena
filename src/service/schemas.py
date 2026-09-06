"""Stable STEP15 response contract helpers."""

from __future__ import annotations


REQUIRED_RESPONSE_KEYS = {
    "company",
    "stock_code",
    "analysis_as_of_date",
    "prototype_data_cutoff_notice",
    "bull",
    "bear",
    "dropped_or_revised_claims",
    "claim_evaluations",
    "fact_room_summary",
    "evidence_board",
    "debate",
    "citations",
    "ml_context",
    "warnings",
    "execution_metadata",
}


def validate_service_response(result: dict) -> None:
    missing = REQUIRED_RESPONSE_KEYS - set(result)
    if missing:
        raise ValueError("STEP15 response is missing: " + ", ".join(sorted(missing)))
    if result["stock_code"] != str(result["stock_code"]).zfill(6):
        raise ValueError("Response stock_code must be six digits")
    if result["execution_metadata"].get("max_revision_round") != 1:
        raise ValueError("STEP15 permits exactly one maximum revision round")
    if result["analysis_as_of_date"] != "2025-12-30":
        raise ValueError("STEP15 service snapshot must be 2025-12-30")
    if result["debate"]["shared_evidence_hash"] != result["execution_metadata"].get("shared_evidence_hash"):
        raise ValueError("Bull/Bear debate must share one immutable Evidence Packet")
