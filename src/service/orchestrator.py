"""Evidence Arena STEP15 orchestration over Frozen STEP8~14 engines."""

from __future__ import annotations

import copy
import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src import claim_router, closed_loop, evidence_adapters, verdict_engine
from src.agents.base import _validated_candidates, deterministic_claims
from src.agents.bear_agent import generate_bear_claims
from src.agents.bull_agent import generate_bull_claims
from src.agents.debate import run_post_validation_debate, seal_evidence_packet
from src.config.paths import PATHS
from src.config.settings import Settings, get_settings
from src.evidence.fundamental_adapter import fact_room_fundamental_packet
from src.evidence.ml_adapter import infer_latest_ml_context, to_frozen_ml_row

from .fact_room import build_service_fact_room, fact_room_summary
from .safety import evaluate_safety_gates
from .schemas import validate_service_response


ProgressCallback = Callable[[str], None]


def _progress(callback: ProgressCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _service_route_claim(claim: dict, batch_id: str, settings: Settings) -> dict:
    """Run STEP12, separating its Frozen development cutoff from service time."""
    routed = claim_router.structured_route_claim(copy.deepcopy(claim), batch_id)
    if routed["validated_claim"]["validation"]["execution_allowed"]:
        routed["service_date_policy"] = {
            "analysis_as_of_date": claim["as_of_date"],
            "development_cutoff_applied": False,
        }
        return routed

    service_date = claim["as_of_date"]
    if service_date <= settings.development_cutoff_date.isoformat():
        return routed

    validation_claim = copy.deepcopy(claim)
    validation_claim["as_of_date"] = settings.development_cutoff_date.isoformat()
    bridged = claim_router.structured_route_claim(validation_claim, batch_id)
    if not bridged["validated_claim"]["validation"]["execution_allowed"]:
        return routed

    bridged["validated_claim"]["as_of_date"] = service_date
    bridged["router_hash"] = claim_router.sha256_text(bridged["validated_claim"])
    bridged["service_date_policy"] = {
        "analysis_as_of_date": service_date,
        "development_cutoff_date": settings.development_cutoff_date.isoformat(),
        "development_cutoff_applied": bridged.get("step8_claim") is not None,
        "policy": "STEP12 registry validation and Frozen STEP8 execution use the development window; service evidence keeps its real source date.",
    }
    return bridged


def _route_agent_claims(
    role: str,
    claims: list[dict],
    fact_room: dict,
    settings: Settings,
    telemetry: dict,
) -> list[dict]:
    routed: list[dict] = []
    for claim in claims:
        item = _service_route_claim(claim, "STEP15-ROUND0", settings)
        if item["validated_claim"]["validation"]["execution_allowed"]:
            routed.append(item)
        else:
            telemetry["step12_invalid_count"] += 1
            telemetry["error_codes"].append(f"{role.upper()}_STEP12_INVALID")
    if routed:
        return routed

    telemetry["fallback_used"] = True
    telemetry["error_codes"].append(f"{role.upper()}_STEP12_FALLBACK")
    fallback = deterministic_claims(role, fact_room)
    for claim in fallback:
        item = _service_route_claim(claim, "STEP15-ROUND0-FALLBACK", settings)
        if item["validated_claim"]["validation"]["execution_allowed"]:
            routed.append(item)
        else:
            telemetry["step12_invalid_count"] += 1
    return routed


def _collect_and_synthesize(
    routed: dict,
    fact_room: dict,
    ml_context: dict,
    telemetry: dict,
) -> dict:
    claim = routed["validated_claim"]
    engines = {route["engine"] for route in claim["routes"] if route["execution_allowed"]}
    started = time.perf_counter()
    raw_by_engine: dict = {}
    try:
        raw_by_engine = evidence_adapters.execute_evidence_routes([routed]).get(claim["claim_id"], {})
    except Exception as error:
        telemetry["error_codes"].append(f"EVIDENCE_PARTIAL:{claim['claim_id']}:{type(error).__name__}")

    if claim["template_id"] == "T5":
        raw_by_engine["T5"] = fact_room_fundamental_packet(fact_room, claim)
    if "ML" in engines:
        ml_row = to_frozen_ml_row(ml_context)
        if ml_row is not None:
            raw_by_engine["ML"] = ml_row

    elapsed = time.perf_counter() - started
    if "RAG" in engines:
        telemetry["rag_latency_sec"] += elapsed
    if engines & {"T1", "T2", "T3", "T4", "T5", "T6"}:
        telemetry["statistics_latency_sec"] += elapsed

    try:
        request = verdict_engine.collect_evidence_for_claim(claim, routed["claim_scope"], raw_by_engine)
        return verdict_engine.synthesize_claim_verdict(request)
    except Exception as error:
        telemetry["error_codes"].append(f"VERDICT_FAILURE:{claim['claim_id']}:{type(error).__name__}")
        unavailable = [
            verdict_engine.make_unavailable_evidence(claim, route["engine"], "service_partial_failure")
            for route in claim["routes"]
        ]
        request = {
            "request_id": "SYN-" + claim["claim_id"],
            "validated_claim": claim,
            "claim_scope": routed["claim_scope"],
            "evidence_packets": unavailable,
        }
        return verdict_engine.synthesize_claim_verdict(request)


def _extract_citations(board: list[dict], changed_claims: list[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    packet_groups = [
        item["primary_evidence"] + item["auxiliary_context"]
        for item in board
    ] + [item.get("evidence_packets", []) for item in changed_claims]
    for packets in packet_groups:
        for packet in packets:
            for citation in packet.get("citations", []):
                key = (citation.get("rcept_no"), citation.get("chunk_id"), citation.get("source_url"))
                unique[key] = copy.deepcopy(citation)
    return list(unique.values())


def _build_claim_evaluations(
    all_routed: list[dict],
    round0_verdicts: dict[str, dict],
    revisions: list[dict],
    round1_results: dict[str, tuple[dict, dict]],
) -> list[dict]:
    revision_map = {item["claim_id"]: item for item in revisions}
    evaluations = []
    for routed in all_routed:
        claim = routed["validated_claim"]
        verdict = round0_verdicts[claim["claim_id"]]
        revision = revision_map[claim["claim_id"]]
        final_claim = claim if revision["revision_action"] == "KEEP" else None
        final_verdict = verdict if final_claim is not None else None
        final_packets: list[dict] = []
        if claim["claim_id"] in round1_results:
            revised_route, revised_verdict = round1_results[claim["claim_id"]]
            final_claim = revised_route["validated_claim"]
            final_verdict = revised_verdict
            final_packets = copy.deepcopy(revised_verdict.get("evidence_packets", []))
        final_retained = bool(
            final_claim is not None
            and final_verdict is not None
            and final_verdict["claim_verdict"] in closed_loop.FINAL_ACCEPTED_VERDICTS
        )
        evaluations.append(
            {
                "claim_id": claim["claim_id"],
                "source_agent": claim["source_agent"],
                "claim": copy.deepcopy(claim),
                "initial_verdict": verdict["claim_verdict"],
                "verdict_reason_code": verdict.get("verdict_reason_code"),
                "revision_action": revision["revision_action"],
                "revision_reason": revision["revision_reason"],
                "final_retained": final_retained,
                "final_claim": copy.deepcopy(final_claim),
                "final_verdict": final_verdict["claim_verdict"] if final_verdict else None,
                "evidence_packets": copy.deepcopy(verdict.get("evidence_packets", [])),
                "final_evidence_packets": final_packets,
            }
        )
    return evaluations


def _shared_evidence_packet(room: dict, evaluations: list[dict]) -> dict:
    evidence_by_id: dict[str, dict] = {}
    claims = []
    for item in evaluations:
        initial_packets = item["evidence_packets"]
        final_packets = item["final_evidence_packets"]
        for packet in initial_packets + final_packets:
            evidence_by_id[packet["evidence_id"]] = copy.deepcopy(packet)
        claims.append(
            {
                "claim_id": item["claim_id"],
                "source_agent": item["source_agent"],
                "claim_text": item["claim"]["atomic_claim_text"],
                "template_id": item["claim"]["template_id"],
                "feature": item["claim"].get("feature"),
                "deterministic_initial_verdict": item["initial_verdict"],
                "revision_action": item["revision_action"],
                "final_retained": item["final_retained"],
                "final_claim_id": item["final_claim"].get("claim_id") if item.get("final_claim") else None,
                "final_claim_text": item["final_claim"].get("atomic_claim_text") if item.get("final_claim") else None,
                "deterministic_final_verdict": item["final_verdict"],
                "evidence_ids": [packet["evidence_id"] for packet in initial_packets],
                "final_evidence_ids": [packet["evidence_id"] for packet in final_packets],
            }
        )
    return seal_evidence_packet(
        {
            "company": room["company_name"],
            "stock_code": room["stock_code"],
            "analysis_as_of_date": room["as_of_date"],
            "claims": claims,
            "evidence": sorted(evidence_by_id.values(), key=lambda item: item["evidence_id"]),
            "policy": {
                "verdict_authority": "Frozen STEP13 deterministic policy only",
                "maximum_revision_round": 1,
                "unknown_evidence_ids_allowed": False,
                "future_information_allowed": False,
                "investment_recommendation_allowed": False,
            },
        }
    )


def _append_request_log(metadata: dict) -> None:
    log_dir = PATHS.artifact_root / "step15"
    log_dir.mkdir(parents=True, exist_ok=True)
    fields = {
        key: metadata.get(key)
        for key in (
            "request_id", "timestamp", "stock_code", "analysis_as_of_date",
            "bull_claim_count", "bear_claim_count", "step12_invalid_count",
            "verdict_counts", "llm_calls", "fallback_used", "fact_room_latency_sec",
            "llm_latency_sec", "rag_latency_sec", "statistics_latency_sec",
            "total_latency_sec", "error_codes", "llm_stage_status",
            "shared_evidence_hash",
        )
    }
    with (log_dir / "request_logs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fields, ensure_ascii=False, separators=(",", ":")) + "\n")


def _run_evidence_arena(
    stock_code: str,
    *,
    settings: Settings | None = None,
    claim_overrides: dict[str, list[dict]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    started_total = time.perf_counter()
    settings = settings or get_settings()
    telemetry = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_calls": 0,
        "llm_latency_sec": 0.0,
        "rag_latency_sec": 0.0,
        "statistics_latency_sec": 0.0,
        "step12_invalid_count": 0,
        "fallback_used": False,
        "error_codes": [],
        "llm_stage_status": {},
    }

    _progress(progress_callback, "Fact Room 구성 중")
    fact_started = time.perf_counter()
    room = build_service_fact_room(stock_code)
    telemetry["fact_room_latency_sec"] = time.perf_counter() - fact_started
    ml_context = infer_latest_ml_context(room["stock_code"], room["as_of_date"])
    if not ml_context.get("available"):
        telemetry["error_codes"].append(str(ml_context.get("error_code", "ML_UNAVAILABLE")))

    _progress(progress_callback, "Bull/Bear 주장 생성 중")
    initial_by_role: dict[str, list[dict]] = {}
    route_by_role: dict[str, list[dict]] = {}
    generators = {"bull": generate_bull_claims, "bear": generate_bear_claims}
    for role in ("bull", "bear"):
        try:
            if claim_overrides and role in claim_overrides:
                claims = _validated_candidates(role, claim_overrides[role], room)
                telemetry["llm_stage_status"][f"{role}_initial"] = {
                    "mode": "claim_override",
                    "attempted": False,
                    "error_code": None,
                }
            else:
                claims = generators[role](room, settings, telemetry)
            routed = _route_agent_claims(role, claims, room, settings, telemetry)
            initial_by_role[role] = [copy.deepcopy(item["validated_claim"]) for item in routed]
            route_by_role[role] = routed
        except Exception as error:
            telemetry["fallback_used"] = True
            telemetry["error_codes"].append(f"{role.upper()}_AGENT_FAILURE:{type(error).__name__}")
            try:
                claims = deterministic_claims(role, room)
                routed = _route_agent_claims(role, claims, room, settings, telemetry)
                initial_by_role[role] = [copy.deepcopy(item["validated_claim"]) for item in routed]
                route_by_role[role] = routed
            except Exception as fallback_error:
                telemetry["error_codes"].append(f"{role.upper()}_UNAVAILABLE:{type(fallback_error).__name__}")
                initial_by_role[role] = []
                route_by_role[role] = []
    _progress(progress_callback, "근거 검증 중")
    all_routed = route_by_role["bull"] + route_by_role["bear"]
    round0_verdicts = {
        item["validated_claim"]["claim_id"]: _collect_and_synthesize(item, room, ml_context, telemetry)
        for item in all_routed
    }

    revisions: list[dict] = []
    round1_results: dict[str, tuple[dict, dict]] = {}
    for item in all_routed:
        claim = item["validated_claim"]
        verdict = round0_verdicts[claim["claim_id"]]
        revision = closed_loop.build_revision_record(claim, verdict, room)
        revisions.append(revision)
        if not revision["requires_revalidation"]:
            continue
        try:
            revised_route = _service_route_claim(revision["revised_claim"], "STEP15-ROUND1", settings)
            if not revised_route["validated_claim"]["validation"]["execution_allowed"]:
                telemetry["step12_invalid_count"] += 1
                continue
            revised_verdict = _collect_and_synthesize(revised_route, room, ml_context, telemetry)
            round1_results[revision["claim_id"]] = (revised_route, revised_verdict)
        except Exception as error:
            telemetry["error_codes"].append(f"ROUND1_FAILURE:{claim['claim_id']}:{type(error).__name__}")

    _progress(progress_callback, "Evidence Board 구성 중")
    final_by_role: dict[str, list[dict]] = {"bull": [], "bear": []}
    dropped_or_revised: list[dict] = []
    for item in all_routed:
        claim = item["validated_claim"]
        verdict = round0_verdicts[claim["claim_id"]]
        revision = next(record for record in revisions if record["claim_id"] == claim["claim_id"])
        final_item = None
        if revision["revision_action"] == "KEEP":
            final_item = {"claim": claim, "verdict": verdict, "final_status": "kept"}
        elif revision["claim_id"] in round1_results:
            revised_route, revised_verdict = round1_results[revision["claim_id"]]
            if revised_verdict["claim_verdict"] in closed_loop.FINAL_ACCEPTED_VERDICTS:
                final_item = {
                    "claim": revised_route["validated_claim"],
                    "verdict": revised_verdict,
                    "final_status": "kept_revised",
                }
        if final_item is not None:
            final_by_role[claim["source_agent"]].append(final_item)
        if revision["revision_action"] != "KEEP":
            dropped_or_revised.append(
                {
                    "claim_id": claim["claim_id"],
                    "source_agent": claim["source_agent"],
                    "initial_claim": claim,
                    "initial_verdict": verdict["claim_verdict"],
                    "revision_action": revision["revision_action"],
                    "revision_reason": revision["revision_reason"],
                    "revised_claim": revision.get("revised_claim"),
                    "final_retained": final_item is not None,
                    "evidence_packets": copy.deepcopy(verdict.get("evidence_packets", [])),
                    "routing_errors": item["validated_claim"]["validation"].get("errors", []),
                    "executed": True,
                }
            )

    board = []
    for role in ("bull", "bear"):
        for item in final_by_role[role]:
            packets = item["verdict"]["evidence_packets"]
            board.append(
                {
                    "agent": role,
                    "claim_id": item["claim"]["claim_id"],
                    "claim_text": item["claim"]["atomic_claim_text"],
                    "claim_verdict": item["verdict"]["claim_verdict"],
                    "primary_evidence": [p for p in packets if p["role"] in {"primary", "required_prerequisite"}],
                    "auxiliary_context": [p for p in packets if p["role"] in {"supporting", "auxiliary", "explanation_only"}],
                    "final_status": item["final_status"],
                }
            )

    claim_evaluations = _build_claim_evaluations(
        all_routed, round0_verdicts, revisions, round1_results
    )
    shared_packet = _shared_evidence_packet(room, claim_evaluations)
    debate = run_post_validation_debate(
        shared_packet,
        settings,
        telemetry,
        progress=lambda stage: _progress(progress_callback, stage),
    )

    warnings = []
    if telemetry["fallback_used"]:
        warnings.append("하나 이상의 LLM 단계가 미설정 또는 실패하여 deterministic fallback으로 대체되었습니다.")
    if ml_context.get("inference_mode") != "runtime_frozen_catboost":
        warnings.append("최신 Frozen CatBoost runtime inference를 사용할 수 없어 기존 ML adapter로 대체했습니다.")
    warnings.append(
        f"통계 Evidence는 Frozen development cutoff {settings.development_cutoff_date.isoformat()}를 유지하며, 서비스 기준일과 별도로 표시됩니다."
    )

    telemetry.update(
        {
            "stock_code": room["stock_code"],
            "analysis_as_of_date": room["as_of_date"],
            "bull_claim_count": len(initial_by_role["bull"]),
            "bear_claim_count": len(initial_by_role["bear"]),
            "verdict_counts": dict(Counter(v["claim_verdict"] for v in round0_verdicts.values())),
            "max_revision_round": 1,
            "llm_provider": settings.llm_provider or None,
            "llm_model": settings.llm_model or None,
            "llm_configured": settings.llm_configured,
            "fact_room_shared_hash": room["fact_room_hash"],
            "shared_evidence_hash": debate["shared_evidence_hash"],
            "total_latency_sec": time.perf_counter() - started_total,
            "error_codes": sorted(set(telemetry["error_codes"])),
        }
    )

    result = {
        "company": room["company_name"],
        "stock_code": room["stock_code"],
        "analysis_as_of_date": room["as_of_date"],
        "development_cutoff_date": settings.development_cutoff_date.isoformat(),
        "prototype_data_cutoff_notice": settings.prototype_notice,
        "bull": {"initial_claims": initial_by_role["bull"], "final_claims": final_by_role["bull"]},
        "bear": {"initial_claims": initial_by_role["bear"], "final_claims": final_by_role["bear"]},
        "dropped_or_revised_claims": dropped_or_revised,
        "claim_evaluations": claim_evaluations,
        "fact_room_summary": fact_room_summary(room),
        "evidence_board": board,
        "debate": debate,
        "citations": _extract_citations(board, dropped_or_revised),
        "ml_context": ml_context,
        "warnings": warnings,
        "execution_metadata": telemetry,
    }
    result["safety_gates"] = evaluate_safety_gates(result)
    validate_service_response(result)
    _append_request_log(telemetry)
    _progress(progress_callback, "완료")
    return result


def run_evidence_arena(stock_code: str) -> dict:
    """Run Evidence Arena at the fixed, data-verified service snapshot."""
    return _run_evidence_arena(stock_code=stock_code)
