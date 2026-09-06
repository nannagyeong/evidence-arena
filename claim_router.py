# Auto-generated adapter around the frozen STEP 12 routing contract.
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from . import statistical_validator as step8

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
ROUTING_DIR = DATA_DIR / "claim_routing"
VALIDATOR_DIR = DATA_DIR / "claim_validator"
RAG_DIR = DATA_DIR / "rag_evidence"
FEATURE_DIR = DATA_DIR / "features"

def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def stable_json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)

def sha256_text(value) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()

output_schema = json.loads((ROUTING_DIR / "02_claim_output_schema.json").read_text(encoding="utf-8"))
output_validator = Draft202012Validator(output_schema)
template_registry_step8 = pd.read_csv(VALIDATOR_DIR / "02_claim_template_registry.csv")
feature_allowlist_step8 = pd.read_csv(VALIDATOR_DIR / "02_claim_feature_allowlist.csv")
control_registry = pd.read_csv(VALIDATOR_DIR / "02_control_set_registry.csv")
feature_registry = pd.read_csv(FEATURE_DIR / "01_feature_registry.csv")
feature_panel = pd.read_parquet(FEATURE_DIR / "feature_panel.parquet")
feature_panel["stock_code"] = feature_panel["stock_code"].astype(str).str.zfill(6)

TEMPLATE_BY_ID = template_registry_step8.set_index("template_id").to_dict("index")
ALLOWLIST_BY_TEMPLATE = {
    template: set(group["feature_name"].dropna().astype(str))
    for template, group in feature_allowlist_step8.groupby("template_id")
}
CONTROL_SETS = {
    row.control_set_id: tuple(part.strip() for part in str(row.controls).split("|") if part.strip())
    for row in control_registry.itertuples(index=False)
}
FINANCE_STOCK_CODES = set(feature_panel.loc[feature_panel["is_finance_industry"].fillna(False), "stock_code"])
FINANCE_DIRECT_BLOCKED = set(feature_registry.loc[
    feature_registry["claim_router_policy"].astype(str).eq("block_direct_claim_for_finance")
    | feature_registry["interpretation_scope"].astype(str).eq("non_financial_only"), "feature_name"
].astype(str))
FINANCE_PEER_ONLY = set(feature_registry.loc[
    feature_registry["claim_router_policy"].astype(str).eq("finance_peer_comparison_only")
    | feature_registry["interpretation_scope"].astype(str).eq("finance_industry_relative_only"), "feature_name"
].astype(str))

frozen_rag_policy = json.loads((RAG_DIR / "09_frozen_retrieval_policy.json").read_text(encoding="utf-8"))
FROZEN_RAG_TOPICS = set(frozen_rag_policy["frozen_policy"]["query_expansion_dictionary"])
ML_ADAPTER_PATH = RAG_DIR / "16_step10_ml_evidence_adapter_2025.parquet"
ml_adapter = pd.read_parquet(ML_ADAPTER_PATH)
ml_adapter["stock_code"] = ml_adapter["stock_code"].astype(str).str.zfill(6)
ml_adapter["as_of_date"] = pd.to_datetime(ml_adapter["as_of_date"], errors="coerce")
ML_DATES_BY_STOCK = {
    stock: np.sort(group["as_of_date"].dropna().values.astype("datetime64[ns]"))
    for stock, group in ml_adapter.groupby("stock_code")
}

def make_route(
    engine: str,
    template_id: str | None = None,
    x_feature: str | None = None,
    expected_direction: str | None = None,
    condition_operator: str | None = None,
    condition_value: float | None = None,
    controls: list[str] | None = None,
    event_type: str | None = None,
    comparison_scope: str | None = None,
    rag_topics: list[str] | None = None,
    route_reason: str = "",
) -> dict:
    policy = TEMPLATE_BY_ID.get(template_id, {})
    return {
        "engine": engine,
        "template_id": template_id,
        "x_feature": x_feature,
        "outcome": policy.get("outcome"),
        "horizon": 5 if template_id in {"T1", "T2", "T3", "T4"} else None,
        "condition_operator": condition_operator,
        "condition_value": condition_value,
        "expected_direction": expected_direction,
        "controls": controls or [],
        "event_type": event_type,
        "event_window": [0, 4] if template_id == "T4" else None,
        "comparison_scope": comparison_scope if template_id == "T6" else None,
        "rag_topics": rag_topics or [],
        "route_reason": route_reason,
        "execution_allowed": False,
        "route_errors": [],
    }


def build_evidence_routes(
    text: str,
    template_id: str | None,
    feature: str | None,
    direction: str | None,
    condition_operator: str | None,
    condition_value: float | None,
    controls: list[str],
    comparison_scope: str | None,
    rag_topics: list[str],
    recommendation_request: bool,
) -> list[dict]:
    routes = []
    if template_id in TEMPLATE_BY_ID:
        routes.append(make_route(
            engine=template_id,
            template_id=template_id,
            x_feature=feature,
            expected_direction=direction,
            condition_operator=condition_operator,
            condition_value=condition_value,
            controls=controls,
            event_type="earnings_improvement_report" if template_id == "T4" else None,
            comparison_scope=comparison_scope,
            route_reason="STEP 8 Template Registry에 의해 결정",
        ))

    explanatory_request = any(term in text for term in ("설명", "원인", "배경", "근거", "위험", "리스크", "계약", "사업"))
    if rag_topics and (template_id in {None, "RAG"} or explanatory_request):
        routes.append(make_route(
            engine="RAG", rag_topics=rag_topics,
            route_reason="STEP 11 Frozen RAG topic과 연결",
        ))

    future_context = any(term in text for term in ("앞으로", "향후", "전망", "지속", "사도", "매수", "매도"))
    if recommendation_request or future_context:
        routes.append(make_route(
            engine="ML",
            route_reason="STEP 10 복합 시장환경의 보조 Evidence; 확률형 투자추천 금지",
        ))
        if not any(route["engine"] == "RAG" for route in routes):
            routes.append(make_route(
                engine="RAG", rag_topics=rag_topics or ["business_growth"],
                route_reason="추천 대신 Evidence Board용 정성 근거 조회",
            ))

    deduplicated = []
    seen = set()
    for route in routes:
        token = (route["engine"], route["template_id"], tuple(route["rag_topics"]))
        if token not in seen:
            deduplicated.append(route)
            seen.add(token)
    return deduplicated


def is_registered_control_set(controls: list[str]) -> bool:
    return tuple(controls) in set(CONTROL_SETS.values())


def ml_evidence_available_for_asof(stock_code: str, as_of_date: str) -> bool:
    dates = ML_DATES_BY_STOCK.get(str(stock_code).zfill(6))
    if dates is None or len(dates) == 0:
        return False
    cutoff = np.datetime64(pd.Timestamp(as_of_date).to_datetime64(), "ns")
    return bool(np.any(dates <= cutoff))


def contains_forbidden_output(value) -> bool:
    forbidden = {"verdict", "final_verdict", "buy_sell", "target_price", "recommendation"}
    if isinstance(value, dict):
        return any(key in forbidden or contains_forbidden_output(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_output(item) for item in value)
    return False


def validate_atomic_claim(atomic: dict) -> dict:
    route_errors = []
    valid_executions = 0
    failed_routes = 0
    for route in atomic["routes"]:
        errors = []
        engine = route["engine"]
        if engine in TEMPLATE_BY_ID:
            if route["template_id"] != engine:
                errors.append("template_engine_mismatch")
            if route["x_feature"] not in ALLOWLIST_BY_TEMPLATE.get(engine, set()):
                errors.append("registry_out_feature")
            if engine in {"T1", "T2", "T3", "T4"} and route["horizon"] != 5:
                errors.append("statistical_horizon_must_be_5")
            if engine in {"T5", "T6"} and route["horizon"] is not None:
                errors.append("fact_or_relative_horizon_must_be_null")
            if engine == "T3" and not is_registered_control_set(route["controls"]):
                errors.append("unregistered_control_set")
            if engine != "T3" and route["controls"]:
                errors.append("controls_only_allowed_for_T3")
            if engine == "T4" and (route["event_type"] != "earnings_improvement_report" or route["event_window"] != [0, 4]):
                errors.append("event_contract_mismatch")
            if engine == "T6" and route["comparison_scope"] not in {"history", "industry"}:
                errors.append("relative_scope_missing")
            if atomic["stock_code"] in FINANCE_STOCK_CODES:
                feature = route["x_feature"]
                if feature in FINANCE_DIRECT_BLOCKED:
                    errors.append("finance_direct_claim_blocked")
                if feature in FINANCE_PEER_ONLY and not (engine == "T6" and route["comparison_scope"] == "industry"):
                    errors.append("finance_peer_scope_required")
        elif engine == "RAG":
            if not route["rag_topics"] or not set(route["rag_topics"]).issubset(FROZEN_RAG_TOPICS):
                errors.append("unknown_rag_topic")
        elif engine == "ML":
            if not ML_ADAPTER_PATH.exists():
                errors.append("step10_adapter_missing")
            elif not ml_evidence_available_for_asof(atomic["stock_code"], atomic["as_of_date"]):
                errors.append("ml_evidence_unavailable_for_asof")
        elif engine not in {"UNSUPPORTED", "CLARIFICATION"}:
            errors.append("invalid_engine")

        route["route_errors"] = list(errors)
        route["execution_allowed"] = not errors and engine not in {"UNSUPPORTED", "CLARIFICATION"}
        valid_executions += int(route["execution_allowed"])
        failed_routes += int(bool(errors))
        route_errors.extend(f"{engine}:{error}" for error in errors)

    schema_errors = sorted(output_validator.iter_errors(atomic), key=lambda error: list(error.path))
    errors = route_errors + ["schema:" + error.message for error in schema_errors]
    if contains_forbidden_output({key: value for key, value in atomic.items() if key != "validation"}):
        errors.append("forbidden_final_output_field")
    return {
        "schema_valid": len(schema_errors) == 0,
        "execution_allowed": valid_executions > 0,
        "partial_route_failure": valid_executions > 0 and failed_routes > 0,
        "valid_execution_count": valid_executions,
        "failed_route_count": failed_routes,
        "errors": errors,
    }


TEMPLATE_FAMILY = {
    "T1": "conditional", "T2": "continuous", "T3": "controlled",
    "T4": "event", "T5": "fact_trend", "T6": "relative", "RAG": "rag",
}

def to_step8_claim(claim: dict, validation_batch_id: str) -> dict | None:
    template_id = claim.get("template_id")
    if template_id not in step8.TEMPLATE_CONFIG:
        return None
    config = step8.TEMPLATE_CONFIG[template_id]
    return {
        "claim_id": claim["claim_id"],
        "validation_batch_id": validation_batch_id,
        "claim_text": claim["atomic_claim_text"],
        "stock_code": str(claim["stock_code"]).zfill(6),
        "as_of_date": str(pd.Timestamp(claim["as_of_date"]).date()),
        "claim_type": config["claim_type"],
        "x_feature": claim["feature"],
        "condition_operator": claim.get("condition_operator"),
        "condition_value": claim.get("condition_value"),
        "outcome": config["outcome"],
        "horizon": 5,
        "expected_direction": claim.get("expected_direction") or "positive",
        "controls": list(claim.get("controls") or []),
        "event_type": "earnings_improvement_report" if template_id == "T4" else None,
        "comparison_scope": claim.get("comparison_scope") if template_id == "T6" else None,
    }

def structured_route_claim(claim: dict, validation_batch_id: str = "STEP14-ROUND0") -> dict:
    """Validate an already-structured Bull/Bear claim with the frozen STEP 12 rules."""
    template_id = claim.get("template_id")
    feature = claim.get("feature")
    direction = claim.get("expected_direction")
    text = str(claim.get("atomic_claim_text", ""))
    routes = build_evidence_routes(
        text=text,
        template_id=template_id if template_id in TEMPLATE_BY_ID else None,
        feature=feature,
        direction=direction,
        condition_operator=claim.get("condition_operator"),
        condition_value=claim.get("condition_value"),
        controls=list(claim.get("controls") or []),
        comparison_scope=claim.get("comparison_scope"),
        rag_topics=list(claim.get("rag_topics") or []),
        recommendation_request=False,
    )
    if template_id == "RAG" and not routes:
        routes = [make_route(engine="RAG", rag_topics=list(claim.get("rag_topics") or []), route_reason="STEP 11 Frozen RAG topic과 연결")]
    atomic = {
        "claim_id": claim["claim_id"],
        "parent_claim_id": claim.get("parent_claim_id"),
        "stock_code": str(claim["stock_code"]).zfill(6),
        "as_of_date": str(pd.Timestamp(claim["as_of_date"]).date()),
        "source_agent": claim["source_agent"],
        "original_text": text,
        "atomic_claim_text": text,
        "claim_family": claim.get("claim_family") or TEMPLATE_FAMILY.get(template_id, "unsupported"),
        "template_id": template_id,
        "feature": feature,
        "condition_operator": claim.get("condition_operator"),
        "condition_value": claim.get("condition_value"),
        "expected_direction": direction,
        "horizon": 5 if template_id in {"T1", "T2", "T3", "T4"} else None,
        "comparison_scope": claim.get("comparison_scope"),
        "event_type": "earnings_improvement_report" if template_id == "T4" else None,
        "event_window": [0, 4] if template_id == "T4" else None,
        "controls": list(claim.get("controls") or []),
        "rag_topics": list(claim.get("rag_topics") or []),
        "routes": routes,
        "routing_status": "routable",
        "extraction_confidence": 1.0,
        "requires_clarification": False,
        "causal_language_detected": False,
        "causal_inference_supported": False,
        "recommendation_request": False,
        "analysis_rewrite": None,
        "policy_notes": [],
        "validation": {},
    }
    atomic["validation"] = validate_atomic_claim(atomic)
    step8_claim = to_step8_claim(atomic, validation_batch_id)
    if step8_claim is not None:
        try:
            step8.validate_claim_contract(step8_claim)
        except Exception as error:
            for route in atomic["routes"]:
                if route["engine"] == template_id:
                    route["execution_allowed"] = False
                    route["route_errors"] = sorted(set(route["route_errors"] + [str(error)]))
            atomic["validation"] = validate_atomic_claim(atomic)
    atomic["routing_status"] = "routable" if atomic["validation"]["execution_allowed"] else "unsupported"
    schema_errors = list(output_validator.iter_errors(atomic))
    if schema_errors:
        atomic["validation"]["schema_valid"] = False
        atomic["validation"]["errors"].extend("schema:" + error.message for error in schema_errors)
    return {
        "step12_status": atomic["routing_status"],
        "routing_status": atomic["routing_status"],
        "validated_claim": atomic,
        "claim_scope": claim.get("claim_scope") or {
            "T1": "historical_relation", "T2": "historical_relation", "T3": "historical_relation",
            "T4": "event", "T5": "fact", "T6": "relative", "RAG": "qualitative",
        }.get(template_id),
        "step8_claim": step8_claim,
        "router_version": "step12_frozen_structured_adapter_v1",
        "router_hash": sha256_text(atomic),
    }

UPSTREAM_PROVENANCE = {
    "source_notebook": str(PROJECT_ROOT / "step12_13.ipynb"),
    "source_notebook_sha256": "2d42d86f19e3e8c5527624626f5d225d20650da82aa18d8090fa587c4724ab66",
    "function_names": ['make_route', 'build_evidence_routes', 'is_registered_control_set', 'ml_evidence_available_for_asof', 'contains_forbidden_output', 'validate_atomic_claim'],
    "implementation": "exact_STEP12_route_building_and_validation_functions",
}
