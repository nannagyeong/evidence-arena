# STEP 14 deterministic rule-based prototype and one-round closed-loop controller.
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from . import claim_router
from . import evidence_adapters
from . import verdict_engine

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
FEATURE_DIR = DATA_DIR / "features"
RAG_DIR = DATA_DIR / "rag_evidence"
STEP14_DIR = DATA_DIR / "closed_loop"
STEP14_DIR.mkdir(parents=True, exist_ok=True)

def json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)

def object_hash(value) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()

feature_panel = pd.read_parquet(FEATURE_DIR / "feature_panel.parquet")
feature_panel["stock_code"] = feature_panel["stock_code"].astype(str).str.zfill(6)
feature_panel["trading_date"] = pd.to_datetime(feature_panel["trading_date"], errors="coerce")
feature_registry = pd.read_csv(FEATURE_DIR / "01_feature_registry.csv")
FEATURE_META = feature_registry.set_index("feature_name").to_dict("index")
document_manifest = evidence_adapters.document_manifest.copy()
parsed_receipts = set(evidence_adapters.parse_audit.loc[evidence_adapters.parse_audit["parse_status"].eq("success"), "rcept_no"])
DATA_MIN_DATE = feature_panel["trading_date"].min().normalize()
DATA_MAX_DATE = feature_panel["trading_date"].max().normalize()

FUNDAMENTAL_FEATURES = ["revenue_yoy", "operating_income_yoy", "net_income_yoy", "operating_margin", "operating_margin_change_yoy", "debt_ratio", "cfo_to_assets", "fy_roe", "fy_roa"]
MARKET_FEATURES = ["adjusted_close", "ret_1", "ret_5", "ret_20", "excess_ret_5", "realized_vol_20", "momentum_60", "drawdown_60", "volume_z_20", "beta_60", "market_corr_60", "market_regime_id"]
MACRO_FEATURES = ["usdkrw_level", "usdkrw_change_20", "base_rate_level", "base_rate_change_20", "treasury_3y_level", "treasury_3y_change_20", "cpi_yoy", "ppi_yoy", "industrial_production_raw_yoy", "export_yoy", "import_yoy"]
FEATURE_UNITS = {"adjusted_close": "KRW", "usdkrw_level": "KRW/USD", "base_rate_level": "%", "treasury_3y_level": "%", "market_regime_id": "category"}

def native_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    return str(value)

def feature_unit(feature: str) -> str:
    if feature in FEATURE_UNITS:
        return FEATURE_UNITS[feature]
    if feature.endswith("_yoy") or "ret_" in feature or feature in {"realized_vol_20", "momentum_60", "drawdown_60", "debt_ratio", "cfo_to_assets", "fy_roe", "fy_roa", "operating_margin", "operating_margin_change_yoy", "beta_60", "market_corr_60"}:
        return "ratio"
    return "value"

def latest_feature_row(stock_code: str, as_of_date) -> pd.Series:
    stock_code = str(stock_code).zfill(6)
    as_of = pd.Timestamp(as_of_date).normalize()
    eligible = feature_panel.loc[feature_panel["stock_code"].eq(stock_code) & feature_panel["trading_date"].le(as_of)].sort_values("trading_date")
    if eligible.empty:
        raise ValueError(f"{stock_code} / {as_of.date()} 이전 Feature가 없습니다.")
    return eligible.iloc[-1]

def build_fact_room(stock_code: str, as_of_date) -> dict:
    stock_code = str(stock_code).zfill(6)
    as_of = pd.Timestamp(as_of_date).normalize()
    if not DATA_MIN_DATE <= as_of <= DATA_MAX_DATE:
        raise ValueError("as_of_date is outside feature panel range")
    row = latest_feature_row(stock_code, as_of)
    fundamental = {}
    for feature in FUNDAMENTAL_FEATURES:
        effective_column = "fy_financial_effective_date" if feature in {"fy_roe", "fy_roa"} else "financial_effective_date"
        receipt_column = "fy_rcept_no" if feature in {"fy_roe", "fy_roa"} else "financial_rcept_no"
        effective_date = pd.to_datetime(row.get(effective_column), errors="coerce")
        if pd.notna(effective_date) and effective_date.normalize() > as_of:
            raise AssertionError(f"PIT violation in {feature}")
        receipt = row.get(receipt_column)
        fundamental[feature] = {"value": native_value(row.get(feature)), "unit": feature_unit(feature), "effective_date": native_value(effective_date), "rcept_no": None if pd.isna(receipt) else str(receipt), "interpretation_scope": FEATURE_META.get(feature, {}).get("interpretation_scope", "unspecified")}
    market_context = {feature: {"value": native_value(row.get(feature)), "unit": feature_unit(feature), "data_date": row["trading_date"].date().isoformat()} for feature in MARKET_FEATURES}
    macro_context = {}
    for feature in MACRO_FEATURES:
        base = feature.split("_change_")[0]
        if feature.startswith("usdkrw"):
            effective_column = "macro_usdkrw__effective_date"
        elif feature.startswith("base_rate"):
            effective_column = "macro_base_rate__effective_date"
        elif feature.startswith("treasury_3y"):
            effective_column = "macro_treasury_3y__effective_date"
        else:
            effective_column = f"{base}__effective_date"
        effective_date = pd.to_datetime(row.get(effective_column), errors="coerce")
        if pd.notna(effective_date) and effective_date.normalize() > as_of:
            raise AssertionError(f"PIT violation in {feature}")
        macro_context[feature] = {"value": native_value(row.get(feature)), "unit": feature_unit(feature), "effective_date": native_value(effective_date)}
    disclosures = document_manifest.loc[document_manifest["stock_code"].eq(stock_code) & document_manifest["effective_date"].notna() & document_manifest["effective_date"].le(as_of) & document_manifest["rcept_no"].isin(parsed_receipts)].copy()
    disclosures = disclosures.sort_values(["report_slot_id", "version_order", "effective_date", "rcept_no"]).groupby("report_slot_id", as_index=False).tail(1).sort_values(["effective_date", "rcept_no"], ascending=[False, False]).head(5)
    recent_disclosures = [{"rcept_no": str(item.rcept_no), "report_name": item.report_nm, "report_type": item.report_type, "effective_date": pd.Timestamp(item.effective_date).date().isoformat(), "source_url": item.source_url} for item in disclosures.itertuples(index=False)]
    room = {"fact_room_id": f"FR-{stock_code}-{as_of.date().isoformat()}", "stock_code": stock_code, "company_name": str(row["company_name"]), "market": str(row["market"]), "industry": str(row["industry"]), "is_finance_industry": bool(row["is_finance_industry"]), "as_of_date": as_of.date().isoformat(), "data_cutoff_trading_date": row["trading_date"].date().isoformat(), "fundamental_snapshot": fundamental, "market_context": market_context, "macro_context": macro_context, "recent_disclosures": recent_disclosures, "pit_guards": {"future_target_included": False, "step8_statistics_included": False, "step13_verdict_included": False, "ml_shap_included": False, "all_source_dates_lte_as_of": True}}
    room["fact_room_hash"] = object_hash(room)
    return room

INITIAL_VIEW_KEYS = {"fact_room_id", "fact_room_hash", "stock_code", "company_name", "market", "industry", "is_finance_industry", "as_of_date", "data_cutoff_trading_date", "fundamental_snapshot", "market_context", "macro_context", "recent_disclosures", "pit_guards"}
INITIAL_FORBIDDEN_KEYS = {"p_value", "q_value", "confidence_interval", "claim_verdict", "step13_verdict", "shap", "shap_values", "opponent_claims", "future_excess_return_5d", "target", "label"}

def recursive_keys(value) -> set[str]:
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key)); keys.update(recursive_keys(child))
    elif isinstance(value, list):
        for child in value: keys.update(recursive_keys(child))
    return keys

def initial_agent_view(fact_room: dict) -> dict:
    view = {key: copy.deepcopy(fact_room[key]) for key in INITIAL_VIEW_KEYS}
    leaked = recursive_keys(view) & INITIAL_FORBIDDEN_KEYS
    if leaked: raise AssertionError(f"Initial Agent information leakage: {sorted(leaked)}")
    return view

MAX_INITIAL_CLAIMS_PER_AGENT = 3
COMPOSITE_PATTERNS = [r"그리고", r"뿐만 아니라", r"동시에", r"이며\s+.*(?:이다|다)"]
FORBIDDEN_TEXT_PATTERNS = [r"매수", r"매도", r"strong\s*buy", r"buy", r"sell", r"목표\s*주가", r"수익(?:률)?\s*(?:보장|확정)", r"반드시\s*(?:상승|하락)", r"\d+(?:\.\d+)?%\s*확률로"]

AGENT_CLAIM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["claim_id", "parent_claim_id", "source_agent", "stock_code", "as_of_date", "atomic_claim_text", "claim_family", "claim_scope", "template_id", "feature", "expected_direction", "comparison_scope", "condition_operator", "condition_value", "horizon", "controls", "rag_topics", "event_type", "event_window"],
    "properties": {
        "claim_id": {"type": "string", "minLength": 3}, "parent_claim_id": {"type": ["string", "null"]},
        "source_agent": {"enum": ["bull", "bear"]}, "stock_code": {"type": "string", "pattern": "^[0-9]{6}$"},
        "as_of_date": {"type": "string"}, "atomic_claim_text": {"type": "string", "minLength": 5},
        "claim_family": {"enum": ["conditional", "continuous", "controlled", "event", "fact_trend", "relative", "rag"]},
        "claim_scope": {"enum": ["historical_relation", "event", "fact", "relative", "qualitative", "broad", "forecast"]},
        "template_id": {"enum": ["T1", "T2", "T3", "T4", "T5", "T6", "RAG"]},
        "feature": {"type": ["string", "null"]}, "expected_direction": {"enum": ["positive", "negative", None]},
        "comparison_scope": {"enum": ["yoy", "history", "industry", None]},
        "condition_operator": {"enum": [">", ">=", "<", "<=", "==", None]}, "condition_value": {"type": ["number", "null"]},
        "horizon": {"type": ["integer", "null"]}, "controls": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "rag_topics": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "event_type": {"type": ["string", "null"]}, "event_window": {"type": ["array", "null"]},
    },
}
agent_claim_validator = Draft202012Validator(AGENT_CLAIM_SCHEMA)

def is_atomic_claim(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    return len([part for part in re.split(r"[.!?]", normalized) if part.strip()]) == 1 and not any(re.search(pattern, normalized) for pattern in COMPOSITE_PATTERNS)

def claim_signature(claim: dict) -> tuple:
    return (claim.get("template_id"), claim.get("feature"), claim.get("comparison_scope"), claim.get("condition_operator"), claim.get("condition_value"), claim.get("expected_direction"))

def validate_agent_claim(claim: dict, fact_room: dict) -> None:
    errors = list(agent_claim_validator.iter_errors(claim))
    if errors: raise ValueError("Agent Claim schema: " + " | ".join(error.message for error in errors))
    if claim["stock_code"] != fact_room["stock_code"] or claim["as_of_date"] != fact_room["as_of_date"]: raise ValueError("Claim scope differs from Fact Room")
    if any(re.search(pattern, claim["atomic_claim_text"], flags=re.IGNORECASE) for pattern in FORBIDDEN_TEXT_PATTERNS): raise ValueError("Forbidden investment/certainty expression")
    if not is_atomic_claim(claim["atomic_claim_text"]): raise ValueError("Claim is not atomic")

def validate_claim_batch(claims: list[dict], source_agent: str, fact_room: dict) -> None:
    if not 1 <= len(claims) <= MAX_INITIAL_CLAIMS_PER_AGENT: raise ValueError("claim count must be 1~3")
    if any(claim["source_agent"] != source_agent for claim in claims): raise ValueError("source_agent mismatch")
    if len({claim_signature(claim) for claim in claims}) != len(claims): raise ValueError("Duplicate Claim detected")
    for claim in claims: validate_agent_claim(claim, fact_room)

T5_LABELS = {"revenue_yoy": "매출", "operating_income_yoy": "영업이익", "net_income_yoy": "순이익", "operating_margin_change_yoy": "영업이익률"}

def make_claim(room: dict, source_agent: str, sequence: int, template_id: str, feature, text: str, expected_direction, comparison_scope, condition_operator=None, condition_value=None, rag_topics=None) -> dict:
    family = {"T1": "conditional", "T2": "continuous", "T3": "controlled", "T4": "event", "T5": "fact_trend", "T6": "relative", "RAG": "rag"}[template_id]
    scope = {"T1": "historical_relation", "T2": "historical_relation", "T3": "historical_relation", "T4": "event", "T5": "fact", "T6": "relative", "RAG": "qualitative"}[template_id]
    prefix = "BULL" if source_agent == "bull" else "BEAR"
    return {"claim_id": f"{prefix}-{room['stock_code']}-{sequence:03d}", "parent_claim_id": None, "source_agent": source_agent, "stock_code": room["stock_code"], "as_of_date": room["as_of_date"], "atomic_claim_text": text, "claim_family": family, "claim_scope": scope, "template_id": template_id, "feature": feature, "expected_direction": expected_direction, "comparison_scope": comparison_scope, "condition_operator": condition_operator, "condition_value": condition_value, "horizon": 5 if template_id in {"T1", "T2", "T3", "T4"} else None, "controls": [], "rag_topics": list(rag_topics or []), "event_type": "earnings_improvement_report" if template_id == "T4" else None, "event_window": [0, 4] if template_id == "T4" else None}

SHARED_AGENT_INSTRUCTION = "PIT-safe Fact Room만 사용하며 최대 3개의 구조화된 Atomic Claim만 생성한다. Verdict·통계량·SHAP·투자 권유는 생성하지 않는다."
AGENT_RUNTIME_CONTRACT = {"implementation": "deterministic_rule_based_prototype", "actual_llm_agent": False, "external_callable_schema_compatible": True, "optional_external_callable": "callable(prompt, fact_room, schema) -> list[claim]", "external_output_must_pass_python_guards": True, "temperature_when_external_llm_is_used": 0}

class StructuredClaimAgent:
    def __init__(self, role: str, external_callable=None):
        if role not in {"bull", "bear"}: raise ValueError("role must be bull or bear")
        self.role = role
        self.prompt = ("긍정적" if role == "bull" else "위험·약점") + " 관점의 검증 가능한 Claim을 생성한다. " + SHARED_AGENT_INSTRUCTION
        self.external_callable = external_callable

    def _candidate_pool(self, room: dict) -> list[dict]:
        candidates = []
        fundamental = room["fundamental_snapshot"]
        desired_positive = self.role == "bull"
        for feature in ["operating_margin_change_yoy", "operating_income_yoy", "revenue_yoy", "net_income_yoy"]:
            value = fundamental.get(feature, {}).get("value")
            if value is None or float(value) == 0 or (float(value) > 0) != desired_positive: continue
            direction = "positive" if float(value) > 0 else "negative"
            ending = "개선됐다" if direction == "positive" else "악화됐다"
            candidates.append(make_claim(room, self.role, len(candidates)+1, "T5", feature, f"{room['company_name']}의 {T5_LABELS[feature]}이 전년동기 대비 {ending}.", direction, "yoy", rag_topics=["profitability"]))
        if room["is_finance_industry"]:
            specs = [("fy_roe", ">=" if self.role == "bull" else "<=", 0.5, "ROE"), ("fy_roa", ">=" if self.role == "bull" else "<=", 0.5, "ROA"), ("debt_ratio", "<=" if self.role == "bull" else ">=", 0.5, "부채비율")]
            for feature, operator, threshold, label in specs:
                if fundamental.get(feature, {}).get("value") is None: continue
                level = "높은" if operator == ">=" else "낮은"
                candidates.append(make_claim(room, self.role, len(candidates)+1, "T6", feature, f"{room['company_name']}의 {label}가 같은 금융업 비교기업 분포에서 {level} 수준이다.", "positive" if self.role == "bull" else "negative", "industry", operator, threshold))
        else:
            specs = [("momentum_60", ">=", 0.5, "최근 60일 모멘텀이 과거 분포에서 높은 수준이다") if self.role == "bull" else ("realized_vol_20", ">=", 0.5, "최근 20일 실현변동성이 과거 분포에서 높은 수준이다"), ("excess_ret_20", ">=", 0.5, "최근 20일 시장초과수익률이 과거 분포에서 높은 수준이다") if self.role == "bull" else ("drawdown_60", "<=", 0.5, "최근 60일 낙폭이 과거 분포에서 낮은 수준이다")]
            for feature, operator, threshold, text in specs:
                if room["market_context"].get(feature, {}).get("value") is None: continue
                candidates.append(make_claim(room, self.role, len(candidates)+1, "T6", feature, f"{room['company_name']}의 {text}.", "positive" if self.role == "bull" else "negative", "history", operator, threshold))
        if self.role == "bull" and room["stock_code"] == "005930":
            candidates.insert(0, make_claim(room, self.role, 90, "RAG", None, "삼성전자는 최근 실적 변화의 배경을 공식 공시에서 설명했다.", "positive", None, rag_topics=["profitability"]))
        return candidates

    def generate_initial_claims(self, fact_room: dict) -> list[dict]:
        view = initial_agent_view(fact_room)
        pool = self._candidate_pool(view) if self.external_callable is None else self.external_callable(self.prompt, copy.deepcopy(view), copy.deepcopy(AGENT_CLAIM_SCHEMA))
        valid = []
        for candidate in pool:
            candidate = copy.deepcopy(candidate)
            candidate["claim_id"] = ("BULL" if self.role == "bull" else "BEAR") + f"-{fact_room['stock_code']}-{len(valid)+1:03d}"
            try:
                validate_agent_claim(candidate, fact_room)
                routed = claim_router.structured_route_claim(candidate, "STEP14-CANDIDATE-GUARD")
                if not routed["validated_claim"]["validation"]["execution_allowed"]: continue
                valid.append(candidate)
            except Exception:
                continue
            if len(valid) == MAX_INITIAL_CLAIMS_PER_AGENT: break
        validate_claim_batch(valid, self.role, fact_room)
        return valid

REVISION_ACTIONS_BY_VERDICT = {"Supported": ("KEEP",), "Partially Supported": ("WEAKEN",), "Insufficient Evidence": ("DROP", "REFRAME"), "Contradicted": ("DROP",)}
DEFAULT_REVISION_ACTION = {"Supported": "KEEP", "Partially Supported": "WEAKEN", "Insufficient Evidence": "DROP", "Contradicted": "DROP"}

def select_revision_action(verdict: str, claim: dict) -> str:
    if verdict == "Insufficient Evidence" and claim.get("template_id") in {"T2", "T3"} and claim.get("feature") in {"revenue_yoy", "operating_income_yoy", "net_income_yoy", "operating_margin"}: return "REFRAME"
    return DEFAULT_REVISION_ACTION[verdict]

def weaken_claim(original_claim: dict) -> dict:
    revised = copy.deepcopy(original_claim)
    revised["claim_id"] = original_claim["claim_id"] + "-R1"
    revised["parent_claim_id"] = original_claim["claim_id"]
    revised["atomic_claim_text"] = "현재 확인 가능한 Evidence 범위에서는 " + original_claim["atomic_claim_text"].rstrip(". ") + "."
    return revised

def make_reframed_fact_claim(original_claim: dict, fact_room: dict) -> dict:
    signal = "operating_margin_change_yoy" if original_claim["feature"] == "operating_margin" else original_claim["feature"]
    value = fact_room["fundamental_snapshot"].get(signal, {}).get("value")
    if value is None or float(value) == 0: raise ValueError("PIT financial fact unavailable for REFRAME")
    direction = "positive" if float(value) > 0 else "negative"
    label = T5_LABELS.get(signal, signal)
    revised = make_claim(fact_room, original_claim["source_agent"], 999, "T5", original_claim["feature"], f"현재 확인 가능한 PIT 재무자료에서 {label}이 전년동기 대비 {'개선됐다' if direction == 'positive' else '악화됐다'}.", direction, "yoy", rag_topics=["profitability"])
    revised["claim_id"] = original_claim["claim_id"] + "-R1"
    revised["parent_claim_id"] = original_claim["claim_id"]
    return revised

def build_revision_record(claim: dict, verdict_result: dict, fact_room: dict) -> dict:
    verdict = verdict_result["claim_verdict"]
    action = select_revision_action(verdict, claim)
    revised = weaken_claim(claim) if action == "WEAKEN" else make_reframed_fact_claim(claim, fact_room) if action == "REFRAME" else None
    return {"claim_id": claim["claim_id"], "parent_claim_id": claim["claim_id"], "source_agent": claim["source_agent"], "revision_round": 1, "revision_action": action, "previous_claim": copy.deepcopy(claim), "previous_verdict": verdict, "step13_verdict": verdict, "revision_reason": {"KEEP": "Primary Evidence가 지지함", "WEAKEN": "근거 한계를 반영해 표현 범위를 약화함", "DROP": "지지되지 않거나 반대 근거가 있어 제외함", "REFRAME": "관계 주장을 PIT Fact 주장으로 낮춤"}[action], "revised_claim": revised, "requires_revalidation": action in {"WEAKEN", "REFRAME"}}

def synthesize_routed_batch(routed_items: list[dict]) -> dict[str, dict]:
    raw_map = evidence_adapters.execute_evidence_routes(routed_items)
    outputs = {}
    for item in routed_items:
        claim = item["validated_claim"]
        request = verdict_engine.collect_evidence_for_claim(claim, item["claim_scope"], raw_map.get(claim["claim_id"], {}))
        outputs[claim["claim_id"]] = verdict_engine.synthesize_claim_verdict(request)
    return outputs

FINAL_ACCEPTED_VERDICTS = {"Supported", "Partially Supported"}

def run_closed_loop_batch(fact_rooms: list[dict], bull_agent=None, bear_agent=None) -> list[dict]:
    bull_agent = bull_agent or StructuredClaimAgent("bull")
    bear_agent = bear_agent or StructuredClaimAgent("bear")
    room_by_code = {room["stock_code"]: room for room in fact_rooms}
    claims = []
    for room in fact_rooms:
        bull_view, bear_view = copy.deepcopy(initial_agent_view(room)), copy.deepcopy(initial_agent_view(room))
        assert object_hash(bull_view) == object_hash(bear_view)
        claims.extend(bull_agent.generate_initial_claims(room)); claims.extend(bear_agent.generate_initial_claims(room))
    round0_routes = [claim_router.structured_route_claim(claim, "STEP14-ROUND0") for claim in claims]
    round0_verdicts = synthesize_routed_batch(round0_routes)
    revisions = [build_revision_record(claim, round0_verdicts[claim["claim_id"]], room_by_code[claim["stock_code"]]) for claim in claims]
    revised_claims = [record["revised_claim"] for record in revisions if record["requires_revalidation"]]
    round1_routes = [claim_router.structured_route_claim(claim, "STEP14-ROUND1") for claim in revised_claims]
    round1_verdicts = synthesize_routed_batch(round1_routes) if round1_routes else {}
    results = []
    for room in fact_rooms:
        room_claims = [claim for claim in claims if claim["stock_code"] == room["stock_code"]]
        round0 = [] ; room_revisions = [] ; round1 = [] ; final_claims = [] ; audit = []
        for claim in room_claims:
            routed = next(item for item in round0_routes if item["validated_claim"]["claim_id"] == claim["claim_id"])
            verdict = round0_verdicts[claim["claim_id"]]
            revision = next(item for item in revisions if item["claim_id"] == claim["claim_id"])
            round0.append({"claim": claim, "routing": routed, "verdict": verdict}); room_revisions.append(revision)
            audit.append({"case_id": f"CASE-{room['stock_code']}-{room['as_of_date']}", "agent": claim["source_agent"], "claim_id": claim["claim_id"], "round": 0, "claim_text": claim["atomic_claim_text"], "step12_status": routed["step12_status"], "step13_verdict": verdict["claim_verdict"], "revision_action": revision["revision_action"], "final_status": "pending_round1" if revision["requires_revalidation"] else "kept" if revision["revision_action"] == "KEEP" else "dropped"})
            if revision["revision_action"] == "KEEP": final_claims.append({"claim": claim, "verdict": verdict, "final_status": "kept"})
            elif revision["requires_revalidation"]:
                revised = revision["revised_claim"]
                routed1 = next(item for item in round1_routes if item["validated_claim"]["claim_id"] == revised["claim_id"])
                verdict1 = round1_verdicts[revised["claim_id"]]
                accepted = verdict1["claim_verdict"] in FINAL_ACCEPTED_VERDICTS
                round1.append({"claim": revised, "routing": routed1, "verdict": verdict1, "parent_claim_id": claim["claim_id"]})
                audit.append({"case_id": f"CASE-{room['stock_code']}-{room['as_of_date']}", "agent": revised["source_agent"], "claim_id": revised["claim_id"], "round": 1, "claim_text": revised["atomic_claim_text"], "step12_status": routed1["step12_status"], "step13_verdict": verdict1["claim_verdict"], "revision_action": "FINAL_AFTER_REVALIDATION", "final_status": "kept_revised" if accepted else "dropped_after_revalidation"})
                if accepted: final_claims.append({"claim": revised, "verdict": verdict1, "final_status": "kept_revised"})
        result = {"case_id": f"CASE-{room['stock_code']}-{room['as_of_date']}", "fact_room_id": room["fact_room_id"], "fact_room_hash": room["fact_room_hash"], "max_revision_round": 1, "state": "CLOSED", "initial_claims": room_claims, "round0": round0, "revisions": room_revisions, "round1": round1, "final_claims": final_claims, "evidence_board": [{"agent": item["claim"]["source_agent"], "claim_id": item["claim"]["claim_id"], "claim_text": item["claim"]["atomic_claim_text"], "claim_verdict": item["verdict"]["claim_verdict"], "primary_evidence": [packet for packet in item["verdict"]["evidence_packets"] if packet["role"] in {"primary", "required_prerequisite"}], "auxiliary_context": [packet for packet in item["verdict"]["evidence_packets"] if packet["role"] in {"supporting", "auxiliary", "explanation_only"}], "final_status": item["final_status"]} for item in final_claims], "audit_log": audit}
        result["closed_loop_hash"] = object_hash(result); results.append(result)
    return results

def finance_policy_fixture(room: dict) -> pd.DataFrame:
    blocked = make_claim(room, "bull", 701, "T5", "operating_margin", f"{room['company_name']}의 영업이익률이 전년동기 대비 개선됐다.", "positive", "yoy", rag_topics=["profitability"])
    tests = [("operating_margin_direct_blocked", blocked, False)]
    for index, feature in enumerate(["debt_ratio", "fy_roe", "fy_roa"], start=702):
        history = make_claim(room, "bull", index, "T6", feature, f"{room['company_name']}의 {feature}가 자체 과거 분포에서 높다.", "positive", "history", ">=", 0.5)
        industry = make_claim(room, "bull", index+10, "T6", feature, f"{room['company_name']}의 {feature}가 같은 금융업 비교기업 분포에서 높다.", "positive", "industry", ">=", 0.5)
        tests.extend([(f"{feature}_history_blocked", history, False), (f"{feature}_industry_allowed", industry, True)])
    rows = []
    for name, claim, expected in tests:
        routed = claim_router.structured_route_claim(claim, "STEP14-FINANCE-QA")
        actual = routed["validated_claim"]["validation"]["execution_allowed"]
        rows.append({"check": name, "expected_execution_allowed": expected, "actual_execution_allowed": actual, "pass": actual == expected, "errors": " | ".join(routed["validated_claim"]["validation"]["errors"])})
    try:
        generated = StructuredClaimAgent("bull").generate_initial_claims(room)
        generation_ok = 1 <= len(generated) <= 3
    except Exception:
        generated = []; generation_ok = False
    rows.append({"check": "blocked_candidate_does_not_crash_agent_and_1_to_3_valid_claims", "expected_execution_allowed": True, "actual_execution_allowed": generation_ok, "pass": generation_ok, "errors": f"generated={len(generated)}"})
    return pd.DataFrame(rows)

def weaken_roundtrip_fixture(room: dict) -> dict:
    claim = StructuredClaimAgent("bull").generate_initial_claims(room)[0]
    synthetic_step13 = {"claim_verdict": "Partially Supported"}
    revision = build_revision_record(claim, synthetic_step13, room)
    routed = claim_router.structured_route_claim(revision["revised_claim"], "STEP14-WEAKEN-FIXTURE")
    verdict = synthesize_routed_batch([routed])[revision["revised_claim"]["claim_id"]]
    return {"initial_policy_verdict": "Partially Supported", "revision_action": revision["revision_action"], "requires_revalidation": revision["requires_revalidation"], "round1_reached": True, "round1_verdict": verdict["claim_verdict"], "pass": revision["revision_action"] == "WEAKEN" and revision["requires_revalidation"] and routed["validated_claim"]["validation"]["execution_allowed"]}

UPSTREAM_PROVENANCE = {"STEP12": claim_router.UPSTREAM_PROVENANCE, "EVIDENCE_ADAPTERS": evidence_adapters.UPSTREAM_PROVENANCE, "STEP13": verdict_engine.UPSTREAM_PROVENANCE, "implementation": "deterministic_rule_based_prototype_with_external_LLM_compatible_schema"}
