# Auto-generated from the frozen STEP 13 synthesis implementation. Do not edit by hand.
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
ROUTING_DIR = DATA_DIR / "claim_routing"
SYNTHESIS_DIR = DATA_DIR / "evidence_synthesis"
RAG_DIR = DATA_DIR / "rag_evidence"

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

VERDICT_LABELS = ("Supported", "Partially Supported", "Insufficient Evidence", "Contradicted")
EVIDENCE_STATUSES = ("support", "contradict", "inconclusive", "unavailable")
EVIDENCE_ROLES = ("unassigned", "primary", "required_prerequisite", "supporting", "auxiliary", "explanation_only")
SOURCE_VERDICT_TO_STATUS = {
    "Supported": "support", "Partially Supported": "support",
    "Insufficient Evidence": "inconclusive", "Contradicted": "contradict",
}
CITATION_REQUIRED_FIELDS = ["company_name", "report_name", "effective_date", "section", "rcept_no", "source_url"]

EVIDENCE_PACKET_SCHEMA = json.loads((SYNTHESIS_DIR / "07_evidence_packet_schema.json").read_text(encoding="utf-8"))
CLAIM_VERDICT_OUTPUT_SCHEMA = json.loads((SYNTHESIS_DIR / "07_claim_verdict_output_schema.json").read_text(encoding="utf-8"))
SYNTHESIS_REQUEST_SCHEMA = json.loads((SYNTHESIS_DIR / "09_synthesis_request_schema.json").read_text(encoding="utf-8"))
RAG_ENTAILMENT_SCHEMA = json.loads((SYNTHESIS_DIR / "08_rag_entailment_schema.json").read_text(encoding="utf-8"))
output_schema = json.loads((ROUTING_DIR / "02_claim_output_schema.json").read_text(encoding="utf-8"))
evidence_packet_validator = Draft202012Validator(EVIDENCE_PACKET_SCHEMA)
claim_verdict_validator = Draft202012Validator(CLAIM_VERDICT_OUTPUT_SCHEMA)
synthesis_request_validator = Draft202012Validator(SYNTHESIS_REQUEST_SCHEMA)
rag_entailment_validator = Draft202012Validator(RAG_ENTAILMENT_SCHEMA)
output_validator = Draft202012Validator(output_schema)

RAG_ENTAILMENT_LABELS = ("support", "contradict", "mixed", "inconclusive")
CLAIM_SCOPE_BY_TEMPLATE = {
    "T1": {"historical_relation"}, "T2": {"historical_relation"}, "T3": {"historical_relation"},
    "T4": {"event"}, "T5": {"fact", "broad", "forecast"}, "T6": {"relative"},
    "RAG": {"qualitative", "broad", "forecast"},
}
BROAD_CLAIM_POLICY = json.loads((SYNTHESIS_DIR / "10_broad_claim_policy.json").read_text(encoding="utf-8"))
TEMPORAL_CLAIM_POLICY = json.loads((SYNTHESIS_DIR / "11_temporal_claim_policy.json").read_text(encoding="utf-8"))
RAG_VERDICT_POLICY = json.loads((SYNTHESIS_DIR / "12_rag_verdict_policy.json").read_text(encoding="utf-8"))
CONFLICT_PRIORITY = ["hard_fact_conflict", "same_family_method_conflict", "rag_version_conflict", "cross_family_context_difference", "missing_evidence", "no_conflict"]

ml_adapter_step10 = pd.read_parquet(RAG_DIR / "16_step10_ml_evidence_adapter_2025.parquet")
ml_adapter_step10["stock_code"] = ml_adapter_step10["stock_code"].astype(str).str.zfill(6)
ml_adapter_step10["as_of_date"] = pd.to_datetime(ml_adapter_step10["as_of_date"], errors="coerce")

def status_set_to_base_verdict(statuses: list[str], source_verdicts: list[str]) -> tuple[str, str]:
    status_set = set(statuses)
    if not statuses or status_set.issubset({"inconclusive", "unavailable"}):
        return "Insufficient Evidence", "primary_not_decisive"
    if "support" in status_set and "contradict" in status_set:
        return "Partially Supported", "same_question_primary_conflict"
    if status_set == {"contradict"}:
        return "Contradicted", "primary_contradicts_claim"
    if "support" in status_set:
        if "Partially Supported" in source_verdicts or "inconclusive" in status_set:
            return "Partially Supported", "primary_partial_or_mixed"
        return "Supported", "primary_supports_claim"
    return "Insufficient Evidence", "primary_not_decisive"


def evidence_fingerprint(packet: dict) -> str:
    excluded = {"evidence_id"}
    canonical = {key: value for key, value in packet.items() if key not in excluded}
    return sha256_text(canonical)


def deduplicate_evidence_packets(packets: list[dict]) -> tuple[list[dict], dict]:
    ordered = sorted(copy.deepcopy(packets), key=lambda p: (evidence_fingerprint(p), p["evidence_id"]))
    seen = {}
    unique = []
    removed_ids = []
    for packet in ordered:
        fingerprint = evidence_fingerprint(packet)
        if fingerprint in seen:
            removed_ids.append(packet["evidence_id"])
            continue
        seen[fingerprint] = packet["evidence_id"]
        unique.append(packet)
    return unique, {
        "input_count": len(packets),
        "unique_count": len(unique),
        "removed_duplicate_count": len(removed_ids),
        "removed_evidence_ids": sorted(removed_ids),
    }


def _python_scalar(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _iso_date(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    timestamp = pd.Timestamp(value)
    return str(timestamp.date())


def _citation_is_valid(citation: dict, as_of_date: str) -> bool:
    if not all(str(citation.get(field, "")).strip() for field in CITATION_REQUIRED_FIELDS):
        return False
    try:
        return pd.Timestamp(citation["effective_date"]) <= pd.Timestamp(as_of_date)
    except Exception:
        return False


def make_evidence_packet(
    *, claim_id: str, stock_code: str, as_of_date: str, evidence_family: str, engine: str,
    evidence_status: str, direction: str = "not_applicable", role: str = "unassigned",
    available: bool = True, count_toward_verdict: bool = True, effect=None, ci_low=None,
    ci_high=None, p_value=None, q_value=None, sample_size=None, citations=None,
    source: str, source_date=None, source_ref: str, independence_key: str,
    limitations=None, metadata=None, evidence_id: str | None = None,
) -> dict:
    citations = citations or []
    citation_valid = bool(citations) and all(_citation_is_valid(citation, as_of_date) for citation in citations)
    source_date_iso = _iso_date(source_date)
    pit_safe = source_date_iso is None or pd.Timestamp(source_date_iso) <= pd.Timestamp(as_of_date)
    packet = {
        "evidence_id": evidence_id or "EVD-" + sha256_text({
            "claim_id": claim_id, "engine": engine, "source_ref": source_ref,
            "status": evidence_status, "direction": direction,
        })[:16].upper(),
        "claim_id": claim_id,
        "stock_code": str(stock_code).zfill(6),
        "as_of_date": _iso_date(as_of_date),
        "evidence_family": evidence_family,
        "engine": engine,
        "role": role,
        "evidence_status": evidence_status,
        "direction": direction,
        "available": bool(available),
        "count_toward_verdict": bool(count_toward_verdict),
        "effect": _python_scalar(effect),
        "ci_low": _python_scalar(ci_low),
        "ci_high": _python_scalar(ci_high),
        "p_value": _python_scalar(p_value),
        "q_value": _python_scalar(q_value),
        "sample_size": None if sample_size is None else int(sample_size),
        "citations": citations,
        "citation_valid": citation_valid,
        "source": source,
        "source_date": source_date_iso,
        "pit_safe": bool(pit_safe),
        "source_ref": source_ref,
        "independence_key": independence_key,
        "limitations": list(limitations or []),
        "metadata": dict(metadata or {}),
    }
    errors = list(evidence_packet_validator.iter_errors(packet))
    if errors:
        raise ValueError("Evidence Packet schema error: " + " | ".join(error.message for error in errors))
    return packet


def normalize_step8_evidence(raw: dict, claim_id: str | None = None) -> dict:
    template = raw["template_id"]
    family = "statistical" if template in {"T1", "T2", "T3"} else "event" if template == "T4" else "structured_fundamental" if template == "T5" else "relative"
    ci = raw.get("ci_95") or [None, None]
    source_date = raw.get("analysis_end")
    if template == "T5":
        source_date = raw.get("details", {}).get("effective_date", source_date)
    metadata = {
        "source_verdict": raw["verdict"],
        "source_status": raw.get("status"),
        "x_feature": raw.get("x_feature"),
        "outcome": raw.get("outcome"),
        "primary_inference_method": raw.get("primary_inference_method"),
        "fdr_pass": raw.get("fdr_pass"),
        "semantic_horizon": None if template in {"T5", "T6"} else raw.get("horizon"),
    }
    if template == "T5":
        metadata.update({
            "numeric_key": raw.get("x_feature"),
            "numeric_value": raw.get("details", {}).get("change", raw.get("effect_size")),
            "numeric_source_of_truth": True,
        })
    return make_evidence_packet(
        claim_id=claim_id or raw["claim_id"], stock_code=raw["stock_code"],
        as_of_date=raw["as_of_date"], evidence_family=family, engine=template,
        evidence_status=SOURCE_VERDICT_TO_STATUS[raw["verdict"]],
        direction=raw.get("direction_observed", "not_applicable"),
        available=raw.get("status") != "rejected", effect=raw.get("effect_size"),
        ci_low=ci[0], ci_high=ci[1], p_value=raw.get("p_value"), q_value=raw.get("q_value"),
        sample_size=raw.get("sample_size"), source="step8_claim_validator",
        source_date=source_date, source_ref=f"{raw.get('validation_batch_id')}:{raw['claim_id']}",
        independence_key=f"{family}:{raw['stock_code']}:{raw.get('x_feature')}:{raw.get('outcome')}:{raw['as_of_date']}",
        limitations=raw.get("limitations", []), metadata=metadata,
    )


def normalize_structured_fundamental(raw: dict, claim: dict) -> dict:
    requested_feature = claim["feature"]
    signal_feature = "operating_margin_change_yoy" if requested_feature == "operating_margin" else requested_feature
    feature_payload = raw.get("features", {}).get(signal_feature)
    available = raw.get("fundamental_status") == "success" and feature_payload is not None
    value = None if not available else feature_payload.get("value")
    if not available or value is None or float(value) == 0:
        evidence_status, observed = ("unavailable", "not_applicable") if not available else ("inconclusive", "neutral")
        source_verdict = "Insufficient Evidence"
    else:
        observed = "positive" if float(value) > 0 else "negative"
        evidence_status = "support" if observed == claim["expected_direction"] else "contradict"
        source_verdict = "Supported" if evidence_status == "support" else "Contradicted"
    financial_source = raw.get("financial_source", {})
    return make_evidence_packet(
        claim_id=claim["claim_id"], stock_code=claim["stock_code"], as_of_date=claim["as_of_date"],
        evidence_family="structured_fundamental", engine="T5", evidence_status=evidence_status,
        direction=observed, available=available, effect=value, sample_size=1 if available else 0,
        source="step11_structured_fundamental", source_date=financial_source.get("effective_date"),
        source_ref=str(financial_source.get("rcept_no") or "structured_fundamental_missing"),
        independence_key=f"structured_fundamental:{claim['stock_code']}:{signal_feature}:{claim['as_of_date']}",
        limitations=["PIT fact verification; no forecast authority"],
        metadata={
            "source_verdict": source_verdict, "numeric_key": signal_feature,
            "numeric_value": value, "unit": None if feature_payload is None else feature_payload.get("unit"),
            "numeric_source_of_truth": True,
        },
    )


def _extract_rag_entries(raw: dict) -> tuple[list[dict], str]:
    if isinstance(raw.get("rag_evidence"), list):
        entries = raw.get("rag_evidence", [])
        retrieval_status = raw.get("retrieval_status", "success")
    elif isinstance(raw.get("rag_evidence"), dict):
        entries = raw["rag_evidence"].get("evidence", [])
        retrieval_status = raw["rag_evidence"].get("retrieval_status", "success")
    else:
        entries = raw.get("evidence", [])
        retrieval_status = raw.get("retrieval_status", "no_document_available")
    normalized_entries = []
    for index, item in enumerate(entries, start=1):
        normalized = dict(item)
        normalized["chunk_id"] = str(item.get("chunk_id") or f"{item.get('rcept_no', 'unknown')}:{index}")
        normalized_entries.append(normalized)
    return normalized_entries, retrieval_status


def _entailment_tokens(text: str) -> set[str]:
    stop = {"회사는", "회사가", "당사는", "최근", "대한", "관련", "무엇", "어떻게", "설명", "설명했다", "것으로"}
    return {
        token for token in re.findall(r"[가-힣A-Za-z0-9]+", unicodedata.normalize("NFKC", str(text)).lower())
        if len(token) >= 2 and token not in stop
    }


def _compact_text(text: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", unicodedata.normalize("NFKC", str(text)).lower())


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = _compact_text(text)
    return {compact[index:index+n] for index in range(max(0, len(compact) - n + 1))}


def _negation_signature(text: str) -> bool:
    normalized = _compact_text(text)
    return any(term in normalized for term in ("아니다", "않았다", "않는다", "없다", "부인", "중단", "취소"))


def _lexical_entailment_score(claim_text: str, excerpt: str) -> float:
    compact_claim = _compact_text(claim_text)
    compact_excerpt = _compact_text(excerpt)
    if compact_claim and compact_claim in compact_excerpt:
        return 1.0
    claim_tokens = _entailment_tokens(claim_text)
    excerpt_tokens = _entailment_tokens(excerpt)
    overlap = len(claim_tokens & excerpt_tokens) / max(1, len(claim_tokens))
    claim_grams = _char_ngrams(claim_text)
    excerpt_grams = _char_ngrams(excerpt)
    char_jaccard = len(claim_grams & excerpt_grams) / max(1, len(claim_grams | excerpt_grams))
    return float(0.75 * overlap + 0.25 * char_jaccard)


def classify_rag_entailment(claim: dict, raw: dict, external_classifier=None) -> dict:
    entries, retrieval_status = _extract_rag_entries(raw)
    eligible = [
        item for item in entries
        if item.get("source_excerpt") or item.get("summary")
        if all(str(item.get(field, "")).strip() for field in CITATION_REQUIRED_FIELDS)
        if pd.Timestamp(item["effective_date"]) <= pd.Timestamp(claim["as_of_date"])
    ]
    eligible_ids = {item["chunk_id"] for item in eligible}
    if retrieval_status != "success" or not eligible:
        return {
            "entailment": "inconclusive", "evidence_chunk_ids": [],
            "reason": "PIT-safe하고 Citation 필수필드를 갖춘 검색 근거가 없음",
            "classifier_mode": "deterministic_conservative",
        }

    if external_classifier is not None:
        payload = {
            "task": "citation_entailment_only_not_final_claim_verdict",
            "allowed_labels": list(RAG_ENTAILMENT_LABELS),
            "claim": claim["atomic_claim_text"],
            "citations": [
                {"chunk_id": item["chunk_id"], "source_excerpt": item.get("source_excerpt") or item.get("summary")}
                for item in eligible
            ],
        }
        result = dict(external_classifier(payload))
        result["classifier_mode"] = "external_constrained_model"
    else:
        support_ids, contradict_ids = [], []
        claim_negative = _negation_signature(claim["atomic_claim_text"])
        for item in eligible:
            excerpt = item.get("source_excerpt") or item.get("summary") or ""
            score = _lexical_entailment_score(claim["atomic_claim_text"], excerpt)
            if score < 0.72:
                continue
            if _negation_signature(excerpt) == claim_negative:
                support_ids.append(item["chunk_id"])
            else:
                contradict_ids.append(item["chunk_id"])
        if support_ids and contradict_ids:
            label, selected = "mixed", sorted(set(support_ids + contradict_ids))
        elif support_ids:
            label, selected = "support", sorted(set(support_ids))
        elif contradict_ids:
            label, selected = "contradict", sorted(set(contradict_ids))
        else:
            label, selected = "inconclusive", []
        result = {
            "entailment": label,
            "evidence_chunk_ids": selected,
            "reason": "보수적 문자열 포함·내용어 중첩·부정 표현 일치 규칙으로 Citation과 Claim의 관계를 분류",
            "classifier_mode": "deterministic_conservative",
        }

    errors = list(rag_entailment_validator.iter_errors(result))
    if errors:
        raise ValueError("RAG entailment schema error: " + " | ".join(error.message for error in errors))
    selected_ids = set(result["evidence_chunk_ids"])
    if not selected_ids.issubset(eligible_ids):
        raise ValueError("RAG entailment selected an unknown or PIT-unsafe chunk_id")
    if result["entailment"] != "inconclusive" and not selected_ids:
        raise ValueError("Decisive RAG entailment requires at least one evidence_chunk_id")
    return result


def normalize_rag_evidence(raw: dict, claim: dict, external_classifier=None) -> dict:
    entries, retrieval_status = _extract_rag_entries(raw)
    classification = classify_rag_entailment(claim, raw, external_classifier=external_classifier)
    selected_ids = set(classification["evidence_chunk_ids"])
    selected_entries = [item for item in entries if item["chunk_id"] in selected_ids]
    # inconclusive일 때 Citation은 추적용으로 보존하되 Verdict 권한은 부여하지 않는다.
    citation_entries = selected_entries if selected_entries else [
        item for item in entries
        if all(str(item.get(field, "")).strip() for field in CITATION_REQUIRED_FIELDS)
        and pd.Timestamp(item["effective_date"]) <= pd.Timestamp(claim["as_of_date"])
    ]
    citations = [{
        "company_name": str(item.get("company_name", "")),
        "report_name": str(item.get("report_name", "")),
        "effective_date": _iso_date(item.get("effective_date")) or "",
        "section": str(item.get("section") or item.get("section_path") or ""),
        "rcept_no": str(item.get("rcept_no", "")),
        "source_url": str(item.get("source_url", "")),
        "chunk_id": item.get("chunk_id"),
        "source_excerpt": item.get("source_excerpt") or item.get("summary"),
    } for item in citation_entries]
    citation_valid = bool(citations) and all(_citation_is_valid(citation, claim["as_of_date"]) for citation in citations)
    entailment = classification["entailment"]
    if retrieval_status in {"no_document_available", "parse_failure"}:
        status, available, source_verdict = "unavailable", False, "Insufficient Evidence"
    elif entailment == "support" and citation_valid:
        status, available, source_verdict = "support", True, "Supported"
    elif entailment == "contradict" and citation_valid:
        status, available, source_verdict = "contradict", True, "Contradicted"
    elif entailment == "mixed" and citation_valid:
        status, available, source_verdict = "support", True, "Partially Supported"
    else:
        status, available, source_verdict = "inconclusive", retrieval_status == "success", "Insufficient Evidence"
    source_dates = [pd.Timestamp(c["effective_date"]) for c in citations if c.get("effective_date")]
    source_date = max(source_dates) if source_dates else None
    return make_evidence_packet(
        claim_id=claim["claim_id"], stock_code=claim["stock_code"], as_of_date=claim["as_of_date"],
        evidence_family="rag", engine="RAG", evidence_status=status, direction="not_applicable",
        available=available, citations=citations, source="step11_frozen_rag",
        source_date=source_date, source_ref=(citations[0]["rcept_no"] if citations else retrieval_status),
        independence_key=f"rag:{claim['stock_code']}:{'|'.join(claim.get('rag_topics', []))}:{claim['as_of_date']}",
        limitations=["Official disclosure context; numeric source of truth remains structured fundamental"],
        metadata={
            "source_verdict": source_verdict, "entailment": entailment,
            "entailment_reason": classification["reason"],
            "entailment_classifier_mode": classification["classifier_mode"],
            "evidence_chunk_ids": classification["evidence_chunk_ids"],
            "retrieval_status": retrieval_status,
        },
    )


def normalize_ml_evidence(row, claim: dict) -> list[dict]:
    if row is None:
        return [make_unavailable_evidence(claim, "ML", "ml_evidence_unavailable_for_asof")]
    record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    state = str(record.get("predicted_state"))
    direction = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}.get(state, "neutral")
    if direction == "neutral":
        status = "inconclusive"
    else:
        status = "support" if direction == claim.get("expected_direction") else "contradict"
    key = f"ml:{record.get('model_version')}:{claim['stock_code']}:{_iso_date(record.get('as_of_date'))}"
    ml_packet = make_evidence_packet(
        claim_id=claim["claim_id"], stock_code=claim["stock_code"], as_of_date=claim["as_of_date"],
        evidence_family="ml", engine="ML", evidence_status=status, direction=direction,
        source="step10_frozen_catboost", source_date=record.get("as_of_date"),
        source_ref=str(record.get("model_version")), independence_key=key,
        limitations=["Historical response-state similarity; not a return probability or recommendation"],
        metadata={
            "source_verdict": "context_only", "predicted_state": state,
            "prediction_confidence": _python_scalar(record.get("prediction_confidence")),
            "probability_semantics": record.get("probability_semantics"),
            "label_policy": record.get("label_policy"),
        },
    )
    shap_packet = make_evidence_packet(
        claim_id=claim["claim_id"], stock_code=claim["stock_code"], as_of_date=claim["as_of_date"],
        evidence_family="shap", engine="SHAP", evidence_status=status, direction=direction,
        count_toward_verdict=False, source="step10_shap_explanation", source_date=record.get("as_of_date"),
        source_ref=str(record.get("model_version")) + ":shap", independence_key=key,
        limitations=["SHAP explains the ML output and is not independent Evidence"],
        metadata={
            "source_verdict": "explanation_only",
            "top_positive_feature": record.get("top_positive_feature"),
            "top_positive_shap": _python_scalar(record.get("top_positive_shap")),
            "top_negative_feature": record.get("top_negative_feature"),
            "top_negative_shap": _python_scalar(record.get("top_negative_shap")),
            "shap_top_features": record.get("shap_top_features"),
        },
    )
    return [ml_packet, shap_packet]


def make_unavailable_evidence(claim: dict, engine: str, reason: str) -> dict:
    family = "ml" if engine == "ML" else "rag" if engine == "RAG" else "structured_fundamental" if engine == "T5" else "relative" if engine == "T6" else "event" if engine == "T4" else "statistical"
    return make_evidence_packet(
        claim_id=claim["claim_id"], stock_code=claim["stock_code"], as_of_date=claim["as_of_date"],
        evidence_family=family, engine=engine, evidence_status="unavailable", direction="not_applicable",
        available=False, count_toward_verdict=False, source="route_availability_guard", source_date=None,
        source_ref=reason, independence_key=f"unavailable:{engine}:{claim['claim_id']}",
        limitations=[reason], metadata={"source_verdict": "Insufficient Evidence", "unavailable_reason": reason},
    )


def validate_step12_structured_claim(claim: dict) -> None:
    schema_errors = list(output_validator.iter_errors(claim))
    if schema_errors:
        raise ValueError("STEP 12 structured claim schema error: " + " | ".join(e.message for e in schema_errors))
    if claim.get("source_agent") not in {"bull", "bear"}:
        raise ValueError("MVP service path accepts Bull/Bear structured claims only")
    validation = claim.get("validation", {})
    if not validation.get("schema_valid") or not validation.get("execution_allowed"):
        raise ValueError("STEP 12 deterministic validation must pass before STEP 13")


def validate_synthesis_request(request: dict) -> None:
    errors = list(synthesis_request_validator.iter_errors(request))
    if errors:
        raise ValueError("Synthesis request schema error: " + " | ".join(e.message for e in errors))
    claim = request["validated_claim"]
    validate_step12_structured_claim(claim)
    if request["claim_scope"] not in CLAIM_SCOPE_BY_TEMPLATE[claim["template_id"]]:
        raise ValueError("claim_scope is not allowed for the validated template")
    if any(packet["claim_id"] != claim["claim_id"] for packet in request["evidence_packets"]):
        raise ValueError("Every Evidence Packet must point to the same claim_id")


def assign_evidence_role(packet: dict, claim: dict, claim_scope: str) -> tuple[str, bool]:
    engine = packet["engine"]
    template = claim["template_id"]
    if engine == "SHAP":
        return "explanation_only", False
    if engine == "ML":
        return "auxiliary", False
    if claim_scope in {"broad", "forecast"}:
        if engine == "T5":
            return "required_prerequisite", True
        if engine == "RAG":
            return "primary", True
    if engine == template:
        return "primary", True
    if engine == "RAG":
        return "supporting", True
    if engine in {"T1", "T2", "T3", "T4", "T5", "T6"}:
        # 같은 데이터/질문의 다른 방법은 보존하되 별도 표로 세지 않는다.
        return "supporting", False
    return "supporting", False


def prepare_evidence_packets(request: dict) -> tuple[list[dict], dict]:
    validate_synthesis_request(request)
    claim = request["validated_claim"]
    prepared = []
    for original in request["evidence_packets"]:
        packet = copy.deepcopy(original)
        role, role_countable = assign_evidence_role(packet, claim, request["claim_scope"])
        packet["role"] = role
        packet["count_toward_verdict"] = bool(packet["count_toward_verdict"] and role_countable)
        if packet["count_toward_verdict"] and packet["source_date"] is None:
            packet["evidence_status"] = "unavailable"
            packet["available"] = False
            packet["count_toward_verdict"] = False
            packet["limitations"] = sorted(set(packet["limitations"] + ["countable_evidence_requires_source_date"]))
        if not packet["pit_safe"]:
            packet["evidence_status"] = "unavailable"
            packet["available"] = False
            packet["count_toward_verdict"] = False
            packet["limitations"] = sorted(set(packet["limitations"] + ["future_evidence_rejected_by_PIT_guard"]))
        if packet["engine"] == "RAG" and packet["evidence_status"] in {"support", "contradict"} and not packet["citation_valid"]:
            packet["evidence_status"] = "inconclusive"
            packet["count_toward_verdict"] = False
            packet["limitations"] = sorted(set(packet["limitations"] + ["citation_required_for_RAG_verdict"]))
        if packet["evidence_status"] == "unavailable":
            packet["available"] = False
            packet["count_toward_verdict"] = False
        if packet["engine"] == "SHAP":
            packet["count_toward_verdict"] = False
        errors = list(evidence_packet_validator.iter_errors(packet))
        if errors:
            raise ValueError("Prepared Evidence schema error: " + " | ".join(e.message for e in errors))
        prepared.append(packet)
    return deduplicate_evidence_packets(prepared)


def latest_ml_row_for_claim(claim: dict):
    eligible = ml_adapter_step10.loc[
        ml_adapter_step10["stock_code"].eq(claim["stock_code"])
        & ml_adapter_step10["as_of_date"].le(pd.Timestamp(claim["as_of_date"]))
    ].sort_values("as_of_date")
    return None if eligible.empty else eligible.iloc[-1]


def collect_evidence_for_claim(claim: dict, claim_scope: str, raw_by_engine: dict) -> dict:
    validate_step12_structured_claim(claim)
    packets = []
    for route in claim["routes"]:
        engine = route["engine"]
        if not route["execution_allowed"]:
            packets.append(make_unavailable_evidence(claim, engine, "|".join(route.get("route_errors", [])) or "route_unavailable"))
            continue
        raw = raw_by_engine.get(engine)
        if engine in {"T1", "T2", "T3", "T4", "T5", "T6"} and raw is not None:
            if engine == "T5" and "fundamental_status" in raw:
                packets.append(normalize_structured_fundamental(raw, claim))
            else:
                packets.append(normalize_step8_evidence(raw, claim_id=claim["claim_id"]))
        elif engine == "RAG" and raw is not None:
            packets.append(normalize_rag_evidence(
                raw, claim, external_classifier=raw_by_engine.get("RAG_ENTAILMENT_CLASSIFIER")
            ))
        elif engine == "ML":
            packets.extend(normalize_ml_evidence(raw if raw is not None else latest_ml_row_for_claim(claim), claim))
        else:
            packets.append(make_unavailable_evidence(claim, engine, "evidence_output_not_supplied"))
    return {"request_id": "SYN-" + claim["claim_id"], "validated_claim": claim, "claim_scope": claim_scope, "evidence_packets": packets}


def rag_packet_has_verdict_authority(packet: dict) -> bool:
    return bool(
        packet["engine"] == "RAG"
        and packet["pit_safe"]
        and packet["citation_valid"]
        and packet["evidence_status"] in {"support", "contradict"}
    )


def detect_evidence_conflicts(packets: list[dict]) -> tuple[str, list[dict]]:
    conflicts = []
    primary_like = [p for p in packets if p["role"] in {"primary", "required_prerequisite"}]
    by_key = defaultdict(list)
    for packet in primary_like:
        if packet["count_toward_verdict"]:
            by_key[packet["independence_key"]].append(packet)
    for key, group in by_key.items():
        statuses = {p["evidence_status"] for p in group}
        if {"support", "contradict"}.issubset(statuses):
            conflicts.append({
                "type": "same_family_method_conflict", "severity": "verdict_relevant",
                "affects_verdict": True, "evidence_ids": sorted(p["evidence_id"] for p in group),
                "reason": "동일 질문·Evidence Family의 결과 방향이 충돌",
            })

    fundamentals = [p for p in primary_like if p["evidence_family"] == "structured_fundamental" and p["evidence_status"] in {"support", "contradict"}]
    rags = [p for p in packets if p["evidence_family"] == "rag" and p["evidence_status"] in {"support", "contradict"}]
    for fundamental in fundamentals:
        for rag in rags:
            if fundamental["evidence_status"] == rag["evidence_status"]:
                continue
            f_key = fundamental["metadata"].get("numeric_key")
            r_key = rag["metadata"].get("numeric_key")
            if f_key and r_key and f_key != r_key:
                continue
            rag_is_old = bool(rag["source_date"] and fundamental["source_date"] and pd.Timestamp(rag["source_date"]) < pd.Timestamp(fundamental["source_date"]))
            conflict_type = "rag_version_conflict" if rag_is_old else "hard_fact_conflict"
            conflicts.append({
                "type": conflict_type, "severity": "source_priority_resolved",
                "affects_verdict": False, "winner": fundamental["evidence_id"],
                "evidence_ids": sorted([fundamental["evidence_id"], rag["evidence_id"]]),
                "reason": "Structured PIT Fundamental이 RAG 숫자/서술보다 우선",
            })

    statistical_primary = [p for p in primary_like if p["evidence_family"] in {"statistical", "event"} and p["evidence_status"] in {"support", "contradict"}]
    contextual = [p for p in packets if p["evidence_family"] in {"ml", "rag"} and p["evidence_status"] in {"support", "contradict"}]
    for primary in statistical_primary:
        for context in contextual:
            if primary["evidence_status"] != context["evidence_status"]:
                conflicts.append({
                    "type": "cross_family_context_difference", "severity": "display_only",
                    "affects_verdict": False,
                    "evidence_ids": sorted([primary["evidence_id"], context["evidence_id"]]),
                    "reason": "과거 단변량/이벤트 관계와 현재 복합상태·공시 맥락은 서로 다른 질문",
                })

    unavailable = [p for p in packets if p["evidence_status"] == "unavailable"]
    if unavailable:
        conflicts.append({
            "type": "missing_evidence", "severity": "availability",
            "affects_verdict": False, "evidence_ids": sorted(p["evidence_id"] for p in unavailable),
            "reason": "일부 Route의 시점 적합 Evidence가 없음",
        })
    if not [p for p in primary_like if p["count_toward_verdict"]]:
        conflicts.append({
            "type": "missing_evidence", "severity": "primary_missing",
            "affects_verdict": False, "evidence_ids": [],
            "reason": "판정 권한이 있는 Primary Evidence가 없음",
        })
    if not conflicts:
        return "no_conflict", []
    present = {item["type"] for item in conflicts}
    status = next(item for item in CONFLICT_PRIORITY if item in present)
    unique = {sha256_text(item): item for item in conflicts}
    return status, sorted(unique.values(), key=lambda item: (CONFLICT_PRIORITY.index(item["type"]), stable_json_dumps(item)))


def build_synthesis_points(verdict: str, primary_basis: list[dict], conflict_status: str, packets: list[dict], claim_scope: str) -> list[str]:
    if primary_basis:
        basis_text = ", ".join(f"{item['engine']}={item['source_verdict']}" for item in primary_basis)
        points = [f"Primary Evidence 판정: {basis_text}."]
    else:
        points = ["판정 가능한 Primary Evidence가 없습니다."]
    points.append(f"Claim Verdict는 {verdict}입니다.")
    if any(packet["engine"] == "ML" for packet in packets):
        points.append("ML은 현재 복합 시장환경의 보조 맥락이며 Claim Verdict를 변경하지 않습니다.")
    if conflict_status != "no_conflict":
        points.append(f"Evidence 상태: {conflict_status}.")
    if claim_scope in {"broad", "forecast"}:
        points.append("미래 지속성 표현에는 Partially Supported 상한을 적용했습니다.")
    return points


def synthesize_claim_verdict(request: dict) -> dict:
    packets, deduplication = prepare_evidence_packets(request)
    claim = request["validated_claim"]
    claim_scope = request["claim_scope"]
    primary_all = [p for p in packets if p["role"] == "primary"]
    prerequisite_all = [p for p in packets if p["role"] == "required_prerequisite"]
    primary_countable = [p for p in primary_all if p["count_toward_verdict"]]
    prerequisite_countable = [p for p in prerequisite_all if p["count_toward_verdict"]]
    prerequisite_required = claim_scope in {"broad", "forecast"}
    required_prerequisite_absent = prerequisite_required and not prerequisite_all

    if required_prerequisite_absent:
        verdict, reason = "Insufficient Evidence", "required_factual_prerequisite_missing"
    elif prerequisite_all and (
        len(prerequisite_countable) != len(prerequisite_all)
        or any(p["evidence_status"] in {"inconclusive", "unavailable"} for p in prerequisite_all)
    ):
        verdict, reason = "Insufficient Evidence", "required_factual_prerequisite_missing"
    elif any(p["evidence_status"] == "contradict" for p in prerequisite_countable):
        verdict, reason = "Contradicted", "required_factual_prerequisite_contradicted"
    else:
        statuses = [p["evidence_status"] for p in primary_countable]
        source_verdicts = [str(p["metadata"].get("source_verdict", "")) for p in primary_countable]
        verdict, reason = status_set_to_base_verdict(statuses, source_verdicts)

    conflict_status, conflicts = detect_evidence_conflicts(packets)
    if required_prerequisite_absent:
        conflicts.append({
            "type": "missing_evidence", "severity": "required_prerequisite_missing",
            "affects_verdict": True, "evidence_ids": [],
            "reason": "Broad/Forecast Claim에 필수인 T5 factual prerequisite가 전달되지 않음",
        })
        if conflict_status == "no_conflict":
            conflict_status = "missing_evidence"
    if verdict == "Supported" and any(item.get("affects_verdict") for item in conflicts):
        verdict, reason = "Partially Supported", "primary_supported_with_same_question_conflict"
    if claim_scope in {"broad", "forecast"} and verdict == "Supported":
        verdict, reason = "Partially Supported", "forecast_or_broad_claim_cap"

    primary_basis = sorted([
        {
            "evidence_id": p["evidence_id"], "engine": p["engine"], "role": p["role"],
            "evidence_status": p["evidence_status"],
            "source_verdict": p["metadata"].get("source_verdict", "Insufficient Evidence"),
            "source": p["source"],
        }
        for p in packets if p["role"] in {"primary", "required_prerequisite"}
    ], key=lambda item: (item["role"], item["engine"], item["evidence_id"]))
    missing_evidence = sorted(p["evidence_id"] for p in packets if p["evidence_status"] == "unavailable")
    if required_prerequisite_absent:
        missing_evidence = sorted(set(missing_evidence + ["required_factual_prerequisite:T5"]))
    if not primary_countable:
        missing_evidence = sorted(set(missing_evidence + ["primary_evidence_missing_or_inconclusive"]))

    limitations = [
        "Evidence Family별 판정 권한을 적용했으며 다수결하지 않음",
        "ML/SHAP는 Claim truth를 결정하지 않음",
    ]
    if claim_scope in {"broad", "forecast"}:
        limitations.append("미래 지속성은 공식 공시와 현재 사실만으로 확정할 수 없음")
    if conflict_status in {"hard_fact_conflict", "rag_version_conflict"}:
        limitations.append("정량 충돌에서는 최신 PIT Structured Fundamental을 우선함")

    result = {
        "claim_id": claim["claim_id"],
        "stock_code": claim["stock_code"],
        "as_of_date": claim["as_of_date"],
        "claim_text": claim["atomic_claim_text"],
        "claim_scope": claim_scope,
        "template_id": claim["template_id"],
        "claim_verdict": verdict,
        "verdict_reason_code": reason,
        "primary_basis": primary_basis,
        "evidence_packets": sorted(packets, key=lambda p: (p["role"], p["engine"], p["evidence_id"])),
        "supporting_evidence_ids": sorted(p["evidence_id"] for p in packets if p["role"] == "supporting"),
        "auxiliary_evidence_ids": sorted(p["evidence_id"] for p in packets if p["role"] == "auxiliary"),
        "explanation_evidence_ids": sorted(p["evidence_id"] for p in packets if p["role"] == "explanation_only"),
        "conflict_status": conflict_status,
        "conflicts": conflicts,
        "missing_evidence": missing_evidence,
        "limitations": sorted(set(limitations)),
        "partial_route_failure": bool(claim.get("validation", {}).get("partial_route_failure") or missing_evidence),
        "deduplication": deduplication,
        "synthesis_points": [],
        "agent_authority": "claim_generation_only_no_verdict_authority",
        "deterministic_policy_version": "step13_verdict_v1",
        "deterministic_verdict_hash": "0" * 64,
        "llm_explanation": None,
    }
    result["synthesis_points"] = build_synthesis_points(verdict, primary_basis, conflict_status, packets, claim_scope)
    hash_payload = {key: value for key, value in result.items() if key not in {"deterministic_verdict_hash", "llm_explanation"}}
    result["deterministic_verdict_hash"] = sha256_text(hash_payload)
    errors = list(claim_verdict_validator.iter_errors(result))
    if errors:
        raise ValueError("Claim Verdict output schema error: " + " | ".join(error.message for error in errors))
    return result


UPSTREAM_PROVENANCE = {
    "source_notebook": str(PROJECT_ROOT / "step12_13.ipynb"),
    "source_notebook_sha256": "2d42d86f19e3e8c5527624626f5d225d20650da82aa18d8090fa587c4724ab66",
    "function_names": ['status_set_to_base_verdict', 'evidence_fingerprint', 'deduplicate_evidence_packets', '_python_scalar', '_iso_date', '_citation_is_valid', 'make_evidence_packet', 'normalize_step8_evidence', 'normalize_structured_fundamental', '_extract_rag_entries', '_entailment_tokens', '_compact_text', '_char_ngrams', '_negation_signature', '_lexical_entailment_score', 'classify_rag_entailment', 'normalize_rag_evidence', 'normalize_ml_evidence', 'make_unavailable_evidence', 'validate_step12_structured_claim', 'validate_synthesis_request', 'assign_evidence_role', 'prepare_evidence_packets', 'latest_ml_row_for_claim', 'collect_evidence_for_claim', 'rag_packet_has_verdict_authority', 'detect_evidence_conflicts', 'build_synthesis_points', 'synthesize_claim_verdict'],
    "implementation": "exact_function_sources_extracted_from_frozen_step13",
}
