"""Evidence-locked post-validation Bull/Bear debate and neutral summary."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

from src.config.settings import Settings
from src.service.presentation import feature_meaning, friendly_feature, internal_display_tokens

from .llm_client import GoogleStructuredClaimClient, LLMUnavailable
from .prompts import build_neutral_summary_prompt, build_role_rebuttal_prompt


FORBIDDEN_TEXT = re.compile(
    r"(?:매수|매도|투자\s*권유|목표\s*(?:주가|가격)|strong\s*buy|\bbuy\b|\bsell\b|수익(?:률)?\s*(?:보장|확정)|반드시\s*(?:상승|하락))",
    re.IGNORECASE,
)
NUMBER_TOKEN = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?%?")


def _statement_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "evidence_ids", "claim_ids"],
        "properties": {
            "text": {"type": "string", "minLength": 5},
            "evidence_ids": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string"},
            },
            "claim_ids": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string"},
            },
        },
    }


def role_response_schema(role: str) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["role", "statements", "withdrawn_claim_ids", "limitations"],
        "properties": {
            "role": {"type": "string", "enum": [role]},
            "statements": {
                "type": "array", "minItems": 1, "maxItems": 1,
                "items": _statement_schema(),
            },
            "withdrawn_claim_ids": {
                "type": "array", "uniqueItems": True,
                "items": {"type": "string"},
            },
            "limitations": {
                "type": "array", "maxItems": 4,
                "items": {"type": "string"},
            },
        },
    }


def neutral_summary_schema() -> dict:
    withdrawal = {
        "type": "object",
        "additionalProperties": False,
        "required": ["claim_id", "reason", "evidence_ids"],
        "properties": {
            "claim_id": {"type": "string"},
            "reason": {"type": "string", "minLength": 3},
            "evidence_ids": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string"},
            },
        },
    }
    hypothesis = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "title", "hypothesis", "metrics", "evidence_status",
            "evidence_ids", "claim_ids",
        ],
        "properties": {
            "title": {"type": "string", "minLength": 3},
            "hypothesis": {"type": "string", "minLength": 5},
            "metrics": {
                "type": "array", "minItems": 1, "maxItems": 4, "uniqueItems": True,
                "items": {"type": "string"},
            },
            "evidence_status": {
                "type": "string",
                "enum": ["Supported", "Partially Supported", "Insufficient Evidence", "Contradicted"],
            },
            "evidence_ids": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string"},
            },
            "claim_ids": {
                "type": "array", "minItems": 1, "maxItems": 1, "uniqueItems": True,
                "items": {"type": "string"},
            },
        },
    }
    summary = {
        "type": "object",
        "additionalProperties": False,
        "required": ["key_issues", "confirmed_evidence", "partially_supported", "uncertainties"],
        "properties": {
            key: {"type": "array", "maxItems": 5, "items": _statement_schema()}
            for key in ("key_issues", "confirmed_evidence", "partially_supported", "uncertainties")
        },
    }
    revised = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bull", "bear"],
        "properties": {
            role: {"type": "array", "maxItems": 3, "items": hypothesis}
            for role in ("bull", "bear")
        },
    }
    closing_exchange = {
        "type": "object",
        "additionalProperties": False,
        "required": ["bull", "bear"],
        "properties": {
            "bull": _statement_schema(),
            "bear": _statement_schema(),
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary", "closing_exchange", "revised_hypotheses", "withdrawn_arguments",
            "additional_information_needed", "uncertainty_statement",
        ],
        "properties": {
            "summary": summary,
            "closing_exchange": closing_exchange,
            "revised_hypotheses": revised,
            "withdrawn_arguments": {"type": "array", "maxItems": 6, "items": withdrawal},
            "additional_information_needed": {
                "type": "array", "maxItems": 4, "items": {"type": "string"},
            },
            "uncertainty_statement": {"type": "string", "minLength": 5},
        },
    }


def seal_evidence_packet(payload: dict) -> dict:
    packet = dict(payload)
    canonical = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    packet["evidence_packet_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return packet


def _allowed_numbers(packet: dict) -> set[str]:
    allowed: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            number = float(value)
            allowed.update({str(value), f"{number:g}", f"{number:.1f}", f"{number:.2f}", f"{number:.3f}"})
            if abs(number) <= 1:
                allowed.update({f"{number * 100:g}%", f"{number * 100:.1f}%", f"{number * 100:.2f}%"})
            return
        if isinstance(value, str):
            allowed.update(NUMBER_TOKEN.findall(value.replace(",", "")))
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, list):
            for child in value:
                visit(child)

    visit(packet)
    return {token.lstrip("+") for token in allowed}


def _validate_text(text: str, allowed_numbers: set[str]) -> None:
    if FORBIDDEN_TEXT.search(text):
        raise ValueError("post-validation output contains a forbidden investment expression")
    unknown = {
        token.lstrip("+")
        for token in NUMBER_TOKEN.findall(text.replace(",", ""))
        if token.lstrip("+") not in allowed_numbers
    }
    if unknown:
        raise ValueError("post-validation output contains ungrounded numeric facts")


def _validate_no_internal_tokens(text: str, packet: dict) -> None:
    lowered = str(text).lower()
    for token in internal_display_tokens(packet):
        if re.search(rf"(?<![a-z0-9_]){re.escape(token.lower())}(?![a-z0-9_])", lowered):
            raise ValueError("post-validation output exposes an internal feature or engine name")


def _packet_sets(packet: dict) -> tuple[set[str], set[str]]:
    evidence_ids = {item["evidence_id"] for item in packet["evidence"]}
    claim_ids = {item["claim_id"] for item in packet["claims"]}
    claim_ids.update(
        item["final_claim_id"] for item in packet["claims"] if item.get("final_claim_id")
    )
    return evidence_ids, claim_ids


def validate_role_response(response: dict, packet: dict, role: str) -> None:
    if not isinstance(response, dict) or response.get("role") != role:
        raise ValueError("role response mismatch")
    if set(response) != {"role", "statements", "withdrawn_claim_ids", "limitations"}:
        raise ValueError("role response contains unsupported fields")
    if len(response.get("statements", [])) != 1:
        raise ValueError("each debate API must return exactly one direct response")
    evidence_ids, claim_ids = _packet_sets(packet)
    allowed_numbers = _allowed_numbers(packet)
    if not set(response.get("withdrawn_claim_ids", [])).issubset(claim_ids):
        raise ValueError("role response references an unknown withdrawn Claim ID")
    for item in response.get("statements", []):
        if set(item) != {"text", "evidence_ids", "claim_ids"}:
            raise ValueError("debate statement contains unsupported fields")
        if not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids):
            raise ValueError("debate statement references an unknown Evidence ID")
        if not item["claim_ids"] or not set(item["claim_ids"]).issubset(claim_ids):
            raise ValueError("debate statement references an unknown Claim ID")
        _validate_text(str(item["text"]), allowed_numbers)
        _validate_no_internal_tokens(str(item["text"]), packet)
    for text in response.get("limitations", []):
        _validate_text(str(text), allowed_numbers)
        _validate_no_internal_tokens(str(text), packet)


def validate_neutral_summary(response: dict, packet: dict) -> None:
    expected = {
        "summary", "closing_exchange", "revised_hypotheses", "withdrawn_arguments",
        "additional_information_needed", "uncertainty_statement",
    }
    if not isinstance(response, dict) or set(response) != expected:
        raise ValueError("neutral response contains unsupported fields")
    evidence_ids, claim_ids = _packet_sets(packet)
    allowed_numbers = _allowed_numbers(packet)
    summary_keys = {"key_issues", "confirmed_evidence", "partially_supported", "uncertainties"}
    if set(response["summary"]) != summary_keys:
        raise ValueError("neutral summary groups mismatch")
    for group in summary_keys:
        for item in response["summary"][group]:
            if set(item) != {"text", "evidence_ids", "claim_ids"}:
                raise ValueError("neutral statement contains unsupported fields")
            if not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids):
                raise ValueError("neutral statement references an unknown Evidence ID")
            if not item["claim_ids"] or not set(item["claim_ids"]).issubset(claim_ids):
                raise ValueError("neutral statement references an unknown Claim ID")
            _validate_text(str(item["text"]), allowed_numbers)
            _validate_no_internal_tokens(str(item["text"]), packet)
    if set(response["closing_exchange"]) != {"bull", "bear"}:
        raise ValueError("closing exchange roles mismatch")
    for role in ("bull", "bear"):
        item = response["closing_exchange"][role]
        if set(item) != {"text", "evidence_ids", "claim_ids"}:
            raise ValueError("closing statement contains unsupported fields")
        if not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids):
            raise ValueError("closing statement references an unknown Evidence ID")
        if not item["claim_ids"] or not set(item["claim_ids"]).issubset(claim_ids):
            raise ValueError("closing statement references an unknown Claim ID")
        _validate_text(str(item["text"]), allowed_numbers)
        _validate_no_internal_tokens(str(item["text"]), packet)
    if set(response["revised_hypotheses"]) != {"bull", "bear"}:
        raise ValueError("revised hypothesis roles mismatch")
    claim_lookup = {}
    allowed_metrics = {item["template_id"] for item in packet["claims"]}
    allowed_metrics.update(item["feature"] for item in packet["claims"] if item.get("feature"))
    allowed_metrics.update(item["engine"] for item in packet["evidence"])
    for claim in packet["claims"]:
        claim_lookup[claim["claim_id"]] = claim
        if claim.get("final_claim_id"):
            claim_lookup[claim["final_claim_id"]] = claim
    hypothesis_keys = {
        "title", "hypothesis", "metrics", "evidence_status", "evidence_ids", "claim_ids"
    }
    for role in ("bull", "bear"):
        for item in response["revised_hypotheses"][role]:
            if set(item) != hypothesis_keys:
                raise ValueError("revised hypothesis contains unsupported fields")
            if not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids):
                raise ValueError("revised hypothesis references an unknown Evidence ID")
            if len(item["claim_ids"]) != 1 or item["claim_ids"][0] not in claim_lookup:
                raise ValueError("revised hypothesis references an unknown Claim ID")
            claim = claim_lookup[item["claim_ids"][0]]
            if claim["source_agent"] != role or not claim.get("final_retained"):
                raise ValueError("revised hypothesis must reference a retained same-role Claim")
            claim_evidence_ids = set(_claim_evidence_ids(claim))
            if not set(item["evidence_ids"]).issubset(claim_evidence_ids):
                raise ValueError("revised hypothesis Evidence must belong to its referenced Claim")
            if item["evidence_status"] != claim.get("deterministic_final_verdict"):
                raise ValueError("LLM cannot change the deterministic Evidence status")
            if not item["metrics"] or not set(item["metrics"]).issubset(allowed_metrics):
                raise ValueError("revised hypothesis contains an unverified metric")
            _validate_text(str(item["title"]), allowed_numbers)
            _validate_text(str(item["hypothesis"]), allowed_numbers)
            _validate_no_internal_tokens(str(item["title"]), packet)
            _validate_no_internal_tokens(str(item["hypothesis"]), packet)
            for metric in item["metrics"]:
                _validate_text(str(metric), allowed_numbers)
    for item in response["withdrawn_arguments"]:
        if item["claim_id"] not in claim_ids or not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(evidence_ids):
            raise ValueError("withdrawal references an unknown Claim/Evidence ID")
        _validate_text(str(item["reason"]), allowed_numbers)
    for text in response["additional_information_needed"]:
        _validate_text(str(text), allowed_numbers)
    _validate_text(str(response["uncertainty_statement"]), allowed_numbers)


def _claim_evidence_ids(claim: dict) -> list[str]:
    return list(dict.fromkeys(claim.get("evidence_ids", []) + claim.get("final_evidence_ids", [])))


def deterministic_role_response(role: str, packet: dict) -> dict:
    claims = [item for item in packet["claims"] if item["source_agent"] == role]
    opponents = [item for item in packet["claims"] if item["source_agent"] != role]
    primary = next((item for item in claims if item.get("final_retained")), claims[0])
    opponent = opponents[-1] if opponents else primary
    evidence_ids = list(dict.fromkeys(_claim_evidence_ids(primary) + _claim_evidence_ids(opponent)))[:6]
    claim_ids = list(dict.fromkeys([
        primary.get("final_claim_id") or primary["claim_id"],
        opponent.get("final_claim_id") or opponent["claim_id"],
    ]))
    label = friendly_feature(primary.get("feature"))
    if role == "bull":
        text = f"Bear가 제기한 위험을 함께 봐야 한다는 점은 타당합니다. 다만 {label}에서 확인된 변화는 현재 범위의 긍정 근거로 남습니다."
    else:
        text = f"Bull이 {label}의 의미를 현재 확인 범위로 한정한 점은 타당합니다. 다만 이 흐름이 이어질지는 현재 근거만으로 확인하기 어렵습니다."
    statements = [{"text": text, "evidence_ids": evidence_ids, "claim_ids": claim_ids}]
    response = {
        "role": role,
        "statements": statements,
        "withdrawn_claim_ids": [item["claim_id"] for item in claims if not item.get("final_retained")],
        "limitations": ["새로운 사실을 추가하지 않고 검증된 Evidence만 반영했습니다."],
    }
    validate_role_response(response, packet, role)
    return response


def deterministic_neutral_summary(packet: dict) -> dict:
    summary = {
        "key_issues": [],
        "confirmed_evidence": [],
        "partially_supported": [],
        "uncertainties": [],
    }
    revised = {"bull": [], "bear": []}
    withdrawn = []
    for claim in packet["claims"]:
        evidence_ids = _claim_evidence_ids(claim)
        if not evidence_ids:
            continue
        metric = claim.get("feature") or claim.get("template_id")
        label = friendly_feature(claim.get("feature"))
        statement = {
            "text": f"{label}은 연결된 근거의 판정 범위 안에서만 해석해야 합니다. {feature_meaning(claim.get('feature'))}",
            "evidence_ids": evidence_ids[:4],
            "claim_ids": [claim.get("final_claim_id") or claim["claim_id"]],
        }
        summary["key_issues"].append(statement)
        if claim.get("final_retained") and claim.get("deterministic_final_verdict") == "Supported":
            summary["confirmed_evidence"].append(statement)
        elif claim.get("final_retained") and claim.get("deterministic_final_verdict") == "Partially Supported":
            summary["partially_supported"].append(statement)
        else:
            summary["uncertainties"].append(statement)
            withdrawn.append({
                "claim_id": claim["claim_id"],
                "reason": "검증 후 유지 요건을 충족하지 못해 논거에서 제외됐습니다.",
                "evidence_ids": evidence_ids[:4],
            })
        if claim.get("final_retained"):
            revised[claim["source_agent"]].append({
                "title": f"{label} 검증 가설",
                "hypothesis": f"{label}에서 확인된 변화는 현재 확인된 근거의 범위에서만 해당 관점의 판단 요소로 남습니다.",
                "metrics": [metric],
                "evidence_status": claim["deterministic_final_verdict"],
                "evidence_ids": evidence_ids[:4],
                "claim_ids": [claim.get("final_claim_id") or claim["claim_id"]],
            })
    bull_claim = next((item for item in packet["claims"] if item["source_agent"] == "bull" and _claim_evidence_ids(item)), None)
    bear_claim = next((item for item in packet["claims"] if item["source_agent"] == "bear" and _claim_evidence_ids(item)), None)
    basis = [item for item in (bull_claim, bear_claim) if item]
    closing_evidence = list(dict.fromkeys(
        evidence_id for item in basis for evidence_id in _claim_evidence_ids(item)
    ))[:6]
    closing_claims = [item.get("final_claim_id") or item["claim_id"] for item in basis]
    closing_exchange = {
        "bull": {
            "text": "Bear가 지적한 불확실성은 인정합니다. 따라서 긍정 논거도 현재 확인된 변화의 범위로 한정하는 것이 적절합니다.",
            "evidence_ids": closing_evidence,
            "claim_ids": closing_claims,
        },
        "bear": {
            "text": "Bull이 주장 범위를 한정한 점은 타당합니다. 그래도 현재의 변화가 이어질지는 추가 자료로 확인해야 합니다.",
            "evidence_ids": closing_evidence,
            "claim_ids": closing_claims,
        },
    }
    response = {
        "summary": {key: value[:5] for key, value in summary.items()},
        "closing_exchange": closing_exchange,
        "revised_hypotheses": {role: items[:3] for role, items in revised.items()},
        "withdrawn_arguments": withdrawn[:6],
        "additional_information_needed": ["기준일 이후 판단에는 이후 공개되는 재무자료와 공시의 추가 검증이 필요합니다."],
        "uncertainty_statement": "현재 근거는 미래 결과를 확정하지 않으며 양쪽 논거의 불확실성이 남아 있습니다.",
    }
    validate_neutral_summary(response, packet)
    return response


def _record_stage(telemetry: dict, stage: str, mode: str, attempted: bool, error_code: str | None, packet_hash: str) -> None:
    telemetry["llm_stage_status"][stage] = {
        "mode": mode,
        "attempted": attempted,
        "error_code": error_code,
        "evidence_packet_hash": packet_hash,
    }


def run_post_validation_debate(
    packet: dict,
    settings: Settings,
    telemetry: dict,
    progress: Callable[[str], None] | None = None,
) -> dict:
    client = GoogleStructuredClaimClient(settings, telemetry)
    packet_hash = packet["evidence_packet_hash"]

    def role_stage(role: str, prior: dict | None = None) -> dict:
        stage = f"{role}_rebuttal"
        if progress:
            progress("Bull 검증 후 재반박 중" if role == "bull" else "Bear 검증 후 재반박 중")
        if not settings.llm_configured:
            telemetry["fallback_used"] = True
            response = deterministic_role_response(role, packet)
            _record_stage(telemetry, stage, "deterministic_fallback", False, "LLM_NOT_CONFIGURED", packet_hash)
            return response
        payload = {"shared_evidence_packet": packet}
        if prior is not None:
            payload["validated_prior_rebuttal"] = prior
        try:
            response = client.generate_structured(
                stage, build_role_rebuttal_prompt(role), payload, role_response_schema(role)
            )
            validate_role_response(response, packet, role)
            _record_stage(telemetry, stage, "actual_llm", True, None, packet_hash)
            return response
        except (LLMUnavailable, ValueError, KeyError, TypeError) as error:
            telemetry["fallback_used"] = True
            detail = getattr(error, "error_code", type(error).__name__)
            code = f"{role.upper()}_REBUTTAL_LLM_FALLBACK:{detail}"
            telemetry["error_codes"].append(code)
            response = deterministic_role_response(role, packet)
            _record_stage(telemetry, stage, "deterministic_fallback", True, detail, packet_hash)
            return response

    bull = role_stage("bull")
    bear = role_stage("bear", bull)
    if progress:
        progress("중립 요약 생성 중")
    if not settings.llm_configured:
        telemetry["fallback_used"] = True
        neutral = deterministic_neutral_summary(packet)
        _record_stage(telemetry, "neutral_summary", "deterministic_fallback", False, "LLM_NOT_CONFIGURED", packet_hash)
    else:
        try:
            neutral = client.generate_structured(
                "neutral_summary",
                build_neutral_summary_prompt(),
                {"shared_evidence_packet": packet, "bull_rebuttal": bull, "bear_rebuttal": bear},
                neutral_summary_schema(),
            )
            validate_neutral_summary(neutral, packet)
            _record_stage(telemetry, "neutral_summary", "actual_llm", True, None, packet_hash)
        except (LLMUnavailable, ValueError, KeyError, TypeError) as error:
            telemetry["fallback_used"] = True
            detail = getattr(error, "error_code", type(error).__name__)
            telemetry["error_codes"].append(f"NEUTRAL_SUMMARY_LLM_FALLBACK:{detail}")
            neutral = deterministic_neutral_summary(packet)
            _record_stage(telemetry, "neutral_summary", "deterministic_fallback", True, detail, packet_hash)

    return {
        "shared_evidence_hash": packet_hash,
        "shared_evidence_packet": packet,
        "bull_rebuttal": bull,
        "bear_rebuttal": bear,
        "neutral_summary": neutral,
        "validation": {
            "unknown_evidence_ids": 0,
            "ungrounded_numeric_facts": 0,
            "verdict_override_fields": 0,
        },
    }
