"""Frozen CAT_012 inference for the latest feature row without future targets."""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config.paths import PATHS
from src.service.date_resolver import normalize_stock_code


STATE_NAMES = ("Negative", "Neutral", "Positive")
PROBABILITY_SEMANTICS = "historical response-state similarity score; not return probability"


@lru_cache(maxsize=1)
def _model_contract() -> dict:
    return json.loads(PATHS.model_contract.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _feature_panel() -> pd.DataFrame:
    contract = _model_contract()
    columns = ["stock_code", "trading_date", "company_name", "market", "industry"] + contract["feature_list"]
    frame = pd.read_parquet(PATHS.feature_panel, columns=columns)
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["trading_date"] = pd.to_datetime(frame["trading_date"], errors="coerce")
    return frame


@lru_cache(maxsize=1)
def _load_frozen_model():
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(PATHS.frozen_model))
    return model


def _latest_feature_row(stock_code: str, analysis_as_of_date) -> pd.Series:
    code = normalize_stock_code(stock_code)
    as_of = pd.Timestamp(analysis_as_of_date)
    eligible = _feature_panel().loc[
        _feature_panel()["stock_code"].eq(code)
        & _feature_panel()["trading_date"].le(as_of)
    ].sort_values("trading_date")
    if eligible.empty:
        raise ValueError(f"ML Feature unavailable: {code} / {as_of.date()}")
    return eligible.iloc[-1]


def _adapter_fallback(stock_code: str, analysis_as_of_date, reason: str) -> dict:
    if not PATHS.ml_adapter.exists():
        return {"available": False, "error_code": "ML_MODEL_AND_ADAPTER_UNAVAILABLE", "reason": reason}
    frame = pd.read_parquet(PATHS.ml_adapter)
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    eligible = frame.loc[
        frame["stock_code"].eq(normalize_stock_code(stock_code))
        & frame["as_of_date"].le(pd.Timestamp(analysis_as_of_date))
    ].sort_values("as_of_date")
    if eligible.empty:
        return {"available": False, "error_code": "ML_EVIDENCE_UNAVAILABLE", "reason": reason}
    row = eligible.iloc[-1]
    return {
        "available": True,
        "inference_mode": "frozen_step10_adapter_fallback",
        "as_of_date": row["as_of_date"].date().isoformat(),
        "predicted_state": str(row["predicted_state"]),
        "confidence": float(row["prediction_confidence"]),
        "entropy": float(row["prediction_entropy"]),
        "model_version": str(row["model_version"]),
        "label_policy": str(row["label_policy"]),
        "probability_semantics": str(row["probability_semantics"]),
        "limitations": [reason, "최신 Feature 날짜보다 이전의 Frozen adapter 행입니다."],
    }


def infer_latest_ml_context(stock_code: str, analysis_as_of_date) -> dict:
    """Infer CAT_012 on the latest PIT feature row; no target/label column is read."""
    try:
        contract = _model_contract()
        row = _latest_feature_row(stock_code, analysis_as_of_date)
        features = list(contract["feature_list"])
        probabilities = np.asarray(_load_frozen_model().predict_proba(row[features].to_frame().T)[0], dtype=float)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        probabilities = probabilities / probabilities.sum()
        state_index = int(np.argmax(probabilities))
        entropy = float(-np.sum([p * math.log(p) for p in probabilities if p > 0]))
        return {
            "available": True,
            "inference_mode": "runtime_frozen_catboost",
            "stock_code": normalize_stock_code(stock_code),
            "as_of_date": row["trading_date"].date().isoformat(),
            "p_negative": float(probabilities[0]),
            "p_neutral": float(probabilities[1]),
            "p_positive": float(probabilities[2]),
            "predicted_state": STATE_NAMES[state_index],
            "confidence": float(probabilities[state_index]),
            "entropy": entropy,
            "model_version": "CatBoost_CAT_012_final_2019_2024",
            "feature_count": len(features),
            "label_policy": contract["final_label_policy"],
            "probability_semantics": PROBABILITY_SEMANTICS,
            "future_target_used": False,
            "evidence_role": "auxiliary_only",
            "limitations": [
                "현재 Feature 조합과 과거 시장반응 상태의 상대적 유사도입니다.",
                "수익률 확률, 투자 추천 또는 독립적인 최종 판정이 아닙니다.",
            ],
        }
    except Exception as error:
        return _adapter_fallback(stock_code, analysis_as_of_date, f"runtime inference unavailable: {type(error).__name__}")


def to_frozen_ml_row(context: dict) -> dict | None:
    if not context.get("available"):
        return None
    return {
        "stock_code": context.get("stock_code"),
        "as_of_date": context.get("as_of_date"),
        "predicted_state": context.get("predicted_state"),
        "prediction_confidence": context.get("confidence"),
        "prediction_entropy": context.get("entropy"),
        "model_version": context.get("model_version"),
        "label_policy": context.get("label_policy"),
        "probability_semantics": context.get("probability_semantics"),
    }

