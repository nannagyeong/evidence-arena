"""Resolve the latest service date from real feature data, never from user input."""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas as pd

from src.config.paths import PATHS


SERVICE_SNAPSHOT_DATE = date(2025, 12, 30)


@lru_cache(maxsize=1)
def _feature_index() -> pd.DataFrame:
    frame = pd.read_parquet(
        PATHS.feature_panel,
        columns=["stock_code", "trading_date", "company_name", "market", "industry", "is_finance_industry"],
    )
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["trading_date"] = pd.to_datetime(frame["trading_date"], errors="coerce")
    return frame.dropna(subset=["trading_date"])


def normalize_stock_code(stock_code: str) -> str:
    normalized = str(stock_code).strip().zfill(6)
    if len(normalized) != 6 or not normalized.isdigit():
        raise ValueError("stock_code must contain six digits")
    return normalized


def resolve_analysis_as_of_date(stock_code: str) -> date:
    """Return the fixed service snapshot after proving the stock supports it."""
    normalized = normalize_stock_code(stock_code)
    eligible = _feature_index().loc[
        lambda x: x["stock_code"].eq(normalized)
        & x["trading_date"].eq(pd.Timestamp(SERVICE_SNAPSHOT_DATE)),
        "trading_date",
    ]
    if eligible.empty:
        raise ValueError(f"{normalized}은 서비스 스냅샷 {SERVICE_SNAPSHOT_DATE.isoformat()}을 지원하지 않습니다.")
    return SERVICE_SNAPSHOT_DATE


def latest_supported_date() -> date:
    if not _feature_index()["trading_date"].eq(pd.Timestamp(SERVICE_SNAPSHOT_DATE)).any():
        raise RuntimeError("고정 서비스 스냅샷을 Feature Panel에서 찾을 수 없습니다.")
    return SERVICE_SNAPSHOT_DATE


@lru_cache(maxsize=1)
def list_supported_companies() -> list[dict]:
    latest = (
        _feature_index()
        .loc[lambda x: x["trading_date"].eq(pd.Timestamp(SERVICE_SNAPSHOT_DATE))]
        .drop_duplicates("stock_code", keep="last")
        .sort_values(["company_name", "stock_code"])
    )
    return [
        {
            "stock_code": row.stock_code,
            "company_name": str(row.company_name),
            "market": str(row.market),
            "industry": str(row.industry),
            "is_finance_industry": bool(row.is_finance_industry),
            "latest_supported_date": SERVICE_SNAPSHOT_DATE.isoformat(),
        }
        for row in latest.itertuples(index=False)
    ]
