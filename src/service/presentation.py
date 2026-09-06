"""User-facing finance language without leaking internal feature or engine names."""

from __future__ import annotations

import re
from urllib.parse import urlparse


FEATURE_LABELS = {
    "revenue_yoy": "전년 동기 대비 매출 흐름",
    "operating_income_yoy": "전년 동기 대비 영업이익 흐름",
    "operating_profit_yoy": "전년 동기 대비 영업이익 흐름",
    "net_income_yoy": "전년 동기 대비 순이익 흐름",
    "operating_margin": "영업이익률 수준",
    "operating_margin_change_yoy": "전년 동기 대비 수익성 변화",
    "debt_ratio": "재무 부담 수준",
    "cfo_to_assets": "자산 대비 영업현금 창출력",
    "assets": "자산 규모",
    "liabilities": "부채 규모",
    "equity": "자기자본 규모",
    "eps": "주당 이익",
    "fy_roe": "자기자본 이익창출력",
    "fy_roa": "자산 이익창출력",
    "momentum_60": "최근 주가 흐름의 방향과 강도",
    "realized_vol_20": "최근 단기 주가 변동 수준",
    "volatility_20d": "최근 단기 주가 변동 수준",
    "excess_ret_20": "시장 대비 최근 주가 흐름",
    "drawdown_60": "최근 고점 대비 주가 하락 폭",
    "market_ret_20": "최근 시장 흐름",
    "foreign_net_buy_20": "최근 외국인 수급 흐름",
    "foreign_net_buy_20d": "최근 외국인 수급 흐름",
    "base_rate_change_60": "최근 기준금리 변화",
    "treasury_3y_change_20": "최근 시장금리 변화",
    "usdkrw_change_20": "최근 원화 환율 변화",
    "kospi_ret_20": "최근 국내 주식시장 흐름",
}

FEATURE_MEANINGS = {
    "revenue_yoy": "기업의 외형 성장과 사업 수요가 어느 방향으로 움직이는지 보여줍니다.",
    "operating_income_yoy": "본업에서 벌어들이는 이익의 회복 또는 부담 정도를 보여줍니다.",
    "operating_profit_yoy": "본업에서 벌어들이는 이익의 회복 또는 부담 정도를 보여줍니다.",
    "net_income_yoy": "최종 이익 창출력이 개선되는지 약해지는지 보여줍니다.",
    "operating_margin": "매출을 실제 영업이익으로 전환하는 능력을 보여줍니다.",
    "operating_margin_change_yoy": "최근 이익 창출력이 회복되는지 악화되는지 보여줍니다.",
    "debt_ratio": "기업이 감당해야 할 재무적 부담의 상대적 크기를 보여줍니다.",
    "cfo_to_assets": "보유 자산이 실제 영업현금으로 이어지는 정도를 보여줍니다.",
    "fy_roe": "주주자본을 활용해 이익을 만드는 효율을 보여줍니다.",
    "fy_roa": "전체 자산을 활용해 이익을 만드는 효율을 보여줍니다.",
    "momentum_60": "최근 시장 참여자의 평가가 어느 방향으로 이어졌는지 보여주는 참고 흐름입니다.",
    "realized_vol_20": "주가 움직임이 평소보다 커졌는지 살펴보는 단기 위험 지표입니다.",
    "volatility_20d": "주가 움직임이 평소보다 커졌는지 살펴보는 단기 위험 지표입니다.",
    "excess_ret_20": "개별 기업의 주가 흐름이 시장 전체보다 강했는지 약했는지 보여줍니다.",
    "drawdown_60": "최근 고점에서 가격이 얼마나 밀렸는지 보여주는 하락 위험 지표입니다.",
    "market_ret_20": "개별 기업을 둘러싼 시장 환경이 우호적인지 살펴보는 참고 정보입니다.",
    "foreign_net_buy_20": "최근 외국인 투자자의 수급 방향을 보여주는 참고 정보입니다.",
    "foreign_net_buy_20d": "최근 외국인 투자자의 수급 방향을 보여주는 참고 정보입니다.",
    "base_rate_change_60": "자금 조달 환경과 시장 할인율에 영향을 줄 수 있는 거시 환경입니다.",
    "treasury_3y_change_20": "시장금리 변화에 따른 기업 가치와 자금 부담의 환경을 보여줍니다.",
    "usdkrw_change_20": "수출입 가격과 해외 실적 환산에 영향을 줄 수 있는 환율 환경입니다.",
}

TEMPLATE_LABELS = {
    "T1": "조건에 따른 과거 시장 반응",
    "T2": "과거 시장 관계",
    "T3": "다른 영향을 고려한 과거 관계",
    "T4": "실적 발표 전후 시장 반응",
    "T5": "최근 실적 변화",
    "T6": "과거·동종기업 비교",
    "RAG": "공식 공시 내용",
}

ENGINE_LABELS = {
    "T1": "조건별 과거 반응 분석",
    "T2": "과거 관계 분석",
    "T3": "복합 관계 분석",
    "T4": "기업 이벤트 분석",
    "T5": "재무 실적 확인",
    "T6": "과거·동종기업 비교",
    "RAG": "공식 공시 확인",
    "ML": "과거 시장 상태 보조 분석",
    "SHAP": "모델 해석 보조 정보",
}

VERDICT_LABELS = {
    "Supported": "근거확인",
    "Partially Supported": "일부 근거확인",
    "Insufficient Evidence": "근거부족",
    "Contradicted": "반대 근거 확인",
}

REPORT_TYPE_LABELS = {
    "periodic_report": "정기보고서",
    "earnings": "영업실적 공시",
    "supply_contract": "공급계약 공시",
    "major_management": "주요경영사항 공시",
}

REVISION_LABELS = {
    "KEEP": "근거 범위에서 유지",
    "WEAKEN": "표현을 완화해 수정",
    "DROP": "검증 후 철회",
    "REFRAME": "확인된 사실 중심으로 수정",
}


def friendly_feature(value: str | None) -> str:
    if not value:
        return "관련 재무·시장 요소"
    key = str(value)
    return FEATURE_LABELS.get(
        key,
        TEMPLATE_LABELS.get(key, ENGINE_LABELS.get(key, "관련 재무·시장 요소")),
    )


def feature_meaning(value: str | None) -> str:
    return FEATURE_MEANINGS.get(str(value), "기업의 현재 여건을 이해하기 위한 참고 요소입니다.")


def friendly_template(value: str | None) -> str:
    return TEMPLATE_LABELS.get(str(value), "데이터 기반 검증")


def friendly_engine(value: str | None) -> str:
    return ENGINE_LABELS.get(str(value), "데이터 기반 검증")


def friendly_verdict(value: str | None) -> str:
    return VERDICT_LABELS.get(str(value), "검증 결과 확인 필요")


def friendly_revision(value: str | None) -> str:
    return REVISION_LABELS.get(str(value), "검증 결과 반영")


def is_dart_source_url(value: str | None) -> bool:
    try:
        parsed = urlparse(str(value or ""))
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and (
        host == "dart.fss.or.kr" or host.endswith(".dart.fss.or.kr")
    )


def friendly_disclosure_name(
    report_name: str | None,
    report_type: str | None = None,
) -> str:
    name = str(report_name or "").strip()
    if name and "�" not in name:
        return naturalize_text(name)
    period = re.search(r"\((20\d{2})\.(\d{2})\)", name)
    if period:
        year, month = period.groups()
        report_label = {
            "12": "사업보고서",
            "06": "반기보고서",
            "03": "1분기보고서",
            "09": "3분기보고서",
        }.get(month, "정기보고서")
        return f"{year}년 {report_label}"
    return REPORT_TYPE_LABELS.get(str(report_type), "공식 공시")


def naturalize_text(text: str) -> str:
    rendered = str(text)
    replacements = {**FEATURE_LABELS, **TEMPLATE_LABELS, **ENGINE_LABELS}
    for internal in sorted(replacements, key=len, reverse=True):
        rendered = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(internal)}(?![A-Za-z0-9_])", replacements[internal], rendered)
    return rendered


def explain_claim(claim: dict) -> str:
    base = naturalize_text(str(claim.get("atomic_claim_text", ""))).strip()
    meaning = feature_meaning(claim.get("feature"))
    if not base:
        return meaning
    return base + " " + meaning


def internal_display_tokens(packet: dict) -> set[str]:
    tokens = {item.get("template_id") for item in packet.get("claims", [])}
    tokens.update(item.get("feature") for item in packet.get("claims", []) if item.get("feature"))
    tokens.update(item.get("engine") for item in packet.get("evidence", []))
    return {str(token) for token in tokens if token}
