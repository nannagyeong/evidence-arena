"""BVB Streamlit UI: evidence-verified Bull vs Bear debate."""

from __future__ import annotations

import html

import streamlit as st

from src.config.settings import get_settings
from src.service.date_resolver import SERVICE_SNAPSHOT_DATE, list_supported_companies
from src.service.orchestrator import _run_evidence_arena
from src.service.presentation import (
    explain_claim,
    feature_meaning,
    friendly_disclosure_name,
    friendly_engine,
    friendly_feature,
    friendly_revision,
    friendly_verdict,
    is_dart_source_url,
    naturalize_text,
)


UI_REVISION = "bvb-compact-hypothesis-ui-20260907-v6"

st.set_page_config(page_title="BVB", page_icon="⚖️", layout="wide")
if st.session_state.get("_ui_revision") != UI_REVISION:
    st.session_state.clear()
    st.session_state["_ui_revision"] = UI_REVISION

st.markdown(
    """
    <style>
      :root {
        --ink: #191f28;
        --muted: #6b7684;
        --line: #e5e8eb;
        --soft: #f8f9fa;
        --bull: #e5484d;
        --bull-soft: #fff1f2;
        --bull-line: #f2b8bc;
        --bear: #3182f6;
        --bear-soft: #eef6ff;
        --bear-line: #b8d7ff;
      }
      html, body, [class*="css"] {font-family: Inter, Pretendard, "Noto Sans KR", sans-serif;}
      .stApp {background: #fff; color: var(--ink);}
      .block-container {max-width: 1160px; padding-top: 1.1rem; padding-bottom: 5rem;}
      header[data-testid="stHeader"] {background: rgba(255,255,255,.92);}
      #MainMenu, footer {visibility: hidden;}

      .topbar {display:flex; align-items:center; justify-content:space-between; min-height:64px; border-bottom:1px solid var(--line); margin-bottom:0;}
      .brand {font-size:1.45rem; font-weight:850; letter-spacing:-.04em; color:var(--ink);}
      .nav {display:flex; gap:2rem; color:#4e5968; font-size:.92rem; font-weight:650;}
      .nav-date {padding:.52rem .8rem; border-radius:999px; background:#f2f4f6; color:#4e5968;}

      .hero {display:grid; grid-template-columns:1.15fr .85fr; min-height:410px; align-items:center; gap:3rem; padding:4.6rem 0 3.7rem; border-bottom:1px solid var(--line);}
      .eyebrow {font-size:.78rem; font-weight:800; letter-spacing:.12em; color:#8b95a1; text-transform:uppercase;}
      .hero h1 {font-size:3.35rem; line-height:1.15; letter-spacing:-.055em; margin:.8rem 0 1.15rem; color:var(--ink);}
      .hero p {font-size:1.08rem; line-height:1.8; color:#6b7684; margin:0; max-width:620px;}
      .hero-visual {position:relative; height:300px; border-radius:36px; background:linear-gradient(150deg,#f7f9fc,#eef4fb); overflow:hidden; border:1px solid #eef0f3;}
      .float-card {position:absolute; width:245px; padding:1.15rem 1.25rem; background:rgba(255,255,255,.94); box-shadow:0 18px 55px rgba(26,46,77,.11); border-radius:18px;}
      .float-card strong {display:block; font-size:.82rem; margin-bottom:.45rem; letter-spacing:.04em;}
      .float-card p {font-size:.88rem; line-height:1.45; color:#4e5968;}
      .float-bull {left:34px; top:38px; border-left:4px solid var(--bull); transform:rotate(-3deg);}
      .float-bear {right:28px; bottom:34px; border-left:4px solid var(--bear); transform:rotate(3deg);}
      .float-check {right:34px; top:40px; width:auto; padding:.7rem 1rem; font-size:.8rem; font-weight:750; color:#4e5968;}

      .select-panel {padding:2rem 2.1rem; margin:2.2rem 0 3.7rem; border:1px solid var(--line); border-radius:20px; background:#fff; box-shadow:0 12px 40px rgba(25,31,40,.055);}
      .select-panel h3 {margin:0 0 .25rem; font-size:1.25rem;}
      .select-panel p {margin:0 0 1.25rem; color:var(--muted); font-size:.92rem;}
      .section-kicker {margin-top:4rem; color:#8b95a1; font-size:.76rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase;}
      .section-title {font-size:2rem; line-height:1.3; letter-spacing:-.04em; margin:.35rem 0 .55rem;}
      .section-desc {color:var(--muted); margin-bottom:1.6rem;}

      .result-head {display:flex; justify-content:space-between; gap:1rem; align-items:center; padding:1.15rem 1.3rem; border:1px solid var(--line); border-radius:14px; margin:1.5rem 0 2.5rem;}
      .result-head strong {font-size:1.08rem;}
      .result-meta {color:var(--muted); font-size:.86rem;}

      .chat-row {display:flex; width:100%; margin:1.1rem 0;}
      .chat-row.bear {justify-content:flex-end;}
      .speaker {font-size:.76rem; font-weight:850; letter-spacing:.06em; margin-bottom:.45rem;}
      .speaker.bull {color:var(--bull);}
      .speaker.bear {color:var(--bear); text-align:right;}
      .bubble-wrap {max-width:72%;}
      .bubble {padding:1.1rem 1.2rem; border-radius:18px; line-height:1.65; font-size:.96rem;}
      .bubble.bull {background:var(--bull-soft); border:1px solid var(--bull-line); border-top-left-radius:5px;}
      .bubble.bear {background:var(--bear-soft); border:1px solid var(--bear-line); border-top-right-radius:5px;}
      .bubble-meta {font-size:.73rem; color:#8b95a1; margin-top:.55rem; overflow-wrap:anywhere;}
      .bubble-wrap.bear .bubble-meta {text-align:right;}

      .evidence-bridge {margin:2.8rem 0; padding:1.6rem; text-align:center; border-top:1px solid var(--line); border-bottom:1px solid var(--line); background:linear-gradient(90deg,#fff,#f8fafc,#fff);}
      .evidence-bridge strong {display:block; font-size:1.05rem; margin-bottom:.75rem;}
      .evidence-checks {display:flex; justify-content:center; flex-wrap:wrap; gap:.55rem;}
      .evidence-chip {padding:.42rem .72rem; border:1px solid #d8dde3; border-radius:999px; background:#fff; color:#4e5968; font-size:.78rem;}

      .verification {display:grid; grid-template-columns:90px 1fr auto; gap:1rem; align-items:center; padding:1rem 0; border-bottom:1px solid var(--line);}
      .side-tag {font-size:.74rem; font-weight:850; letter-spacing:.05em;}
      .side-tag.bull {color:var(--bull);}
      .side-tag.bear {color:var(--bear);}
      .verify-text {font-size:.9rem; line-height:1.5;}
      .badge {display:inline-block; padding:.28rem .58rem; border:1px solid #cfd4da; border-radius:999px; background:#fff; color:#4e5968; font-size:.72rem; font-weight:750; white-space:nowrap;}
      .status-stack {display:flex; align-items:center; justify-content:flex-end; flex-wrap:wrap; gap:.42rem;}
      .badge.status-positive {color:#087443; border-color:#9ed8b9; background:#ecf9f1;}
      .badge.status-negative {color:#c92a2a; border-color:#f0b4b4; background:#fff0f0;}
      .direct-evidence-link {text-decoration:none !important;}
      .direct-source-links {display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.55rem;}
      .direct-source-links a {color:#4e5968 !important; font-size:.72rem; font-weight:700; text-decoration:underline;}
      .source-item {padding:1rem 0; border-bottom:1px solid var(--line);}
      .source-item:last-child {border-bottom:0;}
      .source-item strong {display:block; margin-bottom:.2rem;}
      .source-date {font-size:.78rem; color:var(--muted); margin-bottom:.55rem;}
      .dart-link {display:inline-block; padding:.48rem .72rem; border:1px solid #cfd4da; border-radius:10px; color:var(--ink) !important; text-decoration:none !important; font-size:.8rem; font-weight:750; background:#fff;}
      .dart-link:hover {background:#f2f4f6;}

      .summary-grid {display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.2rem;}
      .summary-card {padding:1.2rem 1.25rem; border:1px solid var(--line); border-radius:14px; background:#fff;}
      .summary-card h4 {margin:0 0 .75rem; font-size:.9rem;}
      .summary-card p {margin:.45rem 0; color:#4e5968; line-height:1.55; font-size:.88rem;}

      .hypothesis-card {padding:1.35rem; border-radius:16px; margin:.8rem 0; min-height:220px;}
      .hypothesis-card.bull {background:var(--bull-soft); border:1px solid var(--bull-line); border-top:4px solid var(--bull);}
      .hypothesis-card.bear {background:var(--bear-soft); border:1px solid var(--bear-line); border-top:4px solid var(--bear);}
      .hypothesis-card h4 {margin:.45rem 0 .7rem; font-size:1.03rem;}
      .hypothesis-card p {line-height:1.62; color:#3e4854; font-size:.9rem;}
      .metric-list {margin-top:1rem; color:#6b7684; font-size:.78rem;}
      .evidence-ref {margin-top:.7rem; color:#8b95a1; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.69rem; overflow-wrap:anywhere;}
      .bull-heading {color:var(--bull); border-bottom:3px solid var(--bull-line); padding-bottom:.55rem;}
      .bear-heading {color:var(--bear); border-bottom:3px solid var(--bear-line); padding-bottom:.55rem;}

      div[data-testid="stMetric"] {border-top:1px solid var(--line); padding-top:.65rem;}
      div[data-testid="stExpander"] {border:1px solid var(--line); border-radius:14px;}
      div[data-testid="stAlert"] {background:#f8f9fa; color:var(--ink); border:1px solid var(--line);}
      button[kind="primary"] {background:#191f28 !important; border-color:#191f28 !important; color:#fff !important; min-height:3rem; border-radius:12px !important;}

      @media (max-width: 760px) {
        .nav {display:none;}
        .hero {grid-template-columns:1fr; padding-top:2.8rem;}
        .hero h1 {font-size:2.5rem;}
        .hero-visual {height:260px;}
        .bubble-wrap {max-width:92%;}
        .summary-grid {grid-template-columns:1fr;}
        .verification {grid-template-columns:70px 1fr;}
        .verification .badge {grid-column:2;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def company_options() -> list[dict]:
    return list_supported_companies()


@st.cache_resource(show_spinner=False)
def runtime_settings():
    return get_settings()


def verdict_badge(verdict: str) -> str:
    css_class = (
        "status-positive"
        if verdict in {"Supported", "Partially Supported"}
        else "status-negative"
    )
    return f'<span class="badge {css_class}">{html.escape(friendly_verdict(verdict))}</span>'


def retention_badge(retained: bool) -> str:
    css_class = "status-positive" if retained else "status-negative"
    label = "유지" if retained else "철회"
    return f'<span class="badge {css_class}">{label}</span>'


def stage_mode(result: dict, stage: str) -> str:
    status = result["execution_metadata"].get("llm_stage_status", {}).get(stage, {})
    return "Gemini" if status.get("mode") == "actual_llm" else "Fallback"


def render_chat_claim(text: str, role: str, meta: str = "") -> None:
    role_upper = role.upper()
    safe_text = html.escape(text)
    safe_meta = html.escape(meta)
    st.markdown(
        f'<div class="chat-row {role}"><div class="bubble-wrap {role}">'
        f'<div class="speaker {role}">{"● " if role == "bull" else ""}{role_upper}{" ●" if role == "bear" else ""}</div>'
        f'<div class="bubble {role}">{safe_text}</div>'
        f'<div class="bubble-meta">{safe_meta}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_verification(item: dict) -> None:
    role = item["source_agent"]
    st.markdown(
        f'<div class="verification"><span class="side-tag {role}">{role.upper()}</span>'
        f'<div class="verify-text">{html.escape(explain_claim(item["claim"]))}'
        f'<div class="bubble-meta">{html.escape(friendly_revision(item["revision_action"]))}</div></div>'
        f'<div class="status-stack">{retention_badge(item["final_retained"])}'
        f'{verdict_badge(item["initial_verdict"])}</div></div>',
        unsafe_allow_html=True,
    )


def evidence_packet_lookup(result: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for evaluation in result["claim_evaluations"]:
        for packet in evaluation["evidence_packets"] + evaluation["final_evidence_packets"]:
            lookup[packet["evidence_id"]] = packet
    return lookup


def hypothesis_sources(item: dict, packet_lookup: dict[str, dict]) -> tuple[list[str], list[dict]]:
    evidence_kinds: list[str] = []
    citations: list[dict] = []
    seen_urls: set[str] = set()
    for evidence_id in item.get("evidence_ids", []):
        packet = packet_lookup.get(evidence_id)
        if not packet:
            continue
        label = friendly_engine(packet.get("engine"))
        if label not in evidence_kinds:
            evidence_kinds.append(label)
        for citation in packet.get("citations", []):
            url = str(citation.get("source_url") or "")
            if not is_dart_source_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            citations.append(citation)
    return evidence_kinds, citations


def render_hypothesis_card(
    item: dict,
    role: str,
    index: int,
    packet_lookup: dict[str, dict],
) -> None:
    metrics = " · ".join(friendly_feature(metric) for metric in item.get("metrics", []))
    _, citations = hypothesis_sources(item, packet_lookup)
    evidence_status = verdict_badge(item["evidence_status"])
    additional_links = ""
    if citations:
        first = citations[0]
        first_url = html.escape(str(first["source_url"]), quote=True)
        first_title = html.escape(
            friendly_disclosure_name(first.get("report_name")),
            quote=True,
        )
        css_class = (
            "status-positive"
            if item["evidence_status"] in {"Supported", "Partially Supported"}
            else "status-negative"
        )
        evidence_status = (
            f'<a class="badge {css_class} direct-evidence-link" href="{first_url}" '
            f'target="_blank" rel="noopener noreferrer" title="{first_title}">'
            f'{html.escape(friendly_verdict(item["evidence_status"]))} →</a>'
        )
        if len(citations) > 1:
            links = []
            for source_index, citation in enumerate(citations[1:], 2):
                url = html.escape(str(citation["source_url"]), quote=True)
                title = html.escape(
                    friendly_disclosure_name(citation.get("report_name")),
                    quote=True,
                )
                links.append(
                    f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                    f'title="{title}">추가 DART 원문 {source_index} →</a>'
                )
            additional_links = f'<div class="direct-source-links">{"".join(links)}</div>'
    st.markdown(
        f'<div class="hypothesis-card {role}"><span class="side-tag {role}">{index:02d}</span>'
        f'<h4>{html.escape(naturalize_text(item["title"]))}</h4>'
        f'<p>{html.escape(naturalize_text(item["hypothesis"]))}</p>'
        f'<div class="metric-list"><strong>확인할 핵심 요소</strong><br>{html.escape(metrics)}</div>'
        f'<div style="margin-top:.75rem">{evidence_status}</div>'
        f'{additional_links}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_evidence_packet(packet: dict) -> None:
    source_date = packet.get("source_date") or "확인 불가"
    st.markdown(
        f"**{html.escape(friendly_engine(packet.get('engine')))}** · "
        f"{verdict_badge(packet.get('evidence_status'))}",
        unsafe_allow_html=True,
    )
    st.caption(f"자료 기준일: {source_date}")
    if packet.get("limitations"):
        st.caption("해당 자료가 확인할 수 있는 범위 안에서만 판단에 반영했습니다.")


def _display_value(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return naturalize_text(str(value))


def render_fact_group(title: str, payloads: dict) -> None:
    st.markdown(f"#### {title}")
    if not payloads:
        st.caption("현재 기준일에 확인 가능한 자료가 없습니다.")
        return
    for name, payload in payloads.items():
        value = payload.get("value") if isinstance(payload, dict) else payload
        source_date = ""
        if isinstance(payload, dict):
            source_date = payload.get("effective_date") or payload.get("source_date") or payload.get("data_date") or ""
        label = friendly_feature(name)
        date_text = f" · 자료 기준일 {source_date}" if source_date else ""
        st.markdown(f"**{html.escape(label)}** · {html.escape(_display_value(value))}{html.escape(date_text)}")
        st.caption(feature_meaning(name))


st.markdown(
    f"""
    <div class="topbar">
      <div class="brand">BVB</div>
      <div class="nav">
        <span>토론 과정</span><span>Evidence</span><span>최종 가설</span>
        <span class="nav-date">{SERVICE_SNAPSHOT_DATE.isoformat()}</span>
      </div>
    </div>
    <section class="hero">
      <div>
        <div class="eyebrow">Bull vs Bear · Verified by Evidence</div>
        <h1>한 종목,<br>두 관점,<br>검증된 가설까지.</h1>
        <p>종목만 선택하세요. Bull과 Bear가 서로 반대되는 가설을 만들고, 실제 데이터 검증을 거쳐 각자의 논거를 다시 수정합니다.</p>
      </div>
      <div class="hero-visual">
        <div class="float-card float-bull"><strong style="color:var(--bull)">BULL HYPOTHESIS</strong><p>상승 관점의 근거를 검증 가능한 Claim으로 제시합니다.</p></div>
        <div class="float-card float-check">✓ EVIDENCE CHECKED</div>
        <div class="float-card float-bear"><strong style="color:var(--bear)">BEAR RISK</strong><p>하락·제약 요인을 같은 데이터 기준으로 검토합니다.</p></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    companies = company_options()
except Exception:
    st.error("서비스 데이터 경로를 확인할 수 없습니다.")
    st.stop()

st.markdown(
    '<div class="select-panel"><h3>분석할 종목을 선택하세요</h3>'
    '<p>사용자 가설이나 날짜 입력 없이 동일한 기준으로 AI 토론을 시작합니다.</p>',
    unsafe_allow_html=True,
)
labels = [f"{item['company_name']} ({item['stock_code']}) · {item['industry']}" for item in companies]
default_index = next((i for i, item in enumerate(companies) if item["stock_code"] == "005930"), 0)
selected_label = st.selectbox("분석 종목 선택", labels, index=default_index, label_visibility="collapsed")
selected = companies[labels.index(selected_label)]
run_clicked = st.button("AI 토론 시작", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if run_clicked:
    try:
        with st.status("Fact Room 구성 중", expanded=True) as status:
            def update_progress(stage: str) -> None:
                status.update(label=stage)
                status.write(stage)

            result = _run_evidence_arena(
                selected["stock_code"],
                settings=runtime_settings(),
                progress_callback=update_progress,
            )
            status.update(label="BVB 토론 완료", state="complete", expanded=False)
        st.session_state["bvb_result"] = result
    except Exception as error:
        st.error(f"토론 처리에 실패했습니다. 데이터와 API 설정을 확인해 주세요. ({type(error).__name__})")

result = st.session_state.get("bvb_result")
if result:
    summary = result["fact_room_summary"]
    st.markdown(
        f'<div class="result-head"><div><strong>{html.escape(result["company"])} ({html.escape(result["stock_code"])})</strong>'
        f'<div class="result-meta">{html.escape(summary["market"])} · {html.escape(summary["industry"])}</div></div>'
        f'<div class="result-meta">분석 기준일<br><strong>{html.escape(result["analysis_as_of_date"])}</strong></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Debate process</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1. 최초 논거</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Bull과 Bear가 아직 서로 반박하지 않고, 각 관점에서 확인할 이유를 독립적으로 제시합니다.</div>', unsafe_allow_html=True)

    for claim in result["bull"]["initial_claims"]:
        render_chat_claim(
            explain_claim(claim), "bull",
            f"최초 논거 · {stage_mode(result, 'bull_initial')}",
        )
    for claim in result["bear"]["initial_claims"]:
        render_chat_claim(
            explain_claim(claim), "bear",
            f"최초 논거 · {stage_mode(result, 'bear_initial')}",
        )

    st.markdown(
        """
        <div class="evidence-bridge">
          <strong>Evidence 검증 완료</strong>
          <div class="evidence-checks">
            <span class="evidence-chip">✓ 재무 데이터</span>
            <span class="evidence-chip">✓ 시장 데이터</span>
            <span class="evidence-chip">✓ 거시경제 데이터</span>
            <span class="evidence-chip">✓ 공시 · RAG</span>
            <span class="evidence-chip">✓ 통계 · ML</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for evaluation in result["claim_evaluations"]:
        render_verification(evaluation)

    neutral = result["debate"]["neutral_summary"]
    st.markdown('<div class="section-kicker">Direct debate</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2. Bull vs Bear 실제 토론</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">각 발언은 검증된 근거를 바탕으로 바로 앞 상대방의 주장에 직접 답합니다.</div>', unsafe_allow_html=True)

    for item in result["debate"]["bull_rebuttal"].get("statements", []):
        render_chat_claim(
            naturalize_text(item["text"]), "bull",
            f"Bear 최초 논거에 직접 답변 · {stage_mode(result, 'bull_rebuttal')} · 검증 근거 연결",
        )
    if not result["debate"]["bull_rebuttal"].get("statements"):
        render_chat_claim("검증 결과 유지할 수 있는 Bull 논거가 없어 주장을 철회합니다.", "bull", stage_mode(result, "bull_rebuttal"))

    for item in result["debate"]["bear_rebuttal"].get("statements", []):
        render_chat_claim(
            naturalize_text(item["text"]), "bear",
            f"Bull의 직전 발언에 직접 답변 · {stage_mode(result, 'bear_rebuttal')} · 검증 근거 연결",
        )
    if not result["debate"]["bear_rebuttal"].get("statements"):
        render_chat_claim("검증 결과 유지할 수 있는 Bear 논거가 없어 주장을 철회합니다.", "bear", stage_mode(result, "bear_rebuttal"))

    closing = neutral["closing_exchange"]
    render_chat_claim(
        naturalize_text(closing["bull"]["text"]), "bull",
        f"Bear의 반박을 반영한 재반박·수정 · {stage_mode(result, 'neutral_summary')} · 검증 근거 연결",
    )
    render_chat_claim(
        naturalize_text(closing["bear"]["text"]), "bear",
        f"Bull의 수정 주장에 직접 답변 · {stage_mode(result, 'neutral_summary')} · 검증 근거 연결",
    )

    neutral_summary = neutral["summary"]

    st.markdown('<div class="section-kicker">Synthesis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">전체 토론 종합</div>', unsafe_allow_html=True)
    st.caption(f"API 5 · {stage_mode(result, 'neutral_summary')} · 어느 쪽의 승자도 결정하지 않습니다.")

    labels_by_group = (
        ("핵심 충돌 쟁점", "key_issues"),
        ("데이터로 확인된 내용", "confirmed_evidence"),
        ("일부만 뒷받침된 내용", "partially_supported"),
        ("근거 부족 · 불확실성", "uncertainties"),
    )
    st.markdown('<div class="summary-grid">', unsafe_allow_html=True)
    for label, key in labels_by_group:
        items = neutral_summary.get(key, [])
        body = "".join(f"<p>• {html.escape(naturalize_text(item['text']))}</p>" for item in items) or "<p>해당 항목이 없습니다.</p>"
        st.markdown(f'<div class="summary-card"><h4>{label}</h4>{body}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(naturalize_text(neutral["uncertainty_statement"]))

    st.markdown('<div class="section-kicker">Final hypotheses</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">토론을 통해 수정된 투자 가설</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">최종 카드는 유지된 Claim과 실제 Evidence 상태만 사용합니다.</div>', unsafe_allow_html=True)
    bull_col, bear_col = st.columns(2)
    revised = neutral["revised_hypotheses"]
    packet_lookup = evidence_packet_lookup(result)
    with bull_col:
        st.markdown('<h3 class="bull-heading">BULL · 상승 관점</h3>', unsafe_allow_html=True)
        if revised["bull"]:
            for index, item in enumerate(revised["bull"], 1):
                render_hypothesis_card(item, "bull", index, packet_lookup)
        else:
            st.caption("검증 후 유지된 Bull 최종 가설이 없습니다.")
    with bear_col:
        st.markdown('<h3 class="bear-heading">BEAR · 하락·위험 관점</h3>', unsafe_allow_html=True)
        if revised["bear"]:
            for index, item in enumerate(revised["bear"], 1):
                render_hypothesis_card(item, "bear", index, packet_lookup)
        else:
            st.caption("검증 후 유지된 Bear 최종 가설이 없습니다.")

    st.markdown('<div class="section-kicker">Evidence detail</div>', unsafe_allow_html=True)
    with st.expander("상세 Evidence / Fact Room", expanded=False):
        fact_tab, evidence_tab, citation_tab, model_tab = st.tabs(
            ["Fact Room", "Evidence Packets", "Citations", "Model 상태"]
        )
        with fact_tab:
            st.caption("Bull과 Bear가 같은 기준일의 동일한 자료를 사용했습니다.")
            render_fact_group("기업 실적과 재무 상태", summary["fundamental"])
            render_fact_group("주가와 시장 흐름", summary["market_context"])
            render_fact_group("거시경제 환경", summary["macro_context"])
            st.markdown("#### 최근 공식 공시")
            if summary["recent_disclosures"]:
                for disclosure in summary["recent_disclosures"]:
                    report_name = friendly_disclosure_name(
                        disclosure.get("report_name"),
                        disclosure.get("report_type"),
                    )
                    effective_date = disclosure.get("effective_date") or "확인 불가"
                    source_url = str(disclosure.get("source_url") or "")
                    st.markdown(f"**{html.escape(report_name)}** · 공시일자 {html.escape(str(effective_date))}")
                    if is_dart_source_url(source_url):
                        st.markdown(
                            f'<a class="dart-link" href="{html.escape(source_url, quote=True)}" '
                            f'target="_blank" rel="noopener noreferrer">DART 원문 보기 →</a>',
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("현재 기준일에 연결된 공식 공시가 없습니다.")
        with evidence_tab:
            for evaluation in result["claim_evaluations"]:
                st.markdown(
                    f"**{evaluation['source_agent'].upper()} 관점 · {friendly_verdict(evaluation['initial_verdict'])}**"
                )
                for packet in evaluation["evidence_packets"]:
                    render_evidence_packet(packet)
                if evaluation["final_evidence_packets"]:
                    st.caption("수정 Claim 재검증 Evidence")
                    for packet in evaluation["final_evidence_packets"]:
                        render_evidence_packet(packet)
                st.divider()
        with citation_tab:
            if result["citations"]:
                for citation in result["citations"]:
                    source_url = str(citation.get("source_url") or "")
                    if not is_dart_source_url(source_url):
                        continue
                    report_name = friendly_disclosure_name(citation.get("report_name"))
                    effective_date = str(citation.get("effective_date") or "확인 불가")
                    st.markdown(
                        f'<div class="source-item"><strong>{html.escape(report_name)}</strong>'
                        f'<div class="source-date">공시일자 {html.escape(effective_date)}</div>'
                        f'<a class="dart-link" href="{html.escape(source_url, quote=True)}" '
                        f'target="_blank" rel="noopener noreferrer">DART 원문 보기 →</a></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("연결된 공식 공시 Citation이 없습니다.")
        with model_tab:
            ml = result["ml_context"]
            ml_status = "과거 시장 반응 패턴 보조 분석 사용" if ml.get("available") else "보조 분석 자료 없음"
            st.write(f"모델 보조 분석: **{ml_status}** · 기준일 **{ml.get('as_of_date', '-')}**")
            stage_status = result["execution_metadata"].get("llm_stage_status", {})
            actual_count = sum(item.get("mode") == "actual_llm" for item in stage_status.values())
            st.write(f"Gemini 실제 실행: **{actual_count} / {len(stage_status)} 단계**")
            safety_status = "통과" if result["safety_gates"]["status"] == "PASS" else "확인 필요"
            st.write(f"안전성 검사: **{safety_status}**")

st.divider()
st.caption("BVB는 특정 투자 행동이나 가격을 제시하지 않는 근거 검증 프로토타입입니다.")
