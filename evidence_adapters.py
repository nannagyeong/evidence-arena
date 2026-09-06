# Frozen STEP 8/10/11 evidence adapters used by STEP 14.
from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import statistical_validator as step8

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
RAG_DIR = DATA_ROOT / "rag_evidence"
FULL_CHUNK_DIR = RAG_DIR / "full_corpus" / "chunks_by_document"

frozen_policy_artifact = json.loads((RAG_DIR / "09_frozen_retrieval_policy.json").read_text(encoding="utf-8"))
FROZEN_POLICY_HASH = frozen_policy_artifact["policy_hash"]
FINANCIAL_SYNONYM_DICTIONARY = frozen_policy_artifact["frozen_policy"]["query_expansion_dictionary"]
SECTION_PRIORITY_BY_TOPIC = frozen_policy_artifact["frozen_policy"]["section_priority"]
HYBRID_SCORE_WEIGHTS = frozen_policy_artifact["frozen_policy"]["retrieval_weights"]
INITIAL_VECTOR_RELEVANCE_FLOOR = float(frozen_policy_artifact["frozen_policy"]["relevance_floor"])

document_manifest = pd.read_parquet(RAG_DIR / "13_full_document_manifest.parquet")
document_manifest["stock_code"] = document_manifest["stock_code"].astype(str).str.zfill(6)
document_manifest["rcept_no"] = document_manifest["rcept_no"].astype(str).str.zfill(14)
document_manifest["effective_date"] = pd.to_datetime(document_manifest["effective_date"], errors="coerce")
parse_audit = pd.read_csv(RAG_DIR / "14_full_parse_audit.csv", dtype={"stock_code": "string", "rcept_no": "string"})
parse_audit["stock_code"] = parse_audit["stock_code"].str.zfill(6)
parse_audit["rcept_no"] = parse_audit["rcept_no"].str.zfill(14)
parse_status = parse_audit[["rcept_no", "parse_status"]].drop_duplicates("rcept_no", keep="last")
full_parse_enriched = document_manifest.drop(columns=["parse_status"], errors="ignore").merge(
    parse_status, on="rcept_no", how="left", validate="one_to_one"
)
ALLOWED_REQUEST_REPORT_TYPES = sorted(document_manifest["report_type"].dropna().astype(str).unique().tolist())

def normalize_block_text(value: str) -> str:
    text = re.sub(r"&cr;?", " ", str(value or ""), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()

def validate_retrieval_request(request: dict) -> dict:
    required = {"stock_code", "as_of_date", "query", "evidence_type", "top_k"}
    missing = required - set(request)
    if missing:
        raise ValueError(f"Missing request fields: {sorted(missing)}")
    normalized = dict(request)
    normalized["stock_code"] = str(normalized["stock_code"]).zfill(6)
    if not re.fullmatch(r"\d{6}", normalized["stock_code"]):
        raise ValueError("stock_code must be six digits")
    normalized["as_of_date"] = pd.Timestamp(normalized["as_of_date"]).normalize()
    normalized["query"] = str(normalized["query"]).strip()
    if len(normalized["query"]) < 2:
        raise ValueError("query is too short")
    if normalized["evidence_type"] != "qualitative_disclosure":
        raise ValueError("STEP 11 supports qualitative_disclosure only")
    normalized["top_k"] = int(normalized["top_k"])
    if not 1 <= normalized["top_k"] <= 20:
        raise ValueError("top_k must be between 1 and 20")
    normalized["report_types"] = list(normalized.get("report_types") or [])
    unknown = set(normalized["report_types"]) - set(ALLOWED_REQUEST_REPORT_TYPES)
    if unknown:
        raise ValueError(f"Unknown report_types: {sorted(unknown)}")
    return normalized

def infer_query_topics(query: str) -> list[str]:
    topics = []
    lowered = query.lower()
    for topic, terms in FINANCIAL_SYNONYM_DICTIONARY.items():
        if any(term.lower() in lowered for term in terms) or topic.lower() in lowered:
            topics.append(topic)
    return topics

def expand_query_controlled(query: str) -> tuple[list[str], list[str]]:
    topics = infer_query_topics(query)
    terms = [query]
    for topic in topics:
        terms.extend(FINANCIAL_SYNONYM_DICTIONARY[topic])
    return list(dict.fromkeys(term for term in terms if term)), topics

def tokenize_financial_text(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[0-9A-Za-z가-힣]+", str(text)) if len(token) > 1]

def bm25_scores(corpus: list[str], query_terms: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized = [tokenize_financial_text(text) for text in corpus]
    query_tokens = list(dict.fromkeys(token for term in query_terms for token in tokenize_financial_text(term)))
    if not corpus or not query_tokens:
        return np.zeros(len(corpus), dtype=float)
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    avg_length = np.mean([len(tokens) for tokens in tokenized]) or 1.0
    scores = np.zeros(len(tokenized), dtype=float)
    for index, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        length = len(tokens)
        for token in query_tokens:
            frequency = counts[token]
            if not frequency:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (len(tokenized) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / avg_length)
            scores[index] += idf * frequency * (k1 + 1.0) / denominator
    return scores

def minmax_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values) or np.nanmax(values) <= 0:
        return np.zeros_like(values)
    minimum, maximum = np.nanmin(values), np.nanmax(values)
    if maximum == minimum:
        return np.ones_like(values)
    return (values - minimum) / (maximum - minimum)

def pit_filter_latest_versions(chunks: pd.DataFrame, stock_code: str, as_of_date: pd.Timestamp, report_types=None) -> pd.DataFrame:
    if chunks.empty:
        return chunks.copy()
    candidates = chunks.loc[
        chunks["stock_code"].astype(str).str.zfill(6).eq(stock_code)
        & pd.to_datetime(chunks["effective_date"]).le(as_of_date)
    ].copy()
    if report_types:
        candidates = candidates.loc[candidates["report_type"].isin(report_types)].copy()
    if candidates.empty:
        return candidates
    allowed_versions = (
        candidates[["report_slot_id", "version_order", "rcept_no", "effective_date"]]
        .drop_duplicates().sort_values(["report_slot_id", "version_order", "effective_date", "rcept_no"])
        .groupby("report_slot_id", as_index=False).tail(1)[["report_slot_id", "rcept_no"]]
    )
    return candidates.merge(allowed_versions, on=["report_slot_id", "rcept_no"], how="inner", validate="many_to_one")

def exact_extractive_excerpt(text: str, query_terms: list[str], max_chars: int = 500) -> str:
    normalized = normalize_block_text(text)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?다])\s+", normalized) if sentence.strip()]
    for sentence in sentences:
        if any(term.lower() in sentence.lower() for term in query_terms):
            return sentence[:max_chars]
    return normalized[:max_chars]

def production_document_selection(stock_code: str, as_of_date, report_types=None) -> pd.DataFrame:
    stock_code = str(stock_code).zfill(6)
    as_of_date = pd.Timestamp(as_of_date).normalize()
    candidates = full_parse_enriched.loc[
        full_parse_enriched["stock_code"].eq(stock_code)
        & full_parse_enriched["fetch_status"].eq("available_local")
        & full_parse_enriched["parse_status"].eq("success")
        & full_parse_enriched["effective_date"].le(as_of_date)
    ].copy()
    if report_types:
        candidates = candidates.loc[candidates["report_type"].isin(report_types)].copy()
    return candidates.sort_values(["report_slot_id", "version_order", "effective_date", "rcept_no"]).groupby("report_slot_id", as_index=False).tail(1)

@lru_cache(maxsize=128)
def _load_production_chunks_cached(stock_code: str, as_of_date_iso: str, report_types_key: tuple[str, ...]) -> pd.DataFrame:
    selected = production_document_selection(stock_code, as_of_date_iso, list(report_types_key))
    frames = []
    for row in selected.itertuples(index=False):
        path = FULL_CHUNK_DIR / str(row.stock_code).zfill(6) / f"{str(row.rcept_no).zfill(14)}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    chunks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not chunks.empty:
        assert chunks["stock_code"].astype(str).str.zfill(6).eq(str(stock_code).zfill(6)).all()
        assert pd.to_datetime(chunks["effective_date"]).le(pd.Timestamp(as_of_date_iso)).all()
    return chunks

def load_production_chunks(stock_code: str, as_of_date, report_types=None) -> pd.DataFrame:
    return _load_production_chunks_cached(
        str(stock_code).zfill(6), str(pd.Timestamp(as_of_date).date()), tuple(report_types or [])
    ).copy()

def retrieve_disclosure_evidence(request: dict, chunks: pd.DataFrame, score_weights=None, relevance_floor=None) -> dict:
    normalized = validate_retrieval_request(request)
    active_weights = dict(score_weights or HYBRID_SCORE_WEIGHTS)
    active_floor = float(INITIAL_VECTOR_RELEVANCE_FLOOR if relevance_floor is None else relevance_floor)
    if not math.isclose(sum(active_weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Hybrid score weights must sum to 1")
    query_terms, topics = expand_query_controlled(normalized["query"])
    allowed = pit_filter_latest_versions(chunks, normalized["stock_code"], normalized["as_of_date"], normalized["report_types"])
    base = {
        "stock_code": normalized["stock_code"], "as_of_date": normalized["as_of_date"].date().isoformat(),
        "query": normalized["query"], "expanded_topics": topics, "expanded_terms": query_terms,
        "final_verdict": None, "verdict_owner": "STEP 13", "corpus_scope": "full_P0_P1_corpus",
        "no_evidence_interpretation": "현재 수집·파싱 완료된 P0/P1 전체 Corpus 범위에서 관련 Evidence를 확인하지 못함",
        "retrieval_policy": {"score_weights": active_weights, "relevance_floor": active_floor},
        "frozen_policy_hash": FROZEN_POLICY_HASH,
    }
    if allowed.empty:
        return {**base, "retrieval_status": "no_document_available", "candidate_count": 0, "evidence": []}
    corpus = allowed["chunk_text"].fillna("").astype(str).tolist()
    bm25_raw = bm25_scores(corpus, query_terms)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(corpus + [" ".join(query_terms)])
    vector_raw = cosine_similarity(tfidf[:-1], tfidf[-1]).ravel()
    keyword_hits = np.asarray([sum(term.lower() in text.lower() for term in query_terms) for text in corpus], dtype=int)
    priority_terms = [term for topic in topics for term in SECTION_PRIORITY_BY_TOPIC.get(topic, [])]
    section_raw = np.asarray([float(any(term.lower() in str(path).lower() for term in priority_terms)) for path in allowed["section_path"].fillna("")])
    scored = allowed.copy()
    scored["bm25_score"] = bm25_raw
    scored["vector_score"] = vector_raw
    scored["keyword_hits"] = keyword_hits
    scored["section_priority_score"] = section_raw
    scored["hybrid_score"] = active_weights["bm25"] * minmax_positive(bm25_raw) + active_weights["char_vector"] * minmax_positive(vector_raw) + active_weights["section_priority"] * section_raw
    relevant = scored.loc[scored["keyword_hits"].gt(0) | scored["vector_score"].ge(active_floor)].sort_values(
        ["hybrid_score", "rag_priority", "effective_date", "rcept_no"], ascending=[False, True, False, False]
    )
    if relevant.empty:
        return {**base, "retrieval_status": "insufficient_relevance", "candidate_count": len(allowed), "evidence": []}
    evidence = []
    for rank, row in enumerate(relevant.head(normalized["top_k"]).itertuples(index=False), 1):
        excerpt = exact_extractive_excerpt(row.chunk_text, query_terms)
        evidence.append({
            "rank": rank, "chunk_id": row.chunk_id, "summary": excerpt,
            "summary_method": "extractive_query_sentence", "source_excerpt": excerpt,
            "company_name": row.company_name, "report_name": row.report_nm,
            "report_type": row.report_type, "effective_date": pd.Timestamp(row.effective_date).date().isoformat(),
            "section": row.section, "subsection": row.subsection, "section_path": row.section_path,
            "rcept_no": str(row.rcept_no), "source_url": row.source_url,
            "report_slot_id": row.report_slot_id, "version_id": row.version_id,
            "hybrid_score": float(row.hybrid_score), "bm25_score": float(row.bm25_score),
            "vector_score": float(row.vector_score), "section_priority_score": float(row.section_priority_score),
        })
    return {**base, "retrieval_status": "success", "candidate_count": len(allowed), "evidence": evidence}

def retrieve_production_disclosure_evidence(request: dict) -> dict:
    normalized = validate_retrieval_request(request)
    chunks = load_production_chunks(normalized["stock_code"], normalized["as_of_date"], normalized["report_types"])
    return retrieve_disclosure_evidence(request, chunks, HYBRID_SCORE_WEIGHTS, INITIAL_VECTOR_RELEVANCE_FLOOR)

def execute_evidence_routes(routed_items: list[dict]) -> dict[str, dict]:
    step8_inputs = []
    for item in routed_items:
        claim = item["validated_claim"]
        template = claim["template_id"]
        executable = any(route["engine"] == template and route["execution_allowed"] for route in claim["routes"])
        if executable and item.get("step8_claim") is not None:
            step8_inputs.append(item["step8_claim"])
    step8_results = step8.run_claim_batch(step8_inputs) if step8_inputs else []
    step8_by_claim = {result["claim_id"]: result for result in step8_results}
    outputs = {}
    for item in routed_items:
        claim = item["validated_claim"]
        raw_by_engine = {}
        if claim["claim_id"] in step8_by_claim:
            raw_by_engine[claim["template_id"]] = step8_by_claim[claim["claim_id"]]
        for route in claim["routes"]:
            if not route["execution_allowed"]:
                continue
            if route["engine"] == "RAG":
                raw_by_engine["RAG"] = retrieve_production_disclosure_evidence({
                    "stock_code": claim["stock_code"], "as_of_date": claim["as_of_date"],
                    "query": claim["atomic_claim_text"], "evidence_type": "qualitative_disclosure",
                    "report_types": [], "top_k": 5,
                })
        outputs[claim["claim_id"]] = raw_by_engine
    return outputs

UPSTREAM_PROVENANCE = {
    "STEP8": step8.UPSTREAM_PROVENANCE,
    "STEP11_policy_hash": FROZEN_POLICY_HASH,
    "STEP11_retrieval": "PIT filter -> latest version -> frozen BM25 + char TF-IDF + section boost",
    "STEP10_adapter": str(RAG_DIR / "16_step10_ml_evidence_adapter_2025.parquet"),
}
