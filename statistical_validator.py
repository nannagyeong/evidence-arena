# Auto-generated from the frozen STEP 8 implementation. Do not edit by hand.
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import warnings

import jsonschema
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
FEATURE_DIR = DATA_ROOT / "features"
TARGET_DIR = DATA_ROOT / "targets"
PIT_DIR = DATA_ROOT / "pit"
CLAIM_DIR = DATA_ROOT / "claim_validator"

FEATURE_PANEL_PATH = FEATURE_DIR / "feature_panel.parquet"
TARGET_PANEL_PATH = TARGET_DIR / "target_panel.parquet"
FEATURE_REGISTRY_PATH = FEATURE_DIR / "01_feature_registry.csv"
FINANCIAL_SNAPSHOT_PATH = FEATURE_DIR / "lineage" / "04_financial_report_snapshot.parquet"
DISCLOSURE_PIT_PATH = PIT_DIR / "disclosure_pit.parquet"
PIT_MASTER_PATH = PIT_DIR / "pit_master_base.parquet"

KEY_COLUMNS = ["stock_code", "trading_date"]
CLAIM_DEVELOPMENT_CUTOFF = pd.Timestamp("2024-12-30")
SUPPORTED_HORIZON = 5
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_ITERATIONS = 1000
BLOCK_LENGTH_5D = 5
HAC_MAXLAGS_5D = 4
FDR_ALPHA = 0.05

feature_panel_full_step8 = pd.read_parquet(FEATURE_PANEL_PATH)
target_panel_full_step8 = pd.read_parquet(TARGET_PANEL_PATH)
feature_registry_step8 = pd.read_csv(FEATURE_REGISTRY_PATH)
financial_snapshot_full_step8 = pd.read_parquet(FINANCIAL_SNAPSHOT_PATH)
disclosure_full_step8 = pd.read_parquet(DISCLOSURE_PIT_PATH)
pit_market_full_step8 = pd.read_parquet(
    PIT_MASTER_PATH,
    columns=["stock_code", "trading_date", "company_name", "market", "adjusted_close", "index_close"],
)
for frame, date_column in [
    (feature_panel_full_step8, "trading_date"),
    (target_panel_full_step8, "trading_date"),
    (financial_snapshot_full_step8, "effective_date"),
    (disclosure_full_step8, "effective_date"),
    (pit_market_full_step8, "trading_date"),
]:
    frame["stock_code"] = frame["stock_code"].astype("string").str.zfill(6)
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")

feature_panel_step8 = feature_panel_full_step8.loc[
    feature_panel_full_step8["trading_date"].le(CLAIM_DEVELOPMENT_CUTOFF)
].copy()
target_panel_step8 = target_panel_full_step8.loc[
    target_panel_full_step8["trading_date"].le(CLAIM_DEVELOPMENT_CUTOFF)
].copy()
financial_snapshot_step8 = financial_snapshot_full_step8.loc[
    financial_snapshot_full_step8["effective_date"].le(CLAIM_DEVELOPMENT_CUTOFF)
].copy()
disclosure_step8 = disclosure_full_step8.loc[
    disclosure_full_step8["effective_date"].le(CLAIM_DEVELOPMENT_CUTOFF)
].copy()
pit_market_step8 = pit_market_full_step8.loc[
    pit_market_full_step8["trading_date"].le(CLAIM_DEVELOPMENT_CUTOFF)
].copy()

target_contract_columns = [
    *KEY_COLUMNS, "target_end_date_5d", "future_excess_return_5d",
    "primary_target_available", "market_transition_within_horizon",
]
claim_panel = feature_panel_step8.merge(
    target_panel_step8[target_contract_columns], on=KEY_COLUMNS, how="inner", validate="one_to_one"
).sort_values(KEY_COLUMNS).reset_index(drop=True)
company_master_step8 = (
    feature_panel_step8[["stock_code", "company_name", "market", "industry", "is_finance_industry"]]
    .drop_duplicates("stock_code").sort_values("stock_code").reset_index(drop=True)
)

feature_registry_step8["feature_name"] = feature_registry_step8["feature_name"].astype(str)
claim_registry_features = feature_registry_step8.loc[
    feature_registry_step8["use_for_claim"].fillna(False).astype(bool), "feature_name"
].tolist()
numeric_claim_features = [
    feature for feature in claim_registry_features
    if feature in claim_panel.columns and pd.api.types.is_numeric_dtype(claim_panel[feature])
]
financial_snapshot_features = [
    feature for feature in [
        "revenue_yoy", "operating_income_yoy", "net_income_yoy", "operating_margin",
        "operating_margin_change_yoy", "debt_ratio", "cfo_to_assets", "assets",
        "liabilities", "equity", "eps",
    ] if feature in financial_snapshot_step8.columns
]
T5_FEATURE_MAPPING = {
    "revenue_yoy": {"evaluation_mode": "direct_signed_change", "signal_feature": "revenue_yoy", "unit": "ratio"},
    "operating_income_yoy": {"evaluation_mode": "direct_signed_change", "signal_feature": "operating_income_yoy", "unit": "ratio"},
    "net_income_yoy": {"evaluation_mode": "direct_signed_change", "signal_feature": "net_income_yoy", "unit": "ratio"},
    "operating_margin_change_yoy": {"evaluation_mode": "direct_signed_change", "signal_feature": "operating_margin_change_yoy", "unit": "ratio_point_change"},
    "operating_margin": {"evaluation_mode": "mapped_yoy_change", "signal_feature": "operating_margin_change_yoy", "unit": "ratio"},
}
T5_ALLOWED_FEATURES = list(T5_FEATURE_MAPPING)
PIT_FINANCIAL_DAILY_FEATURE_DATE = {"fy_roe": "fy_financial_effective_date", "fy_roa": "fy_financial_effective_date"}
PREDEFINED_CONTROL_SETS = {
    "risk_market_core": ("realized_vol_20", "market_ret_20"),
    "risk_market_momentum": ("realized_vol_20", "market_ret_20", "excess_ret_20"),
    "risk_market_macro": ("realized_vol_20", "market_ret_20", "base_rate_change_60", "treasury_3y_change_20"),
}
TEMPLATE_CONFIG = {
    "T1": {"claim_type": "conditional_return_difference", "allowed_features": numeric_claim_features, "outcome": "future_excess_return_5d", "primary_inference_method": "moving_block_bootstrap_mean_difference", "robustness_methods": ["Welch", "Mann-Whitney", "5D_non_overlapping_direction"], "min_sample": 30, "fdr_applicable": True},
    "T2": {"claim_type": "continuous_relation", "allowed_features": numeric_claim_features, "outcome": "future_excess_return_5d", "primary_inference_method": "univariate_OLS-HAC(maxlags=4)", "robustness_methods": ["Spearman_with_moving_block_bootstrap_CI"], "min_sample": 50, "fdr_applicable": True},
    "T3": {"claim_type": "controlled_relation", "allowed_features": numeric_claim_features, "outcome": "future_excess_return_5d", "primary_inference_method": "OLS-HAC(maxlags=4)", "robustness_methods": ["Spearman_uncontrolled_direction"], "min_sample": 80, "fdr_applicable": True},
    "T4": {"claim_type": "event_reaction", "allowed_features": ["operating_margin_change_yoy"], "outcome": "future_excess_return_5d", "primary_inference_method": "clean_event_cluster_bootstrap_mean_CAR_0_4", "robustness_methods": ["all_event_clusters_including_overlap", "CAR_0_1"], "min_sample": 3, "fdr_applicable": True},
    "T5": {"claim_type": "fact_trend", "allowed_features": T5_ALLOWED_FEATURES, "outcome": "fact_value", "primary_inference_method": "PIT_fact_check", "robustness_methods": [], "min_sample": 1, "fdr_applicable": False},
    "T6": {"claim_type": "relative_position", "allowed_features": numeric_claim_features, "outcome": "historical_percentile", "primary_inference_method": "PIT_historical_or_peer_percentile", "robustness_methods": [], "min_sample": 20, "fdr_applicable": False},
}
CLAIM_TYPE_TO_TEMPLATE = {config["claim_type"]: template for template, config in TEMPLATE_CONFIG.items()}

CLAIM_JSON_SCHEMA = json.loads((CLAIM_DIR / "01_claim_json_schema.json").read_text(encoding="utf-8"))
CLAIM_SCHEMA_VALIDATOR = jsonschema.Draft202012Validator(CLAIM_JSON_SCHEMA, format_checker=jsonschema.FormatChecker())
EVIDENCE_RESULT_SCHEMA = json.loads((CLAIM_DIR / "06_evidence_result_schema.json").read_text(encoding="utf-8"))
EVIDENCE_SCHEMA_VALIDATOR = jsonschema.Draft202012Validator(EVIDENCE_RESULT_SCHEMA)

class ClaimValidationError(ValueError):
    pass

feature_policy_lookup = feature_registry_step8.set_index("feature_name").to_dict("index")
company_lookup_step8 = company_master_step8.set_index("stock_code").to_dict("index")

def step8_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def step8_json_default(value):
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def to_builtin(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=step8_json_default))


def validate_claim_json_schema(claim: dict) -> None:
    errors = sorted(CLAIM_SCHEMA_VALIDATOR.iter_errors(claim), key=lambda error: list(error.path))
    if errors:
        message = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ClaimValidationError(f"claim_json_schema_error: {message}")


def stable_claim_seed(claim_id: str) -> int:
    suffix = int(hashlib.sha256(claim_id.encode("utf-8")).hexdigest()[:8], 16)
    return int((BOOTSTRAP_SEED + suffix) % (2**32 - 1))


def direction_from_effect(effect: float | None, tolerance: float = 1e-15) -> str:
    if effect is None or not np.isfinite(effect):
        return "not_applicable"
    if effect > tolerance:
        return "positive"
    if effect < -tolerance:
        return "negative"
    return "neutral"


def spearman_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    return float(np.dot(x_centered, y_centered) / denominator) if denominator > 0 else np.nan


def spearman_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = spearman_statistic(x, y)
    n = len(x)
    if not np.isfinite(rho) or n <= 2:
        return np.nan, np.nan
    if abs(rho) >= 1:
        return float(rho), 0.0
    t_stat = rho * math.sqrt((n - 2) / max(1 - rho**2, np.finfo(float).eps))
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
    return float(rho), p_value


def invert_small_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    augmented = [
        list(matrix[row]) + [1.0 if row == column else 0.0 for column in range(n)]
        for row in range(n)
    ]
    for pivot_column in range(n):
        pivot_row = max(
            range(pivot_column, n),
            key=lambda row: abs(augmented[row][pivot_column]),
        )
        if abs(augmented[pivot_row][pivot_column]) < 1e-12:
            raise ClaimValidationError("singular_ols_design_matrix")
        augmented[pivot_column], augmented[pivot_row] = (
            augmented[pivot_row], augmented[pivot_column]
        )
        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot for value in augmented[pivot_column]]
        for row in range(n):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[pivot_column])
            ]
    return np.asarray([row[n:] for row in augmented], dtype=float)


def multiply_small_matrices(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    result = np.zeros((left.shape[0], right.shape[1]), dtype=float)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            result[row, column] = sum(
                left[row, inner] * right[inner, column]
                for inner in range(left.shape[1])
            )
    return result


def fit_ols_hac(
    y: np.ndarray,
    predictors: pd.DataFrame,
    maxlags: int,
) -> dict:
    predictor_names = predictors.columns.astype(str).tolist()
    predictor_values = predictors.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(predictor_values)), predictor_values])
    names = ["const", *predictor_names]
    k = X.shape[1]
    xtx = np.zeros((k, k), dtype=float)
    xty = np.zeros(k, dtype=float)
    for i in range(k):
        xty[i] = float(np.sum(X[:, i] * y))
        for j in range(k):
            xtx[i, j] = float(np.sum(X[:, i] * X[:, j]))
    xtx_inverse = invert_small_matrix(xtx)
    beta = np.array([
        sum(xtx_inverse[i, j] * xty[j] for j in range(k))
        for i in range(k)
    ])
    fitted = np.array([
        sum(X[row, column] * beta[column] for column in range(k))
        for row in range(len(X))
    ])
    residual = y - fitted
    score = X * residual[:, None]
    meat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            meat[i, j] = float(np.sum(score[:, i] * score[:, j]))
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        for i in range(k):
            for j in range(k):
                cross_ij = float(np.sum(score[lag:, i] * score[:-lag, j]))
                cross_ji = float(np.sum(score[lag:, j] * score[:-lag, i]))
                meat[i, j] += weight * (cross_ij + cross_ji)
    covariance = multiply_small_matrices(
        multiply_small_matrices(xtx_inverse, meat), xtx_inverse
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_statistics = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    p_values = np.array([
        math.erfc(abs(value) / math.sqrt(2)) if np.isfinite(value) else np.nan
        for value in z_statistics
    ])
    ci_low = beta - 1.959963984540054 * standard_errors
    ci_high = beta + 1.959963984540054 * standard_errors
    diagonal = np.diag(xtx)
    diagonal_ratio = float(diagonal.max() / diagonal.min()) if diagonal.min() > 0 else np.inf
    return {
        "coefficients": dict(zip(names, map(float, beta))),
        "standard_errors": dict(zip(names, map(float, standard_errors))),
        "p_values": dict(zip(names, map(float, p_values))),
        "ci_low": dict(zip(names, map(float, ci_low))),
        "ci_high": dict(zip(names, map(float, ci_high))),
        "xtx_diagonal_ratio": diagonal_ratio,
        "maxlags": int(maxlags),
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def apply_operator(values: pd.Series, operator: str, threshold: float) -> pd.Series:
    operators = {
        ">": values.gt, ">=": values.ge, "<": values.lt,
        "<=": values.le, "==": values.eq,
    }
    if operator not in operators:
        raise ClaimValidationError(f"unsupported_condition_operator: {operator}")
    return operators[operator](threshold)


def apply_operator_scalar(value: float, operator: str, threshold: float) -> bool:
    return bool(apply_operator(pd.Series([value]), operator, threshold).iloc[0])


def target_history_for_claim(claim: dict, columns: list[str]) -> pd.DataFrame:
    as_of = pd.Timestamp(claim["as_of_date"])
    required = list(dict.fromkeys([
        "stock_code", "trading_date", "target_end_date_5d",
        "future_excess_return_5d", "primary_target_available", *columns,
    ]))
    sample = claim_panel.loc[
        claim_panel["stock_code"].eq(claim["stock_code"])
        & claim_panel["trading_date"].le(as_of)
        & claim_panel["target_end_date_5d"].notna()
        & claim_panel["target_end_date_5d"].le(as_of)
        & claim_panel["primary_target_available"],
        required,
    ].copy()
    return sample.sort_values("trading_date").reset_index(drop=True)


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks_needed = int(math.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks_needed)
    indices = np.concatenate([
        (start + np.arange(block_length)) % n for start in starts
    ])[:n]
    return indices


def block_bootstrap_difference(
    condition: np.ndarray,
    outcome: np.ndarray,
    seed: int,
) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(seed)
    bootstrap_effects = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        index = moving_block_indices(len(outcome), BLOCK_LENGTH_5D, rng)
        sampled_condition = condition[index]
        sampled_outcome = outcome[index]
        if sampled_condition.any() and (~sampled_condition).any():
            bootstrap_effects.append(
                sampled_outcome[sampled_condition].mean()
                - sampled_outcome[~sampled_condition].mean()
            )
    effects = np.asarray(bootstrap_effects, dtype=float)
    if effects.size == 0:
        return np.nan, np.nan, np.nan, 0
    ci_low, ci_high = np.quantile(effects, [0.025, 0.975])
    p_value = 2 * min(
        (np.sum(effects <= 0) + 1) / (len(effects) + 1),
        (np.sum(effects >= 0) + 1) / (len(effects) + 1),
    )
    return float(ci_low), float(ci_high), float(min(p_value, 1.0)), int(len(effects))


def block_bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        index = moving_block_indices(len(y), BLOCK_LENGTH_5D, rng)
        rho = spearman_statistic(x[index], y[index])
        if np.isfinite(rho):
            estimates.append(rho)
    if not estimates:
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def hedges_like_standardized_difference(group_a: np.ndarray, group_b: np.ndarray) -> float:
    pooled = math.sqrt((np.var(group_a, ddof=1) + np.var(group_b, ddof=1)) / 2)
    return float((np.mean(group_a) - np.mean(group_b)) / pooled) if pooled > 0 else np.nan


def result_common(claim: dict, template_id: str, sample: pd.DataFrame) -> dict:
    return {
        "claim": claim,
        "template_id": template_id,
        "status": "ok",
        "sample_size": int(len(sample)),
        "analysis_start": sample["trading_date"].min() if "trading_date" in sample else None,
        "analysis_end": sample["trading_date"].max() if "trading_date" in sample else None,
        "primary_inference_method": TEMPLATE_CONFIG[template_id]["primary_inference_method"],
        "methods": [
            TEMPLATE_CONFIG[template_id]["primary_inference_method"],
            *TEMPLATE_CONFIG[template_id]["robustness_methods"],
        ],
        "fdr_applicable": TEMPLATE_CONFIG[template_id]["fdr_applicable"],
        "q_value": None,
        "fdr_pass": None,
    }


def validate_t1(claim: dict) -> dict:
    sample = target_history_for_claim(claim, [claim["x_feature"]]).dropna(
        subset=[claim["x_feature"], "future_excess_return_5d"]
    )
    finite = np.isfinite(sample[claim["x_feature"]]) & np.isfinite(sample["future_excess_return_5d"])
    sample = sample.loc[finite].reset_index(drop=True)
    condition = apply_operator(
        sample[claim["x_feature"]], claim["condition_operator"], claim["condition_value"]
    ).to_numpy(dtype=bool)
    y = sample["future_excess_return_5d"].to_numpy(dtype=float)
    group_a, group_b = y[condition], y[~condition]
    minimum = TEMPLATE_CONFIG["T1"]["min_sample"]
    result = result_common(claim, "T1", sample)
    if min(len(group_a), len(group_b)) < minimum:
        result.update({
            "status": "insufficient", "effect_size": None, "ci_95": [None, None],
            "primary_p_value": None, "direction_observed": "not_applicable",
            "robustness_direction_consistent": None,
            "limitations": [f"minimum {minimum} observations required in each group"],
            "details": {"n_A": len(group_a), "n_B": len(group_b)},
        })
        return result
    effect = float(group_a.mean() - group_b.mean())
    ci_low, ci_high, block_p, successful_bootstraps = block_bootstrap_difference(
        condition, y, stable_claim_seed(claim["claim_id"])
    )
    welch = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
    mwu = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    non_overlap = sample.iloc[::BLOCK_LENGTH_5D].copy()
    non_overlap_condition = apply_operator(
        non_overlap[claim["x_feature"]], claim["condition_operator"], claim["condition_value"]
    )
    non_overlap_effect = float(
        non_overlap.loc[non_overlap_condition, "future_excess_return_5d"].mean()
        - non_overlap.loc[~non_overlap_condition, "future_excess_return_5d"].mean()
    )
    result.update({
        "effect_size": effect,
        "ci_95": [ci_low, ci_high],
        "primary_p_value": block_p,
        "direction_observed": direction_from_effect(effect),
        "robustness_direction_consistent": direction_from_effect(effect) == direction_from_effect(non_overlap_effect),
        "limitations": ["historical association, not causality", "overlapping 5D target handled with moving blocks"],
        "details": {
            "n_A": int(len(group_a)), "n_B": int(len(group_b)),
            "mean_A": float(group_a.mean()), "mean_B": float(group_b.mean()),
            "mean_difference": effect,
            "median_A": float(np.median(group_a)), "median_B": float(np.median(group_b)),
            "median_difference": float(np.median(group_a) - np.median(group_b)),
            "standardized_effect": hedges_like_standardized_difference(group_a, group_b),
            "welch_p": float(welch.pvalue), "mann_whitney_p": float(mwu.pvalue),
            "non_overlapping_effect": non_overlap_effect,
            "successful_bootstraps": successful_bootstraps,
            "block_length": BLOCK_LENGTH_5D,
        },
        "diagnostics": {
            "max_target_end_date": sample["target_end_date_5d"].max(),
            "max_trading_date": sample["trading_date"].max(),
        },
    })
    return result


def validate_t2(claim: dict) -> dict:
    sample = target_history_for_claim(claim, [claim["x_feature"]]).dropna(
        subset=[claim["x_feature"], "future_excess_return_5d"]
    )
    finite = np.isfinite(sample[claim["x_feature"]]) & np.isfinite(sample["future_excess_return_5d"])
    sample = sample.loc[finite].reset_index(drop=True)
    result = result_common(claim, "T2", sample)
    minimum = TEMPLATE_CONFIG["T2"]["min_sample"]
    if len(sample) < minimum or sample[claim["x_feature"]].nunique() < 3:
        result.update({
            "status": "insufficient", "effect_size": None, "ci_95": [None, None],
            "primary_p_value": None, "direction_observed": "not_applicable",
            "robustness_direction_consistent": None,
            "limitations": [f"minimum {minimum} observations and variable variation required"],
            "details": {},
        })
        return result
    x = sample[claim["x_feature"]].to_numpy(dtype=float)
    y = sample["future_excess_return_5d"].to_numpy(dtype=float)

    # Primary: 5D 중첩에 대응하는 단변량 OLS-HAC.
    ols = fit_ols_hac(
        y,
        pd.DataFrame({claim["x_feature"]: x}),
        HAC_MAXLAGS_5D,
    )
    beta = ols["coefficients"][claim["x_feature"]]
    beta_p = ols["p_values"][claim["x_feature"]]
    beta_ci_low = ols["ci_low"][claim["x_feature"]]
    beta_ci_high = ols["ci_high"][claim["x_feature"]]

    # Robustness: 순위 기반 Spearman과 Moving Block Bootstrap CI.
    spearman_rho, spearman_naive_p = spearman_test(x, y)
    spearman_ci_low, spearman_ci_high = block_bootstrap_spearman(
        x, y, stable_claim_seed(claim["claim_id"])
    )
    result.update({
        "effect_size": beta,
        "ci_95": [beta_ci_low, beta_ci_high],
        "primary_p_value": beta_p,
        "direction_observed": direction_from_effect(beta),
        "robustness_direction_consistent": (
            direction_from_effect(beta) == direction_from_effect(spearman_rho)
        ),
        "limitations": [
            "historical association, not causality",
            "primary inference uses HAC maxlags fixed before execution",
            "Spearman p-value is descriptive only and is excluded from FDR",
        ],
        "details": {
            "ols_hac_beta": beta,
            "ols_hac_p": beta_p,
            "ols_hac_ci_low": beta_ci_low,
            "ols_hac_ci_high": beta_ci_high,
            "spearman_rho": float(spearman_rho),
            "spearman_naive_p_descriptive_only": float(spearman_naive_p),
            "spearman_block_ci_low": spearman_ci_low,
            "spearman_block_ci_high": spearman_ci_high,
            "hac_maxlags": HAC_MAXLAGS_5D,
            "block_length": BLOCK_LENGTH_5D,
        },
        "diagnostics": {
            "max_target_end_date": sample["target_end_date_5d"].max(),
            "max_trading_date": sample["trading_date"].max(),
            "primary_p_value_source": "univariate_OLS-HAC(maxlags=4)",
            "robustness_p_value_used_for_fdr": False,
        },
    })
    return result


def validate_t3(claim: dict) -> dict:
    columns = [claim["x_feature"], *claim["controls"]]
    sample = target_history_for_claim(claim, columns).dropna(
        subset=[*columns, "future_excess_return_5d"]
    )
    finite = np.isfinite(sample[columns + ["future_excess_return_5d"]]).all(axis=1)
    sample = sample.loc[finite].reset_index(drop=True)
    result = result_common(claim, "T3", sample)
    minimum = TEMPLATE_CONFIG["T3"]["min_sample"]
    if len(sample) < minimum:
        result.update({
            "status": "insufficient", "effect_size": None, "ci_95": [None, None],
            "primary_p_value": None, "direction_observed": "not_applicable",
            "robustness_direction_consistent": None,
            "limitations": [f"minimum {minimum} complete observations required"],
            "details": {"controls": claim["controls"]},
        })
        return result
    y = sample["future_excess_return_5d"].to_numpy(dtype=float)
    model = fit_ols_hac(y, sample[columns].astype(float), HAC_MAXLAGS_5D)
    beta = model["coefficients"][claim["x_feature"]]
    uncontrolled_rho = spearman_statistic(
        sample[claim["x_feature"]].to_numpy(dtype=float), y
    )
    result.update({
        "effect_size": beta,
        "ci_95": [model["ci_low"][claim["x_feature"]], model["ci_high"][claim["x_feature"]]],
        "primary_p_value": model["p_values"][claim["x_feature"]],
        "direction_observed": direction_from_effect(beta),
        "robustness_direction_consistent": direction_from_effect(beta) == direction_from_effect(uncontrolled_rho),
        "limitations": ["controlled association, not causal identification", "controls pre-registered before execution"],
        "details": {
            "ols_hac_beta": beta,
            "ols_hac_p": model["p_values"][claim["x_feature"]],
            "controls": claim["controls"],
            "control_coefficients": {control: model["coefficients"][control] for control in claim["controls"]},
            "xtx_diagonal_ratio": model["xtx_diagonal_ratio"],
            "hac_maxlags": HAC_MAXLAGS_5D,
            "uncontrolled_spearman_rho": uncontrolled_rho,
        },
        "diagnostics": {
            "max_target_end_date": sample["target_end_date_5d"].max(),
            "max_trading_date": sample["trading_date"].max(),
        },
    })
    return result


def preferred_financial_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = frame.copy()
    preferred["fs_priority"] = preferred["fs_div"].map({"CFS": 0, "OFS": 1}).fillna(2)
    preferred = preferred.sort_values(
        ["stock_code", "rcept_no", "fs_priority", "effective_date"]
    ).drop_duplicates(["stock_code", "rcept_no"], keep="first")
    return preferred.drop(columns="fs_priority")


def latest_financial_report_slot_versions(frame: pd.DataFrame) -> pd.DataFrame:
    """경제적 Report Slot별 as-of 최신 정정 버전 하나만 남긴다."""
    preferred = preferred_financial_snapshots(frame)
    if preferred.empty:
        return preferred
    slot_keys = ["stock_code", "bsns_year", "reprt_code"]
    return (
        preferred.sort_values([*slot_keys, "effective_date", "rcept_no"])
        .drop_duplicates(slot_keys, keep="last")
        .reset_index(drop=True)
    )


def event_study_rows(claim: dict) -> pd.DataFrame:
    as_of = pd.Timestamp(claim["as_of_date"])
    stock_code = claim["stock_code"]
    prices = claim_panel.loc[
        claim_panel["stock_code"].eq(stock_code)
        & claim_panel["trading_date"].le(as_of),
        ["trading_date", "excess_ret_1"],
    ].sort_values("trading_date").drop_duplicates("trading_date")
    if prices.empty:
        return pd.DataFrame()
    first_price_date = prices["trading_date"].min()
    snapshots = preferred_financial_snapshots(
        financial_snapshot_step8.loc[
            financial_snapshot_step8["stock_code"].eq(stock_code)
            & financial_snapshot_step8["effective_date"].between(first_price_date, as_of)
            & financial_snapshot_step8[claim["x_feature"]].gt(0)
        ]
    )
    clusters = (
        snapshots.groupby(["stock_code", "effective_date"], observed=True)
        .agg(
            event_record_count=("rcept_no", "size"),
            rcept_nos=("rcept_no", lambda values: "|".join(sorted(set(map(str, values))))),
            mean_improvement=(claim["x_feature"], "mean"),
        )
        .reset_index()
    )
    major_dates = np.array(sorted(
        disclosure_step8.loc[
            disclosure_step8["stock_code"].eq(stock_code)
            & disclosure_step8["pblntf_ty"].isin(["B", "I"])
            & disclosure_step8["effective_date"].between(first_price_date, as_of),
            "effective_date",
        ].dropna().unique()
    ), dtype="datetime64[ns]")
    trading_dates = prices["trading_date"].to_numpy(dtype="datetime64[ns]")
    excess_returns = prices["excess_ret_1"].to_numpy(dtype=float)
    rows = []
    for _, event in clusters.iterrows():
        effective_date = pd.Timestamp(event["effective_date"])
        event_position = int(np.searchsorted(trading_dates, np.datetime64(effective_date)))
        end_position = event_position + 4
        if event_position >= len(trading_dates) or end_position >= len(trading_dates):
            continue
        event_trading_date = pd.Timestamp(trading_dates[event_position])
        window_end_date = pd.Timestamp(trading_dates[end_position])
        if window_end_date > as_of:
            continue
        window_returns = excess_returns[event_position:end_position + 1]
        if len(window_returns) != 5 or not np.isfinite(window_returns).all():
            continue
        overlap_dates = major_dates[
            (major_dates > np.datetime64(effective_date))
            & (major_dates <= np.datetime64(window_end_date))
        ]
        rows.append({
            "stock_code": stock_code,
            "effective_date": effective_date,
            "event_date_source": "effective_date",
            "event_trading_date": event_trading_date,
            "window_end_date": window_end_date,
            "event_record_count": int(event["event_record_count"]),
            "rcept_nos": event["rcept_nos"],
            "mean_improvement": float(event["mean_improvement"]),
            "car_0_1": float(window_returns[:2].sum()),
            "car_0_4": float(window_returns.sum()),
            "window_observation_count": int(len(window_returns)),
            "overlapping_event_flag": bool(len(overlap_dates) > 0),
            "overlapping_major_event_count": int(len(overlap_dates)),
        })
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(BOOTSTRAP_ITERATIONS)
    ])
    ci_low, ci_high = np.quantile(estimates, [0.025, 0.975])
    p_value = 2 * min(
        (np.sum(estimates <= 0) + 1) / (len(estimates) + 1),
        (np.sum(estimates >= 0) + 1) / (len(estimates) + 1),
    )
    return float(ci_low), float(ci_high), float(min(p_value, 1.0))


def validate_t4(claim: dict) -> dict:
    events = event_study_rows(claim)
    clean_events = events.loc[~events["overlapping_event_flag"]].copy() if not events.empty else events
    sample_for_common = clean_events.rename(columns={"event_trading_date": "trading_date"})
    result = result_common(claim, "T4", sample_for_common)
    minimum = TEMPLATE_CONFIG["T4"]["min_sample"]
    if len(clean_events) < minimum:
        result.update({
            "status": "insufficient", "effect_size": None, "ci_95": [None, None],
            "primary_p_value": None, "direction_observed": "not_applicable",
            "robustness_direction_consistent": None,
            "limitations": [f"minimum {minimum} clean event clusters required"],
            "details": {"clean_event_count": len(clean_events), "all_event_count": len(events)},
            "diagnostics": {"event_rows": to_builtin(events.to_dict("records"))},
        })
        return result
    clean_car = clean_events["car_0_4"].to_numpy(dtype=float)
    all_car = events["car_0_4"].to_numpy(dtype=float)
    effect = float(clean_car.mean())
    ci_low, ci_high, p_value = bootstrap_mean(clean_car, stable_claim_seed(claim["claim_id"]))
    result.update({
        "effect_size": effect,
        "ci_95": [ci_low, ci_high],
        "primary_p_value": p_value,
        "direction_observed": direction_from_effect(effect),
        "robustness_direction_consistent": direction_from_effect(effect) == direction_from_effect(float(all_car.mean())),
        "limitations": [
            "market-adjusted event association, not isolated causality",
            "primary estimate excludes event windows containing another major disclosure",
        ],
        "details": {
            "event_type": claim["event_type"],
            "clean_event_count": int(len(clean_events)),
            "all_event_count": int(len(events)),
            "overlap_event_count": int(events["overlapping_event_flag"].sum()),
            "window": "CAR[0,+4]",
            "mean_car_0_4": effect,
            "median_car_0_4": float(np.median(clean_car)),
            "mean_car_0_1": float(clean_events["car_0_1"].mean()),
            "positive_event_ratio": float((clean_car > 0).mean()),
            "all_event_mean_car_0_4": float(all_car.mean()),
        },
        "diagnostics": {"event_rows": to_builtin(events.to_dict("records"))},
    })
    return result


def latest_financial_fact(claim: dict) -> pd.Series | None:
    snapshots = preferred_financial_snapshots(
        financial_snapshot_step8.loc[
            financial_snapshot_step8["stock_code"].eq(claim["stock_code"])
            & financial_snapshot_step8["effective_date"].le(pd.Timestamp(claim["as_of_date"]))
        ]
    ).sort_values(["effective_date", "rcept_no"])
    return None if snapshots.empty else snapshots.iloc[-1]


def validate_t5(claim: dict) -> dict:
    fact = latest_financial_fact(claim)
    empty = pd.DataFrame(columns=["trading_date"])
    result = result_common(claim, "T5", empty)
    feature = claim["x_feature"]
    mapping = T5_FEATURE_MAPPING.get(feature)
    if mapping is None:
        result.update({
            "status": "insufficient", "sample_size": 0, "effect_size": None,
            "ci_95": [None, None], "primary_p_value": None,
            "direction_observed": "not_applicable", "robustness_direction_consistent": None,
            "limitations": ["T5 feature has no explicit semantic mapping"], "details": {},
        })
        return result
    signal_feature = mapping["signal_feature"]
    if (
        fact is None
        or pd.isna(fact.get(feature))
        or pd.isna(fact.get(signal_feature))
    ):
        result.update({
            "status": "insufficient", "sample_size": 0, "effect_size": None,
            "ci_95": [None, None], "primary_p_value": None,
            "direction_observed": "not_applicable", "robustness_direction_consistent": None,
            "limitations": ["PIT financial fact or mapped YoY signal unavailable"],
            "details": {"feature": feature, "signal_feature": signal_feature},
        })
        return result

    current_value = float(fact[feature])
    change = float(fact[signal_feature])
    comparison_value = (
        current_value - change
        if mapping["evaluation_mode"] == "mapped_yoy_change"
        else None
    )
    fact_pass = change > 0 if claim["expected_direction"] == "positive" else change < 0
    result.update({
        "sample_size": 1,
        "analysis_start": fact["effective_date"],
        "analysis_end": fact["effective_date"],
        "effect_size": change,
        "ci_95": [None, None],
        "primary_p_value": None,
        "direction_observed": direction_from_effect(change),
        "robustness_direction_consistent": None,
        "fact_pass": bool(fact_pass),
        "limitations": [
            "fact verification; no statistical causality claim",
            "cumulative report periods are not treated as standalone quarters",
        ],
        "details": {
            "current_value": current_value,
            "comparison_value": comparison_value,
            "change": change,
            "evaluation_mode": mapping["evaluation_mode"],
            "signal_feature": signal_feature,
            "unit": mapping["unit"],
            "period_current": f"{int(fact['bsns_year'])}-{fact['reprt_code']}",
            "period_comparison": f"{int(fact['bsns_year']) - 1}-{fact['reprt_code']}",
            "rcept_no": str(fact["rcept_no"]),
            "effective_date": fact["effective_date"],
            "financial_basis": str(fact["fs_div"]),
            "fact_pass": bool(fact_pass),
        },
        "diagnostics": {"max_effective_date": fact["effective_date"]},
    })
    return result


def financial_history_sample(claim: dict, feature: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, str, str]:
    if feature in financial_snapshot_features:
        history = latest_financial_report_slot_versions(
            financial_snapshot_step8.loc[
                financial_snapshot_step8["stock_code"].eq(claim["stock_code"])
                & financial_snapshot_step8["effective_date"].le(as_of)
            ]
        ).dropna(subset=[feature]).sort_values(["effective_date", "rcept_no"])
        return history, "effective_date", "distinct_economic_financial_report_slot"
    if feature in PIT_FINANCIAL_DAILY_FEATURE_DATE:
        effective_column = PIT_FINANCIAL_DAILY_FEATURE_DATE[feature]
        history = claim_panel.loc[
            claim_panel["stock_code"].eq(claim["stock_code"])
            & claim_panel["trading_date"].le(as_of)
            & pd.to_datetime(claim_panel[effective_column], errors="coerce").le(as_of),
            ["stock_code", "trading_date", effective_column, feature],
        ].dropna(subset=[feature, effective_column])
        history = (
            history.sort_values([effective_column, "trading_date"])
            .drop_duplicates(["stock_code", effective_column], keep="last")
        )
        return history, effective_column, "distinct_financial_effective_snapshot"
    history = claim_panel.loc[
        claim_panel["stock_code"].eq(claim["stock_code"])
        & claim_panel["trading_date"].le(as_of),
        ["stock_code", "trading_date", feature],
    ].dropna(subset=[feature]).sort_values("trading_date")
    return history, "trading_date", "daily_observation"


def industry_peer_sample(claim: dict, feature: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, str, str, str]:
    company = company_lookup_step8[claim["stock_code"]]
    industry = company["industry"]
    peer_codes = company_master_step8.loc[
        company_master_step8["industry"].eq(industry), "stock_code"
    ].tolist()
    if feature in financial_snapshot_features:
        snapshots = latest_financial_report_slot_versions(
            financial_snapshot_step8.loc[
                financial_snapshot_step8["stock_code"].isin(peer_codes)
                & financial_snapshot_step8["effective_date"].le(as_of)
            ]
        ).dropna(subset=[feature])
        peer_sample = (
            snapshots.sort_values(["stock_code", "effective_date", "rcept_no"])
            .groupby("stock_code", observed=True, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        return peer_sample, "effective_date", "peer_latest_financial_report_snapshot", industry
    if feature in PIT_FINANCIAL_DAILY_FEATURE_DATE:
        effective_column = PIT_FINANCIAL_DAILY_FEATURE_DATE[feature]
        peer_sample = claim_panel.loc[
            claim_panel["stock_code"].isin(peer_codes)
            & claim_panel["trading_date"].le(as_of)
            & pd.to_datetime(claim_panel[effective_column], errors="coerce").le(as_of),
            ["stock_code", "trading_date", effective_column, feature],
        ].dropna(subset=[feature, effective_column])
        peer_sample = (
            peer_sample.sort_values(["stock_code", effective_column, "trading_date"])
            .groupby("stock_code", observed=True, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        return peer_sample, effective_column, "peer_latest_financial_effective_snapshot", industry
    peer_sample = claim_panel.loc[
        claim_panel["stock_code"].isin(peer_codes)
        & claim_panel["trading_date"].le(as_of),
        ["stock_code", "trading_date", feature],
    ].dropna(subset=[feature])
    peer_sample = (
        peer_sample.sort_values(["stock_code", "trading_date"])
        .groupby("stock_code", observed=True, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return peer_sample, "trading_date", "peer_latest_daily_observation", industry


def validate_t6(claim: dict) -> dict:
    feature = claim["x_feature"]
    as_of = pd.Timestamp(claim["as_of_date"])
    comparison_scope = claim["comparison_scope"]
    peer_industry = None
    if comparison_scope == "history":
        sample, date_column, unit = financial_history_sample(claim, feature, as_of)
        minimum = 4 if unit != "daily_observation" else TEMPLATE_CONFIG["T6"]["min_sample"]
    else:
        sample, date_column, unit, peer_industry = industry_peer_sample(
            claim, feature, as_of
        )
        minimum = 4

    sample_for_common = sample.copy()
    if date_column != "trading_date":
        sample_for_common = sample_for_common.rename(columns={date_column: "trading_date"})
    result = result_common(claim, "T6", sample_for_common)
    peer_codes = sorted(sample["stock_code"].astype(str).unique().tolist()) if not sample.empty else []
    current_rows = sample.loc[sample["stock_code"].eq(claim["stock_code"])]
    if len(sample) < minimum or (comparison_scope == "industry" and current_rows.empty):
        result.update({
            "status": "insufficient", "effect_size": None, "ci_95": [None, None],
            "primary_p_value": None, "direction_observed": "not_applicable",
            "robustness_direction_consistent": None,
            "limitations": [f"minimum {minimum} distinct observations and focal-stock value required"],
            "details": {
                "comparison_scope": comparison_scope,
                "comparison_unit": unit,
                "peer_industry": peer_industry,
                "peer_count": len(peer_codes),
                "peer_stock_codes": peer_codes,
            },
        })
        return result

    values = sample[feature].astype(float).to_numpy()
    current_value = (
        float(values[-1])
        if comparison_scope == "history"
        else float(current_rows.iloc[-1][feature])
    )
    count_le = int(np.sum(values <= current_value))
    percentile = float(count_le / len(values))
    lower, upper = wilson_interval(count_le, len(values))
    threshold = float(claim["condition_value"])
    comparison_pass = apply_operator_scalar(percentile, claim["condition_operator"], threshold)
    effect = percentile - threshold
    result.update({
        "effect_size": float(effect),
        "ci_95": [float(lower - threshold), float(upper - threshold)],
        "primary_p_value": None,
        "direction_observed": direction_from_effect(effect),
        "robustness_direction_consistent": None,
        "comparison_pass": comparison_pass,
        "limitations": ["relative position is descriptive, not causal"],
        "details": {
            "current_value": current_value,
            "percentile": percentile,
            "historical_percentile": percentile if comparison_scope == "history" else None,
            "industry_peer_percentile": percentile if comparison_scope == "industry" else None,
            "comparison_threshold": threshold,
            "comparison_operator": claim["condition_operator"],
            "comparison_pass": comparison_pass,
            "comparison_scope": comparison_scope,
            "comparison_unit": unit,
            "peer_industry": peer_industry,
            "peer_count": len(peer_codes),
            "peer_stock_codes": peer_codes,
        },
        "diagnostics": {
            "max_observation_date": sample[date_column].max(),
            "economic_report_slot_duplicate_count": int(
                sample.duplicated(["stock_code", "bsns_year", "reprt_code"]).sum()
            ) if {"bsns_year", "reprt_code"}.issubset(sample.columns) else None,
        },
    })
    return result


def enforce_finance_claim_policy(claim: dict) -> list[str]:
    company = company_lookup_step8.get(claim["stock_code"])
    if company is None:
        raise ClaimValidationError(f"unknown_stock_code: {claim['stock_code']}")
    if not bool(company["is_finance_industry"]):
        return []
    feature_policy = feature_policy_lookup.get(claim["x_feature"], {})
    router_policy = str(feature_policy.get("claim_router_policy", ""))
    interpretation_scope = str(feature_policy.get("interpretation_scope", ""))
    if router_policy == "block_direct_claim_for_finance" or interpretation_scope == "non_financial_only":
        raise ClaimValidationError(
            f"finance_direct_claim_blocked: {claim['x_feature']}"
        )
    if router_policy == "finance_peer_comparison_only" or interpretation_scope == "finance_industry_relative_only":
        if claim["claim_type"] != "relative_position" or claim["comparison_scope"] != "industry":
            raise ClaimValidationError(
                f"finance_peer_relative_only: {claim['x_feature']}"
            )
    limitations = []
    if interpretation_scope == "industry_specific_for_finance":
        limitations.append("financial-industry terminology and peer context required")
    return limitations


def validate_claim_contract(claim: dict) -> dict:
    validate_claim_json_schema(claim)
    normalized = dict(claim)
    normalized["stock_code"] = str(normalized["stock_code"]).zfill(6)
    as_of = pd.Timestamp(normalized["as_of_date"])
    if as_of > CLAIM_DEVELOPMENT_CUTOFF:
        raise ClaimValidationError("development_as_of_date_exceeds_2024_cutoff")
    if normalized["horizon"] != SUPPORTED_HORIZON:
        raise ClaimValidationError(f"horizon_not_materialized_in_v1: {normalized['horizon']}")
    template_id = CLAIM_TYPE_TO_TEMPLATE.get(normalized["claim_type"])
    if template_id is None:
        raise ClaimValidationError(f"unregistered_claim_type: {normalized['claim_type']}")
    config = TEMPLATE_CONFIG[template_id]
    if normalized["x_feature"] not in config["allowed_features"]:
        raise ClaimValidationError(
            f"feature_not_allowlisted_for_{template_id}: {normalized['x_feature']}"
        )
    if normalized["outcome"] != config["outcome"]:
        raise ClaimValidationError(
            f"outcome_mismatch_for_{template_id}: {normalized['outcome']}"
        )
    if template_id == "T1":
        if normalized["condition_operator"] is None or normalized["condition_value"] is None:
            raise ClaimValidationError("T1_requires_condition_operator_and_value")
    elif template_id in {"T2", "T3", "T4", "T5"}:
        if normalized["condition_operator"] is not None or normalized["condition_value"] is not None:
            raise ClaimValidationError(f"{template_id}_does_not_accept_condition_threshold")
    elif template_id == "T6":
        if normalized["condition_operator"] is None or normalized["condition_value"] is None:
            raise ClaimValidationError("T6_requires_explicit_percentile_threshold")
    if template_id == "T3":
        controls = tuple(normalized["controls"])
        if controls not in set(PREDEFINED_CONTROL_SETS.values()):
            raise ClaimValidationError("T3_controls_not_pre_registered")
        if normalized["x_feature"] in controls:
            raise ClaimValidationError("x_feature_cannot_be_duplicated_in_controls")
    elif normalized["controls"]:
        raise ClaimValidationError(f"controls_not_allowed_for_{template_id}")
    if template_id == "T4" and normalized["event_type"] != "earnings_improvement_report":
        raise ClaimValidationError("unsupported_event_type")
    if template_id != "T4" and normalized["event_type"] is not None:
        raise ClaimValidationError(f"event_type_not_allowed_for_{template_id}")
    if template_id == "T6" and normalized["comparison_scope"] not in {"history", "industry"}:
        raise ClaimValidationError("T6_requires_comparison_scope")
    if template_id != "T6" and normalized["comparison_scope"] is not None:
        raise ClaimValidationError(f"comparison_scope_not_allowed_for_{template_id}")
    normalized["finance_policy_limitations"] = enforce_finance_claim_policy(normalized)
    normalized["template_id"] = template_id
    return normalized


def interval_excludes_zero(ci_95: list) -> bool:
    if not ci_95 or len(ci_95) != 2 or any(value is None for value in ci_95):
        return False
    low, high = ci_95
    return bool(low > 0 or high < 0)


def opposite_direction(direction: str) -> str:
    return {"positive": "negative", "negative": "positive"}.get(direction, "not_applicable")


def verdict_engine(raw_result: dict) -> str:
    if raw_result.get("status") != "ok":
        return "Insufficient Evidence"
    template_id = raw_result.get("template_id")
    expected = raw_result["claim"]["expected_direction"]
    observed = raw_result.get("direction_observed", "not_applicable")
    if template_id == "T5":
        return "Supported" if raw_result.get("fact_pass") else "Contradicted"
    if template_id == "T6":
        return "Supported" if raw_result.get("comparison_pass") else "Contradicted"

    ci_excludes = interval_excludes_zero(raw_result.get("ci_95"))
    fdr_pass = raw_result.get("fdr_pass") is True
    robustness_consistent = raw_result.get("robustness_direction_consistent")
    primary_supported = observed == expected and ci_excludes and fdr_pass
    primary_contradicted = (
        observed == opposite_direction(expected) and ci_excludes and fdr_pass
    )
    if primary_contradicted:
        return "Contradicted"
    if primary_supported and robustness_consistent is True:
        return "Supported"
    if primary_supported and robustness_consistent is False:
        return "Partially Supported"
    return "Insufficient Evidence"


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def apply_batch_fdr(raw_results: list[dict]) -> list[dict]:
    batch_ids = sorted({
        result["claim"]["validation_batch_id"] for result in raw_results
    })
    for batch_id in batch_ids:
        eligible_indices = [
            index for index, result in enumerate(raw_results)
            if result["claim"]["validation_batch_id"] == batch_id
            and result.get("status") == "ok"
            and result.get("fdr_applicable") is True
            and result.get("primary_p_value") is not None
        ]
        if eligible_indices:
            p_values = np.array([
                raw_results[index]["primary_p_value"] for index in eligible_indices
            ], dtype=float)
            q_values = benjamini_hochberg(p_values)
            for index, q_value in zip(eligible_indices, q_values):
                raw_results[index]["q_value"] = float(q_value)
                raw_results[index]["fdr_pass"] = bool(q_value <= FDR_ALPHA)
    for result in raw_results:
        result["verdict"] = verdict_engine(result)
    return raw_results


def rejected_raw_result(claim: dict, error: Exception) -> dict:
    template_id = CLAIM_TYPE_TO_TEMPLATE.get(claim.get("claim_type"))
    return {
        "claim": claim,
        "template_id": template_id,
        "status": "rejected",
        "sample_size": 0,
        "analysis_start": None,
        "analysis_end": None,
        "primary_inference_method": TEMPLATE_CONFIG.get(template_id, {}).get("primary_inference_method"),
        "methods": [],
        "fdr_applicable": False,
        "primary_p_value": None,
        "q_value": None,
        "fdr_pass": None,
        "effect_size": None,
        "ci_95": [None, None],
        "direction_observed": "not_applicable",
        "robustness_direction_consistent": None,
        "limitations": [str(error)],
        "details": {},
        "diagnostics": {},
    }


def standardize_evidence_result(raw_result: dict) -> dict:
    claim = raw_result["claim"]
    finance_limitations = claim.get("finance_policy_limitations", [])
    evidence = {
        "claim_id": claim.get("claim_id", "UNKNOWN"),
        "validation_batch_id": claim.get("validation_batch_id", "UNKNOWN"),
        "stock_code": str(claim.get("stock_code", "000000")).zfill(6),
        "as_of_date": str(claim.get("as_of_date", "")),
        "template_id": raw_result.get("template_id"),
        "claim_text": claim.get("claim_text", ""),
        "status": raw_result.get("status", "rejected"),
        "verdict": raw_result.get("verdict", "Insufficient Evidence"),
        "direction_expected": claim.get("expected_direction", "positive"),
        "direction_observed": raw_result.get("direction_observed", "not_applicable"),
        "effect_size": raw_result.get("effect_size"),
        "ci_95": raw_result.get("ci_95", [None, None]),
        "p_value": raw_result.get("primary_p_value"),
        "q_value": raw_result.get("q_value"),
        "sample_size": int(raw_result.get("sample_size", 0)),
        "analysis_start": raw_result.get("analysis_start"),
        "analysis_end": raw_result.get("analysis_end"),
        "x_feature": claim.get("x_feature", "UNKNOWN"),
        "outcome": claim.get("outcome", "UNKNOWN"),
        "horizon": int(claim.get("horizon", SUPPORTED_HORIZON)),
        "controls": list(claim.get("controls", [])),
        "primary_inference_method": raw_result.get("primary_inference_method"),
        "methods": raw_result.get("methods", []),
        "limitations": [*raw_result.get("limitations", []), *finance_limitations],
        "fdr_applicable": bool(raw_result.get("fdr_applicable", False)),
        "fdr_pass": raw_result.get("fdr_pass"),
        "details": raw_result.get("details", {}),
        "diagnostics": raw_result.get("diagnostics", {}),
        "data_lineage": {
            "feature_panel_sha256": step8_file_sha256(FEATURE_PANEL_PATH),
            "target_panel_sha256": step8_file_sha256(TARGET_PANEL_PATH),
            "feature_registry_sha256": step8_file_sha256(FEATURE_REGISTRY_PATH),
            "development_cutoff": CLAIM_DEVELOPMENT_CUTOFF,
            "target_pit_rule": "target_end_date_5d<=as_of_date",
        },
    }
    return to_builtin(evidence)


def run_claim_batch(claims: list[dict]) -> list[dict]:
    raw_results = []
    for input_claim in claims:
        try:
            claim = validate_claim_contract(dict(input_claim))
            raw_result = VALIDATOR_FUNCTIONS[claim["template_id"]](claim)
        except (ClaimValidationError, jsonschema.ValidationError, KeyError, ValueError) as error:
            raw_result = rejected_raw_result(dict(input_claim), error)
        raw_results.append(raw_result)
    raw_results = apply_batch_fdr(raw_results)
    evidence_results = [standardize_evidence_result(result) for result in raw_results]
    for evidence in evidence_results:
        errors = list(EVIDENCE_SCHEMA_VALIDATOR.iter_errors(evidence))
        if errors:
            raise AssertionError(
                f"Evidence schema failure for {evidence['claim_id']}: "
                + "; ".join(error.message for error in errors)
            )
    return evidence_results


VALIDATOR_FUNCTIONS = {
    "T1": validate_t1, "T2": validate_t2, "T3": validate_t3,
    "T4": validate_t4, "T5": validate_t5, "T6": validate_t6,
}

UPSTREAM_PROVENANCE = {
    "source_notebook": str(PROJECT_ROOT / "step8_10.ipynb"),
    "source_notebook_sha256": "c2eb0e25cf280e500f83ede8ff4f06637f7b2d90d1d00d4bc4a3aa98f078e58c",
    "function_names": ['step8_file_sha256', 'step8_json_default', 'to_builtin', 'validate_claim_json_schema', 'stable_claim_seed', 'direction_from_effect', 'spearman_statistic', 'spearman_test', 'invert_small_matrix', 'multiply_small_matrices', 'fit_ols_hac', 'wilson_interval', 'apply_operator', 'apply_operator_scalar', 'target_history_for_claim', 'moving_block_indices', 'block_bootstrap_difference', 'block_bootstrap_spearman', 'hedges_like_standardized_difference', 'result_common', 'validate_t1', 'validate_t2', 'validate_t3', 'preferred_financial_snapshots', 'latest_financial_report_slot_versions', 'event_study_rows', 'bootstrap_mean', 'validate_t4', 'latest_financial_fact', 'validate_t5', 'financial_history_sample', 'industry_peer_sample', 'validate_t6', 'enforce_finance_claim_policy', 'validate_claim_contract', 'interval_excludes_zero', 'opposite_direction', 'verdict_engine', 'benjamini_hochberg', 'apply_batch_fdr', 'rejected_raw_result', 'standardize_evidence_result', 'run_claim_batch'],
    "implementation": "exact_function_sources_extracted_from_frozen_step8",
}
