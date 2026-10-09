# =============================================================================
# mpj_spark/resource_prediction/models.py
# Phase 6 — per-driver CPU / memory demand models (P6-02 CPU, P6-03 memory)
#
# PURPOSE
# -------
# Small, transparent regressors that predict one driver's resource demand
# from PRE-RUN inputs only (what is known before the job is launched):
#
#   workload_type, dataset_mb, num_workers, cores_per_driver, heap_mb_per_driver
#
# Models (one per target; CPU and memory are trained separately):
#   lookup  median of the most specific matching configuration seen in
#           training, falling back to coarser matches  (baseline)
#   ridge   regularised linear regression on scaled / one-hot features
#   rf      small random forest (non-linear comparator)
#
# Preprocessing lives inside each scikit-learn Pipeline, so scaling and
# encoding are fitted on training rows only (no leakage).
#
# An empirical UPPER bound per model is calibrated on validation residuals:
#   upper = prediction + q-quantile of (actual − prediction) on validation
# The allocator (later step) adds its own safety margin on top.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# =============================================================================
# Feature contract
# =============================================================================

CATEGORICAL_FEATURES = ["workload_type"]
NUMERIC_FEATURES = ["dataset_mb", "num_workers", "cores_per_driver", "heap_mb_per_driver"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Most specific → least specific key for the lookup baseline.
LOOKUP_LEVELS: list[list[str]] = [
    FEATURES,
    ["workload_type", "dataset_mb", "num_workers", "cores_per_driver"],
    ["workload_type", "dataset_mb", "num_workers"],
    ["workload_type"],
    [],
]

TARGETS: dict[str, str] = {
    "cpu_cores_mean": "cores",
    "cpu_cores_p95": "cores",
    "mem_mb_peak": "MB",
}

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
RF_GRID = [{"max_depth": d, "min_samples_leaf": leaf} for d in (3, 5, 8) for leaf in (2, 5)]


def check_features(df: pd.DataFrame) -> None:
    """Reject rows that miss a mandatory pre-run input."""
    missing = [c for c in FEATURES if c not in df]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    bad = df[FEATURES].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have missing pre-run inputs")


# =============================================================================
# Models
# =============================================================================


def _log2(x):
    return np.log2(np.asarray(x, dtype=float))


def make_ridge(alpha: float = 1.0) -> Pipeline:
    """Ridge on one-hot workload + standardised log2(size, heap) and counts."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "log",
                Pipeline([("log2", FunctionTransformer(_log2)), ("scale", StandardScaler())]),
                ["dataset_mb", "heap_mb_per_driver"],
            ),
            ("lin", StandardScaler(), ["num_workers", "cores_per_driver"]),
        ]
    )
    return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])


def make_rf(max_depth: int = 5, min_samples_leaf: int = 2, seed: int = 0) -> Pipeline:
    """Small random forest; trees need no scaling, only the one-hot workload."""
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    )
    return Pipeline([("pre", pre), ("model", rf)])


class LookupModel:
    """
    Median of the training rows that share the most specific configuration
    key available (see LOOKUP_LEVELS).  `level_used_` records which level
    answered each prediction, so fallbacks are visible in the report.
    """

    def fit(self, X: pd.DataFrame, y) -> LookupModel:
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.tables_ = []
        for keys in LOOKUP_LEVELS:
            if keys:
                self.tables_.append(y.groupby([X[k] for k in keys]).median().to_dict())
            else:
                self.tables_.append({(): float(y.median())})
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds, levels = [], []
        for _, row in X.iterrows():
            for level, (keys, table) in enumerate(zip(LOOKUP_LEVELS, self.tables_, strict=True)):
                key = (
                    tuple(row[k] for k in keys) if len(keys) > 1 else (row[keys[0]] if keys else ())
                )
                if key in table:
                    preds.append(table[key])
                    levels.append(level)
                    break
        self.level_used_ = np.array(levels)
        return np.array(preds, dtype=float)


# =============================================================================
# Metrics
# =============================================================================


def regression_metrics(y_true, y_pred, upper=None) -> dict:
    """Errors in physical units; R² only meaningful with enough spread."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_signed_error": float(np.mean(err)),
        "mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100),
        "underprediction_rate": float(np.mean(err < 0)),
        "underprediction_gt10pct_rate": float(np.mean(err < -0.10 * np.abs(y_true))),
    }
    var = np.var(y_true)
    out["r2"] = float(1 - np.mean(err**2) / var) if var > 0 else None
    if upper is not None:
        out["upper_bound_coverage"] = float(np.mean(np.asarray(upper) >= y_true))
    return out


def upper_margin(y_true, y_pred, quantile: float = 0.9) -> float:
    """Additive margin so that `pred + margin` covers `quantile` of residuals."""
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(max(np.quantile(resid, quantile), 0.0))
