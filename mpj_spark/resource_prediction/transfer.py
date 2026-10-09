# =============================================================================
# mpj_spark/resource_prediction/transfer.py
# Phase 6 — Scout → local transfer experiment (Objective 2b)
#
# QUESTION
# --------
# Does knowledge from the Scout dataset (AWS EC2, whole-VM sar traces) improve
# per-driver CPU / memory prediction on the local testbed, compared with
# models trained on local data only — in particular when local data is scarce?
#
# SOURCE MODEL (deliberately simple and readable)
# ----------------------------------------------
# One straight line per workload, fitted on Scout single-node hosts:
#     source_target ≈ a_w + b_w · log2(cores)
# Only `workload_type` and `cores` are shared between the domains:
#   * Scout vcpus_per_host  ↔ local cores_per_driver
#   * data size is NOT used: local inputs (25–50 MB per driver) lie far
#     below every Scout input (≥ 769 MiB) — that would be pure extrapolation
#   * local heap has no Scout counterpart; only the local correction sees it
# Label mapping (scope differs — documented, not hidden):
#   cpu_cores_mean ← host_cpu_busy_mean_pct / 100 × vcpus   (whole VM)
#   cpu_cores_p95  ← host_cpu_busy_p95_cores                 (whole VM)
#   mem_mb_peak    ← host_mem_app_peak_mib                   (whole VM, no cache)
#
# TRANSFER METHODS (all see the source prediction as column `source_pred`)
# -----------------------------------------------------------------------
#   scout_only      the source line applied unchanged (zero-shot)
#   scout_offset    source line + per-workload mean local residual, shrunk
#                   towards 0 with few local rows: n / (n + k)
#   scout_residual  source line + Ridge (per-workload slopes) fitted on the
#                   local residual (y − source_pred), using all local inputs
# Local-only comparators come from models.py (lookup, ridge_wl, rf).
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from mpj_spark.resource_prediction import models as m

# local target → (Scout column or derivation, unit note)
SOURCE_LABELS: dict[str, str] = {
    "cpu_cores_mean": "host_cpu_busy_mean_pct / 100 * vcpus_per_host",
    "cpu_cores_p95": "host_cpu_busy_p95_cores",
    "mem_mb_peak": "host_mem_app_peak_mib",
}
SOURCE_SETTING = "single_node"  # one Spark application per VM ≈ one driver


def scout_source_frame(scout_hosts: pd.DataFrame) -> pd.DataFrame:
    """Scout host rows mapped onto the local column names used for transfer."""
    s = scout_hosts[scout_hosts["setting"] == SOURCE_SETTING].copy()
    return pd.DataFrame(
        {
            "workload_type": s["workload_type"],
            "cores_per_driver": s["vcpus_per_host"].astype(float),
            "cpu_cores_mean": s["host_cpu_busy_mean_pct"] / 100.0 * s["vcpus_per_host"],
            "cpu_cores_p95": s["host_cpu_busy_p95_cores"],
            "mem_mb_peak": s["host_mem_app_peak_mib"],
        }
    )


# =============================================================================
# Source model
# =============================================================================


class SourceLines:
    """Per-workload least-squares line  y = a + b·log2(cores)  fitted on Scout."""

    def fit(self, source: pd.DataFrame, target: str) -> SourceLines:
        self.target = target
        self.lines_ = {}
        self.core_range_ = {}
        for wl, g in source.dropna(subset=[target]).groupby("workload_type"):
            x = np.log2(g["cores_per_driver"].to_numpy(float))
            y = g[target].to_numpy(float)
            if len(np.unique(x)) >= 2:
                b, a = np.polyfit(x, y, 1)
            else:
                a, b = float(np.mean(y)), 0.0
            self.lines_[wl] = (float(a), float(b))
            self.core_range_[wl] = (
                float(g["cores_per_driver"].min()),
                float(g["cores_per_driver"].max()),
            )
        all_a = np.mean([a for a, _ in self.lines_.values()])
        all_b = np.mean([b for _, b in self.lines_.values()])
        self.fallback_ = (float(all_a), float(all_b))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for i, (wl, cores) in enumerate(
            zip(df["workload_type"], df["cores_per_driver"], strict=True)
        ):
            a, b = self.lines_.get(wl, self.fallback_)
            out[i] = a + b * np.log2(float(cores))
        return out

    def extrapolating(self, df: pd.DataFrame) -> np.ndarray:
        """True where local cores lie outside the Scout range of that workload."""
        flags = []
        for wl, cores in zip(df["workload_type"], df["cores_per_driver"], strict=True):
            lo, hi = self.core_range_.get(wl, (np.inf, -np.inf))
            flags.append(not (lo <= cores <= hi))
        return np.array(flags)

    def describe(self) -> dict:
        return {
            wl: {"intercept": a, "slope_per_log2_core": b, "scout_core_range": self.core_range_[wl]}
            for wl, (a, b) in self.lines_.items()
        }


# =============================================================================
# Transfer estimators — X must contain FEATURES + "source_pred"
# =============================================================================


class ScoutOnly:
    """Zero-shot: the source prediction, unchanged."""

    def fit(self, X: pd.DataFrame, y) -> ScoutOnly:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["source_pred"].to_numpy(float)


class ScoutOffset:
    """Source prediction + per-workload mean local residual, shrunk by n/(n+k)."""

    def __init__(self, shrink_k: float = 2.0):
        self.shrink_k = shrink_k

    def fit(self, X: pd.DataFrame, y) -> ScoutOffset:
        resid = pd.Series(np.asarray(y, float) - X["source_pred"].to_numpy(float), index=X.index)
        stats = resid.groupby(X["workload_type"]).agg(["mean", "count"])
        self.offsets_ = {
            wl: r["mean"] * r["count"] / (r["count"] + self.shrink_k) for wl, r in stats.iterrows()
        }
        n = len(resid)
        self.global_offset_ = float(resid.mean() * n / (n + self.shrink_k)) if n else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        off = X["workload_type"].map(self.offsets_).fillna(self.global_offset_).to_numpy(float)
        return X["source_pred"].to_numpy(float) + off


class ScoutResidual:
    """Source prediction + per-workload Ridge on the local residual (all local inputs)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y) -> ScoutResidual:
        resid = np.asarray(y, float) - X["source_pred"].to_numpy(float)
        self.model_ = m.make_ridge_wl(self.alpha).fit(X[m.FEATURES], resid)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["source_pred"].to_numpy(float) + self.model_.predict(X[m.FEATURES])


def method_factories(rf_trees: int = 200) -> dict:
    """All compared methods; every factory returns an object with fit/predict on X."""
    return {
        "local_lookup": m.LookupModel,
        "local_ridge_wl": lambda: m.make_ridge_wl(1.0),
        "local_rf": lambda: m.make_rf(5, 2, n_estimators=rf_trees),
        "scout_only": ScoutOnly,
        "scout_offset": ScoutOffset,
        "scout_residual": lambda: ScoutResidual(1.0),
    }


def _fit_predict(factory, train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    cols = [*m.FEATURES, "source_pred"]
    model = factory()
    # local-only pipelines must not see the source prediction
    if isinstance(model, (ScoutOnly, ScoutOffset, ScoutResidual)):
        return model.fit(train[cols], train[target]).predict(test[cols])
    return model.fit(train[m.FEATURES], train[target]).predict(test[m.FEATURES])


# =============================================================================
# Evaluation protocols
# =============================================================================


def group_cv(df: pd.DataFrame, target: str, factories: dict) -> pd.DataFrame:
    """Leave-one-configuration-out predictions for every method."""
    frames = []
    for group in df["config_group"].unique():
        held, rest = df[df["config_group"] == group], df[df["config_group"] != group]
        for name, factory in factories.items():
            frames.append(
                pd.DataFrame(
                    {
                        "config_group": group,
                        "workload_type": held["workload_type"].values,
                        "model": name,
                        "actual": held[target].values,
                        "predicted": _fit_predict(factory, rest, held, target),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def sample_groups(groups: pd.DataFrame, k: int, rng: np.random.Generator) -> list[str]:
    """k configuration groups, at least one per workload when k allows."""
    by_wl = {wl: list(g["config_group"]) for wl, g in groups.groupby("workload_type")}
    chosen = []
    for wl in sorted(by_wl):
        if len(chosen) < k:
            chosen.append(by_wl[wl][rng.integers(len(by_wl[wl]))])
    rest = [g for g in groups["config_group"] if g not in chosen]
    extra = rng.choice(rest, size=max(0, min(k, len(groups)) - len(chosen)), replace=False)
    return chosen + list(extra)


def learning_curve(
    df: pd.DataFrame,
    target: str,
    factories: dict,
    ks: list[int],
    reps: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For each k, repeatedly train on k random configuration groups (the
    'adaptation set', identical for every method) and evaluate on all other
    groups.  Uses train+val only — the frozen test split is never touched.
    """
    rng = np.random.default_rng(seed)
    groups = df[["config_group", "workload_type"]].drop_duplicates("config_group")
    rows = []
    for k in ks:
        if k >= len(groups):
            continue
        for rep in range(reps):
            picked = sample_groups(groups, k, rng)
            train = df[df["config_group"].isin(picked)]
            evals = df[~df["config_group"].isin(picked)]
            for name, factory in factories.items():
                pred = _fit_predict(factory, train, evals, target)
                err = pred - evals[target].to_numpy(float)
                rows.append(
                    {
                        "target": target,
                        "k_groups": k,
                        "rep": rep,
                        "model": name,
                        "n_train_rows": len(train),
                        "n_eval_rows": len(evals),
                        "mae": float(np.mean(np.abs(err))),
                        "mean_signed_error": float(np.mean(err)),
                    }
                )
    return pd.DataFrame(rows)
