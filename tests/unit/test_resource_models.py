# =============================================================
# tests/unit/test_resource_models.py
# Phase 6 — per-driver resource models and the training script.
# Synthetic driver tables only.
# =============================================================
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from mpj_spark.resource_prediction import models as m

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")


def _frame(n_per_group=2, seed=0):
    """Configs × repeats with a known demand formula per workload."""
    rng = np.random.default_rng(seed)
    rows = []
    for wl, base in (("kmeans", 2.0), ("logreg", 6.0)):
        for size in (50, 100):
            for cores in (1, 2, 4, 8):
                for heap in (1536, 3072):
                    for rep in range(n_per_group):
                        cpu = base + 0.1 * cores + rng.normal(0, 0.02)
                        mem = 500 + 0.3 * heap + 4 * size + rng.normal(0, 5)
                        rows.append(
                            {
                                "workload_type": wl,
                                "dataset_mb": size,
                                "num_workers": 2,
                                "cores_per_driver": cores,
                                "heap_mb_per_driver": heap,
                                "cpu_cores_mean": cpu,
                                "cpu_cores_p95": cpu * 1.2,
                                "mem_mb_peak": mem,
                                "rep": rep,
                            }
                        )
    return pd.DataFrame(rows)


# ── feature contract ──────────────────────────────────────────


def test_check_features_rejects_missing_inputs():
    df = _frame()
    m.check_features(df)
    df.loc[0, "heap_mb_per_driver"] = np.nan
    with pytest.raises(ValueError, match="missing pre-run inputs"):
        m.check_features(df)
    with pytest.raises(ValueError, match="missing feature columns"):
        m.check_features(df.drop(columns=["num_workers"]))


# ── models ────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", [lambda: m.make_ridge(1.0), lambda: m.make_rf(5, 2)])
def test_pipelines_learn_the_signal(factory):
    df = _frame()
    model = factory().fit(df[m.FEATURES], df["mem_mb_peak"])
    pred = model.predict(df[m.FEATURES])
    assert m.regression_metrics(df["mem_mb_peak"], pred)["mape_pct"] < 5


def test_ridge_scaler_fitted_on_training_rows_only():
    df = _frame()
    train = df[df["dataset_mb"] == 50]
    model = m.make_ridge().fit(train[m.FEATURES], train["cpu_cores_mean"])
    scaler = model.named_steps["pre"].named_transformers_["log"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(np.log2(50))  # never saw size 100


def test_lookup_falls_back_to_coarser_keys():
    df = _frame()
    train = df[df["cores_per_driver"] != 8]
    lk = m.LookupModel().fit(train[m.FEATURES], train["cpu_cores_mean"])
    seen = df[(df["cores_per_driver"] == 2)].iloc[[0]]
    unseen_cores = df[(df["cores_per_driver"] == 8)].iloc[[0]]
    new_workload = seen.assign(workload_type="wordcount")
    lk.predict(seen)
    assert lk.level_used_[0] == 0
    lk.predict(unseen_cores)
    assert lk.level_used_[0] == 2  # workload + size + workers
    pred = lk.predict(new_workload)
    assert lk.level_used_[0] == 4 and pred[0] == pytest.approx(train["cpu_cores_mean"].median())


# ── metrics ───────────────────────────────────────────────────


def test_regression_metrics_units_and_signs():
    met = m.regression_metrics([10, 20], [12, 18], upper=[15, 19])
    assert met["mae"] == 2 and met["mean_signed_error"] == 0
    assert met["underprediction_rate"] == 0.5
    assert met["upper_bound_coverage"] == 0.5
    assert m.regression_metrics([5, 5], [5, 5])["r2"] is None


def test_upper_margin_covers_quantile():
    rng = np.random.default_rng(1)
    y = rng.normal(100, 10, 1000)
    pred = np.full_like(y, 100)
    margin = m.upper_margin(y, pred, 0.9)
    assert np.mean(pred + margin >= y) == pytest.approx(0.9, abs=0.01)
    assert m.upper_margin([1, 2], [5, 5]) == 0.0  # never negative


# ── training script end-to-end ────────────────────────────────


def test_training_script_end_to_end(tmp_path):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import train_resource_predictor as trp

    df = _frame(n_per_group=2)
    df["config_group"] = df[m.FEATURES].astype(str).agg("|".join, axis=1)
    groups = sorted(df["config_group"].unique())
    split = {
        g: ("test" if i % 7 == 0 else "val" if i % 7 == 1 else "train")
        for i, g in enumerate(groups)
    }
    df["split"] = df["config_group"].map(split)
    df["in_main_set"], df["status"], df["alloc_known"] = True, "ok", True
    df["run_key"], df["driver_idx"] = df.index.astype(str), 1
    extra = df.iloc[:2].assign(alloc_known=False, workload_type="wordcount")
    path = tmp_path / "drivers.csv"
    pd.concat([df, extra]).to_csv(path, index=False)

    out = tmp_path / "models"
    metrics = trp.main(["--drivers", str(path), "--out-dir", str(out)])
    assert set(metrics["model"]) == {"lookup", "ridge", "rf"}
    assert set(metrics["target"]) == set(m.TARGETS)
    card = json.loads((out / "model_card.json").read_text())
    assert card["rows"]["rows_dropped_unknown_alloc"] == 2
    assert card["rows"]["workloads_dropped_entirely"] == ["wordcount"]
    assert (out / "mem_mb_peak__rf.joblib").exists()
    preds = pd.read_csv(out / "predictions_test.csv")
    test_keys = set(df.loc[df["split"] == "test", "run_key"])
    assert set(preds["run_key"].astype(str)) == test_keys  # only TEST rows scored
