# =============================================================
# tests/unit/test_transfer.py
# Phase 6 — Scout -> local transfer experiment.
# Synthetic source and target tables only.
# =============================================================
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from mpj_spark.resource_prediction import models as m
from mpj_spark.resource_prediction import transfer as t

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")


def _source():
    """Scout-like rows: y = a_w + b_w * log2(cores) exactly."""
    rows = []
    for wl, a, b in (("kmeans", 0.0, 1.0), ("logreg", 1.0, 0.5)):
        for cores in (2, 4, 8):
            for _ in range(3):
                y = a + b * np.log2(cores)
                rows.append(
                    {
                        "workload_type": wl,
                        "cores_per_driver": float(cores),
                        "cpu_cores_mean": y,
                        "cpu_cores_p95": 2 * y,
                        "mem_mb_peak": 1000 * y + 500,
                    }
                )
    return pd.DataFrame(rows)


def _local(shift=3.0, seed=0):
    """Local rows = source line + shift + heap effect + noise; configs x 2 repeats."""
    rng = np.random.default_rng(seed)
    rows = []
    for wl, a, b in (("kmeans", 0.0, 1.0), ("logreg", 1.0, 0.5)):
        for cores in (1, 2, 4, 8):
            for heap in (1024, 2048, 4096):
                group = f"{wl}|{cores}|{heap}"
                for rep in range(2):
                    base = a + b * np.log2(cores)
                    rows.append(
                        {
                            "workload_type": wl,
                            "dataset_mb": 100,
                            "num_workers": 2,
                            "cores_per_driver": float(cores),
                            "heap_mb_per_driver": float(heap),
                            "cpu_cores_mean": base + shift + rng.normal(0, 0.01),
                            "cpu_cores_p95": 2 * base + shift,
                            "mem_mb_peak": 1000 * base + 200 * np.log2(heap) + rng.normal(0, 5),
                            "config_group": group,
                            "rep": rep,
                        }
                    )
    return pd.DataFrame(rows)


# ── Scout mapping and source lines ────────────────────────────


def test_scout_source_frame_maps_columns_and_setting():
    hosts = pd.DataFrame(
        {
            "setting": ["single_node", "multi_node"],
            "workload_type": ["kmeans", "kmeans"],
            "vcpus_per_host": [4, 4],
            "host_cpu_busy_mean_pct": [50.0, 50.0],
            "host_cpu_busy_p95_cores": [3.0, 3.0],
            "host_mem_app_peak_mib": [2000.0, 2000.0],
        }
    )
    src = t.scout_source_frame(hosts)
    assert len(src) == 1  # single-node only
    assert src["cpu_cores_mean"].iloc[0] == pytest.approx(2.0)  # 50 % of 4 vCPUs
    assert src["cores_per_driver"].iloc[0] == 4.0


def test_source_lines_recover_per_workload_line_and_flag_extrapolation():
    lines = t.SourceLines().fit(_source(), "cpu_cores_mean")
    assert lines.lines_["kmeans"] == pytest.approx((0.0, 1.0))
    assert lines.lines_["logreg"] == pytest.approx((1.0, 0.5))
    q = pd.DataFrame(
        {"workload_type": ["kmeans", "logreg", "wordcount"], "cores_per_driver": [16.0, 4.0, 4.0]}
    )
    pred = lines.predict(q)
    assert pred[0] == pytest.approx(4.0) and pred[1] == pytest.approx(2.0)
    assert pred[2] == pytest.approx(np.mean([0.0, 1.0]) + np.mean([1.0, 0.5]) * 2)  # fallback
    assert list(lines.extrapolating(q)) == [True, False, True]


# ── transfer estimators ───────────────────────────────────────


def _with_source(target="cpu_cores_mean", shift=3.0):
    local = _local(shift)
    lines = t.SourceLines().fit(_source(), target)
    local["source_pred"] = lines.predict(local)
    return local


def test_scout_only_is_zero_shot():
    d = _with_source()
    pred = t.ScoutOnly().fit(d, d["cpu_cores_mean"]).predict(d)
    assert np.allclose(pred, d["source_pred"])


def test_scout_offset_learns_shift_with_shrinkage():
    d = _with_source(shift=3.0)
    few = d.groupby("workload_type").head(1)  # one row per workload
    off_few = t.ScoutOffset(shrink_k=2.0).fit(few, few["cpu_cores_mean"])
    assert off_few.offsets_["kmeans"] == pytest.approx(3.0 / 3.0, abs=0.05)  # 3 * 1/(1+2)
    off_all = t.ScoutOffset(shrink_k=2.0).fit(d, d["cpu_cores_mean"])
    assert off_all.offsets_["kmeans"] == pytest.approx(3.0, rel=0.1)


def test_scout_residual_uses_local_inputs():
    target = "mem_mb_peak"
    d = _with_source(target)
    res = t.ScoutResidual(alpha=0.1).fit(d, d[target])
    mae = m.regression_metrics(d[target], res.predict(d))["mae"]
    assert mae < 30  # heap effect learned on top of the source line


def test_local_models_never_see_source_pred():
    d = _with_source()
    factories = t.method_factories(rf_trees=5)
    model = factories["local_rf"]()
    pred = t._fit_predict(factories["local_rf"], d, d, "cpu_cores_mean")
    assert len(pred) == len(d)
    model.fit(d[m.FEATURES], d["cpu_cores_mean"])
    assert model.named_steps["model"].n_features_in_ == len(m.NUMERIC_FEATURES) + 2  # one-hot


# ── protocols ─────────────────────────────────────────────────


def test_sample_groups_covers_every_workload():
    d = _with_source()
    groups = d[["config_group", "workload_type"]].drop_duplicates("config_group")
    rng = np.random.default_rng(0)
    for k in (2, 3, 6):
        picked = t.sample_groups(groups, k, rng)
        assert len(picked) == k and len(set(picked)) == k
        assert {g.split("|")[0] for g in picked} == {"kmeans", "logreg"}


def test_learning_curve_keeps_train_and_eval_disjoint():
    d = _with_source()
    fac = {"scout_offset": t.ScoutOffset, "scout_only": t.ScoutOnly}
    curve = t.learning_curve(d, "cpu_cores_mean", fac, ks=[2, 4], reps=3)
    assert set(curve["k_groups"]) == {2, 4} and set(curve["model"]) == set(fac)
    assert (curve["n_train_rows"] + curve["n_eval_rows"] == len(d)).all()
    # local data must correct the constant domain shift that zero-shot misses
    med = curve.groupby("model")["mae"].median()
    assert med["scout_offset"] < med["scout_only"]


def test_group_cv_predicts_every_row_once_per_method():
    d = _with_source()
    fac = {"scout_only": t.ScoutOnly, "local_lookup": m.LookupModel}
    oof = t.group_cv(d, "cpu_cores_mean", fac)
    assert len(oof) == 2 * len(d)


# ── script end-to-end ─────────────────────────────────────────


def test_transfer_script_end_to_end(tmp_path):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import transfer_experiment

    src = _source()
    hosts = pd.DataFrame(
        {
            "setting": "single_node",
            "workload_type": src["workload_type"],
            "vcpus_per_host": src["cores_per_driver"].astype(int),
            "host_cpu_busy_mean_pct": src["cpu_cores_mean"] / src["cores_per_driver"] * 100,
            "host_cpu_busy_p95_cores": src["cpu_cores_p95"],
            "host_mem_app_peak_mib": src["mem_mb_peak"],
        }
    )
    local = _local()
    groups = sorted(local["config_group"].unique())
    local["split"] = local["config_group"].map(
        {g: ("test" if i % 6 == 0 else "train") for i, g in enumerate(groups)}
    )
    local["in_main_set"], local["status"], local["alloc_known"] = True, "ok", True
    hp, lp = tmp_path / "hosts.csv", tmp_path / "drivers.csv"
    hosts.to_csv(hp, index=False)
    local.to_csv(lp, index=False)

    out = tmp_path / "out"
    report = transfer_experiment.main(
        [
            "--scout-hosts",
            str(hp),
            "--drivers",
            str(lp),
            "--out-dir",
            str(out),
            "--ks",
            "2,4",
            "--reps",
            "2",
            "--rf-trees",
            "5",
        ]
    )
    assert set(report["targets"]) == set(m.TARGETS)
    for name in (
        "cv_metrics.csv",
        "test_metrics.csv",
        "learning_curve.csv",
        "learning_curve_summary.csv",
    ):
        assert (out / name).exists()
    saved = json.loads((out / "report.json").read_text())
    assert "dataset_mb" in saved["source"]["not_transferred"]
    summary = pd.read_csv(out / "learning_curve_summary.csv")
    assert "scout_residual_beats_local_ridge_wl" in summary.columns
