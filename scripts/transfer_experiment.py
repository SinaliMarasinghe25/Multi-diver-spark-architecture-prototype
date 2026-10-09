#!/usr/bin/env python
# =============================================================================
# scripts/transfer_experiment.py
# Phase 6 — does Scout help? (transfer-learning experiment, Objective 2b)
#
# Compares, per target (cpu_cores_mean, cpu_cores_p95, mem_mb_peak):
#   local_lookup, local_ridge_wl, local_rf          — local data only
#   scout_only, scout_offset, scout_residual        — Scout source line (+ local)
# (methods: mpj_spark/resource_prediction/transfer.py)
#
# Three protocols:
#   1. leave-one-configuration-out CV on TRAIN+VAL (all local data),
#      with 95 % group-bootstrap intervals and paired differences
#   2. LEARNING CURVE on TRAIN+VAL: train on k random configuration groups
#      (same groups for every method), evaluate on the rest; repeated
#   3. single final evaluation on the frozen TEST split
#
# Inputs:  data/scout/scout_hosts.csv     (scripts/build_scout_dataset.py)
#          data/local/local_drivers.csv   (scripts/build_local_dataset.py)
# Output:  data/models/transfer/  (git-ignored)
#   cv_metrics.csv, test_metrics.csv, learning_curve.csv,
#   learning_curve_summary.csv, report.json
#
#   python scripts/transfer_experiment.py
#
# CLI FLAGS
#   --scout-hosts PATH   --drivers PATH   --out-dir PATH
#   --ks 2,4,8,16,32     training-set sizes (configuration groups) for the curve
#   --reps 20            repetitions per size
#   --rf-trees 200       trees in the local random forest
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction import models as m  # noqa: E402
from mpj_spark.resource_prediction import transfer as t  # noqa: E402

# (method, baseline) pairs reported with paired bootstrap differences
COMPARISONS = [
    ("scout_residual", "local_ridge_wl"),  # same model family: does Scout add value?
    ("scout_offset", "local_lookup"),
    ("scout_residual", "local_rf"),
    ("scout_only", "local_lookup"),
]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scout -> local transfer experiment.")
    p.add_argument(
        "--scout-hosts", default=os.path.join(_PROJECT_ROOT, "data", "scout", "scout_hosts.csv")
    )
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "transfer"))
    p.add_argument("--ks", default="2,4,8,16,32")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--rf-trees", type=int, default=200)
    return p.parse_args(argv)


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_local(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["in_main_set"] & (df["status"] == "ok") & df["alloc_known"]].copy()
    m.check_features(df)
    return df


def _metric_rows(frame: pd.DataFrame, target: str, protocol: str) -> list[dict]:
    rows = []
    for name, g in frame.groupby("model"):
        for wl, gw in [("all", g)] + list(g.groupby("workload_type")):
            rows.append(
                {
                    "protocol": protocol,
                    "target": target,
                    "unit": m.TARGETS[target],
                    "model": name,
                    "workload": wl,
                    **m.regression_metrics(gw["actual"], gw["predicted"]),
                }
            )
    return rows


def run_target(local: pd.DataFrame, source: pd.DataFrame, target: str, args) -> dict:
    lines = t.SourceLines().fit(source, target)
    d = local[local[target].notna()].copy()
    d["source_pred"] = lines.predict(d)
    trval = d[d["split"].isin(["train", "val"])]
    test = d[d["split"] == "test"]
    factories = t.method_factories(args.rf_trees)

    # 1. leave-one-configuration-out CV on train+val
    oof = t.group_cv(trval, target, factories)
    cv_rows = _metric_rows(oof, target, "cv_train_val")
    paired = {f"{a}_vs_{b}": m.group_bootstrap(oof, a, baseline=b) for a, b in COMPARISONS}

    # 2. learning curve on train+val (identical adaptation groups per method)
    ks = [int(k) for k in args.ks.split(",")]
    curve = t.learning_curve(trval, target, factories, ks, reps=args.reps)

    # 3. single final evaluation on the frozen test split
    test_frames = []
    for name, factory in factories.items():
        test_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "workload_type": test["workload_type"].values,
                    "actual": test[target].values,
                    "predicted": t._fit_predict(factory, trval, test, target),
                }
            )
        )
    test_rows = _metric_rows(pd.concat(test_frames), target, "test")

    return {
        "cv_rows": cv_rows,
        "test_rows": test_rows,
        "curve": curve,
        "info": {
            "source_lines": lines.describe(),
            "source_label": t.SOURCE_LABELS[target],
            "share_local_rows_outside_scout_core_range": float(lines.extrapolating(d).mean()),
            "n_trainval_groups": int(trval["config_group"].nunique()),
            "n_test_groups": int(test["config_group"].nunique()),
            "paired_cv": paired,
        },
    }


def summarise_curve(curve: pd.DataFrame) -> pd.DataFrame:
    g = curve.groupby(["target", "k_groups", "model"])["mae"]
    out = g.agg(
        mae_median="median",
        mae_q25=lambda x: x.quantile(0.25),
        mae_q75=lambda x: x.quantile(0.75),
        reps="count",
    ).reset_index()
    # how often the Scout residual model beats the same-family local model
    piv = curve.pivot_table(
        index=["target", "k_groups", "rep"], columns="model", values="mae"
    ).reset_index()
    if {"scout_residual", "local_ridge_wl"} <= set(piv.columns):
        win = (
            (piv["scout_residual"] < piv["local_ridge_wl"])
            .groupby([piv["target"], piv["k_groups"]])
            .mean()
            .rename("scout_residual_beats_local_ridge_wl")
            .reset_index()
        )
        out = out.merge(win, on=["target", "k_groups"], how="left")
    return out


def main(argv=None):
    args = _parse_args(argv)
    local = load_local(args.drivers)
    source = t.scout_source_frame(pd.read_csv(args.scout_hosts))
    os.makedirs(args.out_dir, exist_ok=True)

    cv_rows, test_rows, curves, info = [], [], [], {}
    for target in m.TARGETS:
        res = run_target(local, source, target, args)
        cv_rows += res["cv_rows"]
        test_rows += res["test_rows"]
        curves.append(res["curve"])
        info[target] = res["info"]

    cv = pd.DataFrame(cv_rows)
    test = pd.DataFrame(test_rows)
    curve = pd.concat(curves, ignore_index=True)
    curve_summary = summarise_curve(curve)
    cv.to_csv(os.path.join(args.out_dir, "cv_metrics.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test_metrics.csv"), index=False)
    curve.to_csv(os.path.join(args.out_dir, "learning_curve.csv"), index=False)
    curve_summary.to_csv(os.path.join(args.out_dir, "learning_curve_summary.csv"), index=False)

    report = {
        "question": "Does the Scout source model improve per-driver prediction on local data?",
        "source": {
            "file": os.path.relpath(args.scout_hosts, _PROJECT_ROOT),
            "sha256": _sha(args.scout_hosts),
            "setting": t.SOURCE_SETTING,
            "rows": len(source),
            "shared_features": ["workload_type", "cores_per_driver (Scout vcpus_per_host)"],
            "not_transferred": {
                "dataset_mb": "local 25-50 MB per driver vs Scout >= 769 MiB: out of range",
                "heap_mb_per_driver": "no Scout counterpart",
            },
        },
        "local": {
            "file": os.path.relpath(args.drivers, _PROJECT_ROOT),
            "sha256": _sha(args.drivers),
            "rows": len(local),
        },
        "protocols": {
            "cv_train_val": "leave-one-configuration-out on train+val",
            "learning_curve": f"k in {args.ks}, {args.reps} repetitions, train+val only",
            "test": "single evaluation on the frozen test split",
        },
        "targets": info,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("── leave-one-configuration-out CV (train+val), MAE ──")
    print(
        cv[cv["workload"] == "all"]
        .pivot(index="model", columns="target", values="mae")
        .round(3)
        .to_string()
    )
    print("── paired differences (method − baseline), 95 % CI; negative = method better ──")
    for target, inf in info.items():
        for key, r in inf["paired_cv"].items():
            lo, hi = r["mae_diff_ci95"]
            print(
                f"   {target:15s} {key:34s} {r['mae_diff_vs_baseline']:+9.3f} [{lo:+.3f}, {hi:+.3f}]"
            )
    print("── learning curve: median MAE by number of local configuration groups ──")
    print(
        curve_summary.pivot_table(
            index=["target", "k_groups"], columns="model", values="mae_median"
        )
        .round(3)
        .to_string()
    )
    print("── final evaluation on TEST, MAE ──")
    print(
        test[test["workload"] == "all"]
        .pivot(index="model", columns="target", values="mae")
        .round(3)
        .to_string()
    )
    print(f"[transfer] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
