#!/usr/bin/env python
# =============================================================================
# scripts/train_resource_predictor.py
# Phase 6 — train and evaluate the per-driver resource predictor (2b)
#
# STAGE 1 — local-only baselines
# ------------------------------
# For each target (cpu_cores_mean, cpu_cores_p95, mem_mb_peak) this script:
#   1. fits lookup / Ridge / Random Forest on the TRAIN split,
#   2. picks Ridge alpha and RF depth/leaf size on the VAL split (MAE),
#   3. calibrates an empirical upper margin from VAL residuals,
#   4. refits the chosen models on TRAIN+VAL,
#   5. evaluates ONCE on the held-out TEST split (overall and per workload).
#
# Input:  data/local/local_drivers.csv  (scripts/build_local_dataset.py)
# Output: data/models/local/  (git-ignored)
#   metrics.csv          test metrics per target × model × workload
#   predictions_test.csv per-row test predictions and upper bounds
#   <target>__<model>.joblib   fitted models (refit on train+val)
#   model_card.json      data, features, ranges, choices, versions
#
#   python scripts/train_resource_predictor.py
#
# CLI FLAGS
#   --drivers PATH         local_drivers.csv     (default: data/local/local_drivers.csv)
#   --out-dir PATH         output directory      (default: data/models/local)
#   --unknown-alloc MODE   drop (default) — rows without recorded cores/heap
#                          are excluded until their allocation is confirmed
#   --upper-quantile Q     residual quantile for the upper bound (default 0.9)
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction import models as m  # noqa: E402


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the per-driver resource predictor.")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "local"))
    p.add_argument("--unknown-alloc", choices=["drop"], default="drop")
    p.add_argument("--upper-quantile", type=float, default=0.9)
    return p.parse_args(argv)


def load_training_rows(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    info = {"rows_in_file": len(df)}
    df = df[df["in_main_set"] & (df["status"] == "ok")]
    info["rows_main_ok"] = len(df)
    unknown = ~df["alloc_known"]
    info["rows_dropped_unknown_alloc"] = int(unknown.sum())
    info["workloads_dropped_entirely"] = sorted(
        set(df.loc[unknown, "workload_type"]) - set(df.loc[~unknown, "workload_type"])
    )
    df = df[~unknown].copy()
    m.check_features(df)
    return df, info


def _fit_predict(model, train, test, target):
    model.fit(train[m.FEATURES], train[target])
    return model, model.predict(test[m.FEATURES])


def select_and_train(df: pd.DataFrame, target: str, q: float) -> tuple[list[dict], list, dict]:
    """Return (metric rows, prediction frames, choices) for one target."""
    d = df[df[target].notna()]
    train, val, test = (d[d["split"] == s] for s in ("train", "val", "test"))
    trval = pd.concat([train, val])

    # ── selection on validation ──────────────────────────────────────────
    ridge_scores = {
        a: m.regression_metrics(val[target], _fit_predict(m.make_ridge(a), train, val, target)[1])[
            "mae"
        ]
        for a in m.RIDGE_ALPHAS
    }
    best_alpha = min(ridge_scores, key=ridge_scores.get)
    rf_scores = {
        i: m.regression_metrics(val[target], _fit_predict(m.make_rf(**p), train, val, target)[1])[
            "mae"
        ]
        for i, p in enumerate(m.RF_GRID)
    }
    best_rf = m.RF_GRID[min(rf_scores, key=rf_scores.get)]

    factories = {
        "lookup": m.LookupModel,
        "ridge": lambda: m.make_ridge(best_alpha),
        "rf": lambda: m.make_rf(**best_rf),
    }

    rows, preds, fitted = [], [], {}
    for name, factory in factories.items():
        # upper margin from validation residuals of a TRAIN-only fit
        _, val_pred = _fit_predict(factory(), train, val, target)
        margin = m.upper_margin(val[target], val_pred, q)
        # final model on TRAIN+VAL, evaluated once on TEST
        model, test_pred = _fit_predict(factory(), trval, test, target)
        upper = test_pred + margin
        fitted[name] = model
        for wl, idx in [("all", test.index)] + [
            (w, g.index) for w, g in test.groupby("workload_type")
        ]:
            mask = test.index.isin(idx)
            rows.append(
                {
                    "target": target,
                    "unit": m.TARGETS[target],
                    "model": name,
                    "workload": wl,
                    "upper_margin": margin,
                    **m.regression_metrics(test[target][mask], test_pred[mask], upper[mask]),
                }
            )
        p = test[["run_key", "driver_idx", "workload_type", *m.NUMERIC_FEATURES]].copy()
        p["target"], p["model"] = target, name
        p["actual"], p["predicted"], p["upper"] = test[target].values, test_pred, upper
        if name == "lookup":
            p["lookup_level"] = model.level_used_
        preds.append(p)

    choices = {
        "ridge_alpha": best_alpha,
        "ridge_val_mae": ridge_scores,
        "rf_params": best_rf,
        "rf_val_mae": {str(m.RF_GRID[i]): s for i, s in rf_scores.items()},
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "fitted": fitted,
    }
    return rows, preds, choices


def main(argv=None):
    args = _parse_args(argv)
    df, info = load_training_rows(args.drivers)
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows, all_preds, card_targets = [], [], {}
    for target in m.TARGETS:
        rows, preds, choices = select_and_train(df, target, args.upper_quantile)
        all_rows += rows
        all_preds += preds
        for name, model in choices.pop("fitted").items():
            joblib.dump(model, os.path.join(args.out_dir, f"{target}__{name}.joblib"))
        card_targets[target] = choices

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    pd.concat(all_preds).to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)

    with open(args.drivers, "rb") as fh:
        data_sha = hashlib.sha256(fh.read()).hexdigest()
    card = {
        "stage": "local_only_baselines",
        "data": {"drivers_csv": os.path.relpath(args.drivers, _PROJECT_ROOT), "sha256": data_sha},
        "rows": info,
        "features": m.FEATURES,
        "targets": m.TARGETS,
        "valid_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in m.NUMERIC_FEATURES}
        | {"workload_type": sorted(df["workload_type"].unique())},
        "rows_by_split": df["split"].value_counts().to_dict(),
        "upper_quantile": args.upper_quantile,
        "choices": card_targets,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with open(os.path.join(args.out_dir, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=2, default=str)

    show = metrics[metrics["workload"] == "all"][
        [
            "target",
            "unit",
            "model",
            "n",
            "mae",
            "rmse",
            "mape_pct",
            "r2",
            "underprediction_rate",
            "upper_bound_coverage",
        ]
    ]
    print(f"[train] rows used: {info}")
    print(show.round(3).to_string(index=False))
    print(f"[train] outputs written to {args.out_dir}")
    return metrics


if __name__ == "__main__":
    main()
