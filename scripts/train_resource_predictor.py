#!/usr/bin/env python
# =============================================================================
# scripts/train_resource_predictor.py
# Phase 6 — train and evaluate the per-driver resource predictor (2b)
#
# STAGE 1 — local-only baselines
# ------------------------------
# For each target (cpu_cores_mean, cpu_cores_p95, mem_mb_peak) this script:
#   1. runs leave-one-configuration-out CV on TRAIN+VAL: every configuration
#      group is predicted by models trained on all other groups,
#   2. picks Ridge alpha and RF depth/leaf size by CV MAE,
#   3. reports CV errors with 95 % group-bootstrap intervals and paired
#      differences against the lookup baseline,
#   4. calibrates the empirical upper margin from out-of-fold residuals,
#   5. refits the chosen models on TRAIN+VAL and evaluates ONCE on TEST.
#
# Input:  data/local/local_drivers.csv  (scripts/build_local_dataset.py)
# Output: data/models/local/  (git-ignored)
#   metrics.csv          test metrics per target × model × workload
#   cv_metrics.csv       out-of-fold CV metrics per target × model × workload
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


def _cv_rows(oof: pd.DataFrame, target: str) -> list[dict]:
    """Pooled out-of-fold metrics per model, overall and per workload."""
    rows = []
    for name, g in oof.groupby("model"):
        for wl, gw in [("all", g)] + list(g.groupby("workload_type")):
            rows.append(
                {
                    "target": target,
                    "unit": m.TARGETS[target],
                    "model": name,
                    "workload": wl,
                    "n_groups": int(gw["config_group"].nunique()),
                    **m.regression_metrics(gw["actual"], gw["predicted"]),
                }
            )
    return rows


def select_and_train(
    df: pd.DataFrame, target: str, q: float
) -> tuple[list[dict], list[dict], list, dict]:
    """Return (test metric rows, CV metric rows, test prediction frames, choices)."""
    d = df[df[target].notna()]
    trval = d[d["split"].isin(["train", "val"])]
    test = d[d["split"] == "test"]

    # ── selection by leave-one-configuration-out CV on TRAIN+VAL ──────────
    candidates = {"lookup": m.LookupModel}
    candidates |= {f"ridge|{a}": (lambda a=a: m.make_ridge(a)) for a in m.RIDGE_ALPHAS}
    candidates |= {
        f"rf|{p['max_depth']}|{p['min_samples_leaf']}": (lambda p=p: m.make_rf(**p))
        for p in m.RF_GRID
    }
    oof_all = m.group_cv_predictions(trval, target, candidates)
    cand_mae = (
        oof_all.assign(ae=(oof_all["predicted"] - oof_all["actual"]).abs())
        .groupby("model")["ae"]
        .mean()
    )
    best = {
        family: cand_mae[[c for c in cand_mae.index if c.split("|")[0] == family]].idxmin()
        for family in ("lookup", "ridge", "rf")
    }
    factories = {family: candidates[key] for family, key in best.items()}
    oof = oof_all[oof_all["model"].isin(best.values())].replace(
        {"model": {v: k for k, v in best.items()}}
    )
    cv_rows = _cv_rows(oof, target)
    uncertainty = {
        name: m.group_bootstrap(oof, name, baseline=None if name == "lookup" else "lookup")
        for name in factories
    }

    rows, preds, fitted = [], [], {}
    for name, factory in factories.items():
        # upper margin from out-of-fold residuals of the selected configuration
        sel = oof[oof["model"] == name]
        margin = m.upper_margin(sel["actual"], sel["predicted"], q)
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
        "selected": best,
        "cv_mae_by_candidate": cand_mae.round(4).to_dict(),
        "cv_uncertainty": uncertainty,
        "n_trainval_rows": len(trval),
        "n_trainval_groups": int(trval["config_group"].nunique()),
        "n_test_rows": len(test),
        "n_test_groups": int(test["config_group"].nunique()),
        "fitted": fitted,
    }
    return rows, cv_rows, preds, choices


def main(argv=None):
    args = _parse_args(argv)
    df, info = load_training_rows(args.drivers)
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows, all_cv, all_preds, card_targets = [], [], [], {}
    for target in m.TARGETS:
        rows, cv_rows, preds, choices = select_and_train(df, target, args.upper_quantile)
        all_rows += rows
        all_cv += cv_rows
        all_preds += preds
        for name, model in choices.pop("fitted").items():
            joblib.dump(model, os.path.join(args.out_dir, f"{target}__{name}.joblib"))
        card_targets[target] = choices

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    cv_metrics = pd.DataFrame(all_cv)
    cv_metrics.to_csv(os.path.join(args.out_dir, "cv_metrics.csv"), index=False)
    pd.concat(all_preds).to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)

    with open(args.drivers, "rb") as fh:
        data_sha = hashlib.sha256(fh.read()).hexdigest()
    card = {
        "stage": "local_only_baselines",
        "protocol": "leave-one-configuration-out CV on train+val for selection, upper-margin calibration and uncertainty; single final evaluation on test",
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
    cv_show = cv_metrics[cv_metrics["workload"] == "all"][
        ["target", "unit", "model", "n_groups", "mae", "mape_pct", "underprediction_rate"]
    ]
    print("── leave-one-configuration-out CV (train+val) ──")
    print(cv_show.round(3).to_string(index=False))
    for target, ch in card_targets.items():
        for name, u in ch["cv_uncertainty"].items():
            diff = (
                f"  diff vs lookup {u['mae_diff_vs_baseline']:+.3f} "
                f"[{u['mae_diff_ci95'][0]:+.3f}, {u['mae_diff_ci95'][1]:+.3f}]"
                if "baseline" in u
                else ""
            )
            print(
                f"   {target:15s} {name:6s} MAE {u['mae']:.3f} "
                f"[{u['mae_ci95'][0]:.3f}, {u['mae_ci95'][1]:.3f}]{diff}"
            )
    print("── final evaluation on TEST ──")
    print(show.round(3).to_string(index=False))
    print(f"[train] outputs written to {args.out_dir}")
    return metrics


if __name__ == "__main__":
    main()
