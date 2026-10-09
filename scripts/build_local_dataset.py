#!/usr/bin/env python
# =============================================================================
# scripts/build_local_dataset.py
# Phase 6 — build the local target-domain tables (P6-01)
#
# PURPOSE
# -------
# Read every per-rank profiling CSV in a folder and write the tables used to
# train and evaluate the per-driver resource predictor. The rules (excluded
# files, experiment tags, timeout codes) live in the CONFIGURATION block of
# mpj_spark/resource_prediction/local_data.py.
#
#   python scripts/build_local_dataset.py                      # defaults
#   python scripts/build_local_dataset.py --raw-dir data/local/raw --out-dir data/local
#
# OUTPUTS  (data/local/ is git-ignored)
#   local_drivers.csv   one row per Spark driver per run
#   local_runs.csv      one row per run
#   build_report.json   files used/excluded, counts, sanity checks
#
# SPLIT FILE  (version-controlled: config/phase6_splits.json)
#   FROZEN train/val/test assignment per configuration group. It holds only
#   configuration names (no measurements) and is committed so every machine
#   uses the same held-out test set.
#
# New result files: drop them into --raw-dir and run the script again.
# Groups already listed in the split file keep their split, so the held-out test
# set never changes; only new configuration groups are assigned.  Duplicate
# copies of the same run (same run_id + timestamp + driver) count once.
# --reset-splits re-assigns every group (only before any model has been
# evaluated on the test split).
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction import local_data  # noqa: E402

DEFAULT_SPLITS_FILE = os.path.join(_PROJECT_ROOT, "config", "phase6_splits.json")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the local per-driver resource tables.")
    p.add_argument("--raw-dir", default=os.path.join(_PROJECT_ROOT, "data", "local", "raw"))
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "local"))
    p.add_argument(
        "--splits-file",
        default=None,
        help="frozen split assignment (default: config/phase6_splits.json)",
    )
    p.add_argument(
        "--reset-splits",
        action="store_true",
        help="ignore the saved split file and re-assign every group",
    )
    return p.parse_args(argv)


def load_splits(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)["groups"]


def save_splits(path: str, previous: dict, current: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # keep groups that vanished from the data too: a removed file must not
    # free its groups to be re-assigned to another split later
    merged = {**previous, **current}
    payload = {
        "policy": "workload-stratified, deterministic; existing groups never move",
        "counts": {s: sum(v == s for v in merged.values()) for s in ("train", "val", "test")},
        "groups": dict(sorted(merged.items())),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def main(argv=None):
    args = _parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    splits_file = args.splits_file or DEFAULT_SPLITS_FILE
    frozen = {} if args.reset_splits else load_splits(splits_file)

    drivers, runs, report = local_data.build_tables(args.raw_dir, frozen)
    save_splits(splits_file, frozen, report.pop("split_map"))
    report["splits_file"] = splits_file
    report = local_data.summarise(drivers, runs, report)

    drivers.to_csv(os.path.join(args.out_dir, "local_drivers.csv"), index=False)
    runs.to_csv(os.path.join(args.out_dir, "local_runs.csv"), index=False)
    with open(os.path.join(args.out_dir, "build_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"[local] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
