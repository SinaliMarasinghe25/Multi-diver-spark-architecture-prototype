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
# New result files: drop them into --raw-dir and run the script again.
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


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the local per-driver resource tables.")
    p.add_argument("--raw-dir", default=os.path.join(_PROJECT_ROOT, "data", "local", "raw"))
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "local"))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    drivers, runs, report = local_data.build_tables(args.raw_dir)
    report = local_data.summarise(drivers, runs, report)

    os.makedirs(args.out_dir, exist_ok=True)
    drivers.to_csv(os.path.join(args.out_dir, "local_drivers.csv"), index=False)
    runs.to_csv(os.path.join(args.out_dir, "local_runs.csv"), index=False)
    with open(os.path.join(args.out_dir, "build_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"[local] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
