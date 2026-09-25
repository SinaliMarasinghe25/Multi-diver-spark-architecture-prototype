#!/usr/bin/env python
# =============================================================================
# scripts/build_scout_dataset.py
# Phase 6 — build the Scout source-domain tables (P6-01)
#
# PURPOSE
# -------
# Walk a local Scout clone, keep the target workloads defined in
# mpj_spark/resource_prediction/scout.py (CONFIGURATION block) and write:
#
#   data/scout/scout_runs.csv      one row per job
#   data/scout/scout_hosts.csv     one row per VM per job
#   data/scout/provenance.json     Scout commit, licence, SHA-256 of every file used
#   data/scout/build_report.json   kept/dropped counts with reasons, sanity checks
#
# data/ is git-ignored; outputs are rebuilt, never committed.
#
#   git clone https://github.com/oxhead/scout.git ../scout      # ~3 GB
#   python scripts/build_scout_dataset.py --scout-dir ../scout
#
# CLI FLAGS
#   --scout-dir PATH   Scout clone (contains dataset/osr_*)   (required)
#   --out-dir PATH     output directory         (default: data/scout)
#   --workers N        parallel processes       (default: CPU count)
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction import scout  # noqa: E402


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the Scout source-domain tables.")
    p.add_argument("--scout-dir", required=True)
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "scout"))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return p.parse_args(argv)


def _scout_commit(scout_dir: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", scout_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _licence(scout_dir: str) -> str | None:
    path = os.path.join(scout_dir, "LICENSE")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return " ".join(fh.readline().split() + fh.readline().split() + fh.readline().split())


def build(args) -> dict:
    records = scout.discover_records(args.scout_dir)
    if not records:
        sys.exit(f"No Scout records under {args.scout_dir!r} (expected dataset/osr_*/)")
    print(f"[scout] {len(records)} records found — processing with {args.workers} workers")

    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(scout.process_record, records, chunksize=64)
    else:
        results = [scout.process_record(r) for r in records]
    status = Counter(r["status"] for r in results)
    print(f"[scout] processed in {time.time() - t0:.1f}s — {dict(status)}")

    runs, hosts, ram_ratio = scout.assemble_tables(results)
    if runs.empty:
        sys.exit("[scout] no records matched the target workloads")

    os.makedirs(args.out_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.out_dir, "scout_runs.csv"), index=False)
    hosts.to_csv(os.path.join(args.out_dir, "scout_hosts.csv"), index=False)

    files = {}
    for r in results:
        files.update(r.get("files", {}))
    digest = hashlib.sha256("".join(f"{k}:{files[k]}\n" for k in sorted(files)).encode())
    provenance = {
        "source": "https://github.com/oxhead/scout",
        "scout_commit": _scout_commit(args.scout_dir),
        "licence": _licence(args.scout_dir),
        "citation": "Hsu, Nair, Menzies, Freeh. Scout: An Experienced Guide to Find the "
        "Best Cloud Configuration. arXiv:1803.01296, 2018.",
        "files_used": len(files),
        "files_sha256_digest": digest.hexdigest(),
        "files_sha256": dict(sorted(files.items())),
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=1)

    groups = runs.groupby("config_group")["split"].nunique()
    report = {
        "config": {
            "target_workloads": {f"{w}|{f}": t for (w, f), t in scout.TARGET_WORKLOADS.items()},
            "expected_program": scout.EXPECTED_PROGRAM,
            "datasize_rank": scout.DATASIZE_RANK,
            "split_fractions": scout.SPLIT_FRACTIONS,
            "min_trace_samples": scout.MIN_TRACE_SAMPLES,
        },
        "records_scanned": len(records),
        "record_status": dict(status),
        "runs": len(runs),
        "host_rows": len(hosts),
        "runs_by_setting_workload": {
            f"{s}|{w}": int(n)
            for (s, w), n in runs.groupby(["setting", "workload_type"]).size().items()
        },
        "runs_by_split": runs["split"].value_counts().to_dict(),
        "config_groups": int(len(groups)),
        "config_groups_spanning_splits": int((groups > 1).sum()),
        "input_size_known_by_workload": runs.groupby("workload_type")["input_size_known"]
        .mean()
        .round(3)
        .to_dict(),
        "checks": {
            "measured_ram_over_spec_median": round(float(np.median(ram_ratio)), 4),
            "measured_ram_over_spec_min": round(float(np.min(ram_ratio)), 4),
            "trace_coverage_median": round(float(runs["trace_coverage"].median()), 4),
            "trace_coverage_min": round(float(runs["trace_coverage"].min()), 4),
            "nan_in_labels": int(runs.filter(like="host_").isna().sum().sum()),
        },
    }
    with open(os.path.join(args.out_dir, "build_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv=None):
    args = _parse_args(argv)
    report = build(args)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))
    print(f"[scout] outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
