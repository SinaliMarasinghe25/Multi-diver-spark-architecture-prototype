# =============================================================================
# mpj_spark/resource_prediction/local_data.py
# Phase 6 — local target-domain tables from per-rank profiling CSVs (P6-01)
#
# PURPOSE
# -------
# Turn the per-rank resource CSVs recorded on the testbed laptop (MPI runs of
# WordCount, K-Means and LogReg with a per-process CPU / RSS sampler) into the
# TARGET-DOMAIN tables of the Phase 6 resource predictor:
#
#   local_drivers.csv   one row per Spark driver (rank >= 1) per run
#   local_runs.csv      one row per run (driver aggregates + run outcome)
#
# Input files share one column layout (older files lack the allocation and
# p95 columns; those values become missing, never invented).
#
# MEASUREMENT SCOPE
# -----------------
# Labels are PER DRIVER PROCESS TREE as recorded by the sampler:
#   cpu_*_pct  — process CPU percent, 100 % = one core  →  divided by 100
#   rss_*_mb   — resident memory of the driver process (unit as recorded)
# Host-wide columns (sys_*) are measured DURING the run, so they are kept as
# diagnostics only and are never predictor inputs.
#
# CHANGING THE SCOPE
# ------------------
# Everything a researcher may adjust lives in the CONFIGURATION block below.
# Rebuild with scripts/build_local_dataset.py after editing.
# =============================================================================

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION — edit here to change what the dataset contains
# =============================================================================

# Files left out completely, with the reason recorded in the build report.
EXCLUDED_FILES: dict[str, str] = {
    "run1_no_cpu.csv": "CPU columns were not recorded (all 0); the same "
    "configurations were re-run with CPU in resource_runs.csv",
}

# Experiment tag per file-name prefix (first match wins).
EXPERIMENT_BY_PREFIX: list[tuple[str, str]] = [
    ("resource_runs", "default_alloc"),
    ("grid_", "grid"),
    ("floor_", "memory_floor"),
    ("blas_test", "blas_test"),
]

# Experiments that changed the runtime environment; kept in the tables but
# flagged out of the main training set (in_main_set = False).
SEPARATE_EXPERIMENTS: set[str] = {"blas_test"}

# exit codes that mean "stopped by the time limit" → censored, not a label.
TIMEOUT_EXIT_CODES: set[int] = {124}

WORKLOAD_CLASS: dict[str, str] = {
    "wordcount": "batch",
    "kmeans": "iterative",
    "logreg": "iterative",
}

SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# =============================================================================
# Output schema
# =============================================================================

TRACKING_COLUMNS = [
    "source",
    "file",
    "experiment",
    "run_id",
    "run_key",
    "timestamp",
    "host",
    "rep",
    "config_group",
    "split",
    "in_main_set",
]
INPUT_COLUMNS = [
    "workload_type",
    "workload_class",
    "dataset_mb",
    "num_workers",
    "cores_per_driver",
    "heap_mb_per_driver",
    "alloc_known",
]
DRIVER_LABEL_COLUMNS = [
    "cpu_cores_mean",
    "cpu_cores_p95",
    "cpu_cores_peak",
    "mem_mb_mean",
    "mem_mb_p95",
    "mem_mb_peak",
]
OUTCOME_COLUMNS = ["wall_s", "exit_code", "status"]
DIAGNOSTIC_COLUMNS = [
    "n_samples",
    "host_cpu_avg_pct",
    "host_mem_avg_mb",
    "host_mem_peak_mb",
    "host_mem_avail_min_mb",
]

_RAW_RENAME = {
    "app": "workload_type",
    "size": "dataset_mb",
    "workers": "num_workers",
    "cores_alloc": "cores_per_driver",
    "heap_mb_alloc": "heap_mb_per_driver",
    "sys_cpu_avg_pct": "host_cpu_avg_pct",
    "sys_mem_avg_mb": "host_mem_avg_mb",
    "sys_mem_peak_mb": "host_mem_peak_mb",
    "sys_mem_avail_min_mb": "host_mem_avail_min_mb",
    "rss_avg_mb": "mem_mb_mean",
    "rss_p95_mb": "mem_mb_p95",
    "rss_peak_mb": "mem_mb_peak",
}

# =============================================================================
# Loading
# =============================================================================


def experiment_for(filename: str) -> str:
    for prefix, tag in EXPERIMENT_BY_PREFIX:
        if filename.startswith(prefix):
            return tag
    return "other"


def load_file(path: str) -> pd.DataFrame:
    """Read one raw CSV and bring it to the common column set."""
    df = pd.read_csv(path)
    name = os.path.basename(path)
    if "role" not in df:  # older layout: rank 0 is the MPI root
        df["role"] = np.where(df["driver_idx"] == 0, "root", "driver")
    for col in ("cores_alloc", "heap_mb_alloc", "cpu_p95_pct", "rss_p95_mb"):
        if col not in df:
            df[col] = np.nan
    df = df.rename(columns=_RAW_RENAME)
    df["file"] = name
    df["experiment"] = experiment_for(name)
    return df


def config_group(row) -> str:
    """Repetitions of one configuration share this key (and therefore a split)."""
    cores = "auto" if pd.isna(row["cores_per_driver"]) else int(row["cores_per_driver"])
    heap = "auto" if pd.isna(row["heap_mb_per_driver"]) else int(row["heap_mb_per_driver"])
    variant = row["experiment"] if row["experiment"] in SEPARATE_EXPERIMENTS else "main"
    return "|".join(
        str(v)
        for v in (
            variant,
            row["workload_type"],
            int(row["dataset_mb"]),
            int(row["num_workers"]),
            cores,
            heap,
        )
    )


def stratified_split(groups: pd.DataFrame, fractions: dict[str, float] = SPLIT_FRACTIONS) -> dict:
    """
    {config_group: split}, stratified by workload so every workload appears in
    val and test (a plain hash split leaves some workloads out when there are
    only a few dozen groups).  Groups are ordered by a hash of their key, so
    the split is deterministic and does not depend on file order.
    Workloads with fewer than 3 groups go entirely to train.
    """
    out = {}
    for _, g in groups.groupby("stratum"):
        keys = sorted(
            g["config_group"].unique(), key=lambda k: hashlib.sha256(k.encode()).hexdigest()
        )
        n = len(keys)
        n_test = max(1, round(n * fractions["test"])) if n >= 3 else 0
        n_val = max(1, round(n * fractions["val"])) if n >= 3 else 0
        for i, key in enumerate(keys):
            out[key] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def status_for(exit_code: int) -> str:
    if exit_code == 0:
        return "ok"
    if exit_code in TIMEOUT_EXIT_CODES:
        return "timeout"
    return "failed"


# =============================================================================
# Table assembly
# =============================================================================


def build_tables(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (drivers, runs, report) from every *.csv in raw_dir."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    report = {"files": {}, "excluded_files": {}}
    frames = []
    for path in files:
        name = os.path.basename(path)
        if name in EXCLUDED_FILES:
            report["excluded_files"][name] = EXCLUDED_FILES[name]
            continue
        df = load_file(path)
        report["files"][name] = {
            "rows": len(df),
            "runs": int(df["run_id"].nunique()),
            "experiment": df["experiment"].iloc[0] if len(df) else None,
        }
        frames.append(df)
    if not frames:
        raise ValueError(f"no usable CSV files in {raw_dir!r}")
    raw = pd.concat(frames, ignore_index=True)

    # run_id alone repeats across batches; file + run_id + timestamp is unique.
    raw["run_key"] = raw["file"] + ":" + raw["run_id"] + "@" + raw["timestamp"].astype(str)
    raw["source"] = "local"
    raw["workload_class"] = raw["workload_type"].map(WORKLOAD_CLASS)
    raw["alloc_known"] = raw["cores_per_driver"].notna() & raw["heap_mb_per_driver"].notna()
    raw["config_group"] = raw.apply(config_group, axis=1)
    raw["in_main_set"] = ~raw["experiment"].isin(SEPARATE_EXPERIMENTS)
    raw["stratum"] = raw["workload_type"] + "|" + raw["in_main_set"].astype(str)
    raw["split"] = raw["config_group"].map(
        stratified_split(raw[["config_group", "stratum"]].drop_duplicates())
    )
    raw["status"] = raw["exit_code"].map(status_for)
    for stat in ("avg", "p95", "peak"):
        name = "mean" if stat == "avg" else stat
        raw[f"cpu_cores_{name}"] = raw[f"cpu_{stat}_pct"] / 100.0

    dup = raw.duplicated(["run_key", "driver_idx"])
    report["duplicate_rows_dropped"] = int(dup.sum())
    raw = raw[~dup]

    drivers = raw[raw["role"] == "driver"].copy()
    report["root_rows_dropped"] = int((raw["role"] != "driver").sum())
    driver_cols = (
        TRACKING_COLUMNS
        + ["driver_idx"]
        + INPUT_COLUMNS
        + DRIVER_LABEL_COLUMNS
        + OUTCOME_COLUMNS
        + DIAGNOSTIC_COLUMNS
    )
    drivers = drivers[driver_cols].reset_index(drop=True)

    agg = drivers.groupby("run_key").agg(
        drivers_recorded=("driver_idx", "count"),
        **{f"{c}_avg_drivers": (c, "mean") for c in DRIVER_LABEL_COLUMNS},
        **{f"{c}_max_drivers": (c, "max") for c in DRIVER_LABEL_COLUMNS},
    )
    run_first = (
        drivers.drop(columns=["driver_idx", *DRIVER_LABEL_COLUMNS])
        .groupby("run_key", as_index=False)
        .first()
    )
    runs = run_first.merge(agg, on="run_key").reset_index(drop=True)
    return drivers, runs, report


def summarise(drivers: pd.DataFrame, runs: pd.DataFrame, report: dict) -> dict:
    """Counts and sanity checks for the build report."""
    ok = drivers[drivers["status"] == "ok"]
    known = ok[ok["alloc_known"]]
    groups = runs.groupby("config_group")["split"].nunique()
    report.update(
        {
            "runs": len(runs),
            "driver_rows": len(drivers),
            "runs_by_status": runs["status"].value_counts().to_dict(),
            "runs_by_experiment": runs["experiment"].value_counts().to_dict(),
            "runs_by_workload": runs["workload_type"].value_counts().to_dict(),
            "runs_by_split": runs["split"].value_counts().to_dict(),
            "config_groups": int(len(groups)),
            "config_groups_spanning_splits": int((groups > 1).sum()),
            "main_set_ok_driver_rows": int(ok["in_main_set"].sum()),
            "checks": {
                "share_cpu_p95_above_cores_given": (
                    round(float((known["cpu_cores_p95"] > known["cores_per_driver"]).mean()), 3)
                    if len(known)
                    else None
                ),
                "share_mem_peak_above_heap_given": (
                    round(float((known["mem_mb_peak"] > known["heap_mb_per_driver"]).mean()), 3)
                    if len(known)
                    else None
                ),
                "missing_cpu_p95_rows": int(ok["cpu_cores_p95"].isna().sum()),
                "missing_allocation_rows": int((~ok["alloc_known"]).sum()),
            },
        }
    )
    return report
