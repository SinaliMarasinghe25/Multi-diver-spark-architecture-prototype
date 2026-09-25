# =============================================================================
# mpj_spark/resource_prediction/scout.py
# Phase 6 — Scout source-domain dataset (P6-01)
#
# PURPOSE
# -------
# Arrange the public Scout dataset (Hsu et al., 2018; MIT licence;
# https://github.com/oxhead/scout) into two tables that follow the Phase 6
# prediction contract:
#
#   scout_runs.csv   one row per job   — pre-run inputs + run-level labels
#   scout_hosts.csv  one row per VM    — pre-run inputs + per-host labels
#
# MEASUREMENT SCOPE (read before using the labels)
# ------------------------------------------------
# Every Scout resource metric comes from `sar` and describes the WHOLE VM
# (host).  Scout never measured a Spark driver, JVM or container.  Labels are
# therefore named host_* and must NOT be treated as per-driver demand, nor
# divided by the number of hosts.  Scout has no iteration count, sync mode,
# feature dimension or row count; those request fields are absent, not
# invented.  See the audit in docs / the build report for the evidence.
#
# CHANGING THE SCOPE
# ------------------
# Everything a researcher may want to adjust lives in the CONFIGURATION block
# below (which workloads, size ranks, split fractions, minimum trace length).
# Rebuild with scripts/build_scout_dataset.py after editing.
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION — edit here to change what the dataset contains
# =============================================================================

# (Scout workload, Scout framework) → our workload_type.
# Keyed on the framework too, so Hadoop WordCount can never slip in.
TARGET_WORKLOADS: dict[tuple[str, str], str] = {
    ("wordcount", "spark"): "wordcount",  # HiBench, Spark 2.1
    ("kmeans", "spark1.5"): "kmeans",  # spark-perf, Spark 1.5
    ("lr", "spark"): "logreg",  # HiBench, Spark 2.1
}

# Expected report.json `program` per Scout workload. A completed record whose
# program differs is dropped (guards against silent relabelling).
EXPECTED_PROGRAM: dict[str, str] = {
    "wordcount": "ScalaSparkWordcount",
    "kmeans": "kmeans",
    "lr": "LogisticRegression",
}

WORKLOAD_CLASS: dict[str, str] = {
    "wordcount": "batch",
    "kmeans": "iterative",
    "logreg": "iterative",
}

# Global size order, confirmed from lr input bytes (0.8 < 1.2 < 1.6 < 24 < 48 GB).
DATASIZE_RANK: dict[str, int] = {"small": 1, "medium": 2, "large": 3, "huge": 4, "bigdata": 5}

SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_TRACE_SAMPLES = 3  # a host trace needs >= 3 samples (10 s) to yield a p95

# =============================================================================
# Static facts
# =============================================================================

# AWS EC2 published capacity: (vCPUs, memory GiB).
EC2_SPECS: dict[str, tuple[int, float]] = {
    "c3.large": (2, 3.75),
    "c3.xlarge": (4, 7.5),
    "c3.2xlarge": (8, 15.0),
    "c4.large": (2, 3.75),
    "c4.xlarge": (4, 7.5),
    "c4.2xlarge": (8, 15.0),
    "m3.large": (2, 7.5),
    "m3.xlarge": (4, 15.0),
    "m3.2xlarge": (8, 30.0),
    "m4.large": (2, 8.0),
    "m4.xlarge": (4, 16.0),
    "m4.2xlarge": (8, 32.0),
    "r3.large": (2, 15.25),
    "r3.xlarge": (4, 30.5),
    "r3.2xlarge": (8, 61.0),
    "r4.large": (2, 15.25),
    "r4.xlarge": (4, 30.5),
    "r4.2xlarge": (8, 61.0),
}
SPARK_VERSION: dict[str, str] = {"spark": "2.1", "spark1.5": "1.5"}
SAMPLE_INTERVAL_S = 5.0

# Column groups of the output tables (order = CSV column order).
TRACKING_COLUMNS = ["source", "scout_run_id", "setting", "config_group", "split"]
INPUT_COLUMNS = [
    "workload_type",
    "workload_class",
    "spark_version",
    "vm_type",
    "vm_family",
    "num_hosts",
    "vcpus_per_host",
    "mem_mib_per_host",
    "total_vcpus",
    "total_mem_mib",
    "datasize_label",
    "datasize_rank",
    "input_mib",
    "input_size_known",
]
HOST_LABEL_COLUMNS = [
    "host_cpu_busy_mean_pct",
    "host_cpu_busy_p95_pct",
    "host_cpu_busy_p95_cores",
    "host_mem_used_peak_mib",
    "host_mem_app_peak_mib",
]
QUALITY_COLUMNS = ["n_samples", "trace_coverage"]

# =============================================================================
# Parsing
# =============================================================================

_SINGLE_RE = re.compile(
    r"^(?P<vm>[a-z]\d\.\w+?)_i-[0-9a-f]+_(?P<workload>.+)_(?P<framework>hadoop|spark1\.5|spark)"
    r"_(?P<datasize>[a-z]+)_(?P<rep>\d+)$"
)
_MULTI_RE = re.compile(
    r"^(?P<hosts>\d+)_(?P<vm>[a-z]\d\.\w+?)_(?P<workload>.+)_(?P<framework>hadoop|spark1\.5|spark)"
    r"_(?P<datasize>[a-z]+)_(?P<rep>\d+)$"
)


def parse_run_id(run_id: str) -> dict | None:
    """
    Decode a Scout record directory name; None if it matches neither form.

    single_node: ``r4.xlarge_i-0f4e4b248a6aa957a_terasort_spark_small_1``
    multi_node:  ``4_m4.2xlarge_naive-bayes_spark1.5_bigdata_1``
    """
    m, setting = _MULTI_RE.match(run_id), "multi_node"
    if m is None:
        m, setting = _SINGLE_RE.match(run_id), "single_node"
    if m is None:
        return None
    g = m.groupdict()
    return {
        "scout_run_id": run_id,
        "setting": setting,
        "vm_type": g["vm"],
        "num_hosts": int(g.get("hosts") or 1),
        "scout_workload": g["workload"],
        "framework": g["framework"],
        "datasize_label": g["datasize"],
        "repetition": int(g["rep"]),
    }


def config_group(row: dict) -> str:
    """Identity of one configuration; all its repetitions share a split."""
    keys = ("setting", "workload_type", "spark_version", "vm_type", "num_hosts", "datasize_label")
    return "|".join(str(row[k]) for k in keys)


def assign_split(group: str, fractions: dict[str, float] = SPLIT_FRACTIONS) -> str:
    """Deterministic hash split (no RNG state, stable across rebuilds)."""
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, frac in fractions.items():
        edge += frac
        if h < edge:
            return name
    return list(fractions)[-1]


# =============================================================================
# Per-host labels from one sar trace
# =============================================================================


def host_labels(sar: pd.DataFrame, vcpus: int, elapsed_s: float) -> dict:
    """
    Whole-VM resource labels from one ``sar`` trace.

    CPU busy = 100 − %idle − %iowait − %steal (time the VM's vCPUs executed
    work; waiting on disk and hypervisor steal are not demand).
    Memory used = kbmemused (includes page cache and buffers);
    memory app  = kbmemused − kbbuffers − kbcached.
    """
    busy = (100.0 - sar["cpu.%idle"] - sar["cpu.%iowait"] - sar["cpu.%steal"]).clip(0.0, 100.0)
    used_kb = sar["memory.kbmemused"]
    app_kb = (used_kb - sar["memory.kbbuffers"] - sar["memory.kbcached"]).clip(lower=0)
    ts = pd.to_datetime(sar["timestamp"])
    duration = (ts.iloc[-1] - ts.iloc[0]).total_seconds() + SAMPLE_INTERVAL_S
    p95 = float(busy.quantile(0.95))
    return {
        "host_cpu_busy_mean_pct": float(busy.mean()),
        "host_cpu_busy_p95_pct": p95,
        "host_cpu_busy_p95_cores": p95 / 100.0 * vcpus,
        "host_mem_used_peak_mib": float(used_kb.max()) / 1024.0,
        "host_mem_app_peak_mib": float(app_kb.max()) / 1024.0,
        "n_samples": len(sar),
        "trace_coverage": min(duration / elapsed_s, 1.0) if elapsed_s > 0 else np.nan,
        # Build-report check only (measured RAM vs published spec)
        "_mem_total_measured_mib": float((used_kb + sar["memory.kbmemfree"]).median()) / 1024.0,
    }


def _sar_files(run_dir: str) -> list[str]:
    files = [f for f in os.listdir(run_dir) if f.startswith("sar") and f.endswith(".csv")]
    return sorted(files, key=lambda f: int(re.sub(r"\D", "", f) or 0))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# One Scout record → rows
# =============================================================================


def process_record(run_dir: str) -> dict:
    """
    Convert one Scout record directory.

    Returns ``{"status": "ok", "run": {...}, "hosts": [...], "files": {...}}``
    or ``{"status": <drop reason>}``.
    """
    meta = parse_run_id(os.path.basename(run_dir.rstrip("/")))
    if meta is None:
        return {"status": "unparsed_id"}
    workload_type = TARGET_WORKLOADS.get((meta["scout_workload"], meta["framework"]))
    if workload_type is None:
        return {"status": "not_target_workload"}
    if meta["vm_type"] not in EC2_SPECS:
        return {"status": "unknown_vm_type"}
    if meta["datasize_label"] not in DATASIZE_RANK:
        return {"status": "unknown_datasize"}

    report_path = os.path.join(run_dir, "report.json")
    with open(report_path) as fh:
        report = json.load(fh)
    if report.get("completed") is not True:
        return {"status": "not_completed"}
    if report.get("program") != EXPECTED_PROGRAM[meta["scout_workload"]]:
        return {"status": "program_mismatch"}
    elapsed = float(report.get("elapsed_time", -1))
    if elapsed <= 0:
        return {"status": "no_elapsed_time"}

    vcpus, mem_gib = EC2_SPECS[meta["vm_type"]]
    input_bytes = float(report.get("input_size", -1))
    run = {
        "source": "scout",
        "scout_run_id": meta["scout_run_id"],
        "setting": meta["setting"],
        "workload_type": workload_type,
        "workload_class": WORKLOAD_CLASS[workload_type],
        "spark_version": SPARK_VERSION[meta["framework"]],
        "vm_type": meta["vm_type"],
        "vm_family": meta["vm_type"][0],
        "num_hosts": meta["num_hosts"],
        "vcpus_per_host": vcpus,
        "mem_mib_per_host": mem_gib * 1024.0,
        "total_vcpus": vcpus * meta["num_hosts"],
        "total_mem_mib": mem_gib * 1024.0 * meta["num_hosts"],
        "datasize_label": meta["datasize_label"],
        "datasize_rank": DATASIZE_RANK[meta["datasize_label"]],
        "input_mib": input_bytes / (1024.0 * 1024.0) if input_bytes > 0 else np.nan,
        "input_size_known": input_bytes > 0,
        "elapsed_time_s": elapsed,
    }
    run["config_group"] = config_group(run)
    run["split"] = assign_split(run["config_group"])

    sar_files = _sar_files(run_dir)
    if len(sar_files) != meta["num_hosts"]:
        return {"status": "host_count_mismatch"}
    hosts, files = (
        [],
        {
            os.path.relpath(report_path, os.path.dirname(os.path.dirname(run_dir))): file_sha256(
                report_path
            )
        },
    )
    for idx, fname in enumerate(sar_files, start=1):
        path = os.path.join(run_dir, fname)
        try:
            sar = pd.read_csv(path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return {"status": "unreadable_trace"}
        if len(sar) < MIN_TRACE_SAMPLES:
            return {"status": "trace_too_short"}
        hosts.append({"host_index": idx, **host_labels(sar, vcpus, elapsed)})
        files[os.path.relpath(path, os.path.dirname(os.path.dirname(run_dir)))] = file_sha256(path)

    frame = pd.DataFrame(hosts)
    for col in HOST_LABEL_COLUMNS:
        run[f"{col}_avg_hosts"] = float(frame[col].mean())
        run[f"{col}_max_hosts"] = float(frame[col].max())
    run["n_samples"] = int(frame["n_samples"].min())
    run["trace_coverage"] = float(frame["trace_coverage"].min())
    return {"status": "ok", "run": run, "hosts": hosts, "files": files}


def discover_records(scout_root: str) -> list[str]:
    """All record directories under <scout>/dataset/osr_{single_node,multiple_nodes}."""
    base = os.path.join(scout_root, "dataset")
    if not os.path.isdir(base):
        base = scout_root
    found = []
    for setting in ("osr_single_node", "osr_multiple_nodes"):
        d = os.path.join(base, setting)
        if os.path.isdir(d):
            found.extend(os.path.join(d, r) for r in sorted(os.listdir(d)))
    return [r for r in found if os.path.isfile(os.path.join(r, "report.json"))]


# =============================================================================
# Table assembly
# =============================================================================


def run_columns() -> list[str]:
    labels = ["elapsed_time_s"] + [
        f"{c}_{agg}" for c in HOST_LABEL_COLUMNS for agg in ("avg_hosts", "max_hosts")
    ]
    return TRACKING_COLUMNS + INPUT_COLUMNS + labels + QUALITY_COLUMNS


def host_columns() -> list[str]:
    return (
        TRACKING_COLUMNS
        + ["host_index"]
        + INPUT_COLUMNS
        + ["elapsed_time_s"]
        + HOST_LABEL_COLUMNS
        + QUALITY_COLUMNS
    )


def assemble_tables(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Build (runs, hosts, measured/spec RAM ratios) from `process_record` outputs."""
    runs, hosts, ram_ratio = [], [], []
    for res in results:
        if res["status"] != "ok":
            continue
        run = res["run"]
        runs.append(run)
        for h in res["hosts"]:
            ram_ratio.append(h["_mem_total_measured_mib"] / run["mem_mib_per_host"])
            hosts.append({**run, **{k: v for k, v in h.items() if not k.startswith("_")}})
    runs_df = pd.DataFrame(runs, columns=run_columns())
    hosts_df = pd.DataFrame(hosts, columns=host_columns())
    return runs_df, hosts_df, ram_ratio
