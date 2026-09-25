# =============================================================
# tests/unit/test_scout_build.py
# Phase 6 — Scout source-domain tables (P6-01).
# Synthetic Scout records only; the real 3 GB clone is not needed.
# =============================================================
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from mpj_spark.resource_prediction import scout

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")

_SAR_COLUMNS = [
    "timestamp",
    "cpu.%usr",
    "cpu.%nice",
    "cpu.%sys",
    "cpu.%iowait",
    "cpu.%steal",
    "cpu.%irq",
    "cpu.%soft",
    "cpu.%guest",
    "cpu.%gnice",
    "cpu.%idle",
    "memory.kbmemfree",
    "memory.kbmemused",
    "memory.kbbuffers",
    "memory.kbcached",
]
_MIB = 1024  # kB per MiB


def _sar(n, busy=50.0, iowait=10.0, steal=0.0, used_mib=4000, cached_mib=1000, total_mib=8192):
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": pd.Timestamp("2018-01-01 00:00:00") + pd.Timedelta(seconds=5 * i),
                "cpu.%usr": busy,
                "cpu.%nice": 0.0,
                "cpu.%sys": 0.0,
                "cpu.%iowait": iowait,
                "cpu.%steal": steal,
                "cpu.%irq": 0.0,
                "cpu.%soft": 0.0,
                "cpu.%guest": 0.0,
                "cpu.%gnice": 0.0,
                "cpu.%idle": 100.0 - busy - iowait - steal,
                "memory.kbmemfree": (total_mib - used_mib) * _MIB,
                "memory.kbmemused": used_mib * _MIB,
                "memory.kbbuffers": 0,
                "memory.kbcached": cached_mib * _MIB,
            }
        )
    return pd.DataFrame(rows, columns=_SAR_COLUMNS)


def _record(root, name, program, completed=True, input_size=-1, hosts=1, n=20, busy=50.0, **sar):
    setting = "osr_multiple_nodes" if name[0].isdigit() else "osr_single_node"
    d = root / "dataset" / setting / name
    d.mkdir(parents=True)
    report = {"completed": completed, "datasize": "x", "framework": "x", "workload": "x"}
    if completed:
        report.update({"elapsed_time": n * 5.0, "input_size": input_size, "program": program})
    (d / "report.json").write_text(json.dumps(report))
    if setting == "osr_single_node":
        _sar(n, busy=busy, **sar).to_csv(d / "sar.csv", index=False)
    else:
        for k in range(1, hosts + 1):
            # host k gets a different busy level so avg != max
            _sar(n, busy=busy + 10 * (k - 1), **sar).to_csv(d / f"sar_node{k}.csv", index=False)
    return str(d)


# ── Configuration sanity ──────────────────────────────────────


def test_only_spark_target_workloads_configured():
    assert set(scout.TARGET_WORKLOADS.values()) == {"wordcount", "kmeans", "logreg"}
    assert all(fw != "hadoop" for _, fw in scout.TARGET_WORKLOADS)
    assert set(scout.WORKLOAD_CLASS) == set(scout.TARGET_WORKLOADS.values())


def test_datasize_rank_is_strictly_increasing():
    order = ["small", "medium", "large", "huge", "bigdata"]
    assert [scout.DATASIZE_RANK[s] for s in order] == [1, 2, 3, 4, 5]


# ── parse_run_id ──────────────────────────────────────────────


def test_parse_single_node():
    m = scout.parse_run_id("r4.xlarge_i-0f4e4b248a6aa957a_wordcount_spark_small_1")
    assert m["setting"] == "single_node" and m["num_hosts"] == 1
    assert m["vm_type"] == "r4.xlarge" and m["scout_workload"] == "wordcount"
    assert m["framework"] == "spark" and m["datasize_label"] == "small"


def test_parse_multi_node():
    m = scout.parse_run_id("12_m4.2xlarge_kmeans_spark1.5_bigdata_1")
    assert m["setting"] == "multi_node" and m["num_hosts"] == 12
    assert m["vm_type"] == "m4.2xlarge" and m["framework"] == "spark1.5"


def test_parse_unknown_returns_none():
    assert scout.parse_run_id("README") is None


# ── Filtering ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,program,completed,status",
    [
        ("4_c4.large_wordcount_hadoop_huge_1", "HadoopWordcount", True, "not_target_workload"),
        (
            "c4.large_i-0a_classification_spark1.5_large_1",
            "classification",
            True,
            "not_target_workload",
        ),
        ("c4.large_i-0a_regression_spark1.5_large_1", "regression", True, "not_target_workload"),
        ("c4.large_i-0a_lr_spark_large_1", "LogisticRegression", False, "not_completed"),
        ("c4.large_i-0a_lr_spark_large_1", "LinearRegression", True, "program_mismatch"),
        ("c9.large_i-0a_lr_spark_large_1", "LogisticRegression", True, "unknown_vm_type"),
    ],
)
def test_records_are_filtered(tmp_path, name, program, completed, status):
    d = _record(tmp_path, name, program, completed=completed, hosts=4)
    assert scout.process_record(d)["status"] == status


def test_short_trace_dropped(tmp_path):
    d = _record(tmp_path, "c4.large_i-0a_kmeans_spark1.5_small_1", "kmeans", n=2)
    assert scout.process_record(d)["status"] == "trace_too_short"


# ── Labels and inputs ─────────────────────────────────────────


def test_host_labels_units_and_scope():
    sar = _sar(10, busy=50.0, iowait=20.0, steal=5.0, used_mib=4000, cached_mib=1000)
    lab = scout.host_labels(sar, vcpus=4, elapsed_s=50.0)
    # iowait and steal are not counted as busy
    assert lab["host_cpu_busy_mean_pct"] == pytest.approx(50.0)
    assert lab["host_cpu_busy_p95_cores"] == pytest.approx(2.0)
    assert lab["host_mem_used_peak_mib"] == pytest.approx(4000)
    assert lab["host_mem_app_peak_mib"] == pytest.approx(3000)
    assert lab["trace_coverage"] == pytest.approx(1.0)


def test_logreg_record_inputs(tmp_path):
    d = _record(
        tmp_path, "c4.xlarge_i-0a_lr_spark_medium_2", "LogisticRegression", input_size=1209030600
    )
    res = scout.process_record(d)
    run = res["run"]
    assert res["status"] == "ok"
    assert run["workload_type"] == "logreg" and run["workload_class"] == "iterative"
    assert run["spark_version"] == "2.1" and run["vm_family"] == "c"
    assert run["vcpus_per_host"] == 4 and run["mem_mib_per_host"] == pytest.approx(7.5 * 1024)
    assert run["datasize_rank"] == 2
    assert run["input_size_known"] and run["input_mib"] == pytest.approx(1209030600 / 2**20)


def test_kmeans_has_no_input_bytes(tmp_path):
    d = _record(tmp_path, "m4.large_i-0a_kmeans_spark1.5_large_1", "kmeans")
    run = scout.process_record(d)["run"]
    assert not run["input_size_known"] and np.isnan(run["input_mib"])
    assert run["datasize_rank"] == 3


def test_multi_host_labels_are_not_divided(tmp_path):
    d = _record(tmp_path, "3_r4.large_kmeans_spark1.5_huge_1", "kmeans", hosts=3, busy=40.0)
    res = scout.process_record(d)
    assert len(res["hosts"]) == 3
    run = res["run"]
    assert run["num_hosts"] == 3 and run["total_vcpus"] == 6
    # hosts at 40/50/60 % busy → avg 50, max 60, never divided by 3
    assert run["host_cpu_busy_mean_pct_avg_hosts"] == pytest.approx(50.0)
    assert run["host_cpu_busy_mean_pct_max_hosts"] == pytest.approx(60.0)


def test_host_count_mismatch_dropped(tmp_path):
    d = _record(tmp_path, "4_r4.large_kmeans_spark1.5_huge_1", "kmeans", hosts=3)
    assert scout.process_record(d)["status"] == "host_count_mismatch"


# ── Splits ────────────────────────────────────────────────────


def test_repetitions_share_config_group_and_split(tmp_path):
    runs = [
        scout.process_record(
            _record(tmp_path, f"c4.large_i-0a_kmeans_spark1.5_small_{r}", "kmeans")
        )["run"]
        for r in (1, 2, 3)
    ]
    assert len({r["config_group"] for r in runs}) == 1
    assert len({r["split"] for r in runs}) == 1


def test_split_is_deterministic_and_balanced():
    splits = [scout.assign_split(f"g{i}") for i in range(5000)]
    assert splits == [scout.assign_split(f"g{i}") for i in range(5000)]
    assert 0.65 < splits.count("train") / 5000 < 0.75


# ── Tables and end-to-end ─────────────────────────────────────


def test_tables_have_contract_columns_only(tmp_path):
    results = [
        scout.process_record(
            _record(tmp_path, "2_c4.large_lr_spark_huge_1", "LogisticRegression", hosts=2)
        )
    ]
    runs, hosts, _ = scout.assemble_tables(results)
    assert list(runs.columns) == scout.run_columns()
    assert list(hosts.columns) == scout.host_columns()
    assert len(runs) == 1 and len(hosts) == 2
    assert not any(c.startswith("_") for c in hosts.columns)


def test_build_script_end_to_end(tmp_path):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import build_scout_dataset

    _record(
        tmp_path,
        "c4.large_i-0a_wordcount_spark_small_1",
        "ScalaSparkWordcount",
        input_size=3 * 2**30,
    )
    _record(tmp_path, "c4.large_i-0a_kmeans_spark1.5_small_1", "kmeans")
    _record(tmp_path, "2_m4.large_lr_spark_huge_1", "LogisticRegression", hosts=2, input_size=2**34)
    _record(tmp_path, "4_m4.large_wordcount_hadoop_huge_1", "HadoopWordcount", hosts=4)
    out = tmp_path / "out"
    build_scout_dataset.main(
        ["--scout-dir", str(tmp_path), "--out-dir", str(out), "--workers", "1"]
    )

    runs = pd.read_csv(out / "scout_runs.csv")
    hosts = pd.read_csv(out / "scout_hosts.csv")
    report = json.loads((out / "build_report.json").read_text())
    prov = json.loads((out / "provenance.json").read_text())
    assert sorted(runs["workload_type"]) == ["kmeans", "logreg", "wordcount"]
    assert len(hosts) == 4  # 1 + 1 + 2
    assert report["record_status"]["not_target_workload"] == 1
    assert report["config_groups_spanning_splits"] == 0
    assert prov["files_used"] == 3 + 4  # 3 report.json + 4 sar files
