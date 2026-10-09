# =============================================================
# tests/unit/test_local_data.py
# Phase 6 — local per-driver resource tables.
# Small synthetic CSVs in both raw layouts; no real results needed.
# =============================================================
import json
import os
import sys

import pandas as pd
import pytest

from mpj_spark.resource_prediction import local_data as ld

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")

_OLD_COLS = (
    "run_id,timestamp,host,app,size,np,workers,rep,driver_idx,pid,wall_s,exit_code,n_samples,"
    "cpu_avg_pct,cpu_peak_pct,rss_avg_mb,rss_peak_mb,sys_cpu_avg_pct,sys_mem_avg_mb"
)
_NEW_COLS = (
    "run_id,timestamp,host,app,size,np,workers,rep,driver_idx,role,pid,wall_s,exit_code,"
    "n_samples,cpu_avg_pct,cpu_peak_pct,rss_avg_mb,rss_peak_mb,sys_cpu_avg_pct,sys_mem_avg_mb,"
    "cores_req,heap_req_mb,cores_alloc,heap_mb_alloc,cpu_p95_pct,rss_p95_mb,pss_avg_mb,"
    "pss_peak_mb,sys_mem_peak_mb,sys_mem_avail_min_mb"
)


def _old_run(app, size, rep, exit_code=0, cpu=200.0):
    rid = f"{app}_{size}_np3_r{rep}"
    ts = f"2026-10-07T10:0{rep}:00"
    rows = [
        f"{rid},{ts},h,{app},{size},3,2,{rep},0,1,50,{exit_code},10,99,101,42,44,20,3000",
        f"{rid},{ts},h,{app},{size},3,2,{rep},1,2,50,{exit_code},10,{cpu},{cpu + 50},900,1800,20,3000",
        f"{rid},{ts},h,{app},{size},3,2,{rep},2,3,50,{exit_code},10,{cpu},{cpu + 60},950,1900,20,3000",
    ]
    return rows


def _new_run(app, size, cores, heap, rep, cpu_p95=250.0):
    rid = f"{app}_{size}_np3_c{cores}_h{heap}_r{rep}"
    ts = f"2026-10-08T10:{cores:02d}:{rep:02d}"
    common = f"{cores},{heap},{cores},{heap}"
    return [
        f"{rid},{ts},h,{app},{size},3,2,{rep},0,root,1,60,0,10,100,101,42,44,20,3000,"
        f"{common},101,42,,,4000,9000",
        f"{rid},{ts},h,{app},{size},3,2,{rep},1,driver,2,60,0,10,200,400,1000,1100,20,3000,"
        f"{common},{cpu_p95},1090,,,4000,9000",
        f"{rid},{ts},h,{app},{size},3,2,{rep},2,driver,3,60,0,10,180,380,1050,1200,20,3000,"
        f"{common},{cpu_p95 - 10},1190,,,4000,9000",
    ]


def _write(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    old = []
    for rep in (1, 2):
        old += _old_run("wordcount", 50, rep)
        old += _old_run("kmeans", 200, rep, exit_code=124)
    _write(d / "resource_runs.csv", _OLD_COLS, old)
    _write(d / "run1_no_cpu.csv", _OLD_COLS, _old_run("wordcount", 50, 1, cpu=0.0))
    grid = []
    for cores in (1, 2, 4, 8):
        for heap in (1536, 3072):
            for rep in (1, 2):
                grid += _new_run("kmeans", 100, cores, heap, rep)
    _write(d / "grid_kmeans100.csv", _NEW_COLS, grid)
    _write(d / "blas_test.csv", _NEW_COLS, _new_run("logreg", 50, 2, 3072, 1, cpu_p95=550))
    return d


def test_excluded_and_root_rows(raw_dir):
    drivers, runs, report = ld.build_tables(str(raw_dir))
    assert "run1_no_cpu.csv" in report["excluded_files"]
    assert set(drivers["file"]) == {"resource_runs.csv", "grid_kmeans100.csv", "blas_test.csv"}
    # only rank >= 1 drivers: 2 per run
    assert (drivers.groupby("run_key").size() == 2).all()
    assert report["root_rows_dropped"] == len(runs)


def test_units_and_labels(raw_dir):
    drivers, _, _ = ld.build_tables(str(raw_dir))
    g = drivers[drivers["experiment"] == "grid"].iloc[0]
    assert g["cpu_cores_mean"] == pytest.approx(2.0)  # 200 % = 2 cores
    assert g["cpu_cores_p95"] == pytest.approx(2.5)
    assert g["mem_mb_peak"] == 1100 and g["alloc_known"]
    old = drivers[drivers["experiment"] == "default_alloc"].iloc[0]
    assert pd.isna(old["cpu_cores_p95"]) and pd.isna(old["cores_per_driver"])
    assert not old["alloc_known"]


def test_timeouts_are_censored(raw_dir):
    _, runs, _ = ld.build_tables(str(raw_dir))
    km200 = runs[(runs["workload_type"] == "kmeans") & (runs["dataset_mb"] == 200)]
    assert set(km200["status"]) == {"timeout"}
    assert set(runs.loc[runs["dataset_mb"] == 50, "status"]) == {"ok"}


def test_blas_test_kept_out_of_main_set(raw_dir):
    _, runs, _ = ld.build_tables(str(raw_dir))
    blas = runs[runs["experiment"] == "blas_test"]
    assert len(blas) == 1 and not blas["in_main_set"].any()
    assert runs.loc[runs["experiment"] != "blas_test", "in_main_set"].all()
    assert blas["config_group"].iloc[0].startswith("blas_test|")


def test_repeats_share_group_and_split(raw_dir):
    _, runs, _ = ld.build_tables(str(raw_dir))
    main = runs[runs["in_main_set"]]  # the blas_test fixture has a single repeat
    per_group = main.groupby("config_group").agg(
        reps=("rep", "nunique"), splits=("split", "nunique")
    )
    assert (per_group["splits"] == 1).all()
    assert (per_group["reps"] == 2).all()


def test_stratified_split_gives_each_workload_a_test_group():
    groups = pd.DataFrame(
        {
            "config_group": [f"kmeans|{i}" for i in range(10)] + [f"wc|{i}" for i in range(3)],
            "stratum": ["kmeans"] * 10 + ["wc"] * 3,
        }
    )
    split = ld.stratified_split(groups)
    for prefix in ("kmeans|", "wc|"):
        vals = [s for k, s in split.items() if k.startswith(prefix)]
        assert "test" in vals and "val" in vals and "train" in vals
    assert split == ld.stratified_split(groups.iloc[::-1])  # order-independent


def test_run_table_aggregates(raw_dir):
    _, runs, _ = ld.build_tables(str(raw_dir))
    r = runs[runs["experiment"] == "grid"].iloc[0]
    assert r["drivers_recorded"] == 2
    assert r["mem_mb_peak_max_drivers"] == 1200
    assert r["mem_mb_peak_avg_drivers"] == pytest.approx(1150)


def test_build_script_end_to_end(raw_dir, tmp_path):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import build_local_dataset

    out = tmp_path / "out"
    build_local_dataset.main(["--raw-dir", str(raw_dir), "--out-dir", str(out)])
    report = json.loads((out / "build_report.json").read_text())
    assert report["runs_by_status"] == {"ok": 19, "timeout": 2}
    assert report["config_groups_spanning_splits"] == 0
    drivers = pd.read_csv(out / "local_drivers.csv")
    assert {"cpu_cores_p95", "mem_mb_peak", "cores_per_driver"} <= set(drivers.columns)
    assert not any(c.startswith(("sys_", "pss_")) for c in drivers.columns)
