# =============================================================
# tests/unit/test_rank_monitor.py
# Phase 6 — per-rank CPU / memory recorder.
# Runs small real child processes (no MPI, no Spark needed).
# =============================================================
import csv
import json
import os
import sys

import pytest

from mpj_spark.resource_prediction import rank_monitor as rm

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")

# Parent burns ~1 core; a grandchild (like the Spark JVM) holds ~200 MB.
_FAKE_DRIVER = """
import subprocess, sys, time
g = subprocess.Popen([sys.executable, "-c",
    "b = bytearray(200 * 1024 * 1024); import time; time.sleep(2)"])
t = time.time()
while time.time() - t < 2:
    pass
g.wait()
sys.exit(3)
"""


@pytest.fixture
def fake_driver(tmp_path):
    path = tmp_path / "fake_driver.py"
    path.write_text(_FAKE_DRIVER)
    return str(path)


def test_wrapper_measures_process_tree(tmp_path, fake_driver, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    out = tmp_path / "res"
    code = rm.main(
        [
            "--out",
            str(out),
            "--run-id",
            "r1",
            "--interval",
            "0.3",
            "--",
            sys.executable,
            fake_driver,
        ]
    )
    assert code == 3  # child's exit code is passed through

    summary = json.loads((out / "rank1_resources_summary.json").read_text())
    assert summary["rank"] == 1 and summary["role"] == "driver" and summary["run_id"] == "r1"
    # grandchild memory is included in the driver's tree
    assert summary["rss_peak_mib"] > 180
    # ~1 core busy loop (loose bounds: CI machines vary)
    assert 0.5 < summary["cpu_cores_p95"] < 4.0
    assert summary["n_samples"] >= 4
    assert summary["host_cores"] >= 1 and summary["host_mem_mib"] > 0

    with open(out / "rank1_resources.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == rm.SAMPLE_COLUMNS
    assert rows[0]["cpu_cores"] == ""  # no CPU rate on the first sample
    assert max(int(r["n_procs"]) for r in rows) >= 2


def test_rank_defaults_to_zero_coordinator(tmp_path, monkeypatch):
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MPJ_RANK"):
        monkeypatch.delenv(var, raising=False)
    out = tmp_path / "res"
    assert rm.main(["--out", str(out), "--", sys.executable, "-c", "pass"]) == 0
    summary = json.loads((out / "rank0_resources_summary.json").read_text())
    assert summary["rank"] == 0 and summary["role"] == "coordinator"


def test_requires_separator():
    with pytest.raises(SystemExit):
        rm.main(["--out", "x", sys.executable])


def test_process_tree_contains_self():
    tree = rm.process_tree(os.getpid())
    assert os.getpid() in tree and tree[os.getpid()] >= 0


def test_summarise_ignores_missing_values():
    rows = [
        {c: None for c in rm.SAMPLE_COLUMNS} | {"rss_mib": 10.0, "host_swap_used_mib": 0.0},
        {c: None for c in rm.SAMPLE_COLUMNS}
        | {"cpu_cores": 1.0, "rss_mib": 30.0, "host_swap_used_mib": 0.0},
        {c: None for c in rm.SAMPLE_COLUMNS}
        | {"cpu_cores": 3.0, "rss_mib": 20.0, "host_swap_used_mib": 0.0},
    ]
    s = rm.summarise(rows)
    assert s["cpu_cores_mean"] == pytest.approx(2.0)
    assert s["cpu_cores_max"] == 3.0
    assert s["rss_peak_mib"] == 30.0
    assert s["cg_mem_peak_mib"] is None and s["pss_peak_mib"] is None


def test_percentile():
    assert rm._percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert rm._percentile([None, 2.0], 0.5) == 2.0
    assert rm._percentile([], 0.95) is None
