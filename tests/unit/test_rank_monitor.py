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


# ── cgroup readers (fake /sys/fs/cgroup trees) ────────────────

_GIB = 1024**3


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v1_docker_layout(tmp_path):
    """Ubuntu 22.04 Docker with cgroupfs driver (partner laptop)."""
    _write(tmp_path, "memory/memory.usage_in_bytes", str(3 * _GIB))
    _write(tmp_path, "memory/memory.max_usage_in_bytes", str(4 * _GIB))
    _write(tmp_path, "memory/memory.stat", f"cache {_GIB}\nrss 123\ntotal_rss {2 * _GIB}\n")
    _write(tmp_path, "memory/memory.limit_in_bytes", str(6 * _GIB))
    _write(tmp_path, "memory/memory.oom_control", "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n")
    _write(tmp_path, "cpuacct/cpuacct.usage", "5000000000")  # 5 s in ns
    _write(tmp_path, "cpu/cpu.cfs_quota_us", "200000")
    _write(tmp_path, "cpu/cpu.cfs_period_us", "100000")
    cg = rm.Cgroup(str(tmp_path))
    assert cg.version == 1 and cg.available
    assert cg.mem_bytes() == 3 * _GIB
    assert cg.anon_bytes() == 2 * _GIB  # total_rss preferred over rss
    assert cg.cpu_usec() == 5_000_000
    assert cg.peak_bytes() == 4 * _GIB
    assert cg.oom_kills() == 1
    assert cg.limits() == {"cgroup_cpu_limit_cores": 2.0, "cgroup_mem_limit_mib": 6 * 1024.0}


def test_cgroup_v1_unlimited_and_combined_cpu_dir(tmp_path):
    _write(tmp_path, "memory/memory.usage_in_bytes", "100")
    _write(tmp_path, "memory/memory.limit_in_bytes", "9223372036854771712")
    _write(tmp_path, "cpu,cpuacct/cpuacct.usage", "1000")
    _write(tmp_path, "cpu,cpuacct/cpu.cfs_quota_us", "-1")
    _write(tmp_path, "cpu,cpuacct/cpu.cfs_period_us", "100000")
    cg = rm.Cgroup(str(tmp_path))
    assert cg.cpu_usec() == 1
    assert cg.limits() == {"cgroup_cpu_limit_cores": None, "cgroup_mem_limit_mib": None}


def test_cgroup_v2_layout(tmp_path):
    _write(tmp_path, "memory.current", str(_GIB))
    _write(tmp_path, "memory.peak", str(2 * _GIB))
    _write(tmp_path, "memory.stat", f"anon {_GIB // 2}\nfile 10\n")
    _write(tmp_path, "memory.events", "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    _write(tmp_path, "cpu.stat", "usage_usec 2500000\nuser_usec 2000000\n")
    _write(tmp_path, "cpu.max", "150000 100000")
    _write(tmp_path, "memory.max", "max")
    cg = rm.Cgroup(str(tmp_path))
    assert cg.version == 2
    assert cg.anon_bytes() == _GIB // 2 and cg.cpu_usec() == 2_500_000
    assert cg.peak_bytes() == 2 * _GIB and cg.oom_kills() == 0
    assert cg.limits() == {"cgroup_cpu_limit_cores": 1.5, "cgroup_mem_limit_mib": None}


def test_no_cgroup(tmp_path):
    cg = rm.Cgroup(str(tmp_path))
    assert cg.version is None and not cg.available


def test_sampler_reads_cgroup(tmp_path):
    _write(tmp_path, "memory/memory.usage_in_bytes", str(_GIB))
    _write(tmp_path, "memory/memory.stat", f"total_rss {_GIB // 4}\n")
    _write(tmp_path, "cpuacct/cpuacct.usage", "0")
    sampler = rm.RankSampler(os.getpid(), cgroup=rm.Cgroup(str(tmp_path)))
    sampler.sample()
    _write(tmp_path, "cpuacct/cpuacct.usage", "1000000000")  # +1 s of CPU
    row = sampler.sample()
    assert row["cg_mem_mib"] == 1024.0 and row["cg_anon_mib"] == 256.0
    assert row["cg_cpu_cores"] > 0
