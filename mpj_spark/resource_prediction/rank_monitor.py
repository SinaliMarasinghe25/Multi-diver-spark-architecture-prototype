# =============================================================================
# mpj_spark/resource_prediction/rank_monitor.py
# Phase 6 — per-rank CPU / memory recorder (P6-01 local target data)
#
# PURPOSE
# -------
# Measure what ONE MPI rank (one Spark driver: the Python worker process plus
# the Spark JVM and any PySpark daemons it starts) actually uses, so the
# Phase 6 predictor has per-driver CPU and peak-memory labels.
#
# It wraps the rank's command, exactly like the P3-12 taskset throttle
# wrapper, so no ML / synchronisation code changes:
#
#   mpirun --oversubscribe -np 3 \
#     python -m mpj_spark.resource_prediction.rank_monitor \
#       --out results/resources/<run_id> --run-id <run_id> -- \
#     python -m mpj_spark.core.main_mpi --app kmeans --generate 200 ...
#
# Each rank writes:
#   rank<R>_resources.csv           one row per sample (default every 1 s)
#   rank<R>_resources_summary.json  peak / mean / p95 labels + host context
#
# WHAT IS MEASURED (Linux only, standard library only)
# ----------------------------------------------------
# Process tree of the wrapped command, re-discovered every sample:
#   cpu_cores    Δ(utime+stime of all tree processes) / Δwall  — core-equivalents
#   rss_mib      Σ VmRSS of the tree (may double-count shared libraries)
#   pss_mib      Σ Pss from smaps_rollup (shared pages split fairly; preferred)
# Container (cgroup v1 or v2), when running inside Docker:
#   cg_mem_mib   container memory incl. page cache
#   cg_anon_mib  anonymous memory (excludes page cache) — closest to demand
#   cg_cpu_cores Δ container CPU usage / Δwall
#   kernel peak  memory.max_usage_in_bytes (v1) / memory.peak (v2)
#   limits       the CPU quota and memory limit actually applied
#   OOM kills    container processes killed for exceeding the memory limit
# Host context: cores, RAM, MemAvailable, load average and swap at start,
# swap during the run (swapping invalidates timing comparisons).
#
# LIMITS OF FIDELITY (document in the write-up)
# ---------------------------------------------
# * CPU time of a process that exits between two samples is lost for that
#   interval; short-lived helpers are therefore slightly under-counted.
# * RSS counts shared pages in every process; PSS corrects this but needs a
#   kernel >= 4.14 (Ubuntu 22.04 is fine).
# * The process tree includes the rank's Python interpreter; on rank 0 it is
#   the coordinator, not a Spark driver.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_CLK_TCK = os.sysconf("SC_CLK_TCK")
_CGROUP = "/sys/fs/cgroup"
_MIB = 1024.0 * 1024.0

SAMPLE_COLUMNS = [
    "t_s",
    "n_procs",
    "cpu_cores",
    "rss_mib",
    "pss_mib",
    "cg_mem_mib",
    "cg_anon_mib",
    "cg_cpu_cores",
    "host_swap_used_mib",
]


# =============================================================================
# /proc readers
# =============================================================================


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _stat_fields(pid: int) -> tuple[int, int] | None:
    """(ppid, utime+stime in ticks) from /proc/<pid>/stat, or None if gone."""
    raw = _read(f"/proc/{pid}/stat")
    if raw is None:
        return None
    # comm may contain spaces/parentheses: split after the last ')'
    rest = raw[raw.rfind(")") + 2 :].split()
    return int(rest[1]), int(rest[11]) + int(rest[12])


def process_tree(root_pid: int) -> dict[int, int]:
    """{pid: cpu_ticks} for root_pid and all its descendants."""
    stats: dict[int, tuple[int, int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            st = _stat_fields(int(entry))
            if st is not None:
                stats[int(entry)] = st
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in stats.items():
        children.setdefault(ppid, []).append(pid)
    tree, stack = {}, [root_pid]
    while stack:
        pid = stack.pop()
        if pid in stats:
            tree[pid] = stats[pid][1]
            stack.extend(children.get(pid, []))
    return tree


def _rss_bytes(pid: int) -> int:
    raw = _read(f"/proc/{pid}/statm")
    return int(raw.split()[1]) * _PAGE_SIZE if raw else 0


def _pss_bytes(pid: int) -> int | None:
    raw = _read(f"/proc/{pid}/smaps_rollup")
    if raw is None:
        return None
    for line in raw.splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1]) * 1024
    return None


def _meminfo() -> dict[str, int]:
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, val = line.partition(":")
        parts = val.split()
        if parts:
            out[key] = int(parts[0]) * 1024
    return out


# =============================================================================
# cgroup readers (Docker) — v1 (cgroupfs, e.g. Ubuntu 22.04 Docker) and v2
# =============================================================================

_V1_UNLIMITED = 1 << 62  # v1 reports "no limit" as a huge number


class Cgroup:
    """
    Read the container's own cgroup counters.  Inside a Docker container the
    container's cgroup is mounted at /sys/fs/cgroup, so these are per-container
    (= per-driver, one rank per container) values.

    version 2: memory.current, memory.stat (anon), cpu.stat (usage_usec),
               cpu.max, memory.max, memory.peak, memory.events (oom_kill)
    version 1: memory/memory.usage_in_bytes, memory/memory.stat (total_rss),
               cpuacct/cpuacct.usage (ns), cpu/cpu.cfs_quota_us + cfs_period_us,
               memory/memory.limit_in_bytes, memory/memory.max_usage_in_bytes,
               memory/memory.oom_control (oom_kill)
    """

    def __init__(self, root: str = _CGROUP):
        self.root = root
        if os.path.exists(os.path.join(root, "memory.current")):
            self.version = 2
        elif os.path.exists(os.path.join(root, "memory", "memory.usage_in_bytes")):
            self.version = 1
        else:
            self.version = None

    @property
    def available(self) -> bool:
        return self.version is not None

    def _get(self, *candidates: str) -> str | None:
        for rel in candidates:
            raw = _read(os.path.join(self.root, rel))
            if raw is not None:
                return raw.strip()
        return None

    def _stat_value(self, rel: str, key: str) -> int | None:
        for line in (self._get(rel) or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == key:
                return int(parts[1])
        return None

    def mem_bytes(self) -> int | None:
        """Current memory incl. page cache."""
        raw = (
            self._get("memory.current")
            if self.version == 2
            else self._get("memory/memory.usage_in_bytes")
        )
        return int(raw) if raw else None

    def anon_bytes(self) -> int | None:
        """Anonymous (non-cache) memory — closest to the driver's real demand."""
        if self.version == 2:
            return self._stat_value("memory.stat", "anon")
        v = self._stat_value("memory/memory.stat", "total_rss")
        return v if v is not None else self._stat_value("memory/memory.stat", "rss")

    def cpu_usec(self) -> int | None:
        if self.version == 2:
            return self._stat_value("cpu.stat", "usage_usec")
        raw = self._get("cpuacct/cpuacct.usage", "cpu,cpuacct/cpuacct.usage")
        return int(raw) // 1000 if raw else None  # ns → µs

    def peak_bytes(self) -> int | None:
        """Kernel-recorded peak (catches spikes between samples), if exposed."""
        raw = (
            self._get("memory.peak")
            if self.version == 2
            else self._get("memory/memory.max_usage_in_bytes")
        )
        return int(raw) if raw and raw.isdigit() else None

    def oom_kills(self) -> int | None:
        if self.version == 2:
            return self._stat_value("memory.events", "oom_kill")
        return self._stat_value("memory/memory.oom_control", "oom_kill")

    def limits(self) -> dict:
        """Applied caps (None = unlimited)."""
        cpu_limit = mem_limit = None
        if self.version == 2:
            cpu_max = self._get("cpu.max")
            if cpu_max and not cpu_max.startswith("max"):
                quota, period = cpu_max.split()
                cpu_limit = int(quota) / int(period)
            mem_max = self._get("memory.max")
            if mem_max and mem_max != "max":
                mem_limit = int(mem_max) / _MIB
        elif self.version == 1:
            quota = self._get("cpu/cpu.cfs_quota_us", "cpu,cpuacct/cpu.cfs_quota_us")
            period = self._get("cpu/cpu.cfs_period_us", "cpu,cpuacct/cpu.cfs_period_us")
            if quota and period and int(quota) > 0:
                cpu_limit = int(quota) / int(period)
            mem_max = self._get("memory/memory.limit_in_bytes")
            if mem_max and int(mem_max) < _V1_UNLIMITED:
                mem_limit = int(mem_max) / _MIB
        return {"cgroup_cpu_limit_cores": cpu_limit, "cgroup_mem_limit_mib": mem_limit}


# =============================================================================
# Sampler
# =============================================================================


class RankSampler:
    """Samples the process tree rooted at `root_pid` (and the cgroup, if any)."""

    def __init__(self, root_pid: int, cgroup: Cgroup | None = None):
        self.root_pid = root_pid
        self.cgroup = Cgroup() if cgroup is None else cgroup
        self.use_cgroup = self.cgroup.available
        self.t0 = time.monotonic()
        self._last_t = self.t0
        self._last_ticks: dict[int, int] = {}
        self._last_cg_usec = self.cgroup.cpu_usec() if self.use_cgroup else None
        self.rows: list[dict] = []

    def sample(self) -> dict:
        now = time.monotonic()
        dt = max(now - self._last_t, 1e-6)
        tree = process_tree(self.root_pid)
        # Processes new since the last sample contribute their whole CPU time
        # (they started inside this interval for all practical purposes).
        dticks = sum(max(t - self._last_ticks.get(pid, 0), 0) for pid, t in tree.items())
        pss = [_pss_bytes(p) for p in tree]
        mem = _meminfo()
        first = not self.rows  # no previous reading → no CPU rate yet
        row = {
            "t_s": round(now - self.t0, 3),
            "n_procs": len(tree),
            "cpu_cores": None if first else dticks / _CLK_TCK / dt,
            "rss_mib": sum(_rss_bytes(p) for p in tree) / _MIB,
            "pss_mib": (sum(p for p in pss if p) / _MIB)
            if any(p is not None for p in pss)
            else None,
            "cg_mem_mib": None,
            "cg_anon_mib": None,
            "cg_cpu_cores": None,
            "host_swap_used_mib": (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / _MIB,
        }
        if self.use_cgroup:
            cur = self.cgroup.mem_bytes()
            anon = self.cgroup.anon_bytes()
            usec = self.cgroup.cpu_usec()
            row["cg_mem_mib"] = cur / _MIB if cur is not None else None
            row["cg_anon_mib"] = anon / _MIB if anon is not None else None
            if not first and usec is not None and self._last_cg_usec is not None:
                row["cg_cpu_cores"] = (usec - self._last_cg_usec) / 1e6 / dt
            self._last_cg_usec = usec
        self._last_t, self._last_ticks = now, tree
        self.rows.append(row)
        return row


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(int(round(q * (len(vals) - 1))), len(vals) - 1)
    return vals[idx]


def summarise(rows: list[dict]) -> dict:
    """Run-level labels from the sample rows (the first sample has no CPU rate)."""
    cpu = [r["cpu_cores"] for r in rows if r["cpu_cores"] is not None]

    def peak(col):
        vals = [r[col] for r in rows if r[col] is not None]
        return max(vals) if vals else None

    cg_cpu = [r["cg_cpu_cores"] for r in rows if r["cg_cpu_cores"] is not None]
    return {
        "n_samples": len(rows),
        "cpu_cores_mean": sum(cpu) / len(cpu) if cpu else None,
        "cpu_cores_p95": _percentile(cpu, 0.95),
        "cpu_cores_max": max(cpu) if cpu else None,
        "rss_peak_mib": peak("rss_mib"),
        "pss_peak_mib": peak("pss_mib"),
        "cg_mem_peak_mib": peak("cg_mem_mib"),
        "cg_anon_peak_mib": peak("cg_anon_mib"),
        "cg_cpu_cores_p95": _percentile(cg_cpu, 0.95),
        "host_swap_used_mib_max": peak("host_swap_used_mib"),
    }


def host_context() -> dict:
    mem = _meminfo()
    load1 = (_read("/proc/loadavg") or "0").split()[0]
    return {
        "host_cores": os.cpu_count(),
        "host_mem_mib": mem.get("MemTotal", 0) / _MIB,
        "host_mem_available_mib_start": mem.get("MemAvailable", 0) / _MIB,
        "host_swap_used_mib_start": (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / _MIB,
        "host_load1_start": float(load1),
    }


def mpi_rank() -> int:
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MPJ_RANK"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    return 0


# =============================================================================
# Wrapper entry point
# =============================================================================


def run_monitored(command: list[str], out_dir: str, run_id: str, interval: float) -> int:
    """Run `command`, sample it until it exits, write CSV + summary, return its exit code."""
    rank = mpi_rank()
    os.makedirs(out_dir, exist_ok=True)
    context = host_context()
    started = time.time()
    proc = subprocess.Popen(command)

    def _forward(signum, _frame):
        if proc.poll() is None:
            proc.send_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward)

    sampler = RankSampler(proc.pid)
    oom_before = sampler.cgroup.oom_kills() if sampler.use_cgroup else None
    sampler.sample()
    while proc.poll() is None:
        time.sleep(interval)
        if proc.poll() is None:
            sampler.sample()
    exit_code = proc.returncode

    csv_path = os.path.join(out_dir, f"rank{rank}_resources.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(sampler.rows)

    summary = {
        "run_id": run_id,
        "rank": rank,
        "role": "coordinator" if rank == 0 else "driver",
        "command": command,
        "exit_code": exit_code,
        "killed_by_signal": -exit_code if exit_code < 0 else None,
        "start_unix": started,
        "duration_s": time.time() - started,
        "interval_s": interval,
        "cgroup_version": sampler.cgroup.version,
        **(sampler.cgroup.limits() if sampler.use_cgroup else {}),
        "cg_mem_kernel_peak_mib": (
            peak / _MIB if sampler.use_cgroup and (peak := sampler.cgroup.peak_bytes()) else None
        ),
        "cg_oom_kills": (
            (sampler.cgroup.oom_kills() or 0) - (oom_before or 0)
            if oom_before is not None
            else None
        ),
        **context,
        **summarise(sampler.rows),
    }
    with open(os.path.join(out_dir, f"rank{rank}_resources_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        sys.exit("usage: rank_monitor [--out DIR] [--run-id ID] [--interval S] -- <command ...>")
    split = argv.index("--")
    p = argparse.ArgumentParser(prog="python -m mpj_spark.resource_prediction.rank_monitor")
    p.add_argument("--out", default="results/resources")
    p.add_argument("--run-id", default="run")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args(argv[:split])
    command = argv[split + 1 :]
    if not command:
        sys.exit("rank_monitor: no command given after --")
    return run_monitored(command, args.out, args.run_id, args.interval)


if __name__ == "__main__":
    sys.exit(main())
