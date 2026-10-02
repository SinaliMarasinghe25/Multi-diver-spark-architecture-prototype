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
# Container (cgroup v2), when running inside Docker and the files exist:
#   cg_mem_mib   memory.current (includes page cache)
#   cg_anon_mib  anonymous memory from memory.stat (excludes page cache)
#   cg_cpu_cores Δ cpu.stat usage_usec / Δwall
#   limits       cpu.max and memory.max (the cap actually applied)
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
# cgroup v2 readers (Docker)
# =============================================================================


def _cg(name: str) -> str | None:
    raw = _read(os.path.join(_CGROUP, name))
    return raw.strip() if raw is not None else None


def cgroup_available() -> bool:
    return os.path.exists(os.path.join(_CGROUP, "memory.current"))


def _cg_cpu_usec() -> int | None:
    for line in (_cg("cpu.stat") or "").splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1])
    return None


def _cg_anon_bytes() -> int | None:
    for line in (_cg("memory.stat") or "").splitlines():
        if line.startswith("anon "):
            return int(line.split()[1])
    return None


def cgroup_limits() -> dict:
    """Applied caps: cpu.max 'quota period' and memory.max (None = unlimited)."""
    cpu_limit = mem_limit = None
    cpu_max = _cg("cpu.max")
    if cpu_max and not cpu_max.startswith("max"):
        quota, period = cpu_max.split()
        cpu_limit = int(quota) / int(period)
    mem_max = _cg("memory.max")
    if mem_max and mem_max != "max":
        mem_limit = int(mem_max) / _MIB
    return {"cgroup_cpu_limit_cores": cpu_limit, "cgroup_mem_limit_mib": mem_limit}


# =============================================================================
# Sampler
# =============================================================================


class RankSampler:
    """Samples the process tree rooted at `root_pid` (and the cgroup, if any)."""

    def __init__(self, root_pid: int, use_cgroup: bool | None = None):
        self.root_pid = root_pid
        self.use_cgroup = cgroup_available() if use_cgroup is None else use_cgroup
        self.t0 = time.monotonic()
        self._last_t = self.t0
        self._last_ticks: dict[int, int] = {}
        self._last_cg_usec = _cg_cpu_usec() if self.use_cgroup else None
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
            cur = _cg("memory.current")
            anon = _cg_anon_bytes()
            usec = _cg_cpu_usec()
            row["cg_mem_mib"] = int(cur) / _MIB if cur else None
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
        "cgroup": sampler.use_cgroup,
        **(cgroup_limits() if sampler.use_cgroup else {}),
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
