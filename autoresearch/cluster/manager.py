"""Cluster manager: discovers nodes, tracks GPU state, allocates resources."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from ..config import GPUNodeConfig, AutoResearchConfig
from ..db.models import GPUInfo, NodeState, NodeStatus, ExperimentStatus
from ..db.registry import Registry
from .node_client import NodeClient, SSHConfig

# Statuses that mean "this experiment owns GPUs on a node right now".
_ACTIVE_RUN_STATUSES = (
    ExperimentStatus.DEPLOYING,
    ExperimentStatus.SCREENING,
    ExperimentStatus.GATING,
    ExperimentStatus.INSPECTING,
    ExperimentStatus.PROMOTING,
)

log = logging.getLogger(__name__)


class ClusterManager:
    """Manages the fleet of GPU nodes.

    Responsibilities:
    - Periodic health checks (probe GPUs, detect offline nodes)
    - GPU allocation across nodes for experiments
    - Job deployment and lifecycle management
    """

    def __init__(self, config: AutoResearchConfig, registry: Registry):
        self.config = config
        self.registry = registry
        self._lock = threading.Lock()
        self._nodes: dict[str, NodeState] = {}
        self._clients: dict[str, NodeClient] = {}
        self._running_jobs: dict[str, _RunningJob] = {}  # exp_id -> job info

        # Initialize clients from config
        for node_cfg in config.nodes:
            if not node_cfg.enabled:
                continue
            ssh_cfg = SSHConfig(
                host=node_cfg.host, user=node_cfg.user,
                port=node_cfg.ssh_port, key_file=node_cfg.ssh_key,
                work_dir=node_cfg.work_dir,
                env_setup=node_cfg.env_setup,
            )
            self._clients[node_cfg.host] = NodeClient(ssh_cfg, label=node_cfg.label)
            self._nodes[node_cfg.host] = NodeState(
                host=node_cfg.host, label=node_cfg.label,
                status=NodeStatus.OFFLINE,
            )

    # ── Discovery & Health ─────────────────────────────────────────────

    def discover_all(self):
        """Probe all configured nodes in parallel."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(self._probe_node, host): host
                for host in self._clients
            }
            for f in concurrent.futures.as_completed(futures):
                host = futures[f]
                try:
                    state = f.result()
                    with self._lock:
                        self._nodes[host] = state
                    # Persist to DB
                    self.registry.upsert_node(
                        host=host, label=state.label,
                        status=state.status.value,
                        gpu_count=state.total_gpus,
                        gpu_info=json.dumps([
                            {"index": g.index, "name": g.name,
                             "memory_total_mb": g.memory_total_mb,
                             "memory_used_mb": g.memory_used_mb,
                             "utilization_pct": g.utilization_pct,
                             "temperature_c": g.temperature_c,
                             "assigned_experiment": g.assigned_experiment}
                            for g in state.gpus
                        ]),
                        error_message=state.error_message,
                    )
                except Exception as e:
                    log.error("Discovery failed for %s: %s", host, e)

    def _probe_node(self, host: str) -> NodeState:
        client = self._clients[host]
        state = client.get_node_state()

        # Preserve the previous NodeState on transient probe failure.
        # A single timed-out nvidia-smi should not flip a healthy node to
        # ERROR/OFFLINE and evict running jobs from the dashboard view.
        # We only accept the ERROR state after a run of consecutive
        # failures exceeds the tolerance (default: 3).
        if state.status in (NodeStatus.OFFLINE, NodeStatus.ERROR):
            prior = self._nodes.get(host)
            fail_counts = getattr(self, "_probe_fails", None)
            if fail_counts is None:
                fail_counts = {}
                self._probe_fails = fail_counts
            fail_counts[host] = fail_counts.get(host, 0) + 1
            tolerance = 3
            if prior and prior.status == NodeStatus.ONLINE and fail_counts[host] < tolerance:
                log.info(
                    "Probe failed for %s (%d/%d) — keeping prior ONLINE state",
                    host, fail_counts[host], tolerance,
                )
                prior.last_heartbeat = datetime.utcnow()
                return prior
        else:
            # Success — reset the failure counter
            if hasattr(self, "_probe_fails"):
                self._probe_fails.pop(host, None)

        state.last_heartbeat = datetime.utcnow()

        # Restore GPU assignments from running jobs.
        #
        # Two sources: (1) the in-process `_running_jobs` dict owned by the
        # scheduler that allocated them, and (2) the experiments table, which
        # is the cross-process source of truth (the dashboard runs in a
        # separate process from the worker and only sees the DB). We union
        # both so the dashboard reports live GPU assignments even though it
        # never called `allocate_gpus` itself.
        assignments: dict[int, str] = {}
        with self._lock:
            for exp_id, job in self._running_jobs.items():
                if job.host == host:
                    for idx in job.gpu_indices:
                        assignments[idx] = exp_id
        for exp_id, idx in self._db_running_jobs_for_host(host):
            assignments.setdefault(idx, exp_id)
        for gpu in state.gpus:
            if gpu.index in assignments:
                gpu.assigned_experiment = assignments[gpu.index]
        return state

    def _db_running_jobs_for_host(self, host: str) -> list[tuple[str, int]]:
        """Return (exp_id, gpu_index) pairs for experiments live on `host`.

        Reads the experiments table — which the scheduler populates via
        `assign_experiment_node()` at launch time — so any process holding
        a `ClusterManager` (dashboard included) can reconstruct the live
        allocation map without sharing in-memory state with the worker.
        """
        out: list[tuple[str, int]] = []
        try:
            for status in _ACTIVE_RUN_STATUSES:
                for exp in self.registry.list_experiments(status=status):
                    if exp.node_host != host:
                        continue
                    for idx in (exp.gpu_indices or []):
                        out.append((exp.id, idx))
        except Exception as e:
            log.debug("DB running-job lookup failed for %s: %s", host, e)
        return out

    def _db_running_experiment_ids(self) -> set[str]:
        ids: set[str] = set()
        try:
            for status in _ACTIVE_RUN_STATUSES:
                for exp in self.registry.list_experiments(status=status):
                    if exp.node_host:
                        ids.add(exp.id)
        except Exception as e:
            log.debug("DB running-exp lookup failed: %s", e)
        return ids

    # ── Allocation ─────────────────────────────────────────────────────

    def get_cluster_summary(self) -> dict:
        """Return a summary of the cluster state.

        GPU→experiment assignments and the running-jobs count are unioned
        across the in-process `_running_jobs` dict (populated by the
        scheduler that owns this manager) and the experiments DB table
        (the cross-process source of truth). This matters because the
        dashboard and the scheduler run as separate processes with
        separate `ClusterManager` instances — without the DB fallback the
        dashboard would always report 0 running jobs and idle GPUs.
        """
        # Build per-host DB assignment map once so we don't re-query inside
        # the node loop.
        db_assign: dict[str, dict[int, str]] = {}
        db_running_ids: set[str] = set()
        try:
            for status in _ACTIVE_RUN_STATUSES:
                for exp in self.registry.list_experiments(status=status):
                    if not exp.node_host:
                        continue
                    db_running_ids.add(exp.id)
                    host_map = db_assign.setdefault(exp.node_host, {})
                    for idx in (exp.gpu_indices or []):
                        host_map.setdefault(idx, exp.id)
        except Exception as e:
            log.debug("DB cluster summary lookup failed: %s", e)

        with self._lock:
            # Union of in-process running ids with DB-derived ids
            running_ids = set(self._running_jobs.keys()) | db_running_ids

            total_gpus = sum(n.total_gpus for n in self._nodes.values()
                             if n.status == NodeStatus.ONLINE)
            online_nodes = sum(1 for n in self._nodes.values()
                               if n.status == NodeStatus.ONLINE)

            def _gpu_exp(host: str, g) -> Optional[str]:
                # In-process state first (freshest), fall back to DB.
                if g.assigned_experiment:
                    return g.assigned_experiment
                return db_assign.get(host, {}).get(g.index)

            def _free_count(host: str, n: NodeState) -> int:
                return sum(
                    1 for g in n.gpus if _gpu_exp(host, g) is None
                )

            free_gpus = sum(
                _free_count(host, n) for host, n in self._nodes.items()
                if n.status == NodeStatus.ONLINE
            )

            return {
                "online_nodes": online_nodes,
                "total_nodes": len(self._nodes),
                "total_gpus": total_gpus,
                "free_gpus": free_gpus,
                "running_jobs": len(running_ids),
                "nodes": {
                    host: {
                        "label": n.label,
                        "status": n.status.value,
                        "total_gpus": n.total_gpus,
                        "free_gpus": _free_count(host, n),
                        "gpus": [
                            {"index": g.index, "name": g.name,
                             "mem_used_mb": g.memory_used_mb,
                             "mem_total_mb": g.memory_total_mb,
                             "util_pct": g.utilization_pct,
                             "temp_c": g.temperature_c,
                             "experiment": _gpu_exp(host, g)}
                            for g in n.gpus
                        ],
                    }
                    for host, n in self._nodes.items()
                },
            }

    def allocate_gpus(self, experiment_id: str, gpu_count: int
                       ) -> Optional[tuple[str, list[int]]]:
        """Find a node with enough free GPUs and allocate them.

        Returns (host, gpu_indices) or None if no capacity.
        """
        with self._lock:
            # Try to allocate on a single node (no cross-node jobs)
            for host, node in self._nodes.items():
                if node.status != NodeStatus.ONLINE:
                    continue
                free = node.free_gpus
                if len(free) >= gpu_count:
                    indices = [g.index for g in free[:gpu_count]]
                    # Mark as assigned
                    for g in node.gpus:
                        if g.index in indices:
                            g.assigned_experiment = experiment_id
                    self._running_jobs[experiment_id] = _RunningJob(
                        host=host, gpu_indices=indices,
                    )
                    log.info("Allocated %s: %s GPUs %s", experiment_id, host, indices)
                    return (host, indices)
        return None

    def release_gpus(self, experiment_id: str):
        """Release GPUs held by an experiment."""
        with self._lock:
            job = self._running_jobs.pop(experiment_id, None)
            if not job:
                return
            node = self._nodes.get(job.host)
            if node:
                for g in node.gpus:
                    if g.assigned_experiment == experiment_id:
                        g.assigned_experiment = None
            log.info("Released GPUs for %s on %s", experiment_id, job.host)

    # ── Job Lifecycle ──────────────────────────────────────────────────

    def deploy_experiment(self, experiment_id: str, job_dir,
                           env_overrides: dict, script: str = "train_gpt.py",
                           wallclock_s: int = 180,
                           branch: Optional[str] = None) -> bool:
        """Deploy an experiment to its allocated node.

        If branch is provided, the node will git-pull that branch before
        launching the job, ensuring reproducible code at a known commit.
        """
        with self._lock:
            job = self._running_jobs.get(experiment_id)
        if not job:
            log.error("No allocation for %s", experiment_id)
            return False

        client = self._clients.get(job.host)
        if not client:
            return False

        from pathlib import Path
        pid = client.deploy_job(
            job_dir=Path(job_dir) if job_dir else Path("/dev/null"),
            experiment_id=experiment_id,
            env_overrides=env_overrides,
            gpu_indices=job.gpu_indices,
            script=script,
            wallclock_s=wallclock_s,
            nproc=len(job.gpu_indices),
            branch=branch,
        )
        if pid:
            with self._lock:
                job.pid = pid
            return True
        return False

    def check_job_alive(self, experiment_id: str) -> bool:
        """Check if a deployed job is still running."""
        with self._lock:
            job = self._running_jobs.get(experiment_id)
        if not job or not job.pid:
            return False
        client = self._clients.get(job.host)
        if not client:
            return False
        return client.check_job_running(job.pid)

    def kill_experiment(self, experiment_id: str, mode: str = "graceful",
                         grace_period_s: int = 10) -> bool:
        """Kill a running experiment on its node.

        mode: 'graceful' (SIGTERM → wait → SIGKILL) or 'force' (SIGKILL).
        """
        with self._lock:
            job = self._running_jobs.get(experiment_id)
        if not job or not job.pid:
            return False
        client = self._clients.get(job.host)
        if not client:
            return False
        killed = client.kill_job(job.pid, mode=mode,
                                  grace_period_s=grace_period_s)
        if killed:
            self.release_gpus(experiment_id)
        return killed

    def get_log_tail(self, experiment_id: str, lines: int = 100) -> str:
        """Get recent log output from a running experiment."""
        with self._lock:
            job = self._running_jobs.get(experiment_id)
        if not job:
            return ""
        client = self._clients.get(job.host)
        if not client:
            return ""
        return client.fetch_log_tail(experiment_id, lines)

    def sync_experiment_results(self, experiment_id: str, local_dir) -> bool:
        """Pull results from remote node to local autoresearch host."""
        with self._lock:
            job = self._running_jobs.get(experiment_id)
        if not job:
            return False
        client = self._clients.get(job.host)
        if not client:
            return False
        from pathlib import Path
        return client.sync_results_back(experiment_id, Path(local_dir))

    # ── Log Streaming ─────────────────────────────────────────────────

    def fetch_log_incremental(self, experiment_id: str,
                               byte_offset: int = 0) -> tuple[str, int]:
        """Fetch new log content since byte_offset for incremental streaming.

        Returns (new_text, new_byte_offset).
        """
        with self._lock:
            job = self._running_jobs.get(experiment_id)
        if not job:
            return "", byte_offset
        client = self._clients.get(job.host)
        if not client:
            return "", byte_offset
        return client.fetch_log_since(experiment_id, byte_offset)

    def sync_log_file(self, experiment_id: str, local_path) -> bool:
        """Rsync the train.log from the remote node to a local path."""
        with self._lock:
            job = self._running_jobs.get(experiment_id)
        if not job:
            return False
        client = self._clients.get(job.host)
        if not client:
            return False
        from pathlib import Path
        return client.sync_log_file(experiment_id, Path(local_path))

    # ── Local mode (no SSH, for testing) ───────────────────────────────

    def add_local_node(self, gpu_count: int = 0):
        """Register localhost as a node (for testing without real GPUs)."""
        gpus = [GPUInfo(index=i, name="virtual-gpu") for i in range(gpu_count)]
        state = NodeState(
            host="localhost", label=f"local-{gpu_count}gpu",
            status=NodeStatus.ONLINE, gpus=gpus,
            last_heartbeat=datetime.utcnow(),
        )
        with self._lock:
            self._nodes["localhost"] = state
        self.registry.upsert_node(
            host="localhost", label=state.label,
            status="online", gpu_count=gpu_count,
        )


class _RunningJob:
    """Tracks a deployed job."""
    __slots__ = ("host", "gpu_indices", "pid", "started_at")

    def __init__(self, host: str, gpu_indices: list[int],
                 pid: Optional[int] = None):
        self.host = host
        self.gpu_indices = gpu_indices
        self.pid = pid
        self.started_at = time.time()
