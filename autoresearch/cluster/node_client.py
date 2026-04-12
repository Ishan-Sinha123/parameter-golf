"""SSH client for communicating with GPU nodes.

Uses subprocess + ssh (not paramiko) for simplicity and to leverage
the user's existing SSH config / agent forwarding.

Deployment model:
  1. Azure VM makes code changes, commits, pushes to a branch
  2. GPU node does `git pull` to get the latest code
  3. Job is launched via `torchrun` in a subprocess
  4. Logs are continuously streamed back to Azure VM
  5. Health checks monitor the subprocess
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time  # noqa: F401 (used by probe_gpus retries)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..db.models import GPUInfo, NodeState, NodeStatus

log = logging.getLogger(__name__)


@dataclass
class SSHConfig:
    host: str
    user: str = "azureuser"
    port: int = 22
    key_file: Optional[str] = None
    work_dir: str = "/workspace/parameter-golf"
    connect_timeout: int = 30
    env_setup: str = ""  # e.g. "source /workspace/parameter-golf/.venv/bin/activate"

    @property
    def ssh_base(self) -> list[str]:
        # ServerAliveInterval/Max keeps the TCP connection fresh across
        # vast.ai's occasional idle drops so `Connection timed out during
        # banner exchange` stops flapping the node offline.
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4",
            "-o", "TCPKeepAlive=yes",
            "-o", "BatchMode=yes",
            "-p", str(self.port),
        ]
        if self.key_file:
            cmd.extend(["-i", self.key_file])
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    @property
    def rsync_base(self) -> list[str]:
        rsh = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout={self.connect_timeout} -p {self.port}"
        if self.key_file:
            rsh += f" -i {self.key_file}"
        return ["rsync", "-az", "--progress", "-e", rsh]


class NodeClient:
    """Manages communication with a single GPU node over SSH."""

    def __init__(self, ssh_cfg: SSHConfig, label: str = ""):
        self.ssh = ssh_cfg
        self.label = label or ssh_cfg.host

    def _run_ssh(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        if self.ssh.env_setup:
            command = f"{self.ssh.env_setup} && {command}"
        full_cmd = self.ssh.ssh_base + [command]
        return subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=timeout,
        )

    def ping(self) -> bool:
        """Check if node is reachable."""
        try:
            r = self._run_ssh("echo ok", timeout=self.ssh.connect_timeout + 5)
            return r.returncode == 0 and "ok" in r.stdout
        except (subprocess.TimeoutExpired, Exception):
            return False

    def probe_gpus(self) -> list[GPUInfo]:
        """Query nvidia-smi on the node and return GPU info.

        Retries on transient SSH failures (banner timeout, connection
        reset) with exponential backoff. Returns [] only after all
        attempts fail — callers should treat [] as "probe failed" not
        "node has zero GPUs".
        """
        query = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
        cmd = f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits 2>/dev/null"
        backoffs = [0, 2, 5]
        last_err = ""
        for attempt, delay in enumerate(backoffs):
            if delay:
                time.sleep(delay)
            try:
                r = self._run_ssh(cmd, timeout=30)
                if r.returncode != 0:
                    last_err = r.stderr.strip()[:200]
                    log.debug("probe_gpus attempt %d on %s failed: %s",
                              attempt + 1, self.ssh.host, last_err)
                    continue
                gpus = []
                for line in r.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 6:
                        continue
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(float(parts[2])),
                        memory_used_mb=int(float(parts[3])),
                        utilization_pct=int(float(parts[4])),
                        temperature_c=int(float(parts[5])),
                    ))
                return gpus
            except subprocess.TimeoutExpired as e:
                last_err = f"timeout: {e}"
                log.debug("probe_gpus attempt %d on %s timed out",
                          attempt + 1, self.ssh.host)
            except Exception as e:
                last_err = str(e)
                log.debug("probe_gpus attempt %d on %s exception: %s",
                          attempt + 1, self.ssh.host, e)
        log.warning("GPU probe failed on %s after %d attempts: %s",
                    self.ssh.host, len(backoffs), last_err)
        return []

    def get_node_state(self) -> NodeState:
        """Full node state: reachability + GPU inventory."""
        if not self.ping():
            return NodeState(
                host=self.ssh.host, label=self.label,
                status=NodeStatus.OFFLINE,
            )
        gpus = self.probe_gpus()
        return NodeState(
            host=self.ssh.host, label=self.label,
            status=NodeStatus.ONLINE if gpus else NodeStatus.ERROR,
            gpus=gpus,
            last_heartbeat=None,  # set by caller
            error_message="" if gpus else "nvidia-smi returned no GPUs",
        )

    # ── Git-based Deployment ──────────────────────────────────────────

    def git_pull(self, branch: str = "main") -> bool:
        """Pull latest code on the GPU node from the given branch.

        The GPU node must already have the repo cloned and the environment
        set up. This just does: git fetch && git checkout <branch> && git pull
        """
        cmd = (
            f"cd {self.ssh.work_dir} && "
            f"git fetch origin && "
            f"git checkout {branch} && "
            f"git pull origin {branch}"
        )
        try:
            r = self._run_ssh(cmd, timeout=60)
            if r.returncode == 0:
                log.info("git pull succeeded on %s (branch=%s)", self.ssh.host, branch)
                return True
            else:
                log.error("git pull failed on %s: %s", self.ssh.host,
                          r.stderr.strip()[:300])
                return False
        except subprocess.TimeoutExpired:
            log.error("git pull timed out on %s", self.ssh.host)
            return False

    def check_repo_status(self) -> dict:
        """Check the git status on the GPU node."""
        cmd = (
            f"cd {self.ssh.work_dir} && "
            f"echo BRANCH=$(git rev-parse --abbrev-ref HEAD) && "
            f"echo COMMIT=$(git rev-parse --short HEAD) && "
            f"echo DIRTY=$(git status --porcelain | wc -l)"
        )
        try:
            r = self._run_ssh(cmd, timeout=15)
            if r.returncode != 0:
                return {"error": r.stderr.strip()[:200]}
            info = {}
            for line in r.stdout.strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
            return info
        except Exception as e:
            return {"error": str(e)}

    # ── Job Deployment ────────────────────────────────────────────────

    def deploy_job(self, job_dir: Path, experiment_id: str,
                   env_overrides: dict, gpu_indices: list[int],
                   script: str = "train_gpt.py",
                   wallclock_s: int = 180,
                   train_wallclock_s: Optional[int] = None,
                   nproc: int = 2,
                   branch: Optional[str] = None) -> Optional[int]:
        """Deploy and launch a training job on this node.

        Deployment flow:
        1. If branch specified, git pull that branch
        2. Create experiment log directory on remote node
        3. Launch torchrun with env overrides via nohup
        4. Return the remote PID for monitoring

        Returns remote PID, or None on failure.
        """
        remote_exp_dir = f"{self.ssh.work_dir}/experiments/{experiment_id}"

        # 1. Git pull if branch specified
        if branch:
            if not self.git_pull(branch):
                log.error("Git pull failed, aborting deploy for %s", experiment_id)
                return None

        # 2. Create remote experiment dir
        r = self._run_ssh(f"mkdir -p {remote_exp_dir}", timeout=10)
        if r.returncode != 0:
            log.error("mkdir failed on %s: %s", self.ssh.host, r.stderr)
            return None

        # 3. Rsync any extra job-specific files (config overrides etc.)
        if job_dir and job_dir.exists():
            rsync_cmd = self.ssh.rsync_base + [
                f"{job_dir}/", f"{self.ssh.user}@{self.ssh.host}:{remote_exp_dir}/",
            ]
            try:
                subprocess.run(rsync_cmd, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                log.error("rsync timeout to %s", self.ssh.host)
                return None

        # 4. Build env string.
        # Stage-budget variables (MAX_WALLCLOCK_SECONDS etc.) must always
        # win over recipe env_overrides — SOTA baselines ship with a 4h
        # budget which, if not clamped, makes screen/gate stages train
        # for the full baseline duration instead of their stage budget.
        cuda_devices = ",".join(str(i) for i in gpu_indices)
        _stage_budget_keys = {"MAX_WALLCLOCK_SECONDS"}
        env_parts = [
            f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_devices)}",
            f"EXPERIMENT_DESC={shlex.quote(experiment_id)}",
            f"RESULTS_TSV_PATH={shlex.quote(f'{remote_exp_dir}/results.tsv')}",
            f"LOG_DIR={shlex.quote(remote_exp_dir)}",
        ]
        for k, v in env_overrides.items():
            if k in _stage_budget_keys:
                continue  # clamped to stage budget below
            env_parts.append(f"{k}={shlex.quote(str(v))}")
        _train_budget = train_wallclock_s if train_wallclock_s is not None else wallclock_s
        env_parts.append(f"MAX_WALLCLOCK_SECONDS={_train_budget}")
        env_str = " ".join(env_parts)

        # 5. Launch via nohup + torchrun.
        # We use a subshell double-fork: `( setsid nohup cmd & )` so the
        # outer bash never tracks the backgrounded job at all. Without the
        # subshell, bash -c holds the SSH channel open waiting on torchrun's
        # descendants even with setsid/disown (observed on vast.ai: bash
        # stayed in S state after `& disown` because torchrun's multiproc
        # workers confused the session leader check).
        # Write PID to a file then cat it so we can still return the PID.
        pid_file = f"{remote_exp_dir}/launch.pid"
        launch_cmd = (
            f"cd {self.ssh.work_dir} && "
            f"( setsid nohup env {env_str} "
            f"torchrun --standalone --nproc_per_node={nproc} {script} "
            f"< /dev/null > {remote_exp_dir}/train.log 2>&1 & "
            f"echo $! > {pid_file} ) && "
            f"sleep 0.5 && cat {pid_file}"
        )
        try:
            r = self._run_ssh(launch_cmd, timeout=30)
            # Parse PID from last non-empty line of stdout. Some env_setup
            # scripts (e.g. conda/uv activate on vast-h100-2) print a
            # banner to stdout before the PID, so `stdout.strip().isdigit()`
            # would incorrectly fail even though the launch succeeded.
            pid_line = ""
            for line in reversed(r.stdout.splitlines()):
                s = line.strip()
                if s:
                    pid_line = s
                    break
            if r.returncode == 0 and pid_line.isdigit():
                pid = int(pid_line)
                log.info("Launched %s on %s GPUs %s, PID=%d",
                         experiment_id, self.ssh.host, gpu_indices, pid)
                return pid
            else:
                log.error(
                    "Launch failed on %s: rc=%d stdout=%r stderr=%r",
                    self.ssh.host, r.returncode, r.stdout, r.stderr,
                )
                return None
        except Exception as e:
            log.error("Launch error on %s: %s", self.ssh.host, e)
            return None

    # ── Job Monitoring ────────────────────────────────────────────────

    def check_job_running(self, pid: int) -> bool:
        """Check if a PID is still alive on the remote node."""
        try:
            r = self._run_ssh(f"kill -0 {pid} 2>/dev/null && echo alive", timeout=10)
            return "alive" in r.stdout
        except Exception:
            return False

    def kill_job(self, pid: int, mode: str = "graceful",
                  grace_period_s: int = 10) -> bool:
        """Kill a running job on the remote node.

        Modes:
        - "graceful": SIGTERM, wait grace_period_s, then SIGKILL if still
          alive. Lets the training script flush checkpoints, write final
          logs, and exit cleanly.
        - "force":    SIGKILL immediately (kill -9). Use only if graceful
          has already failed or the job is clearly wedged.

        We kill the whole process group (negative PID) so that torchrun's
        child workers also die — otherwise they'd orphan and keep GPUs.
        """
        if mode == "force":
            cmd = (
                f"kill -9 -{pid} 2>/dev/null; kill -9 {pid} 2>/dev/null; "
                f"echo done"
            )
            timeout = 10
        else:
            cmd = (
                f"kill -TERM -{pid} 2>/dev/null; kill -TERM {pid} 2>/dev/null; "
                f"for i in $(seq 1 {grace_period_s}); do "
                f"  kill -0 {pid} 2>/dev/null || break; sleep 1; "
                f"done; "
                f"kill -0 {pid} 2>/dev/null && "
                f"  {{ kill -9 -{pid} 2>/dev/null; kill -9 {pid} 2>/dev/null; }}; "
                f"echo done"
            )
            timeout = grace_period_s + 10
        try:
            r = self._run_ssh(cmd, timeout=timeout)
            return "done" in r.stdout
        except Exception:
            return False

    # ── Log Streaming ���────────────────────────────────────────────────

    def fetch_log_tail(self, experiment_id: str, lines: int = 100) -> str:
        """Get the tail of a remote experiment's train.log."""
        remote_log = f"{self.ssh.work_dir}/experiments/{experiment_id}/train.log"
        try:
            r = self._run_ssh(f"tail -n {lines} {remote_log} 2>/dev/null", timeout=10)
            return r.stdout if r.returncode == 0 else ""
        except Exception:
            return ""

    def fetch_log_since(self, experiment_id: str,
                        byte_offset: int = 0) -> tuple[str, int]:
        """Fetch new log content since byte_offset.

        Returns (new_text, new_byte_offset) so the caller can track
        where they left off for incremental streaming.
        """
        remote_log = f"{self.ssh.work_dir}/experiments/{experiment_id}/train.log"
        # Use dd to skip already-read bytes
        cmd = (
            f"stat -c %s {remote_log} 2>/dev/null && "
            f"dd if={remote_log} bs=1 skip={byte_offset} 2>/dev/null"
        )
        try:
            r = self._run_ssh(cmd, timeout=10)
            if r.returncode != 0:
                return "", byte_offset
            lines = r.stdout.split("\n", 1)
            if len(lines) < 2:
                return "", byte_offset
            file_size = int(lines[0].strip())
            new_text = lines[1] if len(lines) > 1 else ""
            return new_text, file_size
        except Exception:
            return "", byte_offset

    def fetch_metrics(self, experiment_id: str) -> str:
        """Fetch the full metrics.jsonl from a remote experiment."""
        remote_metrics = f"{self.ssh.work_dir}/experiments/{experiment_id}/metrics.jsonl"
        try:
            r = self._run_ssh(f"cat {remote_metrics} 2>/dev/null", timeout=10)
            return r.stdout if r.returncode == 0 else ""
        except Exception:
            return ""

    # ── Result Sync ────��──────────────────────────────────────────────

    def sync_results_back(self, experiment_id: str, local_dir: Path) -> bool:
        """Rsync results from remote node back to autoresearch host."""
        remote_exp_dir = f"{self.ssh.user}@{self.ssh.host}:{self.ssh.work_dir}/experiments/{experiment_id}/"
        local_dir.mkdir(parents=True, exist_ok=True)
        rsync_cmd = self.ssh.rsync_base + [remote_exp_dir, f"{local_dir}/"]
        try:
            r = subprocess.run(rsync_cmd, capture_output=True, timeout=120)
            return r.returncode == 0
        except Exception as e:
            log.error("sync_results_back failed for %s: %s", experiment_id, e)
            return False

    def sync_log_file(self, experiment_id: str, local_path: Path) -> bool:
        """Rsync just the train.log from remote to local."""
        remote_log = f"{self.ssh.user}@{self.ssh.host}:{self.ssh.work_dir}/experiments/{experiment_id}/train.log"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        rsync_cmd = self.ssh.rsync_base + [remote_log, str(local_path)]
        try:
            r = subprocess.run(rsync_cmd, capture_output=True, timeout=30)
            return r.returncode == 0
        except Exception:
            return False
