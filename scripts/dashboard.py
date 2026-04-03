#!/usr/bin/env python3
"""
Parameter Golf Experiment Dashboard
====================================
Live tracker for ablation experiments with per-phase stopwatch timers.

Usage:
    python3 scripts/dashboard.py                          # default log dir
    python3 scripts/dashboard.py --log-dir experiment_logs/ablations
    python3 scripts/dashboard.py --once                   # print once and exit
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Experiment manifest (must match run_ablations.sh) ──
EXPERIMENTS = [
    ("0_sota_baseline",              "SOTA baseline (PR #1019, 1.1147 BPB)",   "baseline"),
    ("A1_9L_mlp3.5x",               "9L + MLP 3.5x (PR #1105 width)",         "arch"),
    ("A2_8L_mlp4x",                  "8L + MLP 4x (prior best combo)",         "arch"),
    ("A3_7L_mlp4x",                  "7L + MLP 4x (scaled test winner)",       "arch"),
    ("A4_11L_mlp3.5x",              "11L + MLP 3.5x (wider, same depth)",     "arch"),
    ("B1_muon_lr_0.03",             "Muon LR 0.03",                           "train"),
    ("B2_warmdown_4500",             "Warmdown 4500",                          "train"),
    ("B3_warmdown_5000",             "Warmdown 5000",                          "train"),
    ("B4_bigram_3072x112",           "BigramHash 3072x112",                    "train"),
    ("B5_muon_wd_0.06",             "WD 0.06",                                "train"),
    ("B6_head_lr_0.01",             "Head LR 0.01",                           "train"),
    ("B7_swiglu",                    "SwiGLU activation",                      "train"),
    ("B8_softcap_15",               "Softcap 15",                             "train"),
    ("B9_wd_0.2",                   "WD 0.2 (autoresearch optimal)",          "train"),
    ("C1_stride_32",                "Eval stride 32",                          "eval"),
    ("C2_stride_16",                "Eval stride 16",                          "eval"),
    ("D1_7L_mlp4x_muon03",         "7L/4x + Muon 0.03",                      "combo"),
    ("D2_9L_mlp3.5x_muon03",       "9L/3.5x + Muon 0.03",                    "combo"),
    ("D3_11L_mlp3.5x_bigram3072",  "11L/3.5x + bigger bigram",               "combo"),
    ("D4_best_combo_stride16",      "D1 + stride 16 eval",                    "combo"),
    ("D5_7L_swiglu_muon03",        "7L/4x + SwiGLU + Muon 0.03",             "combo"),
]

# ANSI
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
BG_YELLOW = "\033[43m"
BG_BLUE   = "\033[44m"
# Move cursor home + clear to end of screen (no flicker)
HOME_CLEAR = "\033[H\033[J"

CATEGORY_COLORS = {
    "baseline": CYAN,
    "arch":     BLUE,
    "train":    YELLOW,
    "eval":     GREEN,
    "combo":    BOLD,
}

# Cache start times parsed from log timestamps (persists across refreshes)
_start_times: dict[str, float] = {}

# How recently a file must have been modified to count as "actively running"
ACTIVE_THRESHOLD_S = 30.0


def fmt_time(seconds):
    """Format seconds as M:SS or H:MM:SS."""
    if seconds is None or seconds < 0:
        return ""
    s = int(seconds)
    if s >= 3600:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}"
    m, sec = divmod(s, 60)
    return f"{m}:{sec:02d}"


def _visible_len(s: str) -> int:
    """Length of string ignoring ANSI escape codes."""
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def _pad(s: str, width: int) -> str:
    """Right-justify string to `width` visible characters, ANSI-aware."""
    pad_needed = width - _visible_len(s)
    return (" " * max(0, pad_needed)) + s


def _get_start_time(logfile: Path, text: str) -> float | None:
    """Get stable start time for a log file from its first torch timestamp."""
    key = str(logfile)
    if key in _start_times:
        return _start_times[key]
    ts_match = re.search(r"W(\d{4}) (\d{2}:\d{2}:\d{2})\.\d+", text)
    if ts_match:
        mmdd, hms = ts_match.group(1), ts_match.group(2)
        year = datetime.now().year
        month, day = int(mmdd[:2]), int(mmdd[2:])
        h, m, s = [int(x) for x in hms.split(":")]
        _start_times[key] = datetime(year, month, day, h, m, s).timestamp()
        return _start_times[key]
    return None


def parse_log(logfile: Path, now: float) -> dict:
    """Extract metrics, timing, and phase from an experiment log file."""
    m = {
        "steps": "",
        "train_bpb": "",
        "int6_bpb": "",
        "slide_bpb": "",
        "step_avg": "",
        "params": "",
        "artifact_mb": "",
        "status": "pending",
        "train_str": "",       # formatted train stopwatch
        "eval_str": "",        # formatted eval stopwatch
        "train_secs": 0.0,     # for totals
        "eval_secs": 0.0,
    }
    if not logfile.exists():
        return m

    try:
        stat = logfile.stat()
    except OSError:
        return m

    if stat.st_size == 0:
        return m

    try:
        text = logfile.read_text(errors="replace")
    except Exception:
        return m

    if not text.strip():
        return m

    start_time = _get_start_time(logfile, text)
    has_error = ("Error" in text or "Traceback" in text)
    is_recent = (now - stat.st_mtime) < ACTIVE_THRESHOLD_S

    # ── Metrics ──
    steps = re.findall(r"^step:(\d+)", text, re.MULTILINE)
    if steps:
        m["steps"] = steps[-1]

    train_bpb = re.findall(r"^step:\d+.*?val_bpb:([\d.]+)", text, re.MULTILINE)
    if train_bpb:
        m["train_bpb"] = train_bpb[-1]

    ema_bpb = re.findall(r"post_ema.*?val_bpb:([\d.]+)", text)
    if ema_bpb:
        m["train_bpb"] = ema_bpb[-1]

    int6 = re.findall(r"final_int6_roundtrip_exact.*?val_bpb:([\d.]+)", text)
    if int6:
        m["int6_bpb"] = int6[-1]

    slide = re.findall(r"final_int6_sliding_window_exact.*?val_bpb:([\d.]+)", text)
    if slide:
        m["slide_bpb"] = slide[-1]

    step_avg = re.findall(r"step_avg:([\d.]+)", text)
    if step_avg:
        m["step_avg"] = step_avg[-1]

    params = re.findall(r"model_params:(\d+)", text)
    if params:
        m["params"] = f"{int(params[-1])/1e6:.1f}M"

    artifact = re.findall(r"artifact_bytes:(\d+)", text)
    if artifact:
        m["artifact_mb"] = f"{int(artifact[-1])/1e6:.2f}"

    # ── Timing from log ──
    train_times = re.findall(r"train_time:(\d+)ms", text)
    log_train_s = int(train_times[-1]) / 1000.0 if train_times else None

    wc_match = re.search(r"max_wallclock_seconds:([\d.]+)", text)
    budget = float(wc_match.group(1)) if wc_match else None

    ema_done = "post_ema" in text or "post_swa" in text
    gptq_start = bool(re.search(r"(?:GPTQ|gptq_quantiz|hessian_collection|Quantizing)", text))
    train_done = ema_done or gptq_start
    has_results = bool(int6 or slide)

    # ── Phase / status ──
    if has_results:
        m["status"] = "completed"
    elif has_error and not steps:
        m["status"] = "failed"
    elif train_done and is_recent:
        m["status"] = "eval"
    elif (steps or train_times) and is_recent:
        m["status"] = "training"
    elif steps or train_times or train_done:
        # Has data but file is stale — it finished or crashed
        if has_results:
            m["status"] = "completed"
        elif has_error:
            m["status"] = "failed"
        else:
            m["status"] = "completed"  # best guess: finished without sliding window
    elif has_error:
        m["status"] = "failed"

    # ── Format stopwatches ──
    if m["status"] == "training" and log_train_s is not None:
        # Tick: log's train_time + time since last log write
        train_s = log_train_s + (now - stat.st_mtime)
        m["train_secs"] = train_s
        elapsed_fmt = fmt_time(train_s)
        if budget:
            budget_fmt = fmt_time(budget)
            if train_s < budget:
                m["train_str"] = f"{BOLD}{CYAN}{elapsed_fmt}/{budget_fmt}{RESET}"
            else:
                m["train_str"] = f"{BOLD}{YELLOW}{elapsed_fmt}/{budget_fmt}{RESET}"
        else:
            m["train_str"] = f"{BOLD}{CYAN}{elapsed_fmt}{RESET}"

    elif m["status"] == "eval":
        # Train frozen, eval ticking
        if log_train_s is not None:
            m["train_secs"] = log_train_s
            if budget:
                m["train_str"] = f"{fmt_time(log_train_s)}/{fmt_time(budget)}"
            else:
                m["train_str"] = fmt_time(log_train_s)
        if start_time and log_train_s:
            eval_s = now - (start_time + log_train_s)
            if eval_s > 0:
                m["eval_secs"] = eval_s
                m["eval_str"] = f"{BOLD}{YELLOW}{fmt_time(eval_s)}{RESET}"

    elif m["status"] == "completed":
        if log_train_s is not None:
            m["train_secs"] = log_train_s
            if budget:
                m["train_str"] = f"{fmt_time(log_train_s)}/{fmt_time(budget)}"
            else:
                m["train_str"] = fmt_time(log_train_s)
        if start_time and log_train_s:
            eval_s = stat.st_mtime - (start_time + log_train_s)
            if eval_s > 0:
                m["eval_secs"] = eval_s
                m["eval_str"] = fmt_time(eval_s)

    return m


# Fixed-width status strings (all same visible width = 9)
_STATUS_FMT = {
    "pending":   f"{DIM}  PENDING{RESET}",
    "training":  f"{BG_BLUE}{WHITE}{BOLD} TRAIN..{RESET}",
    "eval":      f"{BG_YELLOW}{WHITE}{BOLD}  EVAL..{RESET}",
    "completed": f"{GREEN}{BOLD}    DONE{RESET}",
    "failed":    f"{RED}{BOLD}  FAILED{RESET}",
}
_STATUS_WIDTH = 8  # visible chars


def render(log_dir: Path) -> str:
    """Render the full dashboard as a string."""
    now = time.time()
    W = 120  # total line width

    status_file = log_dir / "status.json"
    status_data = {}
    if status_file.exists():
        try:
            with open(status_file) as f:
                status_data = json.load(f)
        except Exception:
            pass

    # Parse all logs
    rows = []
    for name, desc, cat in EXPERIMENTS:
        logfile = log_dir / f"{name}.log"
        m = parse_log(logfile, now)
        # Allow status.json to override for edge cases
        if m["status"] == "pending" and name in status_data:
            s = status_data[name].get("status", "pending")
            if s in ("completed", "failed", "timeout"):
                m["status"] = s
        rows.append((name, desc, cat, m))

    lines = []

    # Header
    lines.append(f"{BOLD}{'=' * W}{RESET}")
    lines.append(f"{BOLD}  PARAMETER GOLF EXPERIMENT DASHBOARD{RESET}    {DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    lines.append(f"{BOLD}{'=' * W}{RESET}")
    lines.append("")

    # Progress bar
    statuses = [m["status"] for *_, m in rows]
    n = len(rows)
    n_done = statuses.count("completed")
    n_train = statuses.count("training")
    n_eval = statuses.count("eval")
    n_fail = statuses.count("failed")

    bar = 30
    bd = int(bar * n_done / n)
    bt = int(bar * n_train / n)
    be = int(bar * n_eval / n)
    bf = int(bar * n_fail / n)
    bp = bar - bd - bt - be - bf
    prog = (
        f"  [{GREEN}{'#' * bd}{RESET}"
        f"{BLUE}{'>' * bt}{RESET}"
        f"{YELLOW}{'~' * be}{RESET}"
        f"{RED}{'!' * bf}{RESET}"
        f"{DIM}{'.' * bp}{RESET}]"
        f"  {n_done}/{n} done"
    )
    if n_train: prog += f"  {BLUE}{n_train} training{RESET}"
    if n_eval:  prog += f"  {YELLOW}{n_eval} eval{RESET}"
    if n_fail:  prog += f"  {RED}{n_fail} failed{RESET}"
    lines.append(prog)

    lines.append(f"  {DIM}Rules: 10min train + 10min eval on 8xH100. Budget scales for fewer GPUs.{RESET}")
    lines.append("")

    # Column widths (visible chars)
    C_STAT = 9
    C_NAME = 30
    C_TRAIN = 12
    C_EVAL = 8
    C_NUM = 7  # steps, ms/step
    C_BPB = 11 # bpb columns
    C_PAR = 8
    C_MB = 6

    hdr = (
        f"  {'Status':<{C_STAT}}"
        f"{'ID':<{C_NAME}}"
        f"{'Train':>{C_TRAIN}}"
        f"{'Eval':>{C_EVAL}}"
        f"{'Steps':>{C_NUM}}"
        f"{'TrainBPB':>{C_BPB}}"
        f"{'Int6 BPB':>{C_BPB}}"
        f"{'SlideBPB':>{C_BPB}}"
        f"{'ms/step':>{C_NUM}}"
        f"{'Params':>{C_PAR}}"
        f"{'MB':>{C_MB}}"
    )
    lines.append(f"{BOLD}{hdr}{RESET}")
    lines.append(f"  {'-' * (W - 2)}")

    prev_cat = None
    best_int6 = 999.0
    total_train = 0.0
    total_eval = 0.0

    for name, desc, cat, m in rows:
        if cat != prev_cat and prev_cat is not None:
            lines.append(f"  {DIM}{'-' * (W - 2)}{RESET}")
        prev_cat = cat

        status_str = _STATUS_FMT.get(m["status"], f"{m['status']:>{_STATUS_WIDTH}}")
        cat_color = CATEGORY_COLORS.get(cat, "")

        total_train += m["train_secs"]
        total_eval += m["eval_secs"]

        # Highlight best int6
        int6_str = m["int6_bpb"]
        try:
            v = float(int6_str)
            if v < best_int6:
                best_int6 = v
                int6_str = f"{GREEN}{BOLD}{m['int6_bpb']}{RESET}"
        except ValueError:
            pass

        line = (
            f"  {status_str} "
            f"{cat_color}{name:<{C_NAME}}{RESET}"
            f"{_pad(m['train_str'], C_TRAIN)}"
            f"{_pad(m['eval_str'], C_EVAL)}"
            f"{m['steps']:>{C_NUM}}"
            f"{m['train_bpb']:>{C_BPB}}"
            f"{_pad(int6_str, C_BPB)}"
            f"{m['slide_bpb']:>{C_BPB}}"
            f"{m['step_avg']:>{C_NUM}}"
            f"{m['params']:>{C_PAR}}"
            f"{m['artifact_mb']:>{C_MB}}"
        )
        lines.append(line)

    lines.append(f"  {'-' * (W - 2)}")
    lines.append(
        f"  {DIM}Totals:  Train {fmt_time(total_train)}"
        f"  |  Eval {fmt_time(total_eval)}"
        f"  |  Combined {fmt_time(total_train + total_eval)}"
        f"    Log dir: {log_dir}/{RESET}"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Experiment Dashboard")
    parser.add_argument("--log-dir", default="experiment_logs/ablations", help="Log directory")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.once:
        print(render(log_dir))
        return

    # Hide cursor for cleaner display
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        # Initial full clear, then use cursor-home to overwrite
        sys.stdout.write("\033[2J")
        sys.stdout.flush()
        while True:
            output = render(log_dir)
            # Move cursor to top-left and overwrite; clear any leftover lines
            sys.stdout.write(f"{HOME_CLEAR}{output}")
            sys.stdout.flush()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        # Show cursor again
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
