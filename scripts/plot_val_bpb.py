#!/usr/bin/env python3
"""Plot val BPB curves for record runs.

Scans run_logs/ for .log files produced by run_all_records.sh and plots
val BPB over training steps and wall time for each record.

Usage:
    python3 plot_val_bpb.py                        # defaults: run_logs/ -> run_logs/plots/
    python3 plot_val_bpb.py --log-dir DIR --out-dir DIR
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAL_RE = re.compile(
    r"step:(\d+)/\d+\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:(\d+)ms"
)
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)")

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def parse_log(path: Path) -> dict:
    text = path.read_text()
    val_steps, val_bpbs, val_times = [], [], []
    final_bpb = None

    for m in VAL_RE.finditer(text):
        val_steps.append(int(m.group(1)))
        val_bpbs.append(float(m.group(3)))
        val_times.append(int(m.group(4)))

    m = FINAL_RE.search(text)
    if m:
        final_bpb = float(m.group(2))

    return dict(val_steps=val_steps, val_bpbs=val_bpbs, val_times=val_times, final_bpb=final_bpb)


def short_label(name: str) -> str:
    """Shorten log file names for legend readability."""
    # run_all_records uses track__date_name format
    parts = name.split("__", 1)
    return parts[-1] if len(parts) > 1 else name


def main():
    parser = argparse.ArgumentParser(description="Plot val BPB for record runs")
    parser.add_argument("--log-dir", default="run_logs", help="Directory with .log files")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <log-dir>/plots)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    print(f"Parsing {len(log_files)} log files from {log_dir}/")
    logs = {}
    for lf in log_files:
        data = parse_log(lf)
        if data["val_steps"]:
            logs[lf.stem] = data

    if not logs:
        print("No validation data found in any log file.")
        return

    # Sort by final BPB (best first) for legend ordering
    sorted_names = sorted(logs.keys(), key=lambda n: logs[n]["final_bpb"] if logs[n]["final_bpb"] is not None else (min(logs[n]["val_bpbs"]) if logs[n]["val_bpbs"] else 999))

    # ── Val BPB vs Step ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, name in enumerate(sorted_names):
        data = logs[name]
        color = COLORS[idx % len(COLORS)]
        label = short_label(name)
        final = data["final_bpb"]
        if final is not None:
            label += f" ({final:.4f})"
        ax.plot(data["val_steps"], data["val_bpbs"], marker="o", markersize=2,
                label=label, color=color, linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Val BPB")
    ax.set_title("Record Runs: Val BPB over Steps")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "records_val_bpb_vs_step.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Val BPB vs Wall Time ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, name in enumerate(sorted_names):
        data = logs[name]
        color = COLORS[idx % len(COLORS)]
        label = short_label(name)
        final = data["final_bpb"]
        if final is not None:
            label += f" ({final:.4f})"
        times_s = [t / 1000 for t in data["val_times"]]
        ax.plot(times_s, data["val_bpbs"], marker="o", markersize=2,
                label=label, color=color, linewidth=1.2)

    ax.set_xlabel("Wall Time (s)")
    ax.set_ylabel("Val BPB")
    ax.set_title("Record Runs: Val BPB over Time")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "records_val_bpb_vs_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Summary table ──
    path = out_dir / "records_summary.txt"
    with open(path, "w") as f:
        f.write(f"{'Run':<55} {'Final BPB':>10} {'Steps':>8}\n")
        f.write("-" * 77 + "\n")
        for name in sorted_names:
            data = logs[name]
            bpb = data["final_bpb"]
            if bpb is None and data["val_bpbs"]:
                bpb = data["val_bpbs"][-1]
            bpb_s = f"{bpb:.4f}" if bpb is not None else "N/A"
            steps = max(data["val_steps"]) if data["val_steps"] else 0
            f.write(f"{short_label(name):<55} {bpb_s:>10} {steps:>8}\n")
    print(f"  Saved {path}")
    print("Done.")


if __name__ == "__main__":
    main()
