#!/usr/bin/env python3
"""Parse experiment logs and generate comparison plots.

Usage:
    python3 plot_experiments.py                              # defaults: experiment_logs/ -> experiment_plots/
    python3 plot_experiments.py --log-dir DIR --out-dir DIR   # custom dirs
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Log parsing ──────────────────────────────────────────────────────────────

VAL_RE = re.compile(
    r"step:(\d+)/\d+\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:(\d+)ms"
)
TRAIN_RE = re.compile(
    r"step:(\d+)/\d+\s+train_loss:([\d.]+)\s+train_time:(\d+)ms"
)
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text()
    val_steps, val_losses, val_bpbs, val_times = [], [], [], []
    train_steps, train_losses, train_times = [], [], []
    final_val_loss = final_val_bpb = None

    for m in VAL_RE.finditer(text):
        val_steps.append(int(m.group(1)))
        val_losses.append(float(m.group(2)))
        val_bpbs.append(float(m.group(3)))
        val_times.append(int(m.group(4)))

    for m in TRAIN_RE.finditer(text):
        train_steps.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
        train_times.append(int(m.group(3)))

    m = FINAL_RE.search(text)
    if m:
        final_val_loss = float(m.group(1))
        final_val_bpb = float(m.group(2))

    return dict(
        val_steps=val_steps, val_losses=val_losses, val_bpbs=val_bpbs, val_times=val_times,
        train_steps=train_steps, train_losses=train_losses, train_times=train_times,
        final_val_loss=final_val_loss, final_val_bpb=final_val_bpb,
    )


# ── Phase definitions ───────────────────────────────────────────────────────

PHASES = {
    "phase1": {
        "title": "Phase 1: Architecture Variants",
        "prefix": "p1_",
    },
    "phase2": {
        "title": "Phase 2: TTT + Architecture Mods",
        "prefix": "p2_",
    },
    "phase3": {
        "title": "Phase 3: LoRA vs FFT Showdown",
        "prefix": "p3_",
    },
    "phase4": {
        "title": "Phase 4: Heavy Infrastructure",
        "prefix": "p4_",
    },
    "phase5": {
        "title": "Phase 5: Extended TTT",
        "prefix": "p5_",
    },
    "phase6": {
        "title": "Phase 6: MLP Expansion + Activation",
        "prefix": "p6_",
    },
    "phase7": {
        "title": "Phase 7: Layer Importance",
        "prefix": "p7_",
    },
    "phase8": {
        "title": "Phase 8: Gram Newton-Schulz",
        "prefix": "p8_",
    },
}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def label_from_name(name: str) -> str:
    """Strip phase prefix for cleaner legend labels."""
    for info in PHASES.values():
        if name.startswith(info["prefix"]):
            return name[len(info["prefix"]):]
    return name


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_phase(phase_key: str, logs: dict[str, dict], out_dir: Path):
    info = PHASES[phase_key]
    prefix = info["prefix"]
    runs = {k: v for k, v in logs.items() if k.startswith(prefix)}
    if not runs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(info["title"], fontsize=14, fontweight="bold")

    for idx, (name, data) in enumerate(sorted(runs.items())):
        color = COLORS[idx % len(COLORS)]
        label = label_from_name(name)

        # Val BPB over steps
        if data["val_steps"]:
            axes[0].plot(data["val_steps"], data["val_bpbs"], marker="o", markersize=3,
                         label=label, color=color)
        # Val BPB over wall time
        if data["val_times"]:
            axes[1].plot([t / 1000 for t in data["val_times"]], data["val_bpbs"],
                         marker="o", markersize=3, label=label, color=color)
        # Train loss over steps
        if data["train_steps"]:
            axes[2].plot(data["train_steps"], data["train_losses"],
                         alpha=0.7, label=label, color=color, linewidth=1)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Val BPB")
    axes[0].set_title("Val BPB vs Step")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Wall Time (s)")
    axes[1].set_ylabel("Val BPB")
    axes[1].set_title("Val BPB vs Time")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Train Loss")
    axes[2].set_title("Train Loss vs Step")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{phase_key}_results.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_all_val_bpb(logs: dict[str, dict], out_dir: Path):
    """Single overlay plot of val BPB over time for every experiment."""
    runs_with_val = {k: v for k, v in logs.items() if v["val_steps"]}
    if not runs_with_val:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (name, data) in enumerate(sorted(runs_with_val.items())):
        color = COLORS[idx % len(COLORS)]
        times_s = [t / 1000 for t in data["val_times"]]
        ax.plot(times_s, data["val_bpbs"], marker="o", markersize=3,
                label=label_from_name(name), color=color)

    ax.set_xlabel("Wall Time (s)")
    ax.set_ylabel("Val BPB")
    ax.set_title("All Experiments: Val BPB over Time")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "experiments_val_bpb_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def write_summary_table(logs: dict[str, dict], out_dir: Path):
    """Write a text summary table with final metrics for each run."""
    rows = []
    for name, data in sorted(logs.items()):
        final_bpb = data["final_val_bpb"]
        if final_bpb is None and data["val_bpbs"]:
            final_bpb = data["val_bpbs"][-1]
        final_loss = data["final_val_loss"]
        if final_loss is None and data["val_losses"]:
            final_loss = data["val_losses"][-1]
        steps = max(data["val_steps"]) if data["val_steps"] else (max(data["train_steps"]) if data["train_steps"] else 0)
        rows.append((name, final_loss, final_bpb, steps))

    out_path = out_dir / "summary_table.txt"
    with open(out_path, "w") as f:
        f.write(f"{'Experiment':<30} {'Val Loss':>10} {'Val BPB':>10} {'Steps':>8}\n")
        f.write("-" * 62 + "\n")
        for name, loss, bpb, steps in sorted(rows, key=lambda r: r[2] if r[2] is not None else 999):
            loss_s = f"{loss:.4f}" if loss is not None else "N/A"
            bpb_s = f"{bpb:.4f}" if bpb is not None else "N/A"
            f.write(f"{name:<30} {loss_s:>10} {bpb_s:>10} {steps:>8}\n")
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--log-dir", default="experiment_logs", help="Directory with .log files")
    parser.add_argument("--out-dir", default="experiment_plots", help="Output directory for plots")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    print(f"Parsing {len(log_files)} log files from {log_dir}/")
    logs = {}
    for lf in log_files:
        name = lf.stem
        logs[name] = parse_log(lf)
        n_val = len(logs[name]["val_steps"])
        n_train = len(logs[name]["train_steps"])
        print(f"  {name}: {n_val} val points, {n_train} train points")

    print(f"\nGenerating plots in {out_dir}/")
    for phase_key in PHASES:
        plot_phase(phase_key, logs, out_dir)

    plot_all_val_bpb(logs, out_dir)
    write_summary_table(logs, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
