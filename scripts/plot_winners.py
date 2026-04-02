#!/usr/bin/env python3
"""Generate focused comparison plots for the best architecture experiments."""

import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("experiment_logs_scaled")
OUT_DIR = LOG_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_RE = re.compile(
    r"step:(\d+)/\d+\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:(\d+)ms"
)

def parse_log(path):
    text = path.read_text()
    steps, bpbs, times = [], [], []
    for m in VAL_RE.finditer(text):
        steps.append(int(m.group(1)))
        bpbs.append(float(m.group(3)))
        times.append(int(m.group(4)) / 1000)
    return steps, bpbs, times


# ── 1. Architecture Winners: Top experiments by final val_bpb ──
WINNERS = [
    ("p6_shallow_wide",    "4L/MLP4x (shallow wide)", "#e41a1c"),
    ("p2_trigram_hash",    "Trigram Hash Embed",       "#377eb8"),
    ("p6_swiglu_mlp3x",   "SwiGLU + MLP3x",          "#4daf4a"),
    ("p6_leaky_relu2",    "LeakyReLU²",               "#984ea3"),
    ("p2_alpha_baseline",  "Baseline (6L/MLP2x)",     "#999999"),
    ("p6_mlp3x",          "MLP3x",                    "#ff7f00"),
    ("p2_local_conv3",    "Causal Conv (k=3)",        "#a65628"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Architecture Winners — Val BPB Comparison (Scaled: 90s Training)", fontsize=13, fontweight="bold")

for name, label, color in WINNERS:
    log = LOG_DIR / f"{name}.log"
    if not log.exists():
        continue
    steps, bpbs, times = parse_log(log)
    if not steps:
        continue
    ax1.plot(steps, bpbs, marker="o", markersize=3, label=f"{label} ({bpbs[-1]:.4f})", color=color, linewidth=2)
    ax2.plot(times, bpbs, marker="o", markersize=3, label=f"{label} ({bpbs[-1]:.4f})", color=color, linewidth=2)

for ax, xlabel in [(ax1, "Step"), (ax2, "Wall Time (s)")]:
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Val BPB")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.9, 3.0)

ax1.set_title("Val BPB vs Step")
ax2.set_title("Val BPB vs Wall Time")
plt.tight_layout()
fig.savefig(OUT_DIR / "architecture_winners.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'architecture_winners.png'}")


# ── 2. MLP Expansion Deep Dive ──
MLP_RUNS = [
    ("p2_alpha_baseline",   "Baseline (6L/MLP2x)",     "#999999"),
    ("p6_mlp3x",           "MLP3x (6L)",               "#377eb8"),
    ("p6_leaky_relu2_mlp3x","LeakyReLU² + MLP3x (6L)", "#4daf4a"),
    ("p6_swiglu_mlp3x",    "SwiGLU + MLP3x (6L)",     "#e41a1c"),
    ("p6_mlp4x",           "MLP4x (6L)",               "#ff7f00"),
    ("p6_mlp4x_4h",        "MLP4x + 4 heads (6L)",    "#984ea3"),
    ("p6_shallow_wide",    "MLP4x (4L, shallow wide)", "#a65628"),
    ("p6_deep_narrow",     "MLP2x (8L, deep narrow)",  "#f781bf"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MLP Expansion Comparison — Depth vs Width Tradeoff", fontsize=13, fontweight="bold")

for name, label, color in MLP_RUNS:
    log = LOG_DIR / f"{name}.log"
    if not log.exists():
        continue
    steps, bpbs, times = parse_log(log)
    if not steps:
        continue
    ax1.plot(steps, bpbs, marker="o", markersize=3, label=f"{label} ({bpbs[-1]:.4f})", color=color, linewidth=1.5)
    ax2.plot(times, bpbs, marker="o", markersize=3, label=f"{label} ({bpbs[-1]:.4f})", color=color, linewidth=1.5)

for ax, xlabel in [(ax1, "Step"), (ax2, "Wall Time (s)")]:
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Val BPB")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

ax1.set_title("Val BPB vs Step (per-step efficiency)")
ax2.set_title("Val BPB vs Wall Time (actual budget)")
plt.tight_layout()
fig.savefig(OUT_DIR / "mlp_expansion_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'mlp_expansion_comparison.png'}")


# ── 3. TTT/LoRA vs FFT (training curves identical, note this) ──
TTT_RUNS = [
    ("p2_alpha_baseline",   "Baseline (no TTT)",  "#999999"),
    ("p3_lora_rank16",      "LoRA r16",           "#e41a1c"),
    ("p2_ttt_bias_only",    "Bias TTT",           "#377eb8"),
    ("p2_fft_last2",        "FFT last 2L",        "#4daf4a"),
    ("p3_fft_last4",        "FFT last 4L",        "#984ea3"),
    ("p4_fft_all",          "FFT all layers",     "#ff7f00"),
    ("p2_ttt_chunk128",     "LoRA chunk128",      "#a65628"),
    ("p2_ttt_multistep3",   "LoRA 3-step",        "#f781bf"),
    ("p3_lora_rank32",      "LoRA r32",           "#e7298a"),
]

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("TTT Variants — Training Curves (Note: Identical training, diff only at eval-time TTT)", fontsize=11, fontweight="bold")

for name, label, color in TTT_RUNS:
    log = LOG_DIR / f"{name}.log"
    if not log.exists():
        continue
    steps, bpbs, times = parse_log(log)
    if not steps:
        continue
    ax.plot(steps, bpbs, marker="o", markersize=3, label=f"{label} ({bpbs[-1]:.4f})", color=color, linewidth=1.5)

ax.set_xlabel("Step")
ax.set_ylabel("Val BPB")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title("Val BPB vs Step — Training curves overlap because TTT mode only affects post-training eval")
plt.tight_layout()
fig.savefig(OUT_DIR / "ttt_lora_vs_fft.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'ttt_lora_vs_fft.png'}")


# ── 4. Bar chart of final val_bpb ──
ALL_RUNS = {}
for log in sorted(LOG_DIR.glob("*.log")):
    steps, bpbs, _ = parse_log(log)
    if steps and bpbs[-1] < 10:  # skip NaN/bogus
        ALL_RUNS[log.stem] = bpbs[-1]

# Sort by val_bpb
sorted_runs = sorted(ALL_RUNS.items(), key=lambda x: x[1])
names = [r[0] for r in sorted_runs]
vals = [r[1] for r in sorted_runs]

# Color by category
def get_color(name):
    if "shallow_wide" in name or "deep_narrow" in name or "mlp" in name or "swiglu" in name or "leaky" in name:
        return "#e41a1c"  # red = MLP/depth
    if "ttt" in name or "lora" in name or "fft" in name or "bias" in name:
        return "#377eb8"  # blue = TTT variants
    if "trigram" in name or "bigram" in name or "hash" in name:
        return "#4daf4a"  # green = embedding
    if "baseline" in name:
        return "#999999"  # grey = baseline
    return "#ff7f00"  # orange = other

colors = [get_color(n) for n in names]

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(names))
bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5)

# Add value labels
for i, (v, n) in enumerate(zip(vals, names)):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=7)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Final Val BPB (lower = better)")
ax.set_title("All Experiments — Final Val BPB (Scaled 90s runs)", fontweight="bold")
ax.invert_yaxis()
ax.axvline(x=ALL_RUNS.get("p2_alpha_baseline", 2.27), color="grey", linestyle="--", alpha=0.5, label="baseline")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e41a1c", label="MLP/Depth variants"),
    Patch(facecolor="#377eb8", label="TTT variants (same training)"),
    Patch(facecolor="#4daf4a", label="Embedding variants"),
    Patch(facecolor="#999999", label="Baselines"),
    Patch(facecolor="#ff7f00", label="Other"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "final_bpb_bar_chart.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'final_bpb_bar_chart.png'}")

print("\nDone — all focused comparison plots generated.")
