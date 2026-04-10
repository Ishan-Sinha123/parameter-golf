#!/usr/bin/env python3
"""Generate comprehensive experiment report with plots."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
import os

OUT_DIR = "/home/azureuser/parameter-golf/report_plots"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# ============================================================
# DATA: Scaled ablations (6L/384d, 90s, single GPU)
# ============================================================
scaled_data = [
    ("4L/MLP4x\n(shallow wide)", 2.0058, 150, "depth", True),
    ("Trigram Hash", 2.2104, 110, "embed", True),
    ("SwiGLU+MLP3x", 2.2455, 120, "activation", True),
    ("LoRA r16", 2.2635, 110, "ttt", False),
    ("Bias TTT", 2.2636, 110, "ttt", False),
    ("FFT all", 2.2643, 110, "ttt", False),
    ("FFT last2", 2.2647, 110, "ttt", False),
    ("LeakyReLU²", 2.2651, 110, "activation", True),
    ("Chunk128", 2.2665, 110, "ttt", False),
    ("MultiStep3", 2.2667, 110, "ttt", False),
    ("FFT last4", 2.2669, 110, "ttt", False),
    ("FFT2+MS3", 2.2671, 110, "ttt", False),
    ("Baseline", 2.2729, 110, "baseline", False),
    ("MLP3x", 2.2820, 110, "depth", False),
    ("Causal Conv", 2.2861, 110, "other", False),
    ("16Q/4KV GQA", 2.3110, 110, "other", False),
    ("LN inv_sqrt", 2.3174, 110, "other", False),
    ("LoRA r32", 2.3239, 100, "ttt", False),
    ("16Q/8KV", 2.3315, 120, "other", False),
    ("LeakyReLU²+MLP3x", 2.3380, 100, "activation", False),
    ("MLP4x (6L)", 2.3393, 100, "depth", False),
    ("MLP4x+4h", 2.3666, 90, "depth", False),
    ("Residual Gated", 2.4063, 110, "other", False),
    ("Deep Narrow 8L", 2.5372, 80, "depth", False),
    ("Gram N-S Opt", 2.8827, 110, "other", False),
]

# ============================================================
# DATA: Full-scale experiments (SOTA fork, 8xH100, 90s)
# ============================================================

# Phase 1: Feature isolation (clean GPUs)
phase1 = [
    ("SwiGLU", 4.148, 607, 148, "clean"),
    ("SwiGLU+Trigram", 4.139, 604, 149, "clean"),
    ("LeakyReLU²", 6.622, 564, 160, "clean"),
    ("Trigram Hash", 6.891, 562, 160, "clean"),
    ("Baseline", 6.972, 564, 160, "clean"),
]

# Phase 2: Depth/width (clean GPUs)
phase2 = [
    ("7L/MLP4x", 2.350, 814, 111, "clean"),
    ("8L/MLP4x", 5.136, 713, 126, "clean"),
    ("9L/MLP3x", 5.090, 691, 130, "clean"),
    ("9L/MLP4x", 6.405, 636, 142, "clean"),
]

# Phase 3: Combined (mixed hardware)
phase3_clean = [
    ("8L/MLP4x+SwiGLU+Tri", 3.904, 637, 142, "clean"),
    ("9L/MLP4x+SwiGLU+Tri", 4.094, 695, 130, "clean"),
]
phase3_degraded = [
    ("11L+SwiGLU+Tri", 3.413, 254, 355, "degraded"),
    ("9L/MLP4x+Leaky+Tri", 3.429, 141, 645, "degraded"),
]

# Phase 4: TTT (degraded GPUs, ~240 steps)
phase4_degraded = [
    ("LoRA r16 QVK", 3.500, 244, 370),
    ("Bias TTT", 3.501, 242, 373),
    ("LoRA r8", 3.505, 243, 372),
    ("LoRA r16 QV+MLP", 3.515, 240, 377),
    ("LoRA r16 QVK+MLP", 3.522, 244, 370),
    ("LoRA r16 (QV)", 3.533, 241, 374),
    ("LoRA r32", 3.558, 246, 367),
    ("FFT last2", 3.570, 245, 369),
    ("LoRA r4", 3.577, 251, 359),
]

# Autoresearch data (different dataset — directional only)
autoresearch = [
    ("Muon LR 0.03", 0.9839, "keep"),
    ("TTT 1 SGD", 0.9840, "discard"),
    ("TTT 3 steps", 0.9841, "discard"),
    ("WD 0.3", 0.9843, "discard"),
    ("Cosine warmdown", 0.9843, "discard"),
    ("Softcap 20", 0.9845, "discard"),
    ("Muon LR 0.02", 0.9848, "discard"),
    ("Embed LR 0.8", 0.9848, "discard"),
    ("Unembed LR 0.008", 0.9851, "keep"),
    ("Adam β1=0.9", 0.9855, "discard"),
    ("DEVICE_BS=64", 0.9855, "discard"),
    ("FINAL_LR_FRAC=0.1", 0.9858, "keep"),
    ("Batch 2^18", 0.9859, "keep"),
    ("Embed LR 1.0", 0.9865, "discard"),
    ("VE all layers", 0.9865, "discard"),
    ("Batch 2^17", 0.9864, "discard"),
    ("MLP 3x", 0.9872, "discard"),
    ("7L 768w", 0.9872, "discard"),
    ("All full-context", 0.9878, "discard"),
    ("RMSNorm pre-MLP", 0.9881, "discard"),
    ("Muon LR 0.06", 0.9881, "discard"),
    ("GQA 3 KV", 0.9897, "discard"),
    ("5% warmup", 0.9896, "discard"),
    ("WD 30%", 0.9901, "discard"),
    ("No WD", 0.9902, "discard"),
    ("Softcap 30", 0.9911, "discard"),
    ("No VE", 0.9931, "discard"),
    ("HEAD_DIM=64", 0.9947, "discard"),
    ("Parallel attn+MLP", 1.0038, "discard"),
    ("6L 768w+SwiGLU", 0.9987, "keep"),
    ("SwiGLU", 1.0097, "keep"),
    ("Baseline", 1.0196, "keep"),
    ("4L 768w", 1.0133, "discard"),
    ("6L 960w", 1.0628, "discard"),
    ("6L 896w", 1.0365, "discard"),
]

# ============================================================
# PLOT 1: Master overview — Scaled ablations ranked bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

names = [d[0] for d in scaled_data]
bpbs = [d[1] for d in scaled_data]
cats = [d[3] for d in scaled_data]
winners = [d[4] for d in scaled_data]

cat_colors = {
    'depth': '#e74c3c',
    'embed': '#2ecc71',
    'activation': '#f39c12',
    'ttt': '#3498db',
    'baseline': '#95a5a6',
    'other': '#9b59b6',
}
colors = [cat_colors[c] for c in cats]
edge_colors = ['gold' if w else 'none' for w in winners]
linewidths = [2.5 if w else 0 for w in winners]

bars = ax.barh(range(len(names)), bpbs, color=colors, edgecolor=edge_colors, linewidth=linewidths)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Val BPB (lower is better)', fontsize=12)
ax.set_title('Scaled Ablations: 30 Experiments Ranked\n(6L/384d, 90s training, single GPU)', fontsize=14, fontweight='bold')
ax.axvline(x=2.2729, color='gray', linestyle='--', alpha=0.7, label='Baseline')

# Legend
patches = [
    mpatches.Patch(color=cat_colors['depth'], label='Depth/Width'),
    mpatches.Patch(color=cat_colors['embed'], label='Embedding'),
    mpatches.Patch(color=cat_colors['activation'], label='Activation'),
    mpatches.Patch(color=cat_colors['ttt'], label='TTT'),
    mpatches.Patch(color=cat_colors['other'], label='Other'),
    mpatches.Patch(facecolor='white', edgecolor='gold', linewidth=2, label='Winner'),
]
ax.legend(handles=patches, loc='lower right', fontsize=9)

for i, (name, bpb, _, _, _) in enumerate(scaled_data):
    ax.text(bpb + 0.01, i, f'{bpb:.4f}', va='center', fontsize=8, color='#333')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_scaled_ablations_ranked.png', bbox_inches='tight')
plt.close()
print("Plot 1 done: scaled ablations ranked")

# ============================================================
# PLOT 2: Full-scale Phase 1 — Activation comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Training BPB
names_p1 = [d[0] for d in phase1]
train_bpbs = [d[2] for d in phase1]  # steps as proxy — let's use training bpb from logs
training_bpb_p1 = [1.599, 1.623, 1.632, 1.639, 1.632]
final_bpb_p1 = [d[1] for d in phase1]
ms_p1 = [d[3] for d in phase1]
steps_p1 = [d[2] for d in phase1]

colors_p1 = ['#2ecc71', '#27ae60', '#f39c12', '#9b59b6', '#95a5a6']
bars1 = ax1.bar(names_p1, training_bpb_p1, color=colors_p1, edgecolor='white', linewidth=0.5)
ax1.set_ylabel('Training BPB (lower = better)')
ax1.set_title('Phase 1: Training BPB @ 90s', fontweight='bold')
for i, v in enumerate(training_bpb_p1):
    ax1.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=9)
ax1.set_ylim(1.55, 1.68)

# Right: Final post-quant BPB
bars2 = ax2.bar(names_p1, final_bpb_p1, color=colors_p1, edgecolor='white', linewidth=0.5)
ax2.set_ylabel('Final BPB (post-GPTQ, lower = better)')
ax2.set_title('Phase 1: Post-Quantization BPB', fontweight='bold')
for i, v in enumerate(final_bpb_p1):
    ax2.text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=9)

# Add step counts as secondary info
for i, (s, ms) in enumerate(zip(steps_p1, ms_p1)):
    ax2.text(i, 0.3, f'{s} steps\n{ms}ms/step', ha='center', fontsize=8, color='white', fontweight='bold')

fig.suptitle('Full-Scale Phase 1: Activation Functions on SOTA Stack (8×H100, 90s)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_phase1_activations.png', bbox_inches='tight')
plt.close()
print("Plot 2 done: phase 1 activations")

# ============================================================
# PLOT 3: Full-scale Phase 2 — Depth/Width tradeoff
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

names_p2 = [d[0] for d in phase2]
final_p2 = [d[1] for d in phase2]
steps_p2 = [d[2] for d in phase2]
ms_p2 = [d[3] for d in phase2]

colors_p2 = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

# Left: ms/step
axes[0].bar(names_p2, ms_p2, color=colors_p2)
axes[0].set_ylabel('ms/step')
axes[0].set_title('Step Speed', fontweight='bold')
for i, v in enumerate(ms_p2):
    axes[0].text(i, v + 1, f'{v}ms', ha='center', fontsize=10)

# Middle: Steps completed
axes[1].bar(names_p2, steps_p2, color=colors_p2)
axes[1].set_ylabel('Steps in 90s')
axes[1].set_title('Training Steps', fontweight='bold')
for i, v in enumerate(steps_p2):
    axes[1].text(i, v + 5, str(v), ha='center', fontsize=10)

# Right: Final BPB
axes[2].bar(names_p2, final_p2, color=colors_p2)
axes[2].set_ylabel('Final BPB (post-GPTQ)')
axes[2].set_title('Final BPB', fontweight='bold')
for i, v in enumerate(final_p2):
    axes[2].text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=10)

fig.suptitle('Full-Scale Phase 2: Depth vs Width — Fewer Layers = Faster Steps = Better Loss\n(8×H100, 90s, SOTA stack)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_phase2_depth_width.png', bbox_inches='tight')
plt.close()
print("Plot 3 done: phase 2 depth/width")

# ============================================================
# PLOT 4: Steps vs Final BPB scatter (full-scale)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Combine all full-scale clean experiments
all_clean = []
for name, bpb, steps, ms, cond in phase1:
    all_clean.append((name, bpb, steps, ms, 'P1: Activation'))
for name, bpb, steps, ms, cond in phase2:
    all_clean.append((name, bpb, steps, ms, 'P2: Depth/Width'))
for name, bpb, steps, ms, cond in phase3_clean:
    all_clean.append((name, bpb, steps, ms, 'P3: Combined'))

phase_colors = {
    'P1: Activation': '#e74c3c',
    'P2: Depth/Width': '#3498db',
    'P3: Combined': '#2ecc71',
}

for name, bpb, steps, ms, phase in all_clean:
    c = phase_colors[phase]
    ax.scatter(steps, bpb, c=c, s=120, zorder=5, edgecolors='white', linewidth=1)
    ax.annotate(name, (steps, bpb), textcoords="offset points",
                xytext=(8, 5), fontsize=8, color='#333')

for phase, color in phase_colors.items():
    ax.scatter([], [], c=color, s=80, label=phase)
ax.legend(fontsize=10)

ax.set_xlabel('Training Steps (in 90s)', fontsize=12)
ax.set_ylabel('Final BPB (post-GPTQ, lower = better)', fontsize=12)
ax.set_title('Full-Scale: More Steps → Better BPB\n(Clean GPU experiments only)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_steps_vs_bpb_scatter.png', bbox_inches='tight')
plt.close()
print("Plot 4 done: steps vs BPB scatter")

# ============================================================
# PLOT 5: TTT comparison (Phase 4, degraded GPU)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

names_p4 = [d[0] for d in phase4_degraded]
bpbs_p4 = [d[1] for d in phase4_degraded]
colors_p4 = ['#2ecc71' if b == min(bpbs_p4) else '#3498db' for b in bpbs_p4]

bars = ax.barh(range(len(names_p4)), bpbs_p4, color=colors_p4)
ax.set_yticks(range(len(names_p4)))
ax.set_yticklabels(names_p4)
ax.invert_yaxis()
ax.set_xlabel('Final BPB (lower = better)')
ax.set_title('Phase 4: TTT Strategy Comparison\n(Degraded GPUs, ~240 steps, same base model)', fontsize=13, fontweight='bold')
ax.set_xlim(3.45, 3.62)

for i, v in enumerate(bpbs_p4):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_ttt_comparison.png', bbox_inches='tight')
plt.close()
print("Plot 5 done: TTT comparison")

# ============================================================
# PLOT 6: Autoresearch hyperparameter sweep
# ============================================================
# Sort by BPB
auto_sorted = sorted(autoresearch, key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(14, 10))

names_ar = [d[0] for d in auto_sorted]
bpbs_ar = [d[1] for d in auto_sorted]
status_ar = [d[2] for d in auto_sorted]
colors_ar = ['#2ecc71' if s == 'keep' else '#e74c3c' for s in status_ar]

bars = ax.barh(range(len(names_ar)), bpbs_ar, color=colors_ar, alpha=0.8)
ax.set_yticks(range(len(names_ar)))
ax.set_yticklabels(names_ar, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Val BPB (lower = better)')
ax.set_title('Autoresearch: 36 Hyperparameter Experiments\n(Single GPU, climbmix-400b dataset — directional only)', fontsize=13, fontweight='bold')

baseline_bpb = 1.0196
ax.axvline(x=baseline_bpb, color='gray', linestyle='--', alpha=0.7)
ax.text(baseline_bpb + 0.001, 1, 'Baseline', fontsize=9, color='gray')

patches = [
    mpatches.Patch(color='#2ecc71', label='Keep (improvement)'),
    mpatches.Patch(color='#e74c3c', label='Discard (no improvement)'),
]
ax.legend(handles=patches, loc='lower right')

for i, v in enumerate(bpbs_ar):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7, color='#333')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_autoresearch_sweep.png', bbox_inches='tight')
plt.close()
print("Plot 6 done: autoresearch sweep")

# ============================================================
# PLOT 7: Best combo training curve
# ============================================================
# Parse best combo log
best_combo_steps = []
best_combo_bpb = []
with open('/home/azureuser/parameter-golf/experiment_logs_fullscale/best_combo_7L_swiglu_ttt_qvk_r16.log') as f:
    for line in f:
        m = re.search(r'step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', line)
        if m:
            best_combo_steps.append(int(m.group(1)))
            best_combo_bpb.append(float(m.group(2)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(best_combo_steps, best_combo_bpb, 'b-o', markersize=4, linewidth=2, label='7L/MLP4x + SwiGLU (best combo)')
ax.axhline(y=1.2244, color='gray', linestyle='--', alpha=0.7, label='Competition Baseline (1.2244)')
ax.axhline(y=1.1147, color='red', linestyle='--', alpha=0.7, label='SOTA PR #1019 (1.1147)')

# Mark where it crosses baseline
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Val BPB (lower = better)', fontsize=12)
ax.set_title('Best Combo: 7L/MLP4x + SwiGLU + LoRA TTT\n(8×H100, 600s training, 99.5 ms/step)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

# Add annotations
last_step = best_combo_steps[-1]
last_bpb = best_combo_bpb[-1]
ax.annotate(f'Step {last_step}: {last_bpb:.4f} BPB\n(training only, no quant)',
           (last_step, last_bpb), textcoords="offset points",
           xytext=(-120, 20), fontsize=9,
           arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_best_combo_curve.png', bbox_inches='tight')
plt.close()
print("Plot 7 done: best combo training curve")

# ============================================================
# PLOT 8: Grand summary — all 3 experiment tracks compared
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scaled
ax = axes[0]
top_scaled = scaled_data[:7] + [scaled_data[12]]  # top 7 + baseline
names_s = [d[0].replace('\n', ' ') for d in top_scaled]
bpbs_s = [d[1] for d in top_scaled]
cols_s = [cat_colors[d[3]] for d in top_scaled]
ax.barh(range(len(names_s)), bpbs_s, color=cols_s)
ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Val BPB')
ax.set_title('Scaled (6L/384d)\n30 experiments', fontweight='bold')
ax.axvline(x=2.2729, color='gray', linestyle='--', alpha=0.5)

# Full-scale clean
ax = axes[1]
all_fullscale = phase1 + phase2 + phase3_clean
all_fullscale.sort(key=lambda x: x[1])
top_fs = all_fullscale[:8]
names_fs = [d[0] for d in top_fs]
bpbs_fs = [d[1] for d in top_fs]
ax.barh(range(len(names_fs)), bpbs_fs, color='#3498db')
ax.set_yticks(range(len(names_fs)))
ax.set_yticklabels(names_fs, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Final BPB (post-GPTQ)')
ax.set_title('Full-Scale (SOTA fork)\n26 experiments', fontweight='bold')
for i, v in enumerate(bpbs_fs):
    ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)

# Autoresearch
ax = axes[2]
auto_top = sorted(autoresearch, key=lambda x: x[1])[:8]
names_a = [d[0] for d in auto_top]
bpbs_a = [d[1] for d in auto_top]
ax.barh(range(len(names_a)), bpbs_a, color='#2ecc71')
ax.set_yticks(range(len(names_a)))
ax.set_yticklabels(names_a, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Val BPB')
ax.set_title('Autoresearch (single GPU)\n36 experiments', fontweight='bold')
for i, v in enumerate(bpbs_a):
    ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=8)

fig.suptitle('Grand Summary: Top Results Across All 3 Experiment Tracks (~92 total experiments)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_grand_summary.png', bbox_inches='tight')
plt.close()
print("Plot 8 done: grand summary")

# ============================================================
# PLOT 9: Quantization gap analysis
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Data: (name, training_bpb, post_ema_bpb, final_bpb)
quant_data = [
    ("P1 Baseline", 1.632, 3.639, 6.972),
    ("P1 LeakyReLU²", 1.632, 3.634, 6.622),
    ("P1 SwiGLU", 1.599, 2.174, 4.148),
    ("P1 SwiGLU+Tri", 1.623, 2.445, 4.139),
    ("P2 7L/MLP4x", 1.489, 1.749, 2.350),
    ("P2 8L/MLP4x", 1.534, 2.104, 5.136),
    ("P2 9L/MLP3x", 1.546, 2.215, 5.090),
    ("P2 9L/MLP4x", 1.579, 2.845, 6.405),
]

x = np.arange(len(quant_data))
width = 0.25

train_bpbs = [d[1] for d in quant_data]
ema_bpbs = [d[2] for d in quant_data]
final_bpbs = [d[3] for d in quant_data]

b1 = ax.bar(x - width, train_bpbs, width, label='Training BPB', color='#3498db')
b2 = ax.bar(x, ema_bpbs, width, label='Post-EMA BPB', color='#f39c12')
b3 = ax.bar(x + width, final_bpbs, width, label='Post-GPTQ Final BPB', color='#e74c3c')

ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in quant_data], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('BPB')
ax.set_title('Quantization Gap: Training → EMA → GPTQ\n(Short training makes GPTQ unreliable — 7L/MLP4x has smallest gap)', fontsize=13, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/09_quantization_gap.png', bbox_inches='tight')
plt.close()
print("Plot 9 done: quantization gap")

print(f"\nAll plots saved to {OUT_DIR}/")
