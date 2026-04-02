#!/usr/bin/env python3
"""Plot full-scale experiment results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path

LOG_DIR = Path("/workspace/parameter-golf/experiment_logs_fullscale")
OUT_DIR = LOG_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Collect results ──
results = {}
for log in sorted(LOG_DIR.glob("p*.log")):
    name = log.stem
    text = log.read_bytes().decode('utf-8', errors='replace')

    # Extract metrics
    steps_match = re.findall(r'step:(\d+)/20000', text)
    steps = int(steps_match[-1]) if steps_match else 0

    val_bpb_matches = re.findall(r'val_bpb:([\d.]+)', text)
    # First val_bpb after step 0 is initial, last before stopping is best training
    train_bpb = float(val_bpb_matches[-1]) if len(val_bpb_matches) > 1 else None

    int6_match = re.search(r'final_int6_sliding_window_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    int6_bpb = float(int6_match.group(1)) if int6_match else None

    step_avg_matches = re.findall(r'step_avg:([\d.]+)ms', text)
    step_avg = float(step_avg_matches[-1]) if step_avg_matches else None

    params_match = re.search(r'model_params:(\d+)', text)
    params = int(params_match.group(1)) if params_match else None

    # Extract training curve (step, val_bpb pairs)
    curve = []
    for m in re.finditer(r'step:(\d+)/20000 val_loss:[\d.]+ val_bpb:([\d.]+)', text):
        curve.append((int(m.group(1)), float(m.group(2))))

    # Phase
    phase = int(name[1])

    results[name] = {
        'steps': steps, 'train_bpb': train_bpb, 'int6_bpb': int6_bpb,
        'step_avg': step_avg, 'params': params, 'curve': curve, 'phase': phase
    }

# ── Color schemes ──
phase_colors = {1: '#2196F3', 2: '#4CAF50', 3: '#FF9800'}
run_colors = {
    'p1_baseline': '#9E9E9E', 'p1_swiglu': '#2196F3', 'p1_swiglu_trigram': '#03A9F4',
    'p1_trigram_hash': '#FF9800', 'p1_leaky_relu2': '#F44336',
    'p2_9L_mlp4x': '#4CAF50', 'p2_8L_mlp4x': '#8BC34A',
    'p2_9L_mlp3x': '#CDDC39', 'p2_7L_mlp4x': '#009688',
    'p3_9L_mlp4x_swiglu_trigram': '#FF5722', 'p3_8L_mlp4x_swiglu_trigram': '#E91E63',
    'p3_11L_swiglu_trigram': '#9C27B0', 'p3_9L_mlp4x_leaky_trigram': '#795548',
}

plt.style.use('dark_background')
fig_bg = '#1a1a2e'
ax_bg = '#16213e'

# ═══════════════════════════════════════════════
# PLOT 1: Bar chart — Training BPB by experiment
# ═══════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor=fig_bg)

completed = {k: v for k, v in results.items() if v['train_bpb'] is not None}
names = sorted(completed.keys())
train_bpbs = [completed[n]['train_bpb'] for n in names]
colors = [run_colors.get(n, '#607D8B') for n in names]

ax1.set_facecolor(ax_bg)
bars = ax1.barh(range(len(names)), train_bpbs, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels([n.replace('_', '\n', 1) for n in names], fontsize=8)
ax1.set_xlabel('Last Validation BPB (training)', fontsize=11)
ax1.set_title('Training BPB (lower = better)', fontsize=13, fontweight='bold', color='white')
ax1.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, train_bpbs)):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9, color='white')

# Best line
best_train = min(train_bpbs)
ax1.axvline(best_train, color='#00E676', linestyle='--', alpha=0.7, linewidth=1)
ax1.text(best_train, -0.5, f'best: {best_train:.4f}', color='#00E676', fontsize=9, ha='center')

# ═══════════════════════════════════════════════
# PLOT 2: Bar chart — Post-quantization int6 BPB
# ═══════════════════════════════════════════════
has_int6 = {k: v for k, v in results.items() if v['int6_bpb'] is not None}
names_i6 = sorted(has_int6.keys())
int6_bpbs = [has_int6[n]['int6_bpb'] for n in names_i6]
colors_i6 = [run_colors.get(n, '#607D8B') for n in names_i6]

ax2.set_facecolor(ax_bg)
bars2 = ax2.barh(range(len(names_i6)), int6_bpbs, color=colors_i6, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(names_i6)))
ax2.set_yticklabels([n.replace('_', '\n', 1) for n in names_i6], fontsize=8)
ax2.set_xlabel('Final int6+LZMA BPB (post-quantization)', fontsize=11)
ax2.set_title('Post-Quantization BPB (lower = better)', fontsize=13, fontweight='bold', color='white')
ax2.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars2, int6_bpbs)):
    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9, color='white')

best_int6 = min(int6_bpbs)
ax2.axvline(best_int6, color='#00E676', linestyle='--', alpha=0.7, linewidth=1)
ax2.text(best_int6, -0.5, f'best: {best_int6:.3f}', color='#00E676', fontsize=9, ha='center')

plt.tight_layout()
plt.savefig(OUT_DIR / 'bpb_comparison.png', dpi=150, bbox_inches='tight', facecolor=fig_bg)
plt.close()
print(f"Saved: {OUT_DIR / 'bpb_comparison.png'}")

# ═══════════════════════════════════════════════
# PLOT 3: Training curves
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=fig_bg)

for phase_idx, phase_num in enumerate([1, 2, 3]):
    ax = axes[phase_idx]
    ax.set_facecolor(ax_bg)
    phase_runs = {k: v for k, v in results.items() if v['phase'] == phase_num and v['curve']}

    for name, data in sorted(phase_runs.items()):
        curve = data['curve']
        if len(curve) < 2:
            continue
        steps_c = [c[0] for c in curve]
        bpbs = [c[1] for c in curve]
        color = run_colors.get(name, '#607D8B')
        label = name.replace(f'p{phase_num}_', '')
        ax.plot(steps_c, bpbs, marker='o', markersize=4, linewidth=2,
                color=color, label=label, alpha=0.9)

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Validation BPB', fontsize=11)
    ax.set_title(f'Phase {phase_num} Training Curves', fontsize=13, fontweight='bold', color='white')
    ax.legend(fontsize=8, loc='upper right', facecolor=ax_bg, edgecolor='gray')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight', facecolor=fig_bg)
plt.close()
print(f"Saved: {OUT_DIR / 'training_curves.png'}")

# ═══════════════════════════════════════════════
# PLOT 4: Steps vs BPB (efficiency scatter)
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7), facecolor=fig_bg)
ax.set_facecolor(ax_bg)

for name, data in sorted(results.items()):
    if data['steps'] == 0 or data['train_bpb'] is None:
        continue
    color = run_colors.get(name, '#607D8B')
    marker = {1: 'o', 2: 's', 3: 'D'}.get(data['phase'], 'x')
    ax.scatter(data['steps'], data['train_bpb'], c=color, marker=marker,
               s=120, edgecolors='white', linewidth=0.5, zorder=5)
    # Label
    short = name.replace(f"p{data['phase']}_", '')
    ax.annotate(short, (data['steps'], data['train_bpb']),
                textcoords="offset points", xytext=(8, 4), fontsize=7, color='white', alpha=0.9)

ax.set_xlabel('Steps completed in 90s', fontsize=12)
ax.set_ylabel('Best Validation BPB', fontsize=12)
ax.set_title('Efficiency: Steps vs Training Quality', fontsize=14, fontweight='bold', color='white')
ax.grid(True, alpha=0.2)

# Legend for phases
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=10, label='Phase 1', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50', markersize=10, label='Phase 2', linestyle='None'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF9800', markersize=10, label='Phase 3', linestyle='None'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper right', facecolor=ax_bg, edgecolor='gray')

plt.tight_layout()
plt.savefig(OUT_DIR / 'efficiency_scatter.png', dpi=150, bbox_inches='tight', facecolor=fig_bg)
plt.close()
print(f"Saved: {OUT_DIR / 'efficiency_scatter.png'}")

# ═══════════════════════════════════════════════
# PLOT 5: Quantization gap (training vs int6)
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7), facecolor=fig_bg)
ax.set_facecolor(ax_bg)

for name, data in sorted(results.items()):
    if data['train_bpb'] is None or data['int6_bpb'] is None:
        continue
    color = run_colors.get(name, '#607D8B')
    ax.scatter(data['train_bpb'], data['int6_bpb'], c=color, s=150,
               edgecolors='white', linewidth=0.8, zorder=5)
    short = name.replace(f"p{data['phase']}_", '')
    ax.annotate(short, (data['train_bpb'], data['int6_bpb']),
                textcoords="offset points", xytext=(8, 4), fontsize=7, color='white', alpha=0.9)

# Diagonal reference (perfect quantization = no gap)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, '--', color='gray', alpha=0.4, label='No quant gap')

ax.set_xlabel('Training BPB (last val)', fontsize=12)
ax.set_ylabel('Post-Quantization int6 BPB', fontsize=12)
ax.set_title('Quantization Gap: Training vs int6+LZMA', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=10, facecolor=ax_bg, edgecolor='gray')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR / 'quantization_gap.png', dpi=150, bbox_inches='tight', facecolor=fig_bg)
plt.close()
print(f"Saved: {OUT_DIR / 'quantization_gap.png'}")

print("\nAll plots saved to:", OUT_DIR)
