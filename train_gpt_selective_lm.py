"""train_gpt_selective_lm.py - Rho-1-style selective LM variant.

Reweights per-token cross-entropy loss by token "interestingness", measured as
a normalized unigram surprise computed once at startup from a sample of the
training shards. Common tokens (low surprise) get downweighted; rare tokens
get upweighted. This is a cheap proxy for the Rho-1 reference-model approach -
no reference model required, no extra forward passes per step.

Why this fits the 10-min budget: low-resource undertraining is dominated by
loss on common tokens that the model has already learned. Reweighting toward
rarer-but-still-learnable tokens routes more gradient signal to the cells of
the loss landscape that actually still have headroom.

Mechanic
--------
1. At startup, sample SELECTIVE_LM_SAMPLE_TOKENS tokens from the first training
   shard, compute unigram probs with +1 smoothing, take z-scored surprise.
2. weight_v = clip(1 + Z_SCALE * z, FLOOR, CAP) for v in vocab.
3. forward() computes per-token CE * weight[target], divided by weight.sum().
   In eval mode (model.eval()), forward falls back to plain mean CE so
   reported val_loss/val_bpb stay comparable.

New env vars
------------
  SELECTIVE_LM_ENABLED=1
  SELECTIVE_LM_WEIGHT_FLOOR=0.3
  SELECTIVE_LM_WEIGHT_CAP=3.0
  SELECTIVE_LM_Z_SCALE=0.5            how aggressively to scale z-scores
  SELECTIVE_LM_SAMPLE_TOKENS=5000000  startup sample size for unigram counts

Baseline knobs worth revisiting alongside selective LM
------------------------------------------------------
  GRAD_CLIP_NORM=0.5  was 0.3 - weighted loss has higher gradient variance
  WARMUP_STEPS=40     was 20  - let weight scaling settle before main loop
  MATRIX_LR=0.018     was 0.022 - effective LR is amplified on hard tokens

Run
---
  DATA_DIR=./data/ RUN_ID=selective_seed42 SEED=42 \
  TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 \
  SELECTIVE_LM_ENABLED=1 SELECTIVE_LM_WEIGHT_FLOOR=0.3 SELECTIVE_LM_WEIGHT_CAP=3.0 \
  SELECTIVE_LM_Z_SCALE=0.5 SELECTIVE_LM_SAMPLE_TOKENS=5000000 \
  GRAD_CLIP_NORM=0.5 WARMUP_STEPS=40 MATRIX_LR=0.018 \
  torchrun --nproc_per_node=8 train_gpt_selective_lm.py
"""
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_BASE_PATH = (
    Path(__file__).parent
    / "records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py"
)
_spec = importlib.util.spec_from_file_location("sota_baseline", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules["sota_baseline"] = base
_spec.loader.exec_module(base)


def _build_unigram_weights(h):
    floor = float(os.environ.get("SELECTIVE_LM_WEIGHT_FLOOR", "0.3"))
    cap = float(os.environ.get("SELECTIVE_LM_WEIGHT_CAP", "3.0"))
    z_scale = float(os.environ.get("SELECTIVE_LM_Z_SCALE", "0.5"))
    n_sample = int(os.environ.get("SELECTIVE_LM_SAMPLE_TOKENS", "5000000"))
    files = sorted(glob.glob(h.train_files))
    if not files:
        return torch.ones(h.vocab_size, dtype=torch.float32)
    mm = base._get_shard_memmap(Path(files[0]))
    n = min(n_sample, len(mm))
    sample = np.array(mm[:n], dtype=np.int64)
    counts = np.bincount(sample, minlength=h.vocab_size).astype(np.float64)
    probs = (counts + 1.0) / (counts.sum() + h.vocab_size)
    surprise = -np.log(probs)
    z = (surprise - surprise.mean()) / (surprise.std() + 1e-9)
    weights = np.clip(1.0 + z_scale * z, floor, cap).astype(np.float32)
    return torch.from_numpy(weights)


class SelectiveLMGPT(base.GPT):
    _SEL_WEIGHTS = None  # populated by the patched train_model

    def __init__(self, h):
        super().__init__(h)
        self.sel_enabled = bool(int(os.environ.get("SELECTIVE_LM_ENABLED", "1")))
        w = type(self)._SEL_WEIGHTS
        if w is None:
            w = torch.ones(h.vocab_size, dtype=torch.float32)
        self.register_buffer("sel_weights", w.clone(), persistent=False)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = target_ids.reshape(-1)
        if not self.sel_enabled or not self.training:
            return F.cross_entropy(flat_logits, flat_targets, reduction="mean")
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        w = self.sel_weights[flat_targets].to(loss.dtype)
        return (loss * w).sum() / w.sum().clamp_min(1e-9)


_orig_train_model = base.train_model


def _patched_train_model(h, device, val_data):
    weights = _build_unigram_weights(h)
    SelectiveLMGPT._SEL_WEIGHTS = weights
    base.log(
        f"selective_lm: built unigram weights "
        f"(mean={weights.mean().item():.3f}, "
        f"min={weights.min().item():.3f}, "
        f"max={weights.max().item():.3f})"
    )
    return _orig_train_model(h, device, val_data)


base.GPT = SelectiveLMGPT
base.train_model = _patched_train_model


if __name__ == "__main__":
    base.main()
