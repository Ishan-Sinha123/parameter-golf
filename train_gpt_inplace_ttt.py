"""train_gpt_inplace_ttt.py - In-place fast-weight TTT during training.

Adds a low-rank LoRA adapter to every MLP `proj` (FC2). The adapter is the
"fast weight": it has its own SGD optimizer with high LR, and it is reset to
zero every INPLACE_TTT_RESET_EVERY optimizer steps. The slow weights update
normally via Muon. This approximates the per-chunk fast-weight updates from
In-Place Test-Time Training (Cai et al., 2026), bolted onto the existing
training loop without restructuring it.

The adapter never enters the final 16MB artifact:
  - it gets reset to zero every K optimizer steps,
  - the EMA decays the (mostly-zero) adapter toward zero between resets,
  - the EMA-applied state_dict goes through GPTQ, which only quantizes 2D
    floats with numel > 65536 - lora_a (rank*hidden) and lora_b (dim*rank)
    are well below that and pass through as fp16, ~20KB total before brotli.

Mechanic
--------
FastMLP.forward(x):
    h = leaky_relu(fc(x), 0.5).square()
    out = proj(h) + (h @ lora_a.T) @ lora_b.T   # rank-r LoRA delta on FC2
    return out

FastOptimizers:
  - inherits the baseline scalar/Muon/AdamW setup
  - peels lora_a, lora_b out into a side SGD optimizer
  - on every step, after the parent step, increments a counter and zeros the
    fast weights when (count % RESET_EVERY == 0)

Note on the reset cadence: with RESET_EVERY=1 the fast delta is always zero
at forward time and the SGD step is wasted. RESET_EVERY=4 (default) gives the
adapter four full optimizer steps to accumulate before each wash-out, which
is the smallest cadence where it actually contributes to the forward.

New env vars
------------
  INPLACE_TTT_ENABLED=1
  INPLACE_TTT_RANK=4
  INPLACE_TTT_LR=0.05
  INPLACE_TTT_RESET_EVERY=4    reset every K optimizer steps
  INPLACE_TTT_MOMENTUM=0.0     SGD momentum on the fast weights

Baseline knobs worth revisiting alongside in-place TTT
------------------------------------------------------
  EMA_DECAY=0.9985             was 0.9965 - faster wash-out of the fast delta
  GRAD_CLIP_NORM=0.5           was 0.3 - resets create occasional grad spikes
  TRAIN_LOG_EVERY=200          was 500 - watch the post-reset transient

Run
---
  DATA_DIR=./data/ RUN_ID=ittt_seed42 SEED=42 \
  TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 \
  INPLACE_TTT_ENABLED=1 INPLACE_TTT_RANK=4 INPLACE_TTT_LR=0.05 \
  INPLACE_TTT_RESET_EVERY=4 INPLACE_TTT_MOMENTUM=0.0 \
  EMA_DECAY=0.9985 GRAD_CLIP_NORM=0.5 TRAIN_LOG_EVERY=200 \
  torchrun --nproc_per_node=8 train_gpt_inplace_ttt.py
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Tag lora_{a,b} as "control tensors" so the baseline routes them to the scalar
# group (= float32, AdamW). We then peel them out of that group inside our
# FastOptimizers and hand them to a side SGD optimizer. This must be set BEFORE
# we import the baseline, since CONTROL_TENSOR_NAME_PATTERNS is read at module
# import time.
_DEFAULT_PATTERNS = (
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
    "q_gain,skip_weight,skip_weights,skip_gates"
)
_existing = os.environ.get("CONTROL_TENSOR_NAME_PATTERNS", _DEFAULT_PATTERNS)
if "lora_a" not in _existing:
    os.environ["CONTROL_TENSOR_NAME_PATTERNS"] = _existing + ",lora_a,lora_b"

_BASE_PATH = (
    Path(__file__).parent
    / "records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py"
)
_spec = importlib.util.spec_from_file_location("sota_baseline", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules["sota_baseline"] = base
_spec.loader.exec_module(base)


_TTT_ENABLED = bool(int(os.environ.get("INPLACE_TTT_ENABLED", "1")))
_TTT_RANK = int(os.environ.get("INPLACE_TTT_RANK", "4"))

_OrigMLP = base.MLP
_OrigOptimizers = base.Optimizers


class FastMLP(_OrigMLP):
    def __init__(self, dim, mlp_mult):
        super().__init__(dim, mlp_mult)
        if _TTT_ENABLED and _TTT_RANK > 0:
            hidden = int(mlp_mult * dim)
            self.lora_a = nn.Parameter(torch.zeros(_TTT_RANK, hidden, dtype=torch.float32))
            self.lora_b = nn.Parameter(torch.zeros(dim, _TTT_RANK, dtype=torch.float32))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x):
        h = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
        out = self.proj(h)
        if self.lora_a is not None:
            a = self.lora_a.to(h.dtype)
            b = self.lora_b.to(h.dtype)
            out = out + (h @ a.T) @ b.T
            # DDP safety: when both lora tensors are at zero (right after a
            # reset) the chain-rule grads collapse to zero and DDP can flag
            # the params as unused. A zero-scaled sum keeps them firmly in
            # the autograd graph on every forward.
            out = out + (a.sum() + b.sum()) * 0.0
        return out


class FastOptimizers(_OrigOptimizers):
    def __init__(self, h, base_model):
        # Tag the fast params before parent walks blocks so we can find them
        # again afterwards.
        fast_params = []
        for name, p in base_model.named_parameters():
            if "lora_a" in name or "lora_b" in name:
                p._is_fast_ttt = True
                fast_params.append(p)
        super().__init__(h, base_model)
        # Peel fast params out of whichever parent group claimed them.
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["params"] = [p for p in group["params"] if not getattr(p, "_is_fast_ttt", False)]
        if fast_params:
            ttt_lr = float(os.environ.get("INPLACE_TTT_LR", "0.05"))
            ttt_mom = float(os.environ.get("INPLACE_TTT_MOMENTUM", "0.0"))
            self.fast_optim = torch.optim.SGD(
                [{"params": fast_params, "lr": ttt_lr, "base_lr": ttt_lr}],
                momentum=ttt_mom,
            )
            self.optimizers.append(self.fast_optim)
        else:
            self.fast_optim = None
        self._fast_params = fast_params
        self._reset_every = max(1, int(os.environ.get("INPLACE_TTT_RESET_EVERY", "4")))
        self._step_count = 0

    def step(self):
        super().step()
        self._step_count += 1
        if self._fast_params and self._step_count % self._reset_every == 0:
            with torch.no_grad():
                for p in self._fast_params:
                    p.zero_()


base.MLP = FastMLP
base.Optimizers = FastOptimizers


if __name__ == "__main__":
    base.main()
