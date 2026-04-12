"""train_gpt_mor.py - Mixture-of-Recursions variant of the SP8192 SOTA baseline.

The baseline runs layers [LOOP_START..LOOP_END] for (NUM_LOOPS+1) passes - every
token pays the same depth cost. This variant gates the *repeat* passes per-token
via a tiny scalar router living on blocks[LOOP_START], so easy tokens skip extra
recursion and hard tokens use full depth.

Mechanic
--------
For each call to a block index inside the loop range:
  - first call:    x = block(x, x0)                              (unchanged)
  - later calls:   x = x + sigmoid(x.w + b) * (block(x,x0) - x)  (gated)

Router init: w=0, b=MOR_GATE_INIT (default 5.0 -> sigmoid ~ 0.993), so step 0
behavior is essentially identical to the baseline. The router learns to push
the gate down for tokens that don't benefit from extra depth.

Param cost: model_dim + 1 floats. Both are 1D / 0D so they auto-route into the
existing scalar AdamW group via the Optimizers class (no surgery needed).

New env vars
------------
  MOR_ENABLED=1            off -> falls through to baseline forward_logits
  MOR_GATE_INIT=5.0        bias init for the gate scalar

Baseline knobs worth revisiting alongside MoR
---------------------------------------------
  NUM_LOOPS=3              was 2 - give the router more headroom
  LOOP_END=6               widen the looped sub-stack
  ENABLE_LOOPING_AT=0.20   was 0.35 - turn looping on earlier so the gate
                           gets more steps to learn before warmdown

Run
---
  DATA_DIR=./data/ RUN_ID=mor_seed42 SEED=42 \
  TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 \
  MOR_ENABLED=1 MOR_GATE_INIT=5.0 \
  NUM_LOOPS=3 LOOP_END=6 ENABLE_LOOPING_AT=0.20 \
  torchrun --nproc_per_node=8 train_gpt_mor.py
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_BASE_PATH = (
    Path(__file__).parent
    / "records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py"
)
_spec = importlib.util.spec_from_file_location("sota_baseline", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules["sota_baseline"] = base
_spec.loader.exec_module(base)


class MoRGPT(base.GPT):
    def __init__(self, h):
        super().__init__(h)
        self.mor_enabled = bool(int(os.environ.get("MOR_ENABLED", "1")))
        gate_init = float(os.environ.get("MOR_GATE_INIT", "5.0"))
        # Stash gate params on a Block so they get picked up by the existing
        # scalar AdamW group (Optimizers walks base_model.blocks).
        anchor = self.blocks[h.loop_start]
        anchor.mor_gate_w = nn.Parameter(torch.zeros(h.model_dim, dtype=torch.float32))
        anchor.mor_gate_b = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self._mor_anchor_idx = h.loop_start
        self._mor_loop_set = set(range(h.loop_start, h.loop_end + 1))

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        if self.looping_active:
            enc_iter = self.encoder_indices
            dec_iter = self.decoder_indices
        else:
            enc_iter = list(range(self.num_encoder_layers))
            dec_iter = list(
                range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
            )

        anchor = self.blocks[self._mor_anchor_idx]
        gate_active = self.mor_enabled and self.looping_active
        seen: set[int] = set()
        # Touch gate params unconditionally so DDP sees them used on every
        # forward — otherwise the warmup phase (looping_active=False) leaves
        # them grad-less and DDP errors when looping flips on later.
        x = x + (anchor.mor_gate_w.sum() + anchor.mor_gate_b).to(x.dtype) * 0.0

        def _gate(prev, new):
            w = anchor.mor_gate_w.to(prev.dtype)
            b = anchor.mor_gate_b.to(prev.dtype)
            g = torch.sigmoid((prev * w).sum(-1, keepdim=True) + b)
            return prev + g * (new - prev)

        for i in enc_iter:
            new_x = self.blocks[i](x, x0)
            if gate_active and i in self._mor_loop_set and i in seen:
                x = _gate(x, new_x)
            else:
                x = new_x
            seen.add(i)
            skips.append(x)

        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = (
                    self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                )
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            new_x = self.blocks[i](x, x0)
            if gate_active and i in self._mor_loop_set and i in seen:
                x = _gate(x, new_x)
            else:
                x = new_x
            seen.add(i)

        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


base.GPT = MoRGPT


if __name__ == "__main__":
    base.main()
