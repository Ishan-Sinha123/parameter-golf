"""
Data loading and evaluation for autoresearch experiments.
Uses the competition's fineweb10B_sp1024 dataset and SentencePiece tokenizer
so that val_bpb numbers are directly comparable to the leaderboard.

This file is READ-ONLY for the autoresearch agent.
"""

import os
import glob
import math
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024        # context length (competition standard)
TIME_BUDGET = 300          # training time budget in seconds (5 minutes)

# ---------------------------------------------------------------------------
# Paths — data lives in the parent repo
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.environ.get("DATA_PATH", os.path.join(_REPO_ROOT, "data", "datasets", "fineweb10B_sp1024"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(_REPO_ROOT, "data", "tokenizers", "fineweb_1024_bpe.model"))
TRAIN_PATTERN = os.path.join(DATA_DIR, "fineweb_train_*.bin")
VAL_PATTERN = os.path.join(DATA_DIR, "fineweb_val_*.bin")
VOCAB_SIZE = 1024

# ---------------------------------------------------------------------------
# Binary shard loading (competition format)
# ---------------------------------------------------------------------------

def _load_data_shard(file: Path) -> Tensor:
    """Load a binary shard: 256 int32 header + uint16 tokens."""
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

# ---------------------------------------------------------------------------
# Token stream (sequential shard reader with wraparound)
# ---------------------------------------------------------------------------

class _TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = _load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class Tokenizer:
    """SentencePiece tokenizer wrapper matching the competition setup."""

    def __init__(self, sp):
        self.sp = sp

    @classmethod
    def from_file(cls, path=TOKENIZER_PATH):
        sp = spm.SentencePieceProcessor(model_file=path)
        return cls(sp)

    def get_vocab_size(self):
        return int(self.sp.vocab_size())

# ---------------------------------------------------------------------------
# SentencePiece BPB lookup tables (competition-exact)
# ---------------------------------------------------------------------------

def _build_sentencepiece_luts(sp, vocab_size, device):
    """Build byte-counting lookup tables for BPB evaluation."""
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):  # sentencepiece space marker
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

# ---------------------------------------------------------------------------
# Dataloader (competition-compatible, with distributed support)
# ---------------------------------------------------------------------------

def make_dataloader(B, T, split, rank=0, world_size=1):
    """
    Simple token-stream dataloader. Returns (x, y) pairs of shape (B, T).
    For distributed training, each rank gets a disjoint slice of each batch.
    """
    assert split in ["train", "val"]
    pattern = TRAIN_PATTERN if split == "train" else VAL_PATTERN
    stream = _TokenStream(pattern)
    device = torch.device("cuda")
    local_tokens = B * T
    per_rank_span = local_tokens + 1

    while True:
        chunk = stream.take(per_rank_span * world_size)
        start = rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64, device=device)
        x = local[:-1].reshape(B, T)
        y = local[1:].reshape(B, T)
        yield x, y

# ---------------------------------------------------------------------------
# Evaluation — competition-exact BPB (DO NOT CHANGE)
# ---------------------------------------------------------------------------

def _load_validation_tokens(seq_len):
    """Load all validation tokens into a single tensor."""
    files = [Path(p) for p in sorted(glob.glob(VAL_PATTERN))]
    if not files:
        raise FileNotFoundError(f"No validation files found for pattern: {VAL_PATTERN}")
    tokens = torch.cat([_load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, rank=0, world_size=1):
    """
    Competition-exact BPB evaluation.
    Uses SentencePiece byte-counting with leading-space adjustment.
    Results are directly comparable to the parameter-golf leaderboard.
    """
    device = torch.device("cuda")
    sp = tokenizer.sp
    vocab_size = tokenizer.get_vocab_size()

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = _build_sentencepiece_luts(sp, vocab_size, device)
    val_tokens = _load_validation_tokens(MAX_SEQ_LEN)

    total_seqs = (val_tokens.numel() - 1) // MAX_SEQ_LEN
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    local_batch_seqs = batch_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * MAX_SEQ_LEN
            raw_end = batch_seq_end * MAX_SEQ_LEN + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, MAX_SEQ_LEN)
            y = local[1:].reshape(-1, MAX_SEQ_LEN)

            batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            # Byte counting with leading-space adjustment (competition-exact)
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    # All-reduce across ranks
    if world_size > 1:
        import torch.distributed as dist
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(bits_per_token * tokens_per_byte)
