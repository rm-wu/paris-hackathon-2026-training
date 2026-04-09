"""
Starter training script for the gpu-mode Paris hackathon training track
"""

import gc
import os
import time
import glob
import math
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from model import get_model


# ---------------------------------------------------------------------------
# Muon optimizer — Newton-Schulz orthogonalization for matrix weights
# Adapted from github.com/KellerJordan/modded-nanogpt
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz.
    Faster convergence than AdamW on matrix (2D) weights."""

    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # Newton-Schulz orthogonalization (3 iterations)
                g = buf.clone()
                if g.dim() == 2:
                    g = self._newton_schulz(g)

                p.add_(g, alpha=-lr)

    @staticmethod
    def _newton_schulz(G, steps=5):
        """Approximate the matrix sign function via Newton-Schulz iteration."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        # Reshape to 2D if needed, tall-and-skinny convention
        shape = G.shape
        if G.shape[0] > G.shape[1]:
            G = G.T
            transposed = True
        else:
            transposed = False

        # Normalize
        eps = 1e-7
        G = G / (G.norm() + eps)

        # Newton-Schulz iterations
        for _ in range(steps):
            A = G @ G.T
            G = a * G + b * (A @ G) + c * (A @ (A @ G))

        if transposed:
            G = G.T
        return G.reshape(shape)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_dir:    str   = "data"
    token_dtype: str   = "uint16"
    seq_len:     int   = 1024

    # Model — v3 bigger model
    vocab_size: int   = 32768
    n_layer:    int   = 16
    n_head:     int   = 16
    n_embd:     int   = 1024
    dropout:    float = 0.0

    # Training — larger effective batch for bigger model
    batch_size:       int   = 16
    grad_accum_steps: int   = 4
    max_lr:           float = 4e-4
    min_lr:           float = 4e-5
    warmup_steps:     int   = 100
    max_steps:        int   = 1_500
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    time_limit_seconds: float = 10 * 60

    # Checkpointing
    checkpoint_path: str = "checkpoint.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinDataset:
    """Memory-maps all *.bin files and draws random (seq_len+1)-token windows."""

    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16"):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"No *.bin files found in '{data_dir}'")
        self.seq_len  = seq_len
        np_dtype      = np.dtype(dtype)
        self.shards   = [np.memmap(p, dtype=np_dtype, mode="r") for p in paths]
        self.lengths  = [len(s) for s in self.shards]
        self.total    = sum(self.lengths)
        self.weights  = [l / self.total for l in self.lengths]
        print(f"[data] {len(paths)} shard(s), {self.total:,} tokens total")

    def get_batch(self, batch_size: int, device):
        xs, ys = [], []
        for _ in range(batch_size):
            shard = self.shards[np.random.choice(len(self.shards), p=self.weights)]
            start = np.random.randint(0, len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(shard[start:start + self.seq_len + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay → min_lr
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: Config) -> float:
    """WSD schedule: warmup → stable → linear decay (last 20% of steps).
    Better than cosine for short runs — keeps max LR longer, then decays hard."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    decay_start = int(cfg.max_steps * 0.8)
    if step < decay_start:
        return cfg.max_lr
    progress = (step - decay_start) / (cfg.max_steps - decay_start)
    return cfg.min_lr + (1.0 - progress) * (cfg.max_lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, step: int, cfg: Config):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "step":   step,
        "model":  raw_model.state_dict(),
        "config": asdict(cfg),
    }, cfg.checkpoint_path)
    print(f"[ckpt] saved → {cfg.checkpoint_path}  (step {step})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          default="data")
    parser.add_argument("--checkpoint_path",   default="checkpoint.pt")
    parser.add_argument("--seq_len",           type=int,   default=1024)
    parser.add_argument("--vocab_size",        type=int,   default=32768)
    parser.add_argument("--n_layer",           type=int,   default=16)
    parser.add_argument("--n_head",            type=int,   default=16)
    parser.add_argument("--n_embd",            type=int,   default=1024)
    parser.add_argument("--batch_size",        type=int,   default=16)
    parser.add_argument("--grad_accum_steps",  type=int,   default=4)
    parser.add_argument("--max_steps",         type=int,   default=1_500)
    parser.add_argument("--time_limit_min",    type=float, default=10.0)
    args = parser.parse_args()

    cfg = Config(
        data_dir           = args.data_dir,
        checkpoint_path    = args.checkpoint_path,
        seq_len            = args.seq_len,
        vocab_size         = args.vocab_size,
        n_layer            = args.n_layer,
        n_head             = args.n_head,
        n_embd             = args.n_embd,
        batch_size         = args.batch_size,
        grad_accum_steps   = args.grad_accum_steps,
        max_steps          = args.max_steps,
        time_limit_seconds = args.time_limit_min * 60,
    )

    # ------------------------------------------------------------------ DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master     = rank == 0
    else:
        rank = 0; master = True
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch.manual_seed(1337 + rank)

    # Disable GC during training — reduces stalls from ~100ms collection pauses
    gc.disable()

    if "cuda" in device:
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif device == "mps":
        amp_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    # ------------------------------------------------------------------ Model
    model = get_model(asdict(cfg)).to(device)
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters")

    # torch.compile: fuses operations for ~20-40% speedup on CUDA
    if "cuda" in device and hasattr(torch, "compile"):
        model = torch.compile(model)
        if master:
            print("[compile] torch.compile enabled")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ Optimizer
    # Muon for 2D weights (matrix params) — faster convergence via Newton-Schulz
    # AdamW for everything else (embeddings, norms, biases)
    raw_model = model.module if ddp else model

    muon_params = []   # 2D weight matrices -> Muon
    adam_params_decay = []   # other 2D params that shouldn't use Muon (embeddings)
    adam_params_nodecay = [] # 1D params (norms, biases)

    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            adam_params_nodecay.append(p)
        elif "wte" in name or "lm_head" in name or "val_embed" in name:
            adam_params_decay.append(p)
        else:
            muon_params.append(p)

    optimizer_adam = torch.optim.AdamW(
        [{"params": adam_params_decay,   "weight_decay": cfg.weight_decay},
         {"params": adam_params_nodecay, "weight_decay": 0.0}],
        lr=cfg.max_lr, betas=(0.9, 0.95), fused=("cuda" in device),
    )
    optimizer_muon = Muon(muon_params, lr=0.02, momentum=0.95)
    optimizers = [optimizer_adam, optimizer_muon]

    # ------------------------------------------------------------------ Data
    dataset = BinDataset(cfg.data_dir, cfg.seq_len, cfg.token_dtype)

    # ------------------------------------------------------------------ Train
    step        = 0
    train_start = time.time()
    model.train()
    for opt in optimizers:
        opt.zero_grad()

    while step < cfg.max_steps:

        # Time-limit check — never starts a new step after the deadline
        elapsed = time.time() - train_start
        stop = torch.tensor(int(elapsed >= cfg.time_limit_seconds), device=device)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item():
            if master:
                print(f"\n[time] {elapsed/60:.1f} min elapsed — time limit reached.")
                save_checkpoint(model, step, cfg)
            break

        step_start = time.time()
        # Update LR for AdamW only (Muon uses its own fixed LR)
        for pg in optimizer_adam.param_groups:
            pg["lr"] = get_lr(step, cfg)

        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y     = dataset.get_batch(cfg.batch_size, device)
            sync_ctx = model.no_sync() if (ddp and micro_step < cfg.grad_accum_steps - 1) \
                       else nullcontext()
            with sync_ctx, amp_ctx:
                _, loss = model(x, y)
                loss    = loss / cfg.grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        step += 1

        if master and step % 10 == 0:
            elapsed_total = time.time() - train_start
            remaining     = max(0, cfg.time_limit_seconds - elapsed_total)
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | "
                  f"lr {get_lr(step, cfg):.2e} | "
                  f"{(time.time()-step_start)*1000:.0f}ms/step | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"time left {remaining/60:.1f}m")

    # max_steps reached cleanly
    if step >= cfg.max_steps and master:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg)

    gc.collect()  # re-enable GC before exit

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

