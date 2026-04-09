"""
v6 training — ReLU² + U-Net skips + zero-init + throughput opts + val eval
"""

import gc
import os
import time
import glob
import math
import argparse
import threading
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from model import get_model


# ---------------------------------------------------------------------------
# Muon optimizer — 3 Newton-Schulz iters
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
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
                g = buf.clone()
                if g.dim() == 2:
                    g = self._newton_schulz(g)
                p.add_(g, alpha=-lr)

    @staticmethod
    def _newton_schulz(G, steps=3):
        a, b, c = (3.4445, -4.7750, 2.0315)
        shape = G.shape
        if G.shape[0] > G.shape[1]:
            G = G.T
            transposed = True
        else:
            transposed = False
        G = G / (G.norm() + 1e-7)
        for _ in range(steps):
            A = G @ G.T
            G = a * G + b * (A @ G) + c * (A @ (A @ G))
        if transposed:
            G = G.T
        return G.reshape(shape)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data_dir:    str   = "data"
    token_dtype: str   = "uint16"
    seq_len:     int   = 1024

    vocab_size: int   = 32768
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.0

    batch_size:       int   = 32
    grad_accum_steps: int   = 2
    max_lr:           float = 6e-4
    min_lr:           float = 6e-5
    muon_max_lr:      float = 0.02
    muon_min_lr:      float = 0.002
    warmup_steps:     int   = 100
    max_steps:        int   = 2_500
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    time_limit_seconds: float = 10 * 60

    eval_interval: int = 200
    eval_batches:  int = 10

    checkpoint_path: str = "checkpoint.pt"


# ---------------------------------------------------------------------------
# Dataset with prefetch
# ---------------------------------------------------------------------------

class BinDataset:
    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16"):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"No *.bin files found in '{data_dir}'")
        self.seq_len = seq_len
        np_dtype = np.dtype(dtype)

        if len(paths) > 1:
            train_paths = paths[:-1]
            val_paths = [paths[-1]]
        else:
            train_paths = paths
            val_paths = paths

        self.train_shards = [np.memmap(p, dtype=np_dtype, mode="r") for p in train_paths]
        self.val_shards   = [np.memmap(p, dtype=np_dtype, mode="r") for p in val_paths]

        train_lens = [len(s) for s in self.train_shards]
        self.train_total = sum(train_lens)
        self.train_weights = [l / self.train_total for l in train_lens]

        val_lens = [len(s) for s in self.val_shards]
        self.val_total = sum(val_lens)
        self.val_weights = [l / self.val_total for l in val_lens]

        self._prefetch_result = None
        self._prefetch_thread = None

        print(f"[data] train: {len(train_paths)} shard(s), {self.train_total:,} tokens")
        print(f"[data] val:   {len(val_paths)} shard(s), {self.val_total:,} tokens")

    def _sample_batch_cpu(self, shards, weights, batch_size):
        xs, ys = [], []
        for _ in range(batch_size):
            shard = shards[np.random.choice(len(shards), p=weights)]
            start = np.random.randint(0, len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(shard[start:start + self.seq_len + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs), torch.stack(ys)

    def get_batch(self, batch_size: int, device):
        if self._prefetch_result is not None and self._prefetch_thread is not None:
            self._prefetch_thread.join()
            x, y = self._prefetch_result
            self._prefetch_result = None
            self._prefetch_thread = None
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y = self._sample_batch_cpu(self.train_shards, self.train_weights, batch_size)
        return x.to(device), y.to(device)

    def prefetch(self, batch_size: int):
        def _load():
            self._prefetch_result = self._sample_batch_cpu(
                self.train_shards, self.train_weights, batch_size)
        self._prefetch_thread = threading.Thread(target=_load)
        self._prefetch_thread.start()

    def get_val_batch(self, batch_size: int, device):
        x, y = self._sample_batch_cpu(self.val_shards, self.val_weights, batch_size)
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# LR schedules: WSD for both AdamW and Muon
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    decay_start = int(cfg.max_steps * 0.8)
    if step < decay_start:
        return cfg.max_lr
    progress = (step - decay_start) / (cfg.max_steps - decay_start)
    return cfg.min_lr + (1.0 - progress) * (cfg.max_lr - cfg.min_lr)


def get_muon_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.muon_max_lr * step / cfg.warmup_steps
    decay_start = int(cfg.max_steps * 0.8)
    if step < decay_start:
        return cfg.muon_max_lr
    progress = (step - decay_start) / (cfg.max_steps - decay_start)
    return cfg.muon_min_lr + (1.0 - progress) * (cfg.muon_max_lr - cfg.muon_min_lr)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loss(model, dataset, cfg, device, amp_ctx):
    model.eval()
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = dataset.get_val_batch(cfg.batch_size, device)
        with amp_ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / cfg.eval_batches


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
    parser.add_argument("--n_layer",           type=int,   default=12)
    parser.add_argument("--n_head",            type=int,   default=12)
    parser.add_argument("--n_embd",            type=int,   default=768)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--grad_accum_steps",  type=int,   default=2)
    parser.add_argument("--max_steps",         type=int,   default=2_500)
    parser.add_argument("--time_limit_min",    type=float, default=10.0)
    parser.add_argument("--eval_interval",     type=int,   default=200)
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
        eval_interval      = args.eval_interval,
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

    if "cuda" in device and hasattr(torch, "compile"):
        model = torch.compile(model)
        if master:
            print("[compile] torch.compile enabled")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ Optimizer
    raw_model = model.module if ddp else model

    muon_params = []
    adam_params_decay = []
    adam_params_nodecay = []

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
    optimizer_muon = Muon(muon_params, lr=cfg.muon_max_lr, momentum=0.95)
    optimizers = [optimizer_adam, optimizer_muon]

    # ------------------------------------------------------------------ Data
    dataset = BinDataset(cfg.data_dir, cfg.seq_len, cfg.token_dtype)

    # ------------------------------------------------------------------ Train
    step        = 0
    train_start = time.time()
    best_val    = float("inf")
    loss_history = deque(maxlen=50)
    model.train()
    for opt in optimizers:
        opt.zero_grad()

    dataset.prefetch(cfg.batch_size)

    while step < cfg.max_steps:

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

        adam_lr = get_lr(step, cfg)
        muon_lr = get_muon_lr(step, cfg)
        for pg in optimizer_adam.param_groups:
            pg["lr"] = adam_lr
        for pg in optimizer_muon.param_groups:
            pg["lr"] = muon_lr

        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y = dataset.get_batch(cfg.batch_size, device)
            if micro_step < cfg.grad_accum_steps - 1:
                dataset.prefetch(cfg.batch_size)

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
        loss_history.append(accumulated_loss)

        dataset.prefetch(cfg.batch_size)

        if master and step % 10 == 0:
            elapsed_total = time.time() - train_start
            remaining     = max(0, cfg.time_limit_seconds - elapsed_total)
            avg50 = sum(loss_history) / len(loss_history)
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | avg50 {avg50:.4f} | "
                  f"lr {adam_lr:.2e} | μlr {muon_lr:.3f} | "
                  f"{(time.time()-step_start)*1000:.0f}ms/step | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"time left {remaining/60:.1f}m")

        if step % cfg.eval_interval == 0:
            val = eval_loss(model, dataset, cfg, device, amp_ctx)
            if master:
                tag = " ★ best!" if val < best_val else ""
                if val < best_val:
                    best_val = val
                print(f"[eval] step {step} | val_loss {val:.4f}{tag}")

    if step >= cfg.max_steps and master:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg)

    val = eval_loss(model, dataset, cfg, device, amp_ctx)
    if master:
        print(f"[eval] FINAL | val_loss {val:.4f} | best was {best_val:.4f}")

    gc.collect()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
