# Detailed Results — Edouard's experiments

## Summary of all runs (2x B300 GPUs, node 2, devices 6-7)

| Version | val_loss | Steps | ms/step | Time | Key changes |
|---------|----------|-------|---------|------|-------------|
| v1 | ~4.15 (train) | 1832 | ~300 | 10.0m | SwiGLU + RMSNorm + RoPE + Muon + torch.compile |
| v2 | ~4.07 (train) | 1872 | ~250 | 10.0m | + QK Norm, logit cap, norm after embed, gc.disable, WSD |
| v3 | ~4.28 (train) | 1177 | ~450 | 10.0m | 253M params — regression, too few steps |
| v4s | 4.1689 (val) | 1669 | ~315 | 10.0m | 110M + val eval + WSD max_steps=2000 |
| v5 | 4.0939 (val) | 1964 | ~237 | 10.0m | + val_embed, x0 residual, per-layer lambdas, throughput opts |
| v6 | 4.0127 (val) | 2000 | ~220 | 9.1m | + ReLU², U-Net skips, zero-init (finished early!) |
| **v7** | **3.9199 (val)** | **2500** | **~190** | **9.6m** | max_steps 2000→2500, uses full budget |

## Key lessons learned

1. **Throughput > Model size** (on 2 GPUs): 110M params at ~190ms/step beats 253M at ~450ms/step. More steps = more tokens = better loss.
2. **WSD schedule is crucial**: The decay phase (last 20% of steps) produces massive loss drops (e.g. 4.09 → 3.92 in v7). Must set max_steps correctly so decay activates within time limit.
3. **ReLU² > SwiGLU for speed**: Same param count but one fewer matmul per layer. v6 was 7% faster than v5.
4. **Zero-init + U-Net skips help convergence**: Model starts as near-identity, U-Net skips shorten gradient path.
5. **Muon WSD is important**: Decaying Muon LR (0.02 → 0.002) alongside AdamW LR gives better final loss than keeping Muon LR fixed.

## Architecture details (v7 = v6 model)

- **119.6M parameters** (12 layers, 12 heads, 768 dim)
- ReLU² MLP (hidden=3072, 2 projections instead of SwiGLU's 3)
- RMSNorm (not LayerNorm)
- RoPE (no learned positional embeddings)
- QK Norm (rms_norm on Q and K)
- Logit soft-capping: `30 * tanh(logits/30)`
- Norm after embedding
- Value Embeddings (ResFormer): learned per-position bias on V
- x0 Residual: skip from initial embedding to every block (learnable, init=0)
- Per-layer Lambdas: learnable scaling for attn/mlp residuals
- U-Net Skip Connections: symmetric pairs (layer 0→11, 1→10, ..., 5→6) with learned scalars (init=0)
- Zero-init: c_proj and w_down initialized to 0 (muP-like)
- Weight tying: wte.weight = lm_head.weight

## Training details (v7)

- Muon optimizer (3 Newton-Schulz iters) for 2D matrix weights
- AdamW (fused) for embeddings, norms, 1D params
- WSD schedule: warmup 100 steps → stable (80%) → linear decay (20%)
  - AdamW: 6e-4 → 6e-5
  - Muon: 0.02 → 0.002
- max_steps=2500, time_limit=10min
- Batch 32, grad_accum 2 (effective batch = 64 * 1024 = 65K tokens/step)
- Data prefetch (async CPU thread)
- gc.disable() during training
- torch.compile enabled
- Validation: last shard reserved, evaluated every 200 steps (10 batches)

---

## Full v7 training log (2x B300, 10 min)

```
[model] 119.6M parameters
[compile] torch.compile enabled
[data] train: 48 shard(s), 48,000,000,000 tokens
[data] val:   1 shard(s), 1,000,000,000 tokens

step     10 | loss 14.0185 | avg50 14.0970 | lr 5.40e-05 | μlr 0.002 | 243ms/step | elapsed 0.2m | time left 9.8m
step     20 | loss 13.4438 | avg50 13.9256 | lr 1.14e-04 | μlr 0.004 | 243ms/step | elapsed 0.2m | time left 9.8m
step     30 | loss 12.1188 | avg50 13.5473 | lr 1.74e-04 | μlr 0.006 | 272ms/step | elapsed 0.3m | time left 9.7m
step     40 | loss  9.9623 | avg50 12.8984 | lr 2.34e-04 | μlr 0.008 | 254ms/step | elapsed 0.3m | time left 9.7m
step     50 | loss  8.9076 | avg50 12.1833 | lr 2.94e-04 | μlr 0.010 | 247ms/step | elapsed 0.4m | time left 9.6m
step    100 | loss  7.6424 | avg50  8.1632 | lr 5.94e-04 | μlr 0.020 | 255ms/step | elapsed 0.6m | time left 9.4m
[eval] step 200 | val_loss 6.2995 ★ best!
[eval] step 400 | val_loss 5.1080 ★ best!
[eval] step 600 | val_loss 4.7176 ★ best!
[eval] step 800 | val_loss 4.4476 ★ best!
step   1000 | loss  4.3941 | avg50  4.3853 | lr 6.00e-04 | μlr 0.020 | 163ms/step | elapsed 4.1m | time left 5.9m
[eval] step 1000 | val_loss 4.4042 ★ best!
[eval] step 1200 | val_loss 4.3126 ★ best!
[eval] step 1400 | val_loss 4.2603 ★ best!
[eval] step 1600 | val_loss 4.2204 ★ best!
[eval] step 1800 | val_loss 4.1418 ★ best!
step   2000 | loss  4.1380 | avg50  4.1078 | lr 6.00e-04 | μlr 0.020 | 230ms/step | elapsed 7.3m | time left 2.7m
[eval] step 2000 | val_loss 4.0891 ★ best!
--- WSD decay starts here (step 2000 = 80% of 2500) ---
step   2100 | loss  4.1418 | avg50  4.0610 | lr 4.93e-04 | μlr 0.016 | 293ms/step | elapsed 7.8m | time left 2.2m
[eval] step 2200 | val_loss 4.0335 ★ best!
step   2300 | loss  3.9959 | avg50  3.9908 | lr 2.77e-04 | μlr 0.009 | 227ms/step | elapsed 8.7m | time left 1.3m
[eval] step 2400 | val_loss 3.9361 ★ best!
step   2500 | loss  3.8369 | avg50  3.9339 | lr 6.11e-05 | μlr 0.002 | 220ms/step | elapsed 9.6m | time left 0.4m

[done] Reached max_steps=2500.
[ckpt] saved → checkpoint_v7.pt  (step 2500)
[eval] FINAL | val_loss 3.9199 | best was 3.9361
```

### val_loss progression during v7

| Step | val_loss | Phase |
|------|----------|-------|
| 200 | 6.2995 | warmup → stable |
| 400 | 5.1080 | stable |
| 600 | 4.7176 | stable |
| 800 | 4.4476 | stable |
| 1000 | 4.4042 | stable |
| 1200 | 4.3126 | stable |
| 1400 | 4.2603 | stable |
| 1600 | 4.2204 | stable |
| 1800 | 4.1418 | stable |
| 2000 | 4.0891 | stable → **decay starts** |
| 2200 | 4.0335 | decay |
| 2400 | 3.9361 | decay |
| FINAL | **3.9199** | decay (end) |

### ms/step evolution (torch.compile warmup visible)

- Steps 10-100: ~250ms (compilation overhead)
- Steps 100-800: ~200ms (compiled, settling)
- Steps 800-1800: ~150ms (fully optimized, torch.compile warmed up)
- Steps 1800-2500: ~230ms (some variance from other GPU activity)

## Next steps

- **v8**: max_steps=2700 (use remaining 0.4 min margin)
- **v9**: FP8 training via torchao (potential ~1.3-2x matmul speedup on Blackwell)
- **32 GPU submission**: scale model + batch size for final competition run
