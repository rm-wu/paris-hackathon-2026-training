# Edouard's branch — v7 (val_loss 3.92 on 2 GPUs)

Best result so far: **val_loss 3.9199** in 9.6 min on 2x B300 GPUs (node 2, devices 6-7).

## What changed vs baseline

### Architecture (`model.py` — ~119M params)
| Technique | What it does |
|-----------|-------------|
| **ReLU²** | Squared ReLU MLP — 2 matmuls vs 3 (SwiGLU), same params, faster steps |
| **RMSNorm** | Faster than LayerNorm (no mean computation) |
| **RoPE** | Rotary Position Embeddings, replaces learned positional embeddings |
| **QK Norm** | `rms_norm` on Q and K before attention — stabilizes gradients |
| **Logit soft-capping** | `30 * tanh(logits/30)` prevents logit explosion (Gemma-style) |
| **Norm after embedding** | `rms_norm` on token embeddings — stabilizes early layers |
| **Value Embeddings** | Learned per-position bias on V in attention (ResFormer) |
| **x0 Residual** | Skip from initial embedding to every block (learnable, init=0) |
| **Per-layer Lambdas** | Learnable scaling for attn/mlp residuals per layer |
| **U-Net Skip Connections** | Symmetric skip connections (layer 0→11, 1→10, ...) with learned scalars |
| **Zero-init output projections** | c_proj and w_down init to 0 (muP-like, model starts as near-identity) |
| **Weight tying** | `wte.weight = lm_head.weight` |

### Training (`train.py`)
| Technique | What it does |
|-----------|-------------|
| **Muon optimizer** | Newton-Schulz orthogonalization for 2D weights (3 iters), AdamW for the rest |
| **WSD schedule** | Warmup (100 steps) → Stable (80%) → Linear Decay (20%), for both AdamW and Muon |
| **torch.compile** | Fused kernel generation |
| **gc.disable()** | Eliminates ~100ms GC pauses during training |
| **Batch 32, grad_accum 2** | Same tokens/step as 16/4, but 2x fewer micro-steps → faster steps |
| **Data prefetch** | Async CPU data loading on a thread while GPU computes |
| **Validation eval** | Last shard reserved for val, evaluated every 200 steps |

## Results on 2x B300

| Version | val_loss | Steps | ms/step | Key change |
|---------|----------|-------|---------|------------|
| v1 (baseline+) | ~4.15 (train) | 1832 | ~300 | SwiGLU + RMSNorm + RoPE + Muon |
| v2 (quick wins) | ~4.07 (train) | 1872 | ~250 | + QK Norm, logit cap, WSD, gc.disable |
| v3 (big model) | ~4.28 (train) | 1177 | ~450 | 253M params — too few steps, regression |
| v4s (+ val eval) | 4.1689 (val) | 1669 | ~315 | 110M + val eval + WSD max_steps fix |
| v5 (throughput) | 4.0939 (val) | 1964 | ~237 | + val_embed, x0, lambdas, throughput opts |
| v6 (new arch) | 4.0127 (val) | 2000 | ~220 | + ReLU², U-Net skips, zero-init |
| **v7 (this)** | **3.9199 (val)** | **2500** | **~230** | max_steps 2000→2500, use full 10 min |

## How to run

```bash
# On cluster (2 GPUs example)
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29507 train.py \
    --data_dir /home/data/ \
    --checkpoint_path checkpoint.pt \
    --time_limit_min 10

# Full 32 GPUs (submission)
torchrun --nproc_per_node=32 train.py \
    --data_dir /home/data/ \
    --checkpoint_path checkpoint.pt \
    --time_limit_min 10
```

## Next up (v8/v9)
- v8: max_steps=2700 (v7 finished in 9.6 min, still 0.4 min of margin)
- v9: FP8 training via torchao (~1.3-2x matmul speedup on Blackwell)
- Scale to 32 GPUs for final submission (bigger model, FSDP)
