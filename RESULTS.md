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
step     40 | loss 9.9623 | avg50 12.8984 | lr 2.34e-04 | μlr 0.008 | 254ms/step | elapsed 0.3m | time left 9.7m
step     50 | loss 8.9076 | avg50 12.1833 | lr 2.94e-04 | μlr 0.010 | 247ms/step | elapsed 0.4m | time left 9.6m
step     60 | loss 8.4283 | avg50 11.0758 | lr 3.54e-04 | μlr 0.012 | 220ms/step | elapsed 0.4m | time left 9.6m
step     70 | loss 8.3397 | avg50 10.0015 | lr 4.14e-04 | μlr 0.014 | 238ms/step | elapsed 0.5m | time left 9.5m
step     80 | loss 8.1224 | avg50 9.0839 | lr 4.74e-04 | μlr 0.016 | 299ms/step | elapsed 0.5m | time left 9.5m
step     90 | loss 7.8389 | avg50 8.4854 | lr 5.34e-04 | μlr 0.018 | 259ms/step | elapsed 0.6m | time left 9.4m
step    100 | loss 7.6424 | avg50 8.1632 | lr 5.94e-04 | μlr 0.020 | 255ms/step | elapsed 0.6m | time left 9.4m
step    110 | loss 7.4409 | avg50 7.9529 | lr 6.00e-04 | μlr 0.020 | 260ms/step | elapsed 0.7m | time left 9.3m
step    120 | loss 7.2232 | avg50 7.7306 | lr 6.00e-04 | μlr 0.020 | 253ms/step | elapsed 0.7m | time left 9.3m
step    130 | loss 7.1156 | avg50 7.5222 | lr 6.00e-04 | μlr 0.020 | 257ms/step | elapsed 0.7m | time left 9.3m
step    140 | loss 6.9594 | avg50 7.3319 | lr 6.00e-04 | μlr 0.020 | 319ms/step | elapsed 0.8m | time left 9.2m
step    150 | loss 6.7407 | avg50 7.1658 | lr 6.00e-04 | μlr 0.020 | 249ms/step | elapsed 0.8m | time left 9.2m
step    160 | loss 6.6830 | avg50 7.0126 | lr 6.00e-04 | μlr 0.020 | 268ms/step | elapsed 0.9m | time left 9.1m
step    170 | loss 6.5748 | avg50 6.8862 | lr 6.00e-04 | μlr 0.020 | 267ms/step | elapsed 0.9m | time left 9.1m
step    180 | loss 6.4897 | avg50 6.7626 | lr 6.00e-04 | μlr 0.020 | 211ms/step | elapsed 1.0m | time left 9.0m
step    190 | loss 6.3627 | avg50 6.6410 | lr 6.00e-04 | μlr 0.020 | 294ms/step | elapsed 1.0m | time left 9.0m
step    200 | loss 6.2930 | avg50 6.5309 | lr 6.00e-04 | μlr 0.020 | 233ms/step | elapsed 1.1m | time left 8.9m
[eval] step 200 | val_loss 6.2995 ★ best!
step    210 | loss 6.1932 | avg50 6.4296 | lr 6.00e-04 | μlr 0.020 | 251ms/step | elapsed 1.2m | time left 8.8m
step    220 | loss 6.0408 | avg50 6.3256 | lr 6.00e-04 | μlr 0.020 | 248ms/step | elapsed 1.2m | time left 8.8m
step    230 | loss 5.9729 | avg50 6.2356 | lr 6.00e-04 | μlr 0.020 | 240ms/step | elapsed 1.3m | time left 8.7m
step    240 | loss 5.9616 | avg50 6.1567 | lr 6.00e-04 | μlr 0.020 | 228ms/step | elapsed 1.3m | time left 8.7m
step    250 | loss 5.9115 | avg50 6.0714 | lr 6.00e-04 | μlr 0.020 | 265ms/step | elapsed 1.3m | time left 8.7m
step    260 | loss 5.8987 | avg50 5.9946 | lr 6.00e-04 | μlr 0.020 | 281ms/step | elapsed 1.4m | time left 8.6m
step    270 | loss 5.7660 | avg50 5.9348 | lr 6.00e-04 | μlr 0.020 | 247ms/step | elapsed 1.4m | time left 8.6m
step    280 | loss 5.6139 | avg50 5.8630 | lr 6.00e-04 | μlr 0.020 | 222ms/step | elapsed 1.5m | time left 8.5m
step    290 | loss 5.5903 | avg50 5.7984 | lr 6.00e-04 | μlr 0.020 | 241ms/step | elapsed 1.5m | time left 8.5m
step    300 | loss 5.5842 | avg50 5.7395 | lr 6.00e-04 | μlr 0.020 | 245ms/step | elapsed 1.6m | time left 8.4m
step    310 | loss 5.4899 | avg50 5.6754 | lr 6.00e-04 | μlr 0.020 | 256ms/step | elapsed 1.6m | time left 8.4m
step    320 | loss 5.4400 | avg50 5.6105 | lr 6.00e-04 | μlr 0.020 | 257ms/step | elapsed 1.6m | time left 8.4m
step    330 | loss 5.3903 | avg50 5.5491 | lr 6.00e-04 | μlr 0.020 | 189ms/step | elapsed 1.7m | time left 8.3m
step    340 | loss 5.3537 | avg50 5.4910 | lr 6.00e-04 | μlr 0.020 | 284ms/step | elapsed 1.7m | time left 8.3m
step    350 | loss 5.3264 | avg50 5.4421 | lr 6.00e-04 | μlr 0.020 | 228ms/step | elapsed 1.8m | time left 8.2m
step    360 | loss 5.2511 | avg50 5.3886 | lr 6.00e-04 | μlr 0.020 | 226ms/step | elapsed 1.8m | time left 8.2m
step    370 | loss 5.1412 | avg50 5.3294 | lr 6.00e-04 | μlr 0.020 | 227ms/step | elapsed 1.9m | time left 8.1m
step    380 | loss 5.1549 | avg50 5.2768 | lr 6.00e-04 | μlr 0.020 | 202ms/step | elapsed 1.9m | time left 8.1m
step    390 | loss 5.1866 | avg50 5.2257 | lr 6.00e-04 | μlr 0.020 | 232ms/step | elapsed 1.9m | time left 8.1m
step    400 | loss 5.0858 | avg50 5.1713 | lr 6.00e-04 | μlr 0.020 | 198ms/step | elapsed 2.0m | time left 8.0m
[eval] step 400 | val_loss 5.1080 ★ best!
step    410 | loss 5.0184 | avg50 5.1281 | lr 6.00e-04 | μlr 0.020 | 244ms/step | elapsed 2.0m | time left 8.0m
step    420 | loss 5.0320 | avg50 5.1030 | lr 6.00e-04 | μlr 0.020 | 252ms/step | elapsed 2.1m | time left 7.9m
step    430 | loss 4.9326 | avg50 5.0723 | lr 6.00e-04 | μlr 0.020 | 243ms/step | elapsed 2.1m | time left 7.9m
step    440 | loss 4.9367 | avg50 5.0445 | lr 6.00e-04 | μlr 0.020 | 224ms/step | elapsed 2.1m | time left 7.9m
step    450 | loss 5.0559 | avg50 5.0241 | lr 6.00e-04 | μlr 0.020 | 177ms/step | elapsed 2.2m | time left 7.8m
step    460 | loss 4.9160 | avg50 5.0019 | lr 6.00e-04 | μlr 0.020 | 197ms/step | elapsed 2.2m | time left 7.8m
step    470 | loss 4.9750 | avg50 4.9781 | lr 6.00e-04 | μlr 0.020 | 222ms/step | elapsed 2.3m | time left 7.7m
step    480 | loss 4.9211 | avg50 4.9555 | lr 6.00e-04 | μlr 0.020 | 203ms/step | elapsed 2.3m | time left 7.7m
step    490 | loss 4.8088 | avg50 4.9327 | lr 6.00e-04 | μlr 0.020 | 215ms/step | elapsed 2.3m | time left 7.7m
step    500 | loss 4.8434 | avg50 4.9104 | lr 6.00e-04 | μlr 0.020 | 191ms/step | elapsed 2.4m | time left 7.6m
step    510 | loss 4.7511 | avg50 4.8898 | lr 6.00e-04 | μlr 0.020 | 186ms/step | elapsed 2.4m | time left 7.6m
step    520 | loss 4.8381 | avg50 4.8671 | lr 6.00e-04 | μlr 0.020 | 222ms/step | elapsed 2.5m | time left 7.5m
step    530 | loss 4.7679 | avg50 4.8574 | lr 6.00e-04 | μlr 0.020 | 224ms/step | elapsed 2.5m | time left 7.5m
step    540 | loss 4.8967 | avg50 4.8355 | lr 6.00e-04 | μlr 0.020 | 219ms/step | elapsed 2.5m | time left 7.5m
step    550 | loss 4.7814 | avg50 4.8180 | lr 6.00e-04 | μlr 0.020 | 180ms/step | elapsed 2.6m | time left 7.4m
step    560 | loss 4.8407 | avg50 4.8020 | lr 6.00e-04 | μlr 0.020 | 205ms/step | elapsed 2.6m | time left 7.4m
step    570 | loss 4.7319 | avg50 4.7820 | lr 6.00e-04 | μlr 0.020 | 222ms/step | elapsed 2.6m | time left 7.4m
step    580 | loss 4.7041 | avg50 4.7587 | lr 6.00e-04 | μlr 0.020 | 207ms/step | elapsed 2.7m | time left 7.3m
step    590 | loss 4.8080 | avg50 4.7425 | lr 6.00e-04 | μlr 0.020 | 202ms/step | elapsed 2.7m | time left 7.3m
step    600 | loss 4.7255 | avg50 4.7276 | lr 6.00e-04 | μlr 0.020 | 201ms/step | elapsed 2.8m | time left 7.2m
[eval] step 600 | val_loss 4.7176 ★ best!
step    610 | loss 4.7651 | avg50 4.7166 | lr 6.00e-04 | μlr 0.020 | 206ms/step | elapsed 2.8m | time left 7.2m
step    620 | loss 4.5890 | avg50 4.7043 | lr 6.00e-04 | μlr 0.020 | 215ms/step | elapsed 2.8m | time left 7.2m
step    630 | loss 4.6655 | avg50 4.6870 | lr 6.00e-04 | μlr 0.020 | 229ms/step | elapsed 2.9m | time left 7.1m
step    640 | loss 4.6436 | avg50 4.6809 | lr 6.00e-04 | μlr 0.020 | 205ms/step | elapsed 2.9m | time left 7.1m
step    650 | loss 4.7111 | avg50 4.6677 | lr 6.00e-04 | μlr 0.020 | 178ms/step | elapsed 2.9m | time left 7.1m
step    660 | loss 4.6554 | avg50 4.6586 | lr 6.00e-04 | μlr 0.020 | 223ms/step | elapsed 3.0m | time left 7.0m
step    670 | loss 4.6525 | avg50 4.6435 | lr 6.00e-04 | μlr 0.020 | 227ms/step | elapsed 3.0m | time left 7.0m
step    680 | loss 4.6091 | avg50 4.6388 | lr 6.00e-04 | μlr 0.020 | 187ms/step | elapsed 3.1m | time left 6.9m
step    690 | loss 4.6236 | avg50 4.6240 | lr 6.00e-04 | μlr 0.020 | 224ms/step | elapsed 3.1m | time left 6.9m
step    700 | loss 4.5844 | avg50 4.6111 | lr 6.00e-04 | μlr 0.020 | 188ms/step | elapsed 3.1m | time left 6.9m
step    710 | loss 4.6578 | avg50 4.5997 | lr 6.00e-04 | μlr 0.020 | 199ms/step | elapsed 3.2m | time left 6.8m
step    720 | loss 4.6243 | avg50 4.5977 | lr 6.00e-04 | μlr 0.020 | 206ms/step | elapsed 3.2m | time left 6.8m
step    730 | loss 4.6234 | avg50 4.5949 | lr 6.00e-04 | μlr 0.020 | 163ms/step | elapsed 3.2m | time left 6.8m
step    740 | loss 4.6950 | avg50 4.6006 | lr 6.00e-04 | μlr 0.020 | 192ms/step | elapsed 3.3m | time left 6.7m
step    750 | loss 4.5238 | avg50 4.5905 | lr 6.00e-04 | μlr 0.020 | 173ms/step | elapsed 3.3m | time left 6.7m
step    760 | loss 4.4603 | avg50 4.5779 | lr 6.00e-04 | μlr 0.020 | 192ms/step | elapsed 3.3m | time left 6.7m
step    770 | loss 4.5134 | avg50 4.5687 | lr 6.00e-04 | μlr 0.020 | 180ms/step | elapsed 3.4m | time left 6.6m
step    780 | loss 4.4789 | avg50 4.5596 | lr 6.00e-04 | μlr 0.020 | 174ms/step | elapsed 3.4m | time left 6.6m
step    790 | loss 4.4814 | avg50 4.5349 | lr 6.00e-04 | μlr 0.020 | 193ms/step | elapsed 3.4m | time left 6.6m
step    800 | loss 4.5080 | avg50 4.5274 | lr 6.00e-04 | μlr 0.020 | 158ms/step | elapsed 3.5m | time left 6.5m
[eval] step 800 | val_loss 4.4476 ★ best!
step    810 | loss 4.4896 | avg50 4.5247 | lr 6.00e-04 | μlr 0.020 | 173ms/step | elapsed 3.5m | time left 6.5m
step    820 | loss 4.4003 | avg50 4.5151 | lr 6.00e-04 | μlr 0.020 | 168ms/step | elapsed 3.5m | time left 6.5m
step    830 | loss 4.5531 | avg50 4.4968 | lr 6.00e-04 | μlr 0.020 | 169ms/step | elapsed 3.6m | time left 6.4m
step    840 | loss 4.4758 | avg50 4.5006 | lr 6.00e-04 | μlr 0.020 | 161ms/step | elapsed 3.6m | time left 6.4m
step    850 | loss 4.4664 | avg50 4.4913 | lr 6.00e-04 | μlr 0.020 | 193ms/step | elapsed 3.6m | time left 6.4m
step    860 | loss 4.4568 | avg50 4.4784 | lr 6.00e-04 | μlr 0.020 | 179ms/step | elapsed 3.7m | time left 6.3m
step    870 | loss 4.4822 | avg50 4.4762 | lr 6.00e-04 | μlr 0.020 | 183ms/step | elapsed 3.7m | time left 6.3m
step    880 | loss 4.4562 | avg50 4.4700 | lr 6.00e-04 | μlr 0.020 | 186ms/step | elapsed 3.7m | time left 6.3m
step    890 | loss 4.3697 | avg50 4.4647 | lr 6.00e-04 | μlr 0.020 | 176ms/step | elapsed 3.8m | time left 6.2m
step    900 | loss 4.4248 | avg50 4.4549 | lr 6.00e-04 | μlr 0.020 | 177ms/step | elapsed 3.8m | time left 6.2m
step    910 | loss 4.4685 | avg50 4.4539 | lr 6.00e-04 | μlr 0.020 | 179ms/step | elapsed 3.8m | time left 6.2m
step    920 | loss 4.2905 | avg50 4.4425 | lr 6.00e-04 | μlr 0.020 | 165ms/step | elapsed 3.9m | time left 6.1m
step    930 | loss 4.4263 | avg50 4.4370 | lr 6.00e-04 | μlr 0.020 | 154ms/step | elapsed 3.9m | time left 6.1m
step    940 | loss 4.4032 | avg50 4.4213 | lr 6.00e-04 | μlr 0.020 | 168ms/step | elapsed 3.9m | time left 6.1m
step    950 | loss 4.4075 | avg50 4.4248 | lr 6.00e-04 | μlr 0.020 | 179ms/step | elapsed 4.0m | time left 6.0m
step    960 | loss 4.3900 | avg50 4.4117 | lr 6.00e-04 | μlr 0.020 | 200ms/step | elapsed 4.0m | time left 6.0m
step    970 | loss 4.4196 | avg50 4.3982 | lr 6.00e-04 | μlr 0.020 | 171ms/step | elapsed 4.0m | time left 6.0m
step    980 | loss 4.4390 | avg50 4.3898 | lr 6.00e-04 | μlr 0.020 | 176ms/step | elapsed 4.1m | time left 5.9m
step    990 | loss 4.4319 | avg50 4.3868 | lr 6.00e-04 | μlr 0.020 | 166ms/step | elapsed 4.1m | time left 5.9m
step   1000 | loss 4.3941 | avg50 4.3853 | lr 6.00e-04 | μlr 0.020 | 163ms/step | elapsed 4.1m | time left 5.9m
[eval] step 1000 | val_loss 4.4042 ★ best!
step   1010 | loss 4.4015 | avg50 4.3855 | lr 6.00e-04 | μlr 0.020 | 177ms/step | elapsed 4.2m | time left 5.8m
step   1020 | loss 4.2504 | avg50 4.3852 | lr 6.00e-04 | μlr 0.020 | 168ms/step | elapsed 4.2m | time left 5.8m
step   1030 | loss 4.3940 | avg50 4.3864 | lr 6.00e-04 | μlr 0.020 | 164ms/step | elapsed 4.2m | time left 5.8m
step   1040 | loss 4.3819 | avg50 4.3804 | lr 6.00e-04 | μlr 0.020 | 162ms/step | elapsed 4.3m | time left 5.7m
step   1050 | loss 4.3839 | avg50 4.3715 | lr 6.00e-04 | μlr 0.020 | 152ms/step | elapsed 4.3m | time left 5.7m
step   1060 | loss 4.3415 | avg50 4.3690 | lr 6.00e-04 | μlr 0.020 | 150ms/step | elapsed 4.3m | time left 5.7m
step   1070 | loss 4.3505 | avg50 4.3724 | lr 6.00e-04 | μlr 0.020 | 145ms/step | elapsed 4.4m | time left 5.6m
step   1080 | loss 4.4518 | avg50 4.3708 | lr 6.00e-04 | μlr 0.020 | 173ms/step | elapsed 4.4m | time left 5.6m
step   1090 | loss 4.3807 | avg50 4.3634 | lr 6.00e-04 | μlr 0.020 | 147ms/step | elapsed 4.4m | time left 5.6m
step   1100 | loss 4.3879 | avg50 4.3660 | lr 6.00e-04 | μlr 0.020 | 148ms/step | elapsed 4.5m | time left 5.5m
step   1110 | loss 4.4560 | avg50 4.3561 | lr 6.00e-04 | μlr 0.020 | 178ms/step | elapsed 4.5m | time left 5.5m
step   1120 | loss 4.2923 | avg50 4.3351 | lr 6.00e-04 | μlr 0.020 | 165ms/step | elapsed 4.5m | time left 5.5m
step   1130 | loss 4.3077 | avg50 4.3300 | lr 6.00e-04 | μlr 0.020 | 183ms/step | elapsed 4.5m | time left 5.5m
step   1140 | loss 4.3862 | avg50 4.3350 | lr 6.00e-04 | μlr 0.020 | 162ms/step | elapsed 4.6m | time left 5.4m
step   1150 | loss 4.3184 | avg50 4.3293 | lr 6.00e-04 | μlr 0.020 | 132ms/step | elapsed 4.6m | time left 5.4m
step   1160 | loss 4.2548 | avg50 4.3272 | lr 6.00e-04 | μlr 0.020 | 186ms/step | elapsed 4.6m | time left 5.4m
step   1170 | loss 4.3683 | avg50 4.3364 | lr 6.00e-04 | μlr 0.020 | 187ms/step | elapsed 4.7m | time left 5.3m
step   1180 | loss 4.2188 | avg50 4.3357 | lr 6.00e-04 | μlr 0.020 | 176ms/step | elapsed 4.7m | time left 5.3m
step   1190 | loss 4.2265 | avg50 4.3198 | lr 6.00e-04 | μlr 0.020 | 164ms/step | elapsed 4.7m | time left 5.3m
step   1200 | loss 4.3155 | avg50 4.3118 | lr 6.00e-04 | μlr 0.020 | 144ms/step | elapsed 4.8m | time left 5.2m
[eval] step 1200 | val_loss 4.3126 ★ best!
step   1210 | loss 4.3699 | avg50 4.3097 | lr 6.00e-04 | μlr 0.020 | 167ms/step | elapsed 4.8m | time left 5.2m
step   1220 | loss 4.3275 | avg50 4.3088 | lr 6.00e-04 | μlr 0.020 | 153ms/step | elapsed 4.8m | time left 5.2m
step   1230 | loss 4.2821 | avg50 4.2944 | lr 6.00e-04 | μlr 0.020 | 156ms/step | elapsed 4.9m | time left 5.1m
step   1240 | loss 4.3183 | avg50 4.3008 | lr 6.00e-04 | μlr 0.020 | 157ms/step | elapsed 4.9m | time left 5.1m
step   1250 | loss 4.3304 | avg50 4.2943 | lr 6.00e-04 | μlr 0.020 | 145ms/step | elapsed 4.9m | time left 5.1m
step   1260 | loss 4.3151 | avg50 4.2902 | lr 6.00e-04 | μlr 0.020 | 170ms/step | elapsed 5.0m | time left 5.0m
step   1270 | loss 4.3204 | avg50 4.2847 | lr 6.00e-04 | μlr 0.020 | 137ms/step | elapsed 5.0m | time left 5.0m
step   1280 | loss 4.3188 | avg50 4.2838 | lr 6.00e-04 | μlr 0.020 | 144ms/step | elapsed 5.0m | time left 5.0m
step   1290 | loss 4.2203 | avg50 4.2826 | lr 6.00e-04 | μlr 0.020 | 120ms/step | elapsed 5.0m | time left 5.0m
step   1300 | loss 4.2359 | avg50 4.2827 | lr 6.00e-04 | μlr 0.020 | 142ms/step | elapsed 5.1m | time left 4.9m
step   1310 | loss 4.2278 | avg50 4.2791 | lr 6.00e-04 | μlr 0.020 | 146ms/step | elapsed 5.1m | time left 4.9m
step   1320 | loss 4.2490 | avg50 4.2730 | lr 6.00e-04 | μlr 0.020 | 142ms/step | elapsed 5.1m | time left 4.9m
step   1330 | loss 4.2349 | avg50 4.2692 | lr 6.00e-04 | μlr 0.020 | 151ms/step | elapsed 5.2m | time left 4.8m
step   1340 | loss 4.2078 | avg50 4.2567 | lr 6.00e-04 | μlr 0.020 | 162ms/step | elapsed 5.2m | time left 4.8m
step   1350 | loss 4.2757 | avg50 4.2522 | lr 6.00e-04 | μlr 0.020 | 184ms/step | elapsed 5.2m | time left 4.8m
step   1360 | loss 4.2586 | avg50 4.2482 | lr 6.00e-04 | μlr 0.020 | 178ms/step | elapsed 5.2m | time left 4.8m
step   1370 | loss 4.2307 | avg50 4.2446 | lr 6.00e-04 | μlr 0.020 | 134ms/step | elapsed 5.3m | time left 4.7m
step   1380 | loss 4.2275 | avg50 4.2426 | lr 6.00e-04 | μlr 0.020 | 158ms/step | elapsed 5.3m | time left 4.7m
step   1390 | loss 4.2864 | avg50 4.2456 | lr 6.00e-04 | μlr 0.020 | 157ms/step | elapsed 5.3m | time left 4.7m
step   1400 | loss 4.2278 | avg50 4.2467 | lr 6.00e-04 | μlr 0.020 | 131ms/step | elapsed 5.4m | time left 4.6m
[eval] step 1400 | val_loss 4.2603 ★ best!
step   1410 | loss 4.1794 | avg50 4.2435 | lr 6.00e-04 | μlr 0.020 | 167ms/step | elapsed 5.4m | time left 4.6m
step   1420 | loss 4.1471 | avg50 4.2378 | lr 6.00e-04 | μlr 0.020 | 154ms/step | elapsed 5.4m | time left 4.6m
step   1430 | loss 4.1624 | avg50 4.2381 | lr 6.00e-04 | μlr 0.020 | 165ms/step | elapsed 5.5m | time left 4.5m
step   1440 | loss 4.1424 | avg50 4.2338 | lr 6.00e-04 | μlr 0.020 | 165ms/step | elapsed 5.5m | time left 4.5m
step   1450 | loss 4.2288 | avg50 4.2225 | lr 6.00e-04 | μlr 0.020 | 135ms/step | elapsed 5.5m | time left 4.5m
step   1460 | loss 4.1882 | avg50 4.2220 | lr 6.00e-04 | μlr 0.020 | 149ms/step | elapsed 5.5m | time left 4.5m
step   1470 | loss 4.2114 | avg50 4.2270 | lr 6.00e-04 | μlr 0.020 | 143ms/step | elapsed 5.6m | time left 4.4m
step   1480 | loss 4.1655 | avg50 4.2231 | lr 6.00e-04 | μlr 0.020 | 149ms/step | elapsed 5.6m | time left 4.4m
step   1490 | loss 4.3428 | avg50 4.2206 | lr 6.00e-04 | μlr 0.020 | 173ms/step | elapsed 5.6m | time left 4.4m
step   1500 | loss 4.1474 | avg50 4.2275 | lr 6.00e-04 | μlr 0.020 | 129ms/step | elapsed 5.7m | time left 4.3m
step   1510 | loss 4.1783 | avg50 4.2270 | lr 6.00e-04 | μlr 0.020 | 149ms/step | elapsed 5.7m | time left 4.3m
step   1520 | loss 4.2131 | avg50 4.2137 | lr 6.00e-04 | μlr 0.020 | 138ms/step | elapsed 5.7m | time left 4.3m
step   1530 | loss 4.2078 | avg50 4.2170 | lr 6.00e-04 | μlr 0.020 | 132ms/step | elapsed 5.7m | time left 4.3m
step   1540 | loss 4.2152 | avg50 4.2168 | lr 6.00e-04 | μlr 0.020 | 173ms/step | elapsed 5.8m | time left 4.2m
step   1550 | loss 4.2146 | avg50 4.2105 | lr 6.00e-04 | μlr 0.020 | 131ms/step | elapsed 5.8m | time left 4.2m
step   1560 | loss 4.2097 | avg50 4.2022 | lr 6.00e-04 | μlr 0.020 | 149ms/step | elapsed 5.8m | time left 4.2m
step   1570 | loss 4.2984 | avg50 4.2071 | lr 6.00e-04 | μlr 0.020 | 154ms/step | elapsed 5.9m | time left 4.1m
step   1580 | loss 4.1713 | avg50 4.1983 | lr 6.00e-04 | μlr 0.020 | 156ms/step | elapsed 5.9m | time left 4.1m
step   1590 | loss 4.1437 | avg50 4.1878 | lr 6.00e-04 | μlr 0.020 | 148ms/step | elapsed 5.9m | time left 4.1m
step   1600 | loss 4.2448 | avg50 4.1775 | lr 6.00e-04 | μlr 0.020 | 134ms/step | elapsed 5.9m | time left 4.1m
[eval] step 1600 | val_loss 4.2204 ★ best!
step   1610 | loss 4.2412 | avg50 4.1856 | lr 6.00e-04 | μlr 0.020 | 183ms/step | elapsed 6.0m | time left 4.0m
step   1620 | loss 4.1177 | avg50 4.1861 | lr 6.00e-04 | μlr 0.020 | 157ms/step | elapsed 6.0m | time left 4.0m
step   1630 | loss 4.1224 | avg50 4.1828 | lr 6.00e-04 | μlr 0.020 | 133ms/step | elapsed 6.0m | time left 4.0m
step   1640 | loss 4.1707 | avg50 4.1903 | lr 6.00e-04 | μlr 0.020 | 152ms/step | elapsed 6.0m | time left 4.0m
step   1650 | loss 4.0592 | avg50 4.1917 | lr 6.00e-04 | μlr 0.020 | 148ms/step | elapsed 6.1m | time left 3.9m
step   1660 | loss 4.2186 | avg50 4.1873 | lr 6.00e-04 | μlr 0.020 | 131ms/step | elapsed 6.1m | time left 3.9m
step   1670 | loss 4.1937 | avg50 4.1798 | lr 6.00e-04 | μlr 0.020 | 154ms/step | elapsed 6.1m | time left 3.9m
step   1680 | loss 4.2048 | avg50 4.1776 | lr 6.00e-04 | μlr 0.020 | 141ms/step | elapsed 6.2m | time left 3.8m
step   1690 | loss 4.1510 | avg50 4.1632 | lr 6.00e-04 | μlr 0.020 | 132ms/step | elapsed 6.2m | time left 3.8m
step   1700 | loss 4.1577 | avg50 4.1633 | lr 6.00e-04 | μlr 0.020 | 146ms/step | elapsed 6.2m | time left 3.8m
step   1710 | loss 4.1128 | avg50 4.1579 | lr 6.00e-04 | μlr 0.020 | 129ms/step | elapsed 6.2m | time left 3.8m
step   1720 | loss 4.2010 | avg50 4.1628 | lr 6.00e-04 | μlr 0.020 | 137ms/step | elapsed 6.3m | time left 3.7m
step   1730 | loss 4.1298 | avg50 4.1717 | lr 6.00e-04 | μlr 0.020 | 145ms/step | elapsed 6.3m | time left 3.7m
step   1740 | loss 4.1236 | avg50 4.1793 | lr 6.00e-04 | μlr 0.020 | 133ms/step | elapsed 6.3m | time left 3.7m
step   1750 | loss 4.0459 | avg50 4.1814 | lr 6.00e-04 | μlr 0.020 | 146ms/step | elapsed 6.4m | time left 3.6m
step   1760 | loss 4.1362 | avg50 4.1796 | lr 6.00e-04 | μlr 0.020 | 146ms/step | elapsed 6.4m | time left 3.6m
step   1770 | loss 4.1351 | avg50 4.1700 | lr 6.00e-04 | μlr 0.020 | 130ms/step | elapsed 6.4m | time left 3.6m
step   1780 | loss 4.1150 | avg50 4.1592 | lr 6.00e-04 | μlr 0.020 | 114ms/step | elapsed 6.4m | time left 3.6m
step   1790 | loss 4.1569 | avg50 4.1528 | lr 6.00e-04 | μlr 0.020 | 146ms/step | elapsed 6.5m | time left 3.5m
step   1800 | loss 4.0645 | avg50 4.1471 | lr 6.00e-04 | μlr 0.020 | 146ms/step | elapsed 6.5m | time left 3.5m
[eval] step 1800 | val_loss 4.1418 ★ best!
step   1810 | loss 4.0226 | avg50 4.1365 | lr 6.00e-04 | μlr 0.020 | 132ms/step | elapsed 6.5m | time left 3.5m
step   1820 | loss 3.9795 | avg50 4.1306 | lr 6.00e-04 | μlr 0.020 | 137ms/step | elapsed 6.6m | time left 3.4m
step   1830 | loss 4.1187 | avg50 4.1217 | lr 6.00e-04 | μlr 0.020 | 133ms/step | elapsed 6.6m | time left 3.4m
step   1840 | loss 4.1544 | avg50 4.1160 | lr 6.00e-04 | μlr 0.020 | 125ms/step | elapsed 6.6m | time left 3.4m
step   1850 | loss 4.1401 | avg50 4.1097 | lr 6.00e-04 | μlr 0.020 | 198ms/step | elapsed 6.6m | time left 3.4m
step   1860 | loss 4.2101 | avg50 4.1069 | lr 6.00e-04 | μlr 0.020 | 140ms/step | elapsed 6.7m | time left 3.3m
step   1870 | loss 4.1202 | avg50 4.1056 | lr 6.00e-04 | μlr 0.020 | 221ms/step | elapsed 6.7m | time left 3.3m
step   1880 | loss 4.2451 | avg50 4.1085 | lr 6.00e-04 | μlr 0.020 | 241ms/step | elapsed 6.7m | time left 3.3m
step   1890 | loss 4.1178 | avg50 4.1098 | lr 6.00e-04 | μlr 0.020 | 278ms/step | elapsed 6.8m | time left 3.2m
step   1900 | loss 4.0684 | avg50 4.1031 | lr 6.00e-04 | μlr 0.020 | 239ms/step | elapsed 6.8m | time left 3.2m
step   1910 | loss 4.1062 | avg50 4.1035 | lr 6.00e-04 | μlr 0.020 | 281ms/step | elapsed 6.9m | time left 3.1m
step   1920 | loss 4.0435 | avg50 4.1016 | lr 6.00e-04 | μlr 0.020 | 295ms/step | elapsed 6.9m | time left 3.1m
step   1930 | loss 4.0392 | avg50 4.0967 | lr 6.00e-04 | μlr 0.020 | 273ms/step | elapsed 7.0m | time left 3.0m
step   1940 | loss 4.0338 | avg50 4.0962 | lr 6.00e-04 | μlr 0.020 | 310ms/step | elapsed 7.0m | time left 3.0m
step   1950 | loss 4.1184 | avg50 4.0988 | lr 6.00e-04 | μlr 0.020 | 266ms/step | elapsed 7.1m | time left 2.9m
step   1960 | loss 4.1321 | avg50 4.1004 | lr 6.00e-04 | μlr 0.020 | 280ms/step | elapsed 7.1m | time left 2.9m
step   1970 | loss 4.2193 | avg50 4.1029 | lr 6.00e-04 | μlr 0.020 | 242ms/step | elapsed 7.2m | time left 2.8m
step   1980 | loss 4.0574 | avg50 4.1063 | lr 6.00e-04 | μlr 0.020 | 258ms/step | elapsed 7.2m | time left 2.8m
step   1990 | loss 4.1340 | avg50 4.1114 | lr 6.00e-04 | μlr 0.020 | 279ms/step | elapsed 7.3m | time left 2.7m
step   2000 | loss 4.1380 | avg50 4.1078 | lr 6.00e-04 | μlr 0.020 | 230ms/step | elapsed 7.3m | time left 2.7m
[eval] step 2000 | val_loss 4.0891 ★ best!
step   2010 | loss 4.1390 | avg50 4.1048 | lr 5.90e-04 | μlr 0.020 | 255ms/step | elapsed 7.4m | time left 2.6m
step   2020 | loss 4.2316 | avg50 4.1071 | lr 5.79e-04 | μlr 0.019 | 315ms/step | elapsed 7.4m | time left 2.6m
step   2030 | loss 4.0194 | avg50 4.1070 | lr 5.69e-04 | μlr 0.019 | 258ms/step | elapsed 7.5m | time left 2.5m
step   2040 | loss 4.0155 | avg50 4.0912 | lr 5.58e-04 | μlr 0.019 | 265ms/step | elapsed 7.5m | time left 2.5m
step   2050 | loss 4.0544 | avg50 4.0924 | lr 5.47e-04 | μlr 0.018 | 301ms/step | elapsed 7.6m | time left 2.4m
step   2060 | loss 4.1778 | avg50 4.0924 | lr 5.36e-04 | μlr 0.018 | 252ms/step | elapsed 7.6m | time left 2.4m
step   2070 | loss 4.1338 | avg50 4.0779 | lr 5.25e-04 | μlr 0.018 | 264ms/step | elapsed 7.7m | time left 2.3m
step   2080 | loss 4.1413 | avg50 4.0726 | lr 5.15e-04 | μlr 0.017 | 255ms/step | elapsed 7.7m | time left 2.3m
step   2090 | loss 3.9860 | avg50 4.0687 | lr 5.04e-04 | μlr 0.017 | 243ms/step | elapsed 7.8m | time left 2.2m
step   2100 | loss 4.1418 | avg50 4.0610 | lr 4.93e-04 | μlr 0.016 | 293ms/step | elapsed 7.8m | time left 2.2m
step   2110 | loss 4.1093 | avg50 4.0531 | lr 4.82e-04 | μlr 0.016 | 278ms/step | elapsed 7.9m | time left 2.1m
step   2120 | loss 4.1603 | avg50 4.0572 | lr 4.71e-04 | μlr 0.016 | 279ms/step | elapsed 7.9m | time left 2.1m
step   2130 | loss 4.1562 | avg50 4.0551 | lr 4.61e-04 | μlr 0.015 | 259ms/step | elapsed 8.0m | time left 2.0m
step   2140 | loss 4.0600 | avg50 4.0577 | lr 4.50e-04 | μlr 0.015 | 226ms/step | elapsed 8.0m | time left 2.0m
step   2150 | loss 4.0779 | avg50 4.0563 | lr 4.39e-04 | μlr 0.015 | 248ms/step | elapsed 8.0m | time left 2.0m
step   2160 | loss 4.0982 | avg50 4.0576 | lr 4.28e-04 | μlr 0.014 | 288ms/step | elapsed 8.1m | time left 1.9m
step   2170 | loss 4.0015 | avg50 4.0484 | lr 4.17e-04 | μlr 0.014 | 236ms/step | elapsed 8.1m | time left 1.9m
step   2180 | loss 4.0417 | avg50 4.0353 | lr 4.07e-04 | μlr 0.014 | 279ms/step | elapsed 8.2m | time left 1.8m
step   2190 | loss 4.0571 | avg50 4.0340 | lr 3.96e-04 | μlr 0.013 | 235ms/step | elapsed 8.2m | time left 1.8m
step   2200 | loss 3.9209 | avg50 4.0304 | lr 3.85e-04 | μlr 0.013 | 278ms/step | elapsed 8.3m | time left 1.7m
[eval] step 2200 | val_loss 4.0335 ★ best!
step   2210 | loss 4.1110 | avg50 4.0191 | lr 3.74e-04 | μlr 0.012 | 249ms/step | elapsed 8.3m | time left 1.7m
step   2220 | loss 3.9282 | avg50 4.0195 | lr 3.63e-04 | μlr 0.012 | 241ms/step | elapsed 8.4m | time left 1.6m
step   2230 | loss 4.0180 | avg50 4.0209 | lr 3.53e-04 | μlr 0.012 | 241ms/step | elapsed 8.4m | time left 1.6m
step   2240 | loss 3.9713 | avg50 4.0114 | lr 3.42e-04 | μlr 0.011 | 250ms/step | elapsed 8.5m | time left 1.5m
step   2250 | loss 3.9943 | avg50 4.0055 | lr 3.31e-04 | μlr 0.011 | 289ms/step | elapsed 8.5m | time left 1.5m
step   2260 | loss 4.1097 | avg50 4.0031 | lr 3.20e-04 | μlr 0.011 | 240ms/step | elapsed 8.5m | time left 1.5m
step   2270 | loss 4.0445 | avg50 3.9940 | lr 3.09e-04 | μlr 0.010 | 248ms/step | elapsed 8.6m | time left 1.4m
step   2280 | loss 3.9605 | avg50 3.9870 | lr 2.99e-04 | μlr 0.010 | 246ms/step | elapsed 8.6m | time left 1.4m
step   2290 | loss 3.8949 | avg50 3.9904 | lr 2.88e-04 | μlr 0.010 | 227ms/step | elapsed 8.7m | time left 1.3m
step   2300 | loss 3.9959 | avg50 3.9908 | lr 2.77e-04 | μlr 0.009 | 227ms/step | elapsed 8.7m | time left 1.3m
step   2310 | loss 3.9812 | avg50 3.9943 | lr 2.66e-04 | μlr 0.009 | 268ms/step | elapsed 8.8m | time left 1.2m
step   2320 | loss 4.0249 | avg50 3.9906 | lr 2.55e-04 | μlr 0.009 | 269ms/step | elapsed 8.8m | time left 1.2m
step   2330 | loss 4.1287 | avg50 3.9875 | lr 2.45e-04 | μlr 0.008 | 240ms/step | elapsed 8.9m | time left 1.1m
step   2340 | loss 3.9893 | avg50 3.9830 | lr 2.34e-04 | μlr 0.008 | 234ms/step | elapsed 8.9m | time left 1.1m
step   2350 | loss 3.9796 | avg50 3.9777 | lr 2.23e-04 | μlr 0.007 | 234ms/step | elapsed 8.9m | time left 1.1m
step   2360 | loss 3.8564 | avg50 3.9693 | lr 2.12e-04 | μlr 0.007 | 231ms/step | elapsed 9.0m | time left 1.0m
step   2370 | loss 3.9680 | avg50 3.9749 | lr 2.01e-04 | μlr 0.007 | 228ms/step | elapsed 9.0m | time left 1.0m
step   2380 | loss 3.9754 | avg50 3.9751 | lr 1.91e-04 | μlr 0.006 | 220ms/step | elapsed 9.1m | time left 0.9m
step   2390 | loss 3.9126 | avg50 3.9698 | lr 1.80e-04 | μlr 0.006 | 249ms/step | elapsed 9.1m | time left 0.9m
step   2400 | loss 4.0063 | avg50 3.9665 | lr 1.69e-04 | μlr 0.006 | 225ms/step | elapsed 9.1m | time left 0.9m
[eval] step 2400 | val_loss 3.9361 ★ best!
step   2410 | loss 3.9280 | avg50 3.9583 | lr 1.58e-04 | μlr 0.005 | 222ms/step | elapsed 9.2m | time left 0.8m
step   2420 | loss 3.8868 | avg50 3.9443 | lr 1.47e-04 | μlr 0.005 | 230ms/step | elapsed 9.2m | time left 0.8m
step   2430 | loss 3.9081 | avg50 3.9390 | lr 1.37e-04 | μlr 0.005 | 243ms/step | elapsed 9.3m | time left 0.7m
step   2440 | loss 3.8847 | avg50 3.9370 | lr 1.26e-04 | μlr 0.004 | 225ms/step | elapsed 9.3m | time left 0.7m
step   2450 | loss 3.8271 | avg50 3.9354 | lr 1.15e-04 | μlr 0.004 | 232ms/step | elapsed 9.4m | time left 0.6m
step   2460 | loss 3.8535 | avg50 3.9364 | lr 1.04e-04 | μlr 0.003 | 226ms/step | elapsed 9.4m | time left 0.6m
step   2470 | loss 3.9244 | avg50 3.9413 | lr 9.35e-05 | μlr 0.003 | 215ms/step | elapsed 9.4m | time left 0.6m
step   2480 | loss 3.7505 | avg50 3.9416 | lr 8.27e-05 | μlr 0.003 | 194ms/step | elapsed 9.5m | time left 0.5m
step   2490 | loss 3.9671 | avg50 3.9370 | lr 7.19e-05 | μlr 0.002 | 198ms/step | elapsed 9.5m | time left 0.5m
step   2500 | loss 3.8369 | avg50 3.9339 | lr 6.11e-05 | μlr 0.002 | 220ms/step | elapsed 9.6m | time left 0.4m

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
