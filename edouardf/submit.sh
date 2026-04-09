#!/bin/bash
#SBATCH --job-name=edouardf-v11
#SBATCH --partition=gpus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ── Rendezvous info derived from SLURM ──────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "============================================="
echo "Job $SLURM_JOB_ID — v11 — $(date)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $NNODES × $NPROC_PER_NODE GPUs = $(($NNODES * $NPROC_PER_NODE)) total"
echo "============================================="

export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

# ── v11 on 32 GPUs ──────────────────────────────────────────────────────────
# auto grad_accum = max(1, 8//32) = 1
# effective batch = 32 seqs/GPU × 1 accum × 32 GPUs = 1024 seqs/step
# 50% cooldown: decay starts at step 1750/3500
# batch scheduling: ramp 16→32 over first 1400 steps (40%)
# Based on v7 run: ~3500 steps achievable in 10 min without eval

srun python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    train.py \
        --data_dir         /home/data/ \
        --checkpoint_path  checkpoint_v11.pt \
        --batch_size       32 \
        --grad_accum_steps -1 \
        --max_lr           6e-4 \
        --muon_max_lr      0.02 \
        --warmup_steps     100 \
        --cooldown_frac    0.50 \
        --max_steps        3500 \
        --eval_interval    0 \
        --time_limit_min   10

echo "============================================="
echo "Job $SLURM_JOB_ID finished — $(date)"
echo "============================================="
