#!/bin/bash
# Deploy v11 files to the submission directory and launch
# Usage: bash deploy_final.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Deploying v11 to submission directory ==="
echo "From: $SCRIPT_DIR/v11/"
echo "To:   $SUBMIT_DIR/"
echo ""

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[dry-run] Would copy:"
    echo "  v11/model.py → model.py"
    echo "  v11/train.py → train.py"
    echo "  submit.sh    → submit.sh"
    exit 0
fi

cp "$SCRIPT_DIR/v11/model.py" "$SUBMIT_DIR/model.py"
cp "$SCRIPT_DIR/v11/train.py" "$SUBMIT_DIR/train.py"
cp "$SCRIPT_DIR/submit.sh"    "$SUBMIT_DIR/submit.sh"

echo "✓ model.py copied"
echo "✓ train.py copied"
echo "✓ submit.sh copied"
echo ""
echo "Ready! From the submission directory, run:"
echo "  cd $SUBMIT_DIR"
echo "  sbatch submit.sh"
echo ""
echo "Or for a 2-GPU test run:"
echo "  CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29501 \\"
echo "    train.py --data_dir /home/data --checkpoint_path checkpoint_test.pt \\"
echo "    --max_steps 2500 --cooldown_frac 0.50"
