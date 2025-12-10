#!/usr/bin/env bash
#SBATCH --job-name=reatt
#SBATCH --output=out/slurm/%x-%j.out
#SBATCH --error=out/slurm/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

# Optional: site modules (ignore if not available)
module load python/3.11 || true

# --------- Paths (edit if needed) ----------
REPO_DIR="/home/tbday/minGPT"
VENV_DIR="$REPO_DIR/.venv"
HF_HOME_DIR="/home/tbday/nobackup/autodelete/hf"
DATA_PATH="/home/tbday/minGPT/pile_data_10_first_10jsonl"
# Each job gets its own work dir by default
WORK_DIR="$REPO_DIR/out/reatt_batch_${SLURM_JOB_ID}"
# ------------------------------------------

# Tunables (override via: sbatch --export=ALL,BLOCK_SIZE=256,MAX_ITERS=2000,...)
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_ITERS="${MAX_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"          # auto | cpu | cuda | cuda:0
TEST_SIZE="${TEST_SIZE:-10}"
SWIGLU="${SWIGLU:-False}"

mkdir -p "$HF_HOME_DIR" "$REPO_DIR/out/slurm"
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Using device: $DEVICE"
echo "Data path: $DATA_PATH"
echo "Work dir: $WORK_DIR"

# Use venv python to avoid network installs on compute node
python -m projects.reatt.reatt \
  --data.file_path="$DATA_PATH" \
  --data.block_size="$BLOCK_SIZE" \
  --data.test_size="$TEST_SIZE" \
  --system.work_dir="$WORK_DIR" \
  --trainer.max_iters="$MAX_ITERS" \
  --trainer.batch_size="$BATCH_SIZE" \
  --trainer.num_workers="$NUM_WORKERS" \
  --trainer.device="$DEVICE"
  --model.swiglu="$SWIGLU"



