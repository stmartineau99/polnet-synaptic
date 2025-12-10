#!/bin/bash
#SBATCH -p standard96s:shared
#SBATCH --job-name=snr_array
#SBATCH --array=0-9%5
#SBATCH --output=out/run1/logs/slurm-%A_%a.out
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

PARENT_DIR="out/run1"
OUT_DIR="$PARENT_DIR/task$SLURM_ARRAY_TASK_ID"
LOG_DIR="$PARENT_DIR/logs"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

source ~/.bashrc
micromamba activate polnet

# voxel size deepict
VOI_VSIZE=13.8

# select SNR range based on for current task
LOW_START=0.10
HIGH_START=0.15
STEP=0.05

LOW_VAL=$(printf "%.2f" "$(echo "scale=2; $LOW_START + $SLURM_ARRAY_TASK_ID * $STEP" | bc)")
HIGH_VAL=$(printf "%.2f" "$(echo "scale=2; $HIGH_START + $SLURM_ARRAY_TASK_ID * $STEP" | bc)")

echo "Job array task $SLURM_ARRAY_TASK_ID: running simulation with SNR range ($LOW_VAL, $HIGH_VAL)."

srun python all_features_argument.py \
  --out_dir $OUT_DIR \
  --ntomos 3 \
  --voi_vsize $VOI_VSIZE \
  --detector_snr $LOW_VAL $HIGH_VAL
