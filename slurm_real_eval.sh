#!/bin/bash
#SBATCH --job-name=real-eval
#SBATCH --gres=gpu:1
#SBATCH --output=/data/%u/slurm-logs/%j.out
#SBATCH --error=/data/%u/slurm-logs/%j.err

set -euo pipefail

cd /data/dongjie/1b-omni/Smoke-test-1b
source .venv/bin/activate

export TMPDIR=/data/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /data/$USER/slurm-logs
mkdir -p /data/$USER/smoke-test-1b-results

# 可选：为 COMET 和 CLIPScore 准备依赖
# pip install unbabel-comet

python -m project.run_real_eval --spec eval_spec_real_small.yaml