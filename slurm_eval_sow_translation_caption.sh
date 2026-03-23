#!/bin/bash
#SBATCH --job-name=eval-sow-trans-cap
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

python -m project.tools.build_sow_aligned_manifests --stage stage2
python -m project.tools.build_sow_tts_audio --stage-dir data_manifests/sow_stage2
python -m project.tools.build_sow_translation_noisy --stage-dir data_manifests/sow_stage2 --snr-db 20
python -m project.run_eval_sow_aligned --spec eval_spec_sow_stage2.yaml
