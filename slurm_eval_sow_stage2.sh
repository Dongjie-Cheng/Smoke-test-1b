#!/bin/bash
#SBATCH --job-name=eval-sow-stage2
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

TTS_AUDIO_ROOT=${TTS_AUDIO_ROOT:-}
ESC50_ROOT=${ESC50_ROOT:-}
CMD=(python -m project.tools.build_sow_translation_audio_manifests --stage-dir data_manifests/sow_stage2 --snr-db 20)
if [ -n "$TTS_AUDIO_ROOT" ]; then
  CMD+=(--tts-audio-root "$TTS_AUDIO_ROOT")
fi
if [ -n "$ESC50_ROOT" ]; then
  CMD+=(--esc50-root "$ESC50_ROOT")
fi
"${CMD[@]}" || true

python -m project.run_eval_sow_aligned --spec eval_spec_sow_stage2.yaml
