#!/bin/bash
#SBATCH --job-name=eval-real-small
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

# 1) 数据层探测
python -m project.tools.data_build_probe

# 2) 构建 real_small manifests
python -m project.tools.build_real_small_manifests

# 3) 构建 translation 的 TTS 清单（不直接评测，只为后续补 audio 做准备）
python -m project.tools.build_translation_tts_manifests --limit 100 || true

# 4) 如果你已经有 intent_clean 和 ESC-50，可构建 noisy intent
# ESC50_ROOT 请按你的机器实际路径改
ESC50_ROOT=${ESC50_ROOT:-/path/to/ESC-50}
if [ -d "$ESC50_ROOT" ] && [ -f data_manifests/real_small/intent_clean.jsonl ]; then
  python -m project.tools.build_noisy_manifests \
    --clean-manifest data_manifests/real_small/intent_clean.jsonl \
    --esc50-root "$ESC50_ROOT" \
    --snr-db 20 \
    --output-manifest data_manifests/real_small/intent_noisy_snr20.jsonl \
    --output-audio-dir data_manifests/real_small/noisy_audio || true
fi

# 5) 真正跑 end-to-end eval
python -m project.run_eval_end2end --spec eval_spec_real_small.yaml