# Lightweight Training-Free Tri-modal Validation Framework

本项目用于 **完全不训练（100% training-free）** 的前期验证：评估 AndesVL-0_6B 与 Qwen3-ASR-0.6B 级联的上限、语音入口损失、噪声损失与多模态增益。

## 1) 依赖安装
```bash
pip install torch transformers datasets qwen-asr soundfile librosa pyyaml sacrebleu
# 可选
pip install unbabel-comet vllm
```

## 2) 运行 model_probe
```bash
python -m project.model_adapters.model_probe
```
输出: `project/logs/model_probe_summary.txt`

## 3) 数据集 loader 最小示例
```bash
python - <<'PY'
from project.datasets.fleurs_loader import prepare_fleurs_asr_samples
print(len(prepare_fleurs_asr_samples('zh_cn','test',limit=5)))
PY
```

```bash
python - <<'PY'
from project.datasets.wmt19_loader import prepare_wmt19_translation_samples
print(len(prepare_wmt19_translation_samples('zh_to_en', split='validation', limit=5)))
PY
```

```bash
python - <<'PY'
from project.datasets.flickr30k_loader import prepare_flickr30k_caption_samples
print(len(prepare_flickr30k_caption_samples(split='test', limit=5)))
PY
```

```bash
python - <<'PY'
from project.datasets.esc50_loader import load_esc50
bank = load_esc50('/path/to/ESC-50')
print(len(bank.rows))
PY
```

```bash
python -m project.pipelines.tts_manifest_builder
```

```bash
python - <<'PY'
from project.datasets.fleurs_loader import prepare_fleurs_asr_samples
from project.datasets.esc50_loader import load_esc50
from project.pipelines.noise_augment import build_noisy_asr_samples
samples = prepare_fleurs_asr_samples('en_us','test',limit=5)
bank = load_esc50('/path/to/ESC-50')
noisy = build_noisy_asr_samples(samples, bank.sample_noise_clips(n=5), snr_db=10)
print(len(noisy))
PY
```

## 4) 实验脚本命令
- 实验A（ASR基线）: 在 Python 中调用 `project.pipelines.asr_infer.run_asr(...)`，再运行 `python -m project.eval.eval_asr`
- 实验B（Intent级联）: 调用 `project.pipelines.cascade_intent.run_intent_cascade(...)`，再运行 `python -m project.eval.eval_intent`
- 实验C（Translation级联）: 调用 `project.pipelines.cascade_translation.run_translation_cascade(...)`，再运行 `python -m project.eval.eval_translation`
- 实验D（Caption基线）: 调用 `project.pipelines.cascade_caption.run_caption_cascade(...)`，再运行 `python -m project.eval.eval_caption`
- 实验E（Deploy，可选）: `python -m project.eval.eval_deploy`

### 主线闭环验证（mock，仅验证字段贯通，不代表真实模型效果）
```bash
python -m project.pipelines.mock_closed_loop_validation
python -m project.eval.eval_asr
python -m project.eval.eval_intent
python -m project.eval.eval_translation
python -m project.eval.eval_caption
```

说明：上述 `raw_outputs` 每条记录会包含 `is_mock=true`，用于显式区分真实实验结果。

## 5) 结果表字段
### metrics CSV
`exp_name, task, dataset, condition, model_chain, prompt_name, metric_name, metric_value, num_samples, seed, notes`

### raw_outputs JSONL
`sample_id, task, dataset, condition, input_mode, source_text, audio_path, image_path, transcript, prompt, raw_model_output, parsed_output, reference, metadata`

## 6) 常见失败与 fallback
- AndesVL 无 `.chat`: 记录 `logs/model_probe_andesvl.txt`，对应实验 skip。
- qwen_asr 缺失: 报安装提示，不阻塞其他模块。
- backend=vllm 依赖缺失: 自动降级 transformers，写日志。
- 数据集不可下载: 写 `logs/dataset_failures.log`，可返回空/stub。
- 可选扩展失败: 不影响主线交付。
