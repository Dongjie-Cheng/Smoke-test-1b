# Training-free 三模态级联诊断报告

## 1. 实验设置
- 固定随机种子: 42
- 采样温度: 0
- 模型链路: AndesVL-0_6B + Qwen3-ASR-0.6B

## 2. 诊断地图
| task | upper_bound(text/image/mm) | audio_entry_loss | noise_loss | multimodal_gain | notes |
|---|---:|---:|---:|---:|---|
| intent |  |  |  |  |  |
| translation |  |  |  |  |  |
| caption |  |  |  |  |  |

## 3. 主线实验汇总
- A ASR: `results/asr_metrics.csv`
- B Intent: `results/intent_metrics.csv`
- C Translation: `results/translation_metrics.csv`
- D Caption: `results/caption_metrics.csv`

## 4. 可选扩展
- AndesVL Thinking 对照
- Qwen3-ASR vLLM backend
- streaming inference smoke test
- Flickr30k Entities
- INT8 / INT4 deploy smoke test

## 5. skipped 与失败日志
- `logs/model_probe_summary.txt`
- `logs/dataset_failures.log`
- `logs/qwen3_asr_adapter.log`
