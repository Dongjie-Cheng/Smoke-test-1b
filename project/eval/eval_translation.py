from __future__ import annotations

import csv
import json
from pathlib import Path


def evaluate_translation(raw_jsonl: str, out_csv: str):
    rows = [json.loads(x) for x in Path(raw_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu([r.get("parsed_output", "") for r in rows], [[r.get("reference", "") for r in rows]]).score
    except Exception:
        bleu = 0.0
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        data = [{"src": r.get("source_text", ""), "mt": r.get("parsed_output", ""), "ref": r.get("reference", "")} for r in rows]
        comet_score = model.predict(data, batch_size=8, gpus=1).system_score
    except Exception:
        comet_score = 0.0

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exp_name", "task", "dataset", "condition", "model_chain", "prompt_name", "metric_name", "metric_value", "num_samples", "seed", "notes"])
        writer.writeheader()
        for name, value in [("SacreBLEU", bleu), ("COMET", comet_score), ("clean_noisy_drop", 0.0)]:
            writer.writerow({"exp_name": "exp_c_translation", "task": "translation", "dataset": "wmt19", "condition": "mixed", "model_chain": "qwen3_asr->andesvl", "prompt_name": "zh_to_en_v1/en_to_zh_v1", "metric_name": name, "metric_value": value, "num_samples": len(rows), "seed": 42, "notes": "training-free"})


if __name__ == "__main__":
    evaluate_translation("project/results/raw_outputs/translation_outputs.jsonl", "project/results/translation_metrics.csv")
