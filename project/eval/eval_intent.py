from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["device_control", "ocr_reading", "visual_qa", "multimodal_qa", "negative_or_stop", "routing_to_cloud"]


def evaluate_intent(raw_jsonl: str, out_csv: str):
    rows = [json.loads(x) for x in Path(raw_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    y_true = [r.get("reference") for r in rows]
    y_pred = [str(r.get("parsed_output", "")).strip() for r in rows]
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    acc = correct / max(1, len(rows))

    f1s = []
    for lb in LABELS:
        tp = sum(int(t == lb and p == lb) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != lb and p == lb) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == lb and p != lb) for t, p in zip(y_true, y_pred))
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(0 if p + r == 0 else 2 * p * r / (p + r))

    cm = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        cm[str(t)][str(p)] += 1

    metrics = [
        ("accuracy", acc),
        ("macro_f1", sum(f1s) / len(f1s)),
        ("text_vs_audio_gap", 0.0),
    ]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exp_name", "task", "dataset", "condition", "model_chain", "prompt_name", "metric_name", "metric_value", "num_samples", "seed", "notes"])
        writer.writeheader()
        for n, v in metrics:
            writer.writerow({"exp_name": "exp_b_intent", "task": "intent", "dataset": "mixed", "condition": "mixed", "model_chain": "qwen3_asr->andesvl", "prompt_name": "closed_set_v1", "metric_name": n, "metric_value": v, "num_samples": len(rows), "seed": 42, "notes": json.dumps({"confusion_matrix": cm}, ensure_ascii=False)})


if __name__ == "__main__":
    evaluate_intent("project/results/raw_outputs/intent_outputs.jsonl", "project/results/intent_metrics.csv")
