from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def _norm_en(s: str) -> str:
    return " ".join((s or "").strip().split())


def _norm_zh(s: str) -> str:
    return (s or "").strip()


def _edit_distance(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[-1][-1]


def evaluate_asr(raw_jsonl: str, out_csv: str):
    rows = [json.loads(x) for x in Path(raw_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    scores = {"en": [], "zh": []}
    by_condition = {}
    lid_hits = 0
    lid_total = 0
    for r in rows:
        ref = r.get("reference") or ""
        hyp = r.get("parsed_output") or ""
        meta = r.get("metadata") or {}
        lang = meta.get("reference_language") or meta.get("language") or ""
        pred_lang = meta.get("predicted_language")
        if pred_lang and lang:
            lid_total += 1
            lid_hits += int(str(pred_lang).split("_")[0] == str(lang).split("_")[0])
        if str(lang).startswith("zh"):
            ref_n, hyp_n = _norm_zh(ref), _norm_zh(hyp)
            denom = max(1, len(ref_n))
            err = _edit_distance(list(ref_n), list(hyp_n)) / denom
            scores["zh"].append(err)
        else:
            ref_n, hyp_n = _norm_en(ref), _norm_en(hyp)
            ref_t, hyp_t = ref_n.split(), hyp_n.split()
            denom = max(1, len(ref_t))
            err = _edit_distance(ref_t, hyp_t) / denom
            scores["en"].append(err)
        by_condition.setdefault(r.get("condition") or "unknown", []).append(err)

    clean_mean = sum(by_condition.get("clean", [])) / max(1, len(by_condition.get("clean", [])))
    noisy_vals = [v for k, vals in by_condition.items() if k != "clean" for v in vals]
    noisy_mean = sum(noisy_vals) / max(1, len(noisy_vals))
    clean_noisy_drop = noisy_mean - clean_mean if noisy_vals else 0.0

    metrics = [
        {"metric_name": "WER", "metric_value": sum(scores["en"]) / max(1, len(scores["en"])), "task": "asr"},
        {"metric_name": "CER", "metric_value": sum(scores["zh"]) / max(1, len(scores["zh"])), "task": "asr"},
        {"metric_name": "LID_accuracy", "metric_value": (lid_hits / lid_total) if lid_total else 0.0, "task": "asr"},
        {"metric_name": "clean_noisy_drop", "metric_value": clean_noisy_drop, "task": "asr"},
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exp_name", "task", "dataset", "condition", "model_chain", "prompt_name", "metric_name", "metric_value", "num_samples", "seed", "notes"])
        writer.writeheader()
        for m in metrics:
            writer.writerow({"exp_name": "exp_a_asr", "dataset": "mixed", "condition": "mixed", "model_chain": "qwen3_asr", "prompt_name": "n/a", "num_samples": len(rows), "seed": 42, "notes": "training-free", **m})


if __name__ == "__main__":
    evaluate_asr("project/results/raw_outputs/asr_outputs.jsonl", "project/results/asr_metrics.csv")
