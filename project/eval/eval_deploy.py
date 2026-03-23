from __future__ import annotations

import csv
from pathlib import Path


def write_deploy_stub(out_csv: str = "project/results/deploy_metrics.csv"):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exp_name", "task", "dataset", "condition", "model_chain", "prompt_name", "metric_name", "metric_value", "num_samples", "seed", "notes"])
        writer.writeheader()
        writer.writerow({"exp_name": "exp_e_deploy_optional", "task": "deploy", "dataset": "n/a", "condition": "skipped", "model_chain": "andesvl+qwen3_asr", "prompt_name": "n/a", "metric_name": "status", "metric_value": 0, "num_samples": 0, "seed": 42, "notes": "optional smoke test; execute only if env supports quantization runtime"})


if __name__ == "__main__":
    write_deploy_stub()
