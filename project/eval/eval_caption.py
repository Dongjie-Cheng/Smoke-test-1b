from __future__ import annotations

import csv
import json
from pathlib import Path


def evaluate_caption(raw_jsonl: str, out_csv: str, review_jsonl: str):
    rows = [json.loads(x) for x in Path(raw_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    clips = 0.0
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        scores = []
        for r in rows:
            if not r.get("image_path"):
                continue
            image = Image.open(r["image_path"]).convert("RGB")
            inputs = proc(text=[r.get("parsed_output", "")], images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model(**inputs)
            scores.append(float(out.logits_per_image.squeeze().item()))
        clips = sum(scores) / max(1, len(scores))
    except Exception:
        clips = 0.0

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exp_name", "task", "dataset", "condition", "model_chain", "prompt_name", "metric_name", "metric_value", "num_samples", "seed", "notes"])
        writer.writeheader()
        writer.writerow({"exp_name": "exp_d_caption", "task": "caption", "dataset": "flickr30k", "condition": "mixed", "model_chain": "qwen3_asr->andesvl", "prompt_name": "caption_v1", "metric_name": "CLIPScore", "metric_value": clips, "num_samples": len(rows), "seed": 42, "notes": "training-free"})

    review = [{"sample_id": r.get("sample_id"), "image_path": r.get("image_path"), "pred": r.get("parsed_output"), "ref": r.get("reference")} for r in rows[:200]]
    Path(review_jsonl).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in review) + "\n", encoding="utf-8")


if __name__ == "__main__":
    evaluate_caption("project/results/raw_outputs/caption_outputs.jsonl", "project/results/caption_metrics.csv", "project/results/raw_outputs/caption_human_review.jsonl")
