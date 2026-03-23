from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from project.model_adapters.qwen3_asr_adapter import Qwen3ASRAdapter


def run_asr(samples: List[Dict[str, Any]], adapter: Qwen3ASRAdapter, condition: str, output_jsonl: str):
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            audio = s.get("audio_path")
            if audio is None:
                continue
            pred = adapter.transcribe_file(audio, language=s.get("language"))
            row = {
                "sample_id": s.get("sample_id"),
                "task": "asr",
                "dataset": s.get("dataset"),
                "condition": condition,
                "input_mode": "audio",
                "source_text": None,
                "audio_path": audio,
                "image_path": None,
                "transcript": pred.get("text"),
                "prompt": None,
                "raw_model_output": pred.get("raw"),
                "parsed_output": pred.get("text"),
                "reference": s.get("reference_text"),
                "metadata": {
                    "predicted_language": pred.get("language"),
                    "reference_language": s.get("language"),
                    **(s.get("metadata") or {}),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
