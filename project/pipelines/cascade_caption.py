from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any

from project.datasets.utils import read_prompt
from project.model_adapters.andesvl_adapter import AndesVLAdapter
from project.model_adapters.qwen3_asr_adapter import Qwen3ASRAdapter


def run_caption_cascade(samples: Iterable[Dict[str, Any]], andes: AndesVLAdapter, prompt_path: str, output_jsonl: str, condition: str, asr: Qwen3ASRAdapter | None = None):
    prompt = read_prompt(prompt_path)
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            final_prompt = prompt
            transcript = None
            input_mode = "image"
            if asr and s.get("audio_path"):
                transcript = asr.transcribe_file(s["audio_path"], language=s.get("language"))["text"]
                final_prompt = transcript + "\n" + prompt
                input_mode = "audio+image"
            pred = andes.generate_from_image(s["image_path"], final_prompt)
            row = {
                "sample_id": s.get("sample_id"), "task": "caption", "dataset": s.get("dataset"), "condition": condition,
                "input_mode": input_mode, "source_text": None, "audio_path": s.get("audio_path"), "image_path": s.get("image_path"),
                "transcript": transcript, "prompt": final_prompt, "raw_model_output": pred["raw_response"], "parsed_output": pred["text"],
                "reference": s.get("reference_text"), "metadata": s.get("metadata") or {},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
