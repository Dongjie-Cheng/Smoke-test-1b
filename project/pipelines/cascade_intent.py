from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterable

from project.model_adapters.andesvl_adapter import AndesVLAdapter
from project.model_adapters.qwen3_asr_adapter import Qwen3ASRAdapter
from project.datasets.utils import read_prompt


def run_intent_cascade(samples: Iterable[Dict[str, Any]], andes: AndesVLAdapter, asr: Qwen3ASRAdapter | None, prompt_path: str, condition: str, output_jsonl: str):
    prompt_tmpl = read_prompt(prompt_path)
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            transcript = s.get("text")
            input_mode = "text"
            if asr and s.get("audio_path"):
                t = asr.transcribe_file(s["audio_path"], language=s.get("language"))
                transcript = t["text"]
                input_mode = "audio" if not s.get("image_path") else "audio+image"
            text_prompt = prompt_tmpl.format(input_text=transcript or "")
            pred = andes.generate_from_text_image(text_prompt, s["image_path"]) if s.get("image_path") else andes.generate_from_text(text_prompt, "closed_set_v1")
            row = {
                "sample_id": s.get("sample_id"), "task": "intent", "dataset": s.get("dataset"), "condition": condition,
                "input_mode": input_mode, "source_text": s.get("text"), "audio_path": s.get("audio_path"), "image_path": s.get("image_path"),
                "transcript": transcript, "prompt": text_prompt, "raw_model_output": pred["raw_response"],
                "parsed_output": pred["text"], "reference": s.get("label"), "metadata": s.get("metadata") or {},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
