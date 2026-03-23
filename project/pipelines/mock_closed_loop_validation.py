from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.utils import write_jsonl
from project.pipelines.asr_infer import run_asr
from project.pipelines.cascade_caption import run_caption_cascade
from project.pipelines.cascade_intent import run_intent_cascade
from project.pipelines.cascade_translation import run_translation_cascade


class MockASRAdapter:
    is_mock = True

    def transcribe_file(self, audio_path: str, language: str | None = None) -> Dict[str, Any]:
        text = "mock transcript zh" if (language or "").startswith("zh") else "mock transcript en"
        return {"language": language or "unknown", "text": text, "timestamps": None, "raw": {"mock": True, "audio": audio_path}}


class MockAndesAdapter:
    is_mock = True

    def generate_from_text(self, text: str, prompt_name: str | None = None) -> Dict[str, Any]:
        if "device_control" in text or "分类器" in text:
            out = "device_control"
        elif "翻译" in text or "Translate" in text:
            out = "mock translation output"
        else:
            out = "mock text output"
        return {"text": out, "raw_response": {"mock": True, "input": text}, "metadata": {"prompt_name": prompt_name}}

    def generate_from_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        return {
            "text": "mock caption output",
            "raw_response": {"mock": True, "image_path": image_path, "prompt": prompt},
            "metadata": {"input_mode": "image"},
        }

    def generate_from_text_image(self, text: str, image_path: str) -> Dict[str, Any]:
        return {
            "text": "device_control" if "分类器" in text else "mock text image output",
            "raw_response": {"mock": True, "text": text, "image_path": image_path},
            "metadata": {"input_mode": "text+image"},
        }


def _mk_manifest(path: str, rows: List[Dict[str, Any]]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(out)


def _attach_mock_flag(raw_jsonl_path: str):
    p = Path(raw_jsonl_path)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        r["is_mock"] = True
        md = r.get("metadata") or {}
        md["is_mock"] = True
        r["metadata"] = md
    write_jsonl(rows, p)


def run_closed_loop_mock_validation():
    manifests_dir = Path("project/results/manifests")
    raw_dir = Path("project/results/raw_outputs")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) build minimal custom manifests (1-2 samples per task)
    asr_manifest = _mk_manifest(
        str(manifests_dir / "custom_asr.jsonl"),
        [
            {"sample_id": "asr-1", "task": "asr", "dataset": "custom_manifest", "split": "mock", "audio_path": "/tmp/a.wav", "reference_text": "mock transcript en", "language": "en"},
            {"sample_id": "asr-2", "task": "asr", "dataset": "custom_manifest", "split": "mock", "audio_path": "/tmp/b.wav", "reference_text": "mock transcript zh", "language": "zh"},
        ],
    )
    intent_manifest = _mk_manifest(
        str(manifests_dir / "custom_intent.jsonl"),
        [
            {"sample_id": "intent-1", "task": "intent", "dataset": "custom_manifest", "split": "mock", "text": "打开空调", "label": "device_control"},
            {"sample_id": "intent-2", "task": "intent", "dataset": "custom_manifest", "split": "mock", "audio_path": "/tmp/c.wav", "label": "device_control", "language": "zh"},
        ],
    )
    translation_manifest = _mk_manifest(
        str(manifests_dir / "custom_translation.jsonl"),
        [
            {"sample_id": "tr-1", "task": "translation", "dataset": "custom_manifest", "split": "mock", "text": "你好", "reference_text": "hello", "source_language": "zh", "target_language": "en"},
            {"sample_id": "tr-2", "task": "translation", "dataset": "custom_manifest", "split": "mock", "audio_path": "/tmp/d.wav", "text": "hello", "reference_text": "你好", "source_language": "en", "target_language": "zh"},
        ],
    )
    caption_manifest = _mk_manifest(
        str(manifests_dir / "custom_caption.jsonl"),
        [
            {"sample_id": "cap-1", "task": "caption", "dataset": "custom_manifest", "split": "mock", "image_path": "/tmp/e.jpg", "reference_text": "a caption"},
            {"sample_id": "cap-2", "task": "caption", "dataset": "custom_manifest", "split": "mock", "image_path": "/tmp/f.jpg", "audio_path": "/tmp/g.wav", "reference_text": "a caption 2", "language": "en"},
        ],
    )

    asr_samples = load_custom_manifest(asr_manifest)
    intent_samples = load_custom_manifest(intent_manifest)
    translation_samples = load_custom_manifest(translation_manifest)
    caption_samples = load_custom_manifest(caption_manifest)

    andes = MockAndesAdapter()
    asr = MockASRAdapter()

    # 2) run loader -> pipeline -> raw_outputs
    run_asr(asr_samples, asr, condition="mock_clean", output_jsonl="project/results/raw_outputs/asr_outputs.jsonl")
    run_intent_cascade(
        intent_samples,
        andes,
        asr,
        prompt_path="project/prompts/intent/closed_set_v1.txt",
        condition="mock_clean",
        output_jsonl="project/results/raw_outputs/intent_outputs.jsonl",
    )
    run_translation_cascade(
        translation_samples,
        andes,
        prompt_path="project/prompts/translation/zh_to_en_v1.txt",
        output_jsonl="project/results/raw_outputs/translation_outputs.jsonl",
        condition="mock_clean",
        asr=asr,
    )
    run_caption_cascade(
        caption_samples,
        andes,
        prompt_path="project/prompts/caption/caption_v1.txt",
        output_jsonl="project/results/raw_outputs/caption_outputs.jsonl",
        condition="mock_clean",
        asr=asr,
    )

    for fn in ["asr_outputs.jsonl", "intent_outputs.jsonl", "translation_outputs.jsonl", "caption_outputs.jsonl"]:
        _attach_mock_flag(str(raw_dir / fn))


if __name__ == "__main__":
    run_closed_loop_mock_validation()
