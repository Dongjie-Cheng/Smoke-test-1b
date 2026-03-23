from __future__ import annotations

from typing import Dict, Any, List

from .schema import normalize_sample
from .utils import write_manifest, log_dataset_failure


def load_wmt19(config_name: str = "zh-en", split: str = "validation"):
    from datasets import load_dataset

    return load_dataset("wmt/wmt19", config_name, split=split)


def prepare_wmt19_translation_samples(direction: str, config_name: str = "zh-en", split: str = "validation", limit: int | None = None) -> List[Dict[str, Any]]:
    if direction not in {"zh_to_en", "en_to_zh"}:
        raise ValueError("direction must be zh_to_en|en_to_zh")
    try:
        ds = load_wmt19(config_name=config_name, split=split)
    except Exception as exc:
        log_dataset_failure("wmt19", f"config={config_name} split={split} error={exc}")
        return []
    samples = []
    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break
        tr = row["translation"]
        source, target = (tr["zh"], tr["en"]) if direction == "zh_to_en" else (tr["en"], tr["zh"])
        src_lang, tgt_lang = ("zh", "en") if direction == "zh_to_en" else ("en", "zh")
        samples.append(normalize_sample({
            "sample_id": f"wmt19-{split}-{idx}",
            "dataset": "wmt19",
            "split": split,
            "task": "translation",
            "source_language": src_lang,
            "target_language": tgt_lang,
            "text": source,
            "reference_text": target,
            "language": src_lang,
            "metadata": {"direction": direction},
        }))
    return samples


def build_tts_manifest_from_wmt19(direction: str, output_path: str, config_name: str = "zh-en", split: str = "validation", limit: int | None = None):
    samples = prepare_wmt19_translation_samples(direction=direction, config_name=config_name, split=split, limit=limit)
    return write_manifest(samples, output_path, fmt="jsonl")
