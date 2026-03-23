from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from .schema import normalize_sample
from .utils import write_jsonl, log_dataset_failure


SUPPORTED_DIRECTIONS = {"zh_to_en", "en_to_zh"}
FIXED_CONFIG_NAME = "zh-en"


def load_wmt19(
    config_name: str = FIXED_CONFIG_NAME,
    split: str = "validation",
    trust_remote_code: bool | None = None,
):
    if config_name != FIXED_CONFIG_NAME:
        raise ValueError(f"wmt19 mainline requires config_name={FIXED_CONFIG_NAME}")
    from datasets import load_dataset

    load_kwargs: Dict[str, Any] = {}
    if trust_remote_code is not None:
        load_kwargs["trust_remote_code"] = trust_remote_code
    return load_dataset("wmt/wmt19", config_name, split=split, **load_kwargs)


def prepare_wmt19_translation_samples(
    direction: str,
    config_name: str = FIXED_CONFIG_NAME,
    split: str = "validation",
    limit: int | None = None,
    trust_remote_code: bool | None = None,
) -> List[Dict[str, Any]]:
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError("direction must be zh_to_en|en_to_zh")
    try:
        ds = load_wmt19(config_name=config_name, split=split, trust_remote_code=trust_remote_code)
    except Exception as exc:
        log_dataset_failure(
            "wmt19",
            f"config={config_name} split={split} trust_remote_code={trust_remote_code} error={exc}",
        )
        return []
    samples = []
    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break
        tr = row["translation"]
        source, target = (tr["zh"], tr["en"]) if direction == "zh_to_en" else (tr["en"], tr["zh"])
        src_lang, tgt_lang = ("zh", "en") if direction == "zh_to_en" else ("en", "zh")
        samples.append(
            normalize_sample(
                {
                    "sample_id": f"wmt19-{split}-{direction}-{idx}",
                    "dataset": "wmt19",
                    "split": split,
                    "task": "translation",
                    "source_language": src_lang,
                    "target_language": tgt_lang,
                    "text": source,
                    "reference_text": target,
                    "language": src_lang,
                    "metadata": {
                        "build_source": "wmt/wmt19",
                        "direction": direction,
                        "split_source": split,
                        "limit_applied": limit,
                        "trust_remote_code": trust_remote_code,
                        "config_name": config_name,
                    },
                }
            )
        )
    return samples


def build_tts_manifest_from_wmt19(
    direction: str,
    output_path: str,
    config_name: str = FIXED_CONFIG_NAME,
    split: str = "validation",
    limit: int | None = None,
    trust_remote_code: bool | None = None,
    tts_voice: str | None = None,
):
    samples = prepare_wmt19_translation_samples(
        direction=direction,
        config_name=config_name,
        split=split,
        limit=limit,
        trust_remote_code=trust_remote_code,
    )
    tts_rows = []
    for s in samples:
        tts_rows.append(
            {
                "sample_id": s.get("sample_id"),
                "source_text": s.get("text"),
                "source_language": s.get("source_language"),
                "target_language": s.get("target_language"),
                "reference_text": s.get("reference_text"),
                "tts_voice": tts_voice,
                "tts_output_audio_path": None,
                "metadata": {
                    **(s.get("metadata") or {}),
                    "task": "translation_tts_prep",
                },
            }
        )
    out = Path(output_path)
    write_jsonl(tts_rows, out)
    return out
