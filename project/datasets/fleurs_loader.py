from __future__ import annotations

from typing import List, Dict, Any

from .schema import normalize_sample
from .utils import log_dataset_failure


SUPPORTED_FLEURS = {"en_us", "zh_cn"}


def load_fleurs(lang_config: str, split: str = "test"):
    if lang_config not in SUPPORTED_FLEURS:
        raise ValueError(f"Mainline only supports {sorted(SUPPORTED_FLEURS)}")
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", lang_config, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def prepare_fleurs_asr_samples(lang_config: str, split: str = "test", limit: int | None = None) -> List[Dict[str, Any]]:
    try:
        ds = load_fleurs(lang_config=lang_config, split=split)
    except Exception as exc:
        log_dataset_failure("fleurs", f"lang={lang_config} split={split} error={exc}")
        return []
    samples = []
    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break
        audio = row.get("audio") or {}
        samples.append(
            normalize_sample(
                {
                    "sample_id": str(row.get("id", idx)),
                    "dataset": "fleurs",
                    "split": split,
                    "task": "asr",
                    "language": row.get("lang_id") or lang_config,
                    "reference_text": row.get("transcription"),
                    "audio_path": audio.get("path"),
                    "audio_array": audio.get("array"),
                    "sampling_rate": audio.get("sampling_rate", 16000),
                    "metadata": {
                        "lang_config": lang_config,
                        "audio_path": audio.get("path"),
                        "speaker_id": row.get("speaker_id"),
                        "gender": row.get("gender"),
                    },
                }
            )
        )
    return samples


def prepare_fleurs_lid_samples(lang_config: str, split: str = "test", limit: int | None = None) -> List[Dict[str, Any]]:
    samples = prepare_fleurs_asr_samples(lang_config, split, limit)
    for s in samples:
        s["task"] = "asr"
        s["label"] = s["language"]
    return samples
