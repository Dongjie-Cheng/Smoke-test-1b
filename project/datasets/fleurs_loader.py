from __future__ import annotations

from typing import List, Dict, Any

from .schema import normalize_sample
from .utils import log_dataset_failure


SUPPORTED_FLEURS = {"en_us", "zh_cn"}


def load_fleurs(
    lang_config: str,
    split: str = "test",
    trust_remote_code: bool | None = None,
    sampling_rate: int = 16000,
):
    if lang_config not in SUPPORTED_FLEURS:
        raise ValueError(f"Mainline only supports {sorted(SUPPORTED_FLEURS)}")
    from datasets import Audio, load_dataset

    load_kwargs: Dict[str, Any] = {}
    if trust_remote_code is not None:
        load_kwargs["trust_remote_code"] = trust_remote_code
    ds = load_dataset("google/fleurs", lang_config, split=split, **load_kwargs)
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def prepare_fleurs_asr_samples(
    lang_config: str,
    split: str = "test",
    limit: int | None = None,
    trust_remote_code: bool | None = None,
    sampling_rate: int = 16000,
) -> List[Dict[str, Any]]:
    try:
        ds = load_fleurs(
            lang_config=lang_config,
            split=split,
            trust_remote_code=trust_remote_code,
            sampling_rate=sampling_rate,
        )
    except Exception as exc:
        log_dataset_failure(
            "fleurs",
            f"lang={lang_config} split={split} trust_remote_code={trust_remote_code} error={exc}",
        )
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
                    "sampling_rate": audio.get("sampling_rate", sampling_rate),
                    "metadata": {
                        "build_source": "google/fleurs",
                        "lang_config": lang_config,
                        "split_source": split,
                        "limit_applied": limit,
                        "trust_remote_code": trust_remote_code,
                        "speaker_id": row.get("speaker_id"),
                        "gender": row.get("gender"),
                        "audio_path_present": bool(audio.get("path")),
                    },
                }
            )
        )
    return samples


def prepare_fleurs_lid_samples(
    lang_config: str,
    split: str = "test",
    limit: int | None = None,
    trust_remote_code: bool | None = None,
    sampling_rate: int = 16000,
) -> List[Dict[str, Any]]:
    samples = prepare_fleurs_asr_samples(
        lang_config=lang_config,
        split=split,
        limit=limit,
        trust_remote_code=trust_remote_code,
        sampling_rate=sampling_rate,
    )
    for s in samples:
        s["task"] = "lid"
        s["label"] = s["language"]
        meta = s.get("metadata") or {}
        meta["task_purpose"] = "language_identification"
        meta["lid_label_source"] = "language"
        s["metadata"] = meta
    return samples
