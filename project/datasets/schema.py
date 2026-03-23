from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import json

SAMPLE_FIELDS = [
    "sample_id",
    "dataset",
    "split",
    "task",
    "language",
    "source_language",
    "target_language",
    "text",
    "reference_text",
    "audio_path",
    "audio_array",
    "sampling_rate",
    "image_path",
    "image",
    "captions",
    "label",
    "metadata",
]

MANIFEST_FIELDS = [
    "sample_id",
    "dataset",
    "split",
    "task",
    "text",
    "reference_text",
    "audio_path",
    "image_path",
    "label",
    "language",
    "source_language",
    "target_language",
    "metadata",
]


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw = deepcopy(sample)
    normalized: Dict[str, Any] = {k: None for k in SAMPLE_FIELDS}
    for key in SAMPLE_FIELDS:
        if key in sample:
            normalized[key] = sample[key]

    for k in ("text", "reference_text"):
        if isinstance(normalized[k], str):
            normalized[k] = normalized[k].strip()

    for k in ("audio_path", "image_path"):
        if normalized[k]:
            normalized[k] = str(Path(normalized[k]).expanduser().resolve())

    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("raw", raw)
    normalized["metadata"] = metadata
    return normalized


def to_manifest_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample = normalize_sample(sample)
    record = {k: sample.get(k) for k in MANIFEST_FIELDS}
    meta = record.get("metadata")
    if isinstance(meta, dict):
        meta = dict(meta)
        meta.pop("raw", None)
        serializable_meta = {}
        for k, v in meta.items():
            try:
                json.dumps(v, ensure_ascii=False)
                serializable_meta[k] = v
            except Exception:
                serializable_meta[k] = str(v)
        record["metadata"] = serializable_meta
    else:
        record["metadata"] = {}
    return record
