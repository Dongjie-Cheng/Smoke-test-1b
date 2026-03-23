from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List

from .schema import normalize_sample


def _map_fields(row: Dict[str, Any], field_map: Dict[str, str] | None):
    if not field_map:
        return row
    out = {}
    for src, value in row.items():
        out[field_map.get(src, src)] = value
    return out


def load_custom_manifest(path: str, field_map: Dict[str, str] | None = None, default_fields: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    p = Path(path)
    records = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            records.extend(csv.DictReader(f))
    else:
        raise ValueError("custom manifest only supports .jsonl/.csv")

    out = []
    for idx, row in enumerate(records):
        mapped = _map_fields(row, field_map)
        merged = {**(default_fields or {}), **mapped}
        merged.setdefault("sample_id", f"custom-{idx}")
        merged.setdefault("dataset", "custom_manifest")
        merged.setdefault("split", "custom")
        out.append(normalize_sample(merged))
    return out
