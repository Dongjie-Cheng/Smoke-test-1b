from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

from .schema import to_manifest_record


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def write_manifest(samples: List[Dict[str, Any]], path: str | Path, fmt: str = "jsonl") -> Path:
    records = [to_manifest_record(s) for s in samples]
    out = Path(path)
    ensure_dir(out.parent)
    if fmt.lower() == "jsonl":
        return write_jsonl(records, out)

    fieldnames = list(records[0].keys()) if records else []
    if not fieldnames:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return out


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def log_dataset_failure(name: str, reason: str, log_path: str | Path = "project/logs/dataset_failures.log") -> None:
    out = Path(log_path)
    ensure_dir(out.parent)
    with out.open("a", encoding="utf-8") as f:
        f.write(f"{name}: {reason}\n")
