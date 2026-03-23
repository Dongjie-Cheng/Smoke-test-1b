from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import yaml

from project.datasets.schema import normalize_sample, to_manifest_record


LOADER_MODULES = [
    "project.datasets.fleurs_loader",
    "project.datasets.wmt19_loader",
    "project.datasets.flickr30k_loader",
    "project.datasets.esc50_loader",
    "project.datasets.custom_manifest_loader",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="project/configs/datasets.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    for mod in LOADER_MODULES:
        importlib.import_module(mod)
    print("[ok] loader imports successful")

    wmt_cfg = cfg.get("wmt19", {})
    if wmt_cfg.get("config_name", "zh-en") != "zh-en":
        raise ValueError("wmt19.config_name must be zh-en")
    print("[ok] dataset parameter checks passed")

    out_root = Path(cfg.get("output_root", "data_manifests"))
    out_root.mkdir(parents=True, exist_ok=True)
    probe_file = out_root / ".probe_write_test"
    probe_file.write_text("ok", encoding="utf-8")
    probe_file.unlink(missing_ok=True)
    print(f"[ok] output writable: {out_root}")

    sample = normalize_sample(
        {
            "sample_id": "probe-1",
            "dataset": "probe",
            "split": "test",
            "task": "asr",
            "reference_text": "hello",
            "audio_array": [0.1, 0.2],
            "metadata": {"build_source": "probe"},
        }
    )
    record = to_manifest_record(sample)
    if "audio_array" in record:
        raise AssertionError("manifest should not include audio_array")
    print("[ok] schema manifest serialization passed")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
