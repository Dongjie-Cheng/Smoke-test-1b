from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.fleurs_loader import prepare_fleurs_asr_samples
from project.datasets.flickr30k_loader import prepare_flickr30k_caption_samples
from project.datasets.utils import write_manifest
from project.datasets.wmt19_loader import prepare_wmt19_translation_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="project/configs/datasets.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(cfg.get("output_root", "data_manifests")) / "real_full"
    out_root.mkdir(parents=True, exist_ok=True)
    full_cfg = cfg.get("real_full", {})

    fleurs_cfg = cfg.get("fleurs", {})
    asr_rows = []
    for lang in ("zh_cn", "en_us"):
        asr_rows.extend(
            prepare_fleurs_asr_samples(
                lang_config=lang,
                split=fleurs_cfg.get("split", "test"),
                limit=full_cfg.get("asr", {}).get(lang),
                trust_remote_code=fleurs_cfg.get("trust_remote_code"),
                sampling_rate=fleurs_cfg.get("sampling_rate", 16000),
            )
        )
    write_manifest(asr_rows, out_root / "asr_clean.jsonl")

    ext_intent = (cfg.get("external_manifests", {}) or {}).get("intent_clean")
    if ext_intent:
        intent_samples = load_custom_manifest(ext_intent, default_fields={"task": "intent", "dataset": "intent_custom", "split": "external"})
        write_manifest(intent_samples, out_root / "intent_clean.jsonl")

    wmt_cfg = cfg.get("wmt19", {})
    for direction in ("zh_to_en", "en_to_zh"):
        samples = prepare_wmt19_translation_samples(
            direction=direction,
            config_name=wmt_cfg.get("config_name", "zh-en"),
            split=wmt_cfg.get("split", "validation"),
            limit=full_cfg.get("translation", {}).get(direction),
            trust_remote_code=wmt_cfg.get("trust_remote_code"),
        )
        write_manifest(samples, out_root / f"translation_{direction}_clean.jsonl")

    flickr_cfg = cfg.get("flickr30k", {})
    caption_rows = prepare_flickr30k_caption_samples(
        split=flickr_cfg.get("split", "test"),
        limit=full_cfg.get("caption", 500),
        reference_mode=flickr_cfg.get("reference_mode", "single_reference"),
        trust_remote_code=flickr_cfg.get("trust_remote_code"),
    )
    write_manifest(caption_rows, out_root / "caption_clean.jsonl")

    print(f"[ok] wrote manifests under {out_root}")


if __name__ == "__main__":
    main()
