from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.fleurs_loader import prepare_fleurs_asr_samples
from project.datasets.flickr30k_loader import prepare_flickr30k_caption_samples
from project.datasets.schema import normalize_sample
from project.datasets.utils import write_manifest
from project.datasets.wmt19_loader import prepare_wmt19_translation_samples, build_tts_manifest_from_wmt19


def _stub(path: Path, task: str, reason: str) -> None:
    write_manifest(
        [
            normalize_sample(
                {
                    "sample_id": f"stub-{path.stem}",
                    "dataset": "stub",
                    "split": "stub",
                    "task": task,
                    "metadata": {"build_source": "build_real_small_manifests.py", "reason": reason},
                }
            )
        ],
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="project/configs/datasets.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(cfg.get("output_root", "data_manifests")) / "real_small"
    out_root.mkdir(parents=True, exist_ok=True)

    fleurs_cfg = cfg.get("fleurs", {})
    small = cfg.get("real_small", {})
    asr_rows = []
    for lang in ("zh_cn", "en_us"):
        asr_rows.extend(
            prepare_fleurs_asr_samples(
                lang_config=lang,
                split=fleurs_cfg.get("split", "test"),
                limit=small.get("asr", {}).get(lang, 50),
                trust_remote_code=fleurs_cfg.get("trust_remote_code"),
                sampling_rate=fleurs_cfg.get("sampling_rate", 16000),
            )
        )
    write_manifest(asr_rows, out_root / "asr_clean.jsonl")

    ext_intent = (cfg.get("external_manifests", {}) or {}).get("intent_clean")
    intent_out = out_root / "intent_clean.jsonl"
    if ext_intent:
        intent_samples = load_custom_manifest(ext_intent, default_fields={"task": "intent", "dataset": "intent_custom", "split": "external"})
        for s in intent_samples:
            s["metadata"] = {
                **(s.get("metadata") or {}),
                "build_source": "external_manifest",
                "condition": "clean",
            }
        write_manifest(intent_samples, intent_out)
    else:
        _stub(intent_out, "intent", "external_manifests.intent_clean not provided")

    wmt_cfg = cfg.get("wmt19", {})
    for direction in ("zh_to_en", "en_to_zh"):
        samples = prepare_wmt19_translation_samples(
            direction=direction,
            config_name=wmt_cfg.get("config_name", "zh-en"),
            split=wmt_cfg.get("split", "validation"),
            limit=small.get("translation", {}).get(direction, 100),
            trust_remote_code=wmt_cfg.get("trust_remote_code"),
        )
        out_name = f"translation_{direction}_clean.jsonl"
        write_manifest(samples, out_root / out_name)
        build_tts_manifest_from_wmt19(
            direction=direction,
            output_path=str(out_root / f"translation_{direction}_tts_manifest.jsonl"),
            config_name=wmt_cfg.get("config_name", "zh-en"),
            split=wmt_cfg.get("split", "validation"),
            limit=small.get("translation", {}).get(direction, 100),
            trust_remote_code=wmt_cfg.get("trust_remote_code"),
            tts_voice=(cfg.get("tts", {}) or {}).get("voice"),
        )
        _stub(out_root / f"translation_{direction}_noisy_stub.jsonl", "translation", "source audio missing for translation noisy build")

    flickr_cfg = cfg.get("flickr30k", {})
    caption_rows = prepare_flickr30k_caption_samples(
        split=flickr_cfg.get("split", "test"),
        limit=small.get("caption", 100),
        reference_mode=flickr_cfg.get("reference_mode", "single_reference"),
        trust_remote_code=flickr_cfg.get("trust_remote_code"),
    )
    write_manifest(caption_rows, out_root / "caption_clean.jsonl")

    print(f"[ok] wrote manifests under {out_root}")


if __name__ == "__main__":
    main()
