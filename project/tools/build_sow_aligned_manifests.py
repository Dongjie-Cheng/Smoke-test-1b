from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.flickr30k_entities_loader import prepare_flickr30k_entities_samples
from project.datasets.schema import normalize_sample
from project.datasets.utils import write_manifest, write_jsonl
from project.datasets.wmt19_loader import prepare_wmt19_translation_samples


def _apply_meta(sample: dict[str, Any], **meta_fields: Any) -> dict[str, Any]:
    meta = dict(sample.get("metadata") or {})
    meta.update(meta_fields)
    sample["metadata"] = meta
    return sample


def _stub(task: str, reason: str, build_source: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return normalize_sample(
        {
            "sample_id": f"stub-{task}",
            "dataset": "stub",
            "split": "stub",
            "task": task,
            "metadata": {
                "sow_aligned": False,
                "proxy_baseline": False,
                "pending_audio": True,
                "stub_reason": reason,
                "build_source": build_source,
                "condition": "stub",
                **(extra or {}),
            },
        }
    )


def _write_tts_manifest(samples: list[dict[str, Any]], out_path: Path, direction: str) -> int:
    rows = []
    for s in samples:
        meta = dict(s.get("metadata") or {})
        rows.append(
            {
                "sample_id": s.get("sample_id"),
                "source_text": s.get("text"),
                "source_language": s.get("source_language"),
                "target_language": s.get("target_language"),
                "reference_text": s.get("reference_text"),
                "tts_voice": None,
                "tts_output_audio_path": None,
                "metadata": {
                    **meta,
                    "sow_aligned": True,
                    "proxy_baseline": False,
                    "pending_audio": True,
                    "stub_reason": None,
                    "condition": "tts_input",
                    "direction": direction,
                    "build_source": "build_sow_aligned_manifests.py",
                    "wmt_track_note": "SOW asks WMT14/19; reproducible public track uses WMT19 here",
                },
            }
        )
    write_jsonl(rows, out_path)
    return len(rows)


def _build_stage(out_dir: Path, cfg: dict[str, Any], stage: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {
        "stage2": {"translation_clean": 3000, "translation_noisy": 2000, "caption": 1000, "caption_review": 0},
        "stage3": {"translation_clean": 5000, "translation_noisy": 3000, "caption": 1000, "caption_review": 200},
    }[stage]
    stage_cfg = (cfg.get("sow", {}) or {}).get(stage, {})

    summary: dict[str, Any] = {
        "stage": stage,
        "translation_track": "WMT19 zh-en/en-zh as reproducible public SOW-aligned track",
        "targets": {
            "translation_clean": int(stage_cfg.get("translation_clean", sizes["translation_clean"])),
            "translation_noisy": int(stage_cfg.get("translation_noisy", sizes["translation_noisy"])),
            "caption": int(stage_cfg.get("caption_clean", sizes["caption"])),
        },
        "actual": {},
        "blocking_reason": {},
        "intent": {"proxy_baseline": True, "sow_aligned": False, "pending_self_test_set": True},
    }

    # ---- Translation clean + TTS ----
    trans_limit = summary["targets"]["translation_clean"]
    wmt_cfg = cfg.get("wmt19", {})
    total_clean = 0
    for direction in ("zh_to_en", "en_to_zh"):
        clean = prepare_wmt19_translation_samples(
            direction=direction,
            config_name="zh-en",
            split=wmt_cfg.get("split", "validation"),
            limit=trans_limit,
            trust_remote_code=wmt_cfg.get("trust_remote_code"),
        )
        clean = [
            _apply_meta(
                s,
                sow_aligned=True,
                proxy_baseline=False,
                pending_audio=False,
                stub_reason=None,
                build_source="wmt19",
                condition="clean_text",
                direction=direction,
            )
            for s in clean
        ]
        total_clean += len(clean)
        write_manifest(clean, out_dir / f"translation_{direction}_clean.jsonl")
        _write_tts_manifest(clean, out_dir / f"translation_{direction}_tts_manifest.jsonl", direction)

    summary["actual"]["translation_clean_total"] = total_clean
    if total_clean < summary["targets"]["translation_clean"] * 2:
        summary["blocking_reason"]["translation_clean"] = "dataset_access_or_environment_limits"

    for direction in ("zh_to_en", "en_to_zh"):
        write_manifest(
            [
                _stub(
                    task="translation",
                    reason="real_tts_audio_not_available",
                    build_source="build_sow_aligned_manifests.py",
                    extra={"direction": direction, "pending_audio": True, "condition": "audio_clean_pending", "sow_aligned": True},
                )
            ],
            out_dir / f"translation_{direction}_audio_clean.jsonl",
        )
        write_manifest(
            [
                _stub(
                    task="translation",
                    reason="audio_clean_manifest_not_ready",
                    build_source="build_sow_aligned_manifests.py",
                    extra={"direction": direction, "pending_audio": True, "condition": "noisy_pending", "sow_aligned": True},
                )
            ],
            out_dir / f"translation_{direction}_noisy_snr20.jsonl",
        )

    # ---- Caption (Entities only) ----
    entities_root = stage_cfg.get("flickr30k_entities_root") or (cfg.get("sow", {}) or {}).get("flickr30k_entities_root")
    caption_target = summary["targets"]["caption"]
    caption_samples: list[dict[str, Any]] = []
    if entities_root:
        caption_samples = prepare_flickr30k_entities_samples(root=entities_root, limit=caption_target)
        caption_samples = [
            _apply_meta(
                s,
                sow_aligned=True,
                proxy_baseline=False,
                pending_audio=False,
                stub_reason=None,
                build_source="flickr30k_entities",
                condition="clean",
            )
            for s in caption_samples
        ]

    summary["actual"]["caption_clean"] = len(caption_samples)
    if caption_samples:
        write_manifest(caption_samples, out_dir / "caption_clean.jsonl")
    else:
        summary["blocking_reason"]["caption"] = "flickr30k_entities_raw_files_unavailable"
        write_manifest(
            [
                _stub(
                    task="caption",
                    reason="flickr30k_entities_raw_files_unavailable",
                    build_source="build_sow_aligned_manifests.py",
                    extra={"condition": "clean", "sow_ready": False},
                )
            ],
            out_dir / "caption_clean.jsonl",
        )

    if stage == "stage3":
        review_n = int(stage_cfg.get("caption_human_review_size", sizes["caption_review"]))
        if caption_samples:
            review = [
                _apply_meta(dict(s), condition="human_review_seed", build_source="flickr30k_entities")
                for s in caption_samples[:review_n]
            ]
            write_manifest(review, out_dir / "caption_human_review_seed.jsonl")
            summary["actual"]["caption_human_review_seed"] = len(review)
        else:
            summary["actual"]["caption_human_review_seed"] = 0
            write_manifest(
                [
                    _stub(
                        task="caption",
                        reason="caption_clean_is_stub_so_human_review_seed_is_stub",
                        build_source="build_sow_aligned_manifests.py",
                        extra={"condition": "human_review_seed", "sow_ready": False},
                    )
                ],
                out_dir / "caption_human_review_seed.jsonl",
            )

    # ---- Intent (proxy baseline only) ----
    ext_intent = (cfg.get("external_manifests", {}) or {}).get("intent_clean")
    if ext_intent and Path(ext_intent).exists():
        intent = load_custom_manifest(ext_intent, default_fields={"task": "intent", "dataset": "intent_proxy", "split": "proxy"})
        intent = [
            _apply_meta(
                s,
                sow_aligned=False,
                proxy_baseline=True,
                pending_audio=False,
                pending_self_test_set=True,
                stub_reason=None,
                build_source="intent_proxy",
                condition="proxy_clean",
            )
            for s in intent
        ]
        write_manifest(intent, out_dir / "intent_proxy_clean.jsonl")
        summary["actual"]["intent_proxy"] = len(intent)
    else:
        write_manifest(
            [
                _stub(
                    task="intent",
                    reason="self_test_intent_set_unavailable_proxy_manifest_missing",
                    build_source="build_sow_aligned_manifests.py",
                    extra={
                        "proxy_baseline": True,
                        "sow_aligned": False,
                        "pending_self_test_set": True,
                        "condition": "proxy_clean",
                    },
                )
            ],
            out_dir / "intent_proxy_clean.jsonl",
        )
        summary["actual"]["intent_proxy"] = 0

    (out_dir / "sow_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="project/configs/datasets.yaml")
    parser.add_argument("--stage", choices=["stage2", "stage3", "both"], default="both")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(cfg.get("output_root", "data_manifests"))
    stages = ["stage2", "stage3"] if args.stage == "both" else [args.stage]
    for s in stages:
        _build_stage(output_root / f"sow_{s}", cfg, s)
        print(f"[ok] built sow manifests: sow_{s}")


if __name__ == "__main__":
    main()
