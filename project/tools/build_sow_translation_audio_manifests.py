from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.esc50_loader import build_noise_bank, build_noisy_audio_file
from project.datasets.schema import normalize_sample
from project.datasets.utils import write_manifest, write_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pending_row(direction: str, reason: str, condition: str) -> dict[str, Any]:
    return normalize_sample(
        {
            "sample_id": f"stub-{direction}-{condition}",
            "dataset": "stub",
            "split": "stub",
            "task": "translation",
            "metadata": {
                "sow_aligned": True,
                "proxy_baseline": False,
                "pending_audio": True,
                "stub_reason": reason,
                "build_source": "build_sow_translation_audio_manifests.py",
                "condition": condition,
                "direction": direction,
            },
        }
    )


def _resolve_audio_path(row: dict[str, Any], tts_audio_root: Path | None) -> str | None:
    p = row.get("tts_output_audio_path")
    if p and Path(p).exists():
        return str(Path(p).resolve())
    if tts_audio_root:
        guess = tts_audio_root / f"{row.get('sample_id')}.wav"
        if guess.exists():
            return str(guess.resolve())
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True, help="e.g. data_manifests/sow_stage2")
    parser.add_argument("--tts-audio-root", default=None, help="directory containing <sample_id>.wav")
    parser.add_argument("--esc50-root", default=None)
    parser.add_argument("--snr-db", type=float, default=20.0)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    tts_audio_root = Path(args.tts_audio_root).resolve() if args.tts_audio_root else None

    for direction in ("zh_to_en", "en_to_zh"):
        tts_path = stage_dir / f"translation_{direction}_tts_manifest.jsonl"
        rows = _read_jsonl(tts_path)
        audio_clean: list[dict[str, Any]] = []
        for row in rows:
            audio_path = _resolve_audio_path(row, tts_audio_root)
            if not audio_path:
                continue
            audio_clean.append(
                normalize_sample(
                    {
                        "sample_id": row.get("sample_id"),
                        "dataset": "wmt19_tts_audio",
                        "split": "sow",
                        "task": "translation",
                        "text": row.get("source_text"),
                        "reference_text": row.get("reference_text"),
                        "audio_path": audio_path,
                        "source_language": row.get("source_language"),
                        "target_language": row.get("target_language"),
                        "metadata": {
                            **(row.get("metadata") or {}),
                            "sow_aligned": True,
                            "proxy_baseline": False,
                            "pending_audio": False,
                            "stub_reason": None,
                            "build_source": "build_sow_translation_audio_manifests.py",
                            "condition": "audio_clean",
                            "direction": direction,
                        },
                    }
                )
            )

        clean_path = stage_dir / f"translation_{direction}_audio_clean.jsonl"
        if not audio_clean:
            write_manifest([_pending_row(direction, "real_tts_audio_not_found", "audio_clean_pending")], clean_path)
            write_manifest([_pending_row(direction, "audio_clean_not_ready", "noisy_pending")], stage_dir / f"translation_{direction}_noisy_snr20.jsonl")
            print(f"[pending] {direction} audio clean/noisy not built")
            continue

        write_manifest(audio_clean, clean_path)

        if not args.esc50_root:
            write_manifest([_pending_row(direction, "esc50_root_not_provided", "noisy_pending")], stage_dir / f"translation_{direction}_noisy_snr20.jsonl")
            print(f"[pending] {direction} noisy skipped (esc50 root missing)")
            continue

        try:
            noise_bank = build_noise_bank(args.esc50_root)
        except Exception as exc:
            write_manifest([_pending_row(direction, f"esc50_unavailable:{exc}", "noisy_pending")], stage_dir / f"translation_{direction}_noisy_snr20.jsonl")
            print(f"[pending] {direction} noisy skipped (esc50 unavailable)")
            continue

        noisy_samples = []
        noisy_audio_dir = stage_dir / "translation_noisy_audio"
        for idx, s in enumerate(audio_clean):
            out_wav = noisy_audio_dir / f"{direction}_{idx}.wav"
            info = build_noisy_audio_file(
                clean_audio_path=s["audio_path"],
                noise_bank=noise_bank,
                output_path=str(out_wav),
                snr_db=args.snr_db,
            )
            meta = dict(s.get("metadata") or {})
            meta.update(
                {
                    "condition": f"noisy_snr{int(args.snr_db)}",
                    "pending_audio": False,
                    "noise_category": info.get("noise_category"),
                    "noise_filename": info.get("noise_filename"),
                    "snr_db": info.get("snr_db"),
                }
            )
            noisy_samples.append(normalize_sample({**s, "audio_path": info["output_path"], "metadata": meta}))

        write_manifest(noisy_samples, stage_dir / f"translation_{direction}_noisy_snr20.jsonl")
        print(f"[ok] {direction} audio_clean={len(audio_clean)} noisy={len(noisy_samples)}")


if __name__ == "__main__":
    main()
