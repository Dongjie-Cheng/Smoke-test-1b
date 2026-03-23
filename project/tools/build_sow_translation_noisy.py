from __future__ import annotations

import argparse
from pathlib import Path

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.esc50_loader import build_noise_bank, build_noisy_audio_file
from project.datasets.schema import normalize_sample
from project.datasets.utils import write_manifest


def _pending(direction: str, reason: str):
    return normalize_sample(
        {
            "sample_id": f"stub-{direction}-noisy",
            "dataset": "stub",
            "split": "sow",
            "task": "translation",
            "metadata": {
                "condition": "noisy_snr20",
                "sow_aligned": True,
                "pending_audio": True,
                "stub_reason": reason,
                "direction": direction,
                "build_source": "build_sow_translation_noisy.py",
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--esc50-root", default=None)
    parser.add_argument("--snr-db", type=float, default=20.0)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    if not args.esc50_root:
        for d in ("zh_to_en", "en_to_zh"):
            write_manifest([_pending(d, "esc50_root_not_provided")], stage_dir / f"translation_{d}_noisy_snr20.jsonl")
        print("[pending] esc50_root not provided")
        return

    try:
        noise_bank = build_noise_bank(args.esc50_root)
    except Exception as exc:
        for d in ("zh_to_en", "en_to_zh"):
            write_manifest([_pending(d, f"esc50_unavailable:{exc}")], stage_dir / f"translation_{d}_noisy_snr20.jsonl")
        print("[pending] esc50 unavailable")
        return

    for direction in ("zh_to_en", "en_to_zh"):
        clean_manifest = stage_dir / f"translation_{direction}_audio_clean.jsonl"
        rows = load_custom_manifest(str(clean_manifest), default_fields={"task": "translation"})
        noisy = []
        for idx, row in enumerate(rows):
            if not row.get("audio_path"):
                continue
            out_wav = stage_dir / "translation_noisy_audio" / f"{direction}_{idx}.wav"
            info = build_noisy_audio_file(
                clean_audio_path=row["audio_path"],
                noise_bank=noise_bank,
                output_path=str(out_wav),
                snr_db=args.snr_db,
            )
            meta = dict(row.get("metadata") or {})
            meta.update(
                {
                    "condition": "noisy_snr20",
                    "snr_db": info.get("snr_db"),
                    "noise_category": info.get("noise_category"),
                    "noise_filename": info.get("noise_filename"),
                    "sow_aligned": True,
                    "pending_audio": False,
                    "build_source": "build_sow_translation_noisy.py",
                }
            )
            noisy.append(normalize_sample({**row, "audio_path": info["output_path"], "metadata": meta}))

        out_manifest = stage_dir / f"translation_{direction}_noisy_snr20.jsonl"
        if noisy:
            write_manifest(noisy, out_manifest)
        else:
            write_manifest([_pending(direction, "no_audio_clean_samples")], out_manifest)
        print(f"[noisy] direction={direction} success={len(noisy)}")


if __name__ == "__main__":
    main()
