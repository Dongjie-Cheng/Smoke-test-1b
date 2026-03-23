from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.esc50_loader import build_noise_bank, build_noisy_audio_file
from project.datasets.schema import normalize_sample
from project.datasets.utils import write_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_stub(path: Path, reason: str) -> None:
    row = normalize_sample(
        {
            "sample_id": f"stub-{path.stem}",
            "dataset": "stub",
            "split": "stub",
            "task": "stub",
            "metadata": {"build_source": "stub", "reason": reason},
        }
    )
    write_manifest([row], path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--esc50-root", required=True)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-audio-dir", required=True)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    clean_path = Path(args.clean_manifest)
    samples = load_custom_manifest(str(clean_path)) if clean_path.suffix in {".jsonl", ".csv"} else _read_jsonl(clean_path)

    try:
        noise_bank = build_noise_bank(args.esc50_root)
    except Exception as exc:
        _write_stub(Path(args.output_manifest), f"ESC-50 unavailable: root={args.esc50_root} error={exc}")
        print(f"[stub] ESC-50 unavailable: {exc}")
        return

    noisy_samples = []
    out_audio_dir = Path(args.output_audio_dir)
    for idx, s in enumerate(samples):
        clean_audio_path = s.get("audio_path")
        if not clean_audio_path:
            continue
        out_wav = out_audio_dir / f"{Path(args.output_manifest).stem}_{idx}.wav"
        noise_info = build_noisy_audio_file(
            clean_audio_path=clean_audio_path,
            noise_bank=noise_bank,
            output_path=str(out_wav),
            snr_db=args.snr_db,
            category=args.category,
        )
        meta = {
            **(s.get("metadata") or {}),
            "build_source": "build_noisy_manifests.py",
            "condition": f"noisy_snr{int(args.snr_db)}",
            "snr_db": noise_info["snr_db"],
            "noise_category": noise_info["noise_category"],
            "noise_filename": noise_info["noise_filename"],
        }
        noisy_samples.append(
            normalize_sample(
                {
                    **s,
                    "audio_path": noise_info["output_path"],
                    "metadata": meta,
                }
            )
        )

    if not noisy_samples:
        _write_stub(Path(args.output_manifest), "No audio_path found in clean manifest")
        print("[stub] no samples with audio_path")
        return

    write_manifest(noisy_samples, args.output_manifest)
    print(f"[ok] wrote {args.output_manifest}, rows={len(noisy_samples)}")


if __name__ == "__main__":
    main()
