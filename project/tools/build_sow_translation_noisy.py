from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.datasets.esc50_loader import build_noise_bank, build_noisy_audio_file
from project.datasets.schema import normalize_sample, to_manifest_record


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _done_ids(path: Path) -> set[str]:
    done = set()
    for r in _read_jsonl(path):
        m = r.get("metadata") or {}
        if r.get("sample_id") and r.get("audio_path") and not m.get("stub_reason"):
            done.add(str(r["sample_id"]))
    return done


def _pending(direction: str, reason: str):
    return to_manifest_record(
        normalize_sample(
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--esc50-root", default=None)
    parser.add_argument("--snr-db", type=float, default=20.0)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--commit-every", type=int, default=100)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    if not args.esc50_root:
        for d in ("zh_to_en", "en_to_zh"):
            _append_jsonl(stage_dir / f"translation_{d}_noisy_snr20.jsonl", [_pending(d, "esc50_root_not_provided")])
        print("[pending] esc50_root not provided")
        return

    try:
        noise_bank = build_noise_bank(args.esc50_root)
    except Exception as exc:
        for d in ("zh_to_en", "en_to_zh"):
            _append_jsonl(stage_dir / f"translation_{d}_noisy_snr20.jsonl", [_pending(d, f"esc50_unavailable:{exc}")])
        print("[pending] esc50 unavailable")
        return

    for direction in ("zh_to_en", "en_to_zh"):
        out_manifest = stage_dir / f"translation_{direction}_noisy_snr20.jsonl"
        if out_manifest.exists() and (args.overwrite or not args.resume):
            out_manifest.unlink()

        done = _done_ids(out_manifest) if args.resume and not args.overwrite else set()
        clean_manifest = stage_dir / f"translation_{direction}_audio_clean.jsonl"
        rows = load_custom_manifest(str(clean_manifest), default_fields={"task": "translation"})

        pending: list[dict] = []
        success = skipped = 0
        for idx, row in enumerate(rows):
            sid = str(row.get("sample_id") or f"{direction}-{idx}")
            if sid in done and not args.overwrite:
                skipped += 1
                continue
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
            pending.append(to_manifest_record(normalize_sample({**row, "audio_path": info["output_path"], "metadata": meta})))
            success += 1

            if len(pending) >= max(1, args.commit_every):
                _append_jsonl(out_manifest, pending)
                pending = []

        _append_jsonl(out_manifest, pending)

        if success == 0 and not out_manifest.exists():
            _append_jsonl(out_manifest, [_pending(direction, "no_audio_clean_samples")])
        print(f"[noisy] direction={direction} success={success} skipped={skipped}")


if __name__ == "__main__":
    main()
