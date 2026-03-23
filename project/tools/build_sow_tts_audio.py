from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project.datasets.schema import normalize_sample
from project.datasets.utils import write_manifest, write_jsonl
from project.tools.tts_backend_coqui import synthesize_to_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_one_direction(stage_dir: Path, direction: str, model_name: str | None, speaker_wav: str | None, speaker: str | None) -> tuple[int, int]:
    tts_manifest = stage_dir / f"translation_{direction}_tts_manifest.jsonl"
    rows = _read_jsonl(tts_manifest)
    audio_out_dir = stage_dir / "tts_audio"
    error_log: list[dict[str, Any]] = []
    success_samples: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        sample_id = row.get("sample_id") or f"{direction}-{i}"
        source_text = row.get("source_text")
        if not source_text:
            error_log.append({"sample_id": sample_id, "error": "missing_source_text"})
            continue
        out_wav = audio_out_dir / direction / f"{sample_id}.wav"
        try:
            result = synthesize_to_file(
                text=source_text,
                output_path=str(out_wav),
                language=row.get("source_language") or "en",
                speaker_wav=speaker_wav,
                speaker=speaker,
                model_name=model_name,
            )
            success_samples.append(
                normalize_sample(
                    {
                        "sample_id": sample_id,
                        "dataset": "wmt19_tts_audio",
                        "split": "sow",
                        "task": "translation",
                        "text": source_text,
                        "reference_text": row.get("reference_text"),
                        "audio_path": result["output_path"],
                        "source_language": row.get("source_language"),
                        "target_language": row.get("target_language"),
                        "metadata": {
                            **(row.get("metadata") or {}),
                            "source_text": source_text,
                            "build_source": "wmt19_tts",
                            "condition": "clean",
                            "direction": direction,
                            "tts_backend": result.get("tts_backend"),
                            "tts_model": result.get("tts_model"),
                            "sample_rate": result.get("sample_rate"),
                            "sow_aligned": True,
                            "pending_audio": False,
                            "stub_reason": None,
                        },
                    }
                )
            )
        except Exception as exc:
            error_log.append({"sample_id": sample_id, "error": str(exc), "direction": direction})

    out_manifest = stage_dir / f"translation_{direction}_audio_clean.jsonl"
    if success_samples:
        write_manifest(success_samples, out_manifest)
    else:
        pending = normalize_sample(
            {
                "sample_id": f"stub-{direction}-audio-clean",
                "dataset": "stub",
                "split": "sow",
                "task": "translation",
                "metadata": {
                    "build_source": "wmt19_tts",
                    "condition": "clean",
                    "direction": direction,
                    "tts_backend": "coqui",
                    "tts_model": model_name,
                    "sow_aligned": True,
                    "pending_audio": True,
                    "stub_reason": "no_tts_audio_generated",
                },
            }
        )
        write_manifest([pending], out_manifest)

    err_path = stage_dir / f"translation_{direction}_tts_errors.jsonl"
    write_jsonl(error_log, err_path)
    return len(success_samples), len(error_log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--speaker-wav", default=None)
    parser.add_argument("--speaker", default=None)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    total_ok = 0
    total_err = 0
    for direction in ("zh_to_en", "en_to_zh"):
        ok, err = _build_one_direction(stage_dir, direction, args.tts_model, args.speaker_wav, args.speaker)
        total_ok += ok
        total_err += err
        print(f"[tts] direction={direction} success={ok} failed={err}")
    print(f"[tts] total_success={total_ok} total_failed={total_err}")


if __name__ == "__main__":
    main()
