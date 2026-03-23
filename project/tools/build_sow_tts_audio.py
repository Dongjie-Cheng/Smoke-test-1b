from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project.datasets.schema import normalize_sample, to_manifest_record
from project.tools.tts_backend_coqui import synthesize_to_file, get_tts_engine


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_done_ids(out_manifest: Path) -> set[str]:
    done = set()
    for r in _read_jsonl(out_manifest):
        meta = r.get("metadata") or {}
        if r.get("sample_id") and r.get("audio_path") and not meta.get("stub_reason"):
            done.add(str(r["sample_id"]))
    return done


def _build_one_direction(
    stage_dir: Path,
    direction: str,
    model_name: str | None,
    speaker_wav: str | None,
    speaker: str | None,
    resume: bool,
    overwrite: bool,
    max_samples: int | None,
    commit_every: int,
) -> tuple[int, int, int]:
    tts_manifest = stage_dir / f"translation_{direction}_tts_manifest.jsonl"
    rows = _read_jsonl(tts_manifest)
    if max_samples:
        rows = rows[:max_samples]

    audio_out_dir = stage_dir / "tts_audio"
    out_manifest = stage_dir / f"translation_{direction}_audio_clean.jsonl"
    err_path = stage_dir / f"translation_{direction}_tts_errors.jsonl"

    if out_manifest.exists() and (overwrite or not resume):
        out_manifest.unlink()
    if err_path.exists() and (overwrite or not resume):
        err_path.unlink()

    done_ids = _load_done_ids(out_manifest) if resume and not overwrite else set()

    pending_records: list[dict[str, Any]] = []
    pending_errors: list[dict[str, Any]] = []
    success = failed = skipped = 0

    engine = None
    for i, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"{direction}-{i}")
        if sample_id in done_ids and not overwrite:
            skipped += 1
            continue

        source_text = row.get("source_text")
        if not source_text:
            failed += 1
            pending_errors.append({"sample_id": sample_id, "error": "missing_source_text", "direction": direction})
            continue

        out_wav = audio_out_dir / direction / f"{sample_id}.wav"
        if resume and out_wav.exists() and not overwrite:
            # if wav exists but manifest row missing, still append record without recomputing
            result = {"output_path": str(out_wav.resolve()), "sample_rate": None, "tts_backend": "coqui", "tts_model": model_name}
        else:
            try:
                if engine is None:
                    engine, picked_model = get_tts_engine(language=row.get("source_language") or "en", model_name=model_name)
                    model_name = picked_model
                result = synthesize_to_file(
                    text=source_text,
                    output_path=str(out_wav),
                    language=row.get("source_language") or "en",
                    speaker_wav=speaker_wav,
                    speaker=speaker,
                    model_name=model_name,
                    engine=engine,
                )
            except Exception as exc:
                failed += 1
                pending_errors.append({"sample_id": sample_id, "error": str(exc), "direction": direction})
                if len(pending_errors) >= commit_every:
                    _append_jsonl(err_path, pending_errors)
                    pending_errors = []
                continue

        success += 1
        rec = normalize_sample(
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
        pending_records.append(to_manifest_record(rec))

        if len(pending_records) >= commit_every:
            _append_jsonl(out_manifest, pending_records)
            pending_records = []

    _append_jsonl(out_manifest, pending_records)
    _append_jsonl(err_path, pending_errors)

    if success == 0 and not out_manifest.exists():
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
        _append_jsonl(out_manifest, [to_manifest_record(pending)])

    return success, failed, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--speaker-wav", default=None)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--commit-every", type=int, default=50)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    total_ok = total_err = total_skip = 0
    for direction in ("zh_to_en", "en_to_zh"):
        ok, err, skipped = _build_one_direction(
            stage_dir=stage_dir,
            direction=direction,
            model_name=args.tts_model,
            speaker_wav=args.speaker_wav,
            speaker=args.speaker,
            resume=args.resume,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
            commit_every=max(1, args.commit_every),
        )
        total_ok += ok
        total_err += err
        total_skip += skipped
        print(f"[tts] direction={direction} success={ok} failed={err} skipped={skipped}")
    print(f"[tts] total_success={total_ok} total_failed={total_err} total_skipped={total_skip}")


if __name__ == "__main__":
    main()
