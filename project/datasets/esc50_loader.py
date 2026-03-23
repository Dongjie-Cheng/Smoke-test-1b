from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any


def _np():
    import numpy as np  # type: ignore

    return np


class ESC50NoiseBank:
    def __init__(self, root: str, rows: List[Dict[str, Any]]):
        self.root = Path(root)
        self.rows = rows

    def sample_noise_clips(self, category: str | None = None, fold: int | None = None, n: int = 1):
        np = _np()
        candidates = self.rows
        if category is not None:
            candidates = [r for r in candidates if r.get("category") == category]
        if fold is not None:
            candidates = [r for r in candidates if int(r.get("fold", -1)) == fold]
        rng = np.random.default_rng(42)
        if not candidates:
            return []
        idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
        return [candidates[i] for i in np.atleast_1d(idx)]


def load_esc50(root: str) -> ESC50NoiseBank:
    root_path = Path(root)
    meta = root_path / "meta" / "esc50.csv"
    audio_dir = root_path / "audio"
    if not meta.exists() or not audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 missing required files under {root}")
    rows = []
    with meta.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["audio_path"] = str((audio_dir / row["filename"]).resolve())
            rows.append(row)
    return ESC50NoiseBank(root=root, rows=rows)


def build_noise_bank(root: str) -> ESC50NoiseBank:
    return load_esc50(root)


def _fit_noise_length(noise, target_len: int, rng):
    np = _np()
    if len(noise) == target_len:
        return noise
    if len(noise) < target_len:
        reps = int(np.ceil(target_len / len(noise)))
        return np.tile(noise, reps)[:target_len]
    start = int(rng.integers(0, len(noise) - target_len + 1))
    return noise[start : start + target_len]


def add_noise(audio, noise, snr_db: float):
    np = _np()
    rng = np.random.default_rng(42)
    noise = _fit_noise_length(noise, len(audio), rng)
    audio_power = np.mean(audio**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    noisy = audio + noise * scale
    return noisy.astype(audio.dtype, copy=False)


def build_noisy_audio_file(
    clean_audio_path: str,
    noise_bank: ESC50NoiseBank,
    output_path: str,
    snr_db: float,
    category: str | None = None,
):
    import soundfile as sf  # type: ignore

    clean, clean_sr = sf.read(clean_audio_path)
    noise_rows = noise_bank.sample_noise_clips(category=category, n=1)
    if not noise_rows:
        raise ValueError(f"No noise clip available for category={category}")
    noise_row = noise_rows[0]
    noise, noise_sr = sf.read(noise_row["audio_path"])
    if noise_sr != clean_sr:
        import librosa  # type: ignore

        noise = librosa.resample(noise.astype(float), orig_sr=noise_sr, target_sr=clean_sr)
    noisy = add_noise(clean.astype(float), noise.astype(float), snr_db=snr_db)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), noisy, clean_sr)
    return {
        "output_path": str(out.resolve()),
        "snr_db": snr_db,
        "noise_category": noise_row.get("category"),
        "noise_filename": noise_row.get("filename"),
        "noise_audio_path": noise_row.get("audio_path"),
    }
