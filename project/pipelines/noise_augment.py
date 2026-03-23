from __future__ import annotations

from typing import Dict, Any, List

from project.datasets.esc50_loader import add_noise


def build_noisy_asr_samples(clean_samples: List[Dict[str, Any]], noise_rows: List[Dict[str, Any]], snr_db: float = 10.0):
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    rng = np.random.default_rng(42)
    out = []
    for s in clean_samples:
        if not s.get("audio_path"):
            continue
        audio, sr = sf.read(s["audio_path"])
        row = noise_rows[int(rng.integers(0, len(noise_rows)))]
        noise, nsr = sf.read(row["audio_path"])
        if nsr != sr:
            import librosa

            noise = librosa.resample(noise.astype(float), orig_sr=nsr, target_sr=sr)
        noisy = add_noise(audio.astype(float), noise.astype(float), snr_db=snr_db)
        out.append({
            **s,
            "audio_array": noisy,
            "sampling_rate": sr,
            "metadata": {
                **(s.get("metadata") or {}),
                "snr_db": snr_db,
                "noise_category": row.get("category"),
                "noise_filename": row.get("filename"),
            },
        })
    return out
