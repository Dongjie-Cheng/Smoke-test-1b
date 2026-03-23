from __future__ import annotations

from pathlib import Path
from typing import Any


def _pick_model(language: str | None, model_name: str | None) -> str:
    if model_name:
        return model_name
    # multilingual first, then language fallback
    if language in {"zh", "zh-cn", "zh_cn"}:
        return "tts_models/zh-CN/baker/tacotron2-DDC-GST"
    if language in {"en", "en-us", "en_us"}:
        return "tts_models/en/ljspeech/tacotron2-DDC"
    return "tts_models/multilingual/multi-dataset/xtts_v2"


def synthesize_to_file(
    text: str,
    output_path: str,
    language: str,
    speaker_wav: str | None = None,
    speaker: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    from TTS.api import TTS  # type: ignore

    chosen_model = _pick_model(language, model_name)
    tts = TTS(model_name=chosen_model)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {"text": text, "file_path": str(out)}
    # XTTS-like multilingual models support language/speaker params
    if "multilingual" in chosen_model or "xtts" in chosen_model:
        kwargs["language"] = language
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif speaker:
            kwargs["speaker"] = speaker

    tts.tts_to_file(**kwargs)
    sample_rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None)
    return {
        "output_path": str(out.resolve()),
        "sample_rate": sample_rate,
        "tts_backend": "coqui",
        "tts_model": chosen_model,
    }
