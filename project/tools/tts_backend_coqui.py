from __future__ import annotations

from pathlib import Path
from typing import Any

_ENGINE_CACHE: dict[tuple[str, str], Any] = {}


def _pick_model(language: str | None, model_name: str | None) -> str:
    if model_name:
        return model_name
    # multilingual first, then language fallback
    if language in {"zh", "zh-cn", "zh_cn"}:
        return "tts_models/zh-CN/baker/tacotron2-DDC-GST"
    if language in {"en", "en-us", "en_us"}:
        return "tts_models/en/ljspeech/tacotron2-DDC"
    return "tts_models/multilingual/multi-dataset/xtts_v2"


def get_tts_engine(language: str, model_name: str | None = None):
    from TTS.api import TTS  # type: ignore

    chosen_model = _pick_model(language, model_name)
    cache_key = (chosen_model, language or "")
    if cache_key not in _ENGINE_CACHE:
        _ENGINE_CACHE[cache_key] = TTS(model_name=chosen_model)
    return _ENGINE_CACHE[cache_key], chosen_model


def synthesize_to_file(
    text: str,
    output_path: str,
    language: str,
    speaker_wav: str | None = None,
    speaker: str | None = None,
    model_name: str | None = None,
    engine: Any | None = None,
) -> dict[str, Any]:
    if engine is None:
        tts, chosen_model = get_tts_engine(language=language, model_name=model_name)
    else:
        tts = engine
        chosen_model = _pick_model(language, model_name)

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
