from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List


class Qwen3ASRAdapter:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-ASR-0.6B",
        backend: str = "transformers",
        device: str = "auto",
        dtype: str = "auto",
        max_inference_batch_size: int = 4,
        max_new_tokens: int = 256,
    ):
        self.model_name_or_path = model_name_or_path
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self._log_file = Path("project/logs/qwen3_asr_adapter.log")
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            qwen_asr = importlib.import_module("qwen_asr")
        except Exception as exc:
            raise ImportError("qwen_asr is required. Install with: pip install qwen-asr") from exc

        if backend == "vllm":
            try:
                importlib.import_module("vllm")
            except Exception:
                self._log("backend=vllm requested but vllm missing; falling back to transformers")
                self.backend = "transformers"

        self.model = qwen_asr.Qwen3ASRModel.from_pretrained(model_name_or_path)

    def _log(self, msg: str):
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    @staticmethod
    def _parse_result(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return {
                "language": raw.get("language"),
                "text": (raw.get("text") or "").strip(),
                "timestamps": raw.get("timestamps"),
                "raw": raw,
            }
        return {"language": None, "text": str(raw).strip(), "timestamps": None, "raw": raw}

    def transcribe_file(self, audio_path: str, language: str | None = None) -> dict:
        result = self.model.transcribe(audio=audio_path, language=language)

        # 情况 1：官方对象列表，例如 [ASRTranscription(...)]
        if isinstance(result, list) and len(result) > 0:
            first = result[0]

            lang = getattr(first, "language", None)
            text = getattr(first, "text", "")
            timestamps = getattr(first, "time_stamps", None)

            return {
                "language": lang or None,
                "text": text or "",
                "timestamps": timestamps,
                "raw": result,
            }

        # 情况 2：dict
        if isinstance(result, dict):
            return {
                "language": result.get("language"),
                "text": result.get("text", ""),
                "timestamps": result.get("timestamps") or result.get("time_stamps"),
                "raw": result,
            }

        # 情况 3：单对象
        lang = getattr(result, "language", None)
        text = getattr(result, "text", None)
        timestamps = getattr(result, "time_stamps", None)

        if text is not None:
            return {
                "language": lang,
                "text": text,
                "timestamps": timestamps,
                "raw": result,
            }

        # 兜底
        return {
            "language": None,
            "text": str(result),
            "timestamps": None,
            "raw": result,
        }

    def transcribe_batch(self, audio_paths: List[str], languages: List[str | None] | None = None) -> List[Dict[str, Any]]:
        languages = languages or [None] * len(audio_paths)
        out = []
        for ap, lang in zip(audio_paths, languages):
            out.append(self.transcribe_file(ap, lang))
        return out

    def detect_language(self, audio_path: str) -> str:
        result = self.transcribe_file(audio_path, language=None)
        return result.get("language") or "unknown"
