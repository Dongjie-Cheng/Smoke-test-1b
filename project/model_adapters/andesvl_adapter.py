from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List


class AndesVLAdapter:
    def __init__(
        self,
        model_name_or_path: str = "OPPOer/AndesVL-0_6B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        thinking: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        try:
            from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
        except Exception as exc:
            raise ImportError("transformers is required for AndesVLAdapter. Install with: pip install transformers") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        if not hasattr(self.model, "chat"):
            log_path = Path("project/logs/model_probe_andesvl.txt")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "AndesVL model missing .chat interface\n"
                + f"model={model_name_or_path}\n"
                + "dir(model):\n"
                + "\n".join(dir(self.model)),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{model_name_or_path} does not expose .chat; see {log_path} for model probe details"
            )

    def _call_chat(self, messages: List[Dict[str, Any]]):
        kwargs = {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature}
        if self.thinking:
            chat_sig = inspect.signature(self.model.chat)
            for key in ("thinking", "use_thinking", "enable_thinking"):
                if key in chat_sig.parameters:
                    kwargs[key] = True
                    break
        return self.model.chat(messages, self.tokenizer, self.image_processor, **kwargs)

    @staticmethod
    def _parse_text(resp: Any) -> str:
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            for key in ("text", "response", "output", "content"):
                if isinstance(resp.get(key), str):
                    return resp[key].strip()
        return str(resp).strip()

    def _wrap(self, response: Any, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "text": self._parse_text(response),
            "raw_response": response,
            "metadata": metadata or {},
        }

    def generate_from_text(self, text: str, prompt_name: str | None = None) -> Dict[str, Any]:
        msg = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        resp = self._call_chat(msg)
        return self._wrap(resp, {"input_mode": "text", "prompt_name": prompt_name})

    def generate_from_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        msg = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_path}}]}]
        resp = self._call_chat(msg)
        return self._wrap(resp, {"input_mode": "image", "image_path": image_path})

    def generate_from_text_image(self, text: str, image_path: str) -> Dict[str, Any]:
        msg = [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image_path}}]}]
        resp = self._call_chat(msg)
        return self._wrap(resp, {"input_mode": "text+image", "image_path": image_path})

    def batch_generate(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for s in samples:
            if s.get("text") and s.get("image_path"):
                outputs.append(self.generate_from_text_image(s["text"], s["image_path"]))
            elif s.get("image_path"):
                outputs.append(self.generate_from_image(s["image_path"], s.get("prompt", "Describe this image.")))
            else:
                outputs.append(self.generate_from_text(s.get("text", ""), s.get("prompt_name")))
        return outputs
