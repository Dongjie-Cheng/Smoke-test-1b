from __future__ import annotations

import importlib
import platform
from pathlib import Path
from typing import Dict, Any


def _safe_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"


def run_model_probe(andes_model: str = "OPPOer/AndesVL-0_6B-Instruct") -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["python"] = platform.python_version()
    summary["torch"] = _safe_version("torch")
    summary["transformers"] = _safe_version("transformers")
    summary["datasets"] = _safe_version("datasets")

    try:
        importlib.import_module("qwen_asr")
        summary["qwen_asr_importable"] = True
    except Exception:
        summary["qwen_asr_importable"] = False

    try:
        import torch

        summary["cuda_available"] = bool(torch.cuda.is_available())
        summary["num_gpus"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        summary["device_suggestion"] = "cuda" if torch.cuda.is_available() else "cpu"
        summary["dtype_suggestion"] = "float16" if torch.cuda.is_available() else "float32"
    except Exception:
        summary["cuda_available"] = False
        summary["num_gpus"] = 0
        summary["device_suggestion"] = "cpu"
        summary["dtype_suggestion"] = "float32"

    summary["andes_has_chat"] = False
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(andes_model, trust_remote_code=True)
        summary["andes_has_chat"] = hasattr(model, "chat")
    except Exception as exc:
        summary["andes_probe_error"] = str(exc)

    summary["qwen_backend_transformers"] = summary["qwen_asr_importable"]
    try:
        importlib.import_module("vllm")
        summary["qwen_backend_vllm"] = summary["qwen_asr_importable"]
    except Exception:
        summary["qwen_backend_vllm"] = False

    out_path = Path("project/logs/model_probe_summary.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v}" for k, v in summary.items()]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[model_probe] summary")
    for k in ["python", "torch", "transformers", "datasets", "qwen_asr_importable", "cuda_available", "andes_has_chat"]:
        print(f"- {k}: {summary.get(k)}")
    return summary


if __name__ == "__main__":
    run_model_probe()
