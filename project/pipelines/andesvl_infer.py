from __future__ import annotations

from typing import Dict, Any

from project.model_adapters.andesvl_adapter import AndesVLAdapter


def infer_one(sample: Dict[str, Any], adapter: AndesVLAdapter):
    if sample.get("text") and sample.get("image_path"):
        return adapter.generate_from_text_image(sample["text"], sample["image_path"])
    if sample.get("image_path"):
        return adapter.generate_from_image(sample["image_path"], sample.get("text") or "Describe this image.")
    return adapter.generate_from_text(sample.get("text") or "")
