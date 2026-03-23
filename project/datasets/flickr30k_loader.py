from __future__ import annotations

from typing import List, Dict, Any

from .schema import normalize_sample
from .utils import log_dataset_failure


def load_flickr30k(split: str = "test", trust_remote_code: bool | None = None):
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {}
    if trust_remote_code is not None:
        kwargs["trust_remote_code"] = trust_remote_code
    return load_dataset("nlphuji/flickr30k", split=split, **kwargs)


def prepare_flickr30k_caption_samples(
    split: str = "test",
    reference_mode: str = "single_reference",
    limit: int | None = None,
    trust_remote_code: bool | None = None,
) -> List[Dict[str, Any]]:
    try:
        ds = load_flickr30k(split=split, trust_remote_code=trust_remote_code)
    except Exception as exc:
        log_dataset_failure("flickr30k", f"split={split} trust_remote_code={trust_remote_code} error={exc}")
        return []
    out = []
    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break
        captions = row.get("caption") or row.get("captions") or row.get("sentences") or []
        if isinstance(captions, str):
            captions = [captions]
        reference_text = captions[0] if captions else None
        if reference_mode == "multi_reference":
            reference_text = None
        image = row.get("image")
        image_path = None
        if hasattr(image, "filename"):
            image_path = image.filename
        out.append(
            normalize_sample(
                {
                    "sample_id": str(row.get("img_id", idx)),
                    "dataset": "flickr30k",
                    "split": split,
                    "task": "caption",
                    "image": image,
                    "image_path": image_path,
                    "captions": list(captions),
                    "reference_text": reference_text,
                    "metadata": {
                        "build_source": "nlphuji/flickr30k",
                        "split_source": split,
                        "limit_applied": limit,
                        "trust_remote_code": trust_remote_code,
                        "reference_mode": reference_mode,
                        "captions": list(captions),
                    },
                }
            )
        )
    return out
