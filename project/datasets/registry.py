from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

from .custom_manifest_loader import load_custom_manifest
from .esc50_loader import build_noise_bank
from .fleurs_loader import prepare_fleurs_asr_samples, prepare_fleurs_lid_samples
from .flickr30k_entities_loader import prepare_flickr30k_entities_samples
from .flickr30k_loader import prepare_flickr30k_caption_samples
from .utils import write_manifest
from .wmt19_loader import prepare_wmt19_translation_samples


DATASET_LOADERS: Dict[str, Callable[..., Any]] = {
    "fleurs": prepare_fleurs_asr_samples,
    "wmt19": prepare_wmt19_translation_samples,
    "flickr30k": prepare_flickr30k_caption_samples,
    "flickr30k_entities": prepare_flickr30k_entities_samples,
    "esc50": build_noise_bank,
    "custom_manifest": load_custom_manifest,
}


def get_dataset_loader(name: str):
    if name not in DATASET_LOADERS:
        raise KeyError(f"Unknown dataset loader: {name}")
    return DATASET_LOADERS[name]


def load_samples(dataset_name: str, config: Dict[str, Any]):
    config = dict(config)
    loader = get_dataset_loader(dataset_name)
    if dataset_name == "fleurs" and config.get("task") == "lid":
        loader = prepare_fleurs_lid_samples
    config.pop("task", None)
    return loader(**config)


def build_eval_manifest(task_name: str, config: Dict[str, Any]):
    dataset_name = config["dataset_name"]
    samples = load_samples(dataset_name, config.get("loader_args", {}))
    out_path = config.get("output_path", Path("project/results/manifests") / f"{task_name}_{dataset_name}.jsonl")
    return write_manifest(samples, out_path, fmt=config.get("format", "jsonl"))
