from .schema import normalize_sample, to_manifest_record

__all__ = [
    "normalize_sample",
    "to_manifest_record",
    "get_dataset_loader",
    "load_samples",
    "build_eval_manifest",
]


def get_dataset_loader(name: str):
    from .registry import get_dataset_loader as _impl

    return _impl(name)


def load_samples(dataset_name: str, config):
    from .registry import load_samples as _impl

    return _impl(dataset_name, config)


def build_eval_manifest(task_name: str, config):
    from .registry import build_eval_manifest as _impl

    return _impl(task_name, config)
