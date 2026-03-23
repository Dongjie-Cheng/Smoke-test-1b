"""Model adapter package.

Keep this module dependency-light so `python -m project.model_adapters.model_probe`
can run even when optional model runtimes are not installed.
"""

__all__ = ["AndesVLAdapter", "Qwen3ASRAdapter"]


def __getattr__(name):
    if name == "AndesVLAdapter":
        from .andesvl_adapter import AndesVLAdapter

        return AndesVLAdapter
    if name == "Qwen3ASRAdapter":
        from .qwen3_asr_adapter import Qwen3ASRAdapter

        return Qwen3ASRAdapter
    raise AttributeError(name)
