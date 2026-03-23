from __future__ import annotations

from project.datasets.wmt19_loader import build_tts_manifest_from_wmt19


def main():
    build_tts_manifest_from_wmt19(
        direction="zh_to_en",
        output_path="project/results/manifests/wmt19_zh_to_en_tts.jsonl",
        split="validation",
    )


if __name__ == "__main__":
    main()
