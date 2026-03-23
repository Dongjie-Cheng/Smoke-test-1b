from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from project.datasets.wmt19_loader import build_tts_manifest_from_wmt19


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="project/configs/datasets.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output_dir or (Path(cfg.get("output_root", "data_manifests")) / "real_small"))
    root.mkdir(parents=True, exist_ok=True)
    wmt_cfg = cfg.get("wmt19", {})
    tts_cfg = cfg.get("tts", {})

    for direction in ("zh_to_en", "en_to_zh"):
        out = root / f"translation_{direction}_tts_manifest.jsonl"
        build_tts_manifest_from_wmt19(
            direction=direction,
            output_path=str(out),
            config_name=wmt_cfg.get("config_name", "zh-en"),
            split=wmt_cfg.get("split", "validation"),
            limit=args.limit,
            trust_remote_code=wmt_cfg.get("trust_remote_code"),
            tts_voice=tts_cfg.get("voice"),
        )
        print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
