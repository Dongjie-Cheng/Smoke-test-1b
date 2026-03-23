from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import yaml

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.model_adapters.andesvl_adapter import AndesVLAdapter
from project.model_adapters.qwen3_asr_adapter import Qwen3ASRAdapter
from project.pipelines.asr_infer import run_asr
from project.pipelines.cascade_intent import run_intent_cascade
from project.pipelines.cascade_translation import run_translation_cascade
from project.pipelines.cascade_caption import run_caption_cascade
from project.eval.eval_asr import evaluate_asr
from project.eval.eval_intent import evaluate_intent
from project.eval.eval_translation import evaluate_translation
from project.eval.eval_caption import evaluate_caption


def _resolve(spec_dir: Path, p: str | None) -> str | None:
    if not p:
        return None
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((spec_dir / path).resolve())


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(src: Path, dst: Path) -> int:
    rows = _read_jsonl(src)
    if not rows:
        return 0
    _ensure_dir(dst.parent)
    mode = "a" if dst.exists() else "w"
    with dst.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _load_manifest(manifest_path: str, task: str) -> list[dict[str, Any]]:
    return load_custom_manifest(
        manifest_path,
        default_fields={"task": task},
    )


def _need_asr(samples: list[dict[str, Any]]) -> bool:
    return any(bool(s.get("audio_path")) for s in samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="YAML spec path")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    spec_dir = spec_path.parent
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    run_name = spec["run_name"]
    out_root = Path(_resolve(spec_dir, spec["out_root"]))
    run_root = out_root / run_name
    raw_root = _ensure_dir(run_root / "raw_outputs")
    metrics_root = _ensure_dir(run_root / "metrics")
    logs_root = _ensure_dir(run_root / "logs")

    prompts = spec["prompts"]
    models = spec["models"]
    tasks = spec["tasks"]

    andes: AndesVLAdapter | None = None
    asr: Qwen3ASRAdapter | None = None

    def get_andes() -> AndesVLAdapter:
        nonlocal andes
        if andes is None:
            andes = AndesVLAdapter(
                model_name_or_path=models["andesvl"],
                device=spec.get("device", "cuda"),
                dtype=spec.get("dtype", "auto"),
                thinking=spec.get("thinking", False),
                max_new_tokens=int(spec.get("andes_max_new_tokens", 256)),
                temperature=float(spec.get("temperature", 0.0)),
                trust_remote_code=True,
            )
        return andes

    def get_asr() -> Qwen3ASRAdapter:
        nonlocal asr
        if asr is None:
            asr = Qwen3ASRAdapter(
                model_name_or_path=models["qwen3_asr"],
                backend=models.get("qwen3_asr_backend", "transformers"),
                device=spec.get("device", "cuda"),
                dtype=spec.get("dtype", "auto"),
                max_inference_batch_size=int(spec.get("asr_batch_size", 1)),
                max_new_tokens=int(spec.get("asr_max_new_tokens", 256)),
            )
        return asr

    summary: dict[str, Any] = {"run_name": run_name, "tasks": {}}

    # ---------- A. ASR ----------
    if "asr" in tasks:
        merged = raw_root / "asr_outputs.jsonl"
        if merged.exists():
            merged.unlink()

        total = 0
        for condition, manifest in tasks["asr"].items():
            manifest = _resolve(spec_dir, manifest)
            samples = _load_manifest(manifest, task="asr")
            if not samples:
                continue
            tmp = raw_root / f"asr_{condition}.jsonl"
            run_asr(samples, get_asr(), condition, str(tmp))
            total += _append_jsonl(tmp, merged)

        if merged.exists():
            out_csv = metrics_root / "asr_metrics.csv"
            evaluate_asr(str(merged), str(out_csv))
            summary["tasks"]["asr"] = {
                "raw_jsonl": str(merged),
                "metrics_csv": str(out_csv),
                "num_rows": total,
            }

    # ---------- B. Intent ----------
    if "intent" in tasks:
        merged = raw_root / "intent_outputs.jsonl"
        if merged.exists():
            merged.unlink()

        total = 0
        for condition, manifest in tasks["intent"].items():
            manifest = _resolve(spec_dir, manifest)
            samples = _load_manifest(manifest, task="intent")
            if not samples:
                continue
            tmp = raw_root / f"intent_{condition}.jsonl"
            run_intent_cascade(
                samples=samples,
                andes=get_andes(),
                asr=get_asr() if _need_asr(samples) else None,
                prompt_path=_resolve(spec_dir, prompts["intent"]),
                condition=condition,
                output_jsonl=str(tmp),
            )
            total += _append_jsonl(tmp, merged)

        if merged.exists():
            out_csv = metrics_root / "intent_metrics.csv"
            evaluate_intent(str(merged), str(out_csv))
            summary["tasks"]["intent"] = {
                "raw_jsonl": str(merged),
                "metrics_csv": str(out_csv),
                "num_rows": total,
            }

    # ---------- C1. Translation zh->en ----------
    if "translation_zh_to_en" in tasks:
        merged = raw_root / "translation_zh_to_en_outputs.jsonl"
        if merged.exists():
            merged.unlink()

        total = 0
        for condition, manifest in tasks["translation_zh_to_en"].items():
            manifest = _resolve(spec_dir, manifest)
            samples = _load_manifest(manifest, task="translation")
            if not samples:
                continue
            tmp = raw_root / f"translation_zh_to_en_{condition}.jsonl"
            run_translation_cascade(
                samples=samples,
                andes=get_andes(),
                prompt_path=_resolve(spec_dir, prompts["zh_to_en"]),
                output_jsonl=str(tmp),
                condition=condition,
                asr=get_asr() if _need_asr(samples) else None,
            )
            total += _append_jsonl(tmp, merged)

        if merged.exists():
            out_csv = metrics_root / "translation_zh_to_en_metrics.csv"
            evaluate_translation(str(merged), str(out_csv))
            summary["tasks"]["translation_zh_to_en"] = {
                "raw_jsonl": str(merged),
                "metrics_csv": str(out_csv),
                "num_rows": total,
            }

    # ---------- C2. Translation en->zh ----------
    if "translation_en_to_zh" in tasks:
        merged = raw_root / "translation_en_to_zh_outputs.jsonl"
        if merged.exists():
            merged.unlink()

        total = 0
        for condition, manifest in tasks["translation_en_to_zh"].items():
            manifest = _resolve(spec_dir, manifest)
            samples = _load_manifest(manifest, task="translation")
            if not samples:
                continue
            tmp = raw_root / f"translation_en_to_zh_{condition}.jsonl"
            run_translation_cascade(
                samples=samples,
                andes=get_andes(),
                prompt_path=_resolve(spec_dir, prompts["en_to_zh"]),
                output_jsonl=str(tmp),
                condition=condition,
                asr=get_asr() if _need_asr(samples) else None,
            )
            total += _append_jsonl(tmp, merged)

        if merged.exists():
            out_csv = metrics_root / "translation_en_to_zh_metrics.csv"
            evaluate_translation(str(merged), str(out_csv))
            summary["tasks"]["translation_en_to_zh"] = {
                "raw_jsonl": str(merged),
                "metrics_csv": str(out_csv),
                "num_rows": total,
            }

    # ---------- D. Caption ----------
    if "caption" in tasks:
        merged = raw_root / "caption_outputs.jsonl"
        if merged.exists():
            merged.unlink()

        total = 0
        for condition, manifest in tasks["caption"].items():
            manifest = _resolve(spec_dir, manifest)
            samples = _load_manifest(manifest, task="caption")
            if not samples:
                continue
            tmp = raw_root / f"caption_{condition}.jsonl"
            run_caption_cascade(
                samples=samples,
                andes=get_andes(),
                prompt_path=_resolve(spec_dir, prompts["caption"]),
                output_jsonl=str(tmp),
                condition=condition,
                asr=get_asr() if _need_asr(samples) else None,
            )
            total += _append_jsonl(tmp, merged)

        if merged.exists():
            out_csv = metrics_root / "caption_metrics.csv"
            review_jsonl = raw_root / "caption_human_review.jsonl"
            evaluate_caption(str(merged), str(out_csv), str(review_jsonl))
            summary["tasks"]["caption"] = {
                "raw_jsonl": str(merged),
                "metrics_csv": str(out_csv),
                "review_jsonl": str(review_jsonl),
                "num_rows": total,
            }

    summary_path = logs_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()