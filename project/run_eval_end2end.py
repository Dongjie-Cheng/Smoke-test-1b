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


def _resolve(base: Path, p: str | None) -> str | None:
    if not p:
        return None
    path = Path(p)
    return str(path if path.is_absolute() else (base / path).resolve())


def _mkdir(p: Path) -> Path:
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


def _append_jsonl(src: Path, dst: Path) -> int:
    rows = _read_jsonl(src)
    if not rows:
        return 0
    _mkdir(dst.parent)
    mode = "a" if dst.exists() else "w"
    with dst.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _load_manifest(path: str, task: str) -> list[dict[str, Any]]:
    samples = load_custom_manifest(path, default_fields={"task": task})
    # 丢掉 stub/fallback 记录，避免被误当成真实样本
    real_samples = []
    for s in samples:
        meta = s.get("metadata") or {}
        if meta.get("stub_reason") or meta.get("is_stub") is True:
            continue
        real_samples.append(s)
    return real_samples


def _need_asr(samples: list[dict[str, Any]]) -> bool:
    return any(bool(s.get("audio_path")) for s in samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    repo_root = spec_path.parent
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    run_name = spec["run_name"]
    out_root = Path(spec["out_root"]).resolve()
    run_root = out_root / run_name
    raw_root = _mkdir(run_root / "raw_outputs")
    metrics_root = _mkdir(run_root / "metrics")
    logs_root = _mkdir(run_root / "logs")

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

    # A. ASR
    if "asr" in tasks:
        merged = raw_root / "asr_outputs.jsonl"
        if merged.exists():
            merged.unlink()
        total = 0

        for condition, manifest in tasks["asr"].items():
            samples = _load_manifest(_resolve(repo_root, manifest), task="asr")
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

    # B. Intent
    if "intent" in tasks:
        merged = raw_root / "intent_outputs.jsonl"
        if merged.exists():
            merged.unlink()
        total = 0

        for condition, manifest in tasks["intent"].items():
            samples = _load_manifest(_resolve(repo_root, manifest), task="intent")
            if not samples:
                continue
            tmp = raw_root / f"intent_{condition}.jsonl"
            run_intent_cascade(
                samples=samples,
                andes=get_andes(),
                asr=get_asr() if _need_asr(samples) else None,
                prompt_path=_resolve(repo_root, prompts["intent"]),
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

    # C1. Translation zh->en
    if "translation_zh_to_en" in tasks:
        merged = raw_root / "translation_zh_to_en_outputs.jsonl"
        if merged.exists():
            merged.unlink()
        total = 0

        for condition, manifest in tasks["translation_zh_to_en"].items():
            samples = _load_manifest(_resolve(repo_root, manifest), task="translation")
            if not samples:
                continue
            tmp = raw_root / f"translation_zh_to_en_{condition}.jsonl"
            run_translation_cascade(
                samples=samples,
                andes=get_andes(),
                prompt_path=_resolve(repo_root, prompts["zh_to_en"]),
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

    # C2. Translation en->zh
    if "translation_en_to_zh" in tasks:
        merged = raw_root / "translation_en_to_zh_outputs.jsonl"
        if merged.exists():
            merged.unlink()
        total = 0

        for condition, manifest in tasks["translation_en_to_zh"].items():
            samples = _load_manifest(_resolve(repo_root, manifest), task="translation")
            if not samples:
                continue
            tmp = raw_root / f"translation_en_to_zh_{condition}.jsonl"
            run_translation_cascade(
                samples=samples,
                andes=get_andes(),
                prompt_path=_resolve(repo_root, prompts["en_to_zh"]),
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

    # D. Caption
    if "caption" in tasks:
        merged = raw_root / "caption_outputs.jsonl"
        if merged.exists():
            merged.unlink()
        total = 0

        for condition, manifest in tasks["caption"].items():
            samples = _load_manifest(_resolve(repo_root, manifest), task="caption")
            if not samples:
                continue
            tmp = raw_root / f"caption_{condition}.jsonl"
            run_caption_cascade(
                samples=samples,
                andes=get_andes(),
                prompt_path=_resolve(repo_root, prompts["caption"]),
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

    (logs_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()