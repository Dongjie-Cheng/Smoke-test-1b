from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from project.datasets.custom_manifest_loader import load_custom_manifest
from project.model_adapters.andesvl_adapter import AndesVLAdapter
from project.model_adapters.qwen3_asr_adapter import Qwen3ASRAdapter
from project.pipelines.cascade_intent import run_intent_cascade
from project.pipelines.cascade_translation import run_translation_cascade
from project.pipelines.cascade_caption import run_caption_cascade
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
    with dst.open("a" if dst.exists() else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _load_manifest(path: str, task: str) -> list[dict[str, Any]]:
    samples = load_custom_manifest(path, default_fields={"task": task})
    out = []
    for s in samples:
        meta = s.get("metadata") or {}
        if meta.get("stub_reason") and not (s.get("audio_path") or s.get("text") or s.get("image_path")):
            continue
        if meta.get("pending_audio") and not s.get("audio_path"):
            continue
        out.append(s)
    return out


def _need_asr(samples: list[dict[str, Any]]) -> bool:
    return any(bool(s.get("audio_path")) for s in samples)


def _ready(path: str, repo_root: Path, require_audio: bool = False) -> tuple[bool, int]:
    rows = _read_jsonl(Path(_resolve(repo_root, path)))
    if not rows:
        return False, 0
    valid = []
    for r in rows:
        m = r.get("metadata") or {}
        if m.get("stub_reason"):
            continue
        if require_audio and not r.get("audio_path"):
            continue
        valid.append(r)
    return (len(valid) > 0), len(valid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    repo_root = spec_path.parent
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    run_name = spec["run_name"]
    out_root = Path(_resolve(repo_root, spec["out_root"]))
    run_root = out_root / run_name
    raw_root = _mkdir(run_root / "raw_outputs")
    metrics_root = _mkdir(run_root / "metrics")
    logs_root = _mkdir(run_root / "logs")

    prompts = spec["prompts"]
    models = spec["models"]
    tasks = spec["tasks"]
    targets = spec.get("targets", {})

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

    # readiness summary
    zh_text_ready, zh_text_n = _ready(tasks["translation_zh_to_en"]["clean_text"], repo_root)
    en_text_ready, en_text_n = _ready(tasks["translation_en_to_zh"]["clean_text"], repo_root)
    zh_audio_ready, zh_audio_n = _ready(tasks["translation_zh_to_en"]["audio_clean"], repo_root, require_audio=True)
    en_audio_ready, en_audio_n = _ready(tasks["translation_en_to_zh"]["audio_clean"], repo_root, require_audio=True)
    zh_noisy_ready, zh_noisy_n = _ready(tasks["translation_zh_to_en"]["noisy_snr20"], repo_root, require_audio=True)
    en_noisy_ready, en_noisy_n = _ready(tasks["translation_en_to_zh"]["noisy_snr20"], repo_root, require_audio=True)
    caption_ready, caption_n = _ready(tasks["caption"]["clean"], repo_root)

    summary: dict[str, Any] = {
        "run_name": run_name,
        "translation_text_ready": zh_text_ready and en_text_ready,
        "translation_audio_ready": zh_audio_ready and en_audio_ready,
        "translation_noisy_ready": zh_noisy_ready and en_noisy_ready,
        "caption_sow_ready": caption_ready,
        "intent_proxy_only": True,
        "targets": {
            "translation_clean": targets.get("translation_clean"),
            "translation_noisy": targets.get("translation_noisy"),
            "caption": targets.get("caption"),
        },
        "actual_size": {
            "translation_clean": zh_text_n + en_text_n,
            "translation_audio_clean": zh_audio_n + en_audio_n,
            "translation_noisy": zh_noisy_n + en_noisy_n,
            "caption": caption_n,
        },
        "blocking_reason": {},
        "tasks": {},
    }
    if not summary["translation_audio_ready"]:
        summary["blocking_reason"]["translation_audio"] = "real_tts_audio_not_generated_or_unavailable"
    if not summary["translation_noisy_ready"]:
        summary["blocking_reason"]["translation_noisy"] = "audio_clean_or_esc50_unavailable"
    if not summary["caption_sow_ready"]:
        summary["blocking_reason"]["caption"] = "flickr30k_entities_unavailable_or_stub"

    # translation zh->en
    for key, prompt_key in (("translation_zh_to_en", "zh_to_en"), ("translation_en_to_zh", "en_to_zh")):
        merged = raw_root / f"{key}_outputs.jsonl"
        if merged.exists():
            merged.unlink()
        total = 0
        for condition, manifest in tasks[key].items():
            samples = _load_manifest(_resolve(repo_root, manifest), task="translation")
            if not samples:
                continue
            tmp = raw_root / f"{key}_{condition}.jsonl"
            run_translation_cascade(
                samples=samples,
                andes=get_andes(),
                prompt_path=_resolve(repo_root, prompts[prompt_key]),
                output_jsonl=str(tmp),
                condition=condition,
                asr=get_asr() if _need_asr(samples) else None,
            )
            total += _append_jsonl(tmp, merged)
        if merged.exists():
            out_csv = metrics_root / f"{key}_metrics.csv"
            evaluate_translation(str(merged), str(out_csv))
            summary["tasks"][key] = {"raw_jsonl": str(merged), "metrics_csv": str(out_csv), "num_rows": total}

    # caption
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

    # intent proxy
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
            summary["tasks"]["intent"] = {"raw_jsonl": str(merged), "metrics_csv": str(out_csv), "num_rows": total}

    (logs_root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
