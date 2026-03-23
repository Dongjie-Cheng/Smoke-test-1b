from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from .schema import normalize_sample
from .utils import log_dataset_failure

logger = logging.getLogger(__name__)


def parse_entities_sentence_file(path: str):
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_entities_xml(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    root = ET.parse(p).getroot()
    boxes = []
    for obj in root.findall("object"):
        boxes.append({"name": obj.findtext("name"), "bbox": [int(obj.findtext(f"bndbox/{k}", "0")) for k in ["xmin", "ymin", "xmax", "ymax"]]})
    return {"boxes": boxes}


def load_flickr30k_entities(root: str):
    root_path = Path(root)
    sent_dir = root_path / "Sentences"
    ann_dir = root_path / "Annotations"
    if not sent_dir.exists() or not ann_dir.exists():
        logger.warning("Flickr30k Entities raw files unavailable; returning empty stub")
        return []
    pairs = []
    for sent_file in sent_dir.glob("*.txt"):
        xml_file = ann_dir / f"{sent_file.stem}.xml"
        pairs.append((sent_file, xml_file))
    return pairs


def prepare_flickr30k_entities_samples(root: str, limit: int | None = None):
    pairs = load_flickr30k_entities(root)
    if not pairs:
        log_dataset_failure("flickr30k_entities", "missing Sentences/ or Annotations/; loader returns empty stub")
        return []
    out = []
    for idx, (sent_file, xml_file) in enumerate(pairs):
        if limit and idx >= limit:
            break
        out.append(normalize_sample({
            "sample_id": sent_file.stem,
            "dataset": "flickr30k_entities",
            "split": "custom",
            "task": "caption",
            "captions": parse_entities_sentence_file(str(sent_file)),
            "metadata": parse_entities_xml(str(xml_file)),
        }))
    return out
