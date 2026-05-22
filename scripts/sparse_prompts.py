"""Build sparse (entity-list) prompts from COCO instance category names."""

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_JSON = PROJECT_ROOT / "data/coco/raw/instances_val2017.json"

_cache: dict[int, str] | None = None


def load_sparse_prompts(instances_path: Path | None = None) -> dict[int, str]:
    """Map COCO image_id -> comma-separated category names (no spatial wording)."""
    global _cache
    if _cache is not None:
        return _cache

    path = instances_path or INSTANCES_JSON
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    by_image: dict[int, set[str]] = defaultdict(set)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].add(cat_names[ann["category_id"]])

    _cache = {
        image_id: ", ".join(sorted(names))
        for image_id, names in by_image.items()
        if names
    }
    return _cache


def get_sparse_prompt(coco_id: int, dense_caption: str, instances_path: Path | None = None) -> str:
    """Return sparse prompt for an image; fall back to dense caption if no detections."""
    sparse_map = load_sparse_prompts(instances_path)
    prompt = sparse_map.get(int(coco_id), "").strip()
    if prompt:
        return prompt
    return dense_caption
