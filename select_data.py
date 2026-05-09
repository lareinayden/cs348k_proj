# select_data.py

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import fiftyone.zoo as foz


# -----------------------------
# Config
# -----------------------------
MAX_SAMPLES = 100
TOP_K = 40

COCO_CAPTION_FILE = Path("data/coco/raw/captions_val2017.json")
OUTPUT_DIR = Path("data/selected")
OUTPUT_CSV = OUTPUT_DIR / "selected_coco_candidates.csv"

SPATIAL_WORDS = [
    "on", "under", "above", "below", "next to", "beside",
    "behind", "in front of", "near", "inside", "outside",
    "left", "right", "top", "bottom", "between"
]


# -----------------------------
# Utility functions
# -----------------------------
def load_coco_captions(caption_file):
    """Load COCO captions from captions_val2017.json."""
    with open(caption_file, "r", encoding="utf-8") as f:
        caption_json = json.load(f)

    captions_by_image_id = defaultdict(list)

    for ann in caption_json["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        captions_by_image_id[image_id].append(caption)

    return captions_by_image_id


def get_caption_score(captions):
    """Score captions based on length and spatial relationship words."""
    if not captions:
        return 0.0, ""

    best_caption = max(captions, key=len)
    text = best_caption.lower()

    length_score = min(len(text.split()) / 15.0, 1.0)

    spatial_count = sum(1 for word in SPATIAL_WORDS if word in text)
    spatial_score = min(spatial_count / 3.0, 1.0)

    caption_score = 0.6 * length_score + 0.4 * spatial_score

    return caption_score, best_caption


def get_detection_score(sample):
    """Score image layout complexity using COCO detections."""
    if not sample.has_field("detections"):
        return 0.0, 0, 0, 0.0

    detections = sample["detections"]

    if detections is None or not hasattr(detections, "detections"):
        return 0.0, 0, 0, 0.0

    dets = detections.detections
    num_objects = len(dets)

    if num_objects == 0:
        return 0.0, 0, 0, 0.0

    categories = set(d.label for d in dets)
    num_categories = len(categories)

    centers = []
    areas = []

    for d in dets:
        # FiftyOne bounding boxes are normalized [x, y, width, height]
        x, y, w, h = d.bounding_box
        centers.append([x + w / 2.0, y + h / 2.0])
        areas.append(w * h)

    centers = np.array(centers)
    areas = np.array(areas)

    # Prefer scenes with multiple objects and categories
    object_score = min(num_objects / 8.0, 1.0)
    category_score = min(num_categories / 5.0, 1.0)

    # Prefer objects spread across the image
    if len(centers) > 1:
        spread_score = min(
            (np.std(centers[:, 0]) + np.std(centers[:, 1])) / 0.35,
            1.0
        )
    else:
        spread_score = 0.2

    # Avoid scenes where all detections are tiny
    area_score = min(np.mean(areas) / 0.08, 1.0)

    detection_score = (
        0.35 * object_score
        + 0.25 * category_score
        + 0.25 * spread_score
        + 0.15 * area_score
    )

    return detection_score, num_objects, num_categories, spread_score


def get_edge_score(image_path):
    """Score Canny edge quality using edge density."""
    img = cv2.imread(str(image_path))

    if img is None:
        return 0.0, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Good edge maps are informative but not too noisy.
    # Roughly 2%–18% edge pixels is useful.
    if edge_density < 0.02:
        edge_score = edge_density / 0.02
    elif edge_density > 0.18:
        edge_score = max(0.0, 1.0 - (edge_density - 0.18) / 0.15)
    else:
        edge_score = 1.0

    return edge_score, edge_density


def extract_coco_id_from_path(filepath):
    """
    COCO val filenames look like:
    000000123456.jpg

    The integer stem is the COCO image id.
    """
    return int(Path(filepath).stem)


# -----------------------------
# Main selection pipeline
# -----------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COCO_CAPTION_FILE.exists():
        raise FileNotFoundError(
            f"Could not find caption file: {COCO_CAPTION_FILE}\n"
            "Expected file: data/coco/raw/captions_val2017.json"
        )

    print("Loading COCO captions...")
    captions_by_image_id = load_coco_captions(COCO_CAPTION_FILE)

    print("Loading COCO validation subset from FiftyOne...")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=MAX_SAMPLES,
        label_types=["detections", "segmentations"],
    )

    rows = []

    for sample in dataset:
        image_path = Path(sample.filepath)
        coco_id = extract_coco_id_from_path(image_path)

        captions = captions_by_image_id.get(coco_id, [])
        caption_score, best_caption = get_caption_score(captions)

        detection_score, num_objects, num_categories, spread_score = get_detection_score(sample)
        edge_score, edge_density = get_edge_score(image_path)

        # Weighted total score
        # Detection/layout matters most, edge quality second, caption quality third.
        total_score = (
            0.45 * detection_score
            + 0.35 * edge_score
            + 0.20 * caption_score
        )

        rows.append({
            "coco_id": coco_id,
            "filepath": str(image_path),
            "caption": best_caption,
            "total_score": round(total_score, 4),
            "detection_score": round(detection_score, 4),
            "edge_score": round(edge_score, 4),
            "caption_score": round(caption_score, 4),
            "num_objects": num_objects,
            "num_categories": num_categories,
            "spread_score": round(spread_score, 4),
            "edge_density": round(edge_density, 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("total_score", ascending=False)

    selected = df.head(TOP_K)
    selected.to_csv(OUTPUT_CSV, index=False)

    from shutil import copy2

    SELECTED_IMAGE_DIR = OUTPUT_DIR / "images"
    SELECTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in selected.iterrows():
        src = Path(row["filepath"])
        dst = SELECTED_IMAGE_DIR / f"{int(row['coco_id']):012d}.jpg"
        copy2(src, dst)

    print(f"Copied selected images to: {SELECTED_IMAGE_DIR}")

    print(f"\nSaved selected candidates to: {OUTPUT_CSV}")
    print("\nTop selected candidates:")
    print(
        selected[
            [
                "coco_id",
                "total_score",
                "num_objects",
                "num_categories",
                "edge_density",
                "caption",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()