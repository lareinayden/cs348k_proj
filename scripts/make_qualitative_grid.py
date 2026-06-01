# scripts/make_qualitative_grid.py

from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CFG_FOLDER = "cfg_5p0_control_1p0"

OUTPUT_DIRS = {
    "Target": PROJECT_ROOT / "outputs" / "baseline_semantic_dense",
    "Sparse Text": PROJECT_ROOT / "outputs" / "baseline_semantic_sparse",
    "Dense Text": PROJECT_ROOT / "outputs" / "baseline_semantic_dense",
    "Canny Only": PROJECT_ROOT / "outputs" / "baseline_structural_canny" / CFG_FOLDER,
    "Seg Only": PROJECT_ROOT / "outputs" / "baseline_structural_seg" / CFG_FOLDER,
    "Dense + Canny": PROJECT_ROOT / "outputs" / "controlnet_canny" / CFG_FOLDER,
    "Dense + Seg": PROJECT_ROOT / "outputs" / "controlnet_seg" / CFG_FOLDER,
}

OUT_PATH = PROJECT_ROOT / "figures" / "qualitative_grid.png"

IMAGE_SIZE = 256
LABEL_HEIGHT = 42
PADDING = 10
NUM_EXAMPLES = 2
SEED = 10

# Set manually if desired, e.g. [139, 2685]
SELECTED_IDS = ["12120", "27620"]


# -----------------------------
# Helpers
# -----------------------------
def find_image(folder: Path, coco_id: str, prefer_target: bool = False):
    stems = [
        f"{int(coco_id):012d}",
        str(int(coco_id)),
    ]

    wanted_names = ["target.png"] if prefer_target else ["generated.png"]

    # Search recursively for paths like:
    # outputs/.../000000012120/generated.png
    # outputs/.../cfg_5p0/000000012120/generated.png
    for stem in stems:
        for name in wanted_names:
            matches = sorted(folder.rglob(f"{stem}/{name}"))
            if matches:
                return matches[0]

    # Fallback: any image inside a folder with this ID
    for stem in stems:
        matches = []
        for ext in [".png", ".jpg", ".jpeg"]:
            matches.extend(sorted(folder.rglob(f"{stem}/*{ext}")))

        if matches:
            if prefer_target:
                preferred = [p for p in matches if "target" in p.name.lower()]
            else:
                preferred = [p for p in matches if "generated" in p.name.lower()]

            if preferred:
                return preferred[0]
            return matches[0]

    return None


def get_available_ids():
    source_folder = OUTPUT_DIRS["Dense Text"]

    candidate_ids = []
    for p in source_folder.iterdir():
        if p.is_dir() and p.name.isdigit():
            candidate_ids.append(str(int(p.name)))

    valid_ids = []

    for coco_id in candidate_ids:
        ok = True

        for label, folder in OUTPUT_DIRS.items():
            img_path = find_image(
                folder,
                coco_id,
                prefer_target=(label == "Target"),
            )

            if img_path is None:
                print(f"Missing {label} for ID {coco_id}")
                ok = False
                break

        if ok:
            valid_ids.append(coco_id)

    print(f"Found {len(valid_ids)} IDs with all outputs.")
    return valid_ids


def load_square(path: Path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMAGE_SIZE, IMAGE_SIZE))

    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    x = (IMAGE_SIZE - img.width) // 2
    y = (IMAGE_SIZE - img.height) // 2
    canvas.paste(img, (x, y))

    return canvas


def draw_centered_text(draw, text, box, font):
    x0, y0, x1, y1 = box
    lines = text.split("\n")

    heights = []
    widths = []

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])

    total_h = sum(heights) + 4 * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) // 2

    for line, w, h in zip(lines, widths, heights):
        x = x0 + (x1 - x0 - w) // 2
        draw.text((x, y), line, fill="black", font=font)
        y += h + 4


def make_grid(selected_ids):
    labels = list(OUTPUT_DIRS.keys())
    n_cols = len(labels)
    n_rows = len(selected_ids)

    cell_w = IMAGE_SIZE
    cell_h = LABEL_HEIGHT + IMAGE_SIZE

    grid_w = n_cols * cell_w + (n_cols + 1) * PADDING
    grid_h = n_rows * cell_h + (n_rows + 1) * PADDING

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        small_font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for r, coco_id in enumerate(selected_ids):
        for c, label in enumerate(labels):
            folder = OUTPUT_DIRS[label]
            img_path = find_image(
                folder,
                coco_id,
                prefer_target=(label == "Target"),
            )

            if img_path is None:
                tile = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "lightgray")
            else:
                tile = load_square(img_path)

            x = PADDING + c * (cell_w + PADDING)
            y = PADDING + r * (cell_h + PADDING)

            if label == "Target":
                label_text = f"Target\nID {int(coco_id)}"
                label_font = small_font
            else:
                label_text = label
                label_font = font

            draw_centered_text(
                draw,
                label_text,
                (x, y, x + cell_w, y + LABEL_HEIGHT),
                label_font,
            )

            grid.paste(tile, (x, y + LABEL_HEIGHT))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT_PATH)

    print(f"Saved qualitative grid to: {OUT_PATH}")
    print("Selected COCO IDs:", selected_ids)


def main():
    random.seed(SEED)

    if SELECTED_IDS is None:
        valid_ids = get_available_ids()

        if len(valid_ids) < NUM_EXAMPLES:
            raise RuntimeError(
                f"Only found {len(valid_ids)} IDs with all outputs. "
                "Check your output folder names/layout."
            )

        selected_ids = random.sample(valid_ids, NUM_EXAMPLES)
    else:
        selected_ids = [str(int(x)) for x in SELECTED_IDS]

    make_grid(selected_ids)


if __name__ == "__main__":
    main()