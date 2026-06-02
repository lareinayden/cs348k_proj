# scripts/make_editing_grid.py

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "editing_study"
FIG_DIR = PROJECT_ROOT / "figures"

IMAGE_SIZE = 256
LABEL_HEIGHT = 48
PADDING = 10


def load_square(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMAGE_SIZE, IMAGE_SIZE))

    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    x = (IMAGE_SIZE - img.width) // 2
    y = (IMAGE_SIZE - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_text(draw, text, box, font):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text(
        (x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2),
        text,
        fill="black",
        font=font,
    )


def make_grid(task_name, ids):
    columns = ["Target", "Canny Edit", "Seg Edit"]
    n_rows = len(ids)
    n_cols = len(columns)

    cell_w = IMAGE_SIZE
    cell_h = LABEL_HEIGHT + IMAGE_SIZE

    grid_w = n_cols * cell_w + (n_cols + 1) * PADDING
    grid_h = n_rows * cell_h + (n_rows + 1) * PADDING

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for r, coco_id in enumerate(ids):
        paths = [
            OUT_DIR / task_name / "canny" / f"{coco_id:012d}" / "target.png",
            OUT_DIR / task_name / "canny" / f"{coco_id:012d}" / "generated.png",
            OUT_DIR / task_name / "seg" / f"{coco_id:012d}" / "generated.png",
        ]

        for c, (col, path) in enumerate(zip(columns, paths)):
            x = PADDING + c * (cell_w + PADDING)
            y = PADDING + r * (cell_h + PADDING)

            label = col if c != 0 else f"Target\nID {coco_id}"

            # Simpler single-line label fallback
            label = label.replace("\n", " ")

            draw_text(draw, label, (x, y, x + cell_w, y + LABEL_HEIGHT), font)

            if path.exists():
                img = load_square(path)
            else:
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "lightgray")

            grid.paste(img, (x, y + LABEL_HEIGHT))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"editing_{task_name}_grid.png"
    grid.save(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--ids", nargs="+", type=int, required=True)
    args = parser.parse_args()

    make_grid(args.task, args.ids)


if __name__ == "__main__":
    main()