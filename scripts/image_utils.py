"""Validate and load selected subset images."""

from pathlib import Path

from PIL import Image


def validate_image(path: Path) -> str | None:
    """Return an error message if the image is missing or unreadable, else None."""
    if not path.exists():
        return "file not found"

    try:
        with Image.open(path) as img:
            img.load()
    except OSError as exc:
        return str(exc)
    except Exception as exc:
        return f"unreadable: {exc}"

    return None


def validate_selected_images(coco_ids: list[int], images_dir: Path) -> None:
    """Fail fast before loading models if any selected images are bad."""
    bad = []
    for coco_id in coco_ids:
        path = images_dir / f"{int(coco_id):012d}.jpg"
        error = validate_image(path)
        if error:
            bad.append((coco_id, path, error))

    if not bad:
        return

    lines = [
        f"Found {len(bad)} invalid image(s) under {images_dir}:",
        "",
    ]
    for coco_id, path, error in bad:
        size = path.stat().st_size if path.exists() else 0
        lines.append(f"  COCO {coco_id}: {path.name} ({size} bytes) — {error}")

    lines.extend(
        [
            "",
            "These files are usually incomplete copies. Re-copy them from a full COCO val2017 source, then rerun.",
        ]
    )
    raise ValueError("\n".join(lines))


def load_rgb_image(path: Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Load an image as RGB, validating it is complete before decode."""
    error = validate_image(path)
    if error:
        raise OSError(f"{path}: {error}")

    with Image.open(path) as img:
        return img.convert("RGB").resize(size)


if __name__ == "__main__":
    import argparse
    import sys

    import pandas as pd

    parser = argparse.ArgumentParser(description="Validate selected subset images.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data/selected_200/selected_coco_candidates_200_shortened.csv",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data/selected_200/images",
    )
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).head(args.max_images)
    try:
        validate_selected_images(df["coco_id"].tolist(), args.images_dir)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"All {len(df)} images OK under {args.images_dir}")
