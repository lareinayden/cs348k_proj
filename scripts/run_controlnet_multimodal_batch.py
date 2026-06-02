"""
Batch Multi-ControlNet reconstruction — Config 7.

Dense text + Canny edge map + semantic segmentation mask (dual ControlNet).

Examples:
  python scripts/run_controlnet_multimodal_batch.py
  python scripts/run_controlnet_multimodal_batch.py --guidance-scale 5.0 --controlnet-scale 1.5
  python scripts/run_controlnet_multimodal_batch.py --guidance-scale 7.5 \\
      --canny-controlnet-scale 1.5 --seg-controlnet-scale 0.75
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from conditioning_images import (
    load_segmentation_model,
    make_canny_image,
    make_segmentation_image,
)
from controlnet_batch_utils import (
    append_result,
    is_multimodal_completed,
    load_existing_results,
)
from evaluation import GenerativeEvaluator
from caption_utils import get_coco_caption, get_dense_caption
from image_utils import load_rgb_image, validate_selected_images


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_CSV = PROJECT_ROOT / "data/selected_200/selected_coco_candidates_200_shortened.csv"
SELECTED_IMAGES = PROJECT_ROOT / "data/selected_200/images"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CANNY_CONTROLNET = "lllyasviel/sd-controlnet-canny"
SEG_CONTROLNET = "lllyasviel/sd-controlnet-seg"

EXPERIMENT_ID = 7
MODALITY_NAME = "dense_text_plus_canny_plus_seg"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/controlnet_multimodal"
RESULT_CSV = PROJECT_ROOT / "logs/controlnet_multimodal_results.csv"

MAX_IMAGES = 200
DEFAULT_STEPS = 20
DEFAULT_CFG = 5.0
DEFAULT_CONTROLNET_SCALE = 1.5


def scale_to_str(x: float) -> str:
    return str(x).replace(".", "p")


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def resolve_selected_image_path(coco_id: int) -> Path:
    path = SELECTED_IMAGES / f"{coco_id:012d}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"Selected image not found: {path}")
    return path


def resolve_controlnet_scales(
    controlnet_scale: float | None,
    canny_scale: float | None,
    seg_scale: float | None,
) -> tuple[float, float]:
    """Resolve per-branch scales; --controlnet-scale applies to any unset branch."""
    default = (
        controlnet_scale
        if controlnet_scale is not None
        else DEFAULT_CONTROLNET_SCALE
    )
    if canny_scale is None and seg_scale is None:
        return default, default
    canny = canny_scale if canny_scale is not None else default
    seg = seg_scale if seg_scale is not None else default
    return canny, seg


def multimodal_output_subdir(
    guidance_scale: float,
    canny_scale: float,
    seg_scale: float,
) -> str:
    cfg_part = f"cfg_{scale_to_str(guidance_scale)}"
    if canny_scale == seg_scale:
        return f"{cfg_part}_control_{scale_to_str(canny_scale)}"
    return (
        f"{cfg_part}_canny_{scale_to_str(canny_scale)}"
        f"_seg_{scale_to_str(seg_scale)}"
    )


def load_multicontrolnet_pipeline(device, dtype):
    print(f"Loading dual ControlNet pipeline (Canny + Seg) on {device}...")

    controlnets = [
        ControlNetModel.from_pretrained(CANNY_CONTROLNET, torch_dtype=dtype),
        ControlNetModel.from_pretrained(SEG_CONTROLNET, torch_dtype=dtype),
    ]

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnets,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Config 7 — dense text + Canny + seg (dual ControlNet)."
    )
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_CFG)
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=None,
        help=(
            "ControlNet scale for both branches when --canny-controlnet-scale / "
            "--seg-controlnet-scale are omitted (default: 1.5). Also fills in any "
            "single branch-specific scale that is omitted."
        ),
    )
    parser.add_argument(
        "--canny-controlnet-scale",
        type=float,
        default=None,
        help="ControlNet conditioning scale for the Canny branch only.",
    )
    parser.add_argument(
        "--seg-controlnet-scale",
        type=float,
        default=None,
        help="ControlNet conditioning scale for the segmentation branch only.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--result-csv", type=Path, default=RESULT_CSV)
    return parser.parse_args()


def main():
    args = parse_args()

    canny_scale, seg_scale = resolve_controlnet_scales(
        args.controlnet_scale,
        args.canny_controlnet_scale,
        args.seg_controlnet_scale,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.result_csv.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()

    selected_df = pd.read_csv(SELECTED_CSV).head(args.max_images)
    validate_selected_images(selected_df["coco_id"].tolist(), SELECTED_IMAGES)

    seg_processor, seg_model = load_segmentation_model()
    pipe = load_multicontrolnet_pipeline(device, dtype)
    evaluator = GenerativeEvaluator(device=device)

    results_df = load_existing_results(args.result_csv)
    skipped = 0

    cfg_dir = multimodal_output_subdir(
        args.guidance_scale, canny_scale, seg_scale
    )
    output_root = args.output_root / cfg_dir
    output_root.mkdir(parents=True, exist_ok=True)

    cn_scales = [canny_scale, seg_scale]
    shared_scale = canny_scale if canny_scale == seg_scale else None

    if canny_scale == seg_scale:
        scale_msg = f"ControlNet scale={canny_scale} (Canny + Seg)"
    else:
        scale_msg = f"Canny scale={canny_scale}, Seg scale={seg_scale}"

    print(
        f"Experiment {EXPERIMENT_ID} ({MODALITY_NAME}), "
        f"n={len(selected_df)}, CFG={args.guidance_scale}, "
        f"{scale_msg}, existing rows={len(results_df)}"
    )

    for idx, row in selected_df.iterrows():
        coco_id = int(row["coco_id"])
        if is_multimodal_completed(
            results_df, coco_id, args.guidance_scale, canny_scale, seg_scale
        ):
            skipped += 1
            continue

        dense_caption = get_dense_caption(row)
        coco_caption = get_coco_caption(row)
        image_path = resolve_selected_image_path(coco_id)

        print(f"\n[{idx + 1}/{len(selected_df)}] COCO {coco_id}")
        print(f"COCO caption: {coco_caption}")
        print(f"Dense caption: {dense_caption}")

        sample_dir = output_root / f"{coco_id:012d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        target = load_rgb_image(image_path)
        canny_image = make_canny_image(target)
        seg_image = make_segmentation_image(target, seg_processor, seg_model)

        generator = None
        if device == "cuda":
            generator = torch.Generator(device=device).manual_seed(args.seed)

        result = pipe(
            prompt=dense_caption,
            image=[canny_image, seg_image],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=cn_scales,
            generator=generator,
        ).images[0]

        target_path = sample_dir / "target.png"
        canny_path = sample_dir / "canny.png"
        seg_path = sample_dir / "seg.png"
        generated_path = sample_dir / "generated.png"

        target.save(target_path)
        canny_image.save(canny_path)
        seg_image.save(seg_path)
        result.save(generated_path)

        scores = evaluator.evaluate(target, result, dense_caption)

        result_row = {
            "coco_id": coco_id,
            "caption": coco_caption,
            "dense_caption": dense_caption,
            "prompt": dense_caption,
            "target_path": str(target_path),
            "canny_path": str(canny_path),
            "seg_path": str(seg_path),
            "generated_path": str(generated_path),
            "modality": MODALITY_NAME,
            "experiment_id": EXPERIMENT_ID,
            "guidance_scale": args.guidance_scale,
            "controlnet_conditioning_scale": shared_scale,
            "canny_controlnet_scale": canny_scale,
            "seg_controlnet_scale": seg_scale,
            "num_inference_steps": args.num_inference_steps,
            **scores,
        }

        results_df = append_result(results_df, result_row)
        results_df.to_csv(args.result_csv, index=False)
        print(f"Scores: {scores}")

    total = len(selected_df)
    print(
        f"\nDone. Results saved to {args.result_csv} "
        f"({len(results_df)} rows, skipped {skipped}/{total})"
    )
    print(f"Images saved to {output_root}")


if __name__ == "__main__":
    main()
