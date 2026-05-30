from pathlib import Path
import argparse

import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from evaluation import GenerativeEvaluator
from caption_utils import get_coco_caption, get_dense_caption
from image_utils import load_rgb_image, validate_selected_images


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_CSV = PROJECT_ROOT / "data/selected_200/selected_coco_candidates_200_shortened.csv"
SELECTED_IMAGES = PROJECT_ROOT / "data/selected_200/images"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

MAX_IMAGES = 200


def scale_to_str(x):
    return str(x).replace(".", "p")


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    elif torch.cuda.is_available():
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32


def resolve_selected_image_path(coco_id):
    path = SELECTED_IMAGES / f"{coco_id:012d}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"Selected image not found: {path}")
    return path


def make_canny_image(target_img, low=100, high=200):
    img_np = np.array(target_img)
    edges = cv2.Canny(img_np, low, high)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


def load_controlnet_pipeline(device, dtype):
    print(f"Loading ControlNet pipeline on {device}...")

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    return pipe


def build_prompt(modality, caption):
    if modality == "empty":
        return ""
    if modality == "dense":
        return caption
    raise ValueError(f"Unknown modality: {modality}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        choices=["empty", "dense"],
        required=True,
        help="empty = Config 3; dense = Config 5",
    )
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[7.5])
    parser.add_argument("--controlnet-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.modality == "dense":
        modality_name = "dense_text_plus_canny"
        output_root_base = PROJECT_ROOT / "outputs/controlnet_canny"
        result_csv = PROJECT_ROOT / "logs/controlnet_canny_results.csv"
    else:
        modality_name = "empty_text_plus_canny"
        output_root_base = PROJECT_ROOT / "outputs/baseline_structural_canny"
        result_csv = PROJECT_ROOT / "logs/baseline_structural_canny_results.csv"

    output_root_base.mkdir(parents=True, exist_ok=True)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()

    # Use fixed selected 40 images only
    selected_df = pd.read_csv(SELECTED_CSV).head(args.max_images)
    validate_selected_images(selected_df["coco_id"].tolist(), SELECTED_IMAGES)

    pipe = load_controlnet_pipeline(device, dtype)
    evaluator = GenerativeEvaluator(device=device)

    all_results = []

    for guidance_scale in args.guidance_scales:
        for controlnet_scale in args.controlnet_scales:

            cfg_dir = f"cfg_{scale_to_str(guidance_scale)}_control_{scale_to_str(controlnet_scale)}"
            output_root = output_root_base / cfg_dir
            output_root.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Running {modality_name} | CFG={guidance_scale} | ControlNet={controlnet_scale} ===")

            for idx, row in selected_df.iterrows():
                coco_id = int(row["coco_id"])
                dense_caption = get_dense_caption(row)
                coco_caption = get_coco_caption(row)

                image_path = resolve_selected_image_path(coco_id)

                print(f"\n[{idx + 1}/{len(selected_df)}] Processing COCO {coco_id}")
                print(f"COCO caption: {coco_caption}")
                print(f"Dense caption: {dense_caption}")

                sample_dir = output_root / f"{coco_id:012d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                target = load_rgb_image(image_path)
                canny_image = make_canny_image(target)

                prompt = build_prompt(args.modality, dense_caption)

                generator = None
                if device == "cuda":
                    generator = torch.Generator(device=device).manual_seed(args.seed)

                result = pipe(
                    prompt=prompt,
                    image=canny_image,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_scale,
                    generator=generator,
                ).images[0]

                target_path = sample_dir / "target.png"
                canny_path = sample_dir / "canny.png"
                generated_path = sample_dir / "generated.png"

                target.save(target_path)
                canny_image.save(canny_path)
                result.save(generated_path)

                scores = evaluator.evaluate(target, result, dense_caption)

                result_row = {
                    "coco_id": coco_id,
                    "caption": coco_caption,
                    "dense_caption": dense_caption,
                    "prompt": prompt,
                    "target_path": str(target_path),
                    "canny_path": str(canny_path),
                    "generated_path": str(generated_path),
                    "modality": modality_name,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": controlnet_scale,
                    "num_inference_steps": args.num_inference_steps,
                    **scores,
                }

                all_results.append(result_row)
                pd.DataFrame(all_results).to_csv(result_csv, index=False)

                print(f"Scores: {scores}")

    print(f"\nDone. Results saved to {result_csv}")
    print(f"Images saved to {output_root_base}")


if __name__ == "__main__":
    main()