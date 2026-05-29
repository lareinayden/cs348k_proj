"""
Batch ControlNet-Canny reconstruction on the selected COCO subset.

  Config 3 — Baseline Structural A:  --modality empty   (empty text + Canny)
  Config 5 — Multi-modal A:          --modality dense   (dense caption + Canny)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

from evaluation import GenerativeEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_CSV = PROJECT_ROOT / "data/selected/selected_coco_candidates.csv"
SELECTED_IMAGES = PROJECT_ROOT / "data/selected/images"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
MAX_IMAGES = 40
DEFAULT_GUIDANCE = 7.5
DEFAULT_CN_SCALE = 1.0
DEFAULT_STEPS = 20

# Empty prompt for Config 3 (structure-only intent; CLIP still scored with dense caption)
EMPTY_PROMPT = ""

MODALITY_CONFIG = {
    "empty": {
        "experiment_id": 3,
        "modality": "empty_text_plus_canny",
        "output_root": PROJECT_ROOT / "outputs/baseline_structural_canny",
        "result_csv": PROJECT_ROOT / "logs/baseline_structural_canny_results.csv",
    },
    "dense": {
        "experiment_id": 5,
        "modality": "dense_text_plus_canny",
        "output_root": PROJECT_ROOT / "outputs/controlnet_canny",
        "result_csv": PROJECT_ROOT / "logs/controlnet_canny_results.csv",
    },
}


def scale_to_str(x):
    return str(x).replace(".", "p")


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def resolve_image_path(coco_id: int, filepath: str) -> Path:
    path = Path(filepath)
    if path.is_file():
        return path
    local = SELECTED_IMAGES / f"{coco_id:012d}.jpg"
    if local.is_file():
        return local
    coco_val = PROJECT_ROOT / "data/coco/validation/data" / f"{coco_id:012d}.jpg"
    if coco_val.is_file():
        return coco_val
    raise FileNotFoundError(f"No image for COCO {coco_id}: tried {filepath}, {local}, {coco_val}")


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
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
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
    cfg = MODALITY_CONFIG[args.modality]
    output_root_base = cfg["output_root"]
    result_csv = cfg["result_csv"]
    modality_name = cfg["modality"]

    output_root_base.mkdir(parents=True, exist_ok=True)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()
    selected_df = pd.read_csv(SELECTED_CSV).head(args.max_images)

    pipe = load_controlnet_pipeline(device, dtype)
    evaluator = GenerativeEvaluator(device=device)

    all_results = []
    print(f"Experiment {cfg['experiment_id']} ({modality_name}), n={len(selected_df)}")


    for guidance_scale in args.guidance_scales:
        for controlnet_scale in args.controlnet_scales:

            cfg_dir = f"cfg_{scale_to_str(guidance_scale)}_control_{scale_to_str(controlnet_scale)}"
            output_root = output_root_base / cfg_dir
            output_root.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Running {modality_name} | CFG={guidance_scale} | ControlNet={controlnet_scale} ===")

            for idx, row in selected_df.iterrows():
                coco_id = int(row["coco_id"])
                caption = row["caption"]

                image_path = resolve_selected_image_path(coco_id)

                print(f"\n[{idx + 1}/{len(selected_df)}] Processing COCO {coco_id}")
                print(f"Caption: {caption}")

                sample_dir = output_root / f"{coco_id:012d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                target = Image.open(image_path).convert("RGB").resize((512, 512))
                canny_image = make_canny_image(target)

                prompt = build_prompt(args.modality, caption)

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

                scores = evaluator.evaluate(target, result, caption)

                result_row = {
                    "coco_id": coco_id,
                    "caption": caption,
                    "prompt": prompt,
                    "target_path": str(target_path),
                    "canny_path": str(canny_path),
                    "generated_path": str(generated_path),
                    "modality": modality_name,
                    "experiment_id": cfg["experiment_id"],
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
