"""
Batch ControlNet-Seg reconstruction.

Config 4 — Baseline Structural B:
    --modality empty

Config 6 — Multi-modal B:
    --modality dense

Uses fixed selected images:
    data/selected/images/

Supports CFG sweep:
    --guidance-scales 3.0 5.0 7.5 10.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from evaluation import GenerativeEvaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_CSV = PROJECT_ROOT / "data/selected/selected_coco_candidates.csv"
SELECTED_IMAGES = PROJECT_ROOT / "data/selected/images"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-seg"
SEG_MODEL = "openmmlab/upernet-convnext-small"

MAX_IMAGES = 40
DEFAULT_STEPS = 20
DEFAULT_CONTROL_SCALE = 1.0


ADE_PALETTE = np.asarray([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
], dtype=np.uint8)


def scale_to_str(x):
    return str(x).replace(".", "p")


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def resolve_selected_image_path(coco_id):
    path = SELECTED_IMAGES / f"{coco_id:012d}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"Selected image not found: {path}")
    return path


def build_prompt(modality, caption):
    if modality == "empty":
        return ""
    if modality == "dense":
        return caption
    raise ValueError(f"Unknown modality: {modality}")


def load_segmentation_model():
    print("Loading segmentation model on CPU...")
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = UperNetForSemanticSegmentation.from_pretrained(SEG_MODEL)
    model = model.to("cpu")
    model.eval()
    return processor, model


def make_segmentation_image(image, processor, model):
    original_size = image.size  # (w, h)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )

    seg = logits.argmax(dim=1)[0].detach().cpu().numpy()
    color_seg = ADE_PALETTE[seg % len(ADE_PALETTE)]
    return Image.fromarray(color_seg).resize((512, 512))


def load_controlnet_seg_pipeline(device, dtype):
    print(f"Loading ControlNet-Seg pipeline on {device}...")

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        choices=["empty", "dense"],
        required=True,
        help="empty = Config 4; dense = Config 6",
    )
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[7.5])
    parser.add_argument("--controlnet-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.modality == "dense":
        modality_name = "dense_text_plus_seg"
        experiment_id = 6
        output_root_base = PROJECT_ROOT / "outputs/controlnet_seg"
        result_csv = PROJECT_ROOT / "logs/controlnet_seg_results.csv"
    else:
        modality_name = "empty_text_plus_seg"
        experiment_id = 4
        output_root_base = PROJECT_ROOT / "outputs/baseline_structural_seg"
        result_csv = PROJECT_ROOT / "logs/baseline_structural_seg_results.csv"

    output_root_base.mkdir(parents=True, exist_ok=True)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()

    selected_df = pd.read_csv(SELECTED_CSV).head(args.max_images)

    seg_processor, seg_model = load_segmentation_model()
    pipe = load_controlnet_seg_pipeline(device, dtype)
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
                caption = row["caption"]
                image_path = resolve_selected_image_path(coco_id)

                print(f"\n[{idx + 1}/{len(selected_df)}] Processing COCO {coco_id}")
                print(f"Caption: {caption}")

                sample_dir = output_root / f"{coco_id:012d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                target = Image.open(image_path).convert("RGB").resize((512, 512))
                seg_image = make_segmentation_image(target, seg_processor, seg_model)

                prompt = build_prompt(args.modality, caption)

                generator = None
                if device == "cuda":
                    generator = torch.Generator(device=device).manual_seed(args.seed)

                result = pipe(
                    prompt=prompt,
                    image=seg_image,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_scale,
                    generator=generator,
                ).images[0]

                target_path = sample_dir / "target.png"
                seg_path = sample_dir / "seg.png"
                generated_path = sample_dir / "generated.png"

                target.save(target_path)
                seg_image.save(seg_path)
                result.save(generated_path)

                scores = evaluator.evaluate(target, result, caption)

                result_row = {
                    "coco_id": coco_id,
                    "caption": caption,
                    "prompt": prompt,
                    "target_path": str(target_path),
                    "seg_path": str(seg_path),
                    "generated_path": str(generated_path),
                    "modality": modality_name,
                    "experiment_id": experiment_id,
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