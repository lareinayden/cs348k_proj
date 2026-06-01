# scripts/run_editing_canny_vs_seg.py

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from editing_tasks import EDIT_TASKS
from caption_utils import get_coco_caption
from image_utils import load_rgb_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_CSV = PROJECT_ROOT / "data/selected_200/selected_coco_candidates_200_shortened.csv"
SELECTED_IMAGES = PROJECT_ROOT / "data/selected_200/images"
OUT_DIR = PROJECT_ROOT / "outputs/editing_study"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CANNY_CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
SEG_CONTROLNET_MODEL = "lllyasviel/sd-controlnet-seg"


def find_image_path(coco_id: int) -> Path:
    candidates = [
        SELECTED_IMAGES / f"{int(coco_id):012d}.jpg",
        SELECTED_IMAGES / f"{int(coco_id):012d}.png",
        SELECTED_IMAGES / f"{int(coco_id)}.jpg",
        SELECTED_IMAGES / f"{int(coco_id)}.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"Could not find image for COCO ID {coco_id}")


def load_target_image(coco_id: int, size: int = 512) -> Image.Image:
    image_path = find_image_path(coco_id)
    img = load_rgb_image(image_path)

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    return img.convert("RGB").resize((size, size))


import pandas as pd

selected_df = pd.read_csv(SELECTED_CSV)

def get_row_for_coco_id(coco_id: int):
    matches = selected_df[selected_df["coco_id"].astype(int) == int(coco_id)]

    if matches.empty:
        raise ValueError(f"COCO ID {coco_id} not found in {SELECTED_CSV}")

    return matches.iloc[0]


def get_base_caption(coco_id: int) -> str:
    row = get_row_for_coco_id(coco_id)
    return get_coco_caption(row)


def make_canny_condition(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges).convert("RGB")


def make_seg_condition_from_outputs(coco_id: int) -> Image.Image:
    """
    Loads a previously saved segmentation condition image from earlier segmentation runs.
    This avoids rerunning UperNet.
    """
    all_candidates = list(
        (PROJECT_ROOT / "outputs").rglob(f"{int(coco_id):012d}/seg.png")
    )

    candidates = [
        p for p in all_candidates
        if "seg" in str(p).lower() and "canny" not in str(p).lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find saved segmentation condition for COCO ID {coco_id}. "
            "Expected something like outputs/controlnet_seg/.../<id>/seg.png"
        )

    return Image.open(candidates[0]).convert("RGB").resize((512, 512))


def build_prompt(base_caption: str, task_name: str) -> str:
    suffix = EDIT_TASKS[task_name]["prompt_suffix"]
    prompt = f"{base_caption}. {suffix}."

    # Rough word cap to avoid CLIP text truncation.
    words = prompt.split()
    if len(words) > 65:
        prompt = " ".join(words[:65])

    return prompt


def load_pipeline(modality: str, device: str):
    if modality == "canny":
        controlnet_id = CANNY_CONTROLNET_MODEL
    elif modality == "seg":
        controlnet_id = SEG_CONTROLNET_MODEL
    else:
        raise ValueError(f"Unknown modality: {modality}")

    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    return pipe


def save_prompt(out_subdir: Path, prompt: str, task_name: str, modality: str, coco_id: int):
    with open(out_subdir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(f"COCO ID: {coco_id}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Modality: {modality}\n\n")
        f.write(prompt)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=list(EDIT_TASKS.keys()))

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--control-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])

    args = parser.parse_args()

    if args.ids is None:
        task_to_ids = {
            task_name: EDIT_TASKS[task_name]["sample_ids"]
            for task_name in args.tasks
        }
    else:
        task_to_ids = {
            task_name: args.ids
            for task_name in args.tasks
        }

    print("Loading Canny ControlNet pipeline...")
    canny_pipe = load_pipeline("canny", args.device)

    print("Loading Segmentation ControlNet pipeline...")
    seg_pipe = load_pipeline("seg", args.device)

    for task_name in args.tasks:
        if task_name not in EDIT_TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(EDIT_TASKS.keys())}")

        for coco_id in task_to_ids[task_name]:
            print(f"\nRunning task={task_name}, COCO ID={coco_id}")

            target = load_target_image(coco_id)
            base_caption = get_base_caption(coco_id)
            prompt = build_prompt(base_caption, task_name)

            canny_condition = make_canny_condition(target)
            seg_condition = make_seg_condition_from_outputs(coco_id)

            for modality, pipe, condition in [
                ("canny", canny_pipe, canny_condition),
                ("seg", seg_pipe, seg_condition),
            ]:
                generator = torch.Generator(device=args.device).manual_seed(args.seed)

                out_subdir = OUT_DIR / task_name / modality / f"{coco_id:012d}"
                out_subdir.mkdir(parents=True, exist_ok=True)

                result = pipe(
                    prompt=prompt,
                    image=condition,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    controlnet_conditioning_scale=args.control_scale,
                    generator=generator,
                ).images[0]

                result.save(out_subdir / "generated.png")
                target.save(out_subdir / "target.png")
                condition.save(out_subdir / "condition.png")
                save_prompt(out_subdir, prompt, task_name, modality, coco_id)

                print(f"Saved {task_name} | {modality} | {coco_id} -> {out_subdir}")


if __name__ == "__main__":
    main()