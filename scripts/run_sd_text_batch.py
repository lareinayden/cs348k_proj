"""
Batch text-only Stable Diffusion reconstruction (Configs 1 & 2).

  Config 1 — Baseline Semantic:  --modality sparse
  Config 2 — Advanced Semantic:  --modality dense

Uses only the fixed selected images under:
  data/selected/images/

Supports CFG sweep:
  --guidance-scales 3.0 5.0 7.5 10.0
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from evaluation import GenerativeEvaluator
from caption_utils import get_coco_caption, get_dense_caption
from sparse_prompts import get_sparse_prompt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_CSV = PROJECT_ROOT / "data/selected/selected_coco_candidates.csv"
SELECTED_IMAGES = PROJECT_ROOT / "data/selected/images"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MAX_IMAGES = 40
DEFAULT_STEPS = 20

MODALITY_CONFIG = {
    "sparse": {
        "experiment_id": 1,
        "modality": "sparse_text_only",
        "output_root": PROJECT_ROOT / "outputs/baseline_semantic_sparse",
        "result_csv": PROJECT_ROOT / "logs/baseline_semantic_sparse_results.csv",
    },
    "dense": {
        "experiment_id": 2,
        "modality": "dense_text_only",
        "output_root": PROJECT_ROOT / "outputs/baseline_semantic_dense",
        "result_csv": PROJECT_ROOT / "logs/baseline_semantic_dense_results.csv",
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


def resolve_selected_image_path(coco_id: int) -> Path:
    path = SELECTED_IMAGES / f"{coco_id:012d}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"Selected image not found: {path}")
    return path


def load_sd_pipeline(device: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    print(f"Loading Stable Diffusion pipeline on {device}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def build_prompt(modality: str, coco_id: int, dense_caption: str) -> str:
    if modality == "dense":
        return dense_caption
    return get_sparse_prompt(coco_id, dense_caption)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-only SD batch (sparse or dense prompts)."
    )
    parser.add_argument(
        "--modality",
        choices=["sparse", "dense"],
        required=True,
        help="sparse = Config 1 (entity list); dense = Config 2 (Dense Caption column)",
    )
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument(
        "--guidance-scales",
        type=float,
        nargs="+",
        default=[7.5],
        help="One or more CFG values, e.g. --guidance-scales 3.0 5.0 7.5 10.0",
    )
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0, help="Used on CUDA for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MODALITY_CONFIG[args.modality]

    output_root_base = cfg["output_root"]
    result_csv = cfg["result_csv"]

    output_root_base.mkdir(parents=True, exist_ok=True)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()

    # Use only the fixed selected image list
    selected_df = pd.read_csv(SELECTED_CSV).head(args.max_images)

    pipe = load_sd_pipeline(device, dtype)
    evaluator = GenerativeEvaluator(device=device)

    all_results = []

    print(
        f"Experiment {cfg['experiment_id']} ({cfg['modality']}), "
        f"n={len(selected_df)}, CFG sweep={args.guidance_scales}"
    )

    for guidance_scale in args.guidance_scales:
        cfg_dir = f"cfg_{scale_to_str(guidance_scale)}"
        output_root = output_root_base / cfg_dir
        output_root.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Running {cfg['modality']} | CFG={guidance_scale} ===")

        for idx, row in selected_df.iterrows():
            coco_id = int(row["coco_id"])
            dense_caption = get_dense_caption(row)
            coco_caption = get_coco_caption(row)
            prompt = build_prompt(args.modality, coco_id, dense_caption)

            image_path = resolve_selected_image_path(coco_id)

            print(f"\n[{idx + 1}/{len(selected_df)}] COCO {coco_id}")
            print(f"COCO caption: {coco_caption}")
            print(f"Dense caption: {dense_caption}")
            print(f"Generation prompt: {prompt}")

            sample_dir = output_root / f"{coco_id:012d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            target = Image.open(image_path).convert("RGB").resize((512, 512))

            generator = None
            if device == "cuda":
                generator = torch.Generator(device=device).manual_seed(args.seed)

            result = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            target_path = sample_dir / "target.png"
            generated_path = sample_dir / "generated.png"

            target.save(target_path)
            result.save(generated_path)

            # CLIP-Score always uses the dense ground-truth caption for fair comparison
            scores = evaluator.evaluate(target, result, dense_caption)

            result_row = {
                "coco_id": coco_id,
                "caption": coco_caption,
                "dense_caption": dense_caption,
                "prompt": prompt,
                "target_path": str(target_path),
                "generated_path": str(generated_path),
                "modality": cfg["modality"],
                "experiment_id": cfg["experiment_id"],
                "guidance_scale": guidance_scale,
                "num_inference_steps": args.num_inference_steps,
                **scores,
            }

            all_results.append(result_row)

            # Save after each image so progress is not lost
            pd.DataFrame(all_results).to_csv(result_csv, index=False)

            print(f"Scores: {scores}")

    print(f"\nDone. Results saved to {result_csv}")
    print(f"Images saved to {output_root_base}")


if __name__ == "__main__":
    main()