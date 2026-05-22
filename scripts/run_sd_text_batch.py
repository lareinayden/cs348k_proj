"""
Batch text-only Stable Diffusion reconstruction (Configs 1 & 2).

  Config 1 — Baseline Semantic:  --modality sparse
  Config 2 — Advanced Semantic:  --modality dense
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from evaluation import GenerativeEvaluator
from sparse_prompts import get_sparse_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_CSV = PROJECT_ROOT / "data/selected/selected_coco_candidates.csv"
SELECTED_IMAGES = PROJECT_ROOT / "data/selected/images"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MAX_IMAGES = 40
DEFAULT_GUIDANCE = 7.5
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
    parser = argparse.ArgumentParser(description="Text-only SD batch (sparse or dense prompts).")
    parser.add_argument(
        "--modality",
        choices=["sparse", "dense"],
        required=True,
        help="sparse = Config 1 (entity list); dense = Config 2 (full COCO caption)",
    )
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0, help="Used on CUDA for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MODALITY_CONFIG[args.modality]
    output_root = cfg["output_root"]
    result_csv = cfg["result_csv"]

    output_root.mkdir(parents=True, exist_ok=True)
    result_csv.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()
    selected_df = pd.read_csv(SELECTED_CSV).head(args.max_images)

    pipe = load_sd_pipeline(device, dtype)
    evaluator = GenerativeEvaluator(device=device)

    all_results = []
    print(f"Experiment {cfg['experiment_id']} ({cfg['modality']}), n={len(selected_df)}")

    for idx, row in selected_df.iterrows():
        coco_id = int(row["coco_id"])
        dense_caption = row["caption"]
        prompt = build_prompt(args.modality, coco_id, dense_caption)
        image_path = resolve_image_path(coco_id, row["filepath"])

        print(f"\n[{idx + 1}/{len(selected_df)}] COCO {coco_id}")
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
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        target_path = sample_dir / "target.png"
        generated_path = sample_dir / "generated.png"
        target.save(target_path)
        result.save(generated_path)

        # CLIP-Score always uses ground-truth dense caption for fair comparison across configs
        scores = evaluator.evaluate(target, result, dense_caption)

        result_row = {
            "coco_id": coco_id,
            "caption": dense_caption,
            "prompt": prompt,
            "target_path": str(target_path),
            "generated_path": str(generated_path),
            "modality": cfg["modality"],
            "experiment_id": cfg["experiment_id"],
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            **scores,
        }
        all_results.append(result_row)
        pd.DataFrame(all_results).to_csv(result_csv, index=False)
        print(f"Scores: {scores}")

    print(f"\nDone. Results: {result_csv}")
    print(f"Images: {output_root}")


if __name__ == "__main__":
    main()
