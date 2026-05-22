from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from evaluation import GenerativeEvaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_CSV = PROJECT_ROOT / "data/selected/selected_coco_candidates.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/controlnet_canny"
RESULT_CSV = PROJECT_ROOT / "logs/controlnet_canny_results.csv"

MAX_IMAGES = 40  # start small for debugging; later change to 40

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    elif torch.cuda.is_available():
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32


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


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()

    selected_df = pd.read_csv(SELECTED_CSV)
    selected_df = selected_df.head(MAX_IMAGES)

    pipe = load_controlnet_pipeline(device, dtype)
    evaluator = GenerativeEvaluator(device=device)

    all_results = []

    for idx, row in selected_df.iterrows():
        coco_id = int(row["coco_id"])
        image_path = Path(row["filepath"])
        caption = row["caption"]

        print(f"\n[{idx + 1}/{len(selected_df)}] Processing COCO {coco_id}")
        print(f"Caption: {caption}")

        sample_dir = OUTPUT_ROOT / f"{coco_id:012d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Load target
        target = Image.open(image_path).convert("RGB").resize((512, 512))

        # Generate Canny condition
        canny_image = make_canny_image(target)

        # Use COCO caption as dense text prompt
        prompt = caption

        generator = None
        if device == "cuda":
            generator = torch.Generator(device=device).manual_seed(0)

        # Generate image
        result = pipe(
            prompt=prompt,
            image=canny_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=generator,
        ).images[0]

        # Save images
        target_path = sample_dir / "target.png"
        canny_path = sample_dir / "canny.png"
        generated_path = sample_dir / "generated.png"

        target.save(target_path)
        canny_image.save(canny_path)
        result.save(generated_path)

        # Evaluate generated image against target
        scores = evaluator.evaluate(target, result, caption)

        result_row = {
            "coco_id": coco_id,
            "caption": caption,
            "target_path": str(target_path),
            "canny_path": str(canny_path),
            "generated_path": str(generated_path),
            "modality": "dense_text_plus_canny",
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 1.0,
            **scores,
        }

        all_results.append(result_row)

        # Save after every image so progress is not lost
        pd.DataFrame(all_results).to_csv(RESULT_CSV, index=False)
        print(f"Scores: {scores}")

    print(f"\nDone. Results saved to {RESULT_CSV}")
    print(f"Images saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()