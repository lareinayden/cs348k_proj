from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == "mps" else torch.float16

image_path = Path("data/selected/images/000000000139.jpg")  # change this
prompt = "a realistic photo matching the given edge layout"

output_dir = Path("outputs/controlnet_canny")
output_dir.mkdir(parents=True, exist_ok=True)

# Load target image
target = Image.open(image_path).convert("RGB").resize((512, 512))

# Build Canny edge map
img_np = np.array(target)
edges = cv2.Canny(img_np, 100, 200)
edges = np.stack([edges] * 3, axis=-1)
canny_image = Image.fromarray(edges)

# Load ControlNet + Stable Diffusion
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
)

pipe = pipe.to(device)

# Generate
generator = torch.Generator(device=device).manual_seed(0) if device != "mps" else None

result = pipe(
    prompt=prompt,
    image=canny_image,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=generator,
).images[0]

target.save(output_dir / "target.png")
canny_image.save(output_dir / "canny.png")
result.save(output_dir / "generated.png")

print(f"Saved results to {output_dir}")