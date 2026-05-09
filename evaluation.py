import json
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.datasets import CocoCaptions
import numpy as np
from PIL import Image

# Metrics
import lpips
from torchmetrics.multimodal.clip_score import CLIPScore
from dreamsim import dreamsim

CLIP_MODEL_ID = "openai/clip-vit-base-patch16"


def _clip_feats_tensor(raw):
    """HF Transformers recent CLIP returns BaseModelOutputWithPooling; torchmetrics expects a Tensor."""
    if isinstance(raw, torch.Tensor):
        return raw
    po = getattr(raw, "pooler_output", None)
    if po is not None:
        return po
    raise TypeError(f"Unexpected CLIP feature type: {type(raw)}")


class _CLIPForTorchmetrics(torch.nn.Module):
    """Wrap CLIPModel so get_* methods return embedding tensors (torchmetrics compatibility)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @property
    def config(self):
        return self.inner.config

    def get_image_features(self, *args, **kwargs):
        return _clip_feats_tensor(self.inner.get_image_features(*args, **kwargs))

    def get_text_features(self, *args, **kwargs):
        return _clip_feats_tensor(self.inner.get_text_features(*args, **kwargs))

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


def _load_clip_for_metrics():
    from transformers import CLIPModel, CLIPProcessor

    base = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return _CLIPForTorchmetrics(base), processor


class GenerativeEvaluator:
    def __init__(self, device=None, info_print=print):
        # Dynamically select the best available hardware if none is explicitly provided
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"    # Apple Silicon Mac
            elif torch.cuda.is_available():
                self.device = "cuda"   # NVIDIA GPU
            else:
                self.device = "cpu"    # Standard Fallback
        else:
            self.device = device

        info_print(f"Initializing metrics on {self.device}...")

        # 1. Initialize LPIPS (Perceptual)
        self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_metric.eval()

        # 2. CLIP score: callable loader (safetensors + HF compatibility wrapper for torchmetrics)
        self.clip_metric = CLIPScore(model_name_or_path=_load_clip_for_metrics).to(self.device)

        # 3. Initialize DreamSim (Mid-level Intent/Layout)
        # DreamSim returns a model and a preprocessing function
        self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True, device=self.device)
        self.dreamsim_model.eval()

    def evaluate(self, target_img, generated_img, ground_truth_text):
        """Runs the trio of metrics on a single target/generated pair."""
        
        # --- Preprocessing ---
        # LPIPS expects tensors in [-1, 1] format
        transform_lpips = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        target_tensor_lpips = transform_lpips(target_img).unsqueeze(0).to(self.device)
        gen_tensor_lpips = transform_lpips(generated_img).unsqueeze(0).to(self.device)

        # DreamSim uses its own proprietary preprocessor
        target_tensor_dream = self.dreamsim_preprocess(target_img).to(self.device)
        gen_tensor_dream = self.dreamsim_preprocess(generated_img).to(self.device)

        # CLIP expects standard PIL-to-tensor uint8 for torchmetrics
        transform_clip = T.Compose([
            T.Resize((256, 256)),
            T.PILToTensor() 
        ])
        gen_tensor_clip = transform_clip(generated_img).to(self.device)

        # --- Metric Calculation ---
        with torch.no_grad():
            # LPIPS: Lower is better (0 is perfect match)
            lpips_score = self.lpips_metric(target_tensor_lpips, gen_tensor_lpips).item()

            # DreamSim: Distance metric, lower is better
            dreamsim_distance = self.dreamsim_model(target_tensor_dream, gen_tensor_dream).item()

            # CLIP-Score: Higher is better (measures text-to-image alignment)
            clip_score = self.clip_metric(gen_tensor_clip, ground_truth_text).item()

        return {
            "LPIPS_Distance": round(lpips_score, 4),
            "DreamSim_Distance": round(dreamsim_distance, 4),
            "CLIP_Score": round(clip_score, 4)
        }

def run_checkpoint_baseline(log_dir=None):
    log_root = Path(log_dir) if log_dir else Path(__file__).resolve().parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    # Pick the next incremental log index: evaluation_1.log, evaluation_2.log, ...
    existing_indices = []
    for p in log_root.glob("evaluation_*.log"):
        try:
            existing_indices.append(int(p.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    next_idx = (max(existing_indices) + 1) if existing_indices else 1
    log_path = log_root / f"evaluation_{next_idx}.log"

    results_summary = {"started_at": datetime.now().isoformat(timespec="seconds"), "runs": {}}

    with log_path.open("w", encoding="utf-8") as log_file:

        def log(msg=""):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        # Setup COCO Loader
        coco_val_dir = "./data/coco/validation/data"
        coco_ann_file = "./data/coco/raw/captions_val2017.json"

        log(f"Log file: {log_path}")
        log("Loading COCO Dataset...")
        dataset = CocoCaptions(root=coco_val_dir, annFile=coco_ann_file)

        # Extract one target image and its captions
        target_image, captions = dataset[0]
        # We enforce the "Sparse/Dense" text rule by using caption index 0 as our ground truth
        ground_truth_text = captions[0]
        results_summary["caption"] = ground_truth_text

        log(f"Target Caption: '{ground_truth_text}'")

        # Generate Trivial Baselines (matching target image size)
        w, h = target_image.size

        # Baseline 1: Pure Noise
        noise_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        noise_image = Image.fromarray(noise_array)

        # Baseline 2: Pure White Blank Canvas
        blank_array = np.full((h, w, 3), 255, dtype=np.uint8)
        blank_image = Image.fromarray(blank_array)

        # Initialize Evaluator
        evaluator = GenerativeEvaluator(info_print=log)

        # Run Tests
        log("\n--- Evaluating Target vs Itself (Perfect Baseline) ---")
        perfect_scores = evaluator.evaluate(target_image, target_image, ground_truth_text)
        log(json.dumps(perfect_scores))
        results_summary["runs"]["perfect_baseline"] = perfect_scores

        log("\n--- Evaluating Target vs Random Noise ---")
        noise_scores = evaluator.evaluate(target_image, noise_image, ground_truth_text)
        log(json.dumps(noise_scores))
        results_summary["runs"]["noise_baseline"] = noise_scores

        log("\n--- Evaluating Target vs Blank Canvas ---")
        blank_scores = evaluator.evaluate(target_image, blank_image, ground_truth_text)
        log(json.dumps(blank_scores))
        results_summary["runs"]["blank_baseline"] = blank_scores

        results_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        json_path = log_path.with_suffix(".json")
        json_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
        log(f"\nStructured summary: {json_path}")

    print(f"\nSaved evaluation log to {log_path}")

if __name__ == "__main__":
    run_checkpoint_baseline()