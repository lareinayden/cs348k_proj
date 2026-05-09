# Evaluating Modality Efficacy for Intent Expression in Generative Image Reconstruction

## Team
Sophia Huang (sophiacc)  
Yixiao Zhang (yixiaoz)

## Summary
We are going to build a visual computing system that evaluates how different conditioning modalities such as text prompts and spatial layouts express user intent during generative image reconstruction. Rather than focusing on training new models, our goal is to design a framework that measures how effectively various inputs guide a pretrained generative model to reconstruct a target image.

We will demonstrate success by performing a “generative reconstruction” task on a standard image dataset, where the system attempts to recreate a given target image using different forms of user input. Our approach includes an interactive feedback loop that allows iterative refinement of inputs and outputs. By the end of the project, we will provide a comparative analysis of modality efficacy using perceptual and semantic metrics, identifying which input modalities best capture and communicate user intent in generative systems.

## Research Questions & Goals
Our project aims to answer the following core questions:
- Perceptual Efficacy: Which single modality (Text Prompts, Canny Edge Maps, or Semantic Segmentation Masks) yields the highest perceptual similarity to the original target image?
- Multi-modal Synergy: Does combining a semantic modality (Text) with a structural modality (Edge/Mask) statistically improve reconstruction fidelity compared to using a single modality?
- Semantic vs. Structural Trade-offs: At what point do strict structural constraints degrade the generative model's ability to render perceptually realistic textures?

## Experimental Design & Success Criteria
To answer these questions, we will conduct a series of reconstruction experiments.

### The Experiment:
To isolate the efficacy of each modality, we categorize our text inputs into two tiers:
- Dense Text (Spatial): Highly descriptive captions explicitly detailing spatial relationships (e.g., "A yellow banana resting on top of a wooden table").
- Sparse Text (Semantic): Simplified entity lists stripping away spatial context (e.g., "A banana, a table").

For a sampled subset of target images, we will extract ground-truth conditioning data and attempt a generative reconstruction using the following configurations:
1. Baseline Semantic: Sparse Text Prompt.
2. Advanced Semantic: Dense Text Prompt.
3. Baseline Structural A: Empty/Sparse Text Prompt + Canny Edge Map.
4. Baseline Structural B: Empty/Sparse Text Prompt + Semantic Segmentation Mask.
5. Multi-modal A: Dense Text Prompt + Canny Edge Map.
6. Multi-modal B: Dense Text Prompt + Semantic Segmentation Mask (To observe if redundant spatial instructions degrade quality).

Hyperparameter Sweeps (Intent Guidance):

To measure how forcefully the model applies our inputs, we will evaluate the above configurations across varied scales of Classifier-Free Guidance (CFG). We will test outputs at a low CFG (e.g., 3.0 - allowing model prior to dominate) and a high CFG (e.g., 7.5 to 10.0 - forcing strict adherence to the input intent).

Note: We will similarly adjust the ControlNet Conditioning Scale when evaluating structural modalities to find the optimal balance between text intent and structural intent.

### Success Criteria:
We will know the experiment is successful when we can produce a robust comparative analysis cross-referencing our generated outputs against the original target images. Success does not mean "perfect" reconstructions; it means our evaluation pipeline can definitively measure and rank the efficacy of each modality setup.
- Checkpoint 1 Success: Our evaluation code is functional and can definitively identify failed generations (e.g., scoring empty pictures or white noise heavily negatively compared to the target).
- Final Success: A comprehensive set of plots (Modality vs. LPIPS, Modality vs. CLIP-score) proving which input method best captures user intent.

## System Architecture

### Inputs
Target Datasets:
- COCO: For evaluating complex, multi-object compositional intent and region-based masks.
- LAION (if resources allow): As it is a much larger dataset, we will incorporate it to evaluate broad, unconstrained generative capabilities across a massive variety of subjects and structural compositions if computational limits permit.

Conditioning Modalities:
- Ground-truth dataset captions (Text)
- Extracted Canny edge maps (High-frequency structure)
- Extracted Semantic segmentation masks (Region-based layout)

Generative Core: 
- Stable Diffusion 1.5 + ControlNet (Weights frozen): Given limited computational resources, we decided to start with this architecture. Although its performance is not the state of the art now, it provides a well-documented and controllable baseline.
- Advanced Models (if resources allow): We will also try evaluating on more advanced state-of-the-art models such as Flux, Hunyuan-image, and Qwen-image.

### Outputs
Qualitative Analysis:
- Reconstructed images for each modality configuration.
- Visual comparisons (target vs reconstructions).

Quantitative Evaluation Metrics:
- LPIPS: For measuring human perceptual similarity.
- CLIP-Score: For measuring semantic alignment with the ground-truth text.
- DreamSim: For measuring mid-level structural and perceptual similarity (e.g., spatial layout, object pose, and intent alignment) without being penalized by strict pixel-level deviations.

## Repository Layout

Top-level layout (paths are relative to the project root):

```
cs348k_proj/
├── data/coco/
│   ├── info.json                 # Metadata for the local COCO subset (e.g. FiftyOne export: split, sample count)
│   ├── raw/                      # Official COCO annotation JSON
│   │   ├── captions_val2017.json   # Used by the evaluation script for captions
│   │   ├── instances_*.json
│   │   └── person_keypoints_*.json
│   └── validation/
│       └── data/                 # Validation images aligned with COCO val IDs
├── download.py                   # Pulls COCO-2017 validation subset via FiftyOne
├── evaluation.py                 # Metric stack + trivial baselines
├── logs/                         # Log files for evaluation metrics
└── models/                       # Local pretrained weights (CLIP/DINO/DreamSim-related assets, etc.)
```


## Scripts

### `download.py`

Loads a small COCO-2017 **validation** split through [FiftyOne](https://voxel51.com/docs/fiftyone/) (`fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=100)`). Use this if you need to (re)populate `data/coco/`; it requires a working FiftyOne install and sufficient disk space for the zoo download.

### `evaluation.py`

Baseline **evaluation pipeline** for the three metrics above:

- Loads the first COCO validation sample (image + captions) via `torchvision.datasets.CocoCaptions`.
- Uses the **first caption** as ground-truth text for CLIP-Score.
- Runs three checks: target vs. itself (sanity / “perfect” baseline), target vs. **random noise**, and target vs. **blank white** image.
- Writes artifacts under `logs/`:
  - **`evaluation_N.log`** — console-style transcript.
  - **`evaluation_N.json`** — same run, structured (`started_at`, `finished_at`, `caption`, and per-baseline metric dicts).

Run from the repo root so relative paths resolve:

```bash
python evaluation.py
```

**Dependencies:** Reproduce the conda environment from the pinned spec at [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml   # once
conda activate cs348k
```

The file pins package builds for reproducibility. Otherwise, you need roughly: PyTorch and matching `torchvision`/`torchaudio`, `torchvision` (COCO loader), `lpips`, `torchmetrics` (CLIP-Score), `transformers` + Hugging Face CLIP weights (safetensors when available), and `dreamsim`. First CLIP/DreamSim runs may download weights; ensure PyTorch and `torchvision` versions are paired to avoid native-op errors.

## Implementation Roadmap

### Phase 1: Evaluation Pipeline MVP

- [x] Initialize LPIPS, CLIP-score, and DreamSim metric functions.

- [x] Build a trivial baseline testing script.

- [x] Verify the evaluation code correctly penalizes random noise and blank canvases against a target image.

- [x] Standardize the data loader to extract target images and their corresponding ground-truth text captions (e.g., COCO JSON parsing).

### Phase 2: Conditioning Interface & Generative Setup

- [ ] Set up the Stable Diffusion 1.5 + ControlNet pipeline.

- [ ] Implement extraction scripts for spatial modalities (Canny edge detection and Segmentation map generation from target images).

- [ ] Run initial generation passes using single modalities to verify the pipeline is connected.

### Phase 3: Generative Reconstruction & Analysis

- [ ] Execute the full suite of four experimental configurations across the dataset subsets.

- [ ] Implement the interactive feedback loop: evaluate outputs against targets and iteratively refine the conditioning inputs (e.g., prompt adjustment).

- [ ] Aggregate results and generate comparative plots (LPIPS vs. Modality, etc.).

### Nice-to-Haves
- Extension to video reconstruction (if compute allows)
- Automated prompt refinement or suggestion system
- Visualization tools for comparing modality contributions
- User study to evaluate perceived controllability

## Risks and Mitigation
**Difficulty controlling pretrained model outputs**
Use constrained inputs (e.g., layouts) and iterative refinement

**Ambiguity in defining “user intent”**
Use reconstruction tasks with known targets as a proxy

**Evaluation metrics may not fully capture quality**
Combine perceptual and semantic metrics with visual inspection

**High computational cost**
Limit resolution or number of samples; reuse cached results when possible
