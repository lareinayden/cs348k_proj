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

### Experiment Results:

Evaluations run on the **40-image COCO subset** (`data/selected`) against each target. Metrics are aggregated across the subset (mean unless noted otherwise). Lower is better for **LPIPS** and **DreamSim**; higher is better for **CLIP-Score**.

| ID | Configuration | Text conditioning | Structural conditioning | CFG | ControlNet scale | LPIPS ↓ | CLIP-Score ↑ | DreamSim ↓ | Notes |
|:--:|---------------|---------------------|-------------------------|-----|------------------|:-------:|:------------:|:----------:|-------|
| 1 | Baseline Semantic | Sparse | — | | | | | | |
| 2 | Advanced Semantic | Dense | — | | | | | | |
| 3 | Baseline Structural A | Sparse / empty | Canny edge map | | | | | | |
| 4 | Baseline Structural B | Sparse / empty | Segmentation mask | | | | | | |
| 5 | Multi-modal A | Dense | Canny edge map | | | | | | |
| 6 | Multi-modal B | Dense | Segmentation mask | | | | | | |

*CFG and ControlNet scale columns: report the setting used per run (e.g., low CFG ≈ 3.0, high CFG ≈ 7.5–10.0; structural configs sweep ControlNet conditioning scale).*

### Progress on Project Goals:

#### What we have demonstrated so far

| Area | Status | Evidence |
|------|--------|----------|
| Evaluation pipeline (LPIPS, CLIP-Score, DreamSim) | Done (Checkpoint 1) | `evaluation.py`; sanity checks penalize noise and blank outputs vs. target |
| Test subset & conditioning data | Done (partial) | 40 images in `data/selected`; captions from COCO; Canny path via `run_controlnet_canny.py` |
| Perceptual efficacy (text vs. edge vs. mask) | *TBD* | |
| Multi-modal synergy (text + structure) | *TBD* | |
| Semantic vs. structural trade-offs (CFG / ControlNet scale) | *TBD* | |

- **Checkpoint 1:** We can score any generated image against a target and caption and distinguish failed generations from plausible ones.
- **Generative reconstruction:** *TBD — describe initial ControlNet / text runs, metric trends, and example figures.*
- **Comparison to baselines:** *TBD — e.g., trivial baselines vs. first structural or text-only reconstructions on the 40-image set.*

#### What is still open or not up to par

| Research question | Gap | Why it matters |
|-------------------|-----|----------------|
| **Perceptual efficacy** — which single modality wins? | *TBD* | Need scored runs for configs 1–4 (and agreed sparse/dense prompts) before ranking modalities. |
| **Multi-modal synergy** — does text + structure beat single modality? | *TBD* | Configs 5–6 not fully run or evaluated vs. 2–4. |
| **Semantic vs. structural trade-offs** | *TBD* | CFG and ControlNet scale sweeps not yet reported; may need plots vs. guidance strength. |
| Full coverage of six configurations on 40 images | *TBD* | Phase 2 may only cover a subset; table rows still empty. |
| Interactive refinement loop | Not started | Deferred to Phase 3 unless time allows. |

### Success Criteria:
We will know the experiment is successful when we can produce a robust comparative analysis cross-referencing our generated outputs against the original target images. Success does not mean "perfect" reconstructions; it means our evaluation pipeline can definitively measure and rank the efficacy of each modality setup.
- Checkpoint 1 Success: Our evaluation code is functional and can definitively identify failed generations (e.g., scoring empty pictures or white noise heavily negatively compared to the target).
- Checkpoint 2 Success: Generations on the test subset, scored and compared to baselines, with intermediate tables or plots and a clear summary of what is done vs. still open.
- Final Success: A comprehensive set of plots (Modality vs. LPIPS, Modality vs. CLIP-score) proving which input method best captures user intent.

## Implementation Roadmap

### Phase 1: Evaluation Pipeline MVP

- [x] Initialize LPIPS, CLIP-score, and DreamSim metric functions.

- [x] Build a trivial baseline testing script.

- [x] Verify the evaluation code correctly penalizes random noise and blank canvases against a target image.

- [x] Standardize the data loader to extract target images and their corresponding ground-truth text captions (e.g., COCO JSON parsing).

- [x] Selected 40 images from the COCO dataset with rich features and objects, clear canny edges, and detailed captions.

### Phase 2: Checkpoint 2 — Generative Pipeline & Intermediate Results

- [ ] Set up the Stable Diffusion 1.5 + ControlNet pipeline (Canny and additional modalities as needed).

- [ ] Implement extraction scripts for spatial modalities. Define sparse vs. dense text prompts for each image in the 40-image test subset.

- [ ] Run batch generation on the test subset for an initial subset of experimental configurations (not necessarily all six). Present intermediate results as tables.

- [ ] Summarize what is answered vs. still open relative to project goals.

### Phase 3: Generative Reconstruction & Final Analysis

- [ ] Run hyperparameter sweeps (CFG; ControlNet conditioning scale for structural configs). Batch-evaluate all reconstructions and complete the experiment results table.

- [ ] Generate final comparative plots (modality vs. LPIPS, CLIP-Score, DreamSim; CFG sweeps where applicable).

- [ ] Answer research questions on perceptual efficacy, multi-modal synergy, and semantic vs. structural trade-offs using the evaluation framework.

- [ ] Implement the interactive feedback loop: refine prompts or conditioning from metric feedback and report whether scores improve.

### Nice-to-Haves
- Extension to video reconstruction (if compute allows)
- Automated prompt refinement or suggestion system
- Visualization tools for comparing modality contributions
- User study to evaluate perceived controllability

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
├── data/selected                 # Selected images by the heuristic
├── download.py                   # Pulls COCO-2017 validation subset via FiftyOne
├── evaluation.py                 # Metric stack + trivial baselines
├── run_controlnet_canny.py       # Runs ControlNet pipeline with Canny edge map
├── select_data.py                # Heuristic selection of data from downloaded set
├── logs/                         # Log files for evaluation metrics
├── outputs/controlnet_canny      # Generated images by ControlNet
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
- Writes artifacts under `logs/`.

Run from the repo root so relative paths resolve:

```bash
python evaluation.py
```

### `select_data.py`

Heuristic selection of data from the downloaded set with rich features and objects, clear canny edges, and detailed captions.

### `run_controlnet_canny.py`

Tests image reconstruction using pretrained models from ControlNet using the modality of canny edge detection.


**Dependencies:** Reproduce the conda environment from the pinned spec at [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml   # once
conda activate cs348k
```

The file pins package builds for reproducibility. Otherwise, you need roughly: PyTorch and matching `torchvision`/`torchaudio`, `torchvision` (COCO loader), `lpips`, `torchmetrics` (CLIP-Score), `transformers` + Hugging Face CLIP weights (safetensors when available), and `dreamsim`. First CLIP/DreamSim runs may download weights; ensure PyTorch and `torchvision` versions are paired to avoid native-op errors.


## Risks and Mitigation
**Difficulty controlling pretrained model outputs**
Use constrained inputs (e.g., layouts) and iterative refinement

**Ambiguity in defining “user intent”**
Use reconstruction tasks with known targets as a proxy

**Evaluation metrics may not fully capture quality**
Combine perceptual and semantic metrics with visual inspection

**High computational cost**
Limit resolution or number of samples; reuse cached results when possible
