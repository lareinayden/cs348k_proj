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
| 1 | Baseline Semantic: Sparse Text | Sparse | — | 7.5 | — | 0.754 | 24.09 | 0.662 | 40-image mean; `logs/baseline_semantic_sparse_results.csv` |
| 2 | Advanced Semantic: Dense Text | Dense | — | 7.5 | — | 0.752 | 31.30 | 0.544 | 40-image mean; `logs/baseline_semantic_dense_results.csv` |
| 3 | Baseline Structural A | Empty | Canny edge map | 7.5 | 1.0 | 0.561 | 25.45 | 0.459 | 40-image mean; `logs/baseline_structural_canny_results.csv` |
| 4 | Baseline Structural B | Sparse / empty | Segmentation mask | | | | | | |
| 5 | Multi-modal A | Dense | Canny edge map | 7.5 | 1.0 | 0.533 | 30.62 | 0.357 | 40-image mean; see `logs/controlnet_canny_results.csv` |
| 6 | Multi-modal B | Dense | Segmentation mask | | | | | | |

*CFG and ControlNet scale columns: report the setting used per run (e.g., low CFG ≈ 3.0, high CFG ≈ 7.5–10.0; structural configs sweep ControlNet conditioning scale).*

**Cross-config takeaways:**
- **Structure vs. text alone:** Empty + Canny (3) lowers LPIPS **~0.75 → 0.56** and DreamSim **~0.60 → 0.46** vs. both text-only configs.
- **Multi-modal synergy (5 vs. 3):** Adding dense text to Canny improves CLIP **25.5 → 30.6** and perceptual metrics modestly (LPIPS **0.561 → 0.533**, DreamSim **0.459 → 0.357**). Gains are not only from edges—caption helps semantics and refinement.
- **Best overall so far:** Config **5** on all three metrics among completed runs.

**Checkpoint 1 reference** (single image; `logs/evaluation_1.json`): target vs. itself — LPIPS 0.00, DreamSim 0.00; vs. noise — 0.89 / 0.90; vs. blank — 0.84 / 0.93. All generative configs beat trivial failure baselines on CLIP-Score.

### Progress on Project Goals:

#### What we have demonstrated so far

| Area | Status | Evidence |
|------|--------|----------|
| Evaluation pipeline (LPIPS, CLIP-Score, DreamSim) | Done | `scripts/evaluation.py`; sanity checks in `logs/evaluation_1.json` |
| Test subset & conditioning data | Done | 40 images in `data/selected`; Canny under `outputs/baseline_structural_canny/` and `outputs/controlnet_canny/` |
| Text-only SD (Configs 1–2) | Done | `scripts/run_sd_text_batch.py`; `logs/baseline_semantic_*_results.csv` |
| ControlNet-Canny (Configs 3 & 5) | Done | Empty + dense text variants; `logs/baseline_structural_canny_results.csv`, `logs/controlnet_canny_results.csv` |
| Perceptual efficacy (text vs. edge vs. mask) | Partial | Canny path scored (3, 5); **segmentation mask** (4, 6) not run |
| Multi-modal synergy (text + structure) | Partial | Config 5 > Config 3 on all metrics; structure-alone (3) already beats text-only on LPIPS/DreamSim |

- **Text-only (Configs 1 vs. 2):** Dense captions raise CLIP **24.1 → 31.3** and lower DreamSim **0.66 → 0.54**; LPIPS ~0.75 for both.
- **Structure-only (Config 3):** Empty prompt + Canny reaches LPIPS **0.56**, DreamSim **0.46**, CLIP **25.5** — large perceptual gain over text-only, but CLIP stays near sparse-text levels (limited semantic guidance).
- **Multi-modal (Config 5 vs. 3):** Dense caption + Canny improves CLIP **+5.2** and perceptual metrics vs. empty + Canny; confirms **synergy** beyond structure alone, though most layout gains come from Canny vs. text-only.

#### What is still open or not up to par

| Research question | Gap | Comment |
|-------------------|-----|----------------|
| **Perceptual efficacy** — mask vs. Canny vs. text | Configs 4 & 6 not run | Cannot compare segmentation vs. edge structure |
| **Semantic vs. structural trade-offs** | No hyperparameter sweep | CFG and ControlNet scale fixed at 7.5 / 1.0 |
| Sparse text + Canny (variant of 3) | Not run separately | Config 3 used **empty** prompt; sparse entity list + Canny optional ablation |
| Segmentation mask path (configs 4, 6) | Not started | No mask extraction or ControlNet-seg runs |
| Interactive refinement loop | Not started | Phase 3 |

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

- [x] Set up the Stable Diffusion 1.5 + ControlNet pipeline (Canny and additional modalities as needed).

- [x] Implement extraction scripts for spatial modalities (Canny edge maps per image). Sparse vs. dense text runs complete (`run_sd_text_batch.py`).

- [x] Run batch generation on the test subset for an initial subset of experimental configurations. Present intermediate results as tables (`logs`, experiment table above).

- [x] Summarize what is answered vs. still open relative to project goals ([Progress on Project Goals](#progress-on-project-goals)).

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
├── scripts/
│   ├── evaluation.py             # Metric stack + trivial baselines
│   ├── sparse_prompts.py         # COCO category lists for sparse prompts
│   ├── run_sd_text_batch.py      # Configs 1–2: text-only SD 1.5 (--modality sparse|dense)
│   └── run_controlnet_canny_batch.py  # Config 3 (empty+Canny) & Config 5 (dense+Canny)
├── download.py                   # Pulls COCO-2017 validation subset via FiftyOne
├── select_data.py                # Heuristic selection of data from downloaded set
├── logs/
│   ├── evaluation_*.json         # Checkpoint 1 sanity runs
│   ├── baseline_semantic_sparse_results.csv   # Config 1
│   ├── baseline_semantic_dense_results.csv    # Config 2
│   ├── baseline_structural_canny_results.csv  # Config 3
│   └── controlnet_canny_results.csv   # Config 5 (40 rows)
├── outputs/baseline_semantic_sparse/  # Config 1 generations
├── outputs/baseline_semantic_dense/   # Config 2 generations
├── outputs/baseline_structural_canny/  # Config 3: empty text + Canny
├── outputs/controlnet_canny/     # Config 5: dense text + Canny
└── models/                       # Local pretrained weights (CLIP/DINO/DreamSim-related assets, etc.)
```


## Scripts

### `download.py`

Loads a small COCO-2017 **validation** split through [FiftyOne](https://voxel51.com/docs/fiftyone/) (`fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=100)`). Use this if you need to (re)populate `data/coco/`; it requires a working FiftyOne install and sufficient disk space for the zoo download.

### `scripts/evaluation.py`

Baseline **evaluation pipeline** for the three metrics above:

- Loads the first COCO validation sample (image + captions) via `torchvision.datasets.CocoCaptions`.
- Uses the **first caption** as ground-truth text for CLIP-Score.
- Runs three checks: target vs. itself (sanity / “perfect” baseline), target vs. **random noise**, and target vs. **blank white** image.
- Writes artifacts under `logs/`.

```bash
python scripts/evaluation.py
```

### `select_data.py`

Heuristic selection of data from the downloaded set with rich features and objects, clear canny edges, and detailed captions.

### `scripts/run_controlnet_canny_batch.py`

Batch ControlNet-Canny on `data/selected`. CLIP-Score always uses the dense ground-truth caption; only the **generation prompt** changes.

| Flag | Experiment |
|------|------------|
| `--modality empty` | **Config 3** — empty text + Canny (`outputs/baseline_structural_canny/`, `logs/baseline_structural_canny_results.csv`) |
| `--modality dense` | **Config 5** — dense caption + Canny (default; `outputs/controlnet_canny/`) |

```bash
# Config 3 — Baseline Structural A (empty text + Canny)
python scripts/run_controlnet_canny_batch.py --modality empty

# Config 5 — Multi-modal A (dense caption + Canny)
python scripts/run_controlnet_canny_batch.py --modality dense
```

### `scripts/run_sd_text_batch.py`

Text-only **Stable Diffusion 1.5** (no ControlNet) on the 40-image subset. Sparse prompts are comma-separated **COCO category names** from `instances_val2017.json` (see `sparse_prompts.py`). CLIP-Score always uses the dense ground-truth caption for evaluation.

| Flag | Experiment |
|------|------------|
| `--modality sparse` | **Config 1** — Baseline Semantic |
| `--modality dense` | **Config 2** — Advanced Semantic |

```bash
# Config 1 — Baseline Semantic (sparse entity list)
python scripts/run_sd_text_batch.py --modality sparse

# Config 2 — Advanced Semantic (full COCO caption)
python scripts/run_sd_text_batch.py --modality dense
```

Outputs: `outputs/baseline_semantic_{sparse,dense}/<coco_id>/` and `logs/baseline_semantic_{sparse,dense}_results.csv`. Optional: `--max-images`, `--guidance-scale 7.5`, `--seed 0` (CUDA).


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
