# Comparative Evaluation of Conditioning Modalities for Target Scene Reconstruction

## Team
Sophia Huang (sophiacc)  
Yixiao Zhang (yixiaoz)

## Summary

Modern diffusion models such as ControlNet support a variety of conditioning modalities, including text prompts, edge maps, and semantic segmentation masks. While these modalities are widely available, it remains unclear which forms of conditioning provide the most useful **semantic and structural scene information** for guiding image generation. Understanding the strengths and limitations of each modality can help practitioners choose more effective conditioning strategies and avoid unnecessary inputs during generation.

We study this through a **generative reconstruction** task: given a target scene, we derive different conditioning signals (text, Canny edges, segmentation masks, and combinations) and measure how well a frozen Stable Diffusion + ControlNet pipeline recreates that target. We ask how well each modality preserves semantic and structural information **relevant to that target**, scored with perceptual and semantic metrics (LPIPS, CLIP-Score, DreamSim).

## Problem & Research Questions

### Problem

Practitioners can supply text, edges, masks, or combinations to ControlNet-style models, but there is no clear guidance on which inputs carry the most reconstructive signal for a given scene. We compare modalities under a controlled reconstruction setup on a fixed COCO subset.

### Core question

**Which modality provides the strongest conditioning signal for reconstructing a target scene?**

### Supporting questions

- **Semantic vs. structural signal:** Which single modality—sparse text, dense text, Canny edges, or segmentation masks—best preserves target semantics (CLIP-Score) and layout/perception (LPIPS, DreamSim)?
- **Multi-modal synergy:** Does combining text with a structural modality improve reconstruction beyond either alone?
- **Conditioning strength:** How do CFG and ControlNet conditioning scale affect the trade-off between following structure and preserving realistic texture?


## System Architecture

### Inputs
Target dataset:
- **COCO** (40-image subset in `data/selected`): diverse multi-object scenes with captions, edges, and masks derived from each target.

Conditioning modalities (extracted from or paired with the target):
- **Text:** sparse entity lists (COCO categories) or **dense** captions (`Dense Caption` column in `selected_coco_candidates.csv`)
- **Canny edge maps:** high-frequency structure from the target
- **Semantic segmentation masks:** region layout from the target (UperNet → ControlNet-seg)

Generative core:
- **Stable Diffusion 1.5 + ControlNet** (weights frozen): documented, controllable baseline given our compute budget.

### Outputs
Qualitative Analysis:
- Reconstructed images for each modality configuration.
- Visual comparisons (target vs reconstructions).

Quantitative Evaluation Metrics (vs. target scene):
- **LPIPS:** perceptual similarity to the target image
- **CLIP-Score:** semantic alignment with the dense caption (proxy for scene semantics)
- **DreamSim:** mid-level layout and perceptual similarity to the target


## Experimental Design & Success Criteria

We hold the **target image** fixed and vary only the conditioning modality. Each configuration produces a reconstruction; metrics measure how much semantic and structural information from the target scene is preserved.

### Configurations

Text tiers (semantic conditioning):
- **Sparse text:** comma-separated COCO category names (entity list, minimal spatial wording)
- **Dense text:** enriched captions from the `Dense Caption` column in `data/selected/selected_coco_candidates.csv`

Six reconstruction configurations on the 40-image subset:
1. **Baseline Semantic:** sparse text only
2. **Advanced Semantic:** dense text only
3. **Baseline Structural A:** empty text + Canny edge map
4. **Baseline Structural B:** empty text + semantic segmentation mask
5. **Multi-modal A:** dense text + Canny edge map
6. **Multi-modal B:** dense text + semantic segmentation mask

### Hyperparameter sweeps (conditioning strength)

We sweep **Classifier-Free Guidance (CFG)** (e.g., 3.0 vs. 7.5–10.0) and **ControlNet conditioning scale** for structural configs to see how strongly the model follows each signal vs. its prior.

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

### Success Criteria

Success means ranking **which conditioning modality strongest reconstructs the target scene** under our metric framework—not achieving perfect pixel-level copies.

- **Checkpoint 1:** Evaluation pipeline distinguishes failed generations (noise, blank) from plausible reconstructions.
- **Checkpoint 2:** Batch reconstructions on the test subset, scored vs. targets, with tables and a progress summary.
- **Final:** Completed experiment table and modality-vs-metric plots identifying the strongest semantic and structural conditioning signals; limitations of target-as-proxy intent stated clearly.

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

- [ ] Answer which modality provides the strongest conditioning signal (semantic vs. structural vs. combined) using the evaluation framework.

- [ ] Document limitations of target-as-proxy intent and what a future human study would add.

### Nice-to-Haves
- Human study linking conditioning choice to perceived user intent (beyond target-as-proxy)
- Automated prompt refinement from metric feedback
- Visualization tools comparing modality contributions
- Extension to additional ControlNet modalities or newer backbones

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

Batch ControlNet-Canny on `data/selected`. Dense generation uses the **`Dense Caption`** column; CLIP-Score always uses that dense caption as the semantic reference for the target scene.

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

Text-only **Stable Diffusion 1.5** (no ControlNet) on the 40-image subset. Sparse prompts are COCO category lists (`sparse_prompts.py`); dense prompts use the **`Dense Caption`** column. CLIP-Score uses the dense caption as the semantic reference for the target scene.

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
Use fixed checkpoints, documented CFG / ControlNet scales, and per-image metric logging.

**Proxy intent vs. real user intent**  
We use the target image and dense caption as stand-ins for “what should be preserved”; conclusions are about reconstructive conditioning signal, not live user preferences. A human study would be needed to validate intent in practice.

**Evaluation metrics may not fully capture scene fidelity**  
Combine LPIPS, DreamSim, and CLIP-Score with qualitative target-vs-generated comparisons.

**High computational cost**  
Fixed 40-image subset; cache generations and sweep hyperparameters on a subset first.
