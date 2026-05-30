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

Evaluations run on the **200-image COCO subset** (`data/selected_200`) against each target. Metrics are aggregated across the subset (mean unless noted otherwise). Lower is better for **LPIPS** and **DreamSim**; higher is better for **CLIP-Score**.

| ID | Configuration | Text conditioning | Structural conditioning | CFG | ControlNet scale | LPIPS ↓ | CLIP-Score ↑ | DreamSim ↓ | Notes |
|:--:|---------------|---------------------|-------------------------|-----|------------------|:-------:|:------------:|:----------:|-------|
| 1 | Baseline Semantic: Sparse Text | Sparse | — | 7.5 | — | 0.763 | 24.10 | 0.687 | 200-image mean; `logs/baseline_semantic_sparse_results.csv` |
| 2 | Advanced Semantic: Dense Text | Dense | — | 7.5 | — | 0.733 | 33.82 | 0.501 | 200-image mean; `logs/baseline_semantic_dense_results.csv` |
| 3 | Baseline Structural A | Empty | Canny edge map | 7.5 | 1.0 | 0.545 | 28.15 | 0.432 | 200-image mean; `logs/baseline_structural_canny_results.csv` |
| 4 | Baseline Structural B | Empty | Segmentation mask | 7.5 | 1.0 | 0.671 | 23.86 | 0.583 | 200-image mean; `logs/baseline_structural_seg_results.csv` |
| 5 | Multi-modal A | Dense | Canny edge map | 7.5 | 1.0 | 0.517 | 34.44 | 0.328 | 200-image mean; `logs/controlnet_canny_results.csv` |
| 6 | Multi-modal B | Dense | Segmentation mask | 7.5 | 1.0 | 0.641 | 34.28 | 0.414 | 200-image mean; `logs/controlnet_seg_results.csv` |

*All-CFG means (CFG 3 / 5 / 7.5 / 10, ControlNet scale 1.0 for structural configs): Config 1 — LPIPS 0.758, CLIP 24.0, DreamSim 0.683; Config 2 — 0.732, 33.5, 0.500; Config 3 — 0.545, 28.1, 0.442; Config 4 — 0.672, 23.9, 0.582; Config 5 — 0.519, 34.1, 0.332; Config 6 — 0.642, 34.2, 0.417.*

*CFG and ControlNet scale columns: report the setting used per run (e.g., low CFG ≈ 3.0, high CFG ≈ 7.5–10.0; structural configs sweep ControlNet conditioning scale).*

**Cross-config takeaways (CFG 7.5, 200 images):**
- **Structure vs. text alone:** Empty + Canny (3) lowers LPIPS **~0.73 → 0.55** and DreamSim **~0.50 → 0.43** vs. dense text (2).
- **Canny vs. seg mask (3 vs. 4):** Canny wins on all mean metrics (LPIPS **0.545 vs. 0.671**, DreamSim **0.432 vs. 0.583**, CLIP **28.2 vs. 23.9**).
- **Multi-modal synergy (5 vs. 3):** Adding dense text to Canny improves CLIP **28.2 → 34.4** and perceptual metrics (LPIPS **0.545 → 0.517**, DreamSim **0.432 → 0.328**).
- **Multi-modal synergy (6 vs. 4):** Dense caption + seg mask improves all metrics vs. empty + seg (LPIPS **0.671 → 0.641**, DreamSim **0.583 → 0.414**, CLIP **23.9 → 34.3**).
- **Canny vs. seg multi-modal (5 vs. 6):** Dense + Canny wins on perceptual metrics (LPIPS **0.517 vs. 0.641**, DreamSim **0.328 vs. 0.414**); CLIP is comparable (**34.4 vs. 34.3**).
- **Best overall:** Config **5** on LPIPS and DreamSim; Config **5** and **6** tie on CLIP.

#### Modality win-rate comparisons (200-image subset, all CFG)

Per-image win-rate: for each image and metric (LPIPS ↓, DreamSim ↓, CLIP-Score ↑), the best config earns **1 point** (ties split 0.5). Totals are normalized over **3 metrics × 200 images = 600 points** max per config. Structural configs use **ControlNet scale 1.0**.

Reproduce: `python scripts/compare_modality_winrate.py --all-cfg` → `logs/modality_winrate_all_cfg.csv`

**Config 1 vs 2** (sparse vs dense text):

| CFG | Winner | Config 1 (sparse) | Config 2 (dense) |
|:---:|:------:|:-----------------:|:------------------:|
| 3.0 | **2** | 13.7% | **86.3%** |
| 5.0 | **2** | 11.0% | **89.0%** |
| 7.5 | **2** | 11.0% | **89.0%** |
| 10.0 | **2** | 11.2% | **88.8%** |

Dense text wins at every CFG; gains are mostly on CLIP and DreamSim.

**Config 2 vs 3 vs 4** (dense text vs empty + Canny vs empty + seg):

| CFG | Winner | Config 2 (dense) | Config 3 (Canny) | Config 4 (seg) |
|:---:|:------:|:----------------:|:----------------:|:--------------:|
| 3.0 | **3** | 36.3% | **60.5%** | 3.3% |
| 5.0 | **3** | 38.3% | **58.8%** | 2.8% |
| 7.5 | **3** | 38.7% | **59.3%** | 2.0% |
| 10.0 | **3** | 38.5% | **58.8%** | 2.8% |

Canny (3) dominates overall; dense text (2) wins CLIP on most images; seg alone (4) rarely wins.

**Config 2 vs 5 vs 6** (dense text vs dense + Canny vs dense + seg):

| CFG | Winner | Config 2 (dense) | Config 5 (dense + Canny) | Config 6 (dense + seg) |
|:---:|:------:|:----------------:|:------------------------:|:----------------------:|
| 3.0 | **5** | 10.0% | **73.5%** | 16.5% |
| 5.0 | **5** | 9.8% | **76.7%** | 13.5% |
| 7.5 | **5** | 10.7% | **74.8%** | 14.5% |
| 10.0 | **5** | 10.7% | **72.5%** | 16.8% |

Adding structure to dense text strongly favors **Config 5**; **Config 6** beats text-only (2) but trails Canny multi-modal (5) at all CFG values.

**Checkpoint 1 reference** (single image; `logs/evaluation_1.json`): target vs. itself — LPIPS 0.00, DreamSim 0.00; vs. noise — 0.89 / 0.90; vs. blank — 0.84 / 0.93. All generative configs beat trivial failure baselines on CLIP-Score.

### Progress on Project Goals:

#### What we have demonstrated so far

| Area | Status | Evidence |
|------|--------|----------|
| Evaluation pipeline (LPIPS, CLIP-Score, DreamSim) | Done | `scripts/evaluation.py`; sanity checks in `logs/evaluation_1.json` |
| Test subset & conditioning data | Done | 200 images in `data/selected_200`; outputs under `outputs/` |
| Text-only SD (Configs 1–2) | Done | `scripts/run_sd_text_batch.py`; `logs/baseline_semantic_*_results.csv` |
| ControlNet-Canny (Configs 3 & 5) | Done | Empty + dense text variants; `logs/baseline_structural_canny_results.csv`, `logs/controlnet_canny_results.csv` |
| ControlNet-Seg (Configs 4 & 6) | Done | Empty + dense text variants; `logs/baseline_structural_seg_results.csv`, `logs/controlnet_seg_results.csv` |
| Perceptual efficacy (text vs. edge vs. mask) | Done | Single-modality configs 2–4 scored; pairwise win-rate in `logs/modality_comparison_234_winrate.txt` |
| Multi-modal synergy (text + structure) | Done | Configs 5 & 6 vs. 3 & 4; dense text improves both Canny and seg paths |

- **Text-only (Configs 1 vs. 2):** Dense captions raise CLIP **24.1 → 33.8** and lower DreamSim **0.69 → 0.50** at CFG 7.5; LPIPS similar (~0.76 vs ~0.73).
- **Structure-only (Config 3):** Empty prompt + Canny reaches LPIPS **0.55**, DreamSim **0.43**, CLIP **28.2** — large perceptual gain over text-only, with moderate CLIP.
- **Multi-modal (Config 5 vs. 3):** Dense caption + Canny improves CLIP **28.2 → 34.4** and perceptual metrics vs. empty + Canny.
- **Multi-modal (Config 6 vs. 4):** Dense caption + seg mask raises CLIP **23.9 → 34.3** and lowers LPIPS/DreamSim vs. empty + seg.

#### What is still open or not up to par

| Research question | Gap | Comment |
|-------------------|-----|----------------|
| **Perceptual efficacy** — mask vs. Canny vs. text | Addressed | Configs 2–4 win-rate done; Canny (3) beats seg (4) on structure-only; dense text wins CLIP |
| **Semantic vs. structural trade-offs** | Done | Full CFG sweep (3–10) for all configs; see experiment table and win-rate section |
| Sparse text + Canny (variant of 3) | Not run separately | Config 3 used **empty** prompt; sparse entity list + Canny optional ablation |
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
│   ├── baseline_structural_seg_results.csv  # Config 4
│   ├── controlnet_canny_results.csv   # Config 5
│   └── controlnet_seg_results.csv     # Config 6 (CFG sweep)
├── outputs/baseline_semantic_sparse/  # Config 1 generations
├── outputs/baseline_semantic_dense/   # Config 2 generations
├── outputs/baseline_structural_canny/  # Config 3: empty text + Canny
├── outputs/baseline_structural_seg/   # Config 4: empty text + seg mask
├── outputs/controlnet_canny/     # Config 5: dense text + Canny
├── outputs/controlnet_seg/       # Config 6: dense text + seg mask
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

### `scripts/run_controlnet_seg_batch.py`

Batch ControlNet-Seg on `data/selected`. Dense generation uses the **`Dense Caption`** column; CLIP-Score always uses that dense caption as the semantic reference for the target scene.

| Flag | Experiment |
|------|------------|
| `--modality empty` | **Config 4** — empty text + seg mask (`outputs/baseline_structural_seg/`, `logs/baseline_structural_seg_results.csv`) |
| `--modality dense` | **Config 6** — dense caption + seg mask (`outputs/controlnet_seg/`, `logs/controlnet_seg_results.csv`) |

```bash
# Config 4 — Baseline Structural B (empty text + seg mask)
python scripts/run_controlnet_seg_batch.py --modality empty

# Config 6 — Multi-modal B (dense caption + seg mask)
python scripts/run_controlnet_seg_batch.py --modality dense
```

Supports CFG sweep via `--guidance-scales 3.0 5.0 7.5 10.0`.

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

### `scripts/compare_modality_winrate.py`

Pairwise and multi-way **win-rate** comparison for **2 or 3** experiment configs (IDs 1–6). For each image and metric, the best config earns 1 point (ties split 0.5).

```bash
# Default: single-modality configs 2 vs 3 vs 4 at one CFG
python scripts/compare_modality_winrate.py --configs 2 3 4 --guidance-scale 7.5

# All comparison groups × all CFG values (tables in README)
python scripts/compare_modality_winrate.py --all-cfg

# Two-way example
python scripts/compare_modality_winrate.py --configs 5 6 --guidance-scale 10.0
```

Outputs: `logs/modality_comparison_<ids>_per_image.csv`, `logs/modality_comparison_<ids>_winrate.txt`, and `logs/modality_winrate_all_cfg.csv` with `--all-cfg`. Optional: `--controlnet-scale 1.0`.


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
