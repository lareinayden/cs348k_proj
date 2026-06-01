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

Six reconstruction configurations on the 200-image subset, plus one **final triple-modal** run:

1. **Baseline Semantic:** sparse text only
2. **Advanced Semantic:** dense text only
3. **Baseline Structural A:** empty text + Canny edge map
4. **Baseline Structural B:** empty text + semantic segmentation mask
5. **Multi-modal A:** dense text + Canny edge map
6. **Multi-modal B:** dense text + semantic segmentation mask
7. **Multi-modal C (final):** dense text + Canny + seg (dual ControlNet); **CFG 5.0**, **ControlNet scale 1.5**

### Hyperparameter sweeps (conditioning strength)

We sweep **Classifier-Free Guidance (CFG)** (3.0 / 5.0 / 7.5 / 10.0) for text-conditioned configs **1, 2, 5, 6**, and **ControlNet conditioning scale** (0.5 / 1.0 / 1.5 / 2.0) for structural configs **3–6**. Configs 3–4 use an empty prompt, so CFG does not affect text guidance; ControlNet scale is the relevant knob for those runs.

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
| 7 | Multi-modal C (final) | Dense | Canny + seg mask | 5.0 | 1.5 | **0.487** | 32.46 | 0.347 | 200-image mean; best LPIPS overall; `logs/controlnet_multimodal_results.csv` |

*CFG sweep applies to text-conditioned configs **1, 2, 5, 6** only. Configs **3–4** use an empty prompt, so CFG has no meaningful text-guidance effect; we report them at **CFG 7.5, ControlNet scale 1.0** in the main table. **Config 7** uses sweep-tuned hyperparameters (**CFG 5.0**, **ControlNet scale 1.5** on both Canny and seg branches) rather than the main-table defaults. ControlNet scale sweep (0.5 / 1.0 / 1.5 / 2.0) at CFG 7.5 writes to `outputs/controlNetScale/` and `logs/controlnet_scale_*_results.csv`.*

#### CFG sweep — configs 1, 2, 5, 6 (200-image means)

Bold = best CFG **for that config** on each metric. Configs 3–4 omitted (empty text → CFG inert).

**LPIPS ↓** (lower is better)

| CFG | Config 1 | Config 2 | Config 5 | Config 6 |
|:---:|:--------:|:--------:|:--------:|:--------:|
| 3.0 | **0.745** | 0.728 | 0.518 | 0.641 |
| 5.0 | 0.754 | **0.727** | **0.513** | **0.640** |
| 7.5 | 0.763 | 0.733 | 0.517 | 0.641 |
| 10.0 | 0.768 | 0.738 | 0.528 | 0.647 |

**CLIP-Score ↑** (higher is better)

| CFG | Config 1 | Config 2 | Config 5 | Config 6 |
|:---:|:--------:|:--------:|:--------:|:--------:|
| 3.0 | 23.8 | 32.8 | 33.4 | 33.6 |
| 5.0 | 23.9 | 33.6 | 34.1 | 34.4 |
| 7.5 | 24.1 | 33.8 | 34.4 | 34.3 |
| 10.0 | **24.2** | **33.8** | **34.4** | **34.5** |

**DreamSim ↓** (lower is better)

| CFG | Config 1 | Config 2 | Config 5 | Config 6 |
|:---:|:--------:|:--------:|:--------:|:--------:|
| 3.0 | **0.675** | 0.503 | 0.340 | 0.421 |
| 5.0 | 0.682 | **0.497** | **0.326** | 0.415 |
| 7.5 | 0.687 | 0.501 | 0.328 | **0.414** |
| 10.0 | 0.686 | 0.500 | 0.335 | 0.419 |

**Best CFG per config (by metric):**

| Config | Best CFG — LPIPS ↓ | Best CFG — CLIP ↑ | Best CFG — DreamSim ↓ |
|:------:|:------------------:|:-----------------:|:---------------------:|
| 1 Sparse text | **3.0** (0.745) | **10.0** (24.2) | **3.0** (0.675) |
| 2 Dense text | **5.0** (0.727) | **10.0** (33.8) | **5.0** (0.497) |
| 5 Dense + Canny | **5.0** (0.513) | **7.5 / 10.0** (34.4) | **5.0** (0.326) |
| 6 Dense + seg | **5.0** (0.640) | **10.0** (34.5) | **7.5** (0.414) |

**CFG takeaways (configs 1, 2, 5, 6):**
- **Text-only (1–2):** Higher CFG improves CLIP but hurts perceptual metrics. Config 1 prefers **CFG 3.0** for LPIPS/DreamSim; Config 2 splits **CFG 5.0** (perceptual) vs **CFG 10.0** (CLIP).
- **Multi-modal (5–6):** **CFG 5.0** is best for LPIPS and DreamSim; **CFG 7.5–10.0** is best for CLIP. **CFG 7.5** is a practical compromise for Config 5 (near-best on all three metrics).
- **Overall:** No single CFG wins every metric. **CFG 5.0** favors perceptual reconstruction; **CFG 10.0** favors CLIP on dense-text runs (2, 5, 6).

#### ControlNet scale sweep — configs 3, 4, 5, 6 (CFG 7.5 fixed; 200-image means)

Bold = best ControlNet scale **for that config** on each metric. Source: `logs/controlnet_scale_*_results.csv`.

**LPIPS ↓** (lower is better)

| ControlNet scale | Config 3 | Config 4 | Config 5 | Config 6 |
|:----------------:|:--------:|:--------:|:--------:|:--------:|
| 0.5 | 0.642 | 0.719 | 0.610 | 0.685 |
| 1.0 | 0.545 | 0.670 | 0.519 | 0.639 |
| 1.5 | **0.521** | 0.659 | **0.500** | 0.629 |
| 2.0 | 0.523 | **0.656** | 0.504 | **0.624** |

**CLIP-Score ↑** (higher is better)

| ControlNet scale | Config 3 | Config 4 | Config 5 | Config 6 |
|:----------------:|:--------:|:--------:|:--------:|:--------:|
| 0.5 | 24.9 | 19.7 | **34.4** | 34.3 |
| 1.0 | 27.9 | 24.0 | 34.3 | 34.4 |
| 1.5 | **28.2** | **24.6** | 33.6 | **34.5** |
| 2.0 | 27.2 | 24.1 | 32.0 | 33.9 |

**DreamSim ↓** (lower is better)

| ControlNet scale | Config 3 | Config 4 | Config 5 | Config 6 |
|:----------------:|:--------:|:--------:|:--------:|:--------:|
| 0.5 | 0.549 | 0.673 | 0.386 | 0.448 |
| 1.0 | **0.435** | 0.582 | **0.329** | **0.412** |
| 1.5 | 0.440 | **0.568** | 0.337 | 0.415 |
| 2.0 | 0.480 | 0.575 | 0.389 | 0.417 |

**Best ControlNet scale per config (by metric):**

| Config | Best scale — LPIPS ↓ | Best scale — CLIP ↑ | Best scale — DreamSim ↓ |
|:------:|:--------------------:|:-------------------:|:-----------------------:|
| 3 Empty + Canny | **1.5** (0.521) | **1.5** (28.2) | **1.0** (0.435) |
| 4 Empty + seg | **2.0** (0.656) | **1.5** (24.6) | **1.5** (0.568) |
| 5 Dense + Canny | **1.5** (0.500) | **0.5** (34.4) | **1.0** (0.329) |
| 6 Dense + seg | **2.0** (0.624) | **1.5** (34.5) | **1.0** (0.412) |

**ControlNet scale takeaways (configs 3–6, CFG 7.5):**
- **Empty text (3–4):** ControlNet scale is the primary conditioning knob (unlike CFG, which is inert with empty prompts). **Scale 0.5** under-follows structure and hurts all metrics. Raising scale to **1.5–2.0** improves LPIPS; Config 3 peaks at **1.5**, Config 4 at **2.0** for perceptual metrics.
- **Multi-modal (5–6):** Stronger structure conditioning improves LPIPS (best at **1.5** for Config 5, **2.0** for Config 6) but can **reduce CLIP** at high scale — Config 5 CLIP drops **34.4 → 32.0** from scale 0.5 to 2.0. DreamSim is best at **scale 1.0** for both.
- **Trade-off:** Scale **1.0–1.5** is a practical default — near-best perceptual metrics without the CLIP penalty of scale 2.0 on dense-text runs. For structure-only Config 3, **1.5** balances all three metrics; Config 4 benefits from pushing toward **1.5–2.0** since text cannot compensate.

**Cross-config takeaways (CFG 7.5, ControlNet scale 1.0, 200 images):**
- **Structure vs. text alone:** Empty + Canny (3) lowers LPIPS **~0.73 → 0.55** and DreamSim **~0.50 → 0.43** vs. dense text (2).
- **Canny vs. seg mask (3 vs. 4):** Canny wins on all mean metrics (LPIPS **0.545 vs. 0.671**, DreamSim **0.432 vs. 0.583**, CLIP **28.2 vs. 23.9**).
- **Multi-modal synergy (5 vs. 3):** Adding dense text to Canny improves CLIP **28.2 → 34.4** and perceptual metrics (LPIPS **0.545 → 0.517**, DreamSim **0.432 → 0.328**).
- **Multi-modal synergy (6 vs. 4):** Dense caption + seg mask improves all metrics vs. empty + seg (LPIPS **0.671 → 0.641**, DreamSim **0.583 → 0.414**, CLIP **23.9 → 34.3**).
- **Canny vs. seg multi-modal (5 vs. 6):** Dense + Canny wins on perceptual metrics (LPIPS **0.517 vs. 0.641**, DreamSim **0.328 vs. 0.414**); CLIP is comparable (**34.4 vs. 34.3**).
- **Best overall (main table, CFG 7.5 / CN 1.0):** Config **5** on LPIPS and DreamSim; Config **5** and **6** tie on CLIP.
- **Final triple-modal (Config 7, CFG 5.0 / CN 1.5):** Combining dense text + Canny + seg yields the **best mean LPIPS (0.487)** across all configs — beating Config 5 at CFG 5.0 (**0.513**) and Config 5 at CFG 7.5 / CN 1.5 (**0.500**). DreamSim (**0.347**) sits between Config 5 (**0.326**) and Config 6 (**0.415**). CLIP (**32.5**) drops vs. dual single-structure runs (**~34.1–34.3**), consistent with the scale-sweep trade-off where stronger structural conditioning improves perceptual metrics at the cost of CLIP.

#### Config 7 — final triple-modal run (CFG 5.0, ControlNet scale 1.5)

Dense text + dual ControlNet (Canny + seg), using hyperparameters chosen from the CFG and ControlNet scale sweeps. Source: `logs/controlnet_multimodal_results.csv` (200 images).

| Metric | Config 5 (dense + Canny) | Config 6 (dense + seg) | Config 7 (dense + Canny + seg) |
|--------|:------------------------:|:------------------------:|:------------------------------:|
| CFG | 5.0 | 5.0 | 5.0 |
| ControlNet scale | 1.0 | 1.0 | **1.5** (both branches) |
| LPIPS ↓ | 0.513 | 0.640 | **0.487** |
| CLIP-Score ↑ | **34.06** | **34.35** | 32.46 |
| DreamSim ↓ | **0.326** | 0.415 | 0.347 |

**Config 7 takeaways:**
- **LPIPS:** Triple-modal wins — adding both structural signals together beats either branch alone at matched CFG.
- **DreamSim:** Config 7 beats Config 6 but trails Config 5; dual structure helps vs. seg-only multi-modal but Canny-only still has the edge on perceptual distance.
- **CLIP:** Both structural branches at scale 1.5 pull CLIP down ~1.5–2 points vs. Configs 5–6 at CN 1.0 — same perceptual-vs-semantic trade-off seen in the ControlNet scale sweep.
- **Win-rate (5 vs 6 vs 7, CFG 5.0):** Config **7** **43.3%**, Config **5** **41.2%**, Config **6** **15.5%** (600 points; CN 1.0 for 5/6, 1.5 for 7).

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

**Config 5 vs 6 vs 7** (dense + Canny vs dense + seg vs dense + Canny + seg; CFG 5.0):

| Config | ControlNet scale | LPIPS ↓ | CLIP ↑ | DreamSim ↓ | Win-rate |
|:------:|:----------------:|:-------:|:------:|:----------:|:--------:|
| 5 (dense + Canny) | 1.0 | 0.513 | **34.06** | **0.326** | 41.2% |
| 6 (dense + seg) | 1.0 | 0.640 | **34.35** | 0.415 | 15.5% |
| **7 (dense + Canny + seg)** | **1.5** | **0.487** | 32.46 | 0.347 | **43.3%** |

Config **7** edges Config **5** on overall win-rate and achieves the best mean LPIPS; Config **5** still wins on DreamSim and CLIP at CN 1.0.

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
| Triple-modal (Config 7) | Done | Dense + Canny + seg; CFG 5.0 / CN 1.5; `logs/controlnet_multimodal_results.csv` |
| Perceptual efficacy (text vs. edge vs. mask) | Done | Single-modality configs 2–4 scored; pairwise win-rate in `logs/modality_comparison_234_winrate.txt` |
| Multi-modal synergy (text + structure) | Done | Configs 5 & 6 vs. 3 & 4; Config 7 triple-modal final run |

- **Text-only (Configs 1 vs. 2):** Dense captions raise CLIP **24.1 → 33.8** and lower DreamSim **0.69 → 0.50** at CFG 7.5; LPIPS similar (~0.76 vs ~0.73).
- **Structure-only (Config 3):** Empty prompt + Canny reaches LPIPS **0.55**, DreamSim **0.43**, CLIP **28.2** — large perceptual gain over text-only, with moderate CLIP.
- **Multi-modal (Config 5 vs. 3):** Dense caption + Canny improves CLIP **28.2 → 34.4** and perceptual metrics vs. empty + Canny.
- **Multi-modal (Config 6 vs. 4):** Dense caption + seg mask raises CLIP **23.9 → 34.3** and lowers LPIPS/DreamSim vs. empty + seg.
- **Final triple-modal (Config 7):** Dense + Canny + seg at CFG **5.0** / CN **1.5** achieves best mean LPIPS (**0.487**) but CLIP falls to **32.5** vs. **~34** for Configs 5–6.

#### What is still open or not up to par

| Research question | Gap | Comment |
|-------------------|-----|----------------|
| **Perceptual efficacy** — mask vs. Canny vs. text | Addressed | Configs 2–4 win-rate done; Canny (3) beats seg (4) on structure-only; dense text wins CLIP |
| **Semantic vs. structural trade-offs** | Done | CFG sweep for configs 1, 2, 5, 6; ControlNet scale sweep for 3–6; see experiment table |
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
│   ├── controlnet_seg_results.csv     # Config 6 (CFG sweep)
│   ├── controlnet_multimodal_results.csv  # Config 7 (final)
│   └── controlnet_scale_*_results.csv  # ControlNet scale sweep (CFG 7.5)
├── outputs/baseline_semantic_sparse/  # Config 1 generations
├── outputs/baseline_semantic_dense/   # Config 2 generations
├── outputs/baseline_structural_canny/  # Config 3: empty text + Canny
├── outputs/baseline_structural_seg/   # Config 4: empty text + seg mask
├── outputs/controlnet_canny/     # Config 5: dense text + Canny
├── outputs/controlnet_seg/       # Config 6: dense text + seg mask
├── outputs/controlnet_multimodal/  # Config 7: dense + Canny + seg
├── outputs/controlNetScale/      # ControlNet scale sweep (CFG 7.5; configs 3–6)
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

# ControlNet scale sweep at CFG 7.5 (separate outputs/controlNetScale/ + logs/controlnet_scale_*_results.csv)
python scripts/run_controlnet_canny_batch.py --modality empty --guidance-scales 7.5 --controlnet-scales 0.5 1.0 1.5 2.0 --controlnet-scale-sweep
```

Pass `--controlnet-scale-sweep` to write scale-sweep runs under `outputs/controlNetScale/` with separate CSVs in `logs/` (e.g. `controlnet_scale_baseline_structural_canny_results.csv`). Main CFG sweep stays in the default output folders and result CSVs. Already-completed combos are skipped on re-run.

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

Supports CFG and ControlNet scale sweeps (same defaults as Canny script). Resume-safe: skips rows already in the results CSV.

### `scripts/run_controlnet_multimodal_batch.py`

**Config 7** — dense text + **dual ControlNet** (Canny + seg simultaneously). Uses sweep-tuned defaults: **CFG 5.0**, **ControlNet scale 1.5** on both structural branches. 200-image results: LPIPS **0.487**, CLIP **32.46**, DreamSim **0.347**.

```bash
python scripts/run_controlnet_multimodal_batch.py
python scripts/run_controlnet_multimodal_batch.py --guidance-scale 5.0 --controlnet-scale 1.5
```

Outputs: `outputs/controlnet_multimodal/cfg_5p0_control_1p5/<coco_id>/` and `logs/controlnet_multimodal_results.csv`.

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
