# Evaluating Modality Efficacy for Intent Expression in Generative Image Reconstruction

## Team
Sophia Huang (sophiacc)  
Yixiao Zhang (yixiaoz)

## Summary
We are going to build a visual computing system that evaluates how different conditioning modalities such as text prompts and spatial layouts express user intent during generative image reconstruction. Rather than focusing on training new models, our goal is to design a framework that measures how effectively various inputs guide a pretrained generative model to reconstruct a target image.

We will demonstrate success by performing a “generative reconstruction” task on a standard image dataset, where the system attempts to recreate a given target image using different forms of user input. Our approach includes an interactive feedback loop that allows iterative refinement of inputs and outputs. By the end of the project, we will provide a comparative analysis of modality efficacy using perceptual and semantic metrics, identifying which input modalities best capture and communicate user intent in generative systems.

## Inputs and Outputs

### Inputs
- Target images from a standard dataset (e.g., FFHQ or similar)
- Conditioning modalities:
    - Text prompts (semantic descriptions)
    - Spatial layouts (masks, bounding boxes, or structure maps)
    - Combined multi-modal inputs
- Pretrained generative image models (e.g., diffusion-based models)

### Outputs
- Reconstructed images under different conditioning modalities
- Quantitative evaluation metrics:
    - LPIPS (perceptual similarity)
    - CLIP-score (semantic alignment)
    - Pixel-level reconstruction error (optional)
- Comparative analysis plots:
    - Modality vs reconstruction accuracy
    - Single-modality vs multi-modal performance
- Visual comparisons (target vs reconstructions)

### Constraints
- Limited control over pretrained model internals
- Ambiguity in mapping user intent to model inputs
- Computational cost of iterative sampling and evaluation
- Tradeoff between perceptual realism and reconstruction fidelity

## Task List

### Core Tasks
#### 1. Baseline Setup
- Select a pretrained generative image model
- Build an end-to-end pipeline for image generation
- Verify correct reconstruction behavior from basic inputs

#### 2. Conditioning Interface Design
- Implement different input modalities:
    - Text-based prompts
    - Spatial layouts (e.g., masks or structure constraints)
    - Multi-modal combinations
- Standardize input formats for fair comparison

#### 3. Generative Reconstruction Pipeline
- Define a reconstruction task:
    - Given a target image, generate inputs that attempt to reproduce it
- Run reconstruction under different modality settings
- Ensure consistent sampling and evaluation across experiments

#### 4. Interactive Feedback Loop
- Design a loop where:
    - Outputs are evaluated against the target
    - Inputs are iteratively refined (e.g., prompt adjustment, layout updates)
- Analyze how feedback improves reconstruction quality

#### 5. Evaluation Framework
- Compute perceptual and semantic metrics:
    - LPIPS for visual similarity
    - CLIP-score for semantic alignment
- Aggregate results across modalities and datasets
- Generate comparison plots and visual summaries

### Nice-to-Haves
- Extension to video reconstruction (if compute allows)
- Automated prompt refinement or suggestion system
- Visualization tools for comparing modality contributions
- User study to evaluate perceived controllability

## Expected Deliverables and Evaluation

### Deliverables
- Side-by-side visual comparisons of reconstruction results
- Plots comparing modality performance:
    - LPIPS vs modality
    - CLIP-score vs modality
- Analysis of single vs multi-modal conditioning
- Demonstration of the interactive feedback system

### Evaluation Questions
- Which modality best captures user intent for reconstruction tasks?
- How do spatial constraints compare to semantic prompts?
- Does combining modalities significantly improve performance?
- How does iterative feedback affect reconstruction quality?

### Success Criteria
- Clear differences observed between conditioning modalities
- Demonstrated improvement using multi-modal inputs
- Effective use of feedback loop to refine outputs
- Insightful analysis of modality strengths and limitations

## Risks and Mitigation
**Difficulty controlling pretrained model outputs**
Use constrained inputs (e.g., layouts) and iterative refinement

**Ambiguity in defining “user intent”**
Use reconstruction tasks with known targets as a proxy

**Evaluation metrics may not fully capture quality**
Combine perceptual and semantic metrics with visual inspection

**High computational cost**
Limit resolution or number of samples; reuse cached results when possible
