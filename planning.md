# Planning: How similar are MLP hidden states to each other?

## Motivation & Novelty Assessment

### Why This Research Matters
Representation-similarity analyses have been performed extensively for CNNs,
ResNets, ViTs, and (recently) Pythia-style LMs, but pure multi-layer
perceptrons are under-tabulated in the same vocabulary (CKA heatmaps, block
structure, width/depth sweeps). Understanding how similar MLP hidden
states are across layers bears directly on practical questions like
per-layer probing, layer pruning/sharing (MPruner, One-Wide-FFN), layer-wise
attention aggregation (LAYA), and theoretical questions like rich-vs-lazy
feature learning (Tong & Pehlevan 2025) and globally-attracting fixed
points (2024). The hypothesis in the user brief — "they can't be totally
different because there's not that much individual transformation
capacity, but... can we compare what differs?" — is exactly the regime the
representation-similarity literature has studied for CNNs but not cleanly
mapped for MLPs.

### Gap in Existing Work
The literature review (`literature_review.md`) identifies four concrete
gaps:
1. **Pure MLPs are under-tabulated.** Kornblith 2019, Nguyen 2020/2022,
   Raghu 2021 and the block-structure line of work almost exclusively
   study convolutional or transformer architectures. We lack a clean
   MLP-only depth/width sweep with modern metrics.
2. **Cross-metric consistency.** Davari et al. 2022 show linear CKA is
   manipulable by small-subset shifts. No paper has corroborated an MLP
   similarity story with CKA + NSA + cosine + Procrustes simultaneously.
3. **Rich vs. lazy regime separation.** Tong & Pehlevan 2025 predict that
   lazy-regime MLPs keep representations close to init — trivially similar
   — while rich-regime MLPs learn distinct per-layer features. This
   prediction has not been empirically tested on standard image tasks.
4. **Dominant-datapoint controls for MLPs.** Nguyen 2022's finding that
   block structure is driven by a small subset with trivial image
   statistics has not been checked for MLPs.

### Our Novel Contribution
We test the hypothesis that "MLP hidden states at different layers are
neither identical nor totally different, with measurable differences" by
producing the first (to our knowledge) **cross-metric, width × depth**
heatmap of MLP-only cross-layer similarity on CIFAR-10 and MNIST, coupled
with:
- a **rich vs. lazy regime diagnostic** (`‖θ_final - θ_init‖ / ‖θ_init‖`)
  per configuration, so we can separate "similar because close to init"
  from "similar despite learning";
- a **dominant-datapoint ablation** (remove top-norm 10 %) to test whether
  any observed block-like structure has the same cause as in CNNs;
- a **linear-probe curve** per layer as a task-functional reference for
  "what information did the representation change actually affect?".

### Experiment Justification
- **Experiment 1 — Depth/width sweep (CIFAR-10)**: map cross-layer CKA,
  cosine, and Procrustes as functions of MLP (depth, width). Necessary
  because the central question is *how* similar layers are and whether
  that depends on capacity. Directly tests the main hypothesis.
- **Experiment 2 — Multi-metric cross-checking**: for each depth/width,
  compute CKA, cosine, Procrustes, and PC-1 variance. Necessary because
  CKA can be fooled by outliers (Davari 2022) and cosine captures
  different invariances. Addresses the robustness question.
- **Experiment 3 — Rich vs. lazy regime**: track
  `‖θ_final - θ_init‖ / ‖θ_init‖` and CKA-to-init at each depth/width.
  Necessary to separate "layers are similar because none of them learned"
  from "layers are similar despite learning". Directly addresses the
  "limited transformation capacity" phrase in the hypothesis.
- **Experiment 4 — Dominant-datapoint ablation**: recompute CKA after
  dropping the top-decile high-norm probe points. Necessary to test the
  Nguyen-2022 mechanism as the driver of any block-like structure in
  MLPs.
- **Experiment 5 — Linear-probe accuracy per layer**: gives a
  task-functional readout per layer. Necessary to contextualize the
  similarity numbers — if two layers have CKA ≈ 1 but different probe
  accuracies, then CKA is missing meaningful functional differences.
- **Experiment 6 — MNIST replication**: a cheap second dataset to test
  whether the headline findings are task-specific or robust.

---

## Research Question
Are the hidden spaces of MLPs at different layers "totally different", or
are there measurable cross-layer similarities and differences, and if so,
how do they depend on MLP width/depth, training regime (rich vs. lazy),
and metric choice?

## Background and Motivation
Transformation capacity per layer in an MLP is bounded by width and
nonlinearity — a single affine+ReLU layer cannot permute an arbitrary
input representation into any arbitrary output representation. This
bounded per-layer capacity suggests representations at nearby layers
cannot be "totally different", yet the literature also shows
early/middle/late layers serve distinct roles. We want the empirical
heatmap — and its dependence on width/depth/regime — to settle *how*
similar.

## Hypothesis Decomposition
- **H1 (primary)**: For trained MLPs on CIFAR-10, the cross-layer CKA
  heatmap exhibits measurable structure: neither uniformly 1 nor
  uniformly low. Specifically, adjacent-layer CKA is significantly higher
  than far-apart-layer CKA.
- **H2 (width amplifies similarity)**: Wider MLPs have higher
  off-diagonal CKA (more similar representations) than narrow MLPs at
  equal depth.
- **H3 (depth induces block-like structure)**: Deeper MLPs exhibit
  contiguous layer ranges with CKA ≳ 0.9, resembling block structure;
  narrow/shallow MLPs do not.
- **H4 (metric agreement)**: Linear CKA, sample-wise cosine similarity,
  and orthogonal Procrustes broadly agree on the ranking of layer
  similarity — if they disagree, CKA is the outlier (per Davari 2022).
- **H5 (rich vs. lazy)**: Configurations with small
  `‖θ_final - θ_init‖ / ‖θ_init‖` (lazy) have higher inter-layer
  similarity than rich-regime configurations of equal capacity.
- **H6 (dominant datapoints)**: If a configuration shows block-like
  structure, dropping the top-decile high-norm probe samples reduces the
  apparent CKA in those blocks (replicates Nguyen 2022 for MLPs).

Independent variables: depth, width, nonlinearity (ReLU first, optional
GeLU ablation), init scale (controls rich/lazy), dataset (CIFAR-10,
MNIST). Dependent variables: cross-layer CKA heatmap, cosine sample
similarity, Procrustes distance, PC-1 variance ratio per layer, linear
probe accuracy per layer, `‖Δθ‖/‖θ_0‖`.

## Proposed Methodology

### Approach
Train a family of ReLU MLPs with systematic (depth, width) variation on
CIFAR-10 (primary) and MNIST (secondary). After training, extract
post-activation hidden states on a held-out 5k-sample probe set for every
hidden layer. Compute a suite of representation-similarity metrics and
visualize layer × layer heatmaps.

### Experimental Steps
1. **Data pipeline**: download CIFAR-10 and MNIST via torchvision; flatten
   to vectors; normalize with per-pixel mean/std from train set; fixed
   5k-sample probe set from the test split.
2. **Model family**: `MLP(depths, width)` with `input → Linear → ReLU`
   repeated `depth` times → `Linear → num_classes`. We keep **constant
   width** across hidden layers to match the "layer-similarity" framing
   (varying-width MLPs would introduce free parameters in the metrics).
3. **Training protocol**: Adam, lr=1e-3, batch=256, 40 epochs CIFAR /
   20 MNIST, weight-decay 5e-4, cosine LR schedule, random seed fixed,
   3 seeds per configuration for variability. Early-stop on val loss
   patience=6. Save final model and checkpoints at init+best.
4. **Extraction**: in eval mode, dropout off, forward probe set,
   concatenate per-layer post-ReLU activations.
5. **Metrics**:
   - **Linear CKA** (Kornblith 2019 formula; full-batch on 5k probe).
   - **Sample-wise cosine similarity** (mean cosine between rows of L_i
     and L_j after centering; Jiang 2024).
   - **Orthogonal Procrustes distance** (after per-layer width
     projection via PCA to min(d_i, d_j) dims — see robustness).
   - **PC-1 variance ratio** per layer (PCA on 5k activations).
   - **Linear probe accuracy** per layer (ℓ2-regularized logistic
     regression with 5-fold on probe set).
   - **CKA to initialization**: use the initial weights' forward pass as
     the "init state" for regime analysis.
   - **‖θ_final − θ_init‖ / ‖θ_init‖** per parameter block and averaged.
6. **Baselines**: same MLPs at **random init** (untrained); **shuffled-
   label control** (trained on random labels — gives rich-regime without
   task structure); **different seeds** (variability).
7. **Ablations**: rich-vs-lazy by varying init scale α ∈ {0.25, 1.0, 4.0}
   on one representative (depth=8, width=512); dominant-datapoint
   ablation on the same configuration.

### Baselines
- Untrained-random-init MLP (same architecture).
- Multi-seed replicates (3 seeds) of the same configuration.
- Shuffled-label trained MLP (rules out "task itself drives similarity").

### Evaluation Metrics
Primary: Linear CKA layer × layer matrix; diagonal = 1 by construction.
Secondary: Procrustes distance, sample-wise cosine similarity,
PC-1 variance ratio. Task-functional: linear probe accuracy per layer.
Regime: `‖θ_final − θ_init‖ / ‖θ_init‖`.

### Statistical Analysis Plan
- All similarity heatmaps reported as mean ± std over 3 seeds.
- Test H1 with paired t-test comparing adjacent-layer CKA vs. far-layer
  (k=1 vs. k≥3) CKA, Holm-Bonferroni over (depth, width).
- Test H2 (width → more similarity) via Spearman ρ between width and
  off-diagonal CKA at fixed depth.
- Test H4 (metric agreement) via Spearman correlation between CKA and
  cosine / Procrustes rankings of layer pairs.
- Significance α = 0.05 per-family, with multiple-comparison correction.

## Expected Outcomes
Based on the literature:
- Some measurable structure: adjacent-layer CKA ≳ 0.9, distant-layer
  CKA ≲ 0.6. (Supports hypothesis.)
- Width amplifies similarity: wider MLPs have flatter heatmaps.
- Very deep narrow MLPs (depth≥16, width≤256) show less block-like
  structure than wide ones.
- Metrics agree in **rank** but differ in **absolute values**.
- Lazy regime has uniformly high CKA (trivially); rich regime separates.

## Timeline and Milestones
- Env + data download: 10 min.
- Coding MLP + training loop + metrics: 30 min.
- Smoke test (depth=4, width=256, 2 epochs): 5 min.
- Full sweep CIFAR-10 (3 depths × 3 widths × 3 seeds = 27 runs × 40
  epochs ≈ 45 min on A6000). MNIST similar but faster.
- Rich-vs-lazy + dominant-point ablation: 15 min.
- Analysis + plots: 30 min.
- REPORT.md + README.md: 25 min.

Total budget: ~2.5 hours of compute, plus analysis/writing.

## Potential Challenges
- **GPU sharing**: 4× A6000 visible but first one has 15 GB used. We'll
  default to CUDA:1.
- **MLP overfitting CIFAR-10**: raw-pixel MLPs hit ~55 % test accuracy;
  that's fine for representation analysis, but we should not
  over-interpret task accuracy.
- **CKA numerical instability for near-zero activations**: we'll clip
  activations to avoid division-by-near-zero in the HSIC denominator.
- **Probe set size vs. CKA reliability**: 5k samples follows Kornblith
  2019; Davari 2022 notes small subsets manipulate CKA — we use the
  whole probe set for every comparison (no resampling).
- **Init-scale sensitivity**: rich/lazy is knife-edge. We pick init
  scales that demonstrably straddle the transition.

## Success Criteria
- Clear, readable layer × layer heatmap figures for each
  (depth, width) on CIFAR-10 + MNIST.
- Quantitative evidence supporting or refuting each of H1–H6.
- Cross-metric robustness check (at least CKA + cosine + Procrustes).
- Reproducible code with seeded runs in `src/`, `results/`, `figures/`.
- REPORT.md with actual numbers, figures, and clear answer to research
  question.

## Preregistered Decisions (to avoid p-hacking)
- Widths fixed up-front: {128, 512, 2048}.
- Depths fixed up-front: {4, 8, 16}.
- Seeds fixed up-front: {0, 1, 2}.
- Probe set: first 5 000 examples of test split, fixed order.
- Primary metric: linear CKA; all hypothesis tests use it.
- Secondary metrics are reported regardless of whether they agree.
