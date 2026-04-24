# Literature Review: How similar are MLP hidden states to each other?

## Research Area Overview

The central question — *how similar are the hidden representations at
different layers of a multi-layer perceptron?* — sits at the intersection of
three active research threads:

1. **Representation-similarity methodology** (how to even measure similarity
   between two n×p activation matrices when the two layers may have
   different widths and the notion of "similarity" is ill-defined).
2. **Empirical cross-layer studies of deep networks** (what the heatmaps
   actually look like for real trained networks: CNNs, ResNets, transformers,
   and MLPs).
3. **Theoretical analysis of feature learning in MLPs** (why representations
   should or should not differ across layers, from neural-tangent-kernel
   theory to fixed-point analyses of deep nonlinear networks).

The literature consistently reports a recurring finding: **layer
representations are neither identical nor totally different — there is
measurable structure.** Most notably a **block structure** emerges in
over-parameterized networks, where large contiguous runs of layers have
CKA ≈ 1, while the first few and last few layers remain distinguishable.
The research hypothesis in this project is a direct restatement of this
finding, applied specifically to pure MLPs.

---

## Key Papers

### Paper 1: CKA — Similarity of Neural Network Representations Revisited
- **Authors**: Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
- **Year**: 2019 · ICML · arXiv:1905.00414 · 2000+ citations
- **Key contribution**: Introduces **Centered Kernel Alignment (CKA)**. Proves
  that any similarity index invariant to invertible linear transformation
  becomes useless when representation width exceeds dataset size (Theorem 1)
  — ruling out CCA for wide modern networks. CKA is invariant only to
  orthogonal transformation and isotropic scaling.
- **Formula**: `CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))`
  where `K = XX^T`, `L = YY^T` for centered activations.
- **Key findings** relevant to our hypothesis:
  - CKA reliably identifies corresponding layers across different random
    initializations and architectures.
  - **Wider networks learn more similar representations**; similarity of
    earlier layers saturates faster than later layers.
  - CIFAR-10 and CIFAR-100-trained networks share highly similar **early**
    representations but diverge in later layers.
  - For 8× depth networks, CKA exposes pathological plateaus (adjacent
    layers with CKA ≈ 1).
- **Code**: Reference in `google-research/similarity/`; PyTorch port in
  `code/PyTorch-Model-Compare/` and `code/anatome/`.

### Paper 2: SVCCA
- **Authors**: Raghu, Gilmer, Yosinski, Sohl-Dickstein · NeurIPS 2017
- **Year**: 2017 · arXiv:1706.05806 · 800+ citations
- **Key contribution**: **Singular Vector Canonical Correlation Analysis**.
  Treats each neuron as a vector over the dataset, layers as subspaces. SVD
  prunes low-variance directions, then CCA aligns subspaces.
- **Key findings**:
  - Learned intrinsic dimensionality is **much smaller** than layer width —
    an MLP with 200 neurons per layer may use only 10–30 effective
    directions.
  - Networks converge **bottom-up**: lower layers solidify first, later
    layers keep evolving.
  - SVCCA subspaces are distributed across many neurons, validating that
    neuron-aligned analyses miss structure.
- **Limitation**: CCA-style measures fail for wide networks (Kornblith 2019
  Theorem 1); use only with per-layer SVD truncation.

### Paper 3: Do Wide and Deep Networks Learn the Same Things?
- **Authors**: Nguyen, Raghu, Kornblith · ICLR 2021
- **Year**: 2020/2021 · arXiv:2010.15327 · 300+ citations
- **Key contribution**: Identifies the **block structure phenomenon** and
  introduces a **minibatch CKA estimator** for large networks.
- **Key findings directly relevant to our hypothesis**:
  - Over-parameterized ResNets develop **contiguous blocks of layers with
    CKA ≈ 1**; representations outside the blocks are similar across models
    of different sizes but block-interior representations are
    **model-specific** (seed-dependent).
  - Block structure appears when model capacity is **large relative to the
    training set**. Narrow/shallow networks do not show it.
  - Inside a block, a **single dominant principal component** explains most
    of the variance and is preserved across layers.
  - Some block-structure layers can be pruned with no accuracy loss.
- **Implication for MLPs**: the hypothesis that "MLP hidden states are not
  totally different" is empirically consistent with this — MLPs with large
  capacity should exhibit block structure; narrower/shallower ones should
  not.

### Paper 4: On the Origins of Block Structure
- **Authors**: Nguyen, Raghu, Kornblith · 2022 · arXiv:2202.07184
- **Key contribution**: Explains *why* the block structure arises.
- **Key findings**:
  - The dominant first PC inside each block is driven by a **small set of
    "dominant datapoints"** sharing trivial image statistics (background
    color, brightness).
  - Dominant datapoints produce high activation norms in block layers.
  - Different seeds pick different dominant datapoints — explaining
    block-structure inconsistency across runs.
  - Blocks can be **eliminated** via PC-regularization, transfer learning,
    or Shake-Shake regularization without hurting accuracy.

### Paper 5: Vision Transformers See Differently from CNNs (ViT vs CNN)
- **Authors**: Raghu et al. 2021 · arXiv:2108.08810 · 1300+ citations
- **Key findings**:
  - ViTs show **much more uniform CKA across layers** than CNNs — i.e. the
    transformer's MLP+attention stacks give more similar hidden states
    between early and late layers.
  - Skip connections drive this uniformity; removing them makes CKA look
    CNN-like.
- **Implication**: for transformer-style networks whose core is stacked
  MLPs with residual streams, the MLP hidden states should be
  systematically more similar than in plain MLPs.

### Paper 6: Tracing Representation Progression
- **Authors**: Jiang, Zhou, Zhu · arXiv:2406.14479 · 22+ citations
- **Key findings for transformers**:
  - **Simple sample-wise cosine similarity tracks CKA** and is much cheaper
    to compute. Per-sample similarity between layers increases as layers
    get closer.
  - Proves this is a consequence of the geodesic-curve property of
    residual networks.
  - Because of this increasing similarity, the last-layer classifier can be
    directly applied to any hidden layer ("logit lens"), producing the
    **saturation events** of Geva et al. — a token's top prediction often
    stabilizes several layers before the end.

### Paper 7: Self-similarity Analysis in Deep Neural Networks
- **Authors**: Ding et al. · 2025 · arXiv:2507.17785
- **Key findings relevant to MLPs**:
  - Constructs a graph `G_M` from per-layer hidden features, quantifies
    **hierarchical self-similarity** with the `SS_rate` metric.
  - Finds that **MLPs, CNNs, and attention models exhibit different
    degrees of self-similarity** — the degree varies with architecture.
  - Self-similarity generally **degrades during training**.
  - An `SS_rate` regularizer improves MLP and attention model accuracy by
    up to 6%.
- Most directly on-topic paper for the project's hypothesis.

### Paper 8: One Wide Feedforward Is All You Need
- **Authors**: Pessoa Pires et al. · arXiv:2309.01826 · 20+ citations
- **Key findings**:
  - Transformer FFNs across layers are **highly redundant**. Sharing a
    single FFN across all encoder layers and **removing** decoder FFNs
    preserves accuracy within 1 BLEU.
  - Internal CKA stays stable when FFNs are shared — direct empirical
    support that per-layer MLPs are doing similar work.

### Paper 9: Learning richness modulates equality reasoning in MLPs
- **Authors**: Tong & Pehlevan · 2025 · arXiv:2503.09781
- **Key contribution**: Mathematical theory of MLP feature learning on
  same-different tasks.
- **Relevant findings**:
  - **Rich-regime** MLPs learn conceptual, task-specific representations
    that differ substantially across layers.
  - **Lazy-regime** MLPs (small initialization scale, NTK behavior) keep
    representations close to their random initialization — layers remain
    statistically similar because none of them transform features much.
- **Direct implication**: the hypothesis ("limited individual transformation
  capacity causes MLP hidden states to be similar") corresponds to the
  lazy regime, whereas richly-trained MLPs can produce more distinct
  per-layer representations.

### Paper 10: Emergence of Globally Attracting Fixed Points in Deep NNs
- **Authors**: arXiv:2410.20107 · 2024
- **Key finding**: Under broad assumptions, deep nonlinear networks drive
  hidden states toward a **globally attracting fixed point** in feature
  space. This provides a formal mechanism for MLP hidden states to
  converge in similarity as depth grows.

### Paper 11: Inter-layer Information Similarity (NNTS/NNTP)
- **Authors**: Hryniowski, Wong · 2020 · arXiv:2012.03793
- **Key contribution**: Nearest-Neighbour Topological Similarity and
  Persistence. Builds a local k-NN graph per layer, compares graphs across
  layers.
- **Relevant finding**: In LeNet-5 on MNIST, atomic operations **within the
  same layer** produce neighbourhoods more similar to each other than to
  operations outside the layer — quantitatively validates the conventional
  layer partitioning.

### Paper 12: Reliability of CKA
- **Authors**: Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky · 2022 ·
  arXiv:2210.16156
- **Important caveat**: Proves CKA is highly sensitive to translations of
  small subsets (outliers) and linear-separability-preserving transformations.
  Shows CKA values can be moved up or down **without changing the network
  function**. Use CKA together with alternative metrics (cosine, NSA,
  Procrustes) to avoid being fooled.

### Paper 13: Normalized Space Alignment
- **Authors**: Ebadulla, Gulati, Singh · 2024 · arXiv:2411.04512
- **Key contribution**: NSA = Local NSA + Global NSA. Distance-preserving,
  differentiable, handles different dimensions, quadratic in #points.
- **Relevant use**: serves as a robustness check on CKA conclusions and can
  be used as a **loss function** if we want to train MLPs with controlled
  inter-layer structural similarity.

### Paper 14: Understanding Inner Workings of LMs via Representation
Dissimilarity
- **Authors**: Brown, Godfrey, Konz, Tu, Kvinge · 2023 · arXiv:2310.14993
- **Relevant findings**:
  - **Block structure appears in generative Transformers** (Pythia family
    70M–1B) — not just CNNs.
  - Model stitching reveals asymmetries between GeLU and SoLU
    representations that CKA alone cannot detect.
  - One model, pythia-2.8b-deduped, has anomalously low inter-model
    similarity despite normal benchmark performance.

### Paper 15: Batch Normalization Orthogonalizes Representations
- **Authors**: 2021 · arXiv:2106.03970
- **Relevant finding**: BN pushes covariance of hidden states toward
  isotropy, which can **raise** CKA between layers independently of
  learned features. Choice of normalization matters for reproducibility.

### Paper 16: MLP-Mixer & Understanding MLP-Mixer
- **Original**: Tolstikhin et al. 2021 · arXiv:2105.01601
- **Understanding**: arXiv:2306.01470
- **Relevance**: MLP-Mixer is a strong modern example of deep MLP stacks.
  Understanding it as a "wide sparse MLP" lets us import theoretical tools
  (permutation-equivariance, block structure) when testing our hypothesis.

### Other papers (reviewed via abstract only)
- **SVCCA for LMs** (Saphra & Lopez 2018): SVCCA across training time.
- **MPruner** (2024): uses CKA between consecutive layers to prune — shows
  redundancy directly.
- **LAYA** (2025): aggregates all intermediate layers for prediction,
  ablation signal for inter-layer information redundancy.
- **Curvature-based Comparison** (Yu, Long, Hopcroft 2018): Riemann and
  sectional curvature on fully-connected-layer activation manifolds.
- **Subspace Clustering Based Analysis** (2021): cluster neurons per layer,
  measure between-layer subspace overlap.
- **Graph-Based Similarity** (2021): graph representation for per-layer
  activations.

---

## Common Methodologies

- **CKA (linear and RBF)**: the default choice in every recent paper. Use
  the minibatch estimator (Nguyen et al. 2021) for large networks.
- **SVCCA / PWCCA**: still useful for narrow networks or when you want
  direction-level analysis.
- **Cosine similarity (sample-wise)**: for residual architectures where
  representation dimension stays constant, cosine captures what CKA does
  at a fraction of the cost (Jiang et al. 2024).
- **Model stitching**: measures functional similarity rather than
  statistical similarity — a strong complement to CKA.
- **NSA, NNTS/NNTP**: emerging alternatives that respect local structure
  better and sidestep CKA's outlier sensitivity.
- **PC / variance-explained analysis**: diagnose block structure by
  checking if a single PC dominates each layer.

## Standard Baselines

Any paper studying cross-layer similarity in MLPs should report at minimum:
- **Random init MLP** (untrained) — Kornblith 2019 Fig 7 shows untrained
  networks have very different CKA profiles from trained ones.
- **Linear classifier probes** per layer — gives a task-performance handle
  on representation quality (Alain & Bengio 2016 style).
- **Different widths/depths** of the same MLP family — to reproduce the
  block-structure onset and width-saturation effects.
- **Different random seeds** — block-interior representations are
  seed-specific; narrow/shallow ones are seed-stable.

## Evaluation Metrics

- **Layer × layer CKA heatmap** (primary visualization).
- **Variance explained by first PC per layer** (diagnoses block structure
  mechanism).
- **Sample-wise cosine similarity between layer L and layer L+k**.
- **k-NN overlap (NNTS / LNS)**.
- **CKA to initialization** across training — reveals lazy vs. rich regime.
- **Linear probe accuracy** at each layer.

## Datasets in the Literature

- **CIFAR-10, CIFAR-100** — universal benchmarks for CKA/SVCCA/block studies
  (Kornblith 2019; Nguyen 2020, 2022; Raghu 2021).
- **MNIST / FashionMNIST** — used by SVCCA, Inter-layer Information
  Similarity for LeNet-scale analyses.
- **ImageNet** — Nguyen 2020 for large-scale confirmation.
- **Patch Camelyon (medical imaging)** — Nguyen 2022 for domain-shift
  confirmation of block structure.
- **Pile / WMT** — for LM-focused analyses.
- **Synthetic regression** — SVCCA's toy task; Equality-Reasoning-in-MLPs'
  same-different task.

## Gaps and Opportunities

1. **Pure MLP focus is under-served.** The empirical literature almost
   exclusively studies CNNs, ResNets, and transformers. Block-structure and
   layer-similarity results for **pure MLPs** on standard datasets are not
   extensively tabulated. This project can fill that gap cleanly.
2. **Rich vs. lazy regime × layer similarity** is theoretically predicted
   (Tong & Pehlevan 2025) but not empirically mapped across architectures.
3. **Cross-metric consistency for MLPs**: given the CKA reliability
   critique (Davari 2022), confirming findings with CKA + NSA + cosine +
   Procrustes + NNTS on the same MLPs would strengthen any conclusions.
4. **Role of normalization & width** in MLP similarity: theorists have shown
   BN orthogonalizes (Daneshmand 2021), but empirical heatmaps for
   wide-shallow vs. narrow-deep MLPs are sparse.

## Recommendations for Our Experiment

Based on the literature review, the most informative and reproducible
experiment looks like:

- **Recommended primary dataset**: CIFAR-10 (and optionally MNIST for a
  quick second task). The test split (10k examples) provides the probe
  set. Directly comparable to Kornblith 2019, Nguyen 2020, 2022.
- **Recommended architecture family**: a single MLP "class" parametrized
  by (depth, width) — e.g. inputs → [width]*depth → softmax, with ReLU or
  GeLU nonlinearities. Scan depth ∈ {4, 8, 16, 32} and width ∈ {64, 256,
  1024, 4096}. This mirrors the exact "wide vs. deep" sweep of Nguyen 2020
  but restricted to MLPs.
- **Recommended baselines**:
  - Same MLP at random initialization (shows training-induced similarity).
  - Same MLP with different seeds (measures seed variability → relates to
    block-structure dominant-datapoint phenomenon).
  - Linear-probe accuracy per layer (task-functional signal).
- **Recommended metrics** (in order of priority):
  1. **Linear CKA** (minibatch estimator; batch size 256) — primary.
  2. **Sample-wise cosine similarity** — cheap confirmation.
  3. **Variance explained by 1st PC per layer** — diagnoses block structure.
  4. **NSA** or **Procrustes** — CKA-reliability safety net.
  5. Optional: **NNTS** for local-structure perspective.
- **Methodological considerations**:
  - Use the same probe set across all layer comparisons.
  - Normalize inputs consistently (Kornblith 2019 conventions).
  - Track CKA across **training time** (not just at convergence) to
    distinguish representation change from initialization artifact.
  - Be explicit about whether you are in the rich or lazy regime (track
    ||θ_final - θ_init|| relative to ||θ_init||; Chizat & Bach 2019).
  - Report multiple metrics — do not rely on linear CKA alone.

### Expected findings (from literature) that the experiments should
confirm or refute

- As MLP depth grows, we should see **block-like structure** emerge —
  consecutive hidden layers becoming highly similar under CKA — especially
  when width is large relative to the dataset or when training is very
  deep.
- Early layers should be more similar across different random seeds than
  middle/late layers.
- Representations should become **more similar between different MLPs as
  width grows** (Kornblith 2019 Fig 6 for CNNs → plausibly replicated for
  MLPs).
- There should be a **non-trivial distance from the last layer back
  toward the input**, confirming that MLP hidden states are not
  all-identical.
- Rich-regime training should produce more distinct cross-layer
  representations than lazy-regime training.

These together answer the original hypothesis: the hidden spaces of MLPs
at different layers are **not totally different** (there is broad
similarity, growing with width/depth/capacity) **but there are measurable
differences** (early vs. late layers diverge, rich regimes separate
layers, low-variance directions are not preserved).
