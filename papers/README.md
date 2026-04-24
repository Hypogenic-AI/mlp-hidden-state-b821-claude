# Downloaded Papers

Papers gathered for the research project "How similar are MLP hidden states to
each other?". 23 PDFs were downloaded covering foundational
representation-similarity methods, cross-layer empirical studies, MLP-specific
theory, and recent work on transformer MLPs.

Files marked with **★** were read in depth (multiple chunks). Others were
skimmed via chunk 1 (title, abstract, intro) or from their paper-finder
abstracts.

## Foundational representation-similarity methods

1. **★ CKA_Kornblith_2019.pdf** — *Similarity of Neural Network
   Representations Revisited* (Kornblith, Norouzi, Lee, Hinton; ICML 2019,
   arXiv:1905.00414; 2000+ cites).
   Introduces **Centered Kernel Alignment (CKA)**, the de-facto standard for
   cross-layer representation similarity. Shows CKA identifies corresponding
   layers across initializations and architectures, whereas CCA fails for
   wide networks. Key empirical findings: (a) wider networks have more similar
   representations; (b) early-layer similarity saturates at fewer channels;
   (c) 8× depth ResNet produces a pathological block at late layers where
   CKA ≈ 1 across many layers.

2. **★ SVCCA_Raghu_2017.pdf** — *SVCCA: Singular Vector Canonical Correlation
   Analysis* (Raghu, Gilmer, Yosinski, Sohl-Dickstein; NeurIPS 2017,
   arXiv:1706.05806; 800+ cites).
   Foundational method combining SVD + CCA. Treats neurons as vectors over a
   dataset and layers as subspaces. Finds that networks converge **bottom-up**
   during training and that learned intrinsic dimensionality is far smaller
   than layer width.

3. **SVCCA_LM_Saphra_2018.pdf** — *Understanding Learning Dynamics of
   Language Models with SVCCA* (Saphra & Lopez; arXiv:1811.00225;
   Edinburgh, 100+ cites). Applies SVCCA to RNN LMs across time; shows
   subnetworks for POS/syntactic features appear early, semantic features
   emerge later.

4. **CKA_Reliability_2022.pdf** — *Reliability of CKA as a Similarity
   Measure in Deep Learning* (Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky;
   ICLR 2023, arXiv:2210.16156). **Critique**: proves CKA is highly sensitive
   to subset translations (outliers) and linear-separability-preserving
   transformations. CKA values can be manipulated up or down without
   changing model function.

5. **NormalizedSpaceAlign_2024.pdf** — *Normalized Space Alignment: A
   Versatile Metric* (Ebadulla, Gulati, Singh; arXiv:2411.04512). Proposes
   NSA (Local + Global), a distance-preserving, differentiable alternative
   to CKA that can be used as a loss. Handles different dimensions via
   point-to-point correspondence. Quadratic complexity vs. CKA/RTD.

6. **GraphBasedSim_2021.pdf** — *Graph-Based Similarity of Neural Network
   Representations* (arXiv:2111.11165).

## Cross-layer empirical studies (directly on-topic)

7. **★ WideDeep_Nguyen_2020.pdf** — *Do Wide and Deep Networks Learn the
   Same Things?* (Nguyen, Raghu, Kornblith; ICLR 2021, arXiv:2010.15327;
   300+ cites). Identifies the **block structure phenomenon**: over-capacity
   networks contain contiguous blocks of highly-similar layers. Blocks arise
   from a single dominant principal component. Representations **outside**
   blocks are similar across architectures; block representations are
   unique to each model.

8. **★ BlockStructure_Origins_2022.pdf** — *On the Origins of the Block
   Structure Phenomenon* (Nguyen, Raghu, Kornblith; arXiv:2202.07184).
   Follow-up: the block-structure PC is driven by a **small set of
   "dominant datapoints"** sharing simple image statistics (e.g. background
   color). Different seeds pick different dominant sets, explaining
   seed-to-seed block variability. Block structure can be removed by PC
   regularization or excluding dominant datapoints.

9. **ViT_vs_CNN_Raghu_2021.pdf** — *Do Vision Transformers See Like CNNs?*
   (Raghu et al. 2021, arXiv:2108.08810; 1300+ cites). Uses CKA to compare
   ViT and ResNet. ViTs have more uniform representations across layers;
   early ViT layers attend both locally and globally; skip connections are
   critical for the CKA-visible structure.

10. **SelfSimilarity_2025.pdf** — *Self-similarity Analysis in Deep Neural
    Networks* (Ding et al.; arXiv:2507.17785). Builds complex-network
    graphs from hidden-layer features and measures fractal-style
    self-similarity across MLP, CNN, and attention models. Finds that
    intrinsic self-similarity **degrades** during training; adding an
    SS_rate regularizer improves MLP and attention models by up to 6%.
    Directly relevant — MLPs are one of the three architectures studied.

11. **InterLayerInfoSim_Hu_2020.pdf** — *Inter-layer Information Similarity
    Assessment via Topological Similarity and Persistence*
    (Hryniowski & Wong; arXiv:2012.03793). Introduces **NNTS** (Nearest
    Neighbour Topological Similarity) and **NNTP** (persistence). Shows
    sequences of atomic ops within a "layer" are more mutually similar
    than across-layer ops — validates layer boundaries via representation.

12. **TracingRepProgression_2024.pdf** — *Tracing Representation Progression*
    (Jiang, Zhou, Zhu; arXiv:2406.14479). For transformers, simple
    **sample-wise cosine similarity** aligns with CKA. Representations in
    transformers are positively correlated across layers, similarity grows
    with proximity. Justifies the "logit lens" and enables a single-classifier
    multi-exit architecture.

13. **InnerWorkingsLM_RepDiss_2023.pdf** — *Understanding the Inner Workings
    of Language Models Through Representation Dissimilarity*
    (Brown, Godfrey, Konz, Tu, Kvinge; arXiv:2310.14993). Applies CKA and
    model-stitching to Pythia LMs; finds block structure in generative
    transformers of 70M–1B parameters; pythia-2.8b-deduped has anomalous
    low similarity with the rest of the family.

14. **DeepNetsPaths_not_downloaded.md** — *Deep Networks as Paths on the
    Manifold of Neural Representations* (Doimo et al. 2023). Could not be
    located on arXiv at search time; the ICLR version should be obtainable
    later if needed.

## MLP-specific theory & analysis

15. **★ EqualityReasoningMLP_2025.pdf** — *Learning richness modulates
    equality reasoning in neural networks* (Tong & Pehlevan;
    arXiv:2503.09781). Develops a **mathematical theory of MLPs on
    same-different tasks**. Rich-regime MLPs learn conceptual
    task-specific representations; lazy-regime MLPs stay near their NTK
    initialization. Strongly informs our MLP question: transformation
    capacity per layer depends on the rich/lazy regime.

16. **MLPMixer_Original_2021.pdf** — *MLP-Mixer: An all-MLP Architecture
    for Vision* (Tolstikhin et al.; arXiv:2105.01601). The architecture
    that brought MLP stacks back into mainstream vision.

17. **MLPMixer_Wide_Sparse_2023.pdf** — *Understanding MLP-Mixer as a Wide
    and Sparse MLP* (arXiv:2306.01470). Shows Mixer is equivalent to a
    wide sparse MLP, bringing theoretical tools for per-layer analysis.

18. **GloballyAttractingFP_2024.pdf** — *Emergence of Globally Attracting
    Fixed Points in Deep Neural Networks With Nonlinear Activations*
    (arXiv:2410.20107). Shows that under broad conditions, deep MLPs
    drive representations toward a **globally attracting fixed point**
    in feature space — a formal reason representations become more
    similar with depth.

19. **BatchNormOrtho_2021.pdf** — *Batch Normalization Orthogonalizes
    Representations in Deep Random Networks* (arXiv:2106.03970). Proves
    BN drives hidden-state covariances toward isotropy, which affects
    CKA values between layers. Relevant when picking normalization for
    MLP experiments.

20. **OneWideFFN_Pessoa_2023.pdf** — *One Wide Feedforward Is All You Need*
    (Pessoa Pires et al.; arXiv:2309.01826). Shows transformer FFNs are
    highly redundant across layers: sharing one FFN across all encoder
    layers and dropping decoder FFNs loses little accuracy. Direct
    motivation: per-layer MLPs **are** highly similar in function.

## Layer pruning / redundancy (corollary evidence)

21. **MPruner_CKA_2024.pdf** — *MPruner: CKA-based Mutual Information
    Pruning* (arXiv:2408.13482). Uses CKA similarity between consecutive
    layers to prune redundant ones.

22. **LAYA_2025.pdf** — *Layer-wise Attention Aggregation for Interpretable
    Depth-Aware Networks* (Vessio; arXiv:2511.12723). Aggregates all
    layers' representations for the final prediction; shows intermediate
    layers carry useful complementary information.

## Geometry / fine-grained analysis

23. **CurvatureCompare_Yu_2018.pdf** — *Curvature-based Comparison of Two
    Neural Networks* (Yu, Long, Hopcroft; arXiv:1801.06801). Uses Riemann
    and sectional curvature to compare the activation manifolds of fully
    connected layers in two trained MLPs.

24. **SubspaceClustering_2021.pdf** — *Subspace Clustering Based Analysis
    of Neural Networks* (arXiv:2107.01296). Clusters neurons into
    subspaces per layer; quantifies within/between-layer subspace overlap.

---

## Deep-read prioritization that was done

Chunks of the following were read fully:
- CKA_Kornblith_2019 (chunks 1 and 3)
- SVCCA_Raghu_2017 (chunk 1)
- WideDeep_Nguyen_2020 (chunk 1: intro + minibatch CKA formulation)
- BlockStructure_Origins_2022 (chunk 1: abstract + main findings)
- SelfSimilarity_2025 (chunk 1: method)
- InterLayerInfoSim_Hu_2020 (chunk 1)
- TracingRepProgression_2024 (chunk 1)
- OneWideFFN_Pessoa_2023 (chunk 1: methodology)
- InnerWorkingsLM_RepDiss_2023 (chunk 1)
- NormalizedSpaceAlign_2024 (chunk 1)
- CKA_Reliability_2022 (chunk 1)

Remaining papers were used via their abstracts (from paper-finder output)
and their page-1 chunks when stored in `pages/`.

## Chunks location

All papers have been chunked for incremental reading:
- Chunks: `papers/pages/*_chunk_NNN.pdf` (3 pages per chunk)
- Manifests: `papers/pages/*_manifest.txt`
