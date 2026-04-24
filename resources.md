# Resources Catalog

Consolidated inventory of all research resources gathered for the project
**"How similar are MLP hidden states to each other?"**.

## Summary

| Resource type  | Count | Location       | Total size |
|----------------|-------|----------------|------------|
| Papers (PDFs)  | 23    | `papers/`      | ~268 MB    |
| Code repos     | 4     | `code/`        | ~6.2 MB    |
| Datasets       | 5     | `datasets/`    | download-on-demand |

Paper chunks (3 pages each) are also available under `papers/pages/`.

---

## Papers (23)

All papers are downloaded to `papers/` and chunked for incremental reading
under `papers/pages/`. See `papers/README.md` for detailed descriptions and
reading status.

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Similarity of Neural Network Representations Revisited (CKA) | Kornblith et al. | 2019 | CKA_Kornblith_2019.pdf | **Foundational**. Introduces CKA. 2000+ cites. |
| SVCCA | Raghu et al. | 2017 | SVCCA_Raghu_2017.pdf | **Foundational**. SVD+CCA. 800+ cites. |
| Do Wide and Deep Networks Learn the Same Things? | Nguyen, Raghu, Kornblith | 2020 | WideDeep_Nguyen_2020.pdf | **Block structure**. 300+ cites. |
| On the Origins of the Block Structure Phenomenon | Nguyen, Raghu, Kornblith | 2022 | BlockStructure_Origins_2022.pdf | Block structure caused by dominant datapoints. |
| Do Vision Transformers See Like CNNs? | Raghu et al. | 2021 | ViT_vs_CNN_Raghu_2021.pdf | ViT has uniform CKA across layers. 1300+ cites. |
| Tracing Representation Progression | Jiang, Zhou, Zhu | 2024 | TracingRepProgression_2024.pdf | Sample-wise cosine ≈ CKA for transformers. |
| Understanding the Inner Workings of LMs through Representation Dissimilarity | Brown et al. | 2023 | InnerWorkingsLM_RepDiss_2023.pdf | Block structure in Pythia LMs. |
| Self-similarity Analysis in Deep Neural Networks | Ding et al. | 2025 | SelfSimilarity_2025.pdf | **MLP-specific**. Fractal self-similarity regularizer. |
| Inter-layer Information Similarity (NNTS / NNTP) | Hryniowski, Wong | 2020 | InterLayerInfoSim_Hu_2020.pdf | Topological persistence metrics. |
| Reliability of CKA | Davari et al. | 2022 | CKA_Reliability_2022.pdf | **Critique**: CKA can be manipulated. |
| Normalized Space Alignment (NSA) | Ebadulla, Gulati, Singh | 2024 | NormalizedSpaceAlign_2024.pdf | Global + local, differentiable loss. |
| Graph-Based Similarity of NN Representations | various | 2021 | GraphBasedSim_2021.pdf | Graph similarity approach. |
| One Wide Feedforward Is All You Need | Pessoa Pires et al. | 2023 | OneWideFFN_Pessoa_2023.pdf | Transformer FFNs are redundant. |
| Learning Richness Modulates Equality Reasoning in NNs | Tong, Pehlevan | 2025 | EqualityReasoningMLP_2025.pdf | **MLP theory**: rich vs. lazy regime. |
| Emergence of Globally Attracting Fixed Points | various | 2024 | GloballyAttractingFP_2024.pdf | MLPs drive to fixed point. |
| MLP-Mixer | Tolstikhin et al. | 2021 | MLPMixer_Original_2021.pdf | All-MLP vision architecture. |
| Understanding MLP-Mixer as a Wide Sparse MLP | various | 2023 | MLPMixer_Wide_Sparse_2023.pdf | Theoretical view. |
| Batch Normalization Orthogonalizes Representations | various | 2021 | BatchNormOrtho_2021.pdf | BN affects CKA. |
| Understanding Learning Dynamics of LMs with SVCCA | Saphra, Lopez | 2018 | SVCCA_LM_Saphra_2018.pdf | SVCCA across training time. |
| MPruner: CKA-Based Pruning | various | 2024 | MPruner_CKA_2024.pdf | Uses CKA between layers to prune. |
| LAYA: Layer-wise Attention Aggregation | Vessio | 2025 | LAYA_2025.pdf | Aggregates intermediate layers. |
| Curvature-based Comparison of Two NNs | Yu, Long, Hopcroft | 2018 | CurvatureCompare_Yu_2018.pdf | Manifold curvature on FC activations. |
| Subspace Clustering Based Analysis | various | 2021 | SubspaceClustering_2021.pdf | Neuron subspace clustering. |

Details: see `papers/README.md`.

## Datasets (5)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CIFAR-10 | torchvision / HuggingFace | 60k · 32×32 · 3 channels | 10-class image classification | `datasets/cifar10/` | **Primary**. Used by CKA, SVCCA, Wide/Deep, Block-Structure papers. |
| CIFAR-100 | torchvision / HuggingFace | 60k · 32×32 · 3 channels | 100-class image classification | `datasets/cifar100/` | Cross-dataset similarity tests. |
| MNIST | torchvision | 70k · 28×28 · 1 channel | 10-class digit classification | `datasets/mnist/` | Used by SVCCA, Inter-layer Information Similarity. |
| FashionMNIST | torchvision | 70k · 28×28 · 1 channel | 10-class clothing classification | `datasets/fashion_mnist/` | Optional. |
| Synthetic | generated on-the-fly | variable | toy regression / same-different | `datasets/synthetic/` | Used by SVCCA, Equality Reasoning MLP. |

All datasets download on demand via torchvision or HuggingFace — see
`datasets/README.md` for exact code snippets. The `.gitignore` in
`datasets/` excludes all data files from version control.

## Code Repositories (4)

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| PyTorch-Model-Compare (`torch_cka`) | https://github.com/AntixK/PyTorch-Model-Compare | Reference PyTorch CKA incl. minibatch estimator | `code/PyTorch-Model-Compare/` | Apache 2.0. **Primary tool for layer × layer CKA**. |
| anatome | https://github.com/moskomule/anatome | SVCCA, PWCCA, CKA, Procrustes in one library | `code/anatome/` | MIT. Multi-metric sanity checks. |
| sim_metric | https://github.com/js-d/sim_metric | Benchmark for similarity metrics | `code/sim_metric/` | MIT. Requires Zenodo embeddings to fully replicate; `dists/` module reusable. |
| LAYA | https://github.com/gvessio/LAYA | Layer-wise attention aggregation | `code/LAYA/` | Ablation baseline for inter-layer info content. |

Details: see `code/README.md`.

---

## Resource Gathering Notes

### Search Strategy
- Used `paper-finder` (diligent mode) twice: once for general MLP hidden
  state similarity, once for transformer FFN/MLP similarity. Combined 132
  unique papers, 39 with relevance ≥ 2.
- Cross-referenced with arXiv and Semantic Scholar APIs to resolve arxiv
  IDs from Semantic Scholar URLs (handled rate-limits with retry/backoff).
- For each downloaded PDF, ran the paper-finder `pdf_chunker.py` with
  3-page chunks for selective deep reading.

### Selection Criteria
1. **User hypothesis match**: papers directly addressing whether hidden
   states at different layers are similar.
2. **Methodological foundation**: CKA (Kornblith), SVCCA (Raghu) — must-have.
3. **Architecture coverage**: pure MLP, CNN, ResNet, ViT, LM papers.
4. **Recent** (2021+) preferred when available, but foundational 2017–2019
   papers are included.
5. **MLP-specific theory** (rich/lazy, fixed points) kept a higher bar
   given the hypothesis focuses on "limited individual transformation
   capacity" — a regime-theory statement.
6. **Critical perspectives** (CKA Reliability critique) included to
   prevent over-confident single-metric conclusions.

### Challenges Encountered
- Semantic Scholar rate-limited the bulk arxiv-ID lookup; worked around
  with exponential backoff and split batches.
- One paper (*Deep Networks as Paths on the Manifold*, Doimo et al. 2023)
  could not be located on arXiv from the search results; it is a
  published ICLR paper and can be obtained directly if the experiment
  runner needs it.
- `Tracing-Representation-Progression` and `Block-Structure` official
  code repos were not at guessed GitHub URLs. PyTorch-Model-Compare
  implements the minibatch CKA from the same authors, which is the
  functionality we need.

### Gaps and Workarounds
- **No user-specified `code_references`** were provided in the research
  topic, so all cloned repos are reference implementations chosen for
  their alignment with the metrics used in the literature.
- Pure-MLP empirical studies of cross-layer similarity are rare in the
  literature; the project is expected to fill this gap with its own
  experiments rather than rely on prior runs.

---

## Recommendations for Experiment Design

Based on the gathered resources:

1. **Primary dataset(s)**: CIFAR-10 (and MNIST for a quick second
   benchmark). Test split used as the probe set for CKA.
2. **Architecture**: single MLP class parametrized by (depth, width).
   Depth scan: {4, 8, 16, 32}. Width scan: {64, 256, 1024, 4096}.
   Nonlinearity: ReLU first, GeLU as ablation.
3. **Baselines**:
   - Untrained MLP of same shape (compares init vs. learned similarity).
   - Multiple seeds of each (depth, width) — variability signal.
   - Linear probe accuracy per layer (task-functional signal).
4. **Metrics**:
   - **Primary**: Linear CKA via minibatch estimator (torch_cka).
   - **Secondary**: sample-wise cosine similarity (Jiang et al. 2024).
   - **Secondary**: variance explained by 1st PC per layer.
   - **Robustness**: NSA or Procrustes (cross-metric consistency).
5. **Code to reuse**:
   - `torch_cka` for heatmaps.
   - `anatome` for SVCCA/PWCCA/Procrustes comparisons.
   - Custom PyTorch training loop for the MLP family (no existing
     MLP-similarity benchmark code found).
6. **Expected findings** (to confirm or refute):
   - Block structure appears when capacity is large relative to task.
   - Width amplifies per-layer representation similarity.
   - Rich-regime training produces more distinct layers than lazy-regime.
   - Early layers > late layers for cross-seed similarity.
   - MLPs should exhibit a **monotonic increase** of cosine similarity
     between layer L and layer L+k as k→small for sufficiently deep
     networks with residual-like structure.

See `literature_review.md` for the full synthesis and list of citations.
