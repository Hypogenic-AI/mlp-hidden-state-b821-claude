# Cloned Code Repositories

All repositories are shallow clones (`--depth 1`) of libraries implementing
the similarity metrics referenced in `literature_review.md`. None of these
are user-specified `code_references` — they were selected because they are
the de-facto reference implementations for the metrics we need.

## Repo 1: PyTorch-Model-Compare (`torch_cka`)

- **URL**: https://github.com/AntixK/PyTorch-Model-Compare
- **License**: Apache 2.0
- **Location**: `code/PyTorch-Model-Compare/`
- **Purpose**: Drop-in PyTorch implementation of **Centered Kernel
  Alignment**, including the minibatch variant from Nguyen, Raghu,
  Kornblith (2021). Handles module-hook-based activation extraction.
- **Install**: `pip install torch_cka` (or use the cloned source directly).
- **Key entry points**:
  - `torch_cka.CKA(model1, model2, model1_layers=..., model2_layers=...)`
  - `.compare(dataloader)` returns a layer × layer CKA matrix.
- **Notes**: Perfect match for our experiment — we can instantiate CKA on
  the same MLP with two copies of the same module list to get the full
  intra-model layer × layer heatmap.
- **Application to our research**: This is the canonical tool for
  computing the within-model layer-similarity heatmaps we need to answer
  the central research question.

## Repo 2: anatome

- **URL**: https://github.com/moskomule/anatome
- **License**: MIT
- **Location**: `code/anatome/`
- **Purpose**: PyTorch library implementing multiple representation
  similarity methods in one place:
  - SVCCA (Raghu et al. 2017)
  - PWCCA (Morcos et al. 2018)
  - Linear CKA (Kornblith et al. 2019)
  - Orthogonal Procrustes distance (Ding et al. 2021)
- **Install**: `pip install -U git+https://github.com/moskomule/anatome`
- **Key entry points**:
  - `anatome.Distance(model_a, model_b, method="pwcca" | "cka" | "procrustes")`
  - `.between(layer_a_name, layer_b_name)` pairwise comparisons
- **Notes**: Use this when we want **cross-metric sanity checks** — for
  the MLP-similarity question it's valuable to know whether CKA, CCA
  variants, and Procrustes tell the same story.
- **Application to our research**: The multi-metric toolkit validates
  findings; particularly useful given the CKA-reliability critique
  (Davari et al. 2022).

## Repo 3: sim_metric

- **URL**: https://github.com/js-d/sim_metric (Ding, Denain, Steinhardt
  2021, "Grounding Representation Similarity with Statistical Testing")
- **License**: MIT
- **Location**: `code/sim_metric/`
- **Purpose**: Benchmark for evaluating which representation-similarity
  metric tracks ground-truth differences (layer-depth changes, PCA
  deletion, finetuning seeds). Includes their dissimilarity
  implementations.
- **Notes**: Requires pre-computed embeddings from
  https://zenodo.org/record/5117844 to replicate their exact experiments,
  but the `dists/` module alone is directly reusable on our MLP
  activations.

## Repo 4: LAYA

- **URL**: https://github.com/gvessio/LAYA
- **Location**: `code/LAYA/`
- **Purpose**: Reference implementation of *Layer-wise Attention
  Aggregation* (Vessio 2025), which exploits cross-layer
  complementarity. Useful as an ablation baseline: if intermediate-layer
  representations are largely redundant (high CKA block structure), LAYA
  should give smaller gains than if they carry distinct info.

## Optional additions (not yet cloned)

The official CKA reference code lives inside `google-research/google-research`
at `similarity/`. That repo is ~2 GB so it wasn't cloned; `torch_cka`
(Repo 1) already reproduces the relevant functionality.

---

## Suggested usage inside experiments

```python
# Minimum viable experiment for the research question
import torch, torchvision
from torch_cka import CKA
from anatome import Distance

# 1. Define a small MLP with L hidden layers
class MLP(torch.nn.Module):
    def __init__(self, dims=[3072,512,512,512,512,10]):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
    def forward(self, x):
        x = x.flatten(1)
        for l in self.layers[:-1]:
            x = torch.relu(l(x))
        return self.layers[-1](x)

# 2. After training, use torch_cka to get pairwise layer CKA
layer_names = [f"layers.{i}" for i in range(len(mlp.layers)-1)]
cka = CKA(mlp, mlp, model1_layers=layer_names, model2_layers=layer_names)
cka.compare(test_loader)
cka.plot_results()  # heatmap
```
