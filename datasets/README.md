# Datasets

This directory holds datasets used to probe MLP hidden-state similarity. Data
files are **not** committed to git (see `.gitignore`) — follow the download
instructions below.

The research hypothesis is:

> The hidden spaces of MLPs at different layers are not totally different due to
> limited individual transformation capacity, but there are measurable
> differences that can be compared.

For this question we need a task where we can train MLPs of modest and varying
depth/width and extract per-layer activations on a held-out probe set. The
literature on cross-layer representation similarity (CKA, SVCCA, block
structure) uses the datasets below almost exclusively, so reusing them keeps
our results comparable.

---

## Dataset 1: CIFAR-10

### Overview
- **Source**: HuggingFace `uoft-cs/cifar10` (or `torchvision.datasets.CIFAR10`)
- **Size**: 60,000 32×32 color images (50k train, 10k test)
- **Format**: PIL images + integer labels (HuggingFace Dataset)
- **Task**: 10-way image classification
- **Splits**: train (50k), test (10k)
- **License**: MIT / research use
- **Why for this project**: The default benchmark in every major
  representation-similarity paper we review (CKA, SVCCA, Wide/Deep,
  Block-Structure). Small enough to train MLPs in minutes, large enough for
  per-example CKA with a meaningful probe set.

### Download Instructions

**Option A — torchvision (recommended for MLPs on raw pixels):**

```python
from torchvision import datasets, transforms
tfm = transforms.Compose([transforms.ToTensor()])
train = datasets.CIFAR10(root="datasets/cifar10", train=True, download=True, transform=tfm)
test  = datasets.CIFAR10(root="datasets/cifar10", train=False, download=True, transform=tfm)
```

**Option B — HuggingFace:**

```python
from datasets import load_dataset
ds = load_dataset("uoft-cs/cifar10")
ds.save_to_disk("datasets/cifar10_hf")
```

### Loading (after download)

```python
import torchvision, torch
tfm = torchvision.transforms.ToTensor()
train = torchvision.datasets.CIFAR10("datasets/cifar10", train=True, transform=tfm)
loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=False)
```

### Sample usage for similarity analysis

A typical experiment uses ~1000–5000 held-out examples as the "probe set"
whose activations are collected at every hidden layer. CKA and other
similarity indices are then computed between layer pairs.

### Notes
- For pure MLPs, flatten 3×32×32 → 3072-D inputs.
- Standardize per-channel or normalize to [0,1] — follow Kornblith et al.
  (2019) conventions if you want directly comparable numbers.

---

## Dataset 2: CIFAR-100

### Overview
- **Source**: HuggingFace `uoft-cs/cifar100` (or `torchvision.datasets.CIFAR100`)
- **Size**: 60,000 32×32 color images across 100 fine-grained classes
- **Format**: PIL images + integer labels
- **Task**: 100-way image classification
- **Splits**: train (50k), test (10k)
- **License**: MIT / research use
- **Why for this project**: Used by CKA and Wide/Deep papers to compare
  similarity of early-layer representations across **different datasets**
  (Kornblith et al. 2019 Fig. 7). Useful for testing whether cross-dataset
  similarity patterns hold for pure MLPs.

### Download Instructions

```python
from torchvision import datasets, transforms
tfm = transforms.Compose([transforms.ToTensor()])
train = datasets.CIFAR100(root="datasets/cifar100", train=True, download=True, transform=tfm)
test  = datasets.CIFAR100(root="datasets/cifar100", train=False, download=True, transform=tfm)
```

---

## Dataset 3: MNIST

### Overview
- **Source**: `torchvision.datasets.MNIST`
- **Size**: 70,000 28×28 grayscale images (60k train, 10k test)
- **Format**: Tensor + integer labels
- **Task**: 10-way digit classification
- **Splits**: train (60k), test (10k)
- **License**: Creative Commons
- **Why for this project**: Used by SVCCA, Inter-layer Information Similarity
  (LeNet analysis) and the MLP-self-similarity paper. MNIST is small enough
  that MLPs can reach ≥98% accuracy, so depth/width variations can be
  studied without confounding by under-fitting.

### Download Instructions

```python
from torchvision import datasets, transforms
tfm = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(root="datasets/mnist", train=True, download=True, transform=tfm)
test  = datasets.MNIST(root="datasets/mnist", train=False, download=True, transform=tfm)
```

---

## Dataset 4 (optional): FashionMNIST

### Overview
- **Source**: `torchvision.datasets.FashionMNIST`
- **Size**: 70,000 28×28 grayscale images, 10 clothing classes
- **Task**: 10-way classification, harder than MNIST
- **Why for this project**: Useful as a drop-in replacement / secondary
  benchmark to test whether conclusions about MLP hidden-state similarity
  are task-specific.

### Download Instructions

```python
from torchvision import datasets, transforms
tfm = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(root="datasets/fashion_mnist", train=True, download=True, transform=tfm)
test  = datasets.FashionMNIST(root="datasets/fashion_mnist", train=False, download=True, transform=tfm)
```

---

## Dataset 5 (optional): Synthetic / Toy Regression

### Overview
Several foundational papers (SVCCA's toy regression in Raghu et al. 2017,
Kornblith 2019 appendix experiments, Equality-Reasoning-in-MLPs 2025) probe
similarity on synthetic tasks where the data manifold is known.

Examples we may generate in the experiment runner:
- **Toy regression**: 1D input, multiple output functions (SVCCA style).
- **Two-moons / concentric rings**: binary classification, 2-D inputs.
- **Same-different (SD) task**: Two images are "same" iff they share an
  abstract feature; used in Equality-Reasoning-in-MLPs (2025) to
  distinguish rich vs. lazy feature-learning regimes.
- **Gaussian mixtures**: variable number of modes, controlled intrinsic
  dimensionality.

These can be generated on the fly inside the experiment scripts; no
download is needed.

---

## Summary Table

| Dataset       | Size    | Input dim | Classes | Source         |
|---------------|---------|-----------|---------|----------------|
| CIFAR-10      | 60k     | 3072      | 10      | torchvision    |
| CIFAR-100     | 60k     | 3072      | 100     | torchvision    |
| MNIST         | 70k     | 784       | 10      | torchvision    |
| FashionMNIST  | 70k     | 784       | 10      | torchvision    |
| Synthetic     | N/A     | varies    | varies  | generated      |

## Recommendation for experiment runner

Primary: **CIFAR-10** + **MNIST**. They cover the ranges tested by CKA/SVCCA
and allow MLPs of varying width/depth to reach good accuracy, so any
cross-layer similarity differences reflect representation structure rather
than underfitting artifacts. Use the test split (10k examples) as the probe
set for similarity computation.
