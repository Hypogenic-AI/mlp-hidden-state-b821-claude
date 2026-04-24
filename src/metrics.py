"""Representation-similarity metrics.

Implemented here rather than importing from `code/PyTorch-Model-Compare`
because we want a small, audited dependency surface, compute everything
on GPU, and return consistent numpy arrays.

Metrics:
- linear CKA (Kornblith 2019)
- sample-wise cosine similarity (Jiang 2024)
- orthogonal Procrustes distance (Williams et al.; normalized)
- variance explained by first principal component
- CKA-to-reference (layer vs. initial layer)
"""
from __future__ import annotations

import numpy as np
import torch


# -------------------------------------------------------------------
# Linear CKA
# -------------------------------------------------------------------
def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between (n, d1) and (n, d2) activation matrices.

    Uses the feature-space formula CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    which is mathematically equivalent to the Gram-matrix form and runs
    O(n*(d1+d2)*min(d1,d2)) — much cheaper than the n×n Gram matrix when
    n >> d.
    """
    x = _center(x.float())
    y = _center(y.float())
    xty = x.T @ y  # (d1, d2)
    xtx = x.T @ x  # (d1, d1)
    yty = y.T @ y  # (d2, d2)
    num = (xty ** 2).sum()
    den = torch.sqrt((xtx ** 2).sum() * (yty ** 2).sum())
    if den.item() <= 0:
        return float("nan")
    return float((num / den).item())


def cka_matrix(
    activations: list[torch.Tensor],
) -> np.ndarray:
    """Return L x L linear-CKA matrix for a list of (n, d_i) activations."""
    L = len(activations)
    out = np.ones((L, L), dtype=np.float32)
    # Precompute centered matrices and XtX
    centered: list[torch.Tensor] = []
    for a in activations:
        centered.append(_center(a.float()))
    for i in range(L):
        xi = centered[i]
        xixi = (xi.T @ xi)
        fx = (xixi ** 2).sum().sqrt()
        for j in range(i + 1, L):
            xj = centered[j]
            xjxj = (xj.T @ xj)
            fy = (xjxj ** 2).sum().sqrt()
            num = ((xi.T @ xj) ** 2).sum()
            den = fx * fy
            val = float((num / den).item()) if den.item() > 0 else float("nan")
            out[i, j] = val
            out[j, i] = val
    return out


# -------------------------------------------------------------------
# Sample-wise cosine similarity (Jiang 2024)
# -------------------------------------------------------------------
def sample_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean per-sample cosine similarity between two activation matrices.

    Each row of x and y is a sample's representation. We mean-center over
    the dataset axis (consistent with CKA) and then compute the cosine
    similarity row by row.
    """
    x = _center(x.float())
    y = _center(y.float())
    # normalize rows
    x_n = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    y_n = y / (y.norm(dim=1, keepdim=True) + 1e-12)
    # Different-width layers — cosine between rows requires equal dim.
    # Project the larger into the smaller via left singular vectors of the
    # concatenation — but for simplicity we only compute sample-wise
    # cosine when dimensions match. For cross-width comparisons the
    # function returns NaN; the report uses this metric mainly within the
    # constant-width hidden-layer block (which is our setup).
    if x_n.shape[1] != y_n.shape[1]:
        return float("nan")
    return float((x_n * y_n).sum(dim=1).mean().item())


def sample_cosine_matrix(activations: list[torch.Tensor]) -> np.ndarray:
    L = len(activations)
    out = np.ones((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            v = sample_cosine(activations[i], activations[j])
            out[i, j] = v
            out[j, i] = v
    return out


# -------------------------------------------------------------------
# Orthogonal Procrustes distance (normalized)
# -------------------------------------------------------------------
def procrustes_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Normalized orthogonal Procrustes distance.

    After centering and Frobenius-normalizing X, Y, we minimize
    ||X - Y R||_F over orthogonal R (when dims match) or pad the smaller
    to the larger with zeros. The reported value is a similarity:
    1 - 0.5 * ||X - YR||_F^2 in [-1, 1], which equals the CKA-equivalent
    when the representations are rotations of each other.
    """
    x = _center(x.float())
    y = _center(y.float())
    # Normalize Frobenius norms
    x = x / (x.norm() + 1e-12)
    y = y / (y.norm() + 1e-12)
    # Pad to same dim
    d = max(x.shape[1], y.shape[1])
    if x.shape[1] < d:
        x = torch.nn.functional.pad(x, (0, d - x.shape[1]))
    if y.shape[1] < d:
        y = torch.nn.functional.pad(y, (0, d - y.shape[1]))
    u, s, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    nuclear = s.sum().item()
    return float(nuclear)  # in [0, 1], 1 if perfectly aligned


def procrustes_matrix(activations: list[torch.Tensor]) -> np.ndarray:
    L = len(activations)
    out = np.ones((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            v = procrustes_distance(activations[i], activations[j])
            out[i, j] = v
            out[j, i] = v
    return out


# -------------------------------------------------------------------
# PC-1 variance explained
# -------------------------------------------------------------------
def pc_variance_explained(acts: torch.Tensor, k: int = 1) -> float:
    """Fraction of variance explained by the first `k` principal components."""
    x = _center(acts.float())
    # SVD on centered data (n, d). Use economy SVD.
    # s^2 / (n-1) are eigenvalues of covariance.
    try:
        s = torch.linalg.svdvals(x)
    except torch._C._LinAlgError:
        return float("nan")
    var = (s ** 2)
    total = var.sum().item()
    if total <= 0:
        return float("nan")
    return float((var[:k].sum() / var.sum()).item())


def pc_variance_profile(activations: list[torch.Tensor], k: int = 1) -> np.ndarray:
    return np.asarray(
        [pc_variance_explained(a, k=k) for a in activations], dtype=np.float32
    )
