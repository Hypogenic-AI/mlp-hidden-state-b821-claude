"""Train MLP models and collect activations + similarity metrics.

Uses GPU-resident data tensors — training loops have no CPU-side data
loading, which was previously the primary bottleneck.
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data import GPUSplit, load_dataset_to_gpu
from src.metrics import (
    cka_matrix,
    linear_cka,
    pc_variance_profile,
    procrustes_matrix,
    sample_cosine_matrix,
)
from src.models import MLP, MLPConfig, count_parameters, parameter_delta_ratio


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, split: GPUSplit) -> float:
    model.eval()
    correct = total = 0
    for x, y in split:
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


def train_model(
    cfg: MLPConfig,
    dataset: str,
    seed: int,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    batch_size: int = 512,
    shuffle_labels: bool = False,
) -> tuple[MLP, dict, GPUSplit, GPUSplit, GPUSplit]:
    set_seed(seed)
    train_split, val_split, probe_split, input_dim, num_classes, orig_labels = (
        load_dataset_to_gpu(dataset, device, batch_size=batch_size)
    )
    assert input_dim == cfg.input_dim
    assert num_classes == cfg.num_classes

    if shuffle_labels:
        rng = np.random.default_rng(seed)
        perm = orig_labels.numpy().copy()
        rng.shuffle(perm)
        train_split.y = torch.as_tensor(perm, device=device)

    model = MLP(cfg).to(device)
    init_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    history = {"train_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_split.iter_batches(generator=gen):
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        sched.step()
        val = evaluate(model, val_split)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_acc"].append(val)

    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    delta = parameter_delta_ratio(init_state, final_state)
    history["param_delta_ratio"] = delta
    history["n_params"] = count_parameters(model)
    return model, history, train_split, val_split, probe_split


@torch.no_grad()
def collect_activations(
    model: MLP, probe_split: GPUSplit
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    model.eval()
    per_layer_chunks: list[list[torch.Tensor]] = [[] for _ in range(model.cfg.depth)]
    ys, logits_all = [], []
    for x, y in probe_split:
        logits, acts = model.forward_with_activations(x)
        for i, a in enumerate(acts):
            per_layer_chunks[i].append(a.detach())
        ys.append(y)
        logits_all.append(logits.detach())
    per_layer = [torch.cat(chs, dim=0) for chs in per_layer_chunks]
    return per_layer, torch.cat(ys), torch.cat(logits_all)


def linear_probe_accuracy(
    activations: list[torch.Tensor],
    y: torch.Tensor,
    subsample: int = 2000,
    num_classes: int | None = None,
    steps: int = 200,
    lr: float = 0.1,
    weight_decay: float = 1e-3,
) -> list[float]:
    """GPU-native per-layer multinomial logistic-regression probe.

    60/40 split of the probe set (first 60 % train, last 40 % eval).
    Trains a single Linear layer with Adam + cross-entropy for `steps`
    iterations on full-batch normalized activations. Much faster than
    sklearn on high-dim features (100–1000× speedup).
    """
    device = activations[0].device
    n = y.shape[0]
    tr_end = int(0.6 * n)
    tr_idx = torch.arange(tr_end, device=device)
    te_idx = torch.arange(tr_end, n, device=device)
    if subsample is not None and subsample < tr_end:
        g = torch.Generator(device=device).manual_seed(0)
        perm = torch.randperm(tr_end, generator=g, device=device)[:subsample]
        tr_idx = perm

    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    accs = []
    y_tr = y[tr_idx]
    y_te = y[te_idx]
    for a in activations:
        x_tr = a[tr_idx].detach()
        x_te = a[te_idx].detach()
        # standardize
        mu = x_tr.mean(dim=0, keepdim=True)
        sd = x_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_tr = (x_tr - mu) / sd
        x_te = (x_te - mu) / sd
        d = x_tr.shape[1]
        W = torch.zeros(d, num_classes, device=device, requires_grad=True)
        b = torch.zeros(num_classes, device=device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
        for _ in range(steps):
            logits = x_tr @ W + b
            loss = torch.nn.functional.cross_entropy(logits, y_tr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (x_te @ W + b).argmax(dim=1)
            acc = float((pred == y_te).float().mean().item())
        accs.append(acc)
    return accs


def summarize(
    activations_gpu: list[torch.Tensor],
    y_gpu: torch.Tensor,
    with_probes: bool = True,
    probe_subsample: int = 2000,
) -> dict:
    cka = cka_matrix(activations_gpu)
    cos = sample_cosine_matrix(activations_gpu)
    proc = procrustes_matrix(activations_gpu)
    pc1 = pc_variance_profile(activations_gpu, k=1)
    probe_acc = (
        linear_probe_accuracy(activations_gpu, y_gpu, subsample=probe_subsample)
        if with_probes
        else None
    )
    return {
        "cka": cka.tolist(),
        "cosine": cos.tolist(),
        "procrustes": proc.tolist(),
        "pc1_variance": pc1.tolist(),
        "probe_accuracy_per_layer": probe_acc,
    }


def run_one(
    dataset: str,
    depth: int,
    width: int,
    seed: int,
    epochs: int,
    device: torch.device,
    init_scale: float = 1.0,
    shuffle_labels: bool = False,
    out_dir: Path = Path("results/metrics"),
    tag: str | None = None,
    batch_size: int = 512,
) -> dict:
    t0 = time.time()
    # Get dimensions without full load (tiny overhead)
    train_split, val_split, probe_split, input_dim, num_classes, orig_labels = (
        load_dataset_to_gpu(dataset, device, batch_size=batch_size)
    )
    cfg = MLPConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        depth=depth,
        width=width,
        init_scale=init_scale,
        nonlinearity="relu",
    )
    del train_split, val_split, probe_split, orig_labels  # reloaded inside train_model

    model, history, _, val_split, probe_split = train_model(
        cfg, dataset, seed, epochs, device=device, shuffle_labels=shuffle_labels,
        batch_size=batch_size,
    )
    test_acc = evaluate(model, val_split)

    acts, y, _ = collect_activations(model, probe_split)
    summary_trained = summarize(acts, y)

    # Freshly-initialized model (same seed) for comparison
    set_seed(seed)
    init_model = MLP(cfg).to(device)
    init_acts, y_init, _ = collect_activations(init_model, probe_split)
    summary_init = summarize(init_acts, y_init, with_probes=False)

    cka_to_init = [linear_cka(at, ai) for at, ai in zip(acts, init_acts)]

    # Free activations
    del acts, init_acts, model, init_model
    torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    name = tag or f"{dataset}_d{depth}_w{width}_s{seed}_init{init_scale}"
    if shuffle_labels:
        name += "_shufflabel"
    record = {
        "config": asdict(cfg),
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "shuffle_labels": shuffle_labels,
        "test_accuracy": test_acc,
        "history": history,
        "trained": summary_trained,
        "init": summary_init,
        "cka_trained_vs_init_per_layer": cka_to_init,
        "runtime_sec": time.time() - t0,
    }
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(record, f)
    return record
