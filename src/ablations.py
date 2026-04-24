"""Targeted ablations on a reference (depth, width) configuration.

These target individual hypotheses:
- H5: rich vs lazy regime via init-scale sweep
- Control: shuffled-label training
- H6: dominant-datapoint exclusion — recompute CKA after dropping top-
  decile activation-norm probe examples
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data import load_dataset_to_gpu
from src.metrics import cka_matrix
from src.train import (
    collect_activations,
    run_one,
    set_seed,
)


def rich_lazy_sweep(
    depth: int, width: int, dataset: str, device: torch.device,
    scales: list[float], seed: int = 0, epochs: int = 15,
) -> None:
    out_dir = Path("results/metrics")
    for scale in scales:
        tag = f"{dataset}_d{depth}_w{width}_s{seed}_init{scale}"
        if (out_dir / f"{tag}.json").exists():
            print(f"SKIP {tag}")
            continue
        t0 = time.time()
        rec = run_one(
            dataset=dataset, depth=depth, width=width, seed=seed,
            epochs=epochs, init_scale=scale, device=device, out_dir=out_dir, tag=tag,
            batch_size=512,
        )
        print(f"{tag} acc={rec['test_accuracy']:.3f} delta={rec['history']['param_delta_ratio']:.3f} t={time.time()-t0:.1f}s")


def shuffled_label_control(
    depth: int, width: int, dataset: str, device: torch.device, seed: int = 0, epochs: int = 15,
) -> None:
    out_dir = Path("results/metrics")
    tag = f"{dataset}_d{depth}_w{width}_s{seed}_init1.0_shufflabel"
    if (out_dir / f"{tag}.json").exists():
        print(f"SKIP {tag}")
        return
    t0 = time.time()
    rec = run_one(
        dataset=dataset, depth=depth, width=width, seed=seed,
        epochs=epochs, init_scale=1.0, shuffle_labels=True,
        device=device, out_dir=out_dir, tag=tag, batch_size=512,
    )
    print(f"{tag} acc={rec['test_accuracy']:.3f} delta={rec['history']['param_delta_ratio']:.3f} t={time.time()-t0:.1f}s")


def dominant_datapoint_ablation(
    depth: int, width: int, dataset: str, device: torch.device, seed: int = 0, epochs: int = 15,
) -> None:
    """Train model, then recompute CKA after dropping top-decile high-activation-norm samples."""
    from src.models import MLP, MLPConfig

    out_dir = Path("results/metrics")
    _, _, _, input_dim, num_classes, _ = load_dataset_to_gpu(dataset, device, batch_size=512)
    cfg = MLPConfig(input_dim=input_dim, num_classes=num_classes, depth=depth, width=width)

    # Reuse the main trained model if it exists — else train a new one
    tag_base = f"{dataset}_d{depth}_w{width}_s{seed}_init1.0"
    # We always retrain — simpler and we want the activations anyway
    set_seed(seed)
    from src.train import train_model

    model, history, _, _, probe_split = train_model(cfg, dataset, seed, epochs, device=device, batch_size=512)
    acts, y, _ = collect_activations(model, probe_split)

    # Activations already on GPU now
    concat = torch.cat([a for a in acts], dim=1)  # (n, sum(widths))
    norms = concat.norm(dim=1).cpu().numpy()  # (n,)
    threshold = np.quantile(norms, 0.9)
    keep = torch.as_tensor(norms < threshold, device=device)
    print(f"dominant_ablation n={len(norms)} keep_frac={float(keep.float().mean()):.2f} thr={threshold:.3f}")

    acts_gpu_kept = [a[keep] for a in acts]
    cka_full = cka_matrix(acts)
    cka_kept = cka_matrix(acts_gpu_kept)
    del acts_gpu_kept
    torch.cuda.empty_cache()

    record = {
        "dataset": dataset,
        "depth": depth,
        "width": width,
        "seed": seed,
        "threshold_quantile": 0.9,
        "threshold_value": float(threshold),
        "fraction_kept": float(keep.float().mean()),
        "cka_full": cka_full.tolist(),
        "cka_after_dropping_top_decile": cka_kept.tolist(),
        "delta_mean_off_diag": float(
            (cka_full - cka_kept)[~np.eye(cka_full.shape[0], dtype=bool)].mean()
        ),
    }
    tag = f"dominant_ablation_{dataset}_d{depth}_w{width}_s{seed}"
    with open(Path("results/metrics") / f"{tag}.json", "w") as f:
        json.dump(record, f)
    print(f"wrote {tag}  Δmean_offdiag={record['delta_mean_off_diag']:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2", type=str)
    p.add_argument("--dataset", default="cifar10", type=str)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 2.0, 4.0])
    p.add_argument("--skip-rich-lazy", action="store_true")
    p.add_argument("--skip-shuffle", action="store_true")
    p.add_argument("--skip-dominant", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)

    if not args.skip_rich_lazy:
        print("=== rich-lazy sweep ===")
        rich_lazy_sweep(args.depth, args.width, args.dataset, device, args.scales, args.seed, args.epochs)
    if not args.skip_shuffle:
        print("=== shuffled-label control ===")
        shuffled_label_control(args.depth, args.width, args.dataset, device, args.seed, args.epochs)
    if not args.skip_dominant:
        print("=== dominant-datapoint ablation ===")
        dominant_datapoint_ablation(args.depth, args.width, args.dataset, device, args.seed, args.epochs)


if __name__ == "__main__":
    main()
