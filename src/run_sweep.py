"""Run the full (dataset × depth × width × seed) sweep.

Dispatches configurations sequentially but on GPU, writing a JSON record
per run into results/metrics/.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import torch

from src.train import run_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/metrics", type=str)
    p.add_argument("--device", default="cuda:1", type=str)
    p.add_argument("--datasets", nargs="+", default=["cifar10", "mnist"])
    p.add_argument("--depths", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--widths", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--cifar-epochs", type=int, default=25)
    p.add_argument("--mnist-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = list(product(args.datasets, args.depths, args.widths, args.seeds))
    print(f"Total runs: {len(configs)}")
    t_all = time.time()
    for i, (ds, d, w, s) in enumerate(configs):
        name = f"{ds}_d{d}_w{w}_s{s}_init1.0"
        path = out_dir / f"{name}.json"
        if path.exists():
            print(f"[{i+1}/{len(configs)}] SKIP (exists) {name}")
            continue
        ep = args.cifar_epochs if ds == "cifar10" else args.mnist_epochs
        t0 = time.time()
        try:
            rec = run_one(
                dataset=ds, depth=d, width=w, seed=s, epochs=ep, device=device,
                out_dir=out_dir, batch_size=args.batch_size,
            )
            print(
                f"[{i+1}/{len(configs)}] {name}  acc={rec['test_accuracy']:.3f}  "
                f"Δθ={rec['history']['param_delta_ratio']:.3f}  t={time.time()-t0:.1f}s"
            )
        except Exception as e:  # pragma: no cover
            print(f"[{i+1}/{len(configs)}] FAILED {name}: {e}")
    print(f"Total sweep time: {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
