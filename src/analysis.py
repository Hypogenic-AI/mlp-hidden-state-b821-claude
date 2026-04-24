"""Aggregate sweep JSON files and produce figures + summary tables.

Reads all results/metrics/*.json, produces:
- figures/cka_heatmap_grid_{dataset}.png  (depth × width grid of CKA heatmaps)
- figures/cka_vs_k_{dataset}.png           (off-diagonal CKA vs layer-gap k)
- figures/metric_agreement_{dataset}.png    (CKA vs cosine vs Procrustes scatter)
- figures/rich_lazy_{dataset}.png           (Δθ vs mean off-diagonal CKA)
- figures/probe_acc_curves_{dataset}.png    (per-layer probe accuracy)
- results/summary.csv                       (per-run aggregate stats)
- results/hypothesis_tests.json            (H1–H5 statistical tests)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path("results/metrics")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = Path("results")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_records() -> list[dict]:
    records = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
            r["_path"] = str(p)
        records.append(r)
    return records


def off_diag(m: np.ndarray) -> np.ndarray:
    mask = ~np.eye(m.shape[0], dtype=bool)
    return m[mask]


def adj_vs_far(cka: np.ndarray, k: int = 3) -> tuple[float, float]:
    L = cka.shape[0]
    adj = [cka[i, i + 1] for i in range(L - 1)]
    far = [cka[i, j] for i in range(L) for j in range(L) if abs(i - j) >= k]
    return float(np.mean(adj)), float(np.mean(far))


def summarize_records(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        # Skip the dominant-datapoint ablation file which has a different schema
        if "config" not in r:
            continue
        cfg = r["config"]
        cka = np.asarray(r["trained"]["cka"])
        cos = np.asarray(r["trained"]["cosine"])
        proc = np.asarray(r["trained"]["procrustes"])
        pc1 = np.asarray(r["trained"]["pc1_variance"])
        probes = r["trained"]["probe_accuracy_per_layer"]
        adj, far = adj_vs_far(cka)
        rows.append(
            {
                "dataset": r["dataset"],
                "depth": cfg["depth"],
                "width": cfg["width"],
                "seed": r["seed"],
                "init_scale": cfg.get("init_scale", 1.0),
                "shuffle_labels": r.get("shuffle_labels", False),
                "test_accuracy": r["test_accuracy"],
                "param_delta_ratio": r["history"]["param_delta_ratio"],
                "n_params": r["history"]["n_params"],
                "cka_offdiag_mean": float(off_diag(cka).mean()),
                "cosine_offdiag_mean": float(np.nanmean(off_diag(cos))),
                "procrustes_offdiag_mean": float(off_diag(proc).mean()),
                "cka_adj": adj,
                "cka_far_k3": far,
                "pc1_mean": float(pc1.mean()),
                "pc1_max_layer": int(np.argmax(pc1)),
                "probe_acc_first": float(probes[0]) if probes else np.nan,
                "probe_acc_last": float(probes[-1]) if probes else np.nan,
                "cka_to_init_mean": float(np.mean(r["cka_trained_vs_init_per_layer"])),
            }
        )
    return pd.DataFrame(rows)


def fig_heatmap_grid(records: list[dict], dataset: str) -> None:
    subset = [r for r in records if "config" in r and r["dataset"] == dataset and not r.get("shuffle_labels") and r["config"].get("init_scale", 1.0) == 1.0]
    if not subset:
        return
    # Average across seeds
    by_dw: dict[tuple[int, int], list[np.ndarray]] = {}
    for r in subset:
        key = (r["config"]["depth"], r["config"]["width"])
        by_dw.setdefault(key, []).append(np.asarray(r["trained"]["cka"]))
    depths = sorted({d for d, _ in by_dw})
    widths = sorted({w for _, w in by_dw})
    fig, axes = plt.subplots(
        len(depths), len(widths), figsize=(3 * len(widths), 3 * len(depths)),
        squeeze=False, constrained_layout=True
    )
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            ax = axes[i][j]
            mats = by_dw.get((d, w))
            if mats is None:
                ax.set_visible(False)
                continue
            m = np.mean(np.stack(mats), axis=0)
            im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis", origin="lower")
            ax.set_title(f"d={d}, w={w}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Layer×Layer Linear CKA — trained MLP — {dataset.upper()}", fontsize=13)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("CKA")
    out = FIG_DIR / f"cka_heatmap_grid_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_cka_vs_k(records: list[dict], dataset: str) -> None:
    subset = [r for r in records if "config" in r and r["dataset"] == dataset and not r.get("shuffle_labels") and r["config"].get("init_scale", 1.0) == 1.0]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    # Group by (depth, width), average over seeds
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for r in subset:
        key = (r["config"]["depth"], r["config"]["width"])
        groups.setdefault(key, []).append(np.asarray(r["trained"]["cka"]))
    cmap = plt.get_cmap("plasma")
    keys = sorted(groups)
    for idx, (d, w) in enumerate(keys):
        mats = np.stack(groups[(d, w)])  # (S, L, L)
        L = mats.shape[1]
        ks = np.arange(L)
        means = np.zeros(L)
        stds = np.zeros(L)
        for k in range(L):
            diag = np.array([np.diag(m, k=k).mean() for m in mats])
            means[k] = diag.mean()
            stds[k] = diag.std()
        color = cmap(idx / max(1, len(keys) - 1))
        ax.plot(ks, means, color=color, label=f"d={d}, w={w}")
        ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.15)
    ax.set_xlabel("layer gap k")
    ax.set_ylabel("mean CKA at gap k")
    ax.set_title(f"Cross-layer CKA decay — {dataset.upper()} (mean ± std over seeds)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    out = FIG_DIR / f"cka_vs_k_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_metric_agreement(records: list[dict], dataset: str) -> None:
    subset = [r for r in records if "config" in r and r["dataset"] == dataset and not r.get("shuffle_labels") and r["config"].get("init_scale", 1.0) == 1.0]
    if not subset:
        return
    pairs_cka: list[float] = []
    pairs_cos: list[float] = []
    pairs_proc: list[float] = []
    for r in subset:
        cka = np.asarray(r["trained"]["cka"])
        cos = np.asarray(r["trained"]["cosine"])
        proc = np.asarray(r["trained"]["procrustes"])
        L = cka.shape[0]
        for i in range(L):
            for j in range(i + 1, L):
                pairs_cka.append(float(cka[i, j]))
                pairs_cos.append(float(cos[i, j]))
                pairs_proc.append(float(proc[i, j]))
    cka_arr = np.asarray(pairs_cka)
    cos_arr = np.asarray(pairs_cos)
    proc_arr = np.asarray(pairs_proc)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    # Cosine vs CKA
    mask = ~(np.isnan(cka_arr) | np.isnan(cos_arr))
    axs[0].scatter(cka_arr[mask], cos_arr[mask], s=8, alpha=0.5)
    rho, p = stats.spearmanr(cka_arr[mask], cos_arr[mask])
    axs[0].set_xlabel("Linear CKA")
    axs[0].set_ylabel("Sample-wise cosine")
    axs[0].set_title(f"cosine vs CKA (ρ={rho:.3f}, p={p:.1e})")
    axs[0].grid(alpha=0.3)
    # Procrustes vs CKA
    mask = ~(np.isnan(cka_arr) | np.isnan(proc_arr))
    axs[1].scatter(cka_arr[mask], proc_arr[mask], s=8, alpha=0.5, color="darkorange")
    rho, p = stats.spearmanr(cka_arr[mask], proc_arr[mask])
    axs[1].set_xlabel("Linear CKA")
    axs[1].set_ylabel("Procrustes similarity")
    axs[1].set_title(f"Procrustes vs CKA (ρ={rho:.3f}, p={p:.1e})")
    axs[1].grid(alpha=0.3)
    fig.suptitle(f"Metric agreement on layer-pair similarity — {dataset.upper()}")
    out = FIG_DIR / f"metric_agreement_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_rich_lazy(records: list[dict], dataset: str) -> None:
    subset = [r for r in records if "config" in r and r["dataset"] == dataset and not r.get("shuffle_labels")]
    if not subset:
        return
    xs, ys, labels = [], [], []
    for r in subset:
        cka = np.asarray(r["trained"]["cka"])
        xs.append(r["history"]["param_delta_ratio"])
        ys.append(float(off_diag(cka).mean()))
        labels.append(f"d{r['config']['depth']}w{r['config']['width']}_init{r['config'].get('init_scale', 1.0)}_s{r['seed']}")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.scatter(xs, ys, s=30, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\|\theta_{final} - \theta_{init}\| / \|\theta_{init}\|$ (log scale)")
    ax.set_ylabel("Mean off-diagonal CKA")
    ax.set_title(f"Rich vs lazy regime — {dataset.upper()}")
    ax.grid(alpha=0.3)
    out = FIG_DIR / f"rich_lazy_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_probe_acc(records: list[dict], dataset: str) -> None:
    subset = [r for r in records if "config" in r and r["dataset"] == dataset and not r.get("shuffle_labels") and r["config"].get("init_scale", 1.0) == 1.0]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    groups: dict[tuple[int, int], list[list[float]]] = {}
    for r in subset:
        probes = r["trained"].get("probe_accuracy_per_layer")
        if not probes:
            continue
        key = (r["config"]["depth"], r["config"]["width"])
        groups.setdefault(key, []).append(probes)
    cmap = plt.get_cmap("plasma")
    keys = sorted(groups)
    for idx, k in enumerate(keys):
        arr = np.asarray(groups[k])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        xs = np.arange(len(mean))
        color = cmap(idx / max(1, len(keys) - 1))
        ax.plot(xs, mean, color=color, label=f"d={k[0]}, w={k[1]}")
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("layer index")
    ax.set_ylabel("linear-probe accuracy")
    ax.set_title(f"Per-layer probe accuracy — {dataset.upper()}")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    out = FIG_DIR / f"probe_acc_curves_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_init_vs_trained(records: list[dict], dataset: str) -> None:
    """Compare mean off-diag CKA at init vs trained for each config."""
    subset = [r for r in records if "config" in r and r["dataset"] == dataset and not r.get("shuffle_labels") and r["config"].get("init_scale", 1.0) == 1.0]
    if not subset:
        return
    xs, ys, sizes, colors = [], [], [], []
    keys = []
    for r in subset:
        trained = off_diag(np.asarray(r["trained"]["cka"])).mean()
        initm = off_diag(np.asarray(r["init"]["cka"])).mean()
        xs.append(initm)
        ys.append(trained)
        sizes.append(np.log10(r["history"]["n_params"]) * 8)
        colors.append(r["config"]["depth"])
        keys.append((r["config"]["depth"], r["config"]["width"]))
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.75)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_xlabel("Mean off-diagonal CKA (at init)")
    ax.set_ylabel("Mean off-diagonal CKA (trained)")
    ax.set_title(f"Training's effect on cross-layer CKA — {dataset.upper()}")
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("MLP depth")
    out = FIG_DIR / f"init_vs_trained_cka_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def hypothesis_tests(df: pd.DataFrame) -> dict:
    out: dict = {}
    # H1: adj > far via paired t-test across runs (per dataset)
    for dataset in df["dataset"].unique():
        d = df[(df["dataset"] == dataset) & (~df["shuffle_labels"]) & (df["init_scale"] == 1.0)]
        if len(d) == 0:
            continue
        t, p = stats.ttest_rel(d["cka_adj"], d["cka_far_k3"])
        out[f"H1_adj_gt_far_{dataset}"] = {
            "mean_adj": float(d["cka_adj"].mean()),
            "mean_far_k3": float(d["cka_far_k3"].mean()),
            "t": float(t),
            "p_value": float(p),
            "n": int(len(d)),
        }
    # H2: width amplifies similarity — Spearman on (width, off-diag CKA) at fixed depth
    for dataset in df["dataset"].unique():
        d = df[(df["dataset"] == dataset) & (~df["shuffle_labels"]) & (df["init_scale"] == 1.0)]
        for depth in sorted(d["depth"].unique()):
            dd = d[d["depth"] == depth]
            if len(dd) < 4:
                continue
            rho, p = stats.spearmanr(dd["width"], dd["cka_offdiag_mean"])
            out[f"H2_width_vs_cka_{dataset}_depth{depth}"] = {
                "rho": float(rho),
                "p_value": float(p),
                "n": int(len(dd)),
            }
    # H5: rich vs lazy via Spearman on (Δθ, off-diag CKA)
    for dataset in df["dataset"].unique():
        d = df[(df["dataset"] == dataset) & (~df["shuffle_labels"]) ]
        if len(d) < 4:
            continue
        rho, p = stats.spearmanr(d["param_delta_ratio"], d["cka_offdiag_mean"])
        out[f"H5_delta_vs_cka_{dataset}"] = {
            "rho": float(rho),
            "p_value": float(p),
            "n": int(len(d)),
        }
    return out


def main() -> None:
    records = load_records()
    print(f"Loaded {len(records)} records")
    df = summarize_records(records)
    df.to_csv(SUMMARY_DIR / "summary.csv", index=False)
    print(df.groupby(["dataset", "depth", "width"]).agg(
        cka=("cka_offdiag_mean", "mean"),
        acc=("test_accuracy", "mean"),
        delta=("param_delta_ratio", "mean"),
        n=("seed", "count"),
    ))

    for dataset in df["dataset"].unique():
        fig_heatmap_grid(records, dataset)
        fig_cka_vs_k(records, dataset)
        fig_metric_agreement(records, dataset)
        fig_probe_acc(records, dataset)
        fig_rich_lazy(records, dataset)
        fig_init_vs_trained(records, dataset)

    tests = hypothesis_tests(df)
    with open(SUMMARY_DIR / "hypothesis_tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
