"""MLP family for cross-layer similarity studies.

A fixed-width ReLU MLP with `depth` hidden layers, each of `width`
neurons. Registers forward-hook capture points so activations can be
collected from every hidden layer in a single forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    depth: int
    width: int
    init_scale: float = 1.0
    nonlinearity: str = "relu"


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        nl = nn.ReLU if cfg.nonlinearity == "relu" else nn.GELU
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        self.hidden_layers: list[nn.Linear] = []
        for _ in range(cfg.depth):
            lin = nn.Linear(in_dim, cfg.width)
            layers.append(lin)
            layers.append(nl())
            self.hidden_layers.append(lin)
            in_dim = cfg.width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, cfg.num_classes)
        self._rescale_init(cfg.init_scale)

    def _rescale_init(self, scale: float) -> None:
        if abs(scale - 1.0) < 1e-8:
            return
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.mul_(scale)
                    # biases left at default zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.head(self.backbone(x))

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward pass, return (logits, list_of_post_activations).

        Captures the output of every ReLU/GELU nonlinearity (i.e. the
        hidden state that the *next* Linear sees). List length == depth.
        """
        acts: list[torch.Tensor] = []
        h = x.flatten(1)
        for i in range(0, len(self.backbone), 2):
            lin, nl = self.backbone[i], self.backbone[i + 1]
            h = nl(lin(h))
            acts.append(h)
        logits = self.head(h)
        return logits, acts


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_delta_ratio(
    initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor]
) -> float:
    """Return ||θ_f - θ_i|| / ||θ_i|| summed over parameters.

    Used as a rich-vs-lazy diagnostic (Chizat & Bach 2019 convention).
    """
    num_sq = 0.0
    den_sq = 0.0
    for k, v0 in initial.items():
        v1 = final[k]
        num_sq += float((v1 - v0).pow(2).sum().item())
        den_sq += float(v0.pow(2).sum().item())
    if den_sq <= 0:
        return float("nan")
    return (num_sq / den_sq) ** 0.5
