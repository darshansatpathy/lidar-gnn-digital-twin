"""A dynamic graph CNN (DGCNN) for per-point classification, on PyTorch Geometric's EdgeConv.

Each point is a node. The first EdgeConv connects every point to its k nearest neighbours in
space; the next two rebuild the graph in the learned feature space, so points that look alike
(two stretches of the same facade, the two sides of a pole) exchange information even when they
are not adjacent. A global max over the block adds scene context before the per-point classifier.

Neighbours are found with a chunked exact search in plain PyTorch, so the model needs no compiled
extension beyond PyTorch itself.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch_geometric.nn import EdgeConv


def knn_edges(x: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """Edge index (2 x n*k) linking each point to its k nearest neighbours (itself excluded)."""
    n = x.shape[0]
    k = min(k, n - 1)
    cols = []
    for start in range(0, n, chunk):
        d = torch.cdist(x[start : start + chunk], x)
        d[torch.arange(d.shape[0]), torch.arange(start, start + d.shape[0])] = float("inf")
        cols.append(d.topk(k, largest=False).indices)
    nbr = torch.cat(cols)  # n x k
    centre = torch.arange(n, device=x.device).repeat_interleave(k)
    # EdgeConv aggregates messages at edge_index[1] from edge_index[0]: (neighbour -> centre).
    return torch.stack([nbr.reshape(-1), centre])


def _mlp(*sizes: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in itertools.pairwise(sizes):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.LeakyReLU(0.2)]
    return nn.Sequential(*layers)


class DGCNN(nn.Module):
    def __init__(self, in_dim: int = 5, classes: int = 5, k: int = 16, width: int = 64) -> None:
        super().__init__()
        self.k = k
        self.conv1 = EdgeConv(_mlp(2 * in_dim, width, width), aggr="max")
        self.conv2 = EdgeConv(_mlp(2 * width, width), aggr="max")
        self.conv3 = EdgeConv(_mlp(2 * width, width), aggr="max")
        self.lift = _mlp(3 * width, 4 * width)
        self.head = nn.Sequential(_mlp(7 * width, 2 * width), nn.Dropout(0.3), nn.Linear(2 * width, classes))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: points x in_dim for one block; returns points x classes logits."""
        xyz = feats[:, :3]
        x1 = self.conv1(feats, knn_edges(xyz, self.k))
        x2 = self.conv2(x1, knn_edges(x1, self.k))
        x3 = self.conv3(x2, knn_edges(x2, self.k))
        local = torch.cat([x1, x2, x3], dim=1)
        context = self.lift(local).max(dim=0, keepdim=True).values.expand(feats.shape[0], -1)
        out: torch.Tensor = self.head(torch.cat([local, context], dim=1))
        return out


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
