"""Geometry shared by the model and the baseline: ground height, blocks, and local shape features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from lidartwin.scenes import Scene

BLOCK = 20.0  # metres along the street


def height_above_ground(points: np.ndarray, cell: float = 2.0) -> np.ndarray:
    """Height of each point above a terrain model: the lowest return in each grid cell, smoothed
    with a 3 x 3 minimum so a cell covered by a car or a wall still finds the ground nearby."""
    xy = points[:, :2]
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / cell).astype(int)
    shape = ij.max(axis=0) + 1
    grid = np.full(shape, np.inf)
    np.minimum.at(grid, (ij[:, 0], ij[:, 1]), points[:, 2])
    grid = minimum_filter(grid, size=3, mode="nearest")
    known = np.isfinite(grid)
    if not known.all():
        ki = np.argwhere(known)
        ui = np.argwhere(~known)
        grid[~known] = griddata(ki, grid[known], ui, method="nearest")
    return np.asarray(points[:, 2] - grid[ij[:, 0], ij[:, 1]], dtype=np.float32)


def shape_features(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Per-point eigenvalue features of the k nearest neighbours: linearity, planarity, scattering,
    verticality, and the local height range. The standard handcrafted set for point clouds."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    nb = points[idx]  # n x k x 3
    centred = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k
    w, v = np.linalg.eigh(cov)  # ascending
    w = np.clip(w, 1e-12, None)
    l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
    normal = v[:, :, 0]
    feats = np.stack(
        [
            (l1 - l2) / l1,  # linearity: poles, edges
            (l2 - l3) / l1,  # planarity: ground, facades
            l3 / l1,  # scattering: vegetation
            1 - np.abs(normal[:, 2]),  # verticality: walls and poles against ground and roofs
            nb[:, :, 2].max(axis=1) - nb[:, :, 2].min(axis=1),
        ],
        axis=1,
    )
    return np.asarray(feats, dtype=np.float32)


@dataclass
class Block:
    scene_seed: int
    index: np.ndarray  # indices into the scene's points
    x0: float


def blocks(scene: Scene) -> list[Block]:
    out = []
    x = float(scene.points[:, 0].min())
    end = float(scene.points[:, 0].max())
    while x < end:
        idx = np.where((scene.points[:, 0] >= x) & (scene.points[:, 0] < x + BLOCK))[0]
        if len(idx) > 100:
            out.append(Block(scene.seed, idx, x))
        x += BLOCK
    return out


def point_inputs(scene: Scene, hag: np.ndarray, block: Block) -> np.ndarray:
    """Model inputs for a block: position relative to the block (scaled to ~[-1, 1]), height above
    ground and intensity."""
    p = scene.points[block.index]
    centre = np.array([block.x0 + BLOCK / 2, 0.0, 0.0])
    rel = (p - centre) / np.array([BLOCK / 2, 25.0, 10.0])
    return np.concatenate([rel, hag[block.index, None] / 10.0, scene.intensity[block.index, None]], axis=1).astype(np.float32)
