"""Scenes prepared for learning: per-point inputs, labels and blocks, cached in memory."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lidartwin import features, scenes


@dataclass
class Prepared:
    scene: scenes.Scene
    hag: np.ndarray
    blocks: list[features.Block]
    inputs: list[np.ndarray]  # per block, points x 5


def prepare(scene: scenes.Scene) -> Prepared:
    hag = features.height_above_ground(scene.points)
    blks = features.blocks(scene)
    return Prepared(scene, hag, blks, [features.point_inputs(scene, hag, b) for b in blks])


def prepare_all(seeds: Iterable[int], tall: bool = False, density: float = 1.0, noise: float = 0.015) -> list[Prepared]:
    return [prepare(scenes.make(s, tall=tall, density=density, noise=noise)) for s in seeds]


CONDITIONS: dict[str, dict[str, Any]] = {
    "test": {},
    "half density": {"density": 0.5},
    "noisy (5 cm)": {"noise": 0.05},
    "taller buildings": {"tall": True},
}
