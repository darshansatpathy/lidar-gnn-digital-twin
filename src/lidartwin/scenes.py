"""Procedural street scenes scanned by a simulated mobile LiDAR.

A scene is a 60 m street: road and sidewalks, buildings set back on both sides, street-light
poles, trees and parked vehicles. A scanner drives along the road and the simulation keeps the
properties that make real mobile-mapping data hard:

- point density falls with the square of the range, and nothing is returned beyond 40 m;
- surfaces facing away from the scanner return nothing (building roofs and backs are missing);
- tree canopies are porous, so points scatter through the crown instead of lying on a surface;
- ranges carry 1.5 cm noise, and intensity depends on material, incidence angle and range.

Every point carries its class and the asset it belongs to, and every scene comes with its asset
registry (what a city's GIS would hold), so both segmentation and the twin can be scored. A second
epoch of a scene applies known changes: a pole removed, a tree planted, a storey added to a
building, and every vehicle moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

CLASSES = ["ground", "building", "vegetation", "pole", "vehicle"]
GROUND, BUILDING, VEGETATION, POLE, VEHICLE = range(5)

LENGTH = 60.0
SENSOR_Y, SENSOR_Z = -1.5, 2.2
MAX_RANGE = 40.0
REF_RANGE = 8.0  # density at this range is the nominal density


@dataclass(frozen=True)
class Asset:
    kind: str  # building | pole | tree | vehicle
    id: str
    x: float
    y: float
    height: float
    size: tuple[float, float] = (0.0, 0.0)  # footprint along x and y (buildings, vehicles); crown radius (trees)


@dataclass
class Scene:
    points: np.ndarray  # n x 3, metres
    intensity: np.ndarray  # n, 0..1
    labels: np.ndarray  # n, class index
    instances: np.ndarray  # n, index into assets, -1 for ground
    assets: list[Asset]
    seed: int
    epoch: int = 0
    changes: list[tuple[str, str]] = field(default_factory=list)  # (change, asset id)

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class Layout:
    seed: int
    buildings: tuple[Asset, ...]
    poles: tuple[Asset, ...]
    trees: tuple[Asset, ...]
    vehicles: tuple[Asset, ...]
    slope: float
    building_reflectance: tuple[float, ...]


def layout(seed: int, tall: bool = False) -> Layout:
    """Where everything stands. `tall` draws taller buildings than training scenes ever have."""
    rng = np.random.default_rng(seed)
    buildings: list[Asset] = []
    poles: list[Asset] = []
    trees: list[Asset] = []
    for side in (-1, 1):
        x = rng.uniform(-2, 3)
        while x < LENGTH - 6:
            width = rng.uniform(8, 18)
            width = min(width, LENGTH + 2 - x)
            depth = rng.uniform(8, 14)
            setback = rng.uniform(9.0, 11.0)
            height = rng.uniform(14, 30) if tall else rng.uniform(6, 18)
            cy = side * (setback + depth / 2)
            buildings.append(Asset("building", f"B{side:+d}{len(buildings):02d}", x + width / 2, cy, height, (width, depth)))
            x += width + rng.uniform(1.5, 6)
        x = rng.uniform(3, 10)
        while x < LENGTH - 2:
            poles.append(Asset("pole", f"P{side:+d}{len(poles):02d}", x, side * rng.uniform(5.2, 5.8), rng.uniform(6.5, 9.0)))
            x += rng.uniform(14, 24)
        x = rng.uniform(1, 8)
        while x < LENGTH - 2:
            if all(abs(x - p.x) > 2.5 for p in poles if np.sign(p.y) == side):
                trees.append(
                    Asset(
                        "tree",
                        f"T{side:+d}{len(trees):02d}",
                        x,
                        side * rng.uniform(6.6, 7.6),
                        rng.uniform(5, 10),
                        (rng.uniform(1.6, 3.0), 0.0),
                    )
                )
            x += rng.uniform(7, 14)
    vehicles = _vehicles(rng, ())
    reflect = tuple(float(v) for v in rng.uniform(0.3, 0.7, size=len(buildings)))
    return Layout(seed, tuple(buildings), tuple(poles), tuple(trees), tuple(vehicles), float(rng.uniform(-0.02, 0.02)), reflect)


def _vehicles(rng: np.random.Generator, avoid: tuple[float, ...]) -> list[Asset]:
    out: list[Asset] = []
    for side in (-1, 1):
        x = rng.uniform(0, 8)
        while x < LENGTH - 5:
            if rng.random() < 0.55:
                length, width = rng.uniform(4.0, 5.2), rng.uniform(1.7, 2.0)
                out.append(
                    Asset("vehicle", f"V{side:+d}{len(out):02d}", x + length / 2, side * 2.9, rng.uniform(1.4, 1.9), (length, width))
                )
                x += length + rng.uniform(1.0, 3.0)
            else:
                x += rng.uniform(4, 9)
    return out


def changed(base: Layout, seed: int) -> tuple[Layout, list[tuple[str, str]]]:
    """The same street some months later, with the changes listed."""
    rng = np.random.default_rng(seed + 10_000)
    poles = list(base.poles)
    removed = poles.pop(int(rng.integers(len(poles))))
    changes = [("removed", removed.id)]
    trees = list(base.trees)
    side = int(rng.choice([-1, 1]))
    for _ in range(50):
        x = float(rng.uniform(4, LENGTH - 4))
        if all(abs(x - t.x) > 4 for t in trees if np.sign(t.y) == side) and all(abs(x - p.x) > 3 for p in poles if np.sign(p.y) == side):
            new = Asset("tree", f"T{side:+d}new", x, side * 7.1, float(rng.uniform(5, 8)), (float(rng.uniform(1.6, 2.4)), 0.0))
            trees.append(new)
            changes.append(("added", new.id))
            break
    buildings = list(base.buildings)
    i = int(rng.integers(len(buildings)))
    buildings[i] = replace(buildings[i], height=buildings[i].height + 3.2)
    changes.append(("modified", buildings[i].id))
    return replace(base, poles=tuple(poles), trees=tuple(trees), buildings=tuple(buildings), vehicles=tuple(_vehicles(rng, ()))), changes


# ---- the scan -----------------------------------------------------------------------------------


def _sensor_vectors(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors from each point to the closest scanner position on its trajectory, and ranges."""
    sx = np.clip(p[:, 0], 0, LENGTH)
    s = np.stack([sx, np.full(len(p), SENSOR_Y), np.full(len(p), SENSOR_Z)], axis=1)
    v = s - p
    r = np.linalg.norm(v, axis=1)
    return v / np.maximum(r, 1e-6)[:, None], r


def _keep(rng: np.random.Generator, p: np.ndarray, normals: np.ndarray | None, density: float) -> np.ndarray:
    v, r = _sensor_vectors(p)
    prob = np.minimum(1.0, (REF_RANGE / np.maximum(r, 0.5)) ** 2) * density
    keep = (rng.random(len(p)) < prob) & (r < MAX_RANGE)
    if normals is not None:
        keep &= np.einsum("ij,ij->i", normals, v) > 0.05
    return np.asarray(keep, dtype=bool)


def _quad(rng: np.random.Generator, origin: np.ndarray, u: np.ndarray, w: np.ndarray, per_m2: float) -> tuple[np.ndarray, np.ndarray]:
    area = float(np.linalg.norm(np.cross(u, w)))
    n = rng.poisson(area * per_m2)
    a, b = rng.random((n, 1)), rng.random((n, 1))
    pts = origin + a * u + b * w
    normal = np.cross(u, w) / max(area, 1e-9)
    return pts, np.repeat(normal[None, :], n, axis=0)


def _box(
    rng: np.random.Generator, cx: float, cy: float, z0: float, sx: float, sy: float, h: float, per_m2: float, roof: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = cx - sx / 2, cy - sy / 2
    o = np.array([x0, y0, z0])
    X, Y, Z = np.array([sx, 0, 0]), np.array([0, sy, 0]), np.array([0, 0, h])
    faces = [  # (origin, u, w) with u x w pointing out of the box
        (o, X, Z),  # y = y0, outward -y
        (o + Y, Z, X),  # y = y0 + sy, outward +y
        (o, Z, Y),  # x = x0, outward -x
        (o + X, Y, Z),  # x = x0 + sx, outward +x
    ]
    if roof:
        faces.append((o + Z, X, Y))  # roof, outward +z
    pts, nrm = zip(*(_quad(rng, *f, per_m2) for f in faces), strict=True)
    return np.concatenate(pts), np.concatenate(nrm)


def _cylinder(
    rng: np.random.Generator, x: float, y: float, z0: float, radius: float, h: float, per_m2: float
) -> tuple[np.ndarray, np.ndarray]:
    n = rng.poisson(2 * np.pi * radius * h * per_m2)
    t = rng.uniform(0, 2 * np.pi, n)
    z = z0 + rng.uniform(0, h, n)
    nrm = np.stack([np.cos(t), np.sin(t), np.zeros(n)], axis=1)
    return np.stack([x + radius * np.cos(t), y + radius * np.sin(t), z], axis=1), nrm


def ground_z(lay: Layout, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(
        lay.slope * x + 0.04 * np.sin(x / 7.0) * np.cos(y / 9.0) + np.where(np.abs(y) > 4.5, 0.15, 0.0), dtype=float
    )  # sidewalks 15 cm up


def scan(
    lay: Layout,
    density: float = 1.0,
    noise: float = 0.015,
    seed_offset: int = 0,
    epoch: int = 0,
    changes: list[tuple[str, str]] | None = None,
) -> Scene:
    rng = np.random.default_rng(lay.seed * 7919 + seed_offset + 31 * epoch)
    parts: list[tuple[np.ndarray, np.ndarray | None, int, int, float]] = []  # points, normals, class, asset index, reflectance
    assets: list[Asset] = []

    # Ground: a dense grid, flagged as ground everywhere (road, sidewalk, the strip before buildings).
    n = int((LENGTH + 4) * 50 * 30 * density)
    gx, gy = rng.uniform(-2, LENGTH + 2, n), rng.uniform(-25, 25, n)
    hidden = np.zeros(n, dtype=bool)
    for bld in lay.buildings:  # ground under and behind a building is out of sight
        front = abs(bld.y) - bld.size[1] / 2
        hidden |= (np.abs(gx - bld.x) <= bld.size[0] / 2) & (np.sign(gy) == np.sign(bld.y)) & (np.abs(gy) >= front)
    gx, gy = gx[~hidden], gy[~hidden]
    n = len(gx)
    g = np.stack([gx, gy, ground_z(lay, gx, gy)], axis=1)
    parts.append((g, np.tile([0.0, 0.0, 1.0], (n, 1)), GROUND, -1, 0.0))

    for b, refl in zip(lay.buildings, lay.building_reflectance, strict=True):
        assets.append(b)
        z0 = float(ground_z(lay, np.array([b.x]), np.array([b.y]))[0]) + 0.15
        pts, nrm = _box(rng, b.x, b.y, z0, b.size[0], b.size[1], b.height, 60 * density)
        parts.append((pts, nrm, BUILDING, len(assets) - 1, refl))
    for p in lay.poles:
        assets.append(p)
        z0 = float(ground_z(lay, np.array([p.x]), np.array([p.y]))[0])
        pts, nrm = _cylinder(rng, p.x, p.y, z0, 0.12, p.height, 220 * density)
        arm_n = int(120 * density)
        t = rng.uniform(0, 1.8, arm_n)
        arm = np.stack(
            [np.full(arm_n, p.x), p.y - np.sign(p.y) * t, np.full(arm_n, z0 + p.height - 0.2) + rng.normal(0, 0.04, arm_n)], axis=1
        )
        parts.append(
            (np.concatenate([pts, arm]), np.concatenate([nrm, np.tile([0.0, 0.0, -1.0], (arm_n, 1))]), POLE, len(assets) - 1, 0.75)
        )
    for tree in lay.trees:
        assets.append(tree)
        z0 = float(ground_z(lay, np.array([tree.x]), np.array([tree.y]))[0])
        trunk_h = 0.35 * tree.height
        trunk, tn = _cylinder(rng, tree.x, tree.y, z0, 0.16, trunk_h, 250 * density)
        crown_r = tree.size[0]
        m = int(4 * np.pi * crown_r**2 * 18 * density)
        d = rng.normal(size=(m, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rad = crown_r * (1 - 0.45 * rng.random(m) ** 2)  # porous: returns from inside the crown too
        cz = z0 + trunk_h + (tree.height - trunk_h) / 2
        crown = np.array([tree.x, tree.y, cz]) + d * rad[:, None] * np.array([1.0, 1.0, (tree.height - trunk_h) / (2 * crown_r)])
        parts.append((trunk, tn, VEGETATION, len(assets) - 1, 0.45))
        parts.append((crown, None, VEGETATION, len(assets) - 1, 0.55))
    for veh in lay.vehicles:
        assets.append(veh)
        z0 = float(ground_z(lay, np.array([veh.x]), np.array([veh.y]))[0]) + 0.25
        pts, nrm = _box(rng, veh.x, veh.y, z0, veh.size[0], veh.size[1], veh.height - 0.25, 90 * density)
        parts.append((pts, nrm, VEHICLE, len(assets) - 1, float(rng.uniform(0.4, 0.9))))

    pts_l, inten_l, lab_l, inst_l = [], [], [], []
    for pts, part_normals, cls, inst, refl in parts:
        if len(pts) == 0:
            continue
        keep = _keep(rng, pts, part_normals, 1.0)
        pts = pts[keep]
        nrm_k = part_normals[keep] if part_normals is not None else None
        v, r = _sensor_vectors(pts)
        pts = pts - v * rng.normal(0, noise, len(pts))[:, None]  # range noise along the beam
        if cls == GROUND:
            road = np.abs(pts[:, 1]) <= 4.5
            base = np.where(road, 0.15, 0.32)
        else:
            base = np.full(len(pts), refl)
        cos = np.abs(np.einsum("ij,ij->i", nrm_k, v)) if nrm_k is not None else np.full(len(pts), 0.6)
        inten = np.clip(base * (0.4 + 0.6 * cos) * np.exp(-r / 120) + rng.normal(0, 0.03, len(pts)), 0, 1)
        pts_l.append(pts)
        inten_l.append(inten)
        lab_l.append(np.full(len(pts), cls))
        inst_l.append(np.full(len(pts), inst))
    points = np.concatenate(pts_l)
    inside = (points[:, 0] >= -2) & (points[:, 0] <= LENGTH + 2) & (np.abs(points[:, 1]) <= 25)
    return Scene(
        points=points[inside].astype(np.float32),
        intensity=np.concatenate(inten_l)[inside].astype(np.float32),
        labels=np.concatenate(lab_l)[inside].astype(np.int64),
        instances=np.concatenate(inst_l)[inside].astype(np.int64),
        assets=assets,
        seed=lay.seed,
        epoch=epoch,
        changes=changes or [],
    )


def make(seed: int, tall: bool = False, density: float = 1.0, noise: float = 0.015) -> Scene:
    return scan(layout(seed, tall=tall), density=density, noise=noise)


def make_pair(seed: int) -> tuple[Scene, Scene]:
    """Two epochs of one street: before and after the changes."""
    base = layout(seed)
    after, changes = changed(base, seed)
    return scan(base, epoch=0), scan(after, epoch=1, changes=changes)


SPLITS = {"train": range(0, 120), "val": range(1000, 1020), "test": range(2000, 2030)}
