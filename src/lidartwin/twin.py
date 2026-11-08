"""The digital twin: street assets extracted from classified points, checked against the registry,
and compared between two scans to find what changed.

From the predicted classes, poles and trees are clustered in plan (DBSCAN) into individual
assets with a position and a height; building points are clustered into facades with their
extent along the street and the highest point seen. Vehicles are deliberately left out: they are
transient, and a twin that reported every parked car as a change would be useless.

Assets are matched to the registry, or to the other epoch, by position (Hungarian assignment
within a distance limit). A change is an asset that appears or disappears, or a building whose
observed height moves by more than a storey's worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from lidartwin.scenes import BUILDING, POLE, VEGETATION, Asset, Scene

MATCH_M = {"pole": 1.5, "tree": 2.0, "building": 4.0}
MIN_POINTS = {"pole": 25, "tree": 80, "building": 300}
HEIGHT_CHANGE_M = 2.0


@dataclass(frozen=True)
class Detected:
    kind: str
    x: float
    y: float
    height: float
    points: int
    extent: tuple[float, float] = (0.0, 0.0)  # along the street, buildings only


@dataclass
class Twin:
    assets: list[Detected]

    def of(self, kind: str) -> list[Detected]:
        return [a for a in self.assets if a.kind == kind]


def extract(scene: Scene, labels: np.ndarray, hag: np.ndarray) -> Twin:
    out: list[Detected] = _trees(scene, labels, hag)
    for kind, cls, eps in (("pole", POLE, 0.6), ("building", BUILDING, 1.5)):
        idx = np.where(labels == cls)[0]
        if len(idx) < MIN_POINTS[kind]:
            continue
        xy = scene.points[idx, :2]
        cluster = DBSCAN(eps=eps, min_samples=8).fit_predict(xy)
        for c in set(cluster) - {-1}:
            members = idx[cluster == c]
            if len(members) < MIN_POINTS[kind]:
                continue
            p = scene.points[members]
            h = float(np.percentile(hag[members], 99))
            if kind == "pole" and h < 3.0:
                continue  # a fragment, not a pole
            if kind == "building":
                out.append(
                    Detected(
                        kind,
                        float(np.median(p[:, 0])),
                        float(np.median(p[:, 1])),
                        h,
                        len(members),
                        (float(p[:, 0].min()), float(p[:, 0].max())),
                    )
                )
            else:
                # Pole and trunk positions: the lowest points locate the base, not the lamp arm or the crown.
                low = p[hag[members] < np.percentile(hag[members], 20)]
                out.append(Detected(kind, float(np.median(low[:, 0])), float(np.median(low[:, 1])), h, len(members)))
    return Twin(out)


def _trees(scene: Scene, labels: np.ndarray, hag: np.ndarray) -> list[Detected]:
    """Trees are found by their trunks: crowns of neighbouring trees overlap in plan and would merge.
    Each vegetation point is then given to the nearest trunk for the tree's height."""
    veg = np.where(labels == VEGETATION)[0]
    trunk = veg[(hag[veg] > 0.3) & (hag[veg] < 1.6)]
    if len(trunk) < 10:
        return []
    cluster = DBSCAN(eps=0.5, min_samples=6).fit_predict(scene.points[trunk, :2])
    bases = []
    for c in set(cluster) - {-1}:
        members = trunk[cluster == c]
        if len(members) >= 10:
            bases.append(np.median(scene.points[members, :2], axis=0))
    if not bases:
        return []
    centres = np.asarray(bases)
    d = np.linalg.norm(scene.points[veg, None, :2] - centres[None, :, :], axis=2)
    owner = d.argmin(axis=1)
    near = d.min(axis=1) < 4.0
    out = []
    for i, (x, y) in enumerate(centres):
        members = veg[(owner == i) & near]
        if len(members) >= MIN_POINTS["tree"]:
            out.append(Detected("tree", float(x), float(y), float(np.percentile(hag[members], 99)), len(members)))
    return out


def _match(a: list[tuple[float, float]], b: list[tuple[float, float]], limit: float) -> list[tuple[int, int]]:
    if not a or not b:
        return []
    d = np.linalg.norm(np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(np.where(d <= limit, d, 1e6))
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if d[i, j] <= limit]


def _registry_xy(asset: Asset) -> tuple[float, float]:
    if asset.kind == "building":  # the facade the scanner sees, not the footprint centre
        return asset.x, float(np.sign(asset.y) * (abs(asset.y) - asset.size[1] / 2))
    return asset.x, asset.y


@dataclass
class KindScore:
    kind: str
    registry: int
    detected: int
    matched: int
    position_error_m: float
    height_error_m: float

    @property
    def precision(self) -> float:
        return self.matched / self.detected if self.detected else float("nan")

    @property
    def recall(self) -> float:
        return self.matched / self.registry if self.registry else float("nan")


def score(twin: Twin, scene: Scene) -> list[KindScore]:
    """Detected assets against the registry, by kind. Building heights are compared only for
    facades fully in range; heights of tall buildings are capped by what the scanner can see."""
    out = []
    for kind in ("pole", "tree", "building"):
        reg = [a for a in scene.assets if a.kind == kind]
        det = twin.of(kind)
        pairs = _match([_registry_xy(a) for a in reg], [(d.x, d.y) for d in det], MATCH_M[kind])
        pos = [float(np.hypot(*(np.array(_registry_xy(reg[i])) - (det[j].x, det[j].y)))) for i, j in pairs]
        hgt = [abs(reg[i].height - det[j].height) for i, j in pairs]
        out.append(
            KindScore(
                kind,
                len(reg),
                len(det),
                len(pairs),
                float(np.mean(pos)) if pos else float("nan"),
                float(np.median(hgt)) if hgt else float("nan"),
            )
        )
    return out


@dataclass
class Change:
    kind: str
    change: str  # added | removed | modified
    x: float
    y: float
    detail: str = ""


@dataclass
class Diff:
    changes: list[Change] = field(default_factory=list)


def diff(before: Twin, after: Twin) -> Diff:
    out = Diff()
    for kind in ("pole", "tree", "building"):
        a, b = before.of(kind), after.of(kind)
        pairs = _match([(d.x, d.y) for d in a], [(d.x, d.y) for d in b], MATCH_M[kind])
        ma, mb = {i for i, _ in pairs}, {j for _, j in pairs}
        out.changes += [Change(kind, "removed", d.x, d.y) for i, d in enumerate(a) if i not in ma]
        out.changes += [Change(kind, "added", d.x, d.y) for j, d in enumerate(b) if j not in mb]
        if kind == "building":
            for i, j in pairs:
                dh = b[j].height - a[i].height
                if abs(dh) > HEIGHT_CHANGE_M:
                    out.changes.append(Change(kind, "modified", b[j].x, b[j].y, f"height {a[i].height:.1f} m -> {b[j].height:.1f} m"))
    return out


def true_changes(before: Scene, after: Scene) -> list[Change]:
    """The changes the generator applied, located, in the same terms as `diff`."""
    by_id = {a.id: a for a in before.assets} | {a.id: a for a in after.assets}
    kinds = {"P": "pole", "T": "tree", "B": "building"}
    out = []
    for change, aid in after.changes:
        a = by_id[aid]
        x, y = _registry_xy(a)
        out.append(Change(kinds[aid[0]], change, x, y))
    return out


def change_score(found: Diff, truth: list[Change]) -> tuple[int, int, int]:
    """(true positives, false positives, missed) for changes of the same kind and type within the
    kind's match distance."""
    used: set[int] = set()
    tp = 0
    for t in truth:
        for i, f in enumerate(found.changes):
            if i in used or f.kind != t.kind or f.change != t.change:
                continue
            if np.hypot(f.x - t.x, f.y - t.y) <= MATCH_M[t.kind] * (3 if t.kind == "building" else 1.5):
                used.add(i)
                tp += 1
                break
    return tp, len(found.changes) - len(used), len(truth) - tp
