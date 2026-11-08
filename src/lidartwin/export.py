"""Exports a GIS or a point-cloud viewer can open: assets as GeoJSON, classified points as PLY."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lidartwin.scenes import CLASSES, Scene
from lidartwin.twin import Diff, Twin

PALETTE = np.array([[150, 150, 150], [200, 70, 60], [60, 160, 60], [60, 70, 200], [220, 170, 40]], dtype=np.uint8)


def geojson(t: Twin, diff: Diff | None = None) -> dict[str, Any]:
    """Assets as points (poles, trees) and facade lines (buildings), in the scene's local metric
    coordinates (x along the street, y across it). Changes, when given, are features of their own."""
    feats: list[dict[str, Any]] = []
    for a in t.assets:
        if a.kind == "building":
            geom: dict[str, Any] = {
                "type": "LineString",
                "coordinates": [[round(a.extent[0], 2), round(a.y, 2)], [round(a.extent[1], 2), round(a.y, 2)]],
            }
        else:
            geom = {"type": "Point", "coordinates": [round(a.x, 2), round(a.y, 2)]}
        feats.append(
            {"type": "Feature", "geometry": geom, "properties": {"kind": a.kind, "height_m": round(a.height, 2), "points": a.points}}
        )
    for c in diff.changes if diff else []:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(c.x, 2), round(c.y, 2)]},
                "properties": {"kind": c.kind, "change": c.change, "detail": c.detail},
            }
        )
    return {"type": "FeatureCollection", "crs_note": "local metric coordinates of the scan", "features": feats}


def write_geojson(path: Path, t: Twin, diff: Diff | None = None) -> None:
    path.write_text(json.dumps(geojson(t, diff), indent=1) + "\n")


def write_ply(path: Path, scene: Scene, labels: np.ndarray) -> None:
    """Binary little-endian PLY with x, y, z, intensity, class and a colour per class."""
    n = len(scene)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment classes {' '.join(CLASSES)}\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\nproperty float intensity\n"
        "property uchar class\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("c", "u1"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = scene.points[:, 0], scene.points[:, 1], scene.points[:, 2]
    rec["i"], rec["c"] = scene.intensity, labels
    rgb = PALETTE[labels]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with path.open("wb") as fh:
        fh.write(header.encode())
        fh.write(rec.tobytes())


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(next(line.split()[-1] for line in raw[:end].decode().splitlines() if line.startswith("element vertex")))
    rec = np.frombuffer(
        raw[end:],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("c", "u1"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
        count=n,
    )
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1), rec["c"].astype(np.int64)
