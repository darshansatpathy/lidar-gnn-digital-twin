from __future__ import annotations

import numpy as np

from lidartwin import scenes


def test_scenes_are_deterministic_and_labelled():
    a, b = scenes.make(11), scenes.make(11)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.labels, b.labels)
    assert set(np.unique(a.labels)) == set(range(len(scenes.CLASSES)))
    assert len(a) > 50_000
    assert (a.instances[a.labels == scenes.GROUND] == -1).all()
    assert (a.instances[a.labels != scenes.GROUND] >= 0).all()


def test_scanner_physics():
    s = scenes.make(12)
    # Density falls with range: more points near the scanner's lane than at the far edge.
    near = np.abs(s.points[:, 1] - scenes.SENSOR_Y) < 5
    far = np.abs(s.points[:, 1]) > 18
    assert near.sum() / 10 > 3 * far.sum() / 14
    # Roofs face away from a street-level scanner: almost nothing is returned from them.
    lay = scenes.layout(12)
    roof_hits = 0
    for b in lay.buildings:
        on_roof = (np.abs(s.points[:, 0] - b.x) < b.size[0] / 2 - 0.5) & (np.abs(s.points[:, 1] - b.y) < b.size[1] / 2 - 0.5)
        roof_hits += int((on_roof & (s.points[:, 2] > b.height - 0.2)).sum())
    assert roof_hits < 50
    assert s.intensity.min() >= 0 and s.intensity.max() <= 1


def test_conditions_change_the_scan():
    base, sparse, tall = scenes.make(13), scenes.make(13, density=0.5), scenes.make(13, tall=True)
    assert 0.35 < len(sparse) / len(base) < 0.65
    assert max(a.height for a in tall.assets if a.kind == "building") > 18


def test_second_epoch_applies_the_listed_changes():
    before, after = scenes.make_pair(14)
    kinds = {c for c, _ in after.changes}
    assert kinds == {"removed", "added", "modified"}
    ids_before = {a.id for a in before.assets}
    ids_after = {a.id for a in after.assets}
    removed = next(i for c, i in after.changes if c == "removed")
    added = next(i for c, i in after.changes if c == "added")
    assert removed in ids_before and removed not in ids_after
    assert added in ids_after and added not in ids_before
    vb = sorted(a.x for a in before.assets if a.kind == "vehicle")
    va = sorted(a.x for a in after.assets if a.kind == "vehicle")
    assert vb != va  # every vehicle moved


def test_splits_do_not_overlap():
    sets = [set(v) for v in scenes.SPLITS.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
