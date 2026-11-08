from __future__ import annotations

import json

import numpy as np

from lidartwin import export, features, scenes, twin


def _oracle(seed):
    s = scenes.make(seed)
    return s, twin.extract(s, s.labels, features.height_above_ground(s.points))


def test_true_labels_give_an_exact_twin():
    s, t = _oracle(2000)
    for k in twin.score(t, s):
        assert k.matched == k.registry == k.detected, k
        assert k.position_error_m < 0.3


def test_change_detection_with_true_labels():
    before, after = scenes.make_pair(2003)
    hb, ha = features.height_above_ground(before.points), features.height_above_ground(after.points)
    d = twin.diff(twin.extract(before, before.labels, hb), twin.extract(after, after.labels, ha))
    tp, fp, fn = twin.change_score(d, twin.true_changes(before, after))
    assert (tp, fp, fn) == (3, 0, 0)
    assert any(c.change == "modified" and "height" in c.detail for c in d.changes)


def test_vehicles_never_become_assets():
    _, t = _oracle(2004)
    assert {a.kind for a in t.assets} == {"pole", "tree", "building"}


def test_misclassified_noise_does_not_make_assets():
    s = scenes.make(2005)
    labels = s.labels.copy()
    rng = np.random.default_rng(0)
    flip = rng.choice(len(labels), 300, replace=False)  # scattered errors, as a model makes
    labels[flip] = scenes.POLE
    t = twin.extract(s, labels, features.height_above_ground(s.points))
    poles = next(k for k in twin.score(t, s) if k.kind == "pole")
    assert poles.detected == poles.registry


def test_exports(tmp_path):
    s, t = _oracle(2006)
    export.write_geojson(tmp_path / "t.geojson", t)
    doc = json.loads((tmp_path / "t.geojson").read_text())
    kinds = {f["properties"]["kind"] for f in doc["features"]}
    assert kinds == {"pole", "tree", "building"}
    export.write_ply(tmp_path / "s.ply", s, s.labels)
    pts, cls = export.read_ply(tmp_path / "s.ply")
    assert np.allclose(pts, s.points) and np.array_equal(cls, s.labels)
