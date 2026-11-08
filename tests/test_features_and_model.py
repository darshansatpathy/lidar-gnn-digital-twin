from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from lidartwin import data, features, metrics, scenes
from lidartwin.model import DGCNN, knn_edges


def test_height_above_ground_separates_classes():
    s = scenes.make(21)
    h = features.height_above_ground(s.points)
    assert np.median(h[s.labels == scenes.GROUND]) < 0.15
    assert np.median(h[s.labels == scenes.VEHICLE]) > 0.8
    assert np.median(h[s.labels == scenes.BUILDING]) > 2


def test_shape_features_describe_geometry():
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(0, 5, (500, 2)), np.zeros(500)]
    line = np.c_[np.zeros(500), np.zeros(500), rng.uniform(0, 5, 500)] + rng.normal(0, 0.003, (500, 3))
    fp, fl = features.shape_features(plane), features.shape_features(line)
    assert np.median(fp[:, 1]) > 0.5 and np.median(fp[:, 3]) < 0.1  # planar, horizontal
    assert np.median(fl[:, 0]) > 0.8  # linear


def test_knn_matches_a_kd_tree():
    x = torch.rand(3000, 3)
    e = knn_edges(x, k=8, chunk=700)
    assert e.shape == (2, 3000 * 8)
    _, idx = cKDTree(x.numpy()).query(x.numpy(), k=9)
    ours = e[0].reshape(3000, 8).numpy()
    assert all(set(ours[i]) == set(idx[i, 1:]) for i in range(0, 3000, 97))
    assert (e[0] != e[1]).all()


def test_model_shapes_and_permutation_equivariance():
    torch.manual_seed(0)
    m = DGCNN().eval()
    x = torch.rand(600, 5)
    perm = torch.randperm(600)
    with torch.no_grad():
        y = m(x)
        yp = m(x[perm])
    assert y.shape == (600, 5)
    assert torch.allclose(y[perm], yp, atol=1e-4)


def test_blocks_cover_the_scene():
    p = data.prepare(scenes.make(22))
    covered = np.concatenate([b.index for b in p.blocks])
    assert len(np.unique(covered)) == len(p.scene) and len(covered) == len(p.scene)
    assert all(i.shape[1] == 5 for i in p.inputs)
    assert all(np.abs(i[:, :2]).max() <= 1.2 for i in p.inputs)


def test_metrics():
    truth = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    pred = np.array([0, 1, 1, 1, 2, 0, 3, 4])
    s = metrics.summary(metrics.confusion(truth, pred))
    assert s["accuracy"] == 6 / 8
    assert s["iou_ground"] == 1 / 3 and s["iou_pole"] == 1.0
