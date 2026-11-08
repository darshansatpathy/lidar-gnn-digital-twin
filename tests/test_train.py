from __future__ import annotations

import json

import pytest
from conftest import MODEL, needs_model

from lidartwin import data, scenes, train


def test_a_short_training_run_learns(tmp_path):
    tr = data.prepare_all([0, 1, 2])
    va = data.prepare_all([1000])
    model, hist = train.train(tr, va, epochs=2, blocks_per_epoch=12, log=lambda m: None)
    # 24 steps: check that it learns (loss falls, accuracy well above the 1-in-5 chance), not how well.
    assert len(hist) == 2 and hist[-1]["loss"] < hist[0]["loss"] and hist[-1]["accuracy"] > 0.6
    path = tmp_path / "m.pt"
    train.save(model, path, {"epochs": 2})
    assert json.loads(path.with_suffix(".json").read_text())["parameters"] > 100_000
    again = train.load(path)
    assert set(again.state_dict()) == set(model.state_dict())


def test_tampered_weights_are_refused(tmp_path):
    model, _ = train.train(data.prepare_all([3]), data.prepare_all([1001]), epochs=1, blocks_per_epoch=2, log=lambda m: None)
    path = tmp_path / "m.pt"
    train.save(model, path, {})
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="manifest"):
        train.load(path)


def test_baseline_learns_from_handcrafted_features():
    clf = train.train_baseline(data.prepare_all([4, 5]), per_scene=2000)
    m = train.evaluate_baseline(clf, data.prepare_all([2000]))
    assert m["miou"] > 0.5


@needs_model
def test_committed_model_meets_the_gate():
    model = train.load(MODEL)
    m = train.evaluate(model, data.prepare_all(list(scenes.SPLITS["test"])[:3]))
    assert m["miou"] > 0.85
