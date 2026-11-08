"""Training the DGCNN, and the baseline it has to beat."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from lidartwin import features, metrics
from lidartwin.data import Prepared
from lidartwin.model import DGCNN, parameters

POINTS = 4096


def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    if rng.random() < 0.5:
        x[:, 1] = -x[:, 1]  # the street looks the same from the other side
    a = rng.uniform(-0.08, 0.08)  # small heading change
    c, s = np.cos(a), np.sin(a)
    x[:, 0], x[:, 1] = c * x[:, 0] - s * x[:, 1] * 2.5, (s * x[:, 0] / 2.5 + c * x[:, 1])
    x[:, :3] += rng.normal(0, 0.002, (len(x), 3))
    x[:, 4] = np.clip(x[:, 4] * rng.uniform(0.85, 1.15), 0, 1)
    return x


def class_weights(data: list[Prepared]) -> torch.Tensor:
    counts = np.bincount(np.concatenate([d.scene.labels for d in data]), minlength=5).astype(float)
    w = 1 / np.sqrt(counts / counts.sum())
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def predict(model: DGCNN, prepared: Prepared, passes: int = 1, seed: int = 0) -> np.ndarray:
    """Labels for every point of a scene: each block is split into random subsets of POINTS (the
    density the model was trained at) and the subsets are classified; several passes vote."""
    model.eval()
    rng = np.random.default_rng(seed)
    n = len(prepared.scene)
    votes = np.zeros((n, 5), dtype=np.float32)
    with torch.no_grad():
        for _ in range(passes):
            for block, inputs in zip(prepared.blocks, prepared.inputs, strict=True):
                order = rng.permutation(len(block.index))
                parts = max(1, round(len(order) / POINTS))
                for part in np.array_split(order, parts):
                    logits = model(torch.from_numpy(inputs[part]))
                    votes[block.index[part]] += torch.softmax(logits, dim=1).numpy()
    return np.asarray(votes.argmax(axis=1))


def evaluate(model: DGCNN, data: list[Prepared], passes: int = 1) -> dict[str, float]:
    conf = sum(metrics.confusion(d.scene.labels, predict(model, d, passes, seed=i)) for i, d in enumerate(data))
    return metrics.summary(np.asarray(conf))


def train(
    train_data: list[Prepared],
    val_data: list[Prepared],
    epochs: int = 20,
    blocks_per_epoch: int = 160,
    lr: float = 2e-3,
    seed: int = 7,
    log: Callable[[str], None] = print,
) -> tuple[DGCNN, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DGCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * blocks_per_epoch)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_data))
    pool = [(d, i) for d in train_data for i in range(len(d.blocks))]
    history: list[dict[str, float]] = []
    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        start, total = time.time(), 0.0
        for j in rng.choice(len(pool), blocks_per_epoch, replace=False):
            d, i = pool[j]
            idx = rng.choice(len(d.blocks[i].index), POINTS, replace=len(d.blocks[i].index) < POINTS)
            x = torch.from_numpy(_augment(d.inputs[i][idx], rng))
            y = torch.from_numpy(d.scene.labels[d.blocks[i].index[idx]])
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        val = evaluate(model, val_data)
        history.append({"epoch": epoch + 1, "loss": total / blocks_per_epoch, **val})
        log(f"epoch {epoch + 1:2d}  loss {total / blocks_per_epoch:.3f}  val mIoU {val['miou']:.3f}  ({time.time() - start:.0f} s)")
        if val["miou"] > best:
            best, best_state = val["miou"], {k: v.clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def save(model: DGCNN, path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"file": path.name, "sha256": digest, "parameters": parameters(model)} | info
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")


def load(path: Path) -> DGCNN:
    """Load weights only (no pickled code), after checking the digest in the manifest."""
    manifest = json.loads(path.with_suffix(".json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(f"{path} does not match its manifest")
    model = DGCNN()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return model


# ---- baseline -----------------------------------------------------------------------------------


def baseline_features(d: Prepared) -> np.ndarray:
    shape = features.shape_features(d.scene.points)
    return np.concatenate([d.hag[:, None], d.scene.intensity[:, None], shape], axis=1)


def train_baseline(data: list[Prepared], per_scene: int = 4000, seed: int = 7) -> HistGradientBoostingClassifier:
    """Gradient boosting on handcrafted features (height above ground, intensity, eigenvalue shape
    features): the standard pre-deep-learning approach, and the bar the GNN has to clear."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for d in data:
        f = baseline_features(d)
        idx = rng.choice(len(f), min(per_scene, len(f)), replace=False)
        xs.append(f[idx])
        ys.append(d.scene.labels[idx])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed)
    return clf.fit(np.concatenate(xs), np.concatenate(ys))


def evaluate_baseline(clf: HistGradientBoostingClassifier, data: list[Prepared]) -> dict[str, float]:
    conf = sum(metrics.confusion(d.scene.labels, clf.predict(baseline_features(d))) for d in data)
    return metrics.summary(np.asarray(conf))
