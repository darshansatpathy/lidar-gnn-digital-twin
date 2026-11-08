"""ltw: simulate scans, train and evaluate the segmentation model, and build the street twin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lidartwin import __version__

MODEL = Path("models/dgcnn.pt")


def cmd_train(args: argparse.Namespace) -> int:
    from lidartwin import data, scenes, train

    tr = data.prepare_all(list(scenes.SPLITS["train"])[: args.scenes])
    va = data.prepare_all(list(scenes.SPLITS["val"])[: args.val_scenes])
    model, hist = train.train(tr, va, epochs=args.epochs, blocks_per_epoch=args.blocks)
    train.save(model, Path(args.out), {"epochs": args.epochs, "val_scenes": args.val_scenes, "val": max(hist, key=lambda h: h["miou"])})
    print(f"saved {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from lidartwin import data, report, scenes, train

    model = train.load(Path(args.model))
    manifest = json.loads(Path(args.model).with_suffix(".json").read_text())
    base = train.train_baseline(data.prepare_all(list(scenes.SPLITS["train"])[: args.baseline_scenes]))
    test = list(scenes.SPLITS["test"])[: args.scenes]
    r = report.run(model, base, test, test[: args.pairs], report.info(model, manifest) | {"test_scenes": len(test)})
    text = report.markdown(r)
    if args.report:
        Path(args.report).write_text(text)
    g = r.segmentation["test"]["DGCNN"]["miou"]
    b = r.segmentation["test"]["gradient boosting"]["miou"]
    print(
        f"test mIoU: DGCNN {g:.3f}, gradient boosting {b:.3f}; changes found {r.changes[0]}/{r.changes[0] + r.changes[2]}, false alarms {r.changes[1]}"
    )
    failures = []
    if g < args.min_miou:
        failures.append(f"DGCNN mIoU {g:.3f} below {args.min_miou}")
    if g <= b:
        failures.append("DGCNN does not beat the baseline")
    for f in failures:
        print("GATE:", f)
    return 1 if failures else 0


def cmd_twin(args: argparse.Namespace) -> int:
    from lidartwin import data, export, scenes, train, twin

    model = train.load(Path(args.model))
    before, after = scenes.make_pair(args.seed)
    pb, pa = data.prepare(before), data.prepare(after)
    lb, la = train.predict(model, pb, 2), train.predict(model, pa, 2)
    tb, ta = twin.extract(before, lb, pb.hag), twin.extract(after, la, pa.hag)
    d = twin.diff(tb, ta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export.write_geojson(out / "twin_before.geojson", tb)
    export.write_geojson(out / "twin_after.geojson", ta, d)
    export.write_ply(out / "scan_after_classified.ply", after, la)
    for c in d.changes:
        print(f"{c.change:8} {c.kind:8} at ({c.x:5.1f}, {c.y:5.1f}) {c.detail}")
    truth = twin.true_changes(before, after)
    tp, fp, _ = twin.change_score(d, truth)
    print(f"{tp} of {len(truth)} real changes found, {fp} false alarms; files in {out}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from lidartwin import export, scenes

    s = scenes.make(args.seed, tall=args.tall, density=args.density, noise=args.noise)
    export.write_ply(Path(args.out), s, s.labels)
    print(f"{len(s):,} points, {len(s.assets)} assets -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ltw", description=__doc__)
    p.add_argument("--version", action="version", version=f"ltw {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="simulate one street scan and write it as PLY (true classes)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tall", action="store_true")
    s.add_argument("--density", type=float, default=1.0)
    s.add_argument("--noise", type=float, default=0.015)
    s.add_argument("--out", default="scan.ply")
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("train", help="train the DGCNN")
    s.add_argument("--epochs", type=int, default=15)
    s.add_argument("--blocks", type=int, default=160, help="training blocks per epoch")
    s.add_argument("--scenes", type=int, default=120)
    s.add_argument("--val-scenes", type=int, default=8)
    s.add_argument("--out", default=str(MODEL))
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="segmentation and twin report, with a quality gate")
    s.add_argument("--model", default=str(MODEL))
    s.add_argument("--scenes", type=int, default=12)
    s.add_argument("--pairs", type=int, default=12)
    s.add_argument("--baseline-scenes", type=int, default=60)
    s.add_argument("--min-miou", type=float, default=0.85)
    s.add_argument("--report")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("twin", help="scan a street twice, build both twins and list the changes")
    s.add_argument("--model", default=str(MODEL))
    s.add_argument("--seed", type=int, default=2001)
    s.add_argument("--out", default="twin-out")
    s.set_defaults(fn=cmd_twin)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
