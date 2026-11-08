from __future__ import annotations

from conftest import MODEL, needs_model

from lidartwin.cli import main


def test_scan(tmp_path, capsys):
    assert main(["scan", "--seed", "3", "--out", str(tmp_path / "s.ply")]) == 0
    assert "points" in capsys.readouterr().out and (tmp_path / "s.ply").stat().st_size > 1_000_000


@needs_model
def test_twin(tmp_path, capsys):
    assert main(["twin", "--model", str(MODEL), "--seed", "2007", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "real changes found" in out
    assert (tmp_path / "twin_after.geojson").exists() and (tmp_path / "scan_after_classified.ply").exists()


def test_train_and_evaluate_small(tmp_path, capsys):
    m = tmp_path / "m.pt"
    assert main(["train", "--epochs", "1", "--blocks", "4", "--scenes", "2", "--val-scenes", "1", "--out", str(m)]) == 0
    code = main(
        ["evaluate", "--model", str(m), "--scenes", "1", "--pairs", "1", "--baseline-scenes", "2", "--report", str(tmp_path / "r.md")]
    )
    assert code == 1  # a model trained for four steps fails the gate
    assert "GATE" in capsys.readouterr().out
    assert "Evaluation report" in (tmp_path / "r.md").read_text()
