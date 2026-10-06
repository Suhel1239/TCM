import io
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_palm_classifier as ev  # noqa: E402


def test_binary_metrics_and_confusion():
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([1, 1, 0, 1, 0])
    assert ev.confusion(y, p) == {"tp": 2, "fn": 1, "fp": 1, "tn": 1}
    m = ev.binary_metrics(y, p)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["palm_recall"] == pytest.approx(2 / 3) and m["palm_precision"] == pytest.approx(2 / 3)
    assert m["nonpalm_recall"] == pytest.approx(0.5)


def test_auc_and_ap():
    y = np.array([0, 0, 1, 1])
    assert ev.roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert ev.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ev.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert ev.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_report_runs_with_per_video():
    paths = ["palm/003_1_frame_010.png", "palm/003_1_frame_020.png", "non_palm/004_2_frame_000.png",
             "non_palm/004_2_frame_005.png"]
    y = np.array([1, 1, 0, 0])
    out = io.StringIO()
    r = ev.report(paths, y, np.array([0.9, 0.3, 0.2, 0.6]), 0.5, n_boot=50, out=out)
    assert r["confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    text = out.getvalue()
    assert "003_1" in text and "palm frames LOST" in text
    assert ev.video_of("x/003_1_frame_010.png") == "003_1"
