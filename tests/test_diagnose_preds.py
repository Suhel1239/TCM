import io
import pickle
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_preds  # noqa: E402

CATS = [{"id": 1, "name": "yuji"}, {"id": 2, "name": "laogong"}]
COCO = {"categories": CATS,
        "images": [{"id": 1, "file_name": "a.png", "height": 100, "width": 100}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                        {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20]}]}


def pred(labels):
    return {1: {"bboxes": np.array([[10, 10, 30, 30], [50, 50, 70, 70]], float),
                "scores": np.array([0.9, 0.8]), "labels": np.array(labels), "ori_shape": (100, 100)}}


def test_correct_labels():
    r = diagnose_preds.diagnose(pred([0, 1]), COCO, ["yuji", "laogong"], out=io.StringIO())
    assert r["recall_any"] == r["recall_label"] == r["recall_top1"] == 1.0 and r["size_mismatch"] == 0


def test_swapped_labels_detected():
    out = io.StringIO()
    r = diagnose_preds.diagnose(pred([1, 0]), COCO, ["yuji", "laogong"], out=out)
    assert r["recall_any"] == 1.0 and r["recall_label"] == 0.0


def test_prompt_order_mismatch_flagged_and_pkl_loads(tmp_path):
    out = io.StringIO()
    diagnose_preds.diagnose(pred([0, 1]), COCO, ["laogong", "yuji"], out=out)
    assert "MISMATCH" in out.getvalue()
    p = tmp_path / "r.pkl"
    p.write_bytes(pickle.dumps([{"img_id": 1, "ori_shape": (90, 100), "pred_instances": {
        "bboxes": np.zeros((1, 4)), "scores": np.ones(1), "labels": np.zeros(1)}}]))
    r = diagnose_preds.diagnose(diagnose_preds.load_preds(str(p)), COCO, ["yuji", "laogong"], out=io.StringIO())
    assert r["size_mismatch"] == 1 and r["recall_any"] == 0.0
