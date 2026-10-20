import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ood_eval  # noqa: E402

CATS = [{"id": 0, "name": "annotator-2-d2"}, {"id": 1, "name": "laogong"}, {"id": 2, "name": "shaofu"},
        {"id": 3, "name": "yuji"}, {"id": 4, "name": "zhongchong"}]


def make_gt(n_img=12, seed=0):
    rng = np.random.default_rng(seed)
    images, anns = [], []
    for i in range(n_img):
        images.append({"id": i, "file_name": f"img_{i}.jpg", "width": 1000, "height": 1000,
                       "extra": {"user_tags": ["external_image_set_40"], "name": f"img_{i}.jpg"}})
        for cid in (1, 2, 3, 4):
            x, y = rng.uniform(50, 900, 2)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid,
                         "bbox": [float(x), float(y), 40.0, 50.0], "iscrowd": 0, "area": 2000.0})
    return {"images": images, "annotations": anns, "categories": CATS}


def preds_from(gt, shift=0.0, drop_every=0, fp=False, seed=1):
    rng = np.random.default_rng(seed)
    res = []
    for k, a in enumerate(gt["annotations"]):
        if drop_every and k % drop_every == 0:
            continue
        x, y, w, h = a["bbox"]
        res.append({"image_id": a["image_id"], "category_id": a["category_id"],
                    "bbox": [x + shift * w, y, w, h], "score": float(rng.uniform(0.3, 0.99))})
        if fp:
            res.append({"image_id": a["image_id"], "category_id": a["category_id"],
                        "bbox": [x + 300, y + 300, w, h], "score": float(rng.uniform(0.05, 0.6))})
    return {"model": "m", "results": res}


def run(tmp_path, preds, extra=()):
    gt = make_gt()
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    args = ["--ann", str(tmp_path / "gt.json"), "--out", str(tmp_path / "out"), "--bootstrap", "50"]
    for name, p in preds.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(p(gt)))
        args += ["--pred", f"{name}={tmp_path / f'{name}.json'}"]
    assert ood_eval.main(args + list(extra)) == 0
    rows = {r["model"]: r for r in __import__("csv").DictReader(open(tmp_path / "out" / "summary.csv"))}
    return gt, rows


def test_perfect_and_degraded(tmp_path):
    _, rows = run(tmp_path, {"perfect": preds_from, "bad": lambda g: preds_from(g, 0.3, 4, True)},
                  ["--ref", "perfect"])
    p, b = rows["perfect"], rows["bad"]
    assert float(p["mAP50"]) == pytest.approx(1.0) and float(p["F1"]) == pytest.approx(1.0)
    assert float(b["mAP50"]) < 0.9 and float(b["F1"]) < 0.9 and int(b["FN"]) > 0 and int(b["FP"]) > 0
    assert "Difference vs perfect" in (tmp_path / "out" / "summary.md").read_text()


def test_matches_pycocotools(tmp_path):
    coco_mod = pytest.importorskip("pycocotools.coco")
    ce = pytest.importorskip("pycocotools.cocoeval")
    gt = make_gt()
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    preds = preds_from(gt, 0.2, 5, True)["results"]
    cg = coco_mod.COCO(str(tmp_path / "gt.json"))
    e = ce.COCOeval(cg, cg.loadRes(preds), "bbox")
    e.params.catIds = [1, 2, 3, 4]
    e.evaluate()
    e.accumulate()
    e.summarize()
    _, name_of, g = ood_eval.load_gt(str(tmp_path / "gt.json"), ood_eval.DEFAULT_CLASSES)
    (tmp_path / "p.json").write_text(json.dumps(preds))
    pr = ood_eval.load_pred(str(tmp_path / "p.json"), name_of, ood_eval.DEFAULT_CLASSES)
    rec = ood_eval.per_image_records([im["id"] for im in gt["images"]], g, pr, ood_eval.DEFAULT_CLASSES)
    m = ood_eval.metrics(rec, list(rec), ood_eval.DEFAULT_CLASSES, 0.15)
    assert m["macro"]["AP50"] == pytest.approx(e.stats[1], abs=0.01)
    assert m["macro"]["AP50_95"] == pytest.approx(e.stats[0], abs=0.01)
