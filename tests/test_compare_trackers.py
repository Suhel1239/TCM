import csv
import io
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_trackers as ct  # noqa: E402

BASE = {"yuji": (500, 900), "laogong": (520, 640), "zhongchong": (510, 230), "shaofu": (680, 700)}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "point", "x1", "y1", "x2", "y2", "score"])
        w.writerows(rows)


def track_rows(video, n, offset=0.0, noise=0.0, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n):
        for pt, (cx, cy) in BASE.items():
            if (pt, f) in drop:
                continue
            dx, dy = rng.normal(0, noise, 2) + offset
            rows.append([video, f, pt, cx - 20 + dx, cy - 25 + dy, cx + 20 + dx, cy + 25 + dy, 0.9])
    return rows


def test_perfect_vs_noisy_tracker(tmp_path):
    for name, kw in (("good", {}), ("bad", {"offset": 30, "noise": 8, "drop": {("yuji", 5)}})):
        rows = []
        for v in ("v1", "v2", "v3"):
            rows += track_rows(v, 20, **kw)
        write_csv(tmp_path / f"{name}.csv", rows)
    coco = {"categories": [{"id": i + 1, "name": n} for i, n in enumerate(BASE)], "images": [], "annotations": []}
    for v in ("v1", "v2", "v3"):
        for f in (0, 10):
            iid = len(coco["images"])
            coco["images"].append({"id": iid, "file_name": f"{v}_frame_{f:03d}_png.rf.abc.jpg"})
            for i, (cx, cy) in enumerate(BASE.values()):
                coco["annotations"].append({"image_id": iid, "category_id": i + 1, "bbox": [cx - 20, cy - 25, 40, 50]})
    (tmp_path / "gt.json").write_text(json.dumps(coco))
    trackers = {n: ct.load_csv(str(tmp_path / f"{n}.csv")) for n in ("good", "bad")}
    det = ct.load_csv(str(tmp_path / "good.csv"))
    rows = ct.evaluate(trackers, ct.load_gt(str(tmp_path / "gt.json")), det)
    good = [r for r in rows if r["tracker"] == "good"]
    bad = [r for r in rows if r["tracker"] == "bad"]
    assert all(r["err_mean"] == 0 and r["pck@0.05"] == 1 and r["hit@iou0.5"] == 1 for r in good)
    assert all(r["jitter"] == 0 and r["coverage_all"] == 1 and r["breaks_per_100"] == 0 for r in good)
    assert all(r["err_mean"] > 0.05 and r["jitter"] > 0 and r["det_agree"] > 0 for r in bad)
    assert all(r["coverage_all"] < 1 and r["breaks_per_100"] > 0 for r in bad)
    out = io.StringIO()
    ct.summarize(rows, equiv_margin=0.01, out=out)
    assert "err_mean" in out.getvalue()


def test_swap_detected():
    trk = {"yuji": {0: (np.array([0, 0, 10, 10.]), 1, "")}, "laogong": {0: (np.array([0, 0, 10, 11.]), 1, "")}}
    assert ct.free_metrics(trk, [0], 100.0, None)["swap_rate"] == 1.0
