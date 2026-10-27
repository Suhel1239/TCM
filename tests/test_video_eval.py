import csv
import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_trackers as ct  # noqa: E402
import run_trackers as rt  # noqa: E402
import video_dump as vd  # noqa: E402

N, W, H = 60, 320, 240
OFFS = {"yuji": (20, 70), "laogong": (40, 40), "zhongchong": (40, 5), "shaofu": (65, 45)}


def hand_pos(f):
    return 80 + 2 * f, 60 + int(15 * np.sin(f / 8))     # top-left of the moving textured patch


def make_video(path):
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, (90, 90, 3), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 40, np.uint8)
        x, y = hand_pos(f)
        img[y:y + 90, x:x + 90] = patch
        vw.write(img)
    vw.release()


def true_box(pt, f):
    x, y = hand_pos(f)
    cx, cy = x + OFFS[pt][0], y + OFFS[pt][1]
    return [cx - 6, cy - 6, cx + 6, cy + 6]


@pytest.fixture()
def data(tmp_path):
    make_video(tmp_path / "vid1.avi")
    with open(tmp_path / "det.csv", "w", newline="") as f:     # detector fires on every 6th frame only
        w = csv.writer(f)
        w.writerow(["video", "frame", "point", "x1", "y1", "x2", "y2", "score"])
        for fr in range(0, N, 6):
            for pt in OFFS:
                w.writerow(["vid1", fr, pt, *true_box(pt, fr), 0.9])
            w.writerow(["vid1", fr, "yuji", 0, 0, 10, 10, 0.05])   # low-score junk, below DET_THR
    with open(tmp_path / "palm.csv", "w", newline="") as f:    # frames 40-44 are "not a palm"
        w = csv.writer(f)
        w.writerow(["video", "frame", "p_palm"])
        for fr in range(N):
            w.writerow(["vid1", fr, 0.1 if 40 <= fr < 45 else 0.95])
    coco = {"categories": [{"id": i + 1, "name": p} for i, p in enumerate(OFFS)], "images": [], "annotations": []}
    for k, fr in enumerate(range(3, N, 5)):                     # keyframes mostly between detections
        coco["images"].append({"id": k, "file_name": f"vid1_frame_{fr:03d}.png"})
        for i, pt in enumerate(OFFS):
            b = true_box(pt, fr)
            coco["annotations"].append({"image_id": k, "category_id": i + 1, "bbox": [b[0], b[1], 12, 12]})
    (tmp_path / "gt.json").write_text(json.dumps(coco))
    return tmp_path


def test_flow_beats_detector_only_and_gate_applies(data):
    stats = rt.run(rt.load_det(str(data / "det.csv")), rt.load_palm(str(data / "palm.csv")),
                   rt.find_videos([str(data)]), ["none", "flow"], data / "tracks")
    assert stats["vid1"]["gated"] == 5
    trk = {n: ct.load_csv(str(data / "tracks" / f"{n}.csv")) for n in ("none", "flow")}
    for n in trk:      # gated frames have no output at all
        assert not any(f in trk[n]["vid1"][pt] for pt in OFFS for f in range(40, 45))
    rows = {r["tracker"]: r for r in ct.evaluate(trk, ct.load_gt(str(data / "gt.json")))}
    none, flow = rows["none"], rows["flow"]
    assert none["success@a0.5"] < 0.3                        # detector fires on few keyframes
    assert flow["success@a0.5"] > 0.8 and flow["loss_rate"] < 0.2
    assert flow["dropout"] < none["dropout"] and flow["n_keyframes"] == 12
    assert np.isfinite(flow["jitter_px"])


def test_tracker_adapters_with_fake_libraries(data, monkeypatch):
    class FakeOc:   # boxmot-like: returns every detection as its own track
        def __init__(self, **k):
            pass

        def update(self, dets, img):
            return np.array([[*d[:4], 7 + int(d[5]), d[4], d[5], 0] for d in dets]).reshape(-1, 8)
    monkeypatch.setitem(sys.modules, "boxmot", types.SimpleNamespace(OcSort=FakeOc))

    class FakeTrack:
        def __init__(self, ltrb, conf, cls):
            self.track_id, self._b, self._c, self._k, self.time_since_update = cls, ltrb, conf, cls, 0

        def is_confirmed(self):
            return True

        def get_det_class(self):
            return self._k

        def get_det_conf(self):
            return self._c

        def to_ltrb(self):
            return self._b

    class FakeDS:
        def __init__(self, **k):
            pass

        def update_tracks(self, raw, frame=None):
            return [FakeTrack([b[0], b[1], b[0] + b[2], b[1] + b[3]], s, c) for b, s, c in raw]
    mod = types.ModuleType("deep_sort_realtime.deepsort_tracker")
    mod.DeepSort = FakeDS
    monkeypatch.setitem(sys.modules, "deep_sort_realtime", types.ModuleType("deep_sort_realtime"))
    monkeypatch.setitem(sys.modules, "deep_sort_realtime.deepsort_tracker", mod)
    rt.run(rt.load_det(str(data / "det.csv")), None, rt.find_videos([str(data)]), ["ocsort", "deepsort"],
           data / "tracks2")
    for n in ("ocsort", "deepsort"):
        t = ct.load_csv(str(data / "tracks2" / f"{n}.csv"))["vid1"]
        assert set(t) == set(OFFS) and sorted(t["yuji"]) == list(range(0, N, 6))
        assert np.allclose(t["laogong"][6][0], true_box("laogong", 6))


def test_video_dump_writes_det_palm_and_video_info(data):
    def predict(frame):
        return (np.array([[1, 2, 11, 12], [5, 5, 9, 9]], float), np.array([0.8, 0.01]),
                np.array(["yuji", "shaofu"]))
    vd.dump([data / "vid1.avi"], predict, lambda frame: 0.9, list(OFFS), data / "dump", 0.02, 5)
    det = rt.load_det(str(data / "dump" / "det.csv"))
    assert len(det["vid1"]) == N and "shaofu" not in det["vid1"][0]          # 0.01 < score_thr
    assert len(rt.load_palm(str(data / "dump" / "palm.csv"))["vid1"]) == N
    meta = list(csv.DictReader(open(data / "dump" / "videos.csv")))
    assert meta[0]["n_frames"] == str(N) and float(meta[0]["duration_s"]) == pytest.approx(N / 25)


def test_palm_gate_from_segments_and_keyframes(data):
    import eval_palm_gate as pg
    seg = data / "segments.csv"
    seg.write_text("video,start_frame,end_frame,label\nvid1,0,39,palm\nvid1,40,44,non_palm\n")
    pg.main(["--palm", str(data / "palm.csv"), "--segments", str(seg), "--gt", str(data / "gt.json"),
             "--bootstrap", "50", "--out", str(data / "gate")])
    txt = (data / "gate" / "report.txt").read_text()
    assert "48 frames: 43 palm, 5 non-palm" in txt          # 40 segment frames + keyframes 48, 53, 58
    rows = list(csv.DictReader(open(data / "gate" / "predictions.csv")))
    assert all(r["true"] == r["pred"] for r in rows)


def test_compare_without_laogong(data):
    rt.run(rt.load_det(str(data / "det.csv")), None, {}, ["none"], data / "tr3")
    try:
        ct.main(["--tracker", f"none={data / 'tr3' / 'none.csv'}", "--gt", str(data / "gt.json"),
                 "--points", "yuji", "zhongchong", "shaofu", "--out", str(data / "cmp")])
        rows = list(csv.DictReader(open(data / "cmp" / "per_video.csv")))
        assert rows[0]["n_gt"] == "36"                      # 12 keyframes x 3 points
    finally:
        ct.POINTS = ("yuji", "laogong", "zhongchong", "shaofu")


def test_settings_mode_end_to_end(data, monkeypatch):
    def fake_builder(args, classes):        # stands in for the real detector
        def predict(frame):
            return np.array([[1, 2, 11, 12]], float), np.array([0.8]), np.array(["yuji"])
        return predict
    monkeypatch.setattr(vd, "build_predictor", fake_builder)
    monkeypatch.setattr(vd, "has_module", lambda name: True)
    monkeypatch.setattr(vd, "palm_predictor", lambda *a: (lambda frame: 0.9))
    monkeypatch.setattr(vd, "VIDEOS", [str(data / "vid1.avi")])
    monkeypatch.setattr(vd, "OUT_ROOT", str(data / "dumps"))
    (data / "w.pt").write_text("x")
    monkeypatch.setattr(vd, "PALM_WEIGHTS", str(data / "w.pt"))
    monkeypatch.setattr(vd, "MODELS", [dict(name="M1", family="yolo", weights=str(data / "w.pt")),
                                       dict(name="M2", family="yolo", weights=str(data / "missing.pt"))])
    vd.run_settings()
    assert (data / "dumps" / "M1" / "det.csv").is_file() and (data / "dumps" / "palm" / "palm.csv").is_file()
    assert not (data / "dumps" / "M2").exists()                 # weights not found -> skipped

    for k, v in dict(DUMP_ROOT=str(data / "dumps"), VIDEOS=[str(data / "vid1.avi")], OUT_ROOT=str(data / "tracks"),
                     PALM_CSV=str(data / "dumps" / "palm" / "palm.csv"), RUNS={"M1": ["none", "flow"]},
                     ALSO_WITHOUT_GATE=True).items():
        monkeypatch.setattr(rt, k, v)
    rt.run_settings()
    assert (data / "tracks" / "M1" / "flow.csv").is_file() and (data / "tracks" / "M1_nogate" / "none.csv").is_file()

    monkeypatch.setattr(ct, "GT_JSON", str(data / "gt.json"))
    monkeypatch.setattr(ct, "OUT_ROOT", str(data / "results"))
    monkeypatch.setattr(ct, "COMPARISONS", [dict(out="cmp", trackers={
        "none": str(data / "tracks" / "M1" / "none.csv"), "flow": str(data / "tracks" / "M1" / "flow.csv"),
        "missing": str(data / "nope.csv")})])
    ct.run_settings()
    for sub in ("all_points", "without_laogong"):
        rows = list(csv.DictReader(open(data / "results" / "cmp" / sub / "per_video.csv")))
        assert {r["tracker"] for r in rows} == {"none", "flow"}
    assert ct.POINTS == ("yuji", "laogong", "zhongchong", "shaofu")


def test_keyframes_from_one_json_per_video(data):
    coco = json.loads((data / "gt.json").read_text())
    folder = data / "kf"
    for half, imgs in (("a", coco["images"][:6]), ("b", coco["images"][6:])):
        ids = {im["id"] for im in imgs}
        part = dict(coco, images=[dict(im, file_name=f"vid1_frame_{int(im['file_name'][11:14]):03d}_jpg.rf.Xy9.jpg")
                                  for im in imgs],
                    annotations=[a for a in coco["annotations"] if a["image_id"] in ids])
        (folder / half).mkdir(parents=True)
        (folder / half / "_annotations.coco.json").write_text(json.dumps(part))
    whole = ct.load_gt(str(data / "gt.json"))
    for src in (str(folder), [str(folder / "a" / "_annotations.coco.json"), str(folder / "b" / "_annotations.coco.json")]):
        g = ct.load_gt(src)
        assert sorted(g["vid1"]) == sorted(whole["vid1"]) and len(g["vid1"]) == 12
