#!/usr/bin/env python3
"""Apply different trackers to the SAME saved per-frame detections (from video_dump.py) and write
one track CSV per tracker, ready for compare_trackers.py.

Trackers (--trackers):
  none      detector only: best box per point per frame with score >= --det-thr (no tracking)
  flow      Kalman filter + Lucas-Kanade optical flow (the pipeline's tracker): the box is reset from
            the detection when there is one, otherwise moved by the median optical flow of features
            around the point; dropped after --max-lost frames with neither detection nor flow
  ocsort    OC-SORT (pip install boxmot)
  deepsort  DeepSORT (pip install deep-sort-realtime)
All trackers use the same detections, the same --det-thr, the same --max-lost and the same palm gate:
frames with P(palm) < --palm-thr (palm.csv from video_dump.py) get no output and reset the trackers,
exactly like the video pipeline.

EASIEST: fill in the SETTINGS block below and run   python scripts/run_trackers.py
(ocsort needs  pip install boxmot,  deepsort needs  pip install deep-sort-realtime).
Or from the terminal:
    python scripts/run_trackers.py --det video_dumps/YOLO26/det.csv --palm video_dumps/palm.csv \\
        --videos test_videos/ --trackers none flow ocsort deepsort --out video_tracks/YOLO26
writes video_tracks/YOLO26/{none,flow,ocsort,deepsort}.csv
(columns: video, frame, point, x1, y1, x2, y2, score, source; source = det / trk)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

POINTS = ("yuji", "laogong", "zhongchong", "shaofu")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}

# ============================================================================
# SETTINGS - edit these and run:  python scripts/run_trackers.py
# ============================================================================
DUMP_ROOT = r"/home/suhel.khan/TCM/video_eval/video_dumps"      # OUT_ROOT of video_dump.py
VIDEOS = [r"/home/suhel.khan/TCM/video_eval/test_videos"]        # needed by flow / ocsort / deepsort
OUT_ROOT = r"/home/suhel.khan/TCM/video_eval/video_tracks"      # writes OUT_ROOT/<model>/<tracker>.csv
PALM_CSV = DUMP_ROOT + "/palm/palm.csv"                          # "" = no palm gate

# model name (= folder in DUMP_ROOT) : trackers to run on its detections
RUNS = {
    "GroundingDINO": ["none", "flow", "ocsort", "deepsort"],     # tracker comparison on the main model
    "GLIP":          ["none", "flow"],
    "YOLOv8":        ["none", "flow"],
    "YOLO11":        ["none", "flow"],
    "YOLO26":        ["none", "flow"],
    "RT-DETR":       ["none", "flow"],
}
ALSO_WITHOUT_GATE = True   # also write OUT_ROOT/<model>_nogate/ (no palm gate) -> shows what the gate costs

DET_THR = 0.15        # detections used by every tracker (the pipeline threshold)
PALM_THR = 0.5        # palm gate
MAX_LOST = 15         # frames a track may coast without support before it is dropped
FLOW_MIN_PTS = 4      # LK: minimum tracked feature points for a valid flow update
FLOW_ROI = 2.0        # LK: features are taken in a window of FLOW_ROI x the box size around the point
FLOW_FB_ERR = 1.5     # LK: max forward-backward error (px)
# ============================================================================


def load_det(path: str) -> dict:
    """-> det[video][frame][point] = [(box, score), ...] highest score first"""
    det: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pt = str(r["point"]).lower()
            if pt in POINTS:
                box = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
                det[r["video"]][int(r["frame"])][pt].append((box, float(r["score"])))
    for v in det.values():
        for fr in v.values():
            for pt in fr:
                fr[pt].sort(key=lambda t: -t[1])
    return det


def load_palm(path: str) -> dict:
    palm: dict = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            palm[r["video"]][int(r["frame"])] = float(r["p_palm"])
    return palm


# ─────────────────────────────── trackers ───────────────────────────────
class DetectorOnly:
    name = "none"
    needs_frames = False

    def reset(self):
        pass

    def step(self, frame, dets):
        return {pt: (d[0][0], d[0][1], "det") for pt, d in dets.items() if d}


class _PointTrack:
    """one acupoint: constant-velocity Kalman filter on the centre + LK optical flow"""

    def __init__(self, box, score):
        import cv2
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        c = self._centre(box)
        self.kf.statePost = np.array([[c[0]], [c[1]], [0], [0]], np.float32)   # 4x1 column (OpenCV 5)
        self.size = np.array([box[2] - box[0], box[3] - box[1]], float)
        self.score, self.lost, self.feats = score, 0, None

    @staticmethod
    def _centre(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], float)

    def box(self):
        c = self.kf.statePost[:2, 0].astype(float)
        return np.array([c[0] - self.size[0] / 2, c[1] - self.size[1] / 2,
                         c[0] + self.size[0] / 2, c[1] + self.size[1] / 2])

    def init_features(self, gray):
        import cv2
        b = self.box()
        c, half = self._centre(b), np.maximum(self.size * FLOW_ROI / 2, 8)
        x1, y1 = (np.maximum(c - half, 0)).astype(int)
        x2, y2 = (np.minimum(c + half, [gray.shape[1] - 1, gray.shape[0] - 1])).astype(int)
        mask = np.zeros_like(gray)
        mask[y1:y2 + 1, x1:x2 + 1] = 255
        f = cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
        if f is None or len(f) < FLOW_MIN_PTS:     # textureless skin: fall back to a grid
            xs, ys = np.meshgrid(np.linspace(x1, x2, 5), np.linspace(y1, y2, 5))
            f = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
        self.feats = f.astype(np.float32)

    def flow_shift(self, prev_gray, gray):
        import cv2
        if self.feats is None or len(self.feats) == 0:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.feats, None, winSize=(21, 21), maxLevel=3)
        if p1 is None:
            return None
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm((p0r - self.feats).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < FLOW_FB_ERR)
        if good.sum() < FLOW_MIN_PTS:
            return None
        d = (p1 - self.feats).reshape(-1, 2)[good]
        self.feats = p1[good].reshape(-1, 1, 2)
        return np.median(d, axis=0)


class KalmanFlow:
    """Kalman + Lucas-Kanade optical flow, one track per acupoint"""
    name = "flow"
    needs_frames = True

    def __init__(self, max_lost=MAX_LOST):
        self.max_lost = max_lost
        self.reset()

    def reset(self):
        self.tracks, self.prev_gray = {}, None

    def step(self, frame, dets):
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = {}
        for pt in POINTS:
            d = dets.get(pt)
            t = self.tracks.get(pt)
            if t is None:
                if d:
                    t = self.tracks[pt] = _PointTrack(d[0][0], d[0][1])
                    t.init_features(gray)
                    out[pt] = (t.box(), d[0][1], "det")
                continue
            shift = t.flow_shift(self.prev_gray, gray) if self.prev_gray is not None else None
            centre_before = t.kf.statePost[:2, 0].astype(float)
            t.kf.predict()
            if d:                                   # detection: correct with it and re-seed the features
                box, score = d[0]
                t.kf.correct(_PointTrack._centre(box).astype(np.float32).reshape(2, 1))
                t.size = 0.7 * t.size + 0.3 * np.array([box[2] - box[0], box[3] - box[1]])
                t.score, t.lost = score, 0
                t.init_features(gray)
                out[pt] = (t.box(), score, "det")
            elif shift is not None:                 # no detection: follow the optical flow
                t.kf.correct((centre_before + shift).astype(np.float32).reshape(2, 1))
                out[pt] = (t.box(), t.score, "trk")
            else:                                   # neither: coast on the Kalman prediction
                t.kf.statePost = t.kf.statePre.copy()
                t.lost += 1
                if t.lost > self.max_lost:
                    del self.tracks[pt]
                    continue
                out[pt] = (t.box(), t.score, "trk")
        self.prev_gray = gray
        return out


def _pick(cands: dict, prev_id):
    """cands: {track_id: (box, score, source, age_rank)} -> keep the same id as before when possible"""
    if not cands:
        return None, None
    if prev_id in cands:
        return prev_id, cands[prev_id]
    tid = max(cands, key=lambda k: (cands[k][3], cands[k][1]))
    return tid, cands[tid]


class OCSort:
    """OC-SORT from boxmot; one class per acupoint, the same track id is kept per point"""
    name = "ocsort"
    needs_frames = True

    def __init__(self, det_thr=DET_THR, max_lost=MAX_LOST):
        try:
            import boxmot
        except ImportError as e:
            raise SystemExit("ocsort needs boxmot:  pip install boxmot") from e
        cls = getattr(boxmot, "OcSort", None) or getattr(boxmot, "OCSORT", None)
        if cls is None:
            raise SystemExit(f"no OcSort / OCSORT class in boxmot {getattr(boxmot, '__version__', '')}")
        self._cls, self.det_thr, self.max_lost = cls, det_thr, max_lost
        self.reset()

    def reset(self):
        try:
            self.trk = self._cls(det_thresh=self.det_thr, max_age=self.max_lost, min_hits=1)
        except TypeError:                           # other boxmot versions: default settings
            self.trk = self._cls()
        self.ids = {}

    def step(self, frame, dets):
        rows = [[*b, s, POINTS.index(pt)] for pt, d in dets.items() for b, s in d]
        arr = np.array(rows, dtype=np.float32).reshape(-1, 6)
        res = self.trk.update(arr, frame)
        res = np.asarray(res).reshape(-1, np.asarray(res).shape[-1] if np.asarray(res).size else 8)
        cands = defaultdict(dict)
        for r in res:          # x1, y1, x2, y2, id, conf, cls, ...
            cands[POINTS[int(r[6])]][int(r[4])] = (r[:4].astype(float), float(r[5]), "", 0)
        out = {}
        for pt, c in cands.items():
            tid, v = _pick(c, self.ids.get(pt))
            if v is not None:
                self.ids[pt] = tid
                out[pt] = v[:3]
        return out


class DeepSORT:
    """DeepSORT from deep-sort-realtime (appearance embedding: MobileNet); one track id kept per point"""
    name = "deepsort"
    needs_frames = True

    def __init__(self, max_lost=MAX_LOST, gpu=True):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as e:
            raise SystemExit("deepsort needs deep-sort-realtime:  pip install deep-sort-realtime") from e
        self._cls, self.max_lost, self.gpu = DeepSort, max_lost, gpu
        self.reset()

    def reset(self):
        self.trk = self._cls(max_age=self.max_lost, n_init=1, embedder="mobilenet", embedder_gpu=self.gpu)
        self.ids = {}

    def step(self, frame, dets):
        raw = [([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], s, pt)
               for pt, d in dets.items() for b, s in d]
        tracks = self.trk.update_tracks(raw, frame=frame)
        cands = defaultdict(dict)
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > self.max_lost:
                continue
            pt = t.get_det_class()
            if pt not in POINTS:
                continue
            conf = t.get_det_conf()
            cands[pt][t.track_id] = (np.asarray(t.to_ltrb(), float), float(conf) if conf is not None else 0.0,
                                     "det" if t.time_since_update == 0 else "trk", -t.time_since_update)
        out = {}
        for pt, c in cands.items():
            tid, v = _pick(c, self.ids.get(pt))
            if v is not None:
                self.ids[pt] = tid
                out[pt] = v[:3]
        return out


TRACKERS = {"none": DetectorOnly, "flow": KalmanFlow, "ocsort": OCSort, "deepsort": DeepSORT}


# ─────────────────────────────── run ───────────────────────────────
def find_videos(paths: list[str]) -> dict:
    out = {}
    for p in map(Path, paths or []):
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS) if p.is_dir() else [p]
        out.update({f.stem: f for f in files})
    return out


def frames_of(video_path, n_frames: int):
    """yields (frame_index, frame_bgr or None)"""
    if video_path is None:
        for i in range(n_frames):
            yield i, None
        return
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def run(det: dict, palm: dict | None, videos: dict, tracker_names: list[str], out_dir: Path,
        det_thr: float = DET_THR, palm_thr: float = PALM_THR, max_lost: int = MAX_LOST) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trackers = []
    for name in tracker_names:
        cls = TRACKERS[name]
        trackers.append(cls(max_lost=max_lost) if name in ("flow", "deepsort") else
                        cls(det_thr=det_thr, max_lost=max_lost) if name == "ocsort" else cls())
    need_frames = any(t.needs_frames for t in trackers)
    files = {t.name: (out_dir / f"{t.name}.csv").open("w", newline="", encoding="utf-8") for t in trackers}
    writers = {k: csv.writer(f) for k, f in files.items()}
    for w in writers.values():
        w.writerow(["video", "frame", "point", "x1", "y1", "x2", "y2", "score", "source"])
    stats = {}
    names = sorted(set(det) | set(palm or {}))
    for v in names:
        if need_frames and v not in videos:
            raise SystemExit(f"video file for {v!r} not found in --videos (needed by {[t.name for t in trackers]})")
        n_frames = max(list(det.get(v, {})) + list((palm or {}).get(v, {})) + [-1]) + 1
        if palm is not None:
            pv = np.array(list(palm.get(v, {}).values()))
            if len(pv) == 0:
                print(f"  WARNING {v}: no rows in palm.csv for this video -> every frame is gated (check the name)")
            else:
                print(f"  {v}: P(palm) min {pv.min():.3f}  median {np.median(pv):.3f}  max {pv.max():.3f}  "
                      f">= {palm_thr}: {(pv >= palm_thr).mean():.1%} of {len(pv)} frames")
        for t in trackers:
            t.reset()
        gated = 0
        for i, frame in frames_of(videos.get(v) if need_frames else None, n_frames):
            if palm is not None and palm.get(v, {}).get(i, 0.0) < palm_thr:   # gate: no palm -> no output
                gated += 1
                for t in trackers:
                    t.reset()
                continue
            dets = {pt: [(b, s) for b, s in d if s >= det_thr] for pt, d in det.get(v, {}).get(i, {}).items()}
            dets = {pt: d for pt, d in dets.items() if d}
            for t in trackers:
                for pt, (box, score, src) in t.step(frame, dets).items():
                    writers[t.name].writerow([v, i, pt, *(round(float(x), 2) for x in box), round(score, 5), src])
        stats[v] = {"frames": n_frames, "gated": gated}
        print(f"  {v}: {n_frames} frames, {gated} rejected by the palm gate")
    for f in files.values():
        f.close()
    print(f"wrote {', '.join(str(out_dir / (t.name + '.csv')) for t in trackers)}")
    return stats


def run_settings() -> int:
    """run everything from the SETTINGS block"""
    for v in VIDEOS:
        if not Path(v).exists():
            raise SystemExit(f"video folder/file not found: {v}  (set VIDEOS in SETTINGS)")
    if PALM_CSV and not Path(PALM_CSV).is_file():
        raise SystemExit(f"palm.csv not found: {PALM_CSV}\n  run video_dump.py with PALM_WEIGHTS set, "
                         "or set PALM_CSV = \"\" in SETTINGS to run without the palm gate")
    videos = find_videos(VIDEOS)
    palm = load_palm(PALM_CSV) if PALM_CSV else None
    if PALM_CSV and not palm:
        raise SystemExit(f"{PALM_CSV} is empty")
    for model, trackers in RUNS.items():
        det_csv = Path(DUMP_ROOT) / model / "det.csv"
        if not det_csv.is_file():
            print(f"{model}: {det_csv} not found, skipped (run video_dump.py first)")
            continue
        det = load_det(str(det_csv))
        variants = [(model, palm)] + ([(model + "_nogate", None)] if ALSO_WITHOUT_GATE and palm else [])
        for out_name, gate in variants:
            print(f"\n{out_name}: {', '.join(trackers)}")
            run(det, gate, videos, trackers, Path(OUT_ROOT) / out_name, DET_THR, PALM_THR, MAX_LOST)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, help="det.csv from video_dump.py")
    ap.add_argument("--palm", help="palm.csv from video_dump.py (omit = no palm gate)")
    ap.add_argument("--videos", nargs="*", help="video files or folders (needed by flow / ocsort / deepsort)")
    ap.add_argument("--trackers", nargs="+", default=["none", "flow"], choices=list(TRACKERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--det-thr", type=float, default=DET_THR)
    ap.add_argument("--palm-thr", type=float, default=PALM_THR)
    ap.add_argument("--max-lost", type=int, default=MAX_LOST)
    args = ap.parse_args(argv)
    run(load_det(args.det), load_palm(args.palm) if args.palm else None, find_videos(args.videos),
        args.trackers, Path(args.out), args.det_thr, args.palm_thr, args.max_lost)
    return 0


if __name__ == "__main__":
    sys.exit(run_settings() if len(sys.argv) == 1 else main())
