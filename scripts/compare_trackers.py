#!/usr/bin/env python3
"""Compare acupoint trackers (e.g. DeepSORT, OC-SORT, optical flow) on the same videos.

Inputs
  --tracker NAME=tracks.csv   one per tracker (repeat). CSV columns:
        video, frame, point, x1, y1, x2, y2[, score][, source]
      video  = video name (same as in the keyframe file names, e.g. 003_1)
      frame  = frame number (0-based, as in the frame sampler)
      point  = yuji / laogong / zhongchong / shaofu (codes LU10/P8/P9/HT8 accepted)
      source = optional; "det" for frames where the box was (re)set from the detector -
               those rows are skipped for detector agreement
  --gt keyframes.json        COCO json of the annotated keyframes; file names <video>_frame_<n>.*
  --detector det.csv         optional, same CSV format: per-frame detector output

Metrics, per tracker and video (distances are divided by the palm scale = diagonal of the box
around all points in that frame, so videos with different hand sizes are comparable)
  with keyframe GT:  err_mean, err_median, pck@0.05/0.10/0.20, hit@iou0.5, gt_coverage
                     (a missing output counts as a miss)
  without GT:        jitter (2nd difference of the point centre), coverage_all (frames with all
                     points), breaks_per_100 (track gaps), swap_rate (two points closer than
                     SWAP_THR), det_agree (median distance to confident detections)

Statistics over videos: mean +- sd; Friedman test across trackers and pairwise Wilcoxon
(Holm-corrected) - needs scipy; optional paired TOST equivalence test (--equiv-margin).

Usage
    python scripts/compare_trackers.py --tracker deepsort=ds.csv --tracker ocsort=oc.csv \
        --tracker flow=of.csv --gt keyframes_coco.json --detector det.csv --out tracker_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

# ============================================================================
# SETTINGS
# ============================================================================
POINTS = ("yuji", "laogong", "zhongchong", "shaofu")
CODES = {"lu10": "yuji", "p8": "laogong", "p9": "zhongchong", "ht8": "shaofu"}
PCK_THRS = (0.05, 0.10, 0.20)
IOU_THR = 0.5
SWAP_THR = 0.05       # two different points closer than this (x palm scale) = possible swap
DET_SCORE_THR = 0.5   # detections used for det_agree
VIDEO_REGEX = r"(?P<video>.+)_frame_(?P<frame>\d+)"
KEY_METRICS = ("err_mean", "pck@0.10", "hit@iou0.5", "jitter", "coverage_all", "det_agree")
LOWER_IS_BETTER = {"err_mean", "err_median", "jitter", "breaks_per_100", "swap_rate", "det_agree"}
# ============================================================================


def norm_point(name: str) -> str | None:
    k = str(name).strip().lower()
    k = CODES.get(k, k)
    return k if k in POINTS else None


def load_csv(path: str) -> dict:
    """-> {video: {point: {frame: (box[4], score, source)}}}"""
    tracks: dict = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pt = norm_point(r["point"])
            if pt is None:
                continue
            box = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
            score = float(r["score"]) if r.get("score") not in (None, "") else 1.0
            tracks[str(r["video"])][pt][int(float(r["frame"]))] = (box, score, (r.get("source") or "").strip())
    return tracks


def load_gt(path: str, regex: str = VIDEO_REGEX) -> dict:
    """COCO keyframes -> {video: {frame: {point: box xyxy}}}"""
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    cat = {c["id"]: norm_point(c["name"]) for c in coco["categories"]}
    img = {}
    for im in coco["images"]:
        m = re.search(regex, Path(im["file_name"]).stem)
        if m:
            img[im["id"]] = (m.group("video"), int(m.group("frame")))
        else:
            print(f"warning: cannot read video/frame from {im['file_name']}, skipped", file=sys.stderr)
    gt: dict = defaultdict(lambda: defaultdict(dict))
    for a in coco["annotations"]:
        if a["image_id"] in img and cat.get(a["category_id"]):
            v, fr = img[a["image_id"]]
            x, y, w, h = a["bbox"]
            gt[v][fr][cat[a["category_id"]]] = np.array([x, y, x + w, y + h], float)
    return gt


def center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def palm_scale(boxes: list[np.ndarray]) -> float:
    """diagonal of the box around all given point boxes."""
    a = np.array(boxes)
    return float(np.hypot(a[:, 2].max() - a[:, 0].min(), a[:, 3].max() - a[:, 1].min()))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def video_scale(trk: dict) -> float:
    """median palm scale of a video from frames where >= 3 points are present."""
    frames = defaultdict(list)
    for pt, fr in trk.items():
        for f, (box, _, _) in fr.items():
            frames[f].append(box)
    s = [palm_scale(b) for b in frames.values() if len(b) >= 3]
    return float(np.median(s)) if s else float("nan")


def gt_metrics(trk: dict, gt_v: dict, fallback_scale: float) -> dict:
    errs, hits, n, covered = [], 0, 0, 0
    pck = dict.fromkeys(PCK_THRS, 0)
    for fr, pts in gt_v.items():
        scale = palm_scale(list(pts.values())) if len(pts) >= 2 else fallback_scale
        for pt, gbox in pts.items():
            n += 1
            pred = trk.get(pt, {}).get(fr)
            if pred is None:
                continue
            covered += 1
            e = float(np.linalg.norm(center(pred[0]) - center(gbox))) / scale
            errs.append(e)
            hits += iou(pred[0], gbox) >= IOU_THR
            for t in PCK_THRS:
                pck[t] += e <= t
    out = {"n_gt": n, "gt_coverage": covered / n if n else float("nan"),
           "err_mean": float(np.mean(errs)) if errs else float("nan"),
           "err_median": float(np.median(errs)) if errs else float("nan"),
           f"hit@iou{IOU_THR}": hits / n if n else float("nan")}
    out.update({f"pck@{t:.2f}": pck[t] / n if n else float("nan") for t in PCK_THRS})
    return out


def free_metrics(trk: dict, frames: list[int], scale: float, det: dict | None) -> dict:
    jit = []
    breaks = 0
    for pt in POINTS:
        fr = trk.get(pt, {})
        for f in fr:
            if f - 1 in fr and f - 2 in fr:
                c0, c1, c2 = (center(fr[g][0]) for g in (f - 2, f - 1, f))
                jit.append(float(np.linalg.norm(c2 - 2 * c1 + c0)) / scale)
        present = [f in fr for f in frames]
        first = next((i for i, x in enumerate(present) if x), None)
        if first is not None:
            last = len(present) - 1 - present[::-1].index(True)
            seg = present[first:last + 1]
            breaks += sum(1 for i in range(1, len(seg)) if seg[i] and not seg[i - 1])
    all_pts = sum(all(f in trk.get(pt, {}) for pt in POINTS) for f in frames)
    swaps = 0
    for f in frames:
        cs = [center(trk[pt][f][0]) for pt in POINTS if f in trk.get(pt, {})]
        if any(np.linalg.norm(a - b) / scale < SWAP_THR for a, b in combinations(cs, 2)):
            swaps += 1
    agree = []
    if det is not None:
        for pt in POINTS:
            for f, (dbox, dscore, _) in det.get(pt, {}).items():
                t = trk.get(pt, {}).get(f)
                if dscore >= DET_SCORE_THR and t is not None and t[2] != "det":
                    agree.append(float(np.linalg.norm(center(t[0]) - center(dbox))) / scale)
    nf = len(frames)
    return {"n_frames": nf, "jitter": float(np.mean(jit)) if jit else float("nan"),
            "coverage_all": all_pts / nf if nf else float("nan"),
            "breaks_per_100": 100 * breaks / nf if nf else float("nan"),
            "swap_rate": swaps / nf if nf else float("nan"),
            "det_agree": float(np.median(agree)) if agree else float("nan")}


def evaluate(trackers: dict, gt: dict | None = None, det: dict | None = None) -> list[dict]:
    """-> one row per (tracker, video)."""
    videos = sorted(set().union(*(t.keys() for t in trackers.values())) | set(gt or {}))
    rows = []
    for v in videos:
        frames = set()
        for src in list(trackers.values()) + ([det] if det else []):
            for fr in src.get(v, {}).values():
                frames |= set(fr)
        frames = sorted(frames)
        scales = [video_scale(t.get(v, {})) for t in trackers.values()]
        scale = float(np.nanmedian(scales)) if not all(np.isnan(scales)) else float("nan")
        for name, t in trackers.items():
            trk = t.get(v, {})
            row = {"tracker": name, "video": v}
            row.update(free_metrics(trk, frames, scale, det.get(v, {}) if det else None))
            if gt and v in gt:
                row.update(gt_metrics(trk, gt[v], scale))
            rows.append(row)
    return rows


def summarize(rows: list[dict], metrics: tuple[str, ...] = KEY_METRICS, equiv_margin: float | None = None,
              equiv_metric: str = "err_mean", out=sys.stdout) -> dict:
    def p(*a):
        print(*a, file=out)

    trackers = list(dict.fromkeys(r["tracker"] for r in rows))
    table = {m: {t: {r["video"]: r.get(m, float("nan")) for r in rows if r["tracker"] == t} for t in trackers}
             for m in metrics}
    p(f"=== {len(trackers)} trackers, {len({r['video'] for r in rows})} videos (mean +- sd over videos) ===")
    p(f"  {'metric':<14}" + "".join(f"{t:>28}" for t in trackers) + "   better")
    for m in metrics:
        cells = []
        for t in trackers:
            vals = np.array([x for x in table[m][t].values() if not np.isnan(x)])
            cells.append(f"{vals.mean():.4f} +- {vals.std(ddof=1) if len(vals) > 1 else 0:.4f} (n={len(vals)})"
                         if len(vals) else "n/a")
        p(f"  {m:<14}" + "".join(f"{c:>28}" for c in cells) + ("   lower" if m in LOWER_IS_BETTER else "   higher"))

    try:
        from scipy import stats
    except ImportError:
        p("\n(scipy not installed: pip install scipy for Friedman / Wilcoxon / TOST)")
        return table

    p("\n=== tests over videos (paired; videos with a value for every tracker) ===")
    for m in metrics:
        vids = sorted(set.intersection(*(
            {v for v, x in table[m][t].items() if not np.isnan(x)} for t in trackers)))
        if len(vids) < 3:
            p(f"  {m}: too few paired videos ({len(vids)})")
            continue
        data = [np.array([table[m][t][v] for v in vids]) for t in trackers]
        line = f"  {m:<14} n={len(vids):<3}"
        if len(trackers) >= 3 and np.allclose(np.array(data), data[0][None, :]):
            line += " all trackers identical"
        elif len(trackers) >= 3:
            try:
                line += f" Friedman p={stats.friedmanchisquare(*data).pvalue:.4f}"
            except ValueError:
                line += " Friedman n/a"
        pairs = list(combinations(range(len(trackers)), 2))
        raw = []
        for i, j in pairs:
            d = data[i] - data[j]
            raw.append(1.0 if np.allclose(d, 0) else stats.wilcoxon(data[i], data[j]).pvalue)
        order = np.argsort(raw)  # Holm correction
        adj = np.empty(len(raw))
        running = 0.0
        for rank, k in enumerate(order):
            running = max(running, min(1.0, (len(raw) - rank) * raw[k]))
            adj[k] = running
        line += "  " + ", ".join(f"{trackers[i]} vs {trackers[j]} p={adj[k]:.4f}" for k, (i, j) in enumerate(pairs))
        p(line)

    if equiv_margin is not None and equiv_metric in table:
        p(f"\n=== equivalence (paired TOST on {equiv_metric}, margin +-{equiv_margin}) ===")
        for a, b in combinations(trackers, 2):
            vids = sorted({v for v, x in table[equiv_metric][a].items() if not np.isnan(x)}
                          & {v for v, x in table[equiv_metric][b].items() if not np.isnan(x)})
            if len(vids) < 3:
                p(f"  {a} vs {b}: too few paired videos")
                continue
            d = np.array([table[equiv_metric][a][v] - table[equiv_metric][b][v] for v in vids])
            if np.allclose(d, d[0]):
                pv = 0.0 if abs(d[0]) < equiv_margin else 1.0
            else:
                p_low = stats.ttest_1samp(d, -equiv_margin, alternative="greater").pvalue
                p_high = stats.ttest_1samp(d, equiv_margin, alternative="less").pvalue
                pv = max(p_low, p_high)
            verdict = "EQUIVALENT" if pv < 0.05 else "not shown equivalent"
            p(f"  {a} vs {b}: mean diff {d.mean():+.4f}, TOST p={pv:.4f} -> {verdict} (n={len(vids)})")
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--gt", help="COCO json of annotated keyframes")
    ap.add_argument("--detector", help="per-frame detector CSV")
    ap.add_argument("--out", default="tracker_compare")
    ap.add_argument("--equiv-margin", type=float, default=None,
                    help="TOST equivalence margin, e.g. 0.02 (in palm-scale units for err_mean)")
    ap.add_argument("--equiv-metric", default="err_mean")
    args = ap.parse_args(argv)

    trackers = {}
    for spec in args.tracker:
        if "=" not in spec:
            raise SystemExit(f"--tracker must be NAME=path.csv, got {spec!r}")
        name, path = spec.split("=", 1)
        trackers[name] = load_csv(path)
    gt = load_gt(args.gt) if args.gt else None
    det = load_csv(args.detector) if args.detector else None
    rows = evaluate(trackers, gt, det)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "per_video.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    metrics = tuple(m for m in KEY_METRICS if any(m in r for r in rows))
    with (out / "summary.txt").open("w", encoding="utf-8") as f:
        summarize(rows, metrics, args.equiv_margin, args.equiv_metric, out=f)
    print((out / "summary.txt").read_text(encoding="utf-8"))
    print(f"saved {out / 'per_video.csv'} and {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
