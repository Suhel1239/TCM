#!/usr/bin/env python3
"""Evaluate the ResNet-18 palm / non-palm gate on the TEST VIDEOS, from palm.csv (video_dump.py).

No images need to be exported: label the videos as time SEGMENTS (watch each video once and write
down where the palm is visible), and/or use the annotated keyframes as palm frames.

  --segments segments.csv   video, start_frame, end_frame, label      (inclusive; label palm / non_palm)
                            e.g.  003_1,0,41,non_palm
                                  003_1,42,610,palm
                            frames not covered by any segment are not evaluated
  --gt keyframes.json       optional: every annotated keyframe counts as a palm frame
  --palm palm.csv           P(palm) per frame

Reports (same as eval_palm_classifier.py): confusion matrix, accuracy, balanced accuracy, palm /
non-palm precision-recall-F1 with bootstrap CIs, ROC-AUC, PR-AUC, threshold sweep, per-video table,
and the gating impact: palm frames lost by the gate and non-palm frames let through.
Note: video frames are strongly correlated; quote the per-video table and treat the frame-level CIs
as optimistic.

EASIEST: fill in the SETTINGS block below and run   python scripts/eval_palm_gate.py
Or from the terminal:
    python scripts/eval_palm_gate.py --palm video_dumps/palm.csv --segments segments.csv \\
        --gt keyframes.json --out results/palm_gate
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from eval_palm_classifier import THRESHOLD, report  # noqa: E402
except ImportError as e:
    raise SystemExit("eval_palm_gate.py needs eval_palm_classifier.py in the same folder "
                     f"({Path(__file__).resolve().parent}) - copy it there") from e

VIDEO_REGEX = r"(?P<video>.+)_frame_(?P<frame>\d+)"   # how video name and frame number appear in keyframe names

# ============================================================================
# SETTINGS - edit these and run:  python scripts/eval_palm_gate.py
# ============================================================================
PALM_CSV = r"/home/suhel.khan/TCM/video_eval/video_dumps/palm/palm.csv"       # from video_dump.py
SEGMENTS_CSV = r"/home/suhel.khan/TCM/video_eval/palm_segments.csv"          # your palm / non_palm segments
GT_JSON = r"/home/suhel.khan/TCM/video_eval/keyframes/_annotations.coco.json"  # keyframes = palm ("" = not used)
OUT_DIR = r"/home/suhel.khan/TCM/video_eval/results/palm_gate"
PALM_THRESHOLD = THRESHOLD    # threshold used in the pipeline (0.5)
N_BOOTSTRAP = 1000
# ============================================================================

LABELS = {"palm": 1, "non_palm": 0, "non-palm": 0, "nonpalm": 0, "1": 1, "0": 0}


def load_labels(segments: str | None, gt: str | None, regex: str = VIDEO_REGEX) -> dict:
    """-> {(video, frame): 0/1}; segment labels win over keyframes"""
    lab = {}
    if gt:
        import json
        coco = json.loads(Path(gt).read_text(encoding="utf-8"))
        for im in coco["images"]:
            m = re.search(regex, Path(im["file_name"]).stem)
            if m:
                lab[(m.group("video"), int(m.group("frame")))] = 1
    if segments:
        with open(segments, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                y = LABELS.get(str(r["label"]).strip().lower())
                if y is None:
                    raise SystemExit(f"unknown label {r['label']!r} in {segments} (use palm / non_palm)")
                for fr in range(int(r["start_frame"]), int(r["end_frame"]) + 1):
                    lab[(r["video"], fr)] = y
    if not lab:
        raise SystemExit("no labels: give --segments and/or --gt")
    return lab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--palm", default=PALM_CSV)
    ap.add_argument("--segments", default=SEGMENTS_CSV or None)
    ap.add_argument("--gt", default=GT_JSON or None)
    ap.add_argument("--gt-regex", default=VIDEO_REGEX)
    ap.add_argument("--threshold", type=float, default=PALM_THRESHOLD)
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    for name, path in (("palm.csv", args.palm), ("segments", args.segments), ("keyframe json", args.gt)):
        if path and not Path(path).is_file():
            raise SystemExit(f"{name} not found: {path}  (set it in SETTINGS or pass it as an option)")
    lab = load_labels(args.segments, args.gt, args.gt_regex)
    keys, y, p = [], [], []
    with open(args.palm, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["video"], int(r["frame"]))
            if k in lab:
                keys.append(k)
                y.append(lab[k])
                p.append(float(r["p_palm"]))
    missing = len(set(lab) - set(keys))
    if missing:
        print(f"WARNING: {missing} labelled frames have no row in {args.palm}", file=sys.stderr)
    if not keys:
        raise SystemExit("no labelled frame found in palm.csv - check video names / frame numbers")
    y, p = np.array(y), np.array(p)
    names = [f"{v}_frame_{fr}" for v, fr in keys]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "report.txt").open("w", encoding="utf-8") as f:
        report(names, y, p, args.threshold, args.bootstrap, out=f)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "true", "pred", "p_palm"])
        for (v, fr), t, pp in zip(keys, y, p):
            w.writerow([v, fr, "palm" if t else "non_palm", "palm" if pp >= args.threshold else "non_palm",
                        f"{pp:.4f}"])
    print((out / "report.txt").read_text(encoding="utf-8"))
    print(f"saved {out / 'report.txt'} and {out / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
