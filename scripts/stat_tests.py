#!/usr/bin/env python3
"""Per-image F1 of every model on a test set + Shapiro-Wilk + pairwise Wilcoxon signed-rank tests
(Holm-corrected) - for the statistical analysis section of the paper.

Per image and model: predictions with score >= THRESHOLD (optionally only the best box of each
acupoint) are matched to the ground truth per acupoint (greedy, highest score first):
  MATCH = "iou"     correct if IoU >= 0.5
  MATCH = "center"  correct if the predicted box centre lies within ALPHA x (diagonal of the true box)
  per-image F1 = 2 TP / (2 TP + FP + FN)   (an image with no ground truth and no prediction: F1 = 1)

Output (printed and saved to OUT_DIR):
  per_image_f1.csv     image, file name, F1 of every model
  model_summary.csv    mean / median per-image F1, Shapiro-Wilk W and p of every model
  wilcoxon.csv         every model pair: median difference, W, p, Holm-adjusted p, significant
  report.txt           all of the above, plus a ready-to-use sentence for the paper

Fill in the SETTINGS block and run:   python stat_tests.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from itertools import combinations

import numpy as np

# ============================================================================
# SETTINGS
# ============================================================================
ANN_JSON = r"/home/suhel.khan/TCM/dataset/new_data_with_captions/New_DATA_COCO/New_DATA_with_subject_level/test/test_anno.json"
PREDS = {   # model name : prediction file on the SAME images (COCO results list or COCO json with scores)
    "YOLOv8m":       r"/path/to/internal/YOLO8.json",
    "YOLOv11m":      r"/path/to/internal/YOLO11.json",
    "YOLOv26m":      r"/path/to/internal/YOLO26.json",
    "GroundingDINO": r"/path/to/internal/GD_predictions.json",
    "RT-DETR":       r"/path/to/internal/RT-DETR.json",
    "GLIP":          r"/path/to/internal/GLIP_predictions.json",
}
OUT_DIR = r"/home/suhel.khan/TCM/For_Videos/stat_tests"
CLASSES = ["yuji", "laogong", "zhongchong", "shaofu"]   # e.g. drop "laogong" for the without-Laogong test
THRESHOLD = 0.5          # the SAME confidence threshold as used for the tables
BEST_ONLY = False        # True = keep only the highest-scoring box of each acupoint per image
MATCH = "iou"            # "iou" (box criterion) or "center" (point criterion)
IOU_THR = 0.5
ALPHA = 0.5
REF_MODEL = "GroundingDINO"   # highlighted in the paper sentence
SIGNIFICANCE = 0.05
# ============================================================================


def load_gt(path):
    data = json.load(open(path))
    name_of = {c["id"]: str(c["name"]).lower() for c in data["categories"]}
    gt = {im["id"]: {c: [] for c in CLASSES} for im in data["images"]}
    for a in data["annotations"]:
        c = name_of.get(a["category_id"])
        if c in CLASSES and a["image_id"] in gt and not a.get("iscrowd", 0):
            x, y, w, h = a["bbox"]
            gt[a["image_id"]][c].append([x, y, x + w, y + h])
    return data, name_of, gt


def load_pred(path, name_of, gt_images):
    """-> {image_id: {class: (boxes xyxy, scores)}}; maps own image ids to the GT by file name"""
    obj = json.load(open(path))
    res = obj if isinstance(obj, list) else (obj.get("results") or obj.get("annotations") or [])
    if isinstance(obj, dict) and obj.get("images"):
        by_name = {os.path.basename(im["file_name"]): im["id"] for im in gt_images}
        to_gt = {im["id"]: by_name.get(os.path.basename(im["file_name"])) for im in obj["images"]}
        res = [{**r, "image_id": to_gt.get(r["image_id"])} for r in res if to_gt.get(r["image_id"]) is not None]
    gt_ids = {im["id"] for im in gt_images}
    if res and not {r["image_id"] for r in res} & gt_ids:
        raise SystemExit(f"{path}: no prediction image id matches the ground truth - predictions on another json?")
    pred = {}
    for r in res:
        c = name_of.get(r["category_id"])
        if c not in CLASSES or r["image_id"] not in gt_ids:
            continue
        x, y, w, h = r["bbox"]
        boxes, scores = pred.setdefault(r["image_id"], {}).setdefault(c, ([], []))
        boxes.append([x, y, x + w, y + h])
        scores.append(r.get("score", 1.0))
    return pred


def similarity(p, g):
    """(N,M) similarity and the hit threshold; higher = better"""
    if MATCH == "iou":
        x1 = np.maximum(p[:, None, 0], g[None, :, 0])
        y1 = np.maximum(p[:, None, 1], g[None, :, 1])
        x2 = np.minimum(p[:, None, 2], g[None, :, 2])
        y2 = np.minimum(p[:, None, 3], g[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
        ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        return inter / np.maximum(ap[:, None] + ag[None, :] - inter, 1e-12), IOU_THR
    pc, gc = (p[:, :2] + p[:, 2:]) / 2, (g[:, :2] + g[:, 2:]) / 2
    diag = np.maximum(np.hypot(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]), 1e-6)
    return -np.linalg.norm(pc[:, None] - gc[None], axis=2) / diag[None], -ALPHA


def image_counts(gt_img, pred_img):
    tp = fp = fn = 0
    for c in CLASSES:
        g = np.array(gt_img[c], float).reshape(-1, 4)
        pb, ps = pred_img.get(c, ([], [])) if pred_img else ([], [])
        pb, ps = np.array(pb, float).reshape(-1, 4), np.array(ps, float)
        keep = ps >= THRESHOLD
        pb, ps = pb[keep], ps[keep]
        if BEST_ONLY and len(ps):
            b = int(np.argmax(ps))
            pb, ps = pb[b:b + 1], ps[b:b + 1]
        hits = 0
        if len(pb) and len(g):
            sim, thr = similarity(pb[np.argsort(-ps)], g)
            used = np.zeros(len(g), bool)
            for row in sim:
                cand = np.where(~used, row, -np.inf)
                j = int(np.argmax(cand))
                if cand[j] >= thr:
                    used[j] = True
                    hits += 1
        tp, fp, fn = tp + hits, fp + len(pb) - hits, fn + len(g) - hits
    return tp, fp, fn


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> int:
    try:
        from scipy.stats import shapiro, wilcoxon
    except ImportError:
        raise SystemExit("scipy is needed:  pip install scipy")
    missing = [f"{n}: {p}" for n, p in PREDS.items() if not os.path.isfile(p)]
    if missing or not os.path.isfile(ANN_JSON):
        raise SystemExit("file(s) not found:\n  " + "\n  ".join(missing + ([] if os.path.isfile(ANN_JSON) else [ANN_JSON])))
    data, name_of, gt = load_gt(ANN_JSON)
    ids = [im["id"] for im in data["images"]]
    fname = {im["id"]: os.path.basename(im["file_name"]) for im in data["images"]}
    models = list(PREDS)

    f1, pooled = {}, {}
    for m in models:
        pred = load_pred(PREDS[m], name_of, data["images"])
        vals, tot = [], np.zeros(3, int)
        for i in ids:
            tp, fp, fn = image_counts(gt[i], pred.get(i))
            tot += (tp, fp, fn)
            vals.append(1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
        f1[m] = np.array(vals)
        pooled[m] = tot

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "per_image_f1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name"] + models)
        for k, i in enumerate(ids):
            w.writerow([i, fname[i]] + [f"{f1[m][k]:.4f}" for m in models])

    lines = [f"{len(ids)} images | classes {CLASSES} | match {MATCH} | threshold {THRESHOLD} | best_only {BEST_ONLY}", ""]
    lines.append("Pooled over all acupoints (should equal the table values computed with the same settings):")
    lines.append(f"{'model':<15}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for m in models:
        tp, fp, fn = pooled[m]
        p_, r_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        lines.append(f"{m:<15}{tp:>6}{fp:>6}{fn:>6}{p_:>8.3f}{r_:>8.3f}{2 * p_ * r_ / max(p_ + r_, 1e-12):>8.3f}")
    lines += ["", "Per image:"]
    lines.append(f"{'model':<15}{'mean F1':>9}{'median':>9}{'Shapiro W':>11}{'p':>11}")
    summ = []
    for m in models:
        sw, sp = shapiro(f1[m]) if np.ptp(f1[m]) > 0 else (float("nan"), float("nan"))
        summ.append({"model": m, "mean_f1": f1[m].mean(), "median_f1": float(np.median(f1[m])),
                     "shapiro_W": sw, "shapiro_p": sp})
        lines.append(f"{m:<15}{f1[m].mean():>9.3f}{np.median(f1[m]):>9.3f}{sw:>11.3f}{sp:>11.2e}")
    with open(os.path.join(OUT_DIR, "model_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    pairs, rows = list(combinations(models, 2)), []
    for a, b in pairs:
        d = f1[a] - f1[b]
        if np.all(d == 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(f1[a], f1[b])
        rows.append({"model_a": a, "model_b": b, "mean_diff": d.mean(), "median_diff": float(np.median(d)),
                     "W": stat, "p": p})
    adj = holm(np.array([r["p"] for r in rows]))
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        r["significant"] = pa < SIGNIFICANCE
    with open(os.path.join(OUT_DIR, "wilcoxon.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines += ["", f"Wilcoxon signed-rank tests ({len(pairs)} pairs, Holm-corrected; diff = model A - model B)",
              f"{'model A':<15}{'model B':<15}{'mean diff':>11}{'p':>11}{'p (Holm)':>11}  sig"]
    for r in rows:
        lines.append(f"{r['model_a']:<15}{r['model_b']:<15}{r['mean_diff']:>+11.3f}{r['p']:>11.2e}"
                     f"{r['p_holm']:>11.2e}  {'*' if r['significant'] else ''}")

    if REF_MODEL in models:
        better, worse, ns = [], [], []
        for r in rows:
            if REF_MODEL not in (r["model_a"], r["model_b"]):
                continue
            other = r["model_b"] if r["model_a"] == REF_MODEL else r["model_a"]
            diff = r["mean_diff"] if r["model_a"] == REF_MODEL else -r["mean_diff"]
            (better if r["significant"] and diff > 0 else worse if r["significant"] else ns).append(
                (other, r["p_holm"]))
        def fmt(xs):
            return ", ".join(f"{o} ($p_{{\\text{{adj}}}} {'< 0.001' if p < 0.001 else f'= {p:.3f}'}$)"
                             for o, p in xs)
        parts = []
        if better:
            parts.append(f"achieved a significantly higher per-image F1-score than {fmt(better)}")
        if worse:
            parts.append(f"a significantly lower per-image F1-score than {fmt(worse)}")
        if ns:
            parts.append(f"showed no significant difference to {fmt(ns)}")
        sentence = f"{REF_MODEL} " + "; it ".join(parts) + "."
        lines += ["", "Sentence for the paper (check before use):", sentence]
        normal = [s["model"] for s in summ if s["shapiro_p"] == s["shapiro_p"] and s["shapiro_p"] >= 0.05]
        lines.append("Shapiro-Wilk: " + ("all per-image F1 distributions deviate from normality (p < 0.05)"
                                         if not normal else f"normality NOT rejected for {normal}"))

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nsaved per_image_f1.csv, model_summary.csv, wilcoxon.csv, report.txt in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
