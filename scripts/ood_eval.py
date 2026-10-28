#!/usr/bin/env python3
"""Score every compared model on a test set (manual COCO annotations): every image in the
annotation json is evaluated - images without acupoints too (any box there is a false positive).

NO score threshold is used: every saved detection is considered, so no model gets an advantage
from a threshold choice. (Make sure the prediction files were saved with a very low confidence,
e.g. YOLO conf=0.001 - a cut made when the predictions were saved cannot be undone here.)

EASIEST: fill in the SETTINGS block below and run   python ood_eval.py
It runs two protocols, each with and without laogong, box (IoU) and point (centre) matching:
  best_box   only the HIGHEST-SCORE box of each acupoint in each image is kept (like the
             pipeline: one location per point), missed point = FN   -> main comparison
  all_boxes  every saved box is kept (standard COCO mAP), missed point = FN
and writes OUT_ROOT/<protocol>/<points>/<iou|center>/summary.md, summary.csv, per_class.csv.

Or from the terminal (same defaults: no threshold, best box only, missed = FN):
    python ood_eval.py --ann external.coco.json \
        --pred GroundingDINO=preds/GroundingDINO.json --pred GLIP=preds/GLIP.json \
        --pred YOLOv8=preds/YOLOv8.json --pred YOLO11=preds/YOLO11.json \
        --pred YOLO26=preds/YOLO26.json --pred RT-DETR=preds/RT-DETR.json \
        --ref GroundingDINO --out ood_results

Matching (--match):
  iou     box overlap, COCO-style: hit if IoU >= 0.5   -> AP50, AP75, AP50:95
  center  point localisation: hit if the predicted box CENTRE is within alpha x (diagonal of the
          true box) of the true centre                -> AP@a0.5 (main), AP@a0.25 (strict), AP@a0.25:1.0
  both    (default) runs both and writes <out>/iou and <out>/center
With best_box, each model gives at most ONE box per acupoint per image, so P / R / F1 compare
"where the model puts each point": TP = best box on the right place, FP = best box on the wrong
place (or on an image without that point), FN = true point not found.
A MISSED point (true point with no box at all in that image) is handled by --missed:
  fn    (default) standard detection metric: FN only (lowers recall, not precision)
  fp    counted like a wrongly placed box: one FP (precision) AND one FN (recall)
  skip  left out of the evaluation (reported as "skipped"; models then differ in points scored)
Per model and per acupoint (COCO greedy matching, 101-point AP):
  AP at three strictness levels, Precision, Recall, F1
Whole model:
  MACRO  = mean over the points  (MACRO AP = the usual mAP)
  MICRO  = all points pooled (TP / FP / FN counts)
95% CIs by bootstrap over IMAGES (default 1000 resamples), and with --ref the paired
bootstrap difference of every model vs the reference model (mAP50, mAP50:95, micro F1).
Also prints a description of the test set (images, sources, resolutions, boxes per point)
for the figure captions. Writes summary.csv, per_class.csv and summary.md to --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

DEFAULT_CLASSES = ["yuji", "laogong", "zhongchong", "shaofu"]

# ============================================================================
# SETTINGS - edit these and run:  python ood_eval.py
# ============================================================================
ANN_JSON = r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/train/_annotations.coco.json"  # ground truth
PREDS = {   # model name : prediction file (COCO results list, ood_predict.py output, or COCO json with scores)
    "GroundingDINO": r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/GD_predictions.coco.json",
    "GLIP":          r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/GLIP_predictions.coco.json",
    "YOLOv8m":       r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/YOLO8.json",
    "YOLOv11m":      r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/YOLO11.json",
    "YOLOv26m":      r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/YOLO26.json",
    "RT-DETR":       r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/RT-DETR.json",
}
OUT_ROOT = r"/home/suhel.khan/TCM/For_Videos/ood_results"
REF_MODEL = "GroundingDINO"          # paired differences of every model vs this one ("" = none)

PROTOCOLS = {
    #   thr 0.0 = no threshold; missed: fn (standard) / fp / skip; best_only: keep only the best box per point
    "best_box":  dict(thr=0.0, missed="fn", best_only=True),
    "all_boxes": dict(thr=0.0, missed="fn", best_only=False),
}
POINT_SUBSETS = {
    "all_points": ["yuji", "laogong", "zhongchong", "shaofu"],
    "without_laogong": ["yuji", "zhongchong", "shaofu"],
}
MATCH = "both"          # iou / center / both
N_BOOTSTRAP = 1000      # 0 = no confidence intervals
# ============================================================================
IOU_THRS = np.round(np.arange(0.50, 1.00, 0.05), 2)
CENTER_ALPHAS = np.array([0.5, 0.25, 0.75, 1.0])   # first = main level, second = strict level

MODES = {
    # levels: hit thresholds (main first, strict second); names: (main, strict, mean) AP labels
    "iou": {"levels": IOU_THRS, "main": 0, "strict": 5, "names": ("AP50", "AP75", "AP50:95"),
            "hit": "IoU >= 0.5"},
    "center": {"levels": CENTER_ALPHAS, "main": 0, "strict": 1, "names": ("AP@a0.5", "AP@a0.25", "AP@a0.25:1.0"),
               "hit": "centre distance <= 0.5 x true-box diagonal"},
}


# ───────────────────────────── matching / AP ─────────────────────────────
def iou_matrix(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """p (N,4) g (M,4) xyxy -> (N,M)"""
    if len(p) == 0 or len(g) == 0:
        return np.zeros((len(p), len(g)))
    x1 = np.maximum(p[:, None, 0], g[None, :, 0])
    y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    x2 = np.minimum(p[:, None, 2], g[None, :, 2])
    y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / np.maximum(ap[:, None] + ag[None, :] - inter, 1e-12)


def center_dist_matrix(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """distance between box centres divided by the diagonal of the TRUE box -> (N,M)"""
    if len(p) == 0 or len(g) == 0:
        return np.zeros((len(p), len(g)))
    pc = (p[:, :2] + p[:, 2:]) / 2
    gc = (g[:, :2] + g[:, 2:]) / 2
    diag = np.maximum(np.hypot(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]), 1e-6)
    return np.linalg.norm(pc[:, None, :] - gc[None, :, :], axis=2) / diag[None, :]


def match(pred_boxes, pred_scores, gt_boxes, mode: str = "iou") -> np.ndarray:
    """greedy, highest score first; returns tp flags (n_levels, N) for every strictness level.
    iou: similarity = IoU, hit if >= level.  center: similarity = -normalised distance, hit if <= level."""
    order = np.argsort(-pred_scores, kind="mergesort")
    if mode == "iou":
        sim = iou_matrix(pred_boxes[order], gt_boxes)
        thrs = MODES["iou"]["levels"]
    else:
        sim = -center_dist_matrix(pred_boxes[order], gt_boxes)
        thrs = -MODES["center"]["levels"]
    tp = np.zeros((len(thrs), len(order)), dtype=bool)
    for t, thr in enumerate(thrs):
        used = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(order)):
            if len(gt_boxes) == 0:
                break
            cand = np.where(~used, sim[i], -np.inf)
            j = int(np.argmax(cand))
            if cand[j] >= thr:
                used[j] = True
                tp[t, i] = True
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return tp[:, inv]          # back to input order


def ap_101(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="mergesort")
    tps = np.cumsum(tp[o])
    fps = np.cumsum(~tp[o])
    rec = tps / n_gt
    prec = tps / np.maximum(tps + fps, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(q.mean())


def prf(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else float("nan")
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ───────────────────────────── data ─────────────────────────────
def load_gt(path: str, classes: list[str]):
    data = json.load(open(path))
    name_of = {c["id"]: str(c["name"]).lower() for c in data["categories"]}
    missing = [c for c in classes if c not in name_of.values()]
    if missing:
        raise SystemExit(f"{missing} not in {path} categories {list(name_of.values())}")
    gt = defaultdict(lambda: {c: [] for c in classes})
    for a in data["annotations"]:
        c = name_of.get(a["category_id"])
        if c in classes and not a.get("iscrowd", 0):
            x, y, w, h = a["bbox"]
            gt[a["image_id"]][c].append([x, y, x + w, y + h])
    return data, name_of, gt


def load_pred(path: str, name_of: dict, classes: list[str], gt_images: list | None = None):
    """gt_images: the "images" of the ground-truth json - used to match a prediction json that has its own
    "images" list (and possibly other image ids) to the ground truth by file name"""
    obj = json.load(open(path))
    if isinstance(obj, list):                      # COCO results list
        res = obj
    elif "results" in obj:                         # ood_predict.py output
        res = obj["results"]
    else:                                          # full COCO json whose annotations carry a "score"
        res = obj["annotations"]
    if gt_images is not None and isinstance(obj, dict) and obj.get("images"):
        gt_by_name = {os.path.basename(im["file_name"]): im["id"] for im in gt_images}
        to_gt = {im["id"]: gt_by_name.get(os.path.basename(im["file_name"])) for im in obj["images"]}
        if any(to_gt[k] != k for k in to_gt if to_gt[k] is not None) or None in to_gt.values():
            n_lost = sum(v is None for v in to_gt.values())
            print(f"note: {path}: image ids matched to the ground truth by file name"
                  + (f" ({n_lost} prediction images not in the ground truth, ignored)" if n_lost else ""),
                  file=sys.stderr)
            res = [{**r, "image_id": to_gt.get(r["image_id"])} for r in res if to_gt.get(r["image_id"]) is not None]
    if gt_images is not None:
        gt_ids = {im["id"] for im in gt_images}
        pred_ids = {r["image_id"] for r in res}
        if res and not pred_ids & gt_ids:
            raise SystemExit(f"{path}: no prediction image_id matches the ground-truth image ids - "
                             "were the predictions made on another json?")
        n_none = len(gt_ids - pred_ids)
        if n_none:
            print(f"note: {path}: {n_none} of {len(gt_ids)} test images have no prediction (still evaluated)",
                  file=sys.stderr)
    if res and any("score" not in r for r in res):
        print(f"WARNING: {path}: predictions without 'score' are treated as score 1.0 - AP is not meaningful",
              file=sys.stderr)
        res = [{**r, "score": r.get("score", 1.0)} for r in res]
    if res:
        lo = min(r["score"] for r in res)
        print(f"note: {path}: {len(res)} boxes, lowest score {lo:.3f}"
              + (" - predictions were saved with a confidence cut; re-save them with a very low confidence "
                 "(e.g. 0.001) for a threshold-free comparison" if lo > 0.05 else ""), file=sys.stderr)
    pred = defaultdict(lambda: {c: ([], []) for c in classes})
    for r in res:
        c = name_of.get(r["category_id"])
        if c in classes:
            x, y, w, h = r["bbox"]
            pred[r["image_id"]][c][0].append([x, y, x + w, y + h])
            pred[r["image_id"]][c][1].append(r["score"])
    return pred


def per_image_records(image_ids, gt, pred, classes, mode: str = "iou", thr: float | None = None,
                      missed: str = "fn", best_only: bool = False):
    """rec[img][cls] = (scores (N,), tp (n_levels,N), n_gt, n_skipped, n_missed_fp)
    thr: drop predictions below this score BEFORE matching (None = keep all).
    best_only: then keep only the highest-score box of each point in each image.
    missed: what to do with a point that has GT but no prediction left in an image -
      "fn" a miss (FN), "fp" a miss AND a false positive, "skip" not evaluated in that image."""
    if missed not in ("fn", "fp", "skip"):
        raise ValueError(f"missed must be fn, fp or skip, got {missed!r}")
    rec = {}
    for im in image_ids:
        rec[im] = {}
        for c in classes:
            g = np.array(gt[im][c], float).reshape(-1, 4) if im in gt else np.zeros((0, 4))
            pb, ps = pred[im][c] if im in pred else ([], [])
            pb, ps = np.array(pb, float).reshape(-1, 4), np.array(ps, float)
            if thr is not None:
                keep = ps >= thr
                pb, ps = pb[keep], ps[keep]
            if best_only and len(ps):
                b = int(np.argmax(ps))
                pb, ps = pb[b:b + 1], ps[b:b + 1]
            skipped = missed_fp = 0
            if len(ps) == 0 and len(g):
                if missed == "skip":
                    skipped, g = len(g), np.zeros((0, 4))
                elif missed == "fp":
                    missed_fp = len(g)
            rec[im][c] = (ps, match(pb, ps, g, mode), len(g), skipped, missed_fp)
    return rec


def metrics(rec, images, classes, thr: float, mode: str = "iou") -> dict:
    """images may contain repeats (bootstrap)."""
    cfg = MODES[mode]
    n_lv = len(cfg["levels"])
    out = {"per_class": {}}
    tot_tp = tot_pred = tot_gt = tot_skip = 0
    for c in classes:
        s = np.concatenate([rec[i][c][0] for i in images]) if images else np.zeros(0)
        tp = np.concatenate([rec[i][c][1] for i in images], axis=1) if images else np.zeros((n_lv, 0), bool)
        n_gt = int(sum(rec[i][c][2] for i in images))
        n_skip = int(sum(rec[i][c][3] for i in images))
        n_mfp = int(sum(rec[i][c][4] for i in images))
        aps = [ap_101(s, tp[t], n_gt) for t in range(n_lv)]
        keep = s >= thr
        k_tp, k_pred = int(tp[cfg["main"]][keep].sum()), int(keep.sum()) + n_mfp   # missed points as FP
        p, r, f = prf(k_tp, k_pred, n_gt)
        out["per_class"][c] = {"n_gt": n_gt, "AP": aps[cfg["main"]], "AP_strict": aps[cfg["strict"]],
                               "AP_mean": float(np.nanmean(aps)) if n_gt else float("nan"),
                               "P": p, "R": r, "F1": f, "TP": k_tp, "FP": k_pred - k_tp, "FN": n_gt - k_tp,
                               "skipped": n_skip, "missed_fp": n_mfp}
        tot_tp, tot_pred, tot_gt = tot_tp + k_tp, tot_pred + k_pred, tot_gt + n_gt
        tot_skip += n_skip
    pc = out["per_class"].values()
    def nanmean(v):
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")
    out["macro"] = {k: nanmean([d[k] for d in pc]) for k in ("AP", "AP_strict", "AP_mean", "P", "R", "F1")}
    p, r, f = prf(tot_tp, tot_pred, tot_gt)
    out["micro"] = {"P": p, "R": r, "F1": f, "TP": tot_tp, "FP": tot_pred - tot_tp, "FN": tot_gt - tot_tp,
                    "n_gt": tot_gt, "skipped": tot_skip}
    return out


def headline(m) -> dict:
    return {"mAP": m["macro"]["AP"], "mAP_strict": m["macro"]["AP_strict"], "mAP_mean": m["macro"]["AP_mean"],
            "P": m["micro"]["P"], "R": m["micro"]["R"], "F1": m["micro"]["F1"]}


def bootstrap(recs: dict, image_ids, classes, thrs: dict, n: int, seed: int = 0, mode: str = "iou"):
    """same resampled image sets for every model -> per-model CIs and paired differences."""
    rng = np.random.default_rng(seed)
    draws = {name: defaultdict(list) for name in recs}
    ids = list(image_ids)
    for _ in range(n):
        sample = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
        for name, rec in recs.items():
            for k, v in headline(metrics(rec, sample, classes, thrs[name], mode)).items():
                draws[name][k].append(v)
    return {name: {k: np.array(v) for k, v in d.items()} for name, d in draws.items()}


def ci(a: np.ndarray) -> tuple[float, float]:
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2


# ───────────────────────────── dataset description ─────────────────────────────
def describe(data, gt, classes, out=sys.stdout):
    def p(*a):
        print(*a, file=out)
    imgs = data["images"]
    tags = Counter()
    for im in imgs:
        t = (im.get("extra") or {}).get("user_tags") or []
        tags.update(t if t else ["(no tag)"])
    src = Counter()
    for im in imgs:
        name = (im.get("extra") or {}).get("name") or im["file_name"]
        m = re.match(r"([A-Za-z]+[ _]?[A-Za-z]*)", name)
        src[m.group(1).strip(" _") if m else "other"] += 1
    wh = np.array([[im.get("width", 0), im.get("height", 0)] for im in imgs])
    per_cls = Counter()
    n_with = Counter()
    for im in imgs:
        for c in classes:
            k = len(gt[im["id"]][c]) if im["id"] in gt else 0
            per_cls[c] += k
            n_with[c] += k > 0
    p("=== External / OOD test set ===")
    p(f"  images: {len(imgs)}   annotated points: {sum(per_cls.values())}")
    p(f"  resolution (w x h): {wh[:, 0].min()}-{wh[:, 0].max()} x {wh[:, 1].min()}-{wh[:, 1].max()} px")
    p(f"  file-name groups: {dict(src.most_common())}")
    p(f"  dataset tags: {dict(tags)}")
    for c in classes:
        p(f"  {c:<11} boxes {per_cls[c]:>4}  in {n_with[c]:>3}/{len(imgs)} images")
    dup = [(im["id"], c) for im in imgs for c in classes if im["id"] in gt and len(gt[im["id"]][c]) > 1]
    if dup:
        p(f"  NOTE: {len(dup)} image/point pairs have >1 box (check labels): {dup[:10]}")
    tiny = [(a["id"], a["bbox"]) for a in data["annotations"] if min(a["bbox"][2], a["bbox"][3]) < 2]
    if tiny:
        p(f"  NOTE: {len(tiny)} boxes thinner than 2 px (cannot reach IoU 0.5): {tiny[:5]}")


# ───────────────────────────── main ─────────────────────────────
def fmt(v, ci_=None):
    s = "  n/a " if v != v else f"{v:.3f}"
    return s if ci_ is None else f"{s} [{ci_[0]:.3f}, {ci_[1]:.3f}]"


def report(mode, models, recs, res, boot, thrs, classes, ref, out_dir, n_boot, missed="fp",
           best_only=True):
    """print + save the tables of one matching mode"""
    cfg = MODES[mode]
    n_main, n_strict, n_mean = cfg["names"]
    os.makedirs(out_dir, exist_ok=True)
    title = "BOX OVERLAP (IoU)" if mode == "iou" else "POINT LOCALISATION (box centre)"
    print(f"\n{'#' * 100}\n#  {title}: a prediction is correct if {cfg['hit']}\n{'#' * 100}")

    # ── per class
    rows = []
    print(f"\n=== Per acupoint (P/R/F1: {cfg['hit']}) ===")
    print(f"  {'model':<15}{'point':<12}{'GT':>5}{n_main:>9}{n_strict:>10}{n_mean:>14}"
          f"{'P':>8}{'R':>8}{'F1':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'skip':>6}")
    for name in models:
        for c in classes:
            d = res[name]["per_class"][c]
            print(f"  {name:<15}{c:<12}{d['n_gt']:>5}{d['AP']:>9.3f}{d['AP_strict']:>10.3f}{d['AP_mean']:>14.3f}"
                  f"{d['P']:>8.3f}{d['R']:>8.3f}{d['F1']:>8.3f}{d['TP']:>5}{d['FP']:>5}{d['FN']:>5}{d['skipped']:>6}")
            rows.append({"model": name, "point": c, "score_thr": thrs[name], "n_gt": d["n_gt"],
                         n_main: d["AP"], n_strict: d["AP_strict"], n_mean: d["AP_mean"],
                         **{k: d[k] for k in ("P", "R", "F1", "TP", "FP", "FN", "skipped")}})
    with open(os.path.join(out_dir, "per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ── whole model
    label = {"mAP": "m" + n_main, "mAP_strict": "m" + n_strict, "mAP_mean": "m" + n_mean,
             "P": "Precision", "R": "Recall", "F1": "F1"}
    print("\n=== Whole model (mAP = mean over the 4 points; P/R/F1 = all points pooled) ===")
    if boot:
        print(f"  95% CI: bootstrap over images, {n_boot} resamples")
    keys = ["mAP", "mAP_strict", "mAP_mean", "P", "R", "F1"]
    summ = []
    scope = {"fp": "missed points count as FP and FN (every model scored on the same points)",
             "fn": "missed points count as FN",
             "skip": "missed points skipped (only detected points scored)"}[missed]
    no_thr = all(t <= 0 for t in thrs.values())
    boxes = ("the best (highest-score) box of each point per image" if best_only else "every saved box") + \
        (", no score threshold" if no_thr else ", score >= thr")
    print(f"  evaluated: {boxes}; {scope}")
    md = [f"Matching: a prediction is correct if {cfg['hit']}. Evaluated: {boxes}; {scope}.", "",
          f"| Model | thr | {label['mAP']} | {label['mAP_mean']} | Precision | Recall | F1 |",
          "|---|---|---|---|---|---|---|"]
    for name in models:
        h = headline(res[name])
        cis = {k: ci(boot[name][k]) for k in keys} if boot else {}
        mi = res[name]["micro"]
        print(f"  {name:<15} thr {thrs[name]:.2f}")
        for k in keys:
            print(f"      {label[k]:<15} {fmt(h[k], cis.get(k))}")
        print(f"      counts          TP {mi['TP']}  FP {mi['FP']}  FN {mi['FN']}  (of {mi['n_gt']} evaluated GT points;"
              f" {mi['skipped']} missed points skipped)")
        summ.append({"model": name, "score_thr": thrs[name], **{label[k]: h[k] for k in keys},
                     **{f"{label[k]}_lo": cis[k][0] for k in cis}, **{f"{label[k]}_hi": cis[k][1] for k in cis},
                     "TP": mi["TP"], "FP": mi["FP"], "FN": mi["FN"], "n_gt": mi["n_gt"],
                     "skipped": mi["skipped"]})
        cell = (lambda k: f"{h[k]:.3f} ({cis[k][0]:.3f}–{cis[k][1]:.3f})") if boot else (lambda k: f"{h[k]:.3f}")
        md.append(f"| {name} | {thrs[name]:.2f} | {cell('mAP')} | {cell('mAP_mean')} | {cell('P')} "
                  f"| {cell('R')} | {cell('F1')} |")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    # ── paired differences vs reference
    if boot and ref:
        print(f"\n=== Paired bootstrap difference vs {ref} (model - ref; CI excluding 0 = significant *) ===")
        md += ["", f"Difference vs {ref} (95% paired bootstrap CI, * = significant):", "",
               f"| Model | Δ{label['mAP']} | Δ{label['mAP_mean']} | ΔF1 |", "|---|---|---|---|"]
        for name in models:
            if name == ref:
                continue
            cells = []
            line = f"  {name:<15}"
            for k in ("mAP", "mAP_mean", "F1"):
                diff = headline(res[name])[k] - headline(res[ref])[k]
                lo, hi = ci(boot[name][k] - boot[ref][k])
                sig = "*" if (lo > 0 or hi < 0) else " "
                line += f"  Δ{label[k]} {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"
                cells.append(f"{diff:+.3f} ({lo:+.3f} to {hi:+.3f}){sig.strip()}")
            print(line)
            md.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nsaved {out_dir}/summary.csv, per_class.csv, summary.md")


def evaluate_all(ann: str, pred_paths: dict, classes: list[str], thrs: dict, missed: str = "fn",
                 best_only: bool = False, match: str = "both", n_boot: int = 1000, ref: str | None = None,
                 out: str = "ood_results", data_gt=None) -> dict:
    """evaluate every model in pred_paths {name: path} on all images of the ground-truth json"""
    classes = [c.lower() for c in classes]
    data, name_of, gt = data_gt if data_gt else load_gt(ann, classes)
    if data_gt:
        gt = {im: {c: v for c, v in pts.items() if c in classes} for im, pts in gt.items()}
    image_ids = [im["id"] for im in data["images"]]
    os.makedirs(out, exist_ok=True)
    describe(data, defaultdict(lambda: {c: [] for c in classes}, gt), classes)
    preds = {name: load_pred(path, name_of, classes, data["images"]) for name, path in pred_paths.items()}
    modes = ["iou", "center"] if match == "both" else [match]
    results = {}
    for mode in modes:
        recs, res = {}, {}
        for name in pred_paths:
            recs[name] = per_image_records(image_ids, gt, preds[name], classes, mode,
                                           thr=thrs[name], missed=missed, best_only=best_only)
            res[name] = metrics(recs[name], image_ids, classes, thrs[name], mode)
        boot = bootstrap(recs, image_ids, classes, thrs, n_boot, mode=mode) if n_boot else None
        out_dir = os.path.join(out, mode) if len(modes) > 1 else out
        report(mode, list(pred_paths), recs, res, boot, thrs, classes, ref, out_dir, n_boot,
               missed=missed, best_only=best_only)
        results[mode] = res
    return results


def run_settings() -> int:
    """run every protocol x point subset from the SETTINGS block"""
    missing = [f"{n}: {p}" for n, p in PREDS.items() if not os.path.isfile(p)]
    if not os.path.isfile(ANN_JSON):
        raise SystemExit(f"ANN_JSON not found: {ANN_JSON}")
    preds = {n: p for n, p in PREDS.items() if os.path.isfile(p)}
    for m in missing:
        print(f"prediction file not found, model left out -> {m}", file=sys.stderr)
    if not preds:
        raise SystemExit("no prediction file found - check PREDS")
    ref = REF_MODEL if REF_MODEL in preds else None
    data_gt = load_gt(ANN_JSON, DEFAULT_CLASSES)
    written = []
    for proto, cfg in PROTOCOLS.items():
        thr = cfg["thr"]
        thrs = {m: float(thr.get(m, 0.0)) if isinstance(thr, dict) else float(thr) for m in preds}
        for subset, pts in POINT_SUBSETS.items():
            out = os.path.join(OUT_ROOT, proto, subset)
            print(f"\n{'=' * 100}\n  PROTOCOL {proto}  |  {subset}  |  thresholds {thrs}\n{'=' * 100}")
            evaluate_all(ANN_JSON, preds, pts, thrs, cfg["missed"], cfg["best_only"], MATCH, N_BOOTSTRAP,
                         ref, out, data_gt)
            written.append(out)
    print("\nresults:\n  " + "\n  ".join(f"{w}/<iou|center>/summary.md" for w in written))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--pred", action="append", required=True, metavar="NAME=preds.json")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--thr", action="append", default=[],
                    help="score threshold (default 0 = no threshold): only boxes >= thr are evaluated; one value "
                         "for all, or NAME=value (repeatable)")
    ap.add_argument("--keep-all", action="store_true",
                    help="evaluate every box >= thr instead of only the best box of each point per image")
    ap.add_argument("--missed", choices=["fp", "fn", "skip"], default="fn",
                    help="true point with no box above thr: fn = FN only (default, standard); fp = count as "
                         "FP and FN; skip = leave it out")
    ap.add_argument("--match", choices=["iou", "center", "both"], default="both",
                    help="iou = box overlap (COCO); center = predicted centre near the true centre; both = report both")
    ap.add_argument("--bootstrap", type=int, default=1000, help="0 = no CIs")
    ap.add_argument("--ref", help="reference model for paired differences")
    ap.add_argument("--out", default="ood_results")
    args = ap.parse_args(argv)

    bad = [p for p in args.pred if "=" not in p]
    if bad:
        raise SystemExit(f"--pred must be NAME=path, e.g. --pred GroundingDINO=preds.json (got {bad})")
    models = dict(p.split("=", 1) for p in args.pred)
    thrs = {m: 0.0 for m in models}
    for t in args.thr:
        if "=" in t:
            k, v = t.split("=", 1)
            thrs[k] = float(v)
        else:
            thrs = {m: float(t) for m in models}
    if args.ref and args.ref not in models:
        raise SystemExit(f"--ref {args.ref} is not one of {list(models)}")
    evaluate_all(args.ann, models, args.classes, thrs, args.missed, not args.keep_all, args.match,
                 args.bootstrap, args.ref, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(run_settings() if len(sys.argv) == 1 else main())
