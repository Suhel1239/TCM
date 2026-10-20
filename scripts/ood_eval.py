#!/usr/bin/env python3
"""Score every compared model on the external / OOD test set (manual COCO annotations).

    python ood_eval.py --ann external.coco.json \
        --pred GroundingDINO=preds/GroundingDINO.json --pred GLIP=preds/GLIP.json \
        --pred YOLOv8=preds/YOLOv8.json --pred YOLO11=preds/YOLO11.json \
        --pred YOLO26=preds/YOLO26.json --pred RT-DETR=preds/RT-DETR.json \
        --thr 0.15 --ref GroundingDINO --out ood_results

Per model and per acupoint (IoU 0.5 for P/R/F1, COCO greedy matching, 101-point AP):
  AP50, AP75, AP50:95, Precision, Recall, F1 at a fixed score threshold (--thr, or NAME=thr)
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
IOU_THRS = np.round(np.arange(0.50, 1.00, 0.05), 2)


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


def match(pred_boxes, pred_scores, gt_boxes) -> np.ndarray:
    """greedy, highest score first; returns tp flags (n_thr, N) for every IoU threshold."""
    order = np.argsort(-pred_scores, kind="mergesort")
    ious = iou_matrix(pred_boxes[order], gt_boxes)
    tp = np.zeros((len(IOU_THRS), len(order)), dtype=bool)
    for t, thr in enumerate(IOU_THRS):
        used = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(order)):
            if len(gt_boxes) == 0:
                break
            cand = np.where(~used, ious[i], -1.0)
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


def load_pred(path: str, name_of: dict, classes: list[str]):
    obj = json.load(open(path))
    if isinstance(obj, list):                      # COCO results list
        res = obj
    elif "results" in obj:                         # ood_predict.py output
        res = obj["results"]
    else:                                          # full COCO json whose annotations carry a "score"
        res = obj["annotations"]
    if res and any("score" not in r for r in res):
        print(f"WARNING: {path}: predictions without 'score' are treated as score 1.0 - AP is not meaningful",
              file=sys.stderr)
        res = [{**r, "score": r.get("score", 1.0)} for r in res]
    pred = defaultdict(lambda: {c: ([], []) for c in classes})
    for r in res:
        c = name_of.get(r["category_id"])
        if c in classes:
            x, y, w, h = r["bbox"]
            pred[r["image_id"]][c][0].append([x, y, x + w, y + h])
            pred[r["image_id"]][c][1].append(r["score"])
    return pred


def per_image_records(image_ids, gt, pred, classes):
    """rec[img][cls] = (scores (N,), tp (n_thr,N), n_gt)"""
    rec = {}
    for im in image_ids:
        rec[im] = {}
        for c in classes:
            g = np.array(gt[im][c], float).reshape(-1, 4) if im in gt else np.zeros((0, 4))
            pb, ps = pred[im][c] if im in pred else ([], [])
            pb, ps = np.array(pb, float).reshape(-1, 4), np.array(ps, float)
            rec[im][c] = (ps, match(pb, ps, g), len(g))
    return rec


def metrics(rec, images, classes, thr: float) -> dict:
    """images may contain repeats (bootstrap)."""
    out = {"per_class": {}}
    tot_tp = tot_pred = tot_gt = 0
    for c in classes:
        s = np.concatenate([rec[i][c][0] for i in images]) if images else np.zeros(0)
        tp = np.concatenate([rec[i][c][1] for i in images], axis=1) if images else np.zeros((len(IOU_THRS), 0), bool)
        n_gt = int(sum(rec[i][c][2] for i in images))
        aps = [ap_101(s, tp[t], n_gt) for t in range(len(IOU_THRS))]
        keep = s >= thr
        k_tp, k_pred = int(tp[0][keep].sum()), int(keep.sum())
        p, r, f = prf(k_tp, k_pred, n_gt)
        out["per_class"][c] = {"n_gt": n_gt, "AP50": aps[0], "AP75": aps[5], "AP50_95": float(np.nanmean(aps)),
                               "P": p, "R": r, "F1": f, "TP": k_tp, "FP": k_pred - k_tp, "FN": n_gt - k_tp}
        tot_tp, tot_pred, tot_gt = tot_tp + k_tp, tot_pred + k_pred, tot_gt + n_gt
    pc = out["per_class"].values()
    out["macro"] = {k: float(np.nanmean([d[k] for d in pc])) for k in ("AP50", "AP75", "AP50_95", "P", "R", "F1")}
    p, r, f = prf(tot_tp, tot_pred, tot_gt)
    out["micro"] = {"P": p, "R": r, "F1": f, "TP": tot_tp, "FP": tot_pred - tot_tp, "FN": tot_gt - tot_tp,
                    "n_gt": tot_gt}
    return out


def headline(m) -> dict:
    return {"mAP50": m["macro"]["AP50"], "mAP75": m["macro"]["AP75"], "mAP50_95": m["macro"]["AP50_95"],
            "P": m["micro"]["P"], "R": m["micro"]["R"], "F1": m["micro"]["F1"]}


def bootstrap(recs: dict, image_ids, classes, thrs: dict, n: int, seed: int = 0):
    """same resampled image sets for every model -> per-model CIs and paired differences."""
    rng = np.random.default_rng(seed)
    draws = {name: defaultdict(list) for name in recs}
    ids = list(image_ids)
    for _ in range(n):
        sample = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
        for name, rec in recs.items():
            for k, v in headline(metrics(rec, sample, classes, thrs[name])).items():
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--pred", action="append", required=True, metavar="NAME=preds.json")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--thr", action="append", default=[],
                    help="score threshold for P/R/F1: one value for all, or NAME=value (repeatable)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="0 = no CIs")
    ap.add_argument("--ref", help="reference model for paired differences")
    ap.add_argument("--out", default="ood_results")
    args = ap.parse_args(argv)

    classes = [c.lower() for c in args.classes]
    data, name_of, gt = load_gt(args.ann, classes)
    image_ids = [im["id"] for im in data["images"]]
    models = dict(p.split("=", 1) for p in args.pred)
    thrs = {m: 0.15 for m in models}
    for t in args.thr:
        if "=" in t:
            k, v = t.split("=", 1)
            thrs[k] = float(v)
        else:
            thrs = {m: float(t) for m in models}
    if args.ref and args.ref not in models:
        raise SystemExit(f"--ref {args.ref} is not one of {list(models)}")

    os.makedirs(args.out, exist_ok=True)
    describe(data, gt, classes)

    recs, res = {}, {}
    for name, path in models.items():
        recs[name] = per_image_records(image_ids, gt, load_pred(path, name_of, classes), classes)
        res[name] = metrics(recs[name], image_ids, classes, thrs[name])
    boot = bootstrap(recs, image_ids, classes, thrs, args.bootstrap) if args.bootstrap else None

    # ── per class
    rows = []
    print("\n=== Per acupoint (IoU 0.5 for P/R/F1) ===")
    print(f"  {'model':<15}{'point':<12}{'GT':>5}{'AP50':>8}{'AP75':>8}{'AP50:95':>9}"
          f"{'P':>8}{'R':>8}{'F1':>8}{'TP':>5}{'FP':>5}{'FN':>5}")
    for name in models:
        for c in classes:
            d = res[name]["per_class"][c]
            print(f"  {name:<15}{c:<12}{d['n_gt']:>5}{d['AP50']:>8.3f}{d['AP75']:>8.3f}{d['AP50_95']:>9.3f}"
                  f"{d['P']:>8.3f}{d['R']:>8.3f}{d['F1']:>8.3f}{d['TP']:>5}{d['FP']:>5}{d['FN']:>5}")
            rows.append({"model": name, "point": c, "score_thr": thrs[name], **d})
    with open(os.path.join(args.out, "per_class.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ── whole model
    print("\n=== Whole model (mAP = macro mean over points; P/R/F1 = micro, all points pooled) ===")
    if boot:
        print(f"  95% CI: bootstrap over images, {args.bootstrap} resamples")
    keys = ["mAP50", "mAP75", "mAP50_95", "P", "R", "F1"]
    summ = []
    md = ["| Model | thr | mAP50 | mAP50:95 | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for name in models:
        h = headline(res[name])
        cis = {k: ci(boot[name][k]) for k in keys} if boot else {}
        mi = res[name]["micro"]
        print(f"  {name:<15} thr {thrs[name]:.2f}")
        for k in keys:
            print(f"      {k:<9} {fmt(h[k], cis.get(k))}")
        print(f"      counts    TP {mi['TP']}  FP {mi['FP']}  FN {mi['FN']}  (of {mi['n_gt']} GT points)")
        summ.append({"model": name, "score_thr": thrs[name], **h,
                     **{f"{k}_lo": cis[k][0] for k in cis}, **{f"{k}_hi": cis[k][1] for k in cis},
                     "TP": mi["TP"], "FP": mi["FP"], "FN": mi["FN"], "n_gt": mi["n_gt"]})
        cell = (lambda k: f"{h[k]:.3f} ({cis[k][0]:.3f}–{cis[k][1]:.3f})") if boot else (lambda k: f"{h[k]:.3f}")
        md.append(f"| {name} | {thrs[name]:.2f} | {cell('mAP50')} | {cell('mAP50_95')} | {cell('P')} "
                  f"| {cell('R')} | {cell('F1')} |")
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    # ── paired differences vs reference
    if boot and args.ref:
        print(f"\n=== Paired bootstrap difference vs {args.ref} (model - ref; CI excluding 0 = significant) ===")
        md += ["", f"Difference vs {args.ref} (95% paired bootstrap CI):", "",
               "| Model | ΔmAP50 | ΔmAP50:95 | ΔF1 |", "|---|---|---|---|"]
        for name in models:
            if name == args.ref:
                continue
            cells = []
            line = f"  {name:<15}"
            for k in ("mAP50", "mAP50_95", "F1"):
                diff = headline(res[name])[k] - headline(res[args.ref])[k]
                lo, hi = ci(boot[name][k] - boot[args.ref][k])
                sig = "*" if (lo > 0 or hi < 0) else " "
                line += f"  Δ{k} {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"
                cells.append(f"{diff:+.3f} ({lo:+.3f} to {hi:+.3f}){sig.strip()}")
            print(line)
            md.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nsaved {args.out}/summary.csv, per_class.csv, summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
