#!/usr/bin/env python3
"""Find out WHY the COCO mAP of a Grounding DINO checkpoint is low.

Reads the predictions saved by mmdetection's  tools/test.py ... --out result.pkl
and the original COCO json used for that test, and reports:

  1. image size check     json width/height vs. the size the model saw (ori_shape)
  2. class-agnostic hits  how many points get a box with IoU >= 0.5, ignoring the label
  3. label confusion      for each true point, which label the best box was given
  4. examples             true box vs. the top boxes for a few images

Reading the result:
  - (2) high but mAP low       -> boxes are right, labels/prompt are wrong (see 3)
  - (2) low, boxes far off     -> coordinates/scale problem (see 1 and 4)
  - (2) low, boxes nearby      -> model localises poorly with this prompt

Usage:
    python scripts/diagnose_preds.py --pred work_dirs/result.pkl \
        --ann path/to/test/_annotations.coco.json \
        --classes yuji laogong zhongchong shaofu
--classes must be the same tuple, in the same order, as `classes` in the test config.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

IOU_THR = 0.5


def to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def iou_xyxy(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU of each box in `boxes` (N, 4) with one box `gt` (4,), all [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(boxes[:, 0], gt[0])
    y1 = np.maximum(boxes[:, 1], gt[1])
    x2 = np.minimum(boxes[:, 2], gt[2])
    y2 = np.minimum(boxes[:, 3], gt[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_g = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return inter / np.maximum(area_b + area_g - inter, 1e-9)


def load_preds(path: str) -> dict:
    with open(path, "rb") as f:
        results = pickle.load(f)
    preds = {}
    for r in results:
        p = r["pred_instances"]
        preds[r["img_id"]] = {
            "bboxes": to_np(p["bboxes"]).reshape(-1, 4),
            "scores": to_np(p["scores"]).reshape(-1),
            "labels": to_np(p["labels"]).reshape(-1).astype(int),
            "ori_shape": tuple(int(v) for v in r.get("ori_shape", ())[:2]),
        }
    return preds


def diagnose(preds: dict, coco: dict, classes: list[str], n_examples: int = 3, out=sys.stdout) -> dict:
    def p(*a):
        print(*a, file=out)

    name_of_cat = {c["id"]: c["name"] for c in coco["categories"]}
    # same rule as mmdet: label i -> i-th category of the json (json order) whose name is in classes
    cat_ids = [c["id"] for c in coco["categories"] if c["name"] in classes]
    label_to_name = {i: name_of_cat[cid] for i, cid in enumerate(cat_ids)}
    prompt_of_label = dict(enumerate(classes))
    images = {im["id"]: im for im in coco["images"]}
    gts = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in cat_ids:
            x, y, w, h = a["bbox"]
            gts[a["image_id"]].append((name_of_cat[a["category_id"]], np.array([x, y, x + w, y + h], float)))

    p("=== label mapping (prompt text -> json category it is scored as) ===")
    for i in range(len(classes)):
        flag = "" if prompt_of_label[i] == label_to_name.get(i) else "   <-- MISMATCH"
        p(f"  label {i}: prompt '{prompt_of_label[i]}' -> scored as '{label_to_name.get(i)}'{flag}")
    missing = [c for c in classes if c not in name_of_cat.values()]
    if missing:
        p(f"  WARNING: {missing} are not category names in the json")

    p("\n=== 1. image size: json vs. model input (ori_shape) ===")
    size_bad = []
    for img_id, pr in preds.items():
        im = images.get(img_id)
        if im is None:
            continue
        if pr["ori_shape"] and pr["ori_shape"] != (im.get("height"), im.get("width")):
            size_bad.append((im["file_name"], (im.get("height"), im.get("width")), pr["ori_shape"]))
    missing_ids = [i for i in gts if i not in preds]
    p(f"  {len(preds)} predicted images, {len(gts)} annotated images, {len(missing_ids)} annotated images have no prediction")
    if size_bad:
        p(f"  WARNING: {len(size_bad)} images differ in size, e.g. (file, json h/w, real h/w): {size_bad[:3]}")
    else:
        p("  OK: sizes match")

    n_gt = hit_any = hit_label = top1_hit = 0
    confusion = defaultdict(Counter)
    best_ious = []
    for img_id, objs in gts.items():
        pr = preds.get(img_id)
        for name, gt in objs:
            n_gt += 1
            if pr is None or len(pr["bboxes"]) == 0:
                confusion[name]["<no box>"] += 1
                best_ious.append(0.0)
                continue
            ious = iou_xyxy(pr["bboxes"], gt)
            best = int(np.argmax(ious))
            best_ious.append(float(ious[best]))
            if ious[best] >= IOU_THR:
                hit_any += 1
                confusion[name][label_to_name.get(int(pr["labels"][best]), "?")] += 1
            else:
                confusion[name]["<IoU<0.5>"] += 1
            same = np.array([label_to_name.get(int(lb)) == name for lb in pr["labels"]])
            if same.any() and ious[same].max() >= IOU_THR:
                hit_label += 1
            if same.any():
                top = np.argmax(np.where(same, pr["scores"], -1))
                top1_hit += ious[top] >= IOU_THR

    p(f"\n=== 2. recall at IoU {IOU_THR} over {n_gt} true points ===")
    if n_gt:
        p(f"  any label     : {hit_any / n_gt:.3f}  (some box overlaps the point, label ignored)")
        p(f"  correct label : {hit_label / n_gt:.3f}  (some box with the right label overlaps)")
        p(f"  top-1 of label: {top1_hit / n_gt:.3f}  (highest-scoring box of that label overlaps)")
        p(f"  median best IoU: {float(np.median(best_ious)):.3f}")

    p("\n=== 3. which label the best-overlapping box got ===")
    for name in classes:
        p(f"  true {name:<12}: {dict(confusion[name].most_common())}")

    p("\n=== 4. examples (x1, y1, x2, y2) ===")
    for img_id in list(gts)[:n_examples]:
        im, pr = images[img_id], preds.get(img_id)
        p(f"  {im['file_name']}  (json {im.get('width')}x{im.get('height')})")
        for name, gt in gts[img_id]:
            p(f"    true {name:<12} {np.round(gt).astype(int).tolist()}")
        if pr is not None:
            for k in np.argsort(-pr["scores"])[:5]:
                p(f"    pred {label_to_name.get(int(pr['labels'][k]), '?'):<12} "
                  f"{np.round(pr['bboxes'][k]).astype(int).tolist()} score {pr['scores'][k]:.3f}")

    return {"n_gt": n_gt, "recall_any": hit_any / max(n_gt, 1), "recall_label": hit_label / max(n_gt, 1),
            "recall_top1": top1_hit / max(n_gt, 1), "size_mismatch": len(size_bad)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="result.pkl from tools/test.py --out")
    ap.add_argument("--ann", required=True, help="the COCO json used in the test config")
    ap.add_argument("--classes", nargs="+", required=True, help="same names/order as `classes` in the test config")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)
    with open(args.ann, encoding="utf-8") as f:
        coco = json.load(f)
    diagnose(load_preds(args.pred), coco, args.classes, args.examples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
