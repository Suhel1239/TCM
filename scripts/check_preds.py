#!/usr/bin/env python3
"""Quick sanity check of prediction files against the ground truth - finds why a model scores ~0.

Per model it prints:
  boxes / images         how many boxes were saved and on how many test images
  scores                 lowest / median / highest score, and how many boxes are >= 0.25 / 0.5
                         (lowest >> 0.001 means the file was saved with a confidence cut)
  category ids           which category ids the file uses vs the ids of the ground truth
  hit same class         % of true points with a box of the SAME class whose centre is within
                         0.5 x true-box diagonal (any score)
  hit any class          the same, but the box may have ANY class
  -> hit any class >> hit same class  = class ids are shifted (e.g. YOLO 0-based ids vs Roboflow ids)
  -> both low                          = boxes are in the wrong place (image size / resize / file mapping)
                                         or the model really fails on these images

EASIEST: fill in the SETTINGS block below and run   python check_preds.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

# ============================================================================
# SETTINGS
# ============================================================================
ANN_JSON = r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/train/_annotations.coco.json"
PREDS = {
    "GroundingDINO": r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/GD_predictions.coco.json",
    "GLIP":          r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/GLIP_predictions.coco.json",
    "YOLOv8m":       r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/YOLO8.json",
    "YOLOv11m":      r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/YOLO11.json",
    "YOLOv26m":      r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/YOLO26.json",
    "RT-DETR":       r"/home/suhel.khan/TCM/dataset/TCM_new_data/d2/external_test_img_d2.coco/RT-DETR.json",
}
CLASSES = ["yuji", "laogong", "zhongchong", "shaofu"]
ALPHA = 0.5
# ============================================================================


def load_results(path: str, gt_images: list) -> list:
    obj = json.load(open(path))
    if isinstance(obj, list):
        return obj
    res = obj.get("results") or obj.get("annotations") or []
    if obj.get("images"):                       # own image ids -> map to GT ids by file name
        by_name = {os.path.basename(im["file_name"]): im["id"] for im in gt_images}
        to_gt = {im["id"]: by_name.get(os.path.basename(im["file_name"])) for im in obj["images"]}
        res = [{**r, "image_id": to_gt[r["image_id"]]} for r in res if to_gt.get(r["image_id"]) is not None]
    return res


def centre(b):
    x, y, w, h = b
    return np.array([x + w / 2, y + h / 2])


def main() -> int:
    gt = json.load(open(ANN_JSON))
    cat_name = {c["id"]: c["name"].lower() for c in gt["categories"]}
    print("ground truth categories (id: name):", cat_name)
    gt_anns = [a for a in gt["annotations"] if cat_name.get(a["category_id"]) in CLASSES]
    gt_ids = {im["id"] for im in gt["images"]}
    size = {im["id"]: (im.get("width"), im.get("height")) for im in gt["images"]}
    print(f"{len(gt['images'])} images, {len(gt_anns)} true points; GT boxes per class id: "
          f"{dict(Counter(a['category_id'] for a in gt_anns))}")
    for name, path in PREDS.items():
        print(f"\n=== {name}  ({path})")
        if not os.path.isfile(path):
            print("  file not found")
            continue
        res = load_results(path, gt["images"])
        res = [r for r in res if r["image_id"] in gt_ids]
        if not res:
            print("  NO boxes on any ground-truth image id - predictions made on another json / file names differ")
            continue
        s = np.array([r.get("score", 1.0) for r in res])
        n_img = len({r["image_id"] for r in res})
        print(f"  boxes {len(res)} on {n_img}/{len(gt_ids)} images ({len(res) / max(n_img, 1):.1f} per image)")
        print(f"  scores  min {s.min():.3f}  median {np.median(s):.3f}  max {s.max():.3f}   "
              f">=0.25: {(s >= 0.25).sum()}   >=0.5: {(s >= 0.5).sum()}")
        print(f"  category ids used: {dict(Counter(r['category_id'] for r in res))}   "
              f"(names in GT: {[cat_name.get(k, '?') for k in sorted({r['category_id'] for r in res})]})")
        out_of_image = sum(1 for r in res if size[r["image_id"]][0] and
                           (centre(r["bbox"])[0] > size[r["image_id"]][0] or centre(r["bbox"])[1] > size[r["image_id"]][1]))
        if out_of_image:
            print(f"  WARNING: {out_of_image} box centres lie OUTSIDE the GT image size - coordinates are in "
                  "another image size (resize)")
        by_img = {}
        for r in res:
            by_img.setdefault(r["image_id"], []).append(r)
        hit_same = hit_any = 0
        dists = []
        for a in gt_anns:
            c, diag = centre(a["bbox"]), max(np.hypot(a["bbox"][2], a["bbox"][3]), 1e-6)
            cand = by_img.get(a["image_id"], [])
            d_any = [np.linalg.norm(centre(r["bbox"]) - c) / diag for r in cand]
            d_same = [d for d, r in zip(d_any, cand) if r["category_id"] == a["category_id"]]
            hit_any += bool(d_any) and min(d_any) <= ALPHA
            hit_same += bool(d_same) and min(d_same) <= ALPHA
            if d_same:
                dists.append(min(d_same))
        n = len(gt_anns)
        med = f"{np.median(dists):.2f} x diagonal" if dists else "no same-class box at all"
        print(f"  hit same class {hit_same / n:6.1%}   hit any class {hit_any / n:6.1%}   "
              f"median distance to nearest same-class box: {med}")
        if hit_any > 1.5 * max(hit_same, 1):
            print("  -> class ids look SHIFTED/WRONG: boxes are in the right place with the wrong class id")
    return 0


if __name__ == "__main__":
    sys.exit(main())
