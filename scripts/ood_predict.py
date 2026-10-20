#!/usr/bin/env python3
"""Run ONE detector on the external / OOD test images and save its raw predictions as a
COCO *results* file:  [{"image_id", "category_id", "bbox": [x, y, w, h], "score"}, ...]

Run it once per model, each in the conda env that model needs; then score all of them
together with ood_eval.py.

    # Grounding DINO / GLIP (mmdet env)
    python ood_predict.py --family mmdet --name GroundingDINO --config gd.py --weights gd.pth \
        --ann external.coco.json --img-dir images/ --out preds/GroundingDINO.json
    # YOLOv8 / YOLO11 / YOLO26 (ultralytics env)
    python ood_predict.py --family yolo --name YOLO26 --weights best.pt --imgsz 640 \
        --ann external.coco.json --img-dir images/ --out preds/YOLO26.json
    # RT-DETR (transformers env)
    python ood_predict.py --family rtdetr --name RT-DETR --weights checkpoint-5291 \
        --ann external.coco.json --img-dir images/ --out preds/RT-DETR.json

Every model sees the same images with the same preprocessing; labels are mapped to the
json's category ids BY NAME, so class order / id differences between models do not matter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_CLASSES = ["yuji", "laogong", "zhongchong", "shaofu"]


def load_image_bgr(path: str, saturation: float):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if saturation != 1.0:
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        s = np.clip(s * saturation, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return img


def build_predictor(args, classes):
    """returns f(img_bgr) -> (boxes xyxy (N,4), scores (N,), names (N,) lowercase)"""
    if args.family == "mmdet":
        from mmdet.apis import init_detector, inference_detector
        model = init_detector(args.config, args.weights, device=args.device)

        def predict(img):
            r = inference_detector(model, img, text_prompt=tuple(classes), custom_entities=True).pred_instances
            labels = r.labels.cpu().numpy()
            names = np.array([classes[i].lower() if 0 <= i < len(classes) else "" for i in labels])
            return r.bboxes.cpu().numpy(), r.scores.cpu().numpy(), names
        return predict

    if args.family == "yolo":
        from ultralytics import YOLO
        model = YOLO(args.weights)
        dev = 0 if args.device.startswith("cuda") else "cpu"

        def predict(img):
            r = model.predict(img, imgsz=args.imgsz, conf=args.score_thr, iou=0.7, max_det=300,
                              device=dev, verbose=False)[0]
            names = np.array([str(model.names[int(c)]).lower() for c in r.boxes.cls.cpu().numpy()])
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), names
        return predict

    if args.family == "rtdetr":
        import cv2
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        model = AutoModelForObjectDetection.from_pretrained(args.weights).to(args.device).eval()
        proc = AutoImageProcessor.from_pretrained(args.weights)
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

        def predict(img):
            h, w = img.shape[:2]
            inputs = proc(images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                          return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = model(**inputs)
            r = proc.post_process_object_detection(out, target_sizes=[(h, w)], threshold=args.score_thr)[0]
            names = np.array([id2label.get(int(lab), "") for lab in r["labels"].cpu().numpy()])
            return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), names
        return predict

    raise SystemExit(f"unknown --family {args.family}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=["mmdet", "yolo", "rtdetr"])
    ap.add_argument("--name", required=True, help="model name used in the results table")
    ap.add_argument("--config", help="mmdet config (mmdet family only)")
    ap.add_argument("--weights", required=True, help="checkpoint / best.pt / HF checkpoint folder")
    ap.add_argument("--ann", required=True, help="COCO json of the external test set (manual annotations)")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="mmdet: prompt in TRAINING order (label i = i-th name)")
    ap.add_argument("--score-thr", type=float, default=0.01, help="keep low for AP (default 0.01)")
    ap.add_argument("--max-dets", type=int, default=100, help="per image per class (COCO: 100)")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO: the training imgsz")
    ap.add_argument("--saturation", type=float, default=1.0,
                    help="multiply HSV saturation before inference (1.0 = unchanged); use the SAME value for all models")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if args.family == "mmdet" and not args.config:
        ap.error("--config is required for --family mmdet")

    gt = json.load(open(args.ann))
    cat_id = {str(c["name"]).lower(): c["id"] for c in gt["categories"]}
    classes = [c.lower() for c in args.classes]
    missing = [c for c in classes if c not in cat_id]
    if missing:
        raise SystemExit(f"{missing} not in json categories {list(cat_id)}")

    predict = build_predictor(args, args.classes)
    results, n_missing = [], 0
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x
    for im in tqdm(gt["images"], desc=args.name):
        path = os.path.join(args.img_dir, im["file_name"])
        if not os.path.isfile(path):
            n_missing += 1
            continue
        boxes, scores, names = predict(load_image_bgr(path, args.saturation))
        for c in classes:
            idx = np.where((names == c) & (scores >= args.score_thr))[0]
            idx = idx[np.argsort(-scores[idx])][: args.max_dets]
            for i in idx:
                x1, y1, x2, y2 = (float(v) for v in boxes[i])
                results.append({"image_id": im["id"], "category_id": cat_id[c],
                                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                "score": round(float(scores[i]), 5)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"model": args.name, "family": args.family, "weights": args.weights,
               "score_thr": args.score_thr, "saturation": args.saturation, "imgsz": args.imgsz,
               "results": results}, open(args.out, "w"))
    if n_missing:
        print(f"WARNING: {n_missing} images not found in {args.img_dir}", file=sys.stderr)
    print(f"{args.name}: {len(results)} predictions on {len(gt['images']) - n_missing} images -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
