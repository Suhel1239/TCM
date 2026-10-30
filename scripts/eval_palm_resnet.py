#!/usr/bin/env python3
"""Evaluate the ResNet-18 palm classifier EXACTLY as it is used in the tracking pipeline:
  - model: resnet18, fc = Sequential(Linear(512, 1), Sigmoid())  -> output = P(palm)
  - preprocessing: image read with OpenCV, HSV saturation x SATURATION, BGR -> RGB,
    Resize(224, 224), ToTensor()  (no normalisation)
  - decision: palm if P(palm) > THRESHOLD (0.4)

Test folder: one sub-folder per class
    DATA_DIR/Palm/      images with a palm
    DATA_DIR/Not_Palm/  images without a palm

Prints and saves (OUT_DIR): confusion matrix, accuracy, balanced accuracy, precision / recall / F1
for palm and non-palm, ROC-AUC, a threshold sweep, and predictions.csv / misclassified.csv.

Fill in the SETTINGS block and run:   python eval_palm_resnet.py
"""

import csv
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ============================================================================
# SETTINGS
# ============================================================================
WEIGHTS = r"/home/suhel.khan/TCM/scratch/Palm_Dataset/resnet_palm_classifier.pth"
DATA_DIR = r"/home/suhel.khan/TCM/scratch/Palm_Dataset/test"
OUT_DIR = r"/home/suhel.khan/TCM/scratch/Palm_Dataset/results_model_evaluation"
PALM_DIR, NON_PALM_DIR = "Palm", "Not_Palm"
THRESHOLD = 0.4          # as in the pipeline: palm if P(palm) > 0.4
SATURATION = 0.5         # as in the pipeline (frames are desaturated before the classifier); 1.0 = unchanged
DEVICE = "cpu"           # or "cuda"
# ============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_palm_model(weights: str, device: str):
    """the same model as in the pipeline"""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


palm_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def adjust_saturation(img_bgr, factor: float):
    if factor == 1.0:
        return img_bgr
    h, s, v = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def palm_probability(model, img_bgr, device: str) -> float:
    """same steps as is_palm() in the pipeline, but returns the probability"""
    pil_image = Image.fromarray(np.ascontiguousarray(img_bgr[..., ::-1]))   # BGR -> RGB
    x = palm_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(x).item())


def list_images():
    items = []
    for sub, label in ((PALM_DIR, 1), (NON_PALM_DIR, 0)):
        folder = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(folder):
            raise SystemExit(f"folder not found: {folder}")
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    items.append((os.path.join(root, f), label))
    if not items:
        raise SystemExit(f"no images in {DATA_DIR}")
    return items


def div(a, b):
    return a / b if b else float("nan")


def scores_at(y, p, thr):
    pred = (p > thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    pp, pr = div(tp, tp + fp), div(tp, tp + fn)        # palm precision / recall
    np_, nr = div(tn, tn + fn), div(tn, tn + fp)       # non-palm precision / recall
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": div(tp + tn, len(y)), "balanced_accuracy": (pr + nr) / 2,
            "palm_precision": pp, "palm_recall": pr, "palm_f1": div(2 * pp * pr, pp + pr),
            "nonpalm_precision": np_, "nonpalm_recall": nr, "nonpalm_f1": div(2 * np_ * nr, np_ + nr)}


def roc_auc(y, p):
    """probability that a random palm image gets a higher P(palm) than a random non-palm image"""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def main():
    items = list_images()
    print(f"{len(items)} images: {sum(l for _, l in items)} palm, {sum(1 - l for _, l in items)} non-palm")
    model = load_palm_model(WEIGHTS, DEVICE)

    paths, y, p = [], [], []
    for i, (path, label) in enumerate(items, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"  could not read {path}, skipped", file=sys.stderr)
            continue
        p.append(palm_probability(model, adjust_saturation(img, SATURATION), DEVICE))
        y.append(label)
        paths.append(path)
        if i % 100 == 0:
            print(f"\r  {i}/{len(items)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    y, p = np.array(y), np.array(p)

    m = scores_at(y, p, THRESHOLD)
    lines = [f"=== {len(y)} images, threshold {THRESHOLD} (palm if P(palm) > thr), saturation {SATURATION} ===",
             "",
             "confusion matrix (rows = true, columns = predicted)",
             f"{'':>12}{'palm':>10}{'non-palm':>10}",
             f"{'palm':>12}{m['tp']:>10}{m['fn']:>10}",
             f"{'non-palm':>12}{m['fp']:>10}{m['tn']:>10}",
             ""]
    for k in ("accuracy", "balanced_accuracy", "palm_precision", "palm_recall", "palm_f1",
              "nonpalm_precision", "nonpalm_recall", "nonpalm_f1"):
        lines.append(f"  {k:<20} {m[k]:.4f}")
    lines.append(f"  {'roc_auc':<20} {roc_auc(y, p):.4f}")
    lines += ["",
              f"palm images rejected (missed by the gate): {m['fn']} / {m['tp'] + m['fn']}",
              f"non-palm images accepted (let through):    {m['fp']} / {m['fp'] + m['tn']}",
              "",
              "threshold sweep",
              f"  {'thr':>5} {'acc':>7} {'palm_P':>7} {'palm_R':>7} {'palm_F1':>8} {'nonpalm_R':>10}"]
    for t in sorted({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, THRESHOLD}):
        s = scores_at(y, p, t)
        lines.append(f"  {t:>5.2f} {s['accuracy']:>7.4f} {s['palm_precision']:>7.4f} {s['palm_recall']:>7.4f} "
                     f"{s['palm_f1']:>8.4f} {s['nonpalm_recall']:>10.4f}")
    text = "\n".join(lines)
    print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write(text + "\n")
    names = {1: "palm", 0: "non_palm"}
    with open(os.path.join(OUT_DIR, "predictions.csv"), "w", newline="") as fa, \
         open(os.path.join(OUT_DIR, "misclassified.csv"), "w", newline="") as fb:
        wa, wb = csv.writer(fa), csv.writer(fb)
        for w in (wa, wb):
            w.writerow(["path", "true", "pred", "p_palm"])
        for path, t, pp in zip(paths, y, p):
            pred = int(pp > THRESHOLD)
            row = [path, names[int(t)], names[pred], f"{pp:.4f}"]
            wa.writerow(row)
            if pred != t:
                wb.writerow(row)
    print(f"\nsaved report.txt, predictions.csv, misclassified.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
