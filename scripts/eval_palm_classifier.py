#!/usr/bin/env python3
"""Evaluate the ResNet-18 palm / non-palm classifier that gates the video pipeline.

Test data: a folder with one sub-folder per class, e.g.
    palm_eval/
      palm/       003_1_frame_033.png ...
      non_palm/   003_1_frame_000.png ...
(frames from TEST subjects only - not used to train the classifier).
Frames named <video>_frame_<n>.* (as written by the frame sampler) also get a per-video breakdown.

Reports
  - confusion matrix, accuracy, balanced accuracy
  - precision / recall / F1 for palm and non-palm (with 95% bootstrap CIs)
  - ROC-AUC and PR-AUC (palm = positive), metrics at several thresholds
  - gating impact: palm frames lost (false negatives) and non-palm frames let through (false positives)
  - per-video table
and writes predictions.csv (every frame) and misclassified.csv next to --out.

Usage
    python scripts/eval_palm_classifier.py --weights resnet18_palm.pth --data palm_eval/ --out results/palm_eval
Edit the SETTINGS block to match how the classifier was trained.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================================
# SETTINGS - must match how the classifier was trained
# ============================================================================
WEIGHTS = r"path/to/resnet18_palm.pth"
DATA_DIR = r"path/to/palm_eval"     # contains PALM_DIR and NON_PALM_DIR sub-folders
OUT_DIR = r"palm_eval_results"
PALM_DIR, NON_PALM_DIR = "palm", "non_palm"
NUM_OUTPUTS = 2       # 2 = softmax over 2 classes, 1 = single logit with sigmoid
PALM_INDEX = 1        # output index of "palm" (ImageFolder sorts alphabetically: non_palm=0, palm=1)
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet
THRESHOLD = 0.5       # palm if P(palm) >= THRESHOLD (the threshold used in the pipeline)
BATCH_SIZE = 64
N_BOOTSTRAP = 1000
VIDEO_REGEX = r"(?P<video>.+)_frame_(?P<frame>\d+)"
# ============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------- metrics (numpy only)
def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """palm = 1 = positive."""
    return {"tp": int(((y_true == 1) & (y_pred == 1)).sum()), "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()), "tn": int(((y_true == 0) & (y_pred == 0)).sum())}


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    p_prec, p_rec = _div(tp, tp + fp), _div(tp, tp + fn)
    n_prec, n_rec = _div(tn, tn + fn), _div(tn, tn + fp)
    f1 = lambda p, r: _div(2 * p * r, p + r)  # noqa: E731
    return {"accuracy": _div(tp + tn, tp + tn + fp + fn), "balanced_accuracy": (p_rec + n_rec) / 2,
            "palm_precision": p_prec, "palm_recall": p_rec, "palm_f1": f1(p_prec, p_rec),
            "nonpalm_precision": n_prec, "nonpalm_recall": n_rec, "nonpalm_f1": f1(n_prec, n_rec)}


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U formulation (ties count 0.5)."""
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision (palm = positive)."""
    if (y_true == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * (y == 1)).sum() / (y == 1).sum())


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, keys: list[str], n: int = N_BOOTSTRAP,
                 seed: int = 0) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    vals = defaultdict(list)
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        m = binary_metrics(y_true[idx], y_pred[idx])
        for k in keys:
            if not np.isnan(m[k]):
                vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"),) * 2
            for k, v in ((k, vals[k]) for k in keys)}


def video_of(path: str, regex: str = VIDEO_REGEX) -> str:
    m = re.search(regex, Path(path).stem)
    return m.group("video") if m else "?"


def report(paths: list[str], y_true: np.ndarray, p_palm: np.ndarray, threshold: float = THRESHOLD,
           n_boot: int = N_BOOTSTRAP, out=sys.stdout) -> dict:
    def p(*a):
        print(*a, file=out)

    y_pred = (p_palm >= threshold).astype(int)
    c = confusion(y_true, y_pred)
    m = binary_metrics(y_true, y_pred)
    m["roc_auc"], m["pr_auc"] = roc_auc(y_true, p_palm), average_precision(y_true, p_palm)
    keys = ["accuracy", "balanced_accuracy", "palm_precision", "palm_recall", "palm_f1", "nonpalm_recall"]
    ci = bootstrap_ci(y_true, y_pred, keys, n_boot) if n_boot else {}

    p(f"=== {len(y_true)} frames: {int((y_true == 1).sum())} palm, {int((y_true == 0).sum())} non-palm; "
      f"threshold {threshold} ===")
    p("\nconfusion matrix (rows = true, cols = predicted)")
    p(f"{'':>12}{'palm':>10}{'non-palm':>10}")
    p(f"{'palm':>12}{c['tp']:>10}{c['fn']:>10}")
    p(f"{'non-palm':>12}{c['fp']:>10}{c['tn']:>10}")
    p("\nmetric                 value   95% CI (bootstrap)")
    for k in keys + ["nonpalm_precision", "nonpalm_f1", "roc_auc", "pr_auc"]:
        lo, hi = ci.get(k, (float("nan"), float("nan")))
        p(f"  {k:<20} {m[k]:.4f}" + (f"   [{lo:.4f}, {hi:.4f}]" if k in ci else ""))

    p("\ngating impact")
    p(f"  palm frames LOST (classified non-palm, never reach the detector): {c['fn']} / {c['tp'] + c['fn']}"
      f" ({_div(c['fn'], c['tp'] + c['fn']):.2%})")
    p(f"  non-palm frames LET THROUGH to the detector: {c['fp']} / {c['fp'] + c['tn']}"
      f" ({_div(c['fp'], c['fp'] + c['tn']):.2%})")

    p("\nthreshold sweep")
    p(f"  {'thr':>5} {'acc':>7} {'palm_P':>7} {'palm_R':>7} {'palm_F1':>7} {'nonpalm_R':>9}")
    for t in sorted({0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, threshold}):
        mt = binary_metrics(y_true, (p_palm >= t).astype(int))
        p(f"  {t:>5.2f} {mt['accuracy']:>7.4f} {mt['palm_precision']:>7.4f} {mt['palm_recall']:>7.4f} "
          f"{mt['palm_f1']:>7.4f} {mt['nonpalm_recall']:>9.4f}")

    per_video = defaultdict(list)
    for i, path in enumerate(paths):
        per_video[video_of(path)].append(i)
    if len(per_video) > 1 or "?" not in per_video:
        p("\nper video")
        p(f"  {'video':<30} {'n':>5} {'palm':>5} {'acc':>7} {'palm_R':>7} {'nonpalm_R':>9}")
        for v, idx in sorted(per_video.items()):
            idx = np.array(idx)
            mv = binary_metrics(y_true[idx], y_pred[idx])
            p(f"  {v:<30} {len(idx):>5} {int(y_true[idx].sum()):>5} {mv['accuracy']:>7.4f} "
              f"{mv['palm_recall']:>7.4f} {mv['nonpalm_recall']:>9.4f}")
    return {"confusion": c, "metrics": m, "ci": ci}


# ---------------------------------------------------------------- model + data (torch)
def list_images(data_dir: Path) -> tuple[list[str], np.ndarray]:
    paths, labels = [], []
    for sub, label in ((PALM_DIR, 1), (NON_PALM_DIR, 0)):
        d = data_dir / sub
        if not d.is_dir():
            raise SystemExit(f"missing folder {d} (expected {PALM_DIR}/ and {NON_PALM_DIR}/ inside {data_dir})")
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                paths.append(str(f))
                labels.append(label)
    if not paths:
        raise SystemExit(f"no images found in {data_dir}")
    return paths, np.array(labels)


def load_model(weights: str, num_outputs: int, device):
    import torch
    from torchvision.models import resnet18

    obj = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        state = obj
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
        state = {k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()}
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_outputs)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"weights do not match a ResNet-18 with {num_outputs} outputs.\n"
                             f"  missing: {missing[:5]}\n  unexpected: {unexpected[:5]}\n"
                             "Check NUM_OUTPUTS, or whether the fc layer had another structure.")
    return model.to(device).eval()


def predict(model, paths: list[str], device, img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE,
            num_outputs: int = NUM_OUTPUTS, palm_index: int = PALM_INDEX) -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    probs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
            out = model(x)
            if num_outputs == 1:
                pr = torch.sigmoid(out.reshape(-1))
                pr = pr if palm_index == 1 else 1 - pr
            else:
                pr = torch.softmax(out, dim=1)[:, palm_index]
            probs.append(pr.cpu().numpy())
            print(f"\r  predicted {min(i + batch_size, len(paths))}/{len(paths)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return np.concatenate(probs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--num-outputs", type=int, choices=[1, 2], default=NUM_OUTPUTS)
    ap.add_argument("--palm-index", type=int, default=PALM_INDEX)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args(argv)

    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths, y_true = list_images(Path(args.data))
    model = load_model(args.weights, args.num_outputs, device)
    p_palm = predict(model, paths, device, args.img_size, num_outputs=args.num_outputs, palm_index=args.palm_index)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_txt = out / "report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        report(paths, y_true, p_palm, args.threshold, out=f)
    print(report_txt.read_text(encoding="utf-8"))

    y_pred = (p_palm >= args.threshold).astype(int)
    names = {1: "palm", 0: "non_palm"}
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f_all, \
         (out / "misclassified.csv").open("w", newline="", encoding="utf-8") as f_bad:
        w_all, w_bad = csv.writer(f_all), csv.writer(f_bad)
        for w in (w_all, w_bad):
            w.writerow(["path", "video", "true", "pred", "p_palm"])
        for path, t, pr, pp in zip(paths, y_true, y_pred, p_palm):
            row = [path, video_of(path), names[int(t)], names[int(pr)], f"{pp:.4f}"]
            w_all.writerow(row)
            if t != pr:
                w_bad.writerow(row)
    print(f"saved {report_txt}, {out / 'predictions.csv'}, {out / 'misclassified.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
