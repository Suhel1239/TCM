#!/usr/bin/env python3
"""Run ONE detector (and optionally the ResNet-18 palm classifier) on EVERY frame of the test
videos and save the raw outputs, so that trackers can be compared offline on identical detections
(run_trackers.py) and every model is scored the same way (compare_trackers.py).

Writes to --out:
  det.csv     video, frame, point, x1, y1, x2, y2, score   (top --top-k boxes per point per frame)
  palm.csv    video, frame, p_palm                          (only with --palm-weights)
  videos.csv  video, n_frames, fps, duration_s, width, height   (for the paper: number / duration)

EASIEST: fill in the SETTINGS block below and run   python scripts/video_dump.py
(in each conda env: models whose library is not installed in the current env are skipped, so run it
once in the mmdet env, once in the YOLO env and once in the rtdetr env).

Frame numbers are 0-based decode order (cv2.VideoCapture), which must be the numbering used in the
annotated keyframe names (<video>_frame_<n>.*).  The video name is the file stem.

    # Grounding DINO / GLIP (mmdet env)
    python scripts/video_dump.py --family mmdet --name GroundingDINO --config gd.py --weights gd.pth \\
        --videos test_videos/ --out video_dumps/GroundingDINO --palm-weights resnet18_palm.pth
    # YOLO (ultralytics env), RT-DETR (transformers env)
    python scripts/video_dump.py --family yolo --name YOLO26 --weights best.pt --imgsz 640 \\
        --videos test_videos/ --out video_dumps/YOLO26
    python scripts/video_dump.py --family rtdetr --name RT-DETR --weights checkpoint-5291 \\
        --videos test_videos/ --out video_dumps/RT-DETR
The palm classifier does not depend on the detector: run it once (with any model) and pass the same
palm.csv to run_trackers.py for every model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_predict import DEFAULT_CLASSES, build_predictor  # noqa: E402

# ============================================================================
# SETTINGS - edit these and run:  python scripts/video_dump.py
# ============================================================================
VIDEOS = [r"/home/suhel.khan/TCM/video_eval/test_videos"]        # folder(s) and/or video files
OUT_ROOT = r"/home/suhel.khan/TCM/video_eval/video_dumps"        # writes OUT_ROOT/<model name>/det.csv

# every model to evaluate (family: mmdet = Grounding DINO / GLIP, yolo, rtdetr)
MODELS = [
    dict(name="GroundingDINO", family="mmdet",
         config=r"/home/suhel.khan/TCM/For_Videos/model_weights_GD/after_sub_lev_split_only_det/train_pure_detection.py",
         weights=r"/home/suhel.khan/TCM/For_Videos/model_weights_GD/after_sub_lev_split_only_det/epoch_28.pth",
         classes=["laogong", "shaofu", "yuji", "zhongchong"]),     # prompt in TRAINING order
    dict(name="GLIP", family="mmdet",
         config=r"/path/to/glip_config.py",
         weights=r"/path/to/glip_epoch.pth",
         classes=["laogong", "shaofu", "yuji", "zhongchong"]),
    dict(name="YOLOv8", family="yolo", imgsz=640,
         weights=r"/home/suhel.khan/TCM/For_Videos/YOLO/runs/ResultsYOLOv8/experiment1/weights/best.pt"),
    dict(name="YOLO11", family="yolo", imgsz=640,
         weights=r"/home/suhel.khan/TCM/For_Videos/YOLO/runs/ResultsYOLOv11/experiment1/weights/best.pt"),
    dict(name="YOLO26", family="yolo", imgsz=640,
         weights=r"/home/suhel.khan/TCM/For_Videos/YOLO/runs/ResultsYOLOv26/experiment1/weights/best.pt"),
    dict(name="RT-DETR", family="rtdetr",
         weights=r"/home/suhel.khan/TCM/rtdetr/weights_and_outputs/checkpoint-5291"),
]
RUN_ONLY = []            # e.g. ["YOLO26"]; empty = every model whose library is installed in this env
SKIP_DONE = True         # skip a model whose det.csv already exists

# ResNet-18 palm classifier: run once, writes OUT_ROOT/palm/palm.csv ("" = skip)
PALM_WEIGHTS = r"/path/to/resnet18_palm.pth"
PALM_NUM_OUTPUTS = 2     # 2 = softmax over 2 classes, 1 = single logit
PALM_INDEX = 1           # output index of "palm" (ImageFolder: non_palm=0, palm=1)
PALM_IMG_SIZE = 224

SCORE_THR = 0.02         # lowest score kept
TOP_K = 5                # boxes kept per point per frame
SATURATION = 1.0         # HSV saturation factor, the same for every model
DEVICE = "cuda:0"
# ============================================================================

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}
LIBRARY = {"mmdet": "mmdet", "yolo": "ultralytics", "rtdetr": "transformers"}


def list_videos(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    if not out:
        raise SystemExit(f"no videos in {paths}")
    return out


def palm_predictor(weights: str, num_outputs: int, palm_index: int, img_size: int, device: str):
    """-> f(frame_bgr) -> P(palm), same preprocessing as eval_palm_classifier.py"""
    import cv2
    import torch
    from eval_palm_classifier import MEAN, STD, load_model

    dev = torch.device(device)
    model = load_model(weights, num_outputs, dev)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    def predict(frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB)
        x = ((torch.from_numpy(rgb).permute(2, 0, 1).float() / 255 - mean) / std)[None].to(dev)
        with torch.no_grad():
            out = model(x)
        if num_outputs == 1:
            p = float(torch.sigmoid(out.reshape(-1))[0])
            return p if palm_index == 1 else 1 - p
        return float(torch.softmax(out, dim=1)[0, palm_index])
    return predict


def adjust_saturation(img, saturation: float):
    if saturation == 1.0:
        return img
    import cv2
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    s = np.clip(s * saturation, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def dump(videos: list[Path], predict, palm, classes: list[str], out_dir: Path, score_thr: float, top_k: int,
         saturation: float = 1.0, name: str = "") -> None:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = [c.lower() for c in classes]
    f_det = (out_dir / "det.csv").open("w", newline="", encoding="utf-8")
    f_palm = (out_dir / "palm.csv").open("w", newline="", encoding="utf-8") if palm else None
    w_det = csv.writer(f_det)
    w_det.writerow(["video", "frame", "point", "x1", "y1", "x2", "y2", "score"])
    w_palm = csv.writer(f_palm) if f_palm else None
    if w_palm:
        w_palm.writerow(["video", "frame", "p_palm"])
    meta = []
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"WARNING: cannot open {vp}", file=sys.stderr)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % 100 == 0:
                print(f"\r  {name} {vp.stem}: frame {n}/{n_total}", end="", file=sys.stderr)
            if predict is not None:
                boxes, scores, names = predict(adjust_saturation(frame, saturation))
                for c in classes:
                    idx = np.where((names == c) & (scores >= score_thr))[0]
                    for i in idx[np.argsort(-scores[idx])][:top_k]:
                        x1, y1, x2, y2 = (round(float(v), 2) for v in boxes[i])
                        w_det.writerow([vp.stem, n, c, x1, y1, x2, y2, round(float(scores[i]), 5)])
            if w_palm:
                w_palm.writerow([vp.stem, n, round(palm(frame), 5)])
            n += 1
        cap.release()
        print(file=sys.stderr)
        meta.append({"video": vp.stem, "n_frames": n, "fps": round(fps, 3),
                     "duration_s": round(n / fps, 2) if fps else "", "width": w, "height": h})
    f_det.close()
    if f_palm:
        f_palm.close()
    with (out_dir / "videos.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["video", "n_frames", "fps", "duration_s", "width", "height"])
        wr.writeheader()
        wr.writerows(meta)
    tot = sum(m["n_frames"] for m in meta)
    dur = sum(m["duration_s"] or 0 for m in meta)
    print(f"{len(meta)} videos, {tot} frames, {dur:.1f} s -> {out_dir}")
    for m in meta:
        print(f"  {m['video']}: {m['n_frames']} frames, {m['fps']} fps, {m['duration_s']} s, {m['width']}x{m['height']}")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_settings() -> int:
    """run everything from the SETTINGS block"""
    videos = list_videos(VIDEOS)
    out_root = Path(OUT_ROOT)
    if PALM_WEIGHTS:
        palm_dir = out_root / "palm"
        if SKIP_DONE and (palm_dir / "palm.csv").is_file():
            print(f"palm classifier: {palm_dir / 'palm.csv'} exists, skipped")
        elif not has_module("torchvision"):
            print("palm classifier: torchvision not installed in this env, skipped")
        else:
            dev = DEVICE if DEVICE.startswith("cuda") or DEVICE == "cpu" else "cpu"
            palm = palm_predictor(PALM_WEIGHTS, PALM_NUM_OUTPUTS, PALM_INDEX, PALM_IMG_SIZE, dev)
            dump(videos, None, palm, [], palm_dir, SCORE_THR, TOP_K, name="palm")
    for m in MODELS:
        name, family = m["name"], m["family"]
        if RUN_ONLY and name not in RUN_ONLY:
            continue
        if not has_module(LIBRARY[family]):
            print(f"{name}: {LIBRARY[family]} not installed in this env, skipped (run it in its own env)")
            continue
        out = out_root / name
        if SKIP_DONE and (out / "det.csv").is_file():
            print(f"{name}: {out / 'det.csv'} exists, skipped")
            continue
        classes = m.get("classes", DEFAULT_CLASSES)
        predict = build_predictor(SimpleNamespace(family=family, config=m.get("config"), weights=m["weights"],
                                                  device=DEVICE, score_thr=SCORE_THR, imgsz=m.get("imgsz", 640)),
                                  classes)
        dump(videos, predict, None, classes, out, SCORE_THR, TOP_K, SATURATION, name)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["mmdet", "yolo", "rtdetr", "none"], required=True,
                    help="'none' = only the palm classifier (and videos.csv)")
    ap.add_argument("--name", default="")
    ap.add_argument("--config", help="mmdet config")
    ap.add_argument("--weights", help="detector checkpoint / best.pt / HF folder")
    ap.add_argument("--videos", nargs="+", required=True, help="video files or folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="mmdet: prompt in TRAINING order (label i = i-th name)")
    ap.add_argument("--score-thr", type=float, default=0.02, help="lowest score kept (default 0.02)")
    ap.add_argument("--top-k", type=int, default=5, help="boxes kept per point per frame")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO: the training imgsz")
    ap.add_argument("--saturation", type=float, default=1.0, help="HSV saturation factor, same for all models")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--palm-weights", help="ResNet-18 palm classifier; writes palm.csv")
    ap.add_argument("--palm-num-outputs", type=int, choices=[1, 2], default=2)
    ap.add_argument("--palm-index", type=int, default=1, help="output index of 'palm' (ImageFolder: 1)")
    ap.add_argument("--palm-img-size", type=int, default=224)
    args = ap.parse_args(argv)
    if args.family == "mmdet" and not args.config:
        ap.error("--config is required for --family mmdet")
    if args.family != "none" and not args.weights:
        ap.error("--weights is required")
    if args.family == "none" and not args.palm_weights:
        ap.error("--family none needs --palm-weights")

    predict = None
    if args.family != "none":
        predict = build_predictor(SimpleNamespace(family=args.family, config=args.config, weights=args.weights,
                                                  device=args.device, score_thr=args.score_thr, imgsz=args.imgsz),
                                  args.classes)
    palm = (palm_predictor(args.palm_weights, args.palm_num_outputs, args.palm_index, args.palm_img_size,
                           args.device if args.device.startswith("cuda") or args.device == "cpu" else "cpu")
            if args.palm_weights else None)
    dump(list_videos(args.videos), predict, palm, args.classes, Path(args.out), args.score_thr, args.top_k,
         args.saturation, args.name or args.family)
    return 0


if __name__ == "__main__":
    sys.exit(run_settings() if len(sys.argv) == 1 else main())
