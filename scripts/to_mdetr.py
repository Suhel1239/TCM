#!/usr/bin/env python3
"""Convert COCO annotation files into the MDETR format used for TCM acupoint training.

COCO (one image entry, many annotations):
    images:      [{"id": 5, "file_name": "002_5_JPG.rf.<hash>.png", "height": 1024, "width": 1024}]
    annotations: [{"image_id": 5, "category_id": 3, "bbox": [...]}, ... 4 points ...]

MDETR (one image entry *per annotation*, with the point name as the sentence):
    images:      [{"file_name": ..., "height": 1024, "width": 1024, "id": 100, "sentences": "LU10"},
                  {"file_name": ..., "height": 1024, "width": 1024, "id": 101, "sentences": "HT8"}, ...]
    annotations: [{"id": 200, "image_id": 100, "bbox": [...], "category_id": 0, "area": ..., "iscrowd": 0}, ...]
    info: {}, licenses: []

Each acupoint can be written as its pinyin name or its code (yuji / LU10, ...);
SENTENCE_MODE chooses which.

Usage: set DATASET_DIR below (the folder with train/ valid/ test/, e.g. the
output of subject_split.py) and run
    python scripts/to_mdetr.py
Every *.json inside each split folder is converted in place, unless OUTPUT_NAME is set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================================
# SETTINGS - edit these and just run:  python scripts/to_mdetr.py
# ============================================================================
DATASET_DIR = r"path/to/your/dataset_subject_split"  # folder containing train/ valid/ test/
OUTPUT_NAME = None     # None = overwrite each split's json; or e.g. "mdetr.json" to write a new file
SENTENCE_MODE = "random"  # "random" = name or code at random, "name" = always yuji..., "code" = always LU10...
SEED = 42
IMAGE_ID_START = 0     # first image id in each file
ANN_ID_START = 0       # first annotation id in each file
SHUFFLE_POINTS = True  # shuffle the order of the points within one image (as in the original files)

# point name -> (MDETR category_id, code). Names are matched case-insensitively
# against the COCO category names; the codes are accepted as names too.
ACUPOINTS = {
    "yuji": (0, "LU10"),
    "laogong": (1, "P8"),
    "zhongchong": (2, "P9"),
    "shaofu": (3, "HT8"),
}
# ============================================================================

SPLIT_DIRS = {"train", "training", "valid", "val", "validation", "test", "testing"}


def build_lookup(acupoints: dict) -> dict[str, tuple[int, str, str]]:
    """lowercased name or code -> (category_id, name, code)."""
    lookup = {}
    for name, (cid, code) in acupoints.items():
        lookup[name.lower()] = (cid, name, code)
        lookup[code.lower()] = (cid, name, code)
    return lookup


def is_mdetr(data: dict) -> bool:
    return bool(data.get("images")) and all("sentences" in im for im in data["images"])


def to_int(v):
    return int(round(v)) if isinstance(v, (int, float)) else v


def convert(data: dict, rng: random.Random, sentence_mode: str = SENTENCE_MODE,
            acupoints: dict = ACUPOINTS, image_id_start: int = IMAGE_ID_START,
            ann_id_start: int = ANN_ID_START, shuffle_points: bool = SHUFFLE_POINTS,
            source: str = "") -> tuple[dict, list[str]]:
    """Convert one COCO dict to MDETR. Returns (mdetr_dict, warnings)."""
    if sentence_mode not in {"random", "name", "code"}:
        raise ValueError(f"SENTENCE_MODE must be random/name/code, got {sentence_mode!r}")
    lookup = build_lookup(acupoints)
    warnings: list[str] = []

    cat_names = {c["id"]: str(c["name"]) for c in data.get("categories", [])}
    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    skipped = defaultdict(int)
    images, annotations = [], []
    for img in sorted(data["images"], key=lambda im: im["id"]):
        anns = list(anns_by_img.get(img["id"], []))
        if not anns:
            warnings.append(f"{source}: {img['file_name']} has no annotations, left out")
            continue
        if shuffle_points:
            rng.shuffle(anns)
        for a in anns:
            cname = cat_names.get(a.get("category_id"), str(a.get("category_id")))
            point = lookup.get(cname.lower())
            if point is None:
                skipped[cname] += 1
                continue
            cid, name, code = point
            if sentence_mode == "name":
                sentence = name
            elif sentence_mode == "code":
                sentence = code
            else:
                sentence = rng.choice([name, code])

            image_id = image_id_start + len(images)
            images.append({
                "file_name": img["file_name"],
                "height": img.get("height"),
                "width": img.get("width"),
                "id": image_id,
                "sentences": sentence,
            })
            bbox = [to_int(v) for v in a["bbox"]]
            area = a.get("area", bbox[2] * bbox[3])
            annotations.append({
                "id": ann_id_start + len(annotations),
                "image_id": image_id,
                "bbox": bbox,
                "category_id": cid,
                "area": to_int(area),
                "iscrowd": a.get("iscrowd", 0),
            })

    for cname, n in sorted(skipped.items()):
        warnings.append(f"{source}: {n} annotation(s) of category {cname!r} are not in ACUPOINTS, left out")
    return {"images": images, "annotations": annotations, "info": {}, "licenses": []}, warnings


def convert_dataset(root: Path, output_name: str | None = OUTPUT_NAME, seed: int = SEED,
                    **kwargs) -> list[Path]:
    rng = random.Random(seed)
    written = []
    split_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.lower() in SPLIT_DIRS)
    if not split_dirs:
        raise SystemExit(f"no train/valid/test folders found in {root}")
    for split_dir in split_dirs:
        for jp in sorted(split_dir.glob("*.json")):
            data = json.loads(jp.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
                print(f"skip {jp}: not a COCO file", file=sys.stderr)
                continue
            if is_mdetr(data):
                print(f"skip {jp}: already in MDETR format")
                continue
            out, warnings = convert(data, rng, source=str(jp), **kwargs)
            for w in warnings:
                print("warning:", w, file=sys.stderr)
            dst = split_dir / output_name if output_name else jp
            with dst.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            n_files = len({im["file_name"] for im in out["images"]})
            print(f"{dst}: {n_files} images -> {len(out['images'])} MDETR entries")
            written.append(dst)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path(DATASET_DIR),
                    help="folder containing train/ valid/ test/ (default: DATASET_DIR)")
    ap.add_argument("--output-name", default=OUTPUT_NAME,
                    help="write to this file name instead of overwriting each json")
    ap.add_argument("--sentence-mode", choices=["random", "name", "code"], default=SENTENCE_MODE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if not args.dataset.is_dir():
        raise SystemExit(f"dataset folder not found: {args.dataset}\n"
                         "Set DATASET_DIR at the top of scripts/to_mdetr.py (or pass --dataset).")
    convert_dataset(args.dataset, args.output_name, args.seed, sentence_mode=args.sentence_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
