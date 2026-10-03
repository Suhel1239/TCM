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

Usage: set INPUT_JSON (your original COCO json) and OUTPUT_JSON (the new file
to create) below, then run
    python scripts/to_mdetr.py
The original json is never modified.
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
INPUT_JSON = r"path/to/original.json"     # original COCO json
OUTPUT_JSON = r"path/to/new_mdetr.json"   # new MDETR json to create
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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


def convert_file(src: Path, dst: Path, seed: int = SEED, rng: random.Random | None = None,
                 **kwargs) -> Path | None:
    """Read the COCO json `src` and write a new MDETR json `dst`."""
    if src.resolve() == dst.resolve():
        raise SystemExit(f"output {dst} is the same file as the input; choose a new file name")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        print(f"skip {src}: not a COCO file", file=sys.stderr)
        return None
    if is_mdetr(data):
        print(f"skip {src}: already in MDETR format")
        return None
    out, warnings = convert(data, rng or random.Random(seed), source=str(src), **kwargs)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    in_files = {Path(im["file_name"].replace("\\", "/")).name for im in data["images"]}
    n_files = len({im["file_name"] for im in out["images"]})
    print(f"{src}: {len(in_files)} images listed in the input json")
    print(f"{dst}: {n_files} images, {len(out['images'])} MDETR entries written")

    # compare with the image files sitting next to the input json
    folder = {p.name for p in src.parent.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    if folder:
        not_in_json = sorted(folder - in_files)
        print(f"{src.parent}: {len(folder)} image files in this folder")
        if not_in_json:
            print(f"  WARNING: {len(not_in_json)} image file(s) in the folder are NOT in {src.name}, "
                  f"so they have no annotations. e.g. {not_in_json[:3]}\n"
                  f"  Check that INPUT_JSON is the json from the same (new) split folder.", file=sys.stderr)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path(INPUT_JSON), help="original COCO json (default: INPUT_JSON)")
    ap.add_argument("--output", type=Path, default=Path(OUTPUT_JSON), help="new MDETR json (default: OUTPUT_JSON)")
    ap.add_argument("--sentence-mode", choices=["random", "name", "code"], default=SENTENCE_MODE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"input json not found: {args.input}\n"
                         "Set INPUT_JSON at the top of scripts/to_mdetr.py (or pass --input).")
    if convert_file(args.input, args.output, args.seed, sentence_mode=args.sentence_mode) is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
