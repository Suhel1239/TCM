#!/usr/bin/env python3
"""Convert a COCO annotation file into ODVG *detection* (OD) jsonl + label map, for
training Grounding DINO / MM-Grounding-DINO as a 4-class acupoint detector.

One line per image, with ALL its points:
    {"filename": "165_5_JPG.rf.<hash>.png", "height": 1024, "width": 1024,
     "detection": {"instances": [
         {"bbox": [428, 900, 467, 954], "label": 0, "category": "yuji"},
         {"bbox": [530, 634, 574, 688], "label": 1, "category": "laogong"}, ...]}}

and a label map json:  {"0": "yuji", "1": "laogong", "2": "zhongchong", "3": "shaofu"}

In mmdet use it with
    dict(type='ODVGDataset', ann_file=<jsonl>, label_map_file=<label map>, ...)
and RandomSamplingNegPos builds the prompt from the label map ("yuji. laogong. ...").

Usage: set INPUT_JSON / OUTPUT_JSONL below and run  python to_odvg_det.py
(or pass --input / --output / --label-map). The original json is never modified.
This script is standalone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================================
# SETTINGS - edit these and just run:  python scripts/to_odvg_det.py
# ============================================================================
INPUT_JSON = r"path/to/train_anno.json"       # original COCO json
OUTPUT_JSONL = r"path/to/odvg_det_train.jsonl"  # new ODVG OD jsonl to create
LABEL_MAP_JSON = ""   # "" = next to OUTPUT_JSONL as <name>_label_map.json

# point name -> label id (0..3). Also the order of the label map / prompt.
# Keep it the same as the category order of your COCO json and `classes` in the config.
ACUPOINTS = {"yuji": 0, "laogong": 1, "zhongchong": 2, "shaofu": 3}
# point codes are accepted as category names too
CODES = {"lu10": "yuji", "p8": "laogong", "p9": "zhongchong", "ht8": "shaofu"}
# ============================================================================


def xywh_to_xyxy(bbox) -> list[int]:
    x, y, w, h = bbox
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def label_map(acupoints: dict = ACUPOINTS) -> dict[str, str]:
    return {str(i): n for n, i in sorted(acupoints.items(), key=lambda kv: kv[1])}


def convert(data: dict, acupoints: dict = ACUPOINTS, source: str = "") -> tuple[list[dict], list[str]]:
    """COCO dict -> list of ODVG OD records (one per image). Returns (records, warnings)."""
    names = {n.lower(): n for n in acupoints}
    warnings: list[str] = []
    cat_point = {}
    for c in data.get("categories", []):
        key = str(c["name"]).lower()
        key = CODES.get(key, key)
        if key in names:
            cat_point[c["id"]] = names[key]

    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    skipped = defaultdict(int)
    cat_names = {c["id"]: str(c["name"]) for c in data.get("categories", [])}
    records = []
    for img in sorted(data["images"], key=lambda im: im["id"]):
        instances = []
        for a in anns_by_img.get(img["id"], []):
            point = cat_point.get(a.get("category_id"))
            if point is None:
                skipped[cat_names.get(a.get("category_id"), str(a.get("category_id")))] += 1
                continue
            instances.append({"bbox": xywh_to_xyxy(a["bbox"]), "label": acupoints[point], "category": point})
        if not instances:
            warnings.append(f"{source}: {img['file_name']} has no acupoint annotations, left out")
            continue
        records.append({
            "filename": img["file_name"],
            "height": img.get("height"),
            "width": img.get("width"),
            "detection": {"instances": instances},
        })
    for cname, n in sorted(skipped.items()):
        warnings.append(f"{source}: {n} annotation(s) of category {cname!r} are not acupoints, left out")
    return records, warnings


def convert_file(src: Path, dst: Path, label_map_path: Path | None = None) -> tuple[Path, Path]:
    if src.resolve() == dst.resolve():
        raise SystemExit(f"output {dst} is the same file as the input; choose a new file name")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise SystemExit(f"{src}: not a COCO json")
    if data["images"] and all("sentences" in im for im in data["images"]):
        raise SystemExit(f"{src}: is MDETR, expected the original COCO json")
    records, warnings = convert(data, source=str(src))
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lm = label_map_path or dst.with_name(dst.stem + "_label_map.json")
    lm.write_text(json.dumps(label_map(), indent=2), encoding="utf-8")
    n_inst = sum(len(r["detection"]["instances"]) for r in records)
    print(f"{src}: {len(data['images'])} images in the input json")
    print(f"{dst}: {len(records)} images, {n_inst} boxes written")
    print(f"{lm}: label map {label_map()}")
    return dst, lm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path(INPUT_JSON), help="original COCO json")
    ap.add_argument("--output", type=Path, default=Path(OUTPUT_JSONL), help="new ODVG OD jsonl")
    ap.add_argument("--label-map", type=Path, default=Path(LABEL_MAP_JSON) if LABEL_MAP_JSON else None,
                    help="label map json (default: <output>_label_map.json)")
    args = ap.parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"input json not found: {args.input}\n"
                         "Set INPUT_JSON at the top of to_odvg_det.py (or pass --input).")
    convert_file(args.input, args.output, args.label_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
