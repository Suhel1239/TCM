#!/usr/bin/env python3
"""Convert a COCO annotation file into ODVG jsonl (Open-GroundingDino) for TCM acupoint training.

One line per annotation (so the same image appears once per point), e.g.
    {"file_name": "165_5_JPG.rf.<hash>.png", "height": 1024, "width": 1024,
     "grounding": {"caption": "LU10", "regions": [{"bbox": [428, 900, 467, 954],
                   "phrase": "yuji", "tokens_positive": [[0, 1]]}]}}

- caption: the pinyin name or the code of the point (SENTENCE_MODE, as in to_mdetr.py)
- phrase:  always the pinyin name (yuji / laogong / zhongchong / shaofu)
- bbox:    [x1, y1, x2, y2] in integer pixels (COCO [x, y, w, h] converted)

Usage: set INPUT_JSON (original COCO json) and OUTPUT_JSONL below, then run
    python scripts/to_odvg.py
The original json is never modified. This script is standalone (it does not need to_mdetr.py).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================================
# SETTINGS - edit these and just run:  python scripts/to_odvg.py
# ============================================================================
INPUT_JSON = r"path/to/original.json"      # original COCO json
OUTPUT_JSONL = r"path/to/new_odvg.jsonl"   # new ODVG jsonl to create
SENTENCE_MODE = "random"  # "random" = name or code at random, "name" = always yuji..., "code" = always LU10...
SEED = 42
SHUFFLE_POINTS = True     # shuffle the order of the points within one image
TOKENS_POSITIVE = [[0, 1]]

# point name -> (category_id, code), same as in to_mdetr.py. Names are matched
# case-insensitively against the COCO category names; the codes are accepted as names too.
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


def xywh_to_xyxy(bbox) -> list[int]:
    x, y, w, h = bbox
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def convert(data: dict, rng: random.Random, sentence_mode: str = SENTENCE_MODE,
            acupoints: dict = ACUPOINTS, shuffle_points: bool = SHUFFLE_POINTS,
            source: str = "") -> tuple[list[dict], list[str]]:
    """Convert one COCO dict to a list of ODVG records. Returns (records, warnings)."""
    if sentence_mode not in {"random", "name", "code"}:
        raise ValueError(f"SENTENCE_MODE must be random/name/code, got {sentence_mode!r}")
    lookup = build_lookup(acupoints)
    warnings: list[str] = []

    cat_names = {c["id"]: str(c["name"]) for c in data.get("categories", [])}
    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    skipped = defaultdict(int)
    records = []
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
            _, name, code = point
            if sentence_mode == "name":
                caption = name
            elif sentence_mode == "code":
                caption = code
            else:
                caption = rng.choice([name, code])
            records.append({
                "file_name": img["file_name"],
                "height": img.get("height"),
                "width": img.get("width"),
                "grounding": {
                    "caption": caption,
                    "regions": [{
                        "bbox": xywh_to_xyxy(a["bbox"]),
                        "phrase": name,
                        "tokens_positive": [list(t) for t in TOKENS_POSITIVE],
                    }],
                },
            })

    for cname, n in sorted(skipped.items()):
        warnings.append(f"{source}: {n} annotation(s) of category {cname!r} are not in ACUPOINTS, left out")
    return records, warnings


def convert_file(src: Path, dst: Path, seed: int = SEED, rng: random.Random | None = None,
                 **kwargs) -> Path | None:
    """Read the COCO json `src` and write a new ODVG jsonl `dst`."""
    if src.resolve() == dst.resolve():
        raise SystemExit(f"output {dst} is the same file as the input; choose a new file name")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        print(f"skip {src}: not a COCO file", file=sys.stderr)
        return None
    if is_mdetr(data):
        print(f"skip {src}: is MDETR, expected the original COCO json", file=sys.stderr)
        return None
    records, warnings = convert(data, rng or random.Random(seed), source=str(src), **kwargs)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    in_files = {Path(im["file_name"].replace("\\", "/")).name for im in data["images"]}
    n_files = len({r["file_name"] for r in records})
    print(f"{src}: {len(in_files)} images listed in the input json")
    print(f"{dst}: {n_files} images, {len(records)} ODVG lines written")

    folder = {p.name for p in src.parent.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    not_in_json = sorted(folder - in_files)
    if not_in_json:
        print(f"  WARNING: {len(not_in_json)} image file(s) in {src.parent} are NOT in {src.name}, "
              f"so they have no annotations. e.g. {not_in_json[:3]}", file=sys.stderr)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path(INPUT_JSON), help="original COCO json (default: INPUT_JSON)")
    ap.add_argument("--output", type=Path, default=Path(OUTPUT_JSONL), help="new ODVG jsonl (default: OUTPUT_JSONL)")
    ap.add_argument("--sentence-mode", choices=["random", "name", "code"], default=SENTENCE_MODE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"input json not found: {args.input}\n"
                         "Set INPUT_JSON at the top of scripts/to_odvg.py (or pass --input).")
    if convert_file(args.input, args.output, args.seed, sentence_mode=args.sentence_mode) is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
