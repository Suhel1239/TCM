#!/usr/bin/env python3
"""Convert a COCO or MDETR annotation file into ODVG jsonl (Open-GroundingDino) for TCM acupoint training.

The input type is detected automatically:
- MDETR json (made by scripts/to_mdetr.py, every image entry has "sentences"):
  each MDETR entry becomes one ODVG line and its sentence is kept as the caption,
  so the ODVG file matches the MDETR file exactly.
- original COCO json: converted directly (caption chosen by SENTENCE_MODE).

One line per annotation (so the same image appears once per point), e.g.
    {"file_name": "165_5_JPG.rf.<hash>.png", "height": 1024, "width": 1024,
     "grounding": {"caption": "LU10", "regions": [{"bbox": [428, 900, 467, 954],
                   "phrase": "yuji", "tokens_positive": [[0, 1]]}]}}

- caption: the pinyin name or the code of the point (MDETR sentence, or SENTENCE_MODE for COCO)
- phrase:  always the pinyin name (yuji / laogong / zhongchong / shaofu)
- bbox:    [x1, y1, x2, y2] in integer pixels (COCO [x, y, w, h] converted)

Usage: set INPUT_JSON (MDETR or COCO json) and OUTPUT_JSONL below, then run
    python scripts/to_odvg.py
The original json is never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from to_mdetr import ACUPOINTS, IMAGE_EXTS, build_lookup, is_mdetr

# ============================================================================
# SETTINGS - edit these and just run:  python scripts/to_odvg.py
# ============================================================================
INPUT_JSON = r"path/to/new_mdetr.json"     # MDETR json (from to_mdetr.py) or original COCO json
OUTPUT_JSONL = r"path/to/new_odvg.jsonl"   # new ODVG jsonl to create
SENTENCE_MODE = "random"  # COCO input only (MDETR keeps its sentences): "random" = name or code at random, "name" = always yuji..., "code" = always LU10...
SEED = 42
SHUFFLE_POINTS = True     # shuffle the order of the points within one image
TOKENS_POSITIVE = [[0, 1]]
# ============================================================================


def xywh_to_xyxy(bbox) -> list[int]:
    x, y, w, h = bbox
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def odvg_record(img: dict, bbox, caption: str, phrase: str) -> dict:
    return {
        "file_name": img["file_name"],
        "height": img.get("height"),
        "width": img.get("width"),
        "grounding": {
            "caption": caption,
            "regions": [{
                "bbox": xywh_to_xyxy(bbox),
                "phrase": phrase,
                "tokens_positive": [list(t) for t in TOKENS_POSITIVE],
            }],
        },
    }


def convert_mdetr(data: dict, acupoints: dict = ACUPOINTS, source: str = "") -> tuple[list[dict], list[str]]:
    """Convert one MDETR dict to a list of ODVG records, keeping its order and sentences."""
    lookup = build_lookup(acupoints)
    by_cid = {cid: name for name, (cid, _) in acupoints.items()}
    warnings: list[str] = []

    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    skipped = 0
    records = []
    for img in data["images"]:
        anns = anns_by_img.get(img["id"], [])
        if not anns:
            warnings.append(f"{source}: MDETR entry id {img['id']} ({img['file_name']}) has no annotation, left out")
            continue
        sentence = str(img["sentences"]).strip()
        for a in anns:
            point = lookup.get(sentence.lower())
            name = point[1] if point else by_cid.get(a.get("category_id"))
            if name is None:
                skipped += 1
                continue
            cat_name = by_cid.get(a.get("category_id"))
            if cat_name is not None and cat_name != name:
                warnings.append(f"{source}: MDETR entry id {img['id']} sentence {sentence!r} does not match "
                                f"category_id {a.get('category_id')} ({cat_name}); using {name!r}")
            records.append(odvg_record(img, a["bbox"], sentence, name))

    if skipped:
        warnings.append(f"{source}: {skipped} annotation(s) whose sentence/category is not in ACUPOINTS, left out")
    return records, warnings


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
            records.append(odvg_record(img, a["bbox"], caption, name))

    for cname, n in sorted(skipped.items()):
        warnings.append(f"{source}: {n} annotation(s) of category {cname!r} are not in ACUPOINTS, left out")
    return records, warnings


def convert_file(src: Path, dst: Path, seed: int = SEED, rng: random.Random | None = None,
                 **kwargs) -> Path | None:
    """Read the MDETR or COCO json `src` and write a new ODVG jsonl `dst`."""
    if src.resolve() == dst.resolve():
        raise SystemExit(f"output {dst} is the same file as the input; choose a new file name")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        print(f"skip {src}: not a COCO/MDETR file", file=sys.stderr)
        return None
    if is_mdetr(data):
        print(f"{src}: MDETR format, keeping its sentences as captions")
        records, warnings = convert_mdetr(data, source=str(src))
    else:
        print(f"{src}: COCO format, sentence mode {kwargs.get('sentence_mode', SENTENCE_MODE)!r}")
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
    ap.add_argument("--input", type=Path, default=Path(INPUT_JSON), help="MDETR or COCO json (default: INPUT_JSON)")
    ap.add_argument("--output", type=Path, default=Path(OUTPUT_JSONL), help="new ODVG jsonl (default: OUTPUT_JSONL)")
    ap.add_argument("--sentence-mode", choices=["random", "name", "code"], default=SENTENCE_MODE,
                    help="COCO input only; MDETR input keeps its sentences")
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
