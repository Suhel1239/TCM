#!/usr/bin/env python3
"""Re-split an image dataset so that no subject (person) appears in more than one split.

Filenames are expected to look like Roboflow exports:

    002_5_JPG.rf.b8dbfca8bcc51c4d8cdcef3b5c1266b8.png
    ^^^ ^
    |   +-- image number of that subject
    +------ subject (person) ID

All images of one subject (including any augmented copies Roboflow made of
them) are kept together in a single split, which removes subject-level leakage
between train / valid / test.

Supported input layouts (detected automatically, per split folder):

  * classification folders   <split>/<class_name>/<image>
  * YOLO detection/segment.  <split>/images/<image> + <split>/labels/<stem>.txt
  * Roboflow CSV labels      <split>/_classes.csv (column "filename") + images
  * flat folder of images    <split>/<image>

By default the new splits get the same number of images as the original ones
(e.g. if the old split was 700/200/100 the new one will be ~700/200/100).
Use --ratios to choose different proportions instead.

The output is written to a new directory; the input dataset is never modified.

Usage: set INPUT_DIR / OUTPUT_DIR in the SETTINGS block below, then run
    python scripts/subject_split.py

Command-line flags override the settings, e.g.:
    python scripts/subject_split.py --input data/original --output data/subject_split
    python scripts/subject_split.py --input data/original --output data/subject_split \
        --ratios 0.7 0.15 0.15 --stratify --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# SETTINGS - edit these and just run:  python scripts/subject_split.py
# (Windows paths work as-is thanks to the r"..." prefix, e.g. r"C:\Users\me\dataset")
# ============================================================================
INPUT_DIR = r"path/to/your/dataset"            # folder containing train/ valid/ test/
OUTPUT_DIR = r"path/to/your/dataset_subject_split"  # new dataset is written here
RATIOS = None          # None = keep original split sizes, or e.g. (0.7, 0.15, 0.15)
STRATIFY = False       # True = also balance classes across splits
SEED = 42              # change to get a different (but repeatable) split
DRY_RUN = False        # True = only print the report, write nothing
OVERWRITE = False      # True = delete OUTPUT_DIR first if it already exists
# ============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "valid", "test")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
    "testing": "test",
}
# "002_5_JPG.rf.<hash>.png" -> subject "002", image "5"
FILENAME_RE = re.compile(r"^(?P<subject>[^_]+)_(?P<image>[^_.]+)")


@dataclass
class Sample:
    image: Path               # absolute path of the image file
    rel_dir: Path             # directory relative to the split root (keeps class folders)
    subject: str
    image_no: str
    original_split: str
    label_file: Path | None = None   # YOLO .txt next to it, if any
    csv_row: dict | None = None      # row from Roboflow _classes.csv, if any
    label: str | None = None         # class used for stratification


@dataclass
class Subject:
    sid: str
    samples: list[Sample] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def label(self) -> str:
        labels = Counter(s.label for s in self.samples if s.label is not None)
        return labels.most_common(1)[0][0] if labels else "__all__"


def parse_filename(name: str) -> tuple[str, str]:
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"cannot parse subject id from filename: {name!r}")
    return m.group("subject"), m.group("image")


def find_split_dirs(root: Path) -> dict[str, Path]:
    found = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.lower() in SPLIT_ALIASES:
            key = SPLIT_ALIASES[child.name.lower()]
            if key in found:
                raise SystemExit(f"two folders map to the {key!r} split: {found[key].name}, {child.name}")
            found[key] = child
    return found


def csv_label(row: dict, columns: list[str]) -> str | None:
    """Label of a Roboflow one-hot CSV row (columns after 'filename')."""
    active = [c for c in columns if str(row.get(c, "")).strip() in {"1", "1.0"}]
    return "+".join(active) if active else None


def collect_split(split: str, split_dir: Path) -> tuple[list[Sample], list[str] | None]:
    samples: list[Sample] = []

    csv_path = split_dir / "_classes.csv"
    csv_rows: dict[str, dict] = {}
    csv_fields: list[str] | None = None
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            csv_fields = [c.strip() for c in reader.fieldnames or []]
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                csv_rows[row["filename"]] = row
    label_cols = [c for c in (csv_fields or []) if c != "filename"]

    if (split_dir / "_annotations.coco.json").exists():
        raise SystemExit(
            f"{split_dir}: COCO json annotations are not supported by this script. "
            "Export the dataset from Roboflow in folder / YOLO / CSV format instead."
        )

    for img in sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        rel_dir = img.parent.relative_to(split_dir)
        subject, image_no = parse_filename(img.name)
        s = Sample(img, rel_dir, subject, image_no, split)

        # YOLO: <split>/images/x.png <-> <split>/labels/x.txt
        if rel_dir.parts and rel_dir.parts[0] == "images":
            lbl = split_dir.joinpath("labels", *rel_dir.parts[1:], img.stem + ".txt")
            if lbl.exists():
                s.label_file = lbl
                classes = sorted({ln.split()[0] for ln in lbl.read_text().splitlines() if ln.strip()})
                s.label = "+".join(classes) if classes else "background"
        elif rel_dir.parts:
            # classification folders: <split>/<class>/x.png
            s.label = rel_dir.parts[0]

        if csv_rows:
            row = csv_rows.get(img.name)
            if row is None:
                print(f"warning: {img.name} not listed in {csv_path}", file=sys.stderr)
            else:
                s.csv_row = row
                s.label = csv_label(row, label_cols)

        samples.append(s)
    return samples, csv_fields


def compute_targets(total: int, weights: list[float]) -> list[int]:
    """Integer targets proportional to weights that sum exactly to total."""
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    out = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


def assign_subjects(subjects: list[Subject], targets: list[int], rng: random.Random) -> list[list[Subject]]:
    """Greedy balanced assignment: biggest subjects first, each to the split
    that is furthest below its target (relative to the target size)."""
    order = subjects[:]
    rng.shuffle(order)                         # random tie-breaking between equal-size subjects
    order.sort(key=lambda s: s.size, reverse=True)
    buckets: list[list[Subject]] = [[] for _ in targets]
    filled = [0] * len(targets)
    for subj in order:
        def deficit(i: int) -> float:
            return (targets[i] - filled[i]) / targets[i] if targets[i] else float("-inf")
        best = max(range(len(targets)), key=deficit)
        buckets[best].append(subj)
        filled[best] += subj.size
    return buckets


def write_output(in_root: Path, out_root: Path, dir_names: dict[str, str], assignment: dict[str, str],
                 samples: list[Sample], csv_fields: list[str] | None, overwrite: bool) -> None:
    if out_root.exists():
        if not overwrite:
            raise SystemExit(f"{out_root} already exists (use --overwrite to replace it)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # keep top-level files such as data.yaml / README.roboflow.txt
    for f in in_root.iterdir():
        if f.is_file():
            shutil.copy2(f, out_root / f.name)

    csv_out: dict[str, list[dict]] = defaultdict(list)
    used_names: dict[str, set[Path]] = defaultdict(set)
    for s in samples:
        split = dir_names[assignment[s.subject]]
        dst_dir = out_root / split / s.rel_dir
        dst = dst_dir / s.image.name
        if dst in used_names[split]:
            raise SystemExit(f"duplicate file name {dst} (same name existed in two original splits)")
        used_names[split].add(dst)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s.image, dst)
        if s.label_file is not None:
            lbl_dir = out_root.joinpath(split, "labels", *s.rel_dir.parts[1:])
            lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s.label_file, lbl_dir / s.label_file.name)
        if s.csv_row is not None:
            csv_out[split].append(s.csv_row)

    if csv_fields:
        for split in dir_names.values():
            rows = csv_out.get(split, [])
            (out_root / split).mkdir(parents=True, exist_ok=True)
            with (out_root / split / "_classes.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_fields)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: r["filename"]))

    with (out_root / "split_manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "subject_id", "image_no", "label", "original_split", "new_split"])
        for s in sorted(samples, key=lambda s: (assignment[s.subject], s.subject, s.image.name)):
            w.writerow([s.image.name, s.subject, s.image_no, s.label or "",
                        s.original_split, assignment[s.subject]])


def report(samples: list[Sample], assignment: dict[str, str], splits: list[str]) -> str:
    lines = []
    old = Counter(s.original_split for s in samples)
    new = Counter(assignment[s.subject] for s in samples)
    subj_per_split = {sp: {s.subject for s in samples if assignment[s.subject] == sp} for sp in splits}
    old_subj = defaultdict(set)
    for s in samples:
        old_subj[s.subject].add(s.original_split)
    leaked = sum(1 for v in old_subj.values() if len(v) > 1)

    lines.append(f"Total images: {len(samples)}   subjects: {len(old_subj)}")
    lines.append(f"Subjects that were leaking across splits in the ORIGINAL data: {leaked}")
    lines.append("")
    lines.append(f"{'split':<8}{'old imgs':>10}{'new imgs':>10}{'new %':>8}{'subjects':>10}")
    for sp in splits:
        lines.append(f"{sp:<8}{old.get(sp, 0):>10}{new.get(sp, 0):>10}"
                     f"{100 * new.get(sp, 0) / len(samples):>7.1f}%{len(subj_per_split[sp]):>10}")

    labels = sorted({s.label for s in samples if s.label is not None})
    if labels:
        lines.append("")
        lines.append("Images per class in the new splits:")
        lines.append(f"{'class':<24}" + "".join(f"{sp:>8}" for sp in splits))
        for lab in labels:
            row = [sum(1 for s in samples if s.label == lab and assignment[s.subject] == sp) for sp in splits]
            lines.append(f"{lab[:23]:<24}" + "".join(f"{n:>8}" for n in row))

    overlap = [(a, b, subj_per_split[a] & subj_per_split[b])
               for i, a in enumerate(splits) for b in splits[i + 1:]]
    lines.append("")
    for a, b, inter in overlap:
        lines.append(f"subject overlap {a} / {b}: {len(inter)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path(INPUT_DIR),
                    help="dataset root containing train/ valid/ test/ folders (default: INPUT_DIR)")
    ap.add_argument("--output", type=Path, default=Path(OUTPUT_DIR),
                    help="where to write the new dataset (default: OUTPUT_DIR)")
    ap.add_argument("--ratios", nargs=3, type=float, metavar=("TRAIN", "VALID", "TEST"),
                    default=list(RATIOS) if RATIOS else None,
                    help="target fractions; default = keep the original split sizes")
    ap.add_argument("--stratify", action="store_true", default=STRATIFY,
                    help="also balance class distribution across splits (by each subject's majority label)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                    help="only print the report, write nothing")
    ap.add_argument("--overwrite", action="store_true", default=OVERWRITE,
                    help="replace --output if it exists")
    args = ap.parse_args(argv)

    if not args.input.is_dir():
        raise SystemExit(f"input folder not found: {args.input}\n"
                         "Set INPUT_DIR at the top of scripts/subject_split.py (or pass --input).")

    in_root = args.input.resolve()
    out_root = args.output.resolve()
    if out_root == in_root or in_root in out_root.parents:
        raise SystemExit("--output must not be inside --input")

    split_dirs = find_split_dirs(in_root)
    if not split_dirs:
        raise SystemExit(f"no train/valid/test folders found in {in_root}")

    samples: list[Sample] = []
    csv_fields: list[str] | None = None
    for split, d in split_dirs.items():
        ss, fields = collect_split(split, d)
        samples += ss
        csv_fields = csv_fields or fields
    if not samples:
        raise SystemExit("no images found")

    subjects: dict[str, Subject] = {}
    for s in samples:
        subjects.setdefault(s.subject, Subject(s.subject)).samples.append(s)

    splits = list(SPLIT_NAMES)
    if args.ratios:
        weights = args.ratios
    else:
        old = Counter(s.original_split for s in samples)
        weights = [old.get(sp, 0) for sp in splits]
    if sum(weights) <= 0 or any(w < 0 for w in weights):
        raise SystemExit("split ratios must be non-negative and not all zero")

    rng = random.Random(args.seed)
    groups: dict[str, list[Subject]] = defaultdict(list)
    for subj in subjects.values():
        groups[subj.label if args.stratify else "__all__"].append(subj)

    assignment: dict[str, str] = {}
    for key in sorted(groups):
        group = groups[key]
        targets = compute_targets(sum(s.size for s in group), weights)
        for sp, bucket in zip(splits, assign_subjects(group, targets, rng)):
            for subj in bucket:
                assignment[subj.sid] = sp

    print(report(samples, assignment, splits))

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0
    # reuse the original folder names (e.g. "val" stays "val") so data.yaml paths keep working
    dir_names = {sp: split_dirs[sp].name if sp in split_dirs else sp for sp in splits}
    write_output(in_root, out_root, dir_names, assignment, samples, csv_fields, args.overwrite)
    print(f"\nNew dataset written to {out_root}")
    print(f"Per-image assignment saved to {out_root / 'split_manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
