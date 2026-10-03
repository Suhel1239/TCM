# TCM

## Subject-level dataset split

Each person (subject) has ~7 images, named like
`002_5_JPG.rf.b8dbfca8bcc51c4d8cdcef3b5c1266b8.png` (`002` = subject ID, `5` = image number).
The original train/valid/test split mixed images of the same person across splits
(subject-level leakage). `scripts/subject_split.py` rebuilds the splits so that **every
subject is in exactly one split**, while keeping split sizes the same (or within one
subject) as before.

**Easiest:** open `scripts/subject_split.py`, set `INPUT_DIR` and `OUTPUT_DIR` in the
`SETTINGS` block at the top (plus `RATIOS` / `STRATIFY` if you like), then run:

```bash
python scripts/subject_split.py
```

Command-line flags still work and override the settings:

```bash
# keep the original split sizes (default)
python scripts/subject_split.py --input path/to/dataset --output path/to/dataset_subject_split

# or choose ratios, and balance classes across splits too
python scripts/subject_split.py --input path/to/dataset --output path/to/dataset_subject_split \
    --ratios 0.7 0.15 0.15 --stratify --seed 42

# preview only
python scripts/subject_split.py --input path/to/dataset --output out --dry-run
```

`--input` must contain `train/`, `valid/` (or `val/`) and `test/` folders. Supported layouts:
class folders (`train/<class>/img.png`), YOLO (`train/images` + `train/labels`), Roboflow
`_classes.csv`, JSON annotation files (e.g. `train/anno.json`), or a flat folder.

**JSON annotations:** the `*.json` in each split folder (e.g. `anno.json`, or `train.json` / `val.json` / `test.json`) is re-split along
with the images, and each new split gets one file with the same name and format containing
only its images. Supported formats: COCO (`images` / `annotations` / `categories` — image
and annotation ids are renumbered per split, categories and `info`/`licenses` are kept),
a dict keyed by image filename, or a list of records with a `filename` / `file_name` field. The input is never modified. The output contains the new
splits, any top-level files (e.g. `data.yaml`), and `split_manifest.csv` listing every
image's subject, old split and new split. The printed report confirms 0 subject overlap
between splits.

Since all of a subject's images (including Roboflow augmented copies) stay together,
a split can differ from its target by at most about one subject's worth of images.

Tests: `python -m pytest tests`
