#!/usr/bin/env python3
"""Copy only the ANNOTATED images (images that have at least one box in the COCO json) from several
folders into a new folder, together with a new COCO json for them.

Each input folder holds images and a COCO json (e.g. Roboflow's _annotations.coco.json).
  MERGE = True   all folders go into ONE output folder with ONE json (OUT_DIR/images + OUT_DIR/JSON_NAME)
  MERGE = False  one output folder per input folder (OUT_DIR/<folder name>/images + json)
Categories are matched by NAME, so folders with different category ids can be merged.
If two folders contain an image with the same file name, the folder name is put in front of it.

EASIEST: fill in the SETTINGS block below and run   python collect_annotated.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

# ============================================================================
# SETTINGS - edit these and run:  python collect_annotated.py
# ============================================================================
INPUT_FOLDERS = [            # folders that contain images + a COCO json
    r"/home/suhel.khan/TCM/dataset/folder_1",
    r"/home/suhel.khan/TCM/dataset/folder_2",
]
JSON_NAME = "_annotations.coco.json"   # name of the json inside each folder ("" = take the only .json there)
OUT_DIR = r"/home/suhel.khan/TCM/dataset/annotated_only"
MERGE = True                 # True = one combined folder, False = one output folder per input folder
IMAGES_SUBFOLDER = "images"  # where the images go inside the output folder ("" = next to the json)
SKIP_CATEGORIES = []         # category names that do NOT count as an annotation, e.g. ["acupoints"]
# ============================================================================


def find_json(folder: str) -> str:
    if JSON_NAME:
        path = os.path.join(folder, JSON_NAME)
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found - set JSON_NAME")
        return path
    found = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    if len(found) != 1:
        raise SystemExit(f"{folder}: expected one .json, found {found} - set JSON_NAME")
    return os.path.join(folder, found[0])


def find_image(folder: str, file_name: str) -> str | None:
    for path in (os.path.join(folder, file_name), os.path.join(folder, os.path.basename(file_name)),
                 os.path.join(folder, "images", os.path.basename(file_name))):
        if os.path.isfile(path):
            return path
    return None


class Output:
    """collects images / annotations / categories of one output json"""

    def __init__(self, out_dir: str):
        self.dir = out_dir
        self.img_dir = os.path.join(out_dir, IMAGES_SUBFOLDER) if IMAGES_SUBFOLDER else out_dir
        os.makedirs(self.img_dir, exist_ok=True)
        self.images, self.annotations, self.categories = [], [], []
        self.cat_id = {}          # name -> new id
        self.names = set()
        self.info = self.licenses = None

    def category(self, cat: dict) -> int:
        name = cat["name"]
        if name not in self.cat_id:
            self.cat_id[name] = len(self.categories) + 1
            self.categories.append({**cat, "id": self.cat_id[name]})
        return self.cat_id[name]

    def add_folder(self, folder: str) -> tuple[int, int, int]:
        """-> (images in json, annotated images copied, images not found)"""
        coco = json.load(open(find_json(folder), encoding="utf-8"))
        self.info = self.info or coco.get("info")
        self.licenses = self.licenses or coco.get("licenses")
        old_cat = {c["id"]: c for c in coco.get("categories", [])}
        anns_of = {}
        for a in coco.get("annotations", []):
            cat = old_cat.get(a["category_id"])
            if cat is None or cat["name"] in SKIP_CATEGORIES:
                continue
            anns_of.setdefault(a["image_id"], []).append(a)
        prefix = os.path.basename(os.path.normpath(folder))
        copied = not_found = 0
        for im in coco["images"]:
            anns = anns_of.get(im["id"])
            if not anns:
                continue                               # no annotation -> left out
            src = find_image(folder, im["file_name"])
            if src is None:
                print(f"  WARNING: image not found: {os.path.join(folder, im['file_name'])}", file=sys.stderr)
                not_found += 1
                continue
            name = os.path.basename(im["file_name"])
            if name in self.names:                     # same name in another folder
                name = f"{prefix}_{name}"
            self.names.add(name)
            shutil.copy2(src, os.path.join(self.img_dir, name))
            new_id = len(self.images) + 1
            rel = os.path.join(IMAGES_SUBFOLDER, name) if IMAGES_SUBFOLDER else name
            self.images.append({**im, "id": new_id, "file_name": rel.replace(os.sep, "/")})
            for a in anns:
                self.annotations.append({**a, "id": len(self.annotations) + 1, "image_id": new_id,
                                         "category_id": self.category(old_cat[a["category_id"]])})
            copied += 1
        return len(coco["images"]), copied, not_found

    def save(self) -> str:
        out = {"images": self.images, "annotations": self.annotations, "categories": self.categories}
        if self.info:
            out = {"info": self.info, **out}
        if self.licenses:
            out["licenses"] = self.licenses
        path = os.path.join(self.dir, JSON_NAME or "_annotations.coco.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return path


def main() -> int:
    missing = [f for f in INPUT_FOLDERS if not os.path.isdir(f)]
    if missing:
        raise SystemExit(f"folders not found: {missing}")
    merged = Output(OUT_DIR) if MERGE else None
    for folder in INPUT_FOLDERS:
        out = merged or Output(os.path.join(OUT_DIR, os.path.basename(os.path.normpath(folder))))
        n_all, n_copied, n_missing = out.add_folder(folder)
        print(f"{folder}: {n_copied} of {n_all} images have annotations -> copied"
              + (f"  ({n_missing} annotated images not found on disk)" if n_missing else ""))
        if not MERGE:
            print(f"  saved {out.save()}  ({len(out.images)} images, {len(out.annotations)} boxes)")
    if MERGE:
        path = merged.save()
        print(f"\nsaved {path}\n  {len(merged.images)} images, {len(merged.annotations)} boxes, "
              f"categories {[c['name'] for c in merged.categories]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
