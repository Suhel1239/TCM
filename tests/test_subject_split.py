import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import subject_split  # noqa: E402


def make_name(subject: int, img: int) -> str:
    return f"{subject:03d}_{img}_JPG.rf.{random.getrandbits(128):032x}.png"


def build_leaky_dataset(root: Path, layout: str, n_subjects: int = 60, per_subject: int = 7):
    """Random image -> split assignment, so nearly every subject leaks."""
    rng = random.Random(0)
    splits = ["train"] * 70 + ["valid"] * 20 + ["test"] * 10
    csv_rows = defaultdict(list)
    for sid in range(1, n_subjects + 1):
        cls = "classA" if sid % 3 else "classB"
        for k in range(1, per_subject + 1):
            split = rng.choice(splits)
            name = make_name(sid, k)
            if layout == "folders":
                d = root / split / cls
            elif layout == "yolo":
                d = root / split / "images"
                (root / split / "labels").mkdir(parents=True, exist_ok=True)
                (root / split / "labels" / (Path(name).stem + ".txt")).write_text(
                    f"{0 if cls == 'classA' else 1} 0.5 0.5 0.2 0.2\n")
            else:  # csv
                d = root / split
                csv_rows[split].append({"filename": name,
                                        "classA": int(cls == "classA"), "classB": int(cls == "classB")})
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(b"x")
    for split, rows in csv_rows.items():
        with (root / split / "_classes.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "classA", "classB"])
            w.writeheader()
            w.writerows(rows)
    (root / "data.yaml").write_text("train: ../train/images\n")


def subjects_by_split(out: Path) -> dict[str, set[str]]:
    res = defaultdict(set)
    for p in out.rglob("*.png"):
        split = p.relative_to(out).parts[0]
        res[split].add(p.name.split("_")[0])
    return res


@pytest.mark.parametrize("layout", ["folders", "yolo", "csv"])
@pytest.mark.parametrize("stratify", [False, True])
def test_no_subject_leakage_and_sizes_kept(tmp_path, layout, stratify):
    src, out = tmp_path / "src", tmp_path / "out"
    build_leaky_dataset(src, layout)
    old_counts = Counter(p.relative_to(src).parts[0] for p in src.rglob("*.png"))

    args = ["--input", str(src), "--output", str(out)]
    assert subject_split.main(args + (["--stratify"] if stratify else [])) == 0

    by_split = subjects_by_split(out)
    assert not (by_split["train"] & by_split["valid"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["valid"] & by_split["test"])

    new_counts = Counter(p.relative_to(out).parts[0] for p in out.rglob("*.png"))
    assert sum(new_counts.values()) == sum(old_counts.values())
    for sp in ("train", "valid", "test"):
        # within one subject (7 images) per stratum of the original size
        assert abs(new_counts[sp] - old_counts[sp]) <= 7 * (2 if stratify else 1)

    assert (out / "data.yaml").exists()
    if layout == "yolo":
        for img in out.rglob("images/*.png"):
            assert (img.parent.parent / "labels" / (img.stem + ".txt")).exists()
    if layout == "csv":
        for sp in ("train", "valid", "test"):
            with (out / sp / "_classes.csv").open() as f:
                listed = {r["filename"] for r in csv.DictReader(f)}
            assert listed == {p.name for p in (out / sp).glob("*.png")}


def test_custom_ratios(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    build_leaky_dataset(src, "folders", n_subjects=100)
    subject_split.main(["--input", str(src), "--output", str(out), "--ratios", "0.8", "0.1", "0.1"])
    counts = Counter(p.relative_to(out).parts[0] for p in out.rglob("*.png"))
    assert abs(counts["train"] - 560) <= 7
    assert abs(counts["valid"] - 70) <= 7
    assert abs(counts["test"] - 70) <= 7


def test_deterministic(tmp_path):
    src = tmp_path / "src"
    build_leaky_dataset(src, "folders")
    for name in ("a", "b"):
        subject_split.main(["--input", str(src), "--output", str(tmp_path / name)])
    assert (tmp_path / "a" / "split_manifest.csv").read_text() == (tmp_path / "b" / "split_manifest.csv").read_text()


def test_parse_filename():
    assert subject_split.parse_filename("002_5_JPG.rf.b8dbfca8bcc51c4d8cdcef3b5c1266b8.png") == ("002", "5")


def test_refuses_existing_output(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    build_leaky_dataset(src, "folders", n_subjects=10)
    out.mkdir()
    with pytest.raises(SystemExit):
        subject_split.main(["--input", str(src), "--output", str(out)])


def build_json_dataset(root: Path, fmt: str, n_subjects: int = 40, per_subject: int = 7):
    """Each split folder holds images + anno.json in the given format (leaky split)."""
    rng = random.Random(1)
    splits = ["train"] * 70 + ["valid"] * 20 + ["test"] * 10
    per_split = defaultdict(list)
    for sid in range(1, n_subjects + 1):
        cat = 1 if sid % 2 else 2
        for k in range(1, per_subject + 1):
            split = rng.choice(splits)
            name = make_name(sid, k)
            (root / split).mkdir(parents=True, exist_ok=True)
            (root / split / name).write_bytes(b"x")
            per_split[split].append((name, cat))
    for split, items in per_split.items():
        if fmt == "coco":
            data = {"info": {"description": "tcm"},
                    "categories": [{"id": 0, "name": "tongue"}, {"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                    "images": [], "annotations": []}
            for i, (name, cat) in enumerate(items):
                data["images"].append({"id": i, "file_name": name, "width": 10, "height": 10})
                for j in range(2):  # two annotations per image
                    data["annotations"].append({"id": len(data["annotations"]), "image_id": i,
                                                "category_id": cat, "bbox": [j, 0, 1, 1]})
        elif fmt == "dict":
            data = {name: {"label": cat} for name, cat in items}
        else:
            data = [{"filename": name, "label": cat} for name, cat in items]
        (root / split / "anno.json").write_text(json.dumps(data, indent=2))


@pytest.mark.parametrize("fmt", ["coco", "dict", "list"])
def test_json_annotations_resplit(tmp_path, fmt):
    src, out = tmp_path / "src", tmp_path / "out"
    build_json_dataset(src, fmt)
    assert subject_split.main(["--input", str(src), "--output", str(out), "--stratify"]) == 0

    by_split = subjects_by_split(out)
    assert not (by_split["train"] & by_split["valid"] or by_split["train"] & by_split["test"]
                or by_split["valid"] & by_split["test"])

    # gather original per-image annotation content to compare against
    def per_image(root):
        res = {}
        for sp in ("train", "valid", "test"):
            data = json.loads((root / sp / "anno.json").read_text())
            if fmt == "coco":
                assert [im["id"] for im in data["images"]] == list(range(len(data["images"])))
                assert len({a["id"] for a in data["annotations"]}) == len(data["annotations"])
                ids = {im["id"]: im["file_name"] for im in data["images"]}
                for a in data["annotations"]:
                    res.setdefault(ids[a["image_id"]], []).append((a["category_id"], a["bbox"]))
                assert data["info"] == {"description": "tcm"}
                assert len(data["categories"]) == 3
            elif fmt == "dict":
                res.update({k: v for k, v in data.items()})
            else:
                res.update({r["filename"]: r for r in data})
            listed = set(ids.values()) if fmt == "coco" else (set(data) if fmt == "dict"
                                                              else {r["filename"] for r in data})
            assert listed == {p.name for p in (root / sp).glob("*.png")}
        return res

    assert per_image(out) == per_image(src)


def test_json_named_differently_per_split(tmp_path):
    """train/train.json, valid/val.json, test/test.json -> one json per split, names kept."""
    src, out = tmp_path / "src", tmp_path / "out"
    build_json_dataset(src, "coco")
    names = {"train": "train.json", "valid": "val.json", "test": "test.json"}
    for sp, n in names.items():
        (src / sp / "anno.json").rename(src / sp / n)
    total_anns = sum(len(json.loads((src / sp / n).read_text())["annotations"]) for sp, n in names.items())

    assert subject_split.main(["--input", str(src), "--output", str(out)]) == 0
    new_total = 0
    for sp, n in names.items():
        assert sorted(p.name for p in (out / sp).glob("*.json")) == [n]
        data = json.loads((out / sp / n).read_text())
        assert {im["file_name"] for im in data["images"]} == {p.name for p in (out / sp).glob("*.png")}
        new_total += len(data["annotations"])
    assert new_total == total_anns


def test_stops_when_json_does_not_match_images(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    build_json_dataset(src, "coco")
    # simulate a json from another export: rename most train images on disk
    imgs = sorted((src / "train").glob("*.png"))
    for p in imgs[: len(imgs) * 2 // 3]:
        p.rename(p.with_name(p.name.replace(".rf.", ".rf.x")))
    with pytest.raises(SystemExit, match="STOPPED"):
        subject_split.main(["--input", str(src), "--output", str(out)])
    assert not out.exists()
    monkeypatch.setattr(subject_split, "ALLOW_MISSING_ANNOTATIONS", True)
    assert subject_split.main(["--input", str(src), "--output", str(out)]) == 0


def test_annotations_found_in_another_splits_json(tmp_path):
    """Images whose annotations are in a different split's json still get them."""
    src, out = tmp_path / "src", tmp_path / "out"
    build_json_dataset(src, "coco")
    # move half of the train images into valid/test folders without touching the json files
    imgs = sorted((src / "train").glob("*.png"))
    for i, p in enumerate(imgs[: len(imgs) // 2]):
        p.rename(src / ("valid" if i % 2 else "test") / p.name)
    expected = {}
    for sp in ("train", "valid", "test"):
        data = json.loads((src / sp / "anno.json").read_text())
        ids = {im["id"]: im["file_name"] for im in data["images"]}
        for a in data["annotations"]:
            expected.setdefault(ids[a["image_id"]], []).append(a["bbox"])

    assert subject_split.main(["--input", str(src), "--output", str(out), "--ratios", "1", "1", "1"]) == 0
    got = {}
    for sp in ("train", "valid", "test"):
        data = json.loads((out / sp / "anno.json").read_text())
        names = {im["file_name"] for im in data["images"]}
        assert names == {p.name for p in (out / sp).glob("*.png")}  # every image has its annotations
        ids = {im["id"]: im["file_name"] for im in data["images"]}
        for a in data["annotations"]:
            got.setdefault(ids[a["image_id"]], []).append(a["bbox"])
    assert {k: sorted(v) for k, v in got.items()} == {k: sorted(v) for k, v in expected.items()}
