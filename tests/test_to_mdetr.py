import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import subject_split  # noqa: E402
import to_mdetr  # noqa: E402

# Roboflow-style COCO: id 0 is the dataset supercategory, the points follow
CATEGORIES = [{"id": 0, "name": "acupoints", "supercategory": "none"},
              {"id": 1, "name": "laogong", "supercategory": "acupoints"},
              {"id": 2, "name": "shaofu", "supercategory": "acupoints"},
              {"id": 3, "name": "yuji", "supercategory": "acupoints"},
              {"id": 4, "name": "zhongchong", "supercategory": "acupoints"}]
MDETR_CAT = {"yuji": 0, "laogong": 1, "zhongchong": 2, "shaofu": 3}
CODE = {"yuji": "LU10", "laogong": "P8", "zhongchong": "P9", "shaofu": "HT8"}


def coco(files):
    data = {"info": {"x": 1}, "licenses": [], "categories": CATEGORIES, "images": [], "annotations": []}
    for i, name in enumerate(files):
        data["images"].append({"id": i, "file_name": name, "height": 1024, "width": 1024})
        for cat in (1, 2, 3, 4):
            data["annotations"].append({"id": len(data["annotations"]), "image_id": i, "category_id": cat,
                                        "bbox": [10.4 * cat, 20.6, 40.2, 59.7], "area": 2400.6,
                                        "segmentation": [], "iscrowd": 0})
    return data


def test_convert_matches_mdetr_format():
    files = ["160_4_JPG.rf.aaa.png", "100_1_JPG.rf.bbb.png"]
    src = coco(files)
    out, warnings = to_mdetr.convert(src, random.Random(0))
    assert warnings == []
    assert set(out) == {"images", "annotations", "info", "licenses"}
    assert out["info"] == {} and out["licenses"] == []
    assert len(out["images"]) == len(out["annotations"]) == 8
    assert [im["id"] for im in out["images"]] == list(range(8))
    assert [a["id"] for a in out["annotations"]] == list(range(8))
    names = {c["id"]: c["name"] for c in CATEGORIES}
    for im, a in zip(out["images"], out["annotations"]):
        assert set(im) == {"file_name", "height", "width", "id", "sentences"}
        assert set(a) == {"id", "image_id", "bbox", "category_id", "area", "iscrowd"}
        assert a["image_id"] == im["id"]
        assert all(isinstance(v, int) for v in a["bbox"]) and isinstance(a["area"], int)
        point = next(p for p, c in MDETR_CAT.items() if c == a["category_id"])
        assert im["sentences"] in (point, CODE[point])
    # every image keeps all 4 of its points
    for f in files:
        cats = sorted(a["category_id"] for im, a in zip(out["images"], out["annotations"]) if im["file_name"] == f)
        assert cats == [0, 1, 2, 3]
    # bbox of a point is carried over (rounded)
    by_cat = {a["category_id"]: a["bbox"] for a in out["annotations"][:4]}
    assert by_cat[MDETR_CAT["yuji"]] == [31, 21, 40, 60]  # yuji was COCO category 3
    assert names[3] == "yuji"


def test_sentence_modes():
    src = coco(["001_1_JPG.rf.a.png"])
    out, _ = to_mdetr.convert(src, random.Random(0), sentence_mode="code")
    assert {im["sentences"] for im in out["images"]} == {"LU10", "P8", "P9", "HT8"}
    out, _ = to_mdetr.convert(src, random.Random(0), sentence_mode="name")
    assert {im["sentences"] for im in out["images"]} == set(MDETR_CAT)


def test_unknown_category_warns():
    src = coco(["001_1_JPG.rf.a.png"])
    src["annotations"][0]["category_id"] = 0  # the supercategory
    out, warnings = to_mdetr.convert(src, random.Random(0))
    assert len(out["images"]) == 3 and any("acupoints" in w for w in warnings)


def test_split_then_mdetr_end_to_end(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    rng = random.Random(3)
    per_split = {"train": [], "valid": [], "test": []}
    for sid in range(1, 31):
        for k in range(1, 8):
            sp = rng.choice(["train"] * 7 + ["valid"] * 2 + ["test"])
            name = f"{sid:03d}_{k}_JPG.rf.{rng.getrandbits(64):016x}.png"
            (src / sp).mkdir(parents=True, exist_ok=True)
            (src / sp / name).write_bytes(b"x")
            per_split[sp].append(name)
    for sp, files in per_split.items():
        (src / sp / f"{sp}.json").write_text(json.dumps(coco(files)))

    assert subject_split.main(["--input", str(src), "--output", str(out), "--mdetr"]) == 0
    subjects = {}
    for sp in per_split:
        data = json.loads((out / sp / f"{sp}.json").read_text())
        assert set(data) == {"images", "annotations", "info", "licenses"}
        files = {im["file_name"] for im in data["images"]}
        assert files == {p.name for p in (out / sp).glob("*.png")}
        assert len(data["images"]) == 4 * len(files)
        subjects[sp] = {f.split("_")[0] for f in files}
    assert not (subjects["train"] & subjects["valid"] or subjects["train"] & subjects["test"]
                or subjects["valid"] & subjects["test"])

    # an MDETR dataset can itself be re-split; all entries of an image move together
    out2 = tmp_path / "out2"
    assert subject_split.main(["--input", str(out), "--output", str(out2), "--seed", "7"]) == 0
    total = 0
    for sp in per_split:
        data = json.loads((out2 / sp / f"{sp}.json").read_text())
        assert "categories" not in data and all("sentences" in im for im in data["images"])
        assert {im["file_name"] for im in data["images"]} == {p.name for p in (out2 / sp).glob("*.png")}
        assert {a["image_id"] for a in data["annotations"]} == {im["id"] for im in data["images"]}
        total += len(data["images"])
    assert total == 4 * 30 * 7
