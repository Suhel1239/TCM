import csv
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
