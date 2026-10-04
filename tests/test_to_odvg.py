import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import to_odvg  # noqa: E402
from test_to_mdetr import CODE, coco  # noqa: E402

NAMES = ["165_5_JPG.rf.533991b1bff2e3b80fe498b19b64bff7.png", "002_1_JPG.rf.aa.png"]


def run(tmp_path, mode="random"):
    src, dst = tmp_path / "in.json", tmp_path / "out.jsonl"
    src.write_text(json.dumps(coco(NAMES)))
    before = src.read_text()
    assert to_odvg.main(["--input", str(src), "--output", str(dst), "--sentence-mode", mode]) == 0
    assert src.read_text() == before  # original untouched
    return [json.loads(line) for line in dst.read_text().splitlines()]


def test_matches_sample_schema(tmp_path):
    lines = run(tmp_path)
    assert len(lines) == 8  # one line per point
    assert Counter(r["file_name"] for r in lines) == {n: 4 for n in NAMES}
    for r in lines:
        assert set(r) == {"file_name", "height", "width", "grounding"}
        assert (r["height"], r["width"]) == (1024, 1024)
        g = r["grounding"]
        assert set(g) == {"caption", "regions"} and len(g["regions"]) == 1
        reg = g["regions"][0]
        assert set(reg) == {"bbox", "phrase", "tokens_positive"}
        assert reg["tokens_positive"] == [[0, 1]]
        assert g["caption"] in (reg["phrase"], CODE[reg["phrase"]])
        assert all(isinstance(v, int) for v in reg["bbox"])


def test_bbox_xyxy(tmp_path):
    for r in run(tmp_path, "name"):
        reg = r["grounding"]["regions"][0]
        cat = {"laogong": 1, "shaofu": 2, "yuji": 3, "zhongchong": 4}[reg["phrase"]]
        x, y, w, h = 10.4 * cat, 20.6, 40.2, 59.7
        assert reg["bbox"] == [round(x), round(y), round(x + w), round(y + h)]


def test_caption_modes(tmp_path):
    assert all(r["grounding"]["caption"] == r["grounding"]["regions"][0]["phrase"] for r in run(tmp_path, "name"))
    assert all(r["grounding"]["caption"] == CODE[r["grounding"]["regions"][0]["phrase"]]
               for r in run(tmp_path, "code"))
