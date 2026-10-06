import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import to_odvg_det  # noqa: E402
from test_to_mdetr import coco  # noqa: E402

NAMES = ["003_1_JPG.rf.a.png", "004_2_JPG.rf.b.png"]


def test_one_line_per_image_with_all_points(tmp_path):
    src, dst = tmp_path / "in.json", tmp_path / "out.jsonl"
    data = coco(NAMES)
    data["annotations"].append({"id": 99, "image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1]})  # supercategory
    src.write_text(json.dumps(data))
    before = src.read_text()
    _, lm = to_odvg_det.convert_file(src, dst)
    assert src.read_text() == before
    lines = [json.loads(x) for x in dst.read_text().splitlines()]
    assert [r["filename"] for r in lines] == NAMES
    label = {"yuji": 0, "laogong": 1, "zhongchong": 2, "shaofu": 3}
    for r in lines:
        inst = r["detection"]["instances"]
        assert len(inst) == 4 and {i["category"] for i in inst} == set(label)
        for i in inst:
            assert i["label"] == label[i["category"]]
            x1, y1, x2, y2 = i["bbox"]
            assert all(isinstance(v, int) for v in i["bbox"]) and x2 > x1 and y2 > y1
    assert json.loads(lm.read_text()) == {"0": "yuji", "1": "laogong", "2": "zhongchong", "3": "shaofu"}
