import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_palm_classifier as epc  # noqa: E402
import video_dump as vd  # noqa: E402
from torchvision.models import resnet18  # noqa: E402


@pytest.mark.parametrize("head", [
    lambda: torch.nn.Linear(512, 2),                                                     # fc.weight
    lambda: torch.nn.Sequential(torch.nn.Linear(512, 2)),                                # fc.0.weight (yours)
    lambda: torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(512, 2)),         # fc.1.weight
    lambda: torch.nn.Sequential(torch.nn.Linear(512, 128), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                torch.nn.Linear(128, 1)),                                # fc.0 + fc.3, 1 output
])
def test_loader_rebuilds_the_saved_head(tmp_path, head):
    torch.manual_seed(0)
    ref = resnet18(weights=None)
    ref.fc = head()
    ref.eval()
    torch.save(ref.state_dict(), tmp_path / "palm.pth")
    x = torch.randn(2, 3, 224, 224)
    for mod in (vd, epc):
        m = mod.load_model(str(tmp_path / "palm.pth"), 2, torch.device("cpu"))
        with torch.no_grad():
            assert torch.allclose(m(x), ref(x), atol=1e-5)
