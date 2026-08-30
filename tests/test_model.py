import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import get_model


def test_model_output_shape():
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"


def test_model_invalid_architecture():
    try:
        get_model(architecture="vgg99")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_model_num_classes():
    model = get_model(architecture="resnet18", num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape[1] == 5


def test_model_is_nn_module():
    import torch.nn as nn
    model = get_model()
    assert isinstance(model, nn.Module)
