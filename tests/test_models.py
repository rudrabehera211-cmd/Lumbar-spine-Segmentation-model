import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from models import (
    UNet, AttentionUNet, UNetPlusPlus, UNETR,
    SwinUNETR, nnUNet, ATMNet, create_model
)


def _model_forward(model, name, input_shape=(1, 1, 32, 64, 64), num_classes=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = torch.randn(*input_shape, device=device)

    model.train()
    output = model(x)

    if isinstance(output, list):
        for i, o in enumerate(output):
            assert o.shape[1] == num_classes, f"{name} DS output {i} channels: {o.shape[1]} != {num_classes}"
        model.eval()
        output_eval = model(x)
        assert output_eval.shape[1] == num_classes, f"{name} eval output channels: {output_eval.shape[1]} != {num_classes}"
    else:
        assert output.shape[1] == num_classes, f"{name} output channels: {output.shape[1]} != {num_classes}"

    loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters() if p.requires_grad)
    assert has_grad, f"{name} gradient not flowing to model parameters"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name:25s}: PASSED ({n_params:>8,} params, output={output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]})")

    return True


def test_all_models():
    print("=" * 60)
    print("MODEL FORWARD/BACKWARD TESTS")
    print("=" * 60)

    input_shape = (2, 1, 32, 64, 64)
    num_classes = 12

    models_to_test = [
        (UNet(in_channels=1, out_channels=num_classes, deep_supervision=True), 'UNet'),
        (UNet(in_channels=1, out_channels=num_classes, deep_supervision=False), 'UNet (no DS)'),
        (AttentionUNet(in_channels=1, out_channels=num_classes, deep_supervision=True), 'AttentionUNet'),
        (UNetPlusPlus(in_channels=1, out_channels=num_classes, deep_supervision=True), 'UNet++'),
        (nnUNet(in_channels=1, out_channels=num_classes, deep_supervision=True), 'nnUNet'),
        (ATMNet(in_channels=1, out_channels=num_classes, deep_supervision=True), 'ATMNet'),
    ]

    for model, name in models_to_test:
        _model_forward(model, name, input_shape, num_classes)

    print("\n" + "=" * 60)
    print("ALL MODEL TESTS PASSED")
    print("=" * 60)


def test_model_factory():
    print("\nTesting model factory...")
    config = {
        'model': {
            'name': 'atm_net',
            'in_channels': 1,
            'num_classes': 12,
            'deep_supervision': True,
            'params': {'features': [16, 32, 64, 128, 256]},
        }
    }
    model = create_model(config)
    assert model is not None, "Model factory returned None"
    print(f"  Model factory: PASSED ({model.__class__.__name__})")


if __name__ == '__main__':
    test_all_models()
    test_model_factory()
