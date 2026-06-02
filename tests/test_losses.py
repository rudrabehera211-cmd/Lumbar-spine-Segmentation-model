import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from losses import (
    DiceLoss, FocalLoss, TverskyLoss, CombinedLoss, DeepSupervisionLoss, create_loss_fn
)


def _loss_forward(name, loss_fn, pred, target):
    loss = loss_fn(pred, target)
    assert not torch.isnan(loss), f"{name} loss is NaN"
    assert not torch.isinf(loss), f"{name} loss is Inf"
    assert loss.item() >= 0, f"{name} loss is negative"

    loss.backward()
    assert pred.grad is not None, f"{name} gradient is None"
    assert not torch.isnan(pred.grad).any(), f"{name} gradient has NaN"

    print(f"  {name:25s}: PASSED (loss={loss.item():.6f})")
    return True


def test_all_losses():
    print("=" * 60)
    print("LOSS FUNCTION TESTS")
    print("=" * 60)

    B, C, D, H, W = 2, 12, 16, 32, 32
    pred = torch.randn(B, C, D, H, W, requires_grad=True)
    target = torch.randint(0, C, (B, D, H, W))

    losses = [
        ('DiceLoss', DiceLoss()),
        ('DiceLoss (square)', DiceLoss(square=True)),
        ('FocalLoss', FocalLoss(gamma=2.0)),
        ('TverskyLoss', TverskyLoss(alpha=0.7, beta=0.3)),
        ('CrossEntropy', torch.nn.CrossEntropyLoss()),
        ('Dice+CE', CombinedLoss([DiceLoss(), torch.nn.CrossEntropyLoss()], [0.5, 0.5])),
        ('Dice+Focal', CombinedLoss([DiceLoss(), FocalLoss()], [0.5, 0.5])),
        ('Dice+Tversky', CombinedLoss([DiceLoss(), TverskyLoss()], [0.5, 0.5])),
    ]

    for name, loss_fn in losses:
        pred.grad = None
        _loss_forward(name, loss_fn, pred, target)

    print("\n" + "=" * 60)
    print("ALL LOSS TESTS PASSED")
    print("=" * 60)


def test_deep_supervision_loss():
    print("\nTesting Deep Supervision Loss...")
    B, C, D, H, W = 2, 12, 16, 32, 32
    base_loss = CombinedLoss([DiceLoss(), FocalLoss()], [0.5, 0.5])
    ds_loss = DeepSupervisionLoss(base_loss, [0.4, 0.2, 0.2, 0.2])

    preds = [
        torch.randn(B, C, D, H, W, requires_grad=True),
        torch.randn(B, C, D // 2, H // 2, W // 2, requires_grad=True),
        torch.randn(B, C, D // 4, H // 4, W // 4, requires_grad=True),
    ]
    target = torch.randint(0, C, (B, D, H, W))

    loss = ds_loss(preds, target)
    loss.backward()

    assert not torch.isnan(loss), "DS loss is NaN"
    assert all(p.grad is not None for p in preds), "DS gradients not flowing"
    print(f"  DeepSupervisionLoss: PASSED (loss={loss.item():.6f})")


if __name__ == '__main__':
    test_all_losses()
    test_deep_supervision_loss()
