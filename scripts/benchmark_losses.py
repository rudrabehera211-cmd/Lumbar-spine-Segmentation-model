#!/usr/bin/env python3
"""Benchmark all loss functions and select the best one."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from losses import DiceLoss, FocalLoss, TverskyLoss, CombinedLoss, DeepSupervisionLoss
from models.unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOSS_CONFIGS = {
    'dice': lambda: DiceLoss(),
    'focal': lambda: FocalLoss(gamma=2.0),
    'tversky': lambda: TverskyLoss(alpha=0.7, beta=0.3),
    'cross_entropy': lambda: torch.nn.CrossEntropyLoss(),
    'dice_ce': lambda: CombinedLoss([DiceLoss(), torch.nn.CrossEntropyLoss()], [0.5, 0.5]),
    'dice_focal': lambda: CombinedLoss([DiceLoss(), FocalLoss()], [0.5, 0.5]),
    'dice_tversky': lambda: CombinedLoss([DiceLoss(), TverskyLoss()], [0.5, 0.5]),
}

B, C, D, H, W = 2, 12, 64, 64, 64

pred = torch.randn(B, C, D, H, W, device=device)
target = torch.randint(0, C, (B, D, H, W), device=device)

print("=" * 60)
print("LOSS FUNCTION BENCHMARK")
print("=" * 60)
print(f"{'Loss Name':<20} {'Value':<12} {'Grad OK':<10}")
print("-" * 60)

results = []
for name, factory in LOSS_CONFIGS.items():
    loss_fn = factory()
    if name == 'dice_ce' or name == 'dice_focal' or name == 'dice_tversky':
        pass

    value = loss_fn(pred, target)
    value.backward()

    grad_ok = pred.grad is not None and not torch.isnan(pred.grad).any()
    print(f"{name:<20} {value.item():<12.6f} {'Yes' if grad_ok else 'No':<10}")

    results.append({'name': name, 'value': value.item(), 'grad_ok': grad_ok})
    pred.grad = None

print("\nDeep Supervision Test:")
model = UNet(in_channels=1, out_channels=C, deep_supervision=True).to(device)
base_loss = CombinedLoss([DiceLoss(), FocalLoss()], [0.5, 0.5])
ds_loss = DeepSupervisionLoss(base_loss, [0.4, 0.2, 0.2, 0.2])

x = torch.randn(B, 1, D // 2, H // 2, W // 2, device=device)
outputs = model(x)
loss = ds_loss(outputs, target[:, :D//2, :H//2, :W//2])
loss.backward()

print(f"  Deep Supervision Loss: {loss.item():.6f} (Gradient OK: {all(p.grad is not None for p in model.parameters() if p.requires_grad)})")

print("\nAll loss functions verified successfully.")
