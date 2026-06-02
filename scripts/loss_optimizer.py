#!/usr/bin/env python3
"""Automatically select the best loss function for spine segmentation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import json

from losses import (
    DiceLoss, FocalLoss, TverskyLoss, CombinedLoss, DeepSupervisionLoss
)
from models.unet import UNet


def create_simulated_data(num_samples=100, num_classes=12, shape=(32, 64, 64), device='cpu'):
    images = torch.randn(num_samples, 1, *shape, device=device)
    targets = torch.randint(0, num_classes, (num_samples, *shape), device=device)
    return images, targets


def evaluate_loss_on_model(loss_fn, model, images, targets, device='cpu'):
    model.train()
    model.zero_grad()

    pred = model(images)
    loss = loss_fn(pred, targets)
    loss.backward()

    with torch.no_grad():
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    return {
        'loss_value': loss.item(),
        'grad_norm': grad_norm,
        'is_finite': torch.isfinite(loss).item(),
    }


def benchmark_losses():
    print("=" * 60)
    print("AUTOMATED LOSS FUNCTION SELECTION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    images, targets = create_simulated_data(num_samples=4, shape=(16, 32, 32), device=device)

    model_configs = {
        'unet': lambda: UNet(in_channels=1, out_channels=12, deep_supervision=False).to(device),
        'unet_ds': lambda: UNet(in_channels=1, out_channels=12, deep_supervision=True).to(device),
    }

    loss_configs = {
        'Dice': DiceLoss(),
        'Dice (square)': DiceLoss(square=True),
        'Focal (γ=2)': FocalLoss(gamma=2.0),
        'Focal (γ=3)': FocalLoss(gamma=3.0),
        'Tversky (α=0.7)': TverskyLoss(alpha=0.7, beta=0.3),
        'Tversky (α=0.5)': TverskyLoss(alpha=0.5, beta=0.5),
        'CrossEntropy': torch.nn.CrossEntropyLoss(),
        'Dice+CE (1:1)': CombinedLoss([DiceLoss(), torch.nn.CrossEntropyLoss()], [0.5, 0.5]),
        'Dice+CE (7:3)': CombinedLoss([DiceLoss(), torch.nn.CrossEntropyLoss()], [0.7, 0.3]),
        'Dice+Focal (1:1)': CombinedLoss([DiceLoss(), FocalLoss()], [0.5, 0.5]),
        'Dice+Focal (7:3)': CombinedLoss([DiceLoss(), FocalLoss()], [0.7, 0.3]),
        'Dice+Tversky (1:1)': CombinedLoss([DiceLoss(), TverskyLoss()], [0.5, 0.5]),
    }

    results = {}

    for model_name, model_fn in model_configs.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)
        print(f"{'Loss':<25} {'Value':<10} {'Grad':<10}")
        print("-" * 45)

        model = model_fn()
        for loss_name, loss_fn in loss_configs.items():
            if model_name == 'unet_ds':
                wrapped = DeepSupervisionLoss(loss_fn, [0.4, 0.2, 0.2, 0.2])
                model2 = model_fn()
                pred = model2(images)
                loss = wrapped(pred, targets)
                loss.backward()
                value = loss.item()
            else:
                model_copy = model_fn()
                try:
                    result = evaluate_loss_on_model(loss_fn, model_copy, images, targets, device)
                    value = result['loss_value']
                    grad = result['grad_norm']
                except Exception as e:
                    value = float('nan')
                    grad = 0.0

            status = 'OK' if np.isfinite(value) and not np.isnan(value) else 'FAIL'
            print(f"{loss_name:<25} {value:<10.6f} {status:<10}")

            if model_name not in results:
                results[model_name] = {}
            results[model_name][loss_name] = {
                'value': value if np.isfinite(value) else None,
                'status': status,
            }

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    for model_name in results:
        valid = {k: v for k, v in results[model_name].items() if v['status'] == 'OK'}
        if valid:
            best = min(valid, key=lambda k: valid[k]['value'])
            print(f"  {model_name}: Best loss = {best} (value={valid[best]['value']:.6f})")

    print("\nSuggested combined losses (Dice + auxiliary):")
    print("  - Dice+Focal (1:1): Good for hard-to-segment structures")
    print("  - Dice+CE (1:1): Good for overall stability")
    print("  - Dice+Tversky (1:1): Good for imbalanced classes")

    with open(Path('reports') / 'loss_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to reports/loss_benchmark.json")


if __name__ == '__main__':
    Path('reports').mkdir(exist_ok=True)
    benchmark_losses()
