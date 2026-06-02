import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, square: bool = False):
        super().__init__()
        self.smooth = smooth
        self.square = square

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, n_classes).permute(0, 4, 1, 2, 3).float()

        dims = tuple(range(2, pred.ndim))
        if self.square:
            intersection = (pred_softmax * target_one_hot).sum(dim=dims)
            union = (pred_softmax ** 2 + target_one_hot ** 2).sum(dim=dims)
        else:
            intersection = (pred_softmax * target_one_hot).sum(dim=dims)
            union = (pred_softmax + target_one_hot).sum(dim=dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.shape[1]
        log_probs = F.log_softmax(pred, dim=1)
        probs = torch.exp(log_probs)
        target_one_hot = F.one_hot(target, n_classes).permute(0, 4, 1, 2, 3).float()

        pt = (target_one_hot * probs).sum(dim=1)
        log_pt = (target_one_hot * log_probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        loss = -focal_weight * log_pt
        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            alpha_weight = (target_one_hot * alpha.view(1, -1, 1, 1, 1)).sum(dim=1)
            loss = loss * alpha_weight

        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, n_classes).permute(0, 4, 1, 2, 3).float()

        dims = tuple(range(2, pred.ndim))
        tp = (pred_softmax * target_one_hot).sum(dim=dims)
        fp = (pred_softmax * (1 - target_one_hot)).sum(dim=dims)
        fn = ((1 - pred_softmax) * target_one_hot).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0 / len(losses)] * len(losses)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total_loss = total_loss + w * loss_fn(pred, target)
        return total_loss


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, weights: Optional[List[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [0.4, 0.2, 0.2, 0.2]

    def forward(self, preds: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            return self.base_loss(preds, target)

        total_loss = 0.0
        for i, p in enumerate(preds):
            if i >= len(self.weights):
                break
            if p.shape[2:] != target.shape[1:]:
                p = F.interpolate(p, size=target.shape[1:], mode='trilinear', align_corners=False)
            total_loss = total_loss + self.weights[i] * self.base_loss(p, target)
        return total_loss


LOSS_REGISTRY = {
    'dice': DiceLoss,
    'focal': FocalLoss,
    'tversky': TverskyLoss,
    'cross_entropy': nn.CrossEntropyLoss,
}


def create_loss_fn(config: Dict[str, Any]) -> nn.Module:
    loss_config = config.get('training', {}).get('loss', {})
    loss_name = loss_config.get('name', 'dice_focal')

    deep_supervision = config.get('model', {}).get('deep_supervision', True)

    if loss_name == 'dice':
        base_loss = DiceLoss(**loss_config.get('params', {}))
    elif loss_name == 'focal':
        base_loss = FocalLoss(**loss_config.get('params', {}))
    elif loss_name == 'tversky':
        base_loss = TverskyLoss(**loss_config.get('params', {}))
    elif loss_name == 'cross_entropy':
        base_loss = nn.CrossEntropyLoss()
    elif loss_name == 'dice_ce':
        w = loss_config.get('weights', [0.5, 0.5])
        base_loss = CombinedLoss([DiceLoss(), nn.CrossEntropyLoss()], w)
    elif loss_name == 'dice_focal':
        w = loss_config.get('weights', [0.5, 0.5])
        base_loss = CombinedLoss([DiceLoss(), FocalLoss()], w)
    elif loss_name == 'dice_tversky':
        w = loss_config.get('weights', [0.5, 0.5])
        base_loss = CombinedLoss([DiceLoss(), TverskyLoss()], w)
    else:
        logger.warning(f"Unknown loss '{loss_name}', using dice_focal")
        base_loss = CombinedLoss([DiceLoss(), FocalLoss()])

    if deep_supervision:
        ds_weights = loss_config.get('deep_supervision_weights', [0.4, 0.2, 0.2, 0.2])
        return DeepSupervisionLoss(base_loss, ds_weights)

    return base_loss
