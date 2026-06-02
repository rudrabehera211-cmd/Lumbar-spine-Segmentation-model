import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .unet import UNet
from .attention_unet import AttentionUNet
from .nested_unet import UNetPlusPlus
from .unetr import UNETR
from .swin_unetr import SwinUNETR
from .nnunet import nnUNet
from .atm_net import ATMNet

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    'unet': UNet,
    'attention_unet': AttentionUNet,
    'unet_plus_plus': UNetPlusPlus,
    'unetr': UNETR,
    'swin_unetr': SwinUNETR,
    'nnunet': nnUNet,
    'atm_net': ATMNet,
}


def create_model(config: Dict[str, Any]) -> nn.Module:
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'unet').lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    model_params = model_config.get('params', {})

    default_params = {
        'in_channels': model_config.get('in_channels', 1),
        'out_channels': model_config.get('num_classes', 12),
        'deep_supervision': model_config.get('deep_supervision', True),
    }

    if model_name == 'unetr':
        default_params['img_size'] = tuple(model_params.get('img_size', config.get('preprocessing', {}).get('target_size', (128, 128, 128))))
    elif model_name == 'swin_unetr':
        default_params['img_size'] = tuple(model_params.get('img_size', config.get('preprocessing', {}).get('target_size', (128, 128, 128))))

    full_params = {**default_params, **model_params}
    model = model_class(**full_params)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {model_name}: {n_params:,} params ({n_trainable:,} trainable)")

    return model
