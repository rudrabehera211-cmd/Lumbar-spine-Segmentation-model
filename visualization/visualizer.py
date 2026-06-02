import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from utils.mha_utils import SPINE_LABEL_MAP

logger = logging.getLogger(__name__)

SPINE_COLORS = [
    (0, 0, 0),        # Background - black
    (1, 0, 0),        # L1 - red
    (0, 1, 0),        # L2 - green
    (0, 0, 1),        # L3 - blue
    (1, 1, 0),        # L4 - yellow
    (1, 0, 1),        # L5 - magenta
    (0, 1, 1),        # Sacrum - cyan
    (0.5, 0, 0),      # Disc_L1_L2 - dark red
    (0, 0.5, 0),      # Disc_L2_L3 - dark green
    (0, 0, 0.5),      # Disc_L3_L4 - dark blue
    (0.5, 0.5, 0),    # Disc_L4_L5 - olive
    (0.5, 0, 0.5),    # Disc_L5_S1 - purple
]

CUSTOM_CMAP = ListedColormap(SPINE_COLORS)


def plot_slice(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    save_path: Optional[Path] = None,
    title: str = '',
    dpi: int = 150,
):
    if image.ndim == 4:
        ncols = 1
        if mask is not None:
            ncols += 1
        if pred is not None:
            ncols += 1

        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        if ncols == 1:
            axes = [axes]

        ax_idx = 0
        axes[ax_idx].imshow(image[slice_idx], cmap='gray')
        axes[ax_idx].set_title(f'MRI {title}')
        axes[ax_idx].axis('off')
        ax_idx += 1

        if mask is not None:
            axes[ax_idx].imshow(mask[slice_idx], cmap=CUSTOM_CMAP, vmin=0, vmax=11)
            axes[ax_idx].set_title('Ground Truth')
            axes[ax_idx].axis('off')
            ax_idx += 1

        if pred is not None:
            axes[ax_idx].imshow(pred[slice_idx], cmap=CUSTOM_CMAP, vmin=0, vmax=11)
            axes[ax_idx].set_title('Prediction')
            axes[ax_idx].axis('off')

        plt.tight_layout()
    else:
        fig, axes = plt.subplots(1, 3 if mask is not None and pred is not None else
                                    2 if mask is not None or pred is not None else 1,
                                 figsize=(18, 6))
        if mask is None and pred is None:
            axes = [axes]

        ax_idx = 0
        axes[ax_idx].imshow(image, cmap='gray')
        axes[ax_idx].set_title(f'MRI {title}')
        axes[ax_idx].axis('off')
        ax_idx += 1

        if mask is not None:
            axes[ax_idx].imshow(mask, cmap=CUSTOM_CMAP, vmin=0, vmax=11)
            axes[ax_idx].set_title('Ground Truth')
            axes[ax_idx].axis('off')
            ax_idx += 1

        if pred is not None:
            axes[ax_idx].imshow(pred, cmap=CUSTOM_CMAP, vmin=0, vmax=11)
            axes[ax_idx].set_title('Prediction')
            axes[ax_idx].axis('off')

        plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
        plt.close(fig)


def plot_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    slice_idx: Optional[int] = None,
    save_path: Optional[Path] = None,
):
    if image.ndim == 4:
        img_slice = image[slice_idx]
        msk_slice = mask[slice_idx]
    else:
        img_slice = image
        msk_slice = mask

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_slice, cmap='gray')

    colored_mask = np.zeros((*msk_slice.shape, 3))
    for label in range(1, len(SPINE_COLORS)):
        colored_mask[msk_slice == label] = SPINE_COLORS[label][:3]

    ax.imshow(colored_mask, alpha=alpha)
    ax.set_title('Overlay: MRI + Segmentation')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_learning_curves(
    history: Dict[str, List],
    save_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.get('train_loss', []), label='Train Loss', color='blue')
    axes[0].plot(history.get('val_loss', []), label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.get('val_dice', []), label='Val Dice', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history.get('learning_rates', []), label='LR', color='red')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved learning curves to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    labels: Optional[List[str]] = None,
):
    if labels is None:
        labels = [SPINE_LABEL_MAP.get(i, f'C{i}') for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=6, color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.colorbar(im)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_class_metrics(
    metrics: Dict[str, float],
    save_path: Path,
    metric_name: str = 'dice',
):
    labels = [SPINE_LABEL_MAP.get(i, f'C{i}') for i in range(len(SPINE_LABEL_MAP))]
    values = [metrics.get(f'{metric_name}_{label}', 0) for label in [SPINE_LABEL_MAP[i] for i in range(1, len(SPINE_LABEL_MAP))]]
    names = [SPINE_LABEL_MAP[i] for i in range(1, len(SPINE_LABEL_MAP))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(f'{metric_name.capitalize()} Score')
    ax.set_title(f'Per-Class {metric_name.capitalize()} Score')
    ax.set_ylim([0, 1])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved class metrics to {save_path}")


def plot_prediction_grid(
    image: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    num_slices: int = 6,
    save_path: Optional[Path] = None,
):
    d = image.shape[0]
    indices = np.linspace(d // 4, 3 * d // 4, num_slices, dtype=int)

    fig, axes = plt.subplots(num_slices, 3, figsize=(12, 3 * num_slices))

    for i, idx in enumerate(indices):
        axes[i, 0].imshow(image[idx], cmap='gray')
        axes[i, 0].set_title(f'Slice {idx}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask[idx], cmap=CUSTOM_CMAP, vmin=0, vmax=11)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred[idx], cmap=CUSTOM_CMAP, vmin=0, vmax=11)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
