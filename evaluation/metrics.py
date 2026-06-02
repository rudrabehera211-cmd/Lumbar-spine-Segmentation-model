import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, binary_dilation
import logging

logger = logging.getLogger(__name__)

SPINE_LABEL_MAP = {
    0: 'Background', 1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5',
    6: 'Sacrum', 7: 'Disc_L1_L2', 8: 'Disc_L2_L3', 9: 'Disc_L3_L4',
    10: 'Disc_L4_L5', 11: 'Disc_L5_S1',
}


def compute_dice(pred: np.ndarray, target: np.ndarray, label: int, smooth: float = 1e-6) -> float:
    p = (pred == label).astype(np.float64)
    t = (target == label).astype(np.float64)
    intersection = (p * t).sum()
    return (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)


def compute_iou(pred: np.ndarray, target: np.ndarray, label: int, smooth: float = 1e-6) -> float:
    p = (pred == label).astype(np.float64)
    t = (target == label).astype(np.float64)
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_precision_recall(pred: np.ndarray, target: np.ndarray, label: int, smooth: float = 1e-6) -> Tuple[float, float]:
    p = (pred == label).astype(np.float64)
    t = (target == label).astype(np.float64)
    tp = (p * t).sum()
    precision = (tp + smooth) / (p.sum() + smooth)
    recall = (tp + smooth) / (t.sum() + smooth)
    return float(precision), float(recall)


def compute_hd95(pred: np.ndarray, target: np.ndarray, label: int, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    p_points = np.array(np.where(pred == label)).T * np.array(spacing)
    t_points = np.array(np.where(target == label)).T * np.array(spacing)

    if len(p_points) == 0 or len(t_points) == 0:
        return float('inf')

    if len(p_points) > 10000:
        idx = np.random.choice(len(p_points), 10000, replace=False)
        p_points = p_points[idx]
    if len(t_points) > 10000:
        idx = np.random.choice(len(t_points), 10000, replace=False)
        t_points = t_points[idx]

    d1 = directed_hausdorff(p_points, t_points)[0]
    d2 = directed_hausdorff(t_points, p_points)[0]
    return float(max(d1, d2))


def compute_asd(pred: np.ndarray, target: np.ndarray, label: int, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    from scipy.ndimage import distance_transform_edt

    p = (pred == label).astype(np.uint8)
    t = (target == label).astype(np.uint8)

    if p.sum() == 0 or t.sum() == 0:
        return float('inf')

    p_border = p ^ binary_erosion(p, iterations=1)
    t_border = t ^ binary_erosion(t, iterations=1)

    dt_p = distance_transform_edt(~p_border, sampling=spacing)
    dt_t = distance_transform_edt(~t_border, sampling=spacing)

    d1 = dt_t[p_border > 0].mean()
    d2 = dt_p[t_border > 0].mean()
    return float((d1 + d2) / 2)


class SegmentationMetrics:
    def __init__(self, num_classes: int = 12, label_map: Optional[Dict[int, str]] = None):
        self.num_classes = num_classes
        self.label_map = label_map or SPINE_LABEL_MAP

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        return self.compute_all(pred, target)

    def compute_all(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        results = {'mean_dice': 0.0, 'mean_iou': 0.0, 'mean_precision': 0.0, 'mean_recall': 0.0}

        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []

        for label in range(self.num_classes):
            name = self.label_map.get(label, f'Class_{label}')
            dice = compute_dice(pred, target, label)
            iou = compute_iou(pred, target, label)
            precision, recall = compute_precision_recall(pred, target, label)

            results[f'dice_{name}'] = round(dice, 6)
            results[f'iou_{name}'] = round(iou, 6)
            results[f'precision_{name}'] = round(precision, 6)
            results[f'recall_{name}'] = round(recall, 6)

            if label > 0:
                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)

        if dice_scores:
            results['mean_dice'] = round(np.mean(dice_scores), 6)
            results['mean_iou'] = round(np.mean(iou_scores), 6)
            results['mean_precision'] = round(np.mean(precision_scores), 6)
            results['mean_recall'] = round(np.mean(recall_scores), 6)

        return results

    def compute_boundary_metrics(
        self, pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        results = {}
        for label in range(1, self.num_classes):
            name = self.label_map.get(label, f'Class_{label}')
            hd95 = compute_hd95(pred, target, label, spacing)
            asd = compute_asd(pred, target, label, spacing)
            results[f'hd95_{name}'] = round(hd95, 4) if hd95 != float('inf') else float('inf')
            results[f'asd_{name}'] = round(asd, 4) if asd != float('inf') else float('inf')
        return results

    def compute_confusion_matrix(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                cm[i, j] = np.logical_and(pred == i, target == j).sum()
        return cm
