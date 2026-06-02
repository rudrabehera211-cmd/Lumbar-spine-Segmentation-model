import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.ndimage import binary_erosion, binary_dilation, label as scipy_label
from .metrics import compute_dice, compute_iou

logger = logging.getLogger(__name__)

SPINE_LABEL_MAP = {
    0: 'Background', 1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5',
    6: 'Sacrum', 7: 'Disc_L1_L2', 8: 'Disc_L2_L3', 9: 'Disc_L3_L4',
    10: 'Disc_L4_L5', 11: 'Disc_L5_S1',
}


def analyze_undersegmentation(pred: np.ndarray, target: np.ndarray, num_classes: int = 12) -> Dict:
    issues = {}
    for label in range(1, num_classes):
        name = SPINE_LABEL_MAP.get(label, f'Class_{label}')
        t_count = (target == label).sum()
        p_count = (pred == label).sum()
        if t_count > 0:
            ratio = p_count / t_count
            if ratio < 0.5:
                issues[name] = {
                    'type': 'under-segmentation',
                    'ratio': float(ratio),
                    'target_voxels': int(t_count),
                    'pred_voxels': int(p_count),
                    'severity': 'high' if ratio < 0.25 else 'medium',
                }
    return issues


def analyze_oversegmentation(pred: np.ndarray, target: np.ndarray, num_classes: int = 12) -> Dict:
    issues = {}
    for label in range(1, num_classes):
        name = SPINE_LABEL_MAP.get(label, f'Class_{label}')
        t_count = (target == label).sum()
        p_count = (pred == label).sum()
        if t_count > 0:
            ratio = p_count / t_count
            if ratio > 1.5:
                issues[name] = {
                    'type': 'over-segmentation',
                    'ratio': float(ratio),
                    'target_voxels': int(t_count),
                    'pred_voxels': int(p_count),
                    'severity': 'high' if ratio > 2.0 else 'medium',
                }
    return issues


def analyze_class_confusion(pred: np.ndarray, target: np.ndarray, num_classes: int = 12) -> Dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = np.logical_and(pred == i, target == j).sum()

    confusion_pairs = []
    for i in range(1, num_classes):
        for j in range(1, num_classes):
            if i != j and cm[i, j] > 0:
                i_name = SPINE_LABEL_MAP.get(i, f'Class_{i}')
                j_name = SPINE_LABEL_MAP.get(j, f'Class_{j}')
                total_i = target[target == i].size
                confusion_ratio = cm[i, j] / max(total_i, 1)
                if confusion_ratio > 0.05:
                    confusion_pairs.append({
                        'predicted_as': i_name,
                        'should_be': j_name,
                        'voxels': int(cm[i, j]),
                        'ratio': float(confusion_ratio),
                    })

    return {'confusion_matrix': cm.tolist(), 'confusion_pairs': confusion_pairs}


def analyze_boundary_errors(pred: np.ndarray, target: np.ndarray, num_classes: int = 12) -> Dict:
    errors = {}
    for label in range(1, num_classes):
        name = SPINE_LABEL_MAP.get(label, f'Class_{label}')
        p = (pred == label).astype(np.uint8)
        t = (target == label).astype(np.uint8)

        if p.sum() == 0 or t.sum() == 0:
            continue

        p_boundary = p ^ binary_erosion(p, iterations=1)
        t_boundary = t ^ binary_erosion(t, iterations=1)

        boundary_fp = np.logical_and(p_boundary, ~t).sum()
        boundary_fn = np.logical_and(t_boundary, ~p).sum()

        if boundary_fp + boundary_fn > 0:
            errors[name] = {
                'boundary_fp': int(boundary_fp),
                'boundary_fn': int(boundary_fn),
                'total_boundary_errors': int(boundary_fp + boundary_fn),
            }
    return errors


def analyze_dataset_imbalance(class_distribution: Dict) -> Dict:
    if not class_distribution or '_total_voxels' not in class_distribution:
        return {'error': 'Invalid class distribution data'}

    total = class_distribution['_total_voxels']
    imbalances = []

    for label, info in class_distribution.items():
        if label == '_total_voxels':
            continue
        pct = info.get('percentage', 0)
        if pct < 0.1 and label != 0:
            imbalances.append({
                'class': info.get('name', f'Class_{label}'),
                'percentage': pct,
                'severity': 'high' if pct < 0.01 else 'medium',
            })

    return {
        'class_imbalances': imbalances,
        'background_percentage': class_distribution.get(0, {}).get('percentage', 0),
        'foreground_percentage': 100 - class_distribution.get(0, {}).get('percentage', 0),
    }


def generate_error_report(
    pred: np.ndarray,
    target: np.ndarray,
    class_distribution: Optional[Dict] = None,
    num_classes: int = 12,
) -> Dict:
    report = {
        'undersegmentation': analyze_undersegmentation(pred, target, num_classes),
        'oversegmentation': analyze_oversegmentation(pred, target, num_classes),
        'class_confusion': analyze_class_confusion(pred, target, num_classes),
        'boundary_errors': analyze_boundary_errors(pred, target, num_classes),
    }

    if class_distribution:
        report['dataset_imbalance'] = analyze_dataset_imbalance(class_distribution)

    critical_issues = []
    for cls, info in report['undersegmentation'].items():
        if info.get('severity') == 'high':
            critical_issues.append(f"Severe under-segmentation of {cls} (ratio={info['ratio']:.2f})")
    for cls, info in report['oversegmentation'].items():
        if info.get('severity') == 'high':
            critical_issues.append(f"Severe over-segmentation of {cls} (ratio={info['ratio']:.2f})")
    for pair in report['class_confusion'].get('confusion_pairs', []):
        if pair['ratio'] > 0.2:
            critical_issues.append(f"High confusion: {pair['predicted_as']} predicted as {pair['should_be']} ({pair['ratio']:.1%})")

    report['critical_issues'] = critical_issues
    report['num_critical'] = len(critical_issues)

    recommendations = []
    if critical_issues:
        recommendations.append("Increase class weights for under-performed classes")
        recommendations.append("Add more training data for confused classes")
        recommendations.append("Consider boundary-aware loss functions")
    if class_distribution:
        imbalances = report.get('dataset_imbalance', {}).get('class_imbalances', [])
        if imbalances:
            recommendations.append("Apply class-balanced sampling or loss weighting")
    report['recommendations'] = recommendations

    return report
