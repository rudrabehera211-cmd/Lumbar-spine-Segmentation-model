import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPINE_LABEL_MAP = {
    0: 'Background',
    1: 'L1',
    2: 'L2',
    3: 'L3',
    4: 'L4',
    5: 'L5',
    6: 'Sacrum',
    7: 'Disc_L1_L2',
    8: 'Disc_L2_L3',
    9: 'Disc_L3_L4',
    10: 'Disc_L4_L5',
    11: 'Disc_L5_S1',
}

NUM_CLASSES = len(SPINE_LABEL_MAP)


def load_mha(filepath: Union[str, Path]) -> Tuple[sitk.Image, np.ndarray]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MHA file not found: {filepath}")
    if filepath.suffix.lower() not in ['.mha', '.mhd', '.raw']:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Expected .mha/.mhd")

    image = sitk.ReadImage(str(filepath))
    array = sitk.GetArrayFromImage(image)
    logger.debug(f"Loaded {filepath.name}: shape={array.shape}, dtype={array.dtype}, "
                 f"spacing={image.GetSpacing()}, origin={image.GetOrigin()}")
    return image, array


def save_mha(array: np.ndarray, filepath: Union[str, Path], ref_image: Optional[sitk.Image] = None) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    out_image = sitk.GetImageFromArray(array.astype(np.float32))
    if ref_image is not None:
        out_image.CopyInformation(ref_image)
    sitk.WriteImage(out_image, str(filepath))
    logger.info(f"Saved MHA to {filepath}")
    return filepath


def verify_image_mask_alignment(
    image_path: Union[str, Path],
    mask_path: Union[str, Path]
) -> Tuple[bool, dict]:
    img, img_arr = load_mha(image_path)
    mask, mask_arr = load_mha(mask_path)

    diagnostics = {
        'image_path': str(image_path),
        'mask_path': str(mask_path),
        'image_shape': img_arr.shape,
        'mask_shape': mask_arr.shape,
        'image_spacing': img.GetSpacing(),
        'mask_spacing': mask.GetSpacing(),
        'image_origin': img.GetOrigin(),
        'mask_origin': mask.GetOrigin(),
        'image_direction': img.GetDirection(),
        'mask_direction': mask.GetDirection(),
        'aligned_shape': img_arr.shape == mask_arr.shape,
        'aligned_spacing': img.GetSpacing() == mask.GetSpacing(),
        'aligned_origin': img.GetOrigin() == mask.GetOrigin(),
    }

    all_aligned = all([
        diagnostics['aligned_shape'],
        diagnostics['aligned_spacing'],
        diagnostics['aligned_origin'],
    ])

    return all_aligned, diagnostics


def verify_label_integrity(mask_array: np.ndarray, expected_labels: Optional[List[int]] = None) -> dict:
    if expected_labels is None:
        expected_labels = list(SPINE_LABEL_MAP.keys())

    unique_labels = np.unique(mask_array)
    unique_list = sorted(unique_labels.tolist())

    present = [l for l in expected_labels if l in unique_list]
    missing = [l for l in expected_labels if l not in unique_list]
    unexpected = [l for l in unique_list if l not in expected_labels]

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(mask_array, return_counts=True))}

    result = {
        'unique_labels': unique_list,
        'present_labels': present,
        'missing_labels': missing,
        'unexpected_labels': unexpected,
        'label_counts': label_counts,
        'valid': len(missing) == 0 and len(unexpected) == 0,
        'num_classes_found': len(unique_list),
    }
    return result


def generate_data_report(
    image_paths: List[Path],
    mask_paths: List[Path]
) -> dict:
    report = {
        'total_pairs': len(image_paths),
        'passed': 0,
        'failed': 0,
        'alignment_issues': [],
        'label_issues': [],
        'corrupt_files': [],
        'shape_stats': {},
        'spacing_stats': {},
    }

    shapes = []
    spacings = []

    for img_p, msk_p in zip(image_paths, mask_paths):
        try:
            aligned, diag = verify_image_mask_alignment(img_p, msk_p)
            _, mask_arr = load_mha(msk_p)
            label_info = verify_label_integrity(mask_arr)

            if not aligned:
                report['alignment_issues'].append({
                    'image': str(img_p),
                    'mask': str(msk_p),
                    'diagnostics': diag
                })

            if not label_info['valid']:
                report['label_issues'].append({
                    'mask': str(msk_p),
                    'info': label_info
                })

            shapes.append(diag['image_shape'])
            spacings.append(diag['image_spacing'])
            report['passed'] += 1

        except Exception as e:
            report['corrupt_files'].append({
                'file': str(img_p),
                'error': str(e)
            })
            report['failed'] += 1

    if shapes:
        shapes_arr = np.array(shapes)
        report['shape_stats'] = {
            'mean': shapes_arr.mean(axis=0).tolist(),
            'std': shapes_arr.std(axis=0).tolist(),
            'min': shapes_arr.min(axis=0).tolist(),
            'max': shapes_arr.max(axis=0).tolist(),
        }
    if spacings:
        spacings_arr = np.array(spacings)
        report['spacing_stats'] = {
            'mean': spacings_arr.mean(axis=0).tolist(),
            'std': spacings_arr.std(axis=0).tolist(),
            'min': spacings_arr.min(axis=0).tolist(),
            'max': spacings_arr.max(axis=0).tolist(),
        }

    return report


def get_class_distribution(mask_paths: List[Path], num_classes: int = NUM_CLASSES) -> dict:
    class_counts = {i: 0 for i in range(num_classes)}
    total_voxels = 0

    for mpath in mask_paths:
        _, mask_arr = load_mha(mpath)
        for label in range(num_classes):
            class_counts[label] += int(np.sum(mask_arr == label))
        total_voxels += mask_arr.size

    distribution = {}
    for label, count in class_counts.items():
        name = SPINE_LABEL_MAP.get(label, f'Unknown_{label}')
        pct = (count / total_voxels * 100) if total_voxels > 0 else 0
        distribution[label] = {
            'name': name,
            'count': count,
            'percentage': round(pct, 4)
        }

    distribution['_total_voxels'] = total_voxels
    return distribution
