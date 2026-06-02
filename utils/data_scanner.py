from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

MHA_EXTENSIONS = {'.mha', '.mhd'}


def scan_dataset_directory(
    data_dir: str,
    image_suffix: str = '_image',
    mask_suffix: str = '_mask',
    require_masks: bool = True,
    image_subdir: Optional[str] = None,
    mask_subdir: Optional[str] = None,
) -> Tuple[List[Path], List[Path]]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if image_subdir and mask_subdir:
        image_dir = data_path / image_subdir
        mask_dir = data_path / mask_subdir
        if not image_dir.exists():
            alt = data_path / 'images' / 'images'
            if alt.exists():
                image_dir = alt
                mask_dir = data_path / 'masks' / 'masks'

        image_files = sorted([
            f for f in image_dir.rglob('*')
            if f.suffix.lower() in MHA_EXTENSIONS
        ])
        mask_files = sorted([
            f for f in mask_dir.rglob('*')
            if f.suffix.lower() in MHA_EXTENSIONS
        ])

        logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks "
                    f"in subdirs ({image_subdir}, {mask_subdir})")

        if require_masks and len(image_files) == len(mask_files):
            return image_files, mask_files

        paired_images, paired_masks = pair_by_patient_id(image_files, mask_files)
        return paired_images, paired_masks

    all_mha_files = sorted([
        f for f in data_path.rglob('*')
        if f.suffix.lower() in MHA_EXTENSIONS
    ])

    if not all_mha_files:
        all_mha_files = sorted([
            f for f in data_path.rglob('*.*')
            if f.suffix.lower() in MHA_EXTENSIONS or f.suffix.lower() in ['.raw']
        ])
        all_mha_files = [f for f in all_mha_files if f.suffix.lower() in MHA_EXTENSIONS]

    image_files = sorted([f for f in all_mha_files if image_suffix in f.stem.lower()])
    mask_files = sorted([f for f in all_mha_files if mask_suffix in f.stem.lower()])

    if not image_files:
        image_files = [f for f in all_mha_files if 'img' in f.stem.lower() or 'image' in f.stem.lower()]
    if not mask_files and require_masks:
        mask_files = [f for f in all_mha_files if 'mask' in f.stem.lower() or 'seg' in f.stem.lower()]

    if not image_files:
        logger.warning(f"No image files found with suffix '{image_suffix}'. Using all .mha files.")
        image_files = all_mha_files

    if require_masks and not mask_files:
        logger.warning(f"No mask files found with suffix '{mask_suffix}'. Trying pattern matching...")
        image_stems = {f.stem.replace(image_suffix, '') for f in image_files}
        for f in all_mha_files:
            if f not in image_files:
                for stem in image_stems:
                    if stem in f.stem:
                        mask_files.append(f)
                        break
        mask_files = sorted(mask_files)

    logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks in {data_dir}")

    if require_masks and len(image_files) != len(mask_files):
        logger.warning(f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks. "
                       f"Attempting to pair by patient ID...")
        paired_images, paired_masks = pair_by_patient_id(image_files, mask_files)
        return paired_images, paired_masks

    return image_files, mask_files


def pair_by_patient_id(
    image_files: List[Path],
    mask_files: List[Path]
) -> Tuple[List[Path], List[Path]]:
    def get_patient_id(path: Path) -> str:
        parts = path.stem.split('_')
        return parts[0] if parts else path.stem

    image_map = {}
    for f in image_files:
        pid = get_patient_id(f)
        image_map[pid] = f

    mask_map = {}
    for f in mask_files:
        pid = get_patient_id(f)
        mask_map[pid] = f

    common_ids = sorted(set(image_map.keys()) & set(mask_map.keys()))
    paired_images = [image_map[pid] for pid in common_ids]
    paired_masks = [mask_map[pid] for pid in common_ids]

    logger.info(f"Paired {len(paired_images)} image-mask pairs by patient ID")
    return paired_images, paired_masks


def validate_data_directory(data_dir: str) -> dict:
    try:
        images, masks = scan_dataset_directory(data_dir)

        from .mha_utils import generate_data_report
        report = generate_data_report(images, masks)

        report['data_dir'] = data_dir
        report['num_images'] = len(images)
        report['num_masks'] = len(masks)

        if report['failed'] > 0:
            report['status'] = 'FAILED'
            report['message'] = f"{report['failed']} files failed validation"
        elif len(report['alignment_issues']) > 0:
            report['status'] = 'WARNING'
            report['message'] = f"{len(report['alignment_issues'])} alignment issues found"
        elif len(report['label_issues']) > 0:
            report['status'] = 'WARNING'
            report['message'] = f"{len(report['label_issues'])} label issues found"
        else:
            report['status'] = 'PASSED'
            report['message'] = 'All validations passed'

        return report

    except Exception as e:
        return {
            'status': 'ERROR',
            'message': str(e),
            'data_dir': data_dir,
        }
