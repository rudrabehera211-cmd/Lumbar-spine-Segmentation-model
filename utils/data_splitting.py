from pathlib import Path
from typing import List, Tuple, Dict
import random
import json
import logging

logger = logging.getLogger(__name__)


def create_patient_level_split(
    image_paths: List[Path],
    mask_paths: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    patient_id_fn=None,
    stratify_on: str = 'patient'
) -> Dict[str, List[Tuple[Path, Path]]]:
    ratios = train_ratio + val_ratio + test_ratio
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1, got {ratios}")

    if patient_id_fn is None:
        def default_patient_id(path: Path) -> str:
            return path.stem.split('_')[0]
        patient_id_fn = default_patient_id

    patient_map: Dict[str, List[Tuple[Path, Path]]] = {}
    for img_p, msk_p in zip(image_paths, mask_paths):
        pid = patient_id_fn(img_p)
        if pid not in patient_map:
            patient_map[pid] = []
        patient_map[pid].append((img_p, msk_p))

    patients = list(patient_map.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_total = len(patients)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    split = {
        'train': [],
        'val': [],
        'test': [],
    }

    for p in train_patients:
        split['train'].extend(patient_map[p])
    for p in val_patients:
        split['val'].extend(patient_map[p])
    for p in test_patients:
        split['test'].extend(patient_map[p])

    logger.info(f"Split: train={len(split['train'])} samples ({len(train_patients)} patients), "
                f"val={len(split['val'])} samples ({len(val_patients)} patients), "
                f"test={len(split['test'])} samples ({len(test_patients)} patients)")

    return split


def save_split_info(split: Dict, filepath: Path):
    info = {}
    for phase in ['train', 'val', 'test']:
        info[phase] = [
            {'image': str(img), 'mask': str(msk)}
            for img, msk in split[phase]
        ]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Split info saved to {filepath}")


def load_split_info(filepath: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    with open(filepath, 'r') as f:
        info = json.load(f)
    split = {}
    for phase in ['train', 'val', 'test']:
        split[phase] = [(Path(item['image']), Path(item['mask'])) for item in info[phase]]
    return split
