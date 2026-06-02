import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging
from utils.mha_utils import load_mha
from datasets.augmentation import get_training_augmentation, Compose

logger = logging.getLogger(__name__)


class SpineDataset2D(Dataset):
    def __init__(
        self,
        file_pairs: List[Tuple[Path, Path]],
        phase: str = 'train',
        config: Optional[Dict[str, Any]] = None,
        transforms: Optional[Compose] = None,
        cache_slices: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        num_classes: int = 12,
    ):
        super().__init__()
        self.file_pairs = file_pairs
        self.phase = phase
        self.config = config or {}
        self.transforms = transforms
        self.cache_slices = cache_slices
        self.target_size = target_size
        self.num_classes = num_classes
        self.cache = {}
        self.index_map = []
        self._build_index()

    def _build_index(self):
        for idx, (img_path, mask_path) in enumerate(self.file_pairs):
            if self.cache_slices and idx in self.cache:
                img_arr, mask_arr = self.cache[idx]
            else:
                _, img_arr = load_mha(img_path)
                _, mask_arr = load_mha(mask_path)
                if self.cache_slices:
                    self.cache[idx] = (img_arr, mask_arr)

            num_slices = img_arr.shape[0]
            for s in range(num_slices):
                self.index_map.append((idx, s))

        logger.info(f"Built 2D index: {len(self.index_map)} slices from {len(self.file_pairs)} volumes")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        volume_idx, slice_idx = self.index_map[i]

        if volume_idx in self.cache:
            img_arr, mask_arr = self.cache[volume_idx]
        else:
            img_path, mask_path = self.file_pairs[volume_idx]
            _, img_arr = load_mha(img_path)
            _, mask_arr = load_mha(mask_path)

        image = img_arr[slice_idx].astype(np.float32)
        mask = mask_arr[slice_idx].astype(np.int16)

        if self.target_size:
            from preprocessing.resampling import resize_volume, resize_mask
            orig_shape = image.shape
            image_3d = image[np.newaxis, ...]
            mask_3d = mask[np.newaxis, ...]
            image_3d = resize_volume(image_3d, (1, *self.target_size))
            mask_3d = resize_mask(mask_3d, (1, *self.target_size))
            image = image_3d[0]
            mask = mask_3d[0]

        image = np.expand_dims(image, axis=0)

        if self.transforms:
            image_3d = image[np.newaxis, ...]
            mask_3d = mask[np.newaxis, ...]
            image_3d, mask_3d = self.transforms(image_3d, mask_3d)
            image = image_3d[0]
            mask = mask_3d[0]

        mask = torch.from_numpy(mask).long()
        image = torch.from_numpy(image).float()

        return {
            'image': image,
            'mask': mask,
            'volume_idx': volume_idx,
            'slice_idx': slice_idx,
        }


class SpineDataset3D(Dataset):
    def __init__(
        self,
        file_pairs: List[Tuple[Path, Path]],
        phase: str = 'train',
        config: Optional[Dict[str, Any]] = None,
        transforms: Optional[Compose] = None,
        cache_volumes: bool = True,
        target_size: Optional[Tuple[int, int, int]] = None,
        num_classes: int = 12,
    ):
        super().__init__()
        self.file_pairs = file_pairs
        self.phase = phase
        self.config = config or {}
        self.transforms = transforms
        self.cache_volumes = cache_volumes
        self.target_size = target_size
        self.num_classes = num_classes
        self.cache = {}

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        if idx in self.cache:
            img_arr, mask_arr = self.cache[idx]
        else:
            img_path, mask_path = self.file_pairs[idx]
            _, img_arr = load_mha(img_path)
            _, mask_arr = load_mha(mask_path)
            if self.cache_volumes:
                self.cache[idx] = (img_arr, mask_arr)

        image = img_arr.astype(np.float32)
        mask = mask_arr.astype(np.int16)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        if self.target_size:
            from preprocessing.resampling import resize_volume, resize_mask
            image = resize_volume(image, self.target_size, interpolator='linear')
            mask = resize_mask(mask, self.target_size)

        image = np.expand_dims(image, axis=0)
        mask = torch.from_numpy(mask).long()
        image = torch.from_numpy(image).float()

        return {
            'image': image,
            'mask': mask,
            'volume_idx': idx,
        }


def create_dataloaders(
    split: Dict[str, List[Tuple[Path, Path]]],
    config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    dataloaders = {}
    batch_size = config.get('training', {}).get('batch_size', 2)
    num_workers = config.get('training', {}).get('num_workers', 0)
    use_3d = config.get('dataset', {}).get('use_3d', False)
    target_size = config.get('preprocessing', {}).get('target_size')
    num_classes = config.get('model', {}).get('num_classes', 12)

    DatasetClass = SpineDataset3D if use_3d else SpineDataset2D

    for phase in ['train', 'val', 'test']:
        pairs = split.get(phase, [])
        if not pairs:
            dataloaders[phase] = None
            continue

        transforms = get_training_augmentation(config) if phase == 'train' else None

        dataset = DatasetClass(
            file_pairs=pairs,
            phase=phase,
            config=config,
            transforms=transforms,
            target_size=tuple(target_size) if target_size else None,
            num_classes=num_classes,
        )

        shuffle = (phase == 'train')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(phase == 'train'),
        )
        dataloaders[phase] = loader
        logger.info(f"{phase}: {len(dataset)} samples, {len(loader)} batches")

    return dataloaders
