import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import logging
import random
from scipy.ndimage import zoom, rotate, map_coordinates, gaussian_filter

logger = logging.getLogger(__name__)


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomRotation:
    def __init__(self, degrees: float = 10, p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        angle = random.uniform(-self.degrees, self.degrees)
        axes = [(1, 2), (0, 2), (0, 1)]
        chosen_axes = random.choice(axes)

        rotated_img = rotate(image.astype(np.float64), angle, axes=chosen_axes,
                             order=1, mode='nearest', reshape=False)
        rotated_mask = rotate(mask.astype(np.float64), angle, axes=chosen_axes,
                              order=0, mode='nearest', reshape=False)

        return rotated_img.astype(np.float32), np.round(rotated_mask).astype(np.int16)


class RandomTranslation:
    def __init__(self, max_px: int = 10, p: float = 0.5):
        self.max_px = max_px
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        shift = [random.randint(-self.max_px, self.max_px) for _ in range(image.ndim)]
        shifted_img = np.roll(image, shift, axis=tuple(range(image.ndim)))
        shifted_mask = np.roll(mask, shift, axis=tuple(range(mask.ndim)))

        return shifted_img.astype(np.float32), shifted_mask.astype(np.int16)


class RandomScaling:
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        scale = random.uniform(*self.scale_range)
        factors = [scale] * image.ndim
        scaled_img = zoom(image.astype(np.float64), factors, order=1)

        classes = np.unique(mask)
        scaled_mask = np.zeros_like(scaled_img, dtype=np.int16)
        for c in classes:
            if c == 0:
                continue
            binary = (mask == c).astype(np.float64)
            warped = zoom(binary, factors, order=1)
            scaled_mask[warped > 0.5] = c
        scaled_mask[scaled_mask == 0] = 0

        slices = tuple(slice(0, min(s, t)) for s, t in zip(scaled_img.shape, image.shape))
        result_img = np.zeros_like(image, dtype=np.float32)
        result_mask = np.zeros_like(mask, dtype=np.int16)
        result_img[slices] = scaled_img[slices]
        result_mask[slices] = scaled_mask[slices]

        return result_img, result_mask


class ElasticDeformation:
    def __init__(self, alpha: float = 20, sigma: float = 5, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        shape = image.shape
        dims = len(shape)
        dtype = image.dtype

        displacements = [gaussian_filter(
            np.random.randn(*shape).astype(np.float64) * self.alpha,
            self.sigma, mode='constant'
        ) for _ in range(dims)]

        grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
        indices = [grid[d] + displacements[d] for d in range(dims)]

        coords = np.array(indices).reshape(dims, -1)
        warped_img = map_coordinates(image.astype(np.float64), coords, order=1, mode='nearest').reshape(shape)
        warped_mask = map_coordinates(mask.astype(np.float64), coords, order=0, mode='nearest').reshape(shape)

        return warped_img.astype(np.float32), np.round(warped_mask).astype(np.int16)


class GaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
        noisy = image + noise
        return noisy, mask


class RandomContrast:
    def __init__(self, contrast_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        factor = random.uniform(*self.contrast_range)
        mean = np.mean(image)
        adjusted = mean + factor * (image - mean)
        return adjusted.astype(np.float32), mask


class RandomBrightness:
    def __init__(self, brightness_range: Tuple[float, float] = (-0.1, 0.1), p: float = 0.5):
        self.brightness_range = brightness_range
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        shift = random.uniform(*self.brightness_range)
        adjusted = image + shift
        return adjusted.astype(np.float32), mask


class MRISpecificAugmentation:
    def __init__(self, p: float = 0.3):
        self.p = p
        self.ghost_noise = GaussianNoise(0, 0.005, p=0.5)
        self.contrast = RandomContrast((0.9, 1.1), p=0.5)
        self.brightness = RandomBrightness((-0.05, 0.05), p=0.5)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask
        image, mask = self.ghost_noise(image, mask)
        image, mask = self.contrast(image, mask)
        image, mask = self.brightness(image, mask)
        return image, mask


def get_training_augmentation(config: Dict[str, Any]) -> Compose:
    aug_params = config.get('augmentation', {})
    transforms = []

    if aug_params.get('rotation', True):
        transforms.append(RandomRotation(
            degrees=aug_params.get('rotation_degrees', 10),
            p=aug_params.get('rotation_prob', 0.5)
        ))

    if aug_params.get('translation', True):
        transforms.append(RandomTranslation(
            max_px=aug_params.get('translation_px', 10),
            p=aug_params.get('translation_prob', 0.5)
        ))

    if aug_params.get('scaling', True):
        transforms.append(RandomScaling(
            scale_range=tuple(aug_params.get('scaling_range', (0.9, 1.1))),
            p=aug_params.get('scaling_prob', 0.5)
        ))

    if aug_params.get('elastic_deformation', True):
        transforms.append(ElasticDeformation(
            alpha=aug_params.get('elastic_alpha', 20),
            sigma=aug_params.get('elastic_sigma', 5),
            p=aug_params.get('elastic_prob', 0.3)
        ))

    if aug_params.get('gaussian_noise', True):
        transforms.append(GaussianNoise(
            mean=aug_params.get('noise_mean', 0.0),
            std=aug_params.get('noise_std', 0.01),
            p=aug_params.get('noise_prob', 0.3)
        ))

    if aug_params.get('contrast', True):
        transforms.append(RandomContrast(
            contrast_range=tuple(aug_params.get('contrast_range', (0.8, 1.2))),
            p=aug_params.get('contrast_prob', 0.5)
        ))

    if aug_params.get('brightness', True):
        transforms.append(RandomBrightness(
            brightness_range=tuple(aug_params.get('brightness_range', (-0.1, 0.1))),
            p=aug_params.get('brightness_prob', 0.5)
        ))

    if aug_params.get('mri_specific', True):
        transforms.append(MRISpecificAugmentation(p=aug_params.get('mri_prob', 0.3)))

    return Compose(transforms) if transforms else Compose([])
