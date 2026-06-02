import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def resample_to_spacing(
    image: np.ndarray,
    original_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
    interpolator: str = 'linear',
    original_direction: Optional[Tuple[float, ...]] = None,
    original_origin: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    if not HAS_SITK:
        logger.warning("SimpleITK not available. Using zoom resampling.")
        from scipy.ndimage import zoom

        scale_factors = [
            orig / target for orig, target in zip(original_spacing, target_spacing)
        ]
        order = 1 if interpolator == 'linear' else 0
        resampled = zoom(image.astype(np.float64), scale_factors, order=order)
        return resampled.astype(np.float32)

    img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    img_sitk.SetSpacing(original_spacing)

    if original_direction is not None:
        img_sitk.SetDirection(original_direction)
    if original_origin is not None:
        img_sitk.SetOrigin(original_origin)

    interp = sitk.sitkLinear if interpolator == 'linear' else sitk.sitkNearestNeighbor

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([
        int(round(img_sitk.GetSize()[i] * original_spacing[i] / target_spacing[i]))
        for i in range(img_sitk.GetDimension())
    ])
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(img_sitk)
    return sitk.GetArrayFromImage(resampled).astype(np.float32 if interpolator == 'linear' else np.int16)


def resize_volume(
    image: np.ndarray,
    target_size: Tuple[int, ...],
    interpolator: str = 'linear'
) -> np.ndarray:
    from scipy.ndimage import zoom

    scale_factors = [
        target / orig for target, orig in zip(target_size, image.shape)
    ]
    order = 1 if interpolator == 'linear' else 0
    resized = zoom(image.astype(np.float64), scale_factors, order=order)
    return resized.astype(np.float32 if interpolator == 'linear' else np.int16)


def resize_mask(mask: np.ndarray, target_size: Tuple[int, ...]) -> np.ndarray:
    from scipy.ndimage import zoom

    scale_factors = [
        target / orig for target, orig in zip(target_size, mask.shape)
    ]

    classes = np.unique(mask)
    warped = np.zeros(target_size, dtype=np.int16)

    for c in classes:
        if c == 0:
            continue
        binary = (mask == c).astype(np.float64)
        warped_binary = zoom(binary, scale_factors, order=1)
        warped[warped_binary > 0.5] = c

    background = np.ones_like(warped)
    for c in classes:
        if c == 0:
            continue
        background[warped == c] = 0
    warped[background > 0] = 0

    return warped.astype(np.int16)


def extract_spine_roi(
    image: np.ndarray,
    mask: np.ndarray,
    margin: int = 10
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    nonzero = np.array(np.where(mask > 0))
    if nonzero.shape[1] == 0:
        logger.warning("Empty mask for ROI extraction. Returning original.")
        return image, mask, (0,) * 3

    min_coords = nonzero.min(axis=1) - margin
    max_coords = nonzero.max(axis=1) + margin + 1

    min_coords = np.maximum(0, min_coords)
    max_coords = np.minimum(np.array(image.shape), max_coords)

    slices = tuple(slice(int(min_coords[i]), int(max_coords[i])) for i in range(3))
    cropped_image = image[slices]
    cropped_mask = mask[slices]

    return cropped_image, cropped_mask, tuple(min_coords)


def apply_spine_centered_crop(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    roi_img, roi_mask, _ = extract_spine_roi(image, mask, margin=5)
    resized_img = resize_volume(roi_img, output_size)
    resized_mask = resize_mask(roi_mask, output_size)
    return resized_img, resized_mask
