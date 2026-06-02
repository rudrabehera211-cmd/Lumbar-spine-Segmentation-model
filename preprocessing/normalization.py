import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def z_score_normalize(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        voxels = image[mask > 0]
        if len(voxels) == 0:
            voxels = image.flatten()
    else:
        voxels = image.flatten()

    mean = np.mean(voxels).astype(np.float64)
    std = np.std(voxels).astype(np.float64)

    if std < 1e-8:
        logger.warning(f"Near-zero std ({std:.6f}) during z-score normalization. Using image-level stats.")
        mean = np.mean(image).astype(np.float64)
        std = np.std(image).astype(np.float64)
        if std < 1e-8:
            return np.zeros_like(image, dtype=np.float32)

    normalized = (image.astype(np.float64) - mean) / std
    return normalized.astype(np.float32)


def min_max_normalize(image: np.ndarray, percentile: Optional[Tuple[float, float]] = None) -> np.ndarray:
    img = image.astype(np.float64)

    if percentile:
        low, high = np.percentile(img, percentile)
    else:
        low, high = img.min(), img.max()

    if high - low < 1e-8:
        logger.warning("Near-zero range in min-max normalization.")
        return np.zeros_like(img, dtype=np.float32)

    normalized = (img - low) / (high - low)
    return np.clip(normalized, 0, 1).astype(np.float32)


def clip_intensity(image: np.ndarray, low_percentile: float = 0.5, high_percentile: float = 99.5) -> np.ndarray:
    low = np.percentile(image, low_percentile)
    high = np.percentile(image, high_percentile)
    clipped = np.clip(image, low, high)
    logger.debug(f"Clipped intensities: [{low:.2f}, {high:.2f}]")
    return clipped


def bias_field_correction(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    try:
        import SimpleITK as sitk

        img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        if mask is not None:
            mask_sitk = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
        else:
            mask_sitk = sitk.OtsuThreshold(img_sitk, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(img_sitk, mask_sitk)
        result = sitk.GetArrayFromImage(corrected)
        logger.info("Bias field correction applied")
        return result.astype(np.float32)

    except Exception as e:
        logger.warning(f"Bias field correction failed: {e}. Returning original image.")
        return image
