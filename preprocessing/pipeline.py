import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import logging
from .normalization import z_score_normalize, clip_intensity, bias_field_correction, min_max_normalize
from .resampling import resample_to_spacing, extract_spine_roi, resize_volume, resize_mask

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.params = config.get('preprocessing', {})

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, ...]] = None,
        direction: Optional[Tuple[float, ...]] = None,
        origin: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, np.ndarray]:
        result = {'image': image.copy(), 'mask': mask.copy() if mask is not None else None}
        steps_applied = []

        if self.params.get('clip_intensity', False):
            percentiles = self.params.get('clip_percentiles', (0.5, 99.5))
            result['image'] = clip_intensity(result['image'], percentiles[0], percentiles[1])
            steps_applied.append('clip_intensity')

        if self.params.get('bias_correction', False):
            result['image'] = bias_field_correction(result['image'], result['mask'])
            steps_applied.append('bias_correction')

        if self.params.get('zscore_normalize', True):
            result['image'] = z_score_normalize(result['image'], result['mask'])
            steps_applied.append('zscore_normalize')

        if self.params.get('minmax_normalize', False):
            result['image'] = min_max_normalize(
                result['image'],
                tuple(self.params['normalize_percentiles']) if self.params.get('normalize_percentiles') else None
            )
            steps_applied.append('minmax_normalize')

        if self.params.get('extract_roi', False) and result['mask'] is not None:
            margin = self.params.get('roi_margin', 10)
            result['image'], result['mask'], result['roi_offset'] = extract_spine_roi(
                result['image'], result['mask'], margin
            )
            steps_applied.append('extract_roi')

        target_spacing = self.params.get('target_spacing')
        if target_spacing and spacing:
            result['image'] = resample_to_spacing(
                result['image'], spacing, tuple(target_spacing),
                interpolator='linear', original_direction=direction, original_origin=origin
            )
            if result['mask'] is not None:
                result['mask'] = resample_to_spacing(
                    result['mask'], spacing, tuple(target_spacing),
                    interpolator='nearest', original_direction=direction, original_origin=origin
                ).astype(np.int16)
            steps_applied.append('resample_spacing')

        target_size = self.params.get('target_size')
        if target_size:
            result['image'] = resize_volume(result['image'], tuple(target_size))
            if result['mask'] is not None:
                result['mask'] = resize_mask(result['mask'], tuple(target_size))
            steps_applied.append('resize')

        result['steps_applied'] = steps_applied
        logger.debug(f"Preprocessing steps: {steps_applied}")
        return result
