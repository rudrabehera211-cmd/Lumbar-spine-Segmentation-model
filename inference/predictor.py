import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from tqdm import tqdm

from utils.mha_utils import load_mha, save_mha, SPINE_LABEL_MAP
from preprocessing.normalization import z_score_normalize

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.num_classes = config.get('model', {}).get('num_classes', 12)
        self.use_tta = config.get('inference', {}).get('tta', False)

    @torch.no_grad()
    def predict_single(
        self,
        image: np.ndarray,
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if image.ndim == 3:
            image = image[np.newaxis, np.newaxis, ...]
        elif image.ndim == 4:
            image = image[np.newaxis, ...]

        image = torch.from_numpy(image.astype(np.float32)).to(self.device)

        if self.use_tta:
            pred = self._predict_with_tta(image)
        else:
            pred = self.model(image)
            if isinstance(pred, list):
                pred = pred[0]
            pred = F.softmax(pred, dim=1)

        probs = pred.cpu().numpy().squeeze()
        mask = np.argmax(probs, axis=0).astype(np.int16)

        if return_probabilities:
            return mask, probs
        return mask

    @torch.no_grad()
    def _predict_with_tta(self, image: torch.Tensor) -> torch.Tensor:
        preds = []

        pred = self.model(image)
        if isinstance(pred, list):
            pred = pred[0]
        preds.append(F.softmax(pred, dim=1))

        flipped_dims = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4)]

        for dims in flipped_dims:
            img_flip = torch.flip(image, dims=dims)
            pred_flip = self.model(img_flip)
            if isinstance(pred_flip, list):
                pred_flip = pred_flip[0]
            pred_flip = F.softmax(pred_flip, dim=1)
            pred_flip = torch.flip(pred_flip, dims=dims)
            preds.append(pred_flip)

        return torch.stack(preds).mean(dim=0)

    def predict_volume(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        image_path = Path(image_path)
        logger.info(f"Predicting: {image_path.name}")

        sitk_img, image_arr = load_mha(image_path)

        orig_shape = image_arr.shape
        image_norm = z_score_normalize(image_arr.astype(np.float32))

        mask, probs = self.predict_single(image_norm, return_probabilities=True)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_mha(mask, output_path, ref_image=sitk_img)
            logger.info(f"Saved prediction to {output_path}")

            if return_probabilities:
                prob_path = output_path.parent / f"{output_path.stem}_probs{output_path.suffix}"
                save_mha(probs.astype(np.float32), prob_path, ref_image=sitk_img)

        if return_probabilities:
            return mask, probs
        return mask

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
    ) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for img_path in tqdm(image_paths, desc="Batch predicting"):
            out_path = output_dir / f"{Path(img_path).stem}_pred.mha"
            self.predict_volume(img_path, out_path)
            saved_paths.append(out_path)

        return saved_paths

    def predict_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = '*.mha',
    ) -> List[Path]:
        input_dir = Path(input_dir)
        files = sorted(input_dir.glob(pattern))
        if not files:
            files = sorted(input_dir.glob('*.mhd'))

        if not files:
            logger.warning(f"No MHA files found in {input_dir} with pattern {pattern}")
            return []

        logger.info(f"Found {len(files)} files for batch prediction")
        return self.predict_batch(files, output_dir)
