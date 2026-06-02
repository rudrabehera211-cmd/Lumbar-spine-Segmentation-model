import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import tempfile
import SimpleITK as sitk

from inference import Predictor
from models.unet import UNet


def test_inference():
    print("=" * 60)
    print("INFERENCE TESTS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'model': {'name': 'unet', 'num_classes': 12, 'in_channels': 1},
        'inference': {'tta': False},
    }

    model = UNet(in_channels=1, out_channels=12).to(device).eval()
    predictor = Predictor(model, config, device)

    with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as f:
        img_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as f:
        mask_path = f.name

    try:
        shape = (16, 64, 64)
        img_data = np.random.randn(*shape).astype(np.float32)
        mask_data = np.random.randint(0, 12, size=shape).astype(np.int16)

        sitk_img = sitk.GetImageFromArray(img_data)
        sitk.WriteImage(sitk_img, img_path)

        # Test prediction
        pred = predictor.predict_volume(img_path)
        assert pred.shape == shape, f"Prediction shape mismatch: {pred.shape} != {shape}"
        assert pred.dtype == np.int16, f"Prediction dtype: {pred.dtype}"
        assert 0 <= pred.min() <= pred.max() <= 11, f"Prediction label range violation: [{pred.min()}, {pred.max()}]"
        print(f"  Single volume prediction: PASSED (shape={pred.shape})")

        # Test with return probabilities
        pred2, probs = predictor.predict_volume(img_path, return_probabilities=True)
        assert pred2.shape == shape, f"Pred shape mismatch"
        assert probs.shape[0] == 12, f"Probabilities channels: {probs.shape[0]}"
        print(f"  Probability prediction: PASSED (probs shape={probs.shape})")

        # Test batch prediction
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = predictor.predict_batch([img_path], tmpdir)
            assert len(saved) == 1, f"Batch prediction count: {len(saved)}"
            assert saved[0].exists(), f"Batch output not found"
            print(f"  Batch prediction: PASSED")

        # Test folder prediction
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / 'imgs'
            out_dir = Path(tmpdir) / 'out'
            img_dir.mkdir()
            for i in range(3):
                p = img_dir / f'test_{i}.mha'
                d = np.random.randn(*shape).astype(np.float32)
                sitk.WriteImage(sitk.GetImageFromArray(d), str(p))
            saved = predictor.predict_folder(img_dir, out_dir)
            assert len(saved) == 3, f"Folder prediction count: {len(saved)}"
            print(f"  Folder prediction: PASSED")

    finally:
        Path(img_path).unlink(missing_ok=True)
        Path(mask_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("ALL INFERENCE TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    test_inference()
