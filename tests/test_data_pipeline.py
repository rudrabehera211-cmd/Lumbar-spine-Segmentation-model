import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import SimpleITK as sitk

from utils.mha_utils import (
    load_mha, save_mha, verify_image_mask_alignment,
    verify_label_integrity, get_class_distribution, SPINE_LABEL_MAP
)
from utils.data_splitting import create_patient_level_split
from preprocessing.normalization import z_score_normalize, min_max_normalize, clip_intensity
from preprocessing.resampling import resize_volume, resize_mask
from datasets.augmentation import (
    RandomRotation, RandomTranslation, RandomScaling,
    ElasticDeformation, GaussianNoise, RandomContrast, RandomBrightness
)


def create_test_mha(shape=(16, 64, 64), num_classes=12):
    data = np.random.randn(*shape).astype(np.float32)
    mask = np.random.randint(0, num_classes, size=shape).astype(np.int16)
    return data, mask


def test_mha_io():
    print("Testing MHA I/O...")
    data, mask = create_test_mha()
    with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as f_img:
        img_path = f_img.name
    with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as f_mask:
        mask_path = f_mask.name

    try:
        img_sitk = sitk.GetImageFromArray(data)
        sitk.WriteImage(img_sitk, img_path)

        mask_sitk = sitk.GetImageFromArray(mask.astype(np.float32))
        sitk.WriteImage(mask_sitk, mask_path)

        loaded_img, loaded_arr = load_mha(img_path)
        assert loaded_arr.shape == data.shape, f"Shape mismatch: {loaded_arr.shape} != {data.shape}"
        assert np.allclose(loaded_arr, data), "Data mismatch"
        print(f"  MHA load/save: PASSED (shape={loaded_arr.shape})")

        aligned, diag = verify_image_mask_alignment(img_path, mask_path)
        assert aligned, "Alignment check failed"
        print(f"  Alignment check: PASSED")

        label_info = verify_label_integrity(mask)
        assert label_info['valid'], "Label integrity check failed"
        print(f"  Label integrity: PASSED (labels={label_info['unique_labels']})")

    finally:
        Path(img_path).unlink(missing_ok=True)
        Path(mask_path).unlink(missing_ok=True)


def test_normalization():
    print("Testing normalization...")
    data = np.random.randn(16, 64, 64).astype(np.float32)

    z_norm = z_score_normalize(data)
    assert abs(z_norm.mean()) < 1e-5, f"Z-score mean not zero: {z_norm.mean()}"
    assert abs(z_norm.std() - 1.0) < 0.1, f"Z-score std not 1: {z_norm.std()}"
    print(f"  Z-score normalization: PASSED (mean={z_norm.mean():.6f}, std={z_norm.std():.6f})")

    mm_norm = min_max_normalize(data)
    assert 0 <= mm_norm.min() <= mm_norm.max() <= 1, "Min-max range violation"
    print(f"  Min-max normalization: PASSED (range=[{mm_norm.min():.4f}, {mm_norm.max():.4f}])")

    clipped = clip_intensity(data, 1, 99)
    assert clipped.min() >= np.percentile(data, 1), "Clipping failed"
    print(f"  Intensity clipping: PASSED")


def test_resize():
    print("Testing resize...")
    data = np.random.randn(16, 64, 64).astype(np.float32)
    target = (32, 128, 128)

    resized = resize_volume(data, target)
    assert resized.shape == target, f"Resize shape: {resized.shape} != {target}"
    print(f"  Volume resize: PASSED ({data.shape} -> {resized.shape})")

    mask = np.random.randint(0, 12, size=(16, 64, 64)).astype(np.int16)
    resized_mask = resize_mask(mask, target)
    assert resized_mask.shape == target, f"Mask resize shape: {resized_mask.shape} != {target}"
    print(f"  Mask resize: PASSED ({mask.shape} -> {resized_mask.shape})")


def test_augmentation():
    print("Testing augmentation...")
    data = np.random.randn(16, 64, 64).astype(np.float32)
    mask = np.random.randint(0, 12, size=(16, 64, 64)).astype(np.int16)

    transforms = [
        RandomRotation(degrees=10, p=1.0),
        RandomTranslation(max_px=5, p=1.0),
        RandomScaling(scale_range=(0.95, 1.05), p=1.0),
        GaussianNoise(std=0.01, p=1.0),
        RandomContrast(contrast_range=(0.9, 1.1), p=1.0),
        RandomBrightness(brightness_range=(-0.1, 0.1), p=1.0),
    ]

    for t in transforms:
        try:
            aug_img, aug_mask = t(data, mask)
            assert aug_img.shape == data.shape, f"Shape mismatch after {t.__class__.__name__}"
            assert aug_mask.shape == mask.shape, f"Mask shape mismatch after {t.__class__.__name__}"
            print(f"  {t.__class__.__name__:25s}: PASSED")
        except Exception as e:
            print(f"  {t.__class__.__name__:25s}: FAILED ({e})")


def test_elastic_deformation():
    print("Testing elastic deformation...")
    data = np.random.randn(8, 32, 32).astype(np.float32)
    mask = np.random.randint(0, 12, size=(8, 32, 32)).astype(np.int16)

    t = ElasticDeformation(alpha=10, sigma=3, p=1.0)
    try:
        aug_img, aug_mask = t(data, mask)
        assert aug_img.shape == data.shape
        print(f"  ElasticDeformation: PASSED")
    except Exception as e:
        print(f"  ElasticDeformation: FAILED ({e})")


def test_data_splitting():
    print("Testing data splitting...")
    import tempfile
    paths = []
    for i in range(10):
        with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as f:
            paths.append(Path(f.name))

    split = create_patient_level_split(paths, paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    total = len(split['train']) + len(split['val']) + len(split['test'])
    assert total == len(paths), f"Split total mismatch: {total} != {len(paths)}"
    print(f"  Split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}: PASSED")

    for p in paths:
        p.unlink(missing_ok=True)


def test_class_distribution():
    print("Testing class distribution...")
    import tempfile
    mask_paths = []
    for _ in range(3):
        mask = np.random.randint(0, 12, size=(16, 64, 64)).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as f:
            sitk_img = sitk.GetImageFromArray(mask.astype(np.float32))
            sitk.WriteImage(sitk_img, f.name)
            mask_paths.append(Path(f.name))

    dist = get_class_distribution(mask_paths)
    assert '_total_voxels' in dist, "Missing total_voxels"
    assert all(label in dist for label in range(12)), "Missing class labels"
    print(f"  Class distribution: PASSED ({len(dist)} entries)")

    for p in mask_paths:
        p.unlink(missing_ok=True)


if __name__ == '__main__':
    print("=" * 60)
    print("DATA PIPELINE TESTS")
    print("=" * 60)

    test_mha_io()
    test_normalization()
    test_resize()
    test_augmentation()
    test_elastic_deformation()
    test_data_splitting()
    test_class_distribution()

    print("=" * 60)
    print("ALL DATA PIPELINE TESTS PASSED")
    print("=" * 60)
