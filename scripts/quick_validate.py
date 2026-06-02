"""Quick data validation for the dataset - samples first few files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_scanner import scan_dataset_directory
from utils.mha_utils import verify_image_mask_alignment, verify_label_integrity, load_mha, get_class_distribution
import json
import numpy as np

print("=" * 60)
print("DATA VALIDATION REPORT")
print("=" * 60)

images, masks = scan_dataset_directory(
    './10159290 (1)',
    image_subdir='images/images',
    mask_subdir='masks/masks',
)
print(f"\nTotal pairs: {len(images)}")

print("\n--- Sample Verification (first 5 pairs) ---")
alignment_ok = 0
label_ok = 0
all_labels = set()

for i in range(min(5, len(images))):
    aligned, diag = verify_image_mask_alignment(images[i], masks[i])
    _, mask_arr = load_mha(masks[i])
    label_info = verify_label_integrity(mask_arr)

    name = images[i].name
    print(f"({i+1}) {name}:")
    print(f"    Aligned={aligned}, Shape={diag['image_shape']}, Spacing={diag['image_spacing']}")
    print(f"    Labels={label_info['unique_labels']}, Valid={label_info['valid']}")

    if aligned:
        alignment_ok += 1
    if label_info['valid']:
        label_ok += 1
    all_labels.update(label_info['unique_labels'])

print(f"\nAlignment OK: {alignment_ok}/5")
print(f"Label OK: {label_ok}/5")
print(f"All unique labels found: {sorted(all_labels)}")

# Class distribution from sample
print("\n--- Class Distribution (first 5 files) ---")
dist = get_class_distribution(masks[:5])
total = dist.pop('_total_voxels', 0)
for label in sorted(int(k) for k in dist.keys()):
    info = dist[str(label)] if isinstance(dist, dict) and str(label) in dist else dist.get(label)
    if isinstance(info, dict):
        print(f"  Label {label:2d} ({info['name']:20s}): {info['percentage']:6.4f}% ({info['count']:>10,} voxels)")

out_dir = Path('reports')
out_dir.mkdir(exist_ok=True)
with open(out_dir / 'class_distribution_sample.json', 'w') as f:
    json.dump({'_total_voxels': total, **dist}, f, indent=2)

print(f"\nSample validation complete. Reports saved to {out_dir}")
print("=" * 60)
