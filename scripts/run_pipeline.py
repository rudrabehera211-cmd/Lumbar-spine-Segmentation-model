#!/usr/bin/env python3
"""
Complete Lumbar Spine MRI Segmentation Pipeline.
Validates all inputs at every stage before proceeding.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_mha, save_mha, verify_image_mask_alignment, verify_label_integrity,
    generate_data_report, get_class_distribution, create_patient_level_split,
    save_split_info, SPINE_LABEL_MAP, NUM_CLASSES
)
from utils.data_scanner import scan_dataset_directory, validate_data_directory
from preprocessing import PreprocessingPipeline
from datasets import create_dataloaders
from models import create_model, MODEL_REGISTRY
from losses import create_loss_fn
from training import Trainer
from training.hyperopt import run_hyperparameter_search
from evaluation.metrics import SegmentationMetrics
from evaluation.error_analysis import generate_error_report
from visualization.visualizer import (
    plot_learning_curves, plot_confusion_matrix, plot_class_metrics,
    plot_prediction_grid, plot_slice, plot_overlay
)
from inference import Predictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Lumbar Spine MRI Segmentation Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['validate', 'analyze', 'train', 'evaluate', 'infer', 'hyperopt', 'benchmark', 'full'],
                        help='Pipeline mode')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for inference/evaluation')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory')
    parser.add_argument('--model-name', type=str, choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture to use')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    import yaml
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using defaults")
        return _default_config()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def _default_config() -> Dict[str, Any]:
    return {
        'dataset': {'data_dir': './data/raw', 'image_suffix': '_image', 'mask_suffix': '_mask',
                     'use_3d': True, 'cache_slices': True, 'split_ratios': [0.7, 0.15, 0.15], 'seed': 42},
        'preprocessing': {'clip_intensity': True, 'clip_percentiles': [0.5, 99.5],
                          'zscore_normalize': True, 'extract_roi': True, 'roi_margin': 5,
                          'target_size': [128, 128, 128]},
        'augmentation': {'rotation': True, 'rotation_degrees': 10,
                         'elastic_deformation': True, 'gaussian_noise': True},
        'model': {'name': 'atm_net', 'in_channels': 1, 'num_classes': 12,
                  'deep_supervision': True, 'params': {'features': [32, 64, 128, 256, 512]}},
        'training': {'epochs': 100, 'batch_size': 2, 'learning_rate': 1e-4,
                     'weight_decay': 1e-5, 'mixed_precision': True,
                     'early_stopping_patience': 20, 'gradient_clip': 1.0,
                     'num_workers': 0, 'loss': {'name': 'dice_focal', 'weights': [0.5, 0.5]}},
        'inference': {'tta': True},
    }


def validate_data(config: Dict[str, Any]) -> Dict:
    logger.info("=" * 60)
    logger.info("STEP 1: DATA VALIDATION")
    logger.info("=" * 60)

    data_dir = config.get('dataset', {}).get('data_dir', './data/raw')
    logger.info(f"Validating data in: {data_dir}")

    report = validate_data_directory(data_dir)

    logger.info(f"Status: {report['status']}")
    logger.info(f"Total pairs: {report.get('total_pairs', 0)}, "
                f"Passed: {report.get('passed', 0)}, Failed: {report.get('failed', 0)}")
    logger.info(f"Alignment issues: {len(report.get('alignment_issues', []))}")
    logger.info(f"Label issues: {len(report.get('label_issues', []))}")
    logger.info(f"Corrupt files: {len(report.get('corrupt_files', []))}")
    logger.info(f"Message: {report.get('message', '')}")

    if report.get('status') == 'FAILED':
        logger.error("Data validation FAILED. Aborting.")
        raise RuntimeError("Data integrity check failed")

    output_path = Path('reports') / 'data_validation_report.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Validation report saved to {output_path}")

    return report


def analyze_dataset(config: Dict[str, Any]) -> Dict:
    logger.info("=" * 60)
    logger.info("STEP 2: DATASET ANALYSIS")
    logger.info("=" * 60)

    data_dir = config.get('dataset', {}).get('data_dir', './data/raw')
    images, masks = scan_dataset_directory(
        data_dir,
        config.get('dataset', {}).get('image_suffix', '_image'),
        config.get('dataset', {}).get('mask_suffix', '_mask'),
    )

    logger.info(f"Dataset: {len(images)} image-mask pairs")

    class_dist = get_class_distribution(masks)
    logger.info("Class distribution:")
    for label, info in class_dist.items():
        if label == '_total_voxels':
            continue
        logger.info(f"  {info['name']:20s}: {info['percentage']:6.4f}% ({info['count']:>10,} voxels)")

    output_path = Path('reports') / 'class_distribution.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(class_dist, f, indent=2, default=str)
    logger.info(f"Class distribution saved to {output_path}")

    return {'class_distribution': class_dist, 'n_images': len(images), 'n_masks': len(masks)}


def create_split(config: Dict[str, Any]) -> Dict:
    logger.info("=" * 60)
    logger.info("STEP 3: DATA SPLITTING")
    logger.info("=" * 60)

    data_dir = config.get('dataset', {}).get('data_dir', './data/raw')
    images, masks = scan_dataset_directory(
        data_dir,
        config.get('dataset', {}).get('image_suffix', '_image'),
        config.get('dataset', {}).get('mask_suffix', '_mask'),
    )

    ratios = config.get('dataset', {}).get('split_ratios', [0.7, 0.15, 0.15])
    seed = config.get('dataset', {}).get('seed', 42)

    split = create_patient_level_split(images, masks, ratios[0], ratios[1], ratios[2], seed=seed)

    split_path = Path('reports') / 'data_split.json'
    save_split_info(split, split_path)

    return split


def train_model(config: Dict[str, Any], split: Dict, run_dir: Path) -> Dict:
    logger.info("=" * 60)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("=" * 60)

    if args.model_name:
        config['model']['name'] = args.model_name

    device = _get_device(args)
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Loss: {config['training']['loss']['name']}")

    model = create_model(config)
    dataloaders = create_dataloaders(split, config)

    if dataloaders['train'] is None or dataloaders['val'] is None:
        raise RuntimeError("Train or val dataloader is empty")

    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config,
        device=device,
        run_dir=run_dir,
    )

    result = trainer.train()

    logger.info(f"Training complete. Best Dice: {result['best_dice']:.4f} at epoch {result['best_epoch']}")

    return result


def evaluate_model(config: Dict[str, Any], split: Dict, checkpoint_path: Path, run_dir: Path):
    logger.info("=" * 60)
    logger.info("STEP 5: EVALUATION")
    logger.info("=" * 60)

    device = _get_device(args)

    model = create_model(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device).eval()

    predictor = Predictor(model, config, device)
    metrics_calc = SegmentationMetrics(config.get('model', {}).get('num_classes', 12))

    test_pairs = split.get('test', [])
    logger.info(f"Evaluating on {len(test_pairs)} test volumes")

    all_metrics = []
    all_preds = []
    all_targets = []

    for idx, (img_path, mask_path) in enumerate(test_pairs):
        logger.info(f"  [{idx + 1}/{len(test_pairs)}] {img_path.name}")

        pred = predictor.predict_volume(img_path)
        _, target = load_mha(mask_path)

        metrics = metrics_calc(pred, target)
        all_metrics.append(metrics)

        all_preds.append(pred)
        all_targets.append(target)

        mid_slice = target.shape[0] // 2
        vis_dir = run_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        plot_prediction_grid(
            load_mha(img_path)[1], target, pred,
            save_path=vis_dir / f'test_{idx}_grid.png'
        )

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    logger.info(f"Mean Dice: {avg_metrics.get('mean_dice', 0):.4f}")
    logger.info(f"Mean IoU: {avg_metrics.get('mean_iou', 0):.4f}")

    all_preds_np = np.concatenate([p.flatten() for p in all_preds])
    all_targets_np = np.concatenate([t.flatten() for t in all_targets])
    cm = metrics_calc.compute_confusion_matrix(all_preds_np, all_targets_np)

    plot_confusion_matrix(cm, run_dir / 'confusion_matrix.png')

    error_report = generate_error_report(all_preds_np, all_targets_np)
    logger.info(f"Critical issues: {error_report.get('num_critical', 0)}")

    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(avg_metrics, f, indent=2)

    with open(run_dir / 'error_analysis.json', 'w') as f:
        json.dump(error_report, f, indent=2, default=str)

    logger.info(f"Evaluation complete. Results saved to {run_dir}")

    return avg_metrics


def infer(config: Dict[str, Any], checkpoint_path: Path, input_path: str, output_dir: str):
    logger.info("=" * 60)
    logger.info("STEP 6: INFERENCE")
    logger.info("=" * 60)

    device = _get_device(args)
    model = create_model(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    predictor = Predictor(model, config, device)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        logger.info(f"Predicting single file: {input_path}")
        mask, probs = predictor.predict_volume(input_path, output_dir / f"{input_path.stem}_pred.mha",
                                                return_probabilities=True)

        sitk_img, img_arr = load_mha(input_path)
        mid = img_arr.shape[0] // 2
        plot_slice(img_arr, mask, slice_idx=mid, save_path=output_dir / f"{input_path.stem}_vis.png")

    elif input_path.is_dir():
        logger.info(f"Predicting folder: {input_path}")
        saved = predictor.predict_folder(input_path, output_dir)
        logger.info(f"Saved {len(saved)} predictions to {output_dir}")

    logger.info("Inference complete")


def benchmark_models(config: Dict[str, Any], split: Dict, run_dir: Path):
    logger.info("=" * 60)
    logger.info("STEP 7: MODEL BENCHMARKING")
    logger.info("=" * 60)

    device = _get_device(args)
    results = {}

    for model_name in MODEL_REGISTRY.keys():
        logger.info(f"\nBenchmarking: {model_name}")
        config['model']['name'] = model_name
        config['model']['deep_supervision'] = True

        model_run_dir = run_dir / model_name
        model_run_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = create_model(config)
            dataloaders = create_dataloaders(split, config)
            config['training']['epochs'] = 50
            config['training']['early_stopping_patience'] = 10

            trainer = Trainer(
                model=model,
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                config=config,
                device=device,
                run_dir=model_run_dir,
            )

            result = trainer.train()
            results[model_name] = {
                'best_dice': result['best_dice'],
                'best_epoch': result['best_epoch'],
                'total_epochs': result['total_epochs'],
            }

            best_ckpt = model_run_dir / 'checkpoints' / 'best.pt'
            if best_ckpt.exists():
                eval_model = create_model(config)
                state = torch.load(best_ckpt, map_location=device)
                eval_model.load_state_dict(state['model_state_dict'])
                eval_model = eval_model.to(device).eval()
                predictor = Predictor(eval_model, config, device)
                metrics_calc = SegmentationMetrics(config.get('model', {}).get('num_classes', 12))

                test_pairs = split.get('test', [])
                test_metrics = []
                for img_p, msk_p in test_pairs:
                    pred = predictor.predict_volume(img_p)
                    _, target = load_mha(msk_p)
                    test_metrics.append(metrics_calc(pred, target))

                avg_dice = float(np.mean([m.get('mean_dice', 0) for m in test_metrics]))
                results[model_name]['test_dice'] = avg_dice
                logger.info(f"  Test Dice: {avg_dice:.4f}")

        except Exception as e:
            logger.error(f"  Benchmark failed for {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    with open(run_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\nBenchmark Results:")
    for model_name, r in sorted(results.items(), key=lambda x: x[1].get('test_dice', 0), reverse=True):
        logger.info(f"  {model_name:20s}: {r.get('test_dice', 0):.4f} (best val: {r.get('best_dice', 0):.4f})")

    return results


def _get_device(args):
    if args.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(args.device)


def main():
    global args
    args = parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config['dataset']['data_dir'] = args.data_dir

    run_dir = Path(args.output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'validate':
        validate_data(config)

    elif args.mode == 'analyze':
        validate_data(config)
        analyze_dataset(config)

    elif args.mode == 'train':
        validate_data(config)
        analyze_dataset(config)
        split = create_split(config)
        result = train_model(config, split, run_dir)

    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("--checkpoint required for evaluate mode")
        split = create_split(config)
        evaluate_model(config, split, Path(args.checkpoint), run_dir)

    elif args.mode == 'infer':
        if not args.checkpoint:
            raise ValueError("--checkpoint required for infer mode")
        infer(config, Path(args.checkpoint), args.data_dir, str(run_dir / 'predictions'))

    elif args.mode == 'hyperopt':
        validate_data(config)
        analyze_dataset(config)
        split = create_split(config)
        device = _get_device(args)
        result = run_hyperparameter_search(config, split, device, run_dir, n_trials=20, num_epochs=30)
        logger.info(f"Best hyperparameters: {result['best_params']}")

    elif args.mode == 'benchmark':
        validate_data(config)
        analyze_dataset(config)
        split = create_split(config)
        benchmark_models(config, split, run_dir)

    elif args.mode == 'full':
        validate_data(config)
        analyze_dataset(config)
        split = create_split(config)
        result = train_model(config, split, run_dir)

        best_ckpt = run_dir / 'checkpoints' / 'best.pt'
        if best_ckpt.exists():
            evaluate_model(config, split, best_ckpt, run_dir)

    logger.info(f"All done. Results in {run_dir}")


if __name__ == '__main__':
    main()
