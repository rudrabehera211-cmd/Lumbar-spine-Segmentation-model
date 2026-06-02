import optuna
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import logging
import json

from models import create_model
from losses import create_loss_fn
from datasets import create_dataloaders
from .trainer import Trainer

logger = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [1, 2, 4]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 2, 10),
        'min_lr': trial.suggest_float('min_lr', 1e-8, 1e-6, log=True),
        'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 5.0),
        'loss_name': trial.suggest_categorical('loss_name', ['dice', 'dice_focal', 'dice_ce', 'dice_tversky']),
        'rotation_degrees': trial.suggest_float('rotation_degrees', 5, 20),
        'elastic_alpha': trial.suggest_float('elastic_alpha', 10, 40),
        'noise_std': trial.suggest_float('noise_std', 0.001, 0.05, log=True),
    }
    return params


def objective(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    split_data: Dict[str, List],
    device: torch.device,
    run_dir: Path,
    num_epochs: int = 30,
) -> float:
    params = suggest_params(trial)

    config = base_config.copy()
    config['training']['learning_rate'] = params['learning_rate']
    config['training']['batch_size'] = params['batch_size']
    config['training']['weight_decay'] = params['weight_decay']
    config['training']['warmup_epochs'] = params['warmup_epochs']
    config['training']['min_lr'] = params['min_lr']
    config['training']['gradient_clip'] = params['gradient_clip']
    config['training']['loss'] = {'name': params['loss_name'], 'weights': [0.5, 0.5]}
    config['augmentation']['rotation_degrees'] = params['rotation_degrees']
    config['augmentation']['elastic_alpha'] = params['elastic_alpha']
    config['augmentation']['noise_std'] = params['noise_std']

    config['training']['epochs'] = num_epochs
    config['training']['early_stopping_patience'] = num_epochs // 2

    trial_dir = run_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = create_model(config)
        dataloaders = create_dataloaders(split_data, config)

        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            config=config,
            device=device,
            run_dir=trial_dir,
        )

        result = trainer.train()
        best_dice = result['best_dice']

        with open(trial_dir / 'result.json', 'w') as f:
            json.dump({'best_dice': best_dice, 'params': params}, f, indent=2)

        return best_dice

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_hyperparameter_search(
    base_config: Dict[str, Any],
    split_data: Dict[str, List],
    device: torch.device,
    run_dir: Path,
    n_trials: int = 20,
    num_epochs: int = 30,
) -> Dict[str, Any]:
    study = optuna.create_study(
        direction='maximize',
        study_name='spine_segmentation',
        storage=None,
    )

    study.optimize(
        lambda trial: objective(trial, base_config, split_data, device, run_dir, num_epochs),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'study_name': study.study_name,
    }

    with open(run_dir / 'hyperopt_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Hyperparameter search completed. Best Dice: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return results
