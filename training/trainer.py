import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, Any, Optional, Callable
import sys

from losses import create_loss_fn, DiceLoss
from evaluation.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        for i, (param_group, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            if self.current_step <= self.warmup_steps:
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            param_group['lr'] = lr
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        run_dir: Path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.train_config = config.get('training', {})
        self.epochs = self.train_config.get('epochs', 100)
        self.accumulation_steps = self.train_config.get('gradient_accumulation', 1)
        self.clip_grad = self.train_config.get('gradient_clip', 1.0)
        self.mixed_precision = self.train_config.get('mixed_precision', True)
        self.early_stopping_patience = self.train_config.get('early_stopping_patience', 20)

        self.loss_fn = create_loss_fn(config)

        lr = self.train_config.get('learning_rate', 1e-4)
        weight_decay = self.train_config.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = len(train_loader) * self.epochs // self.accumulation_steps
        warmup_steps = self.train_config.get('warmup_epochs', 5) * len(train_loader) // self.accumulation_steps
        self.scheduler = WarmupCosineScheduler(self.optimizer, warmup_steps, total_steps,
                                                min_lr=self.train_config.get('min_lr', 1e-7))

        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.metrics_calculator = SegmentationMetrics(config.get('model', {}).get('num_classes', 12))

        self.best_dice = 0.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'learning_rates': []}
        self.global_step = 0

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            image = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)

            with autocast(enabled=self.mixed_precision):
                pred = self.model(image)
                loss = self.loss_fn(pred, mask)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            image = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)

            with autocast(enabled=self.mixed_precision):
                pred = self.model(image)
                if isinstance(pred, list):
                    pred = pred[0]
                loss = self.loss_fn(pred, mask)

            total_loss += loss.item()
            num_batches += 1

            pred_np = torch.argmax(pred, dim=1).cpu().numpy()
            mask_np = mask.cpu().numpy()
            all_preds.append(pred_np)
            all_targets.append(mask_np)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = self.metrics_calculator(all_preds, all_targets)
        metrics['val_loss'] = total_loss / max(num_batches, 1)

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.__dict__,
            'scaler_state': self.scaler.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config,
        }

        path = self.checkpoint_dir / f'epoch_{epoch:04d}.pt'
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(state, best_path)
            logger.info(f"New best model: Dice={self.best_dice:.4f}")

        # Keep only last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('epoch_*.pt'))
        for cp in checkpoints[:-5]:
            cp.unlink(missing_ok=True)

    def load_checkpoint(self, checkpoint_path: Path):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.best_dice = state.get('best_dice', 0.0)
        self.history = state.get('history', self.history)
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {state.get('epoch', '?')})")
        return state.get('epoch', 0)

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting training for {self.epochs} epochs on {self.device}")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            metrics = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(metrics['val_loss'])
            self.history['val_dice'].append(metrics.get('mean_dice', 0.0))
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            val_dice = metrics.get('mean_dice', 0.0)
            is_best = val_dice > self.best_dice

            if is_best:
                self.best_dice = val_dice
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Loss: {metrics['val_loss']:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Best: {self.best_dice:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")

        self._save_history()
        return {
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch,
            'total_time': total_time,
            'history': self.history,
        }

    def _save_history(self):
        path = self.run_dir / 'training_history.json'
        serializable = {}
        for k, v in self.history.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                serializable[k] = v
            else:
                serializable[k] = str(v)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
