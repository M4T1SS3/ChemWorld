"""
Training loop for ChemJEPA.

Handles:
- Multi-phase training (JEPA, property, planning)
- Logging and checkpointing
- Validation
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ChemJEPATrainer:
    """
    Trainer for ChemJEPA model.

    Args:
        model: ChemJEPA model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        log_dir: Directory for logs
        use_wandb: Whether to use Weights & Biases logging
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        log_dir='logs',
        use_wandb=False,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(
        self,
        train_loader: DataLoader,
        phase: str = "jepa",
        log_every: int = 100,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            phase: Training phase ("jepa", "property", "planning")
            log_every: Log every N steps

        Returns:
            Dictionary of average losses
        """
        self.model.train()

        total_loss = 0.0
        loss_components = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch} [{phase}]")

        for batch_idx, batch in enumerate(pbar):
            # For dummy dataset, we'll create a simple training loop
            # In real training, this would use actual batch data

            # TODO: Implement proper batch processing based on phase
            # For now, just demonstrate the structure

            self.optimizer.zero_grad()

            # Forward pass would go here
            # outputs = self.model(...)

            # Loss computation would go here
            # losses = self.criterion(outputs, targets, phase=phase)

            # For now, skip actual training
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)

            # Backward pass
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "avg_loss": total_loss / num_batches,
            })

            # Log
            if self.step % log_every == 0:
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/step": self.step,
                        "train/epoch": self.epoch,
                    })

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            **loss_components,
        }

    def validate(
        self,
        val_loader: DataLoader,
        phase: str = "jepa",
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            phase: Training phase

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # TODO: Implement validation
                loss = torch.tensor(0.0, device=self.device)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return {"val_loss": avg_loss}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        phase: str = "jepa",
        checkpoint_dir: str = "checkpoints",
        save_every: int = 10,
        log_every: int = 100,
    ):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            phase: Training phase
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            log_every: Log every N steps
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Starting training - Phase: {phase}")
        print(f"{'='*80}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"{'='*80}\n")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            start_time = time.time()
            train_metrics = self.train_epoch(
                train_loader,
                phase=phase,
                log_every=log_every,
            )
            train_time = time.time() - start_time

            # Validate
            val_metrics = self.validate(val_loader, phase=phase)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val loss: {val_metrics['val_loss']:.4f}")
            print(f"  Time: {train_time:.1f}s")

            # Log
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics['loss'],
                    "val/loss": val_metrics['val_loss'],
                    "time/epoch": train_time,
                })

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt",
                    epoch=epoch,
                    val_loss=val_metrics['val_loss'],
                )

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(
                    checkpoint_dir / f"best_{phase}.pt",
                    epoch=epoch,
                    val_loss=val_metrics['val_loss'],
                )
                print(f"  ✓ Saved best model (val_loss: {val_metrics['val_loss']:.4f})")

        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"{'='*80}")
        print(f"Best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        print(f"Loaded checkpoint from epoch {self.epoch}")
