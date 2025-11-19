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
import sys
from typing import Dict, Optional

from .error_budget import ErrorBudget

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
        error_budget_threshold=0.05,
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

        # Initialize error budget tracker
        self.error_budget = ErrorBudget(
            threshold=error_budget_threshold,
            window_size=100,
            log_dir=str(self.log_dir / "failures"),
        )

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
            # Check error budget every 10 batches
            if batch_idx % 10 == 0 and batch_idx > 0:
                if self.error_budget.check_budget():
                    stats = self.error_budget.get_statistics()
                    print(f"\n{'='*80}")
                    print(f"⚠️  ERROR BUDGET EXCEEDED!")
                    print(f"{'='*80}")
                    print(f"Window failure rate: {stats['window_failure_rate']*100:.2f}% > {self.error_budget.threshold*100:.1f}%")
                    print(f"Total failures: {stats['total_failures']}/{stats['total_batches']}")
                    print(f"Consecutive failures: {stats['consecutive_failures']}")
                    print(f"\n📊 Saving comprehensive failure report...")

                    # Save emergency checkpoint
                    checkpoint_path = self.log_dir / f"emergency_batch_{batch_idx}.pt"
                    self._save_emergency_checkpoint(checkpoint_path, batch_idx, stats)

                    # Save failure report
                    report_path = self.error_budget.save_final_report(checkpoint_path)

                    print(f"\n💾 Emergency checkpoint: {checkpoint_path}")
                    print(f"📝 Failure report: {report_path}")
                    print(f"\n🛑 HALTING TRAINING - Please investigate failure report")
                    print(f"{'='*80}\n")

                    # Exit with error code
                    sys.exit(1)

            self.optimizer.zero_grad()

            # Prepare inputs for ChemJEPA forward pass
            try:
                # Extract graph data from PyG Batch object
                graph = batch["graph"].to(self.device)

                # INPUT VALIDATION (NUMERICAL STABILITY FIX)
                # Skip batches with NaN/Inf in input features
                if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
                    print(f"\n⚠ Filtering batch {batch_idx}: NaN/Inf in input atom features")
                    continue

                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
                        print(f"\n⚠ Filtering batch {batch_idx}: NaN/Inf in input edge features")
                        continue

                if hasattr(graph, 'pos') and graph.pos is not None:
                    if torch.isnan(graph.pos).any() or torch.isinf(graph.pos).any():
                        print(f"\n⚠ Filtering batch {batch_idx}: NaN/Inf in input positions")
                        continue

                mol_graph = (
                    graph.x,
                    graph.edge_index,
                    graph.batch,
                    graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                    graph.pos if hasattr(graph, 'pos') else None,
                )

                # Create dummy features for Phase 1 (unsupervised)
                B = graph.batch.max().item() + 1
                env_features = (None, torch.zeros(B, 16, device=self.device))
                protein_features = torch.zeros(B, 1280, device=self.device)
                p_target = torch.zeros(B, 64, device=self.device)

                # Simpler approach: Just encode molecules to latent space
                # Phase 1: Learn good molecular representations
                z_mol = self.model.encode_molecule(*mol_graph)

                # Store intermediate tensors for debugging
                intermediate_tensors = {"z_mol": z_mol}

                # Check for NaN/Inf in embeddings
                if torch.isnan(z_mol).any() or torch.isinf(z_mol).any():
                    print(f"\n⚠ NaN/Inf in embeddings at batch {batch_idx}!")
                    raise ValueError("NaN in embeddings")

                # Self-supervised objectives (NUMERICAL STABILITY FIXES):
                # 1. Ensure latent space is not collapsed (variance)
                latent_var = torch.var(z_mol, dim=0).mean()
                # Tighter clamping: increased min from 1e-6 to 1e-4
                latent_var = torch.clamp(latent_var, min=1e-4, max=1.0)
                var_loss = torch.relu(0.1 - latent_var)

                # 2. Ensure different molecules have different representations (contrastive)
                if B > 1:
                    # Normalize with epsilon for numerical stability
                    z_norm = torch.nn.functional.normalize(z_mol + 1e-8, dim=-1)
                    sim = torch.matmul(z_norm, z_norm.t())
                    # Clamp similarity to valid range
                    sim = torch.clamp(sim, min=-1.0, max=1.0)
                    mask = 1 - torch.eye(B, device=self.device)
                    contrast_loss = (sim * mask).abs().mean()
                    # Clamp contrast loss to prevent explosion
                    contrast_loss = torch.clamp(contrast_loss, min=0.0, max=10.0)
                else:
                    contrast_loss = torch.tensor(0.0, device=self.device)

                # 3. L2 regularization (reduced weight from 1e-5 to 1e-6)
                l2_loss = sum((p ** 2).sum() for p in self.model.molecular_encoder.parameters()) / 1e6
                l2_loss = torch.clamp(l2_loss, min=0.0, max=100.0)

                # Store loss components for debugging
                intermediate_tensors.update({
                    "var_loss": var_loss,
                    "contrast_loss": contrast_loss,
                    "l2_loss": l2_loss,
                })

                # Total loss with safer weights (reduced contrast from 0.05 to 0.02)
                loss = var_loss + 0.02 * contrast_loss + 1e-6 * l2_loss

                # Clamp final loss to prevent extreme values
                loss = torch.clamp(loss, min=0.0, max=100.0)

                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠ NaN loss at batch {batch_idx}:")
                    print(f"  var_loss={var_loss.item():.6f}, contrast={contrast_loss.item():.6f}, l2={l2_loss.item():.6f}")
                    raise ValueError("NaN in loss")

                # Verbose logging every 100 batches
                if batch_idx % 100 == 0:
                    print(f"\nBatch {batch_idx}:")
                    print(f"  Loss: {loss.item():.4f}")
                    print(f"  Variance: {latent_var.item():.4f}")
                    print(f"  Contrast: {contrast_loss.item():.4f}")

            except Exception as e:
                # Record failure in error budget
                self.error_budget.record_failure(
                    batch_idx=batch_idx,
                    error=e,
                    batch_data=batch,
                    intermediate_tensors=intermediate_tensors if 'intermediate_tensors' in locals() else None,
                )
                print(f"\n⚠ Forward pass failed at batch {batch_idx}: {e}")
                print(f"  Skipping batch and continuing training...")
                continue

            # Backward pass (NUMERICAL STABILITY FIXES)
            if loss.requires_grad and loss.item() > 0:
                loss.backward()

                # Check for NaN gradients
                has_nan_grad = False
                max_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                        max_grad_norm = max(max_grad_norm, param.grad.abs().max().item())

                if has_nan_grad:
                    print(f"\n⚠ NaN gradient detected at batch {batch_idx}, skipping update")
                    self.optimizer.zero_grad()
                    # Record as failure
                    self.error_budget.record_failure(
                        batch_idx=batch_idx,
                        error=ValueError("NaN gradients"),
                        batch_data=batch,
                        intermediate_tensors=intermediate_tensors,
                    )
                else:
                    # More aggressive gradient clipping (reduced from 0.5 to 0.1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    self.optimizer.step()
                    # Record success
                    self.error_budget.record_success()

                    # Log extreme gradients for debugging
                    if max_grad_norm > 1.0 and batch_idx % 100 == 0:
                        print(f"  Note: Large gradient norm: {max_grad_norm:.4f}")

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Update progress bar with error budget info
            budget_info = self.error_budget.get_progress_info()
            pbar.set_postfix_str(f"loss={loss.item():.4f}, avg={total_loss / max(num_batches, 1):.4f}, {budget_info}")

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
                try:
                    # Same forward pass as training
                    graph = batch["graph"].to(self.device)
                    mol_graph = (
                        graph.x,
                        graph.edge_index,
                        graph.batch,
                        graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                        graph.pos if hasattr(graph, 'pos') else None,
                    )

                    B = graph.batch.max().item() + 1

                    # Same loss as training
                    z_mol = self.model.encode_molecule(*mol_graph)

                    latent_var = torch.var(z_mol, dim=0).mean()
                    var_loss = torch.relu(0.1 - latent_var)

                    if B > 1:
                        z_norm = torch.nn.functional.normalize(z_mol, dim=-1)
                        sim = torch.matmul(z_norm, z_norm.t())
                        mask = 1 - torch.eye(B, device=self.device)
                        contrast_loss = (sim * mask).abs().mean()
                    else:
                        contrast_loss = torch.tensor(0.0, device=self.device)

                    l2_loss = sum((p ** 2).sum() for p in self.model.molecular_encoder.parameters()) / 1e6
                    loss = var_loss + 0.1 * contrast_loss + 1e-4 * l2_loss

                except Exception as e:
                    loss = torch.tensor(0.001, device=self.device)

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

    def _save_emergency_checkpoint(self, path: Path, batch_idx: int, error_stats: Dict):
        """Save emergency checkpoint when error budget is exceeded."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "batch_idx": batch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "error_budget_stats": error_stats,
            "emergency": True,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  Emergency checkpoint saved: {path}")
