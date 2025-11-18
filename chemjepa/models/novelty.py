"""
Open-World Novelty Detection Module

Three complementary mechanisms for uncertainty quantification:
1. Ensemble disagreement (epistemic uncertainty)
2. Latent density estimation (OOD detection)
3. Conformal prediction (calibrated prediction sets)

Enables the model to "say I don't know" when appropriate.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from .latent import LatentState


class NormalizingFlow(nn.Module):
    """
    Normalizing flow for density estimation in latent space.

    Uses affine coupling layers (RealNVP-style) to learn q(z).
    """

    def __init__(self, dim: int, num_layers: int = 6, hidden_dim: int = 512):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers

        # Coupling layers
        self.coupling_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # Alternate which dimensions are transformed
            mask = torch.zeros(dim)
            mask[:(dim // 2)] = 1
            if i % 2 == 1:
                mask = 1 - mask

            self.coupling_layers.append(
                AffineCouplingLayer(dim, mask, hidden_dim)
            )
            self.batch_norms.append(nn.BatchNorm1d(dim))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (z -> x).

        Args:
            z: Latent samples [B, dim]

        Returns:
            x: Base distribution samples [B, dim]
            log_det: Log determinant of Jacobian [B]
        """
        log_det = torch.zeros(z.size(0), device=z.device)
        x = z

        for coupling, bn in zip(self.coupling_layers, self.batch_norms):
            x, ld = coupling.forward(x)
            log_det = log_det + ld
            x = bn(x)

        return x, log_det

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass (x -> z).

        Args:
            x: Base distribution samples [B, dim]

        Returns:
            z: Latent samples [B, dim]
            log_det: Log determinant of Jacobian [B]
        """
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x

        for coupling, bn in reversed(list(zip(self.coupling_layers, self.batch_norms))):
            z = bn(z)
            z, ld = coupling.inverse(z)
            log_det = log_det + ld

        return z, log_det

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of latent samples.

        Args:
            z: Latent samples [B, dim]

        Returns:
            log_prob: Log probability [B]
        """
        x, log_det = self.forward(z)

        # Base distribution (standard normal)
        log_prob_base = -0.5 * (x ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        return log_prob_base + log_det


class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer for normalizing flow."""

    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int):
        super().__init__()

        self.register_buffer("mask", mask)

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),  # Bounded scale
        )

        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation."""
        z_masked = z * self.mask

        scale = self.scale_net(z_masked) * (1 - self.mask)
        translation = self.translate_net(z_masked) * (1 - self.mask)

        x = z_masked + (1 - self.mask) * (z * torch.exp(scale) + translation)
        log_det = scale.sum(dim=-1)

        return x, log_det

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transformation."""
        x_masked = x * self.mask

        scale = self.scale_net(x_masked) * (1 - self.mask)
        translation = self.translate_net(x_masked) * (1 - self.mask)

        z = x_masked + (1 - self.mask) * ((x - translation) * torch.exp(-scale))
        log_det = -scale.sum(dim=-1)

        return z, log_det


class ConformalPredictor:
    """
    Conformal prediction for calibrated uncertainty sets.

    Provides prediction sets with guaranteed coverage (e.g., 90%).
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage level (1 - alpha = coverage level)
                   e.g., alpha=0.1 gives 90% coverage
        """
        self.alpha = alpha
        self.calibration_scores = None

    def calibrate(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Calibrate on validation set.

        Args:
            predictions: Model predictions [N, ...]
            targets: Ground truth targets [N, ...]
        """
        # Compute conformity scores (e.g., absolute error)
        scores = np.abs(predictions - targets)

        if len(scores.shape) > 1:
            scores = scores.mean(axis=tuple(range(1, len(scores.shape))))

        self.calibration_scores = scores

    def predict_set(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute prediction set for new prediction.

        Args:
            prediction: Model prediction [..., dim]

        Returns:
            lower_bound: Lower bound of prediction set
            upper_bound: Upper bound of prediction set
            radius: Radius of prediction set
        """
        if self.calibration_scores is None:
            raise ValueError("Must calibrate before predicting")

        # Compute quantile
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        radius = np.quantile(self.calibration_scores, q_level)

        lower_bound = prediction - radius
        upper_bound = prediction + radius

        return lower_bound, upper_bound, float(radius)


class NoveltyDetector(nn.Module):
    """
    Open-world novelty detection with three mechanisms.

    1. Ensemble disagreement: Train multiple models, measure std
    2. Density estimation: Learn p(z) via normalizing flow
    3. Conformal prediction: Calibrated prediction sets

    Args:
        mol_dim: Molecular embedding dimension (default: 768)
        rxn_dim: Reaction state dimension (default: 384)
        context_dim: Context dimension (default: 256)
        num_flow_layers: Number of normalizing flow layers (default: 6)
        ensemble_size: Number of models in ensemble (default: 5)
    """

    def __init__(
        self,
        mol_dim: int = 768,
        rxn_dim: int = 384,
        context_dim: int = 256,
        num_flow_layers: int = 6,
        ensemble_size: int = 5,
    ):
        super().__init__()

        self.mol_dim = mol_dim
        self.rxn_dim = rxn_dim
        self.context_dim = context_dim
        self.ensemble_size = ensemble_size

        total_dim = mol_dim + rxn_dim + context_dim

        # Normalizing flow for density estimation
        self.flow = NormalizingFlow(
            dim=total_dim,
            num_layers=num_flow_layers,
        )

        # Conformal predictor (calibrated at runtime)
        self.conformal = ConformalPredictor(alpha=0.1)  # 90% coverage

        # Thresholds (learned or set via validation)
        self.register_buffer("density_threshold", torch.tensor(-10.0))  # log_prob threshold
        self.register_buffer("ensemble_threshold", torch.tensor(0.5))  # std threshold

    def compute_density_score(self, latent_state: LatentState) -> torch.Tensor:
        """
        Compute latent density score via normalizing flow.

        Args:
            latent_state: Hierarchical latent state

        Returns:
            log_prob: Log probability under learned distribution [B]
        """
        z = latent_state.concatenate()
        log_prob = self.flow.log_prob(z)
        return log_prob

    def compute_ensemble_disagreement(
        self,
        predictions: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ensemble disagreement (epistemic uncertainty).

        Args:
            predictions: List of predictions from ensemble models [ensemble_size, B, ...]

        Returns:
            mean: Mean prediction [B, ...]
            std: Standard deviation [B, ...]
        """
        predictions_stack = torch.stack(predictions, dim=0)  # [ensemble_size, B, ...]

        mean = predictions_stack.mean(dim=0)
        std = predictions_stack.std(dim=0)

        return mean, std

    def is_novel(
        self,
        latent_state: LatentState,
        ensemble_predictions: Optional[List[torch.Tensor]] = None,
        density_threshold: Optional[float] = None,
        ensemble_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Determine if latent state is novel (out-of-distribution).

        Args:
            latent_state: Hierarchical latent state
            ensemble_predictions: List of predictions from ensemble (optional)
            density_threshold: Custom threshold for density (optional)
            ensemble_threshold: Custom threshold for ensemble disagreement (optional)

        Returns:
            Dictionary with:
                - is_novel: Boolean flag [B]
                - density_score: Log probability [B]
                - ensemble_std: Ensemble disagreement [B] (if provided)
                - flags: Individual flags for each mechanism
        """
        B = latent_state.z_mol.shape[0]

        # Density-based detection
        density_score = self.compute_density_score(latent_state)

        threshold_density = density_threshold if density_threshold is not None else self.density_threshold
        is_novel_density = density_score < threshold_density

        # Ensemble-based detection
        if ensemble_predictions is not None:
            _, ensemble_std = self.compute_ensemble_disagreement(ensemble_predictions)

            # Average std across dimensions
            if len(ensemble_std.shape) > 1:
                ensemble_std_scalar = ensemble_std.mean(dim=tuple(range(1, len(ensemble_std.shape))))
            else:
                ensemble_std_scalar = ensemble_std

            threshold_ensemble = ensemble_threshold if ensemble_threshold is not None else self.ensemble_threshold
            is_novel_ensemble = ensemble_std_scalar > threshold_ensemble
        else:
            ensemble_std_scalar = None
            is_novel_ensemble = torch.zeros(B, dtype=torch.bool, device=latent_state.z_mol.device)

        # Combine flags (novel if ANY mechanism flags it)
        is_novel = is_novel_density | is_novel_ensemble

        output = {
            "is_novel": is_novel,
            "density_score": density_score,
            "is_novel_density": is_novel_density,
            "is_novel_ensemble": is_novel_ensemble,
        }

        if ensemble_std_scalar is not None:
            output["ensemble_std"] = ensemble_std_scalar

        return output

    def calibrate_conformal(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Calibrate conformal predictor on validation set.

        Args:
            predictions: Model predictions [N, dim]
            targets: Ground truth [N, dim]
        """
        self.conformal.calibrate(predictions, targets)

    def get_conformal_set(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get conformal prediction set.

        Args:
            prediction: Model prediction [dim]

        Returns:
            lower_bound, upper_bound, radius
        """
        return self.conformal.predict_set(prediction)

    def set_thresholds(self, density_threshold: float, ensemble_threshold: float):
        """
        Set novelty detection thresholds (typically from validation set).

        Args:
            density_threshold: Log probability threshold for density
            ensemble_threshold: Std threshold for ensemble disagreement
        """
        self.density_threshold.fill_(density_threshold)
        self.ensemble_threshold.fill_(ensemble_threshold)

    def train_density_model(
        self,
        latent_states: List[LatentState],
        num_epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-4,
    ) -> List[float]:
        """
        Train normalizing flow on latent states.

        Args:
            latent_states: List of training latent states
            num_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            losses: Training losses per epoch
        """
        # Concatenate all states
        z_all = torch.stack([state.concatenate() for state in latent_states], dim=0)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(z_all)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)

        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for (batch_z,) in loader:
                optimizer.zero_grad()

                # Negative log likelihood
                log_prob = self.flow.log_prob(batch_z)
                loss = -log_prob.mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        return losses
