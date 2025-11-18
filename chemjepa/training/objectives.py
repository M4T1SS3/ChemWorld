"""
Training Objectives for ChemJEPA

Multi-phase training:
1. Phase 1: JEPA pre-training (latent prediction + energy contrastive)
2. Phase 2: Property prediction fine-tuning
3. Phase 3: Planning & RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ChemJEPALoss(nn.Module):
    """
    Combined loss function for ChemJEPA training.

    Supports multiple training phases:
    - JEPA prediction loss
    - Energy contrastive loss
    - Consistency regularization
    - Property prediction loss
    - VAE regularization
    - Vector quantization loss
    """

    def __init__(
        self,
        lambda_jepa: float = 1.0,
        lambda_energy: float = 1.0,
        lambda_consistency: float = 0.1,
        lambda_property: float = 1.0,
        lambda_kl: float = 0.01,
        lambda_vq: float = 0.1,
        energy_margin: float = 1.0,
        consistency_temperature: float = 0.1,
    ):
        """
        Args:
            lambda_jepa: Weight for JEPA prediction loss
            lambda_energy: Weight for energy contrastive loss
            lambda_consistency: Weight for consistency regularization
            lambda_property: Weight for property prediction loss
            lambda_kl: Weight for KL divergence (VAE)
            lambda_vq: Weight for vector quantization loss
            energy_margin: Margin for contrastive energy loss
            consistency_temperature: Temperature for consistency loss
        """
        super().__init__()

        self.lambda_jepa = lambda_jepa
        self.lambda_energy = lambda_energy
        self.lambda_consistency = lambda_consistency
        self.lambda_property = lambda_property
        self.lambda_kl = lambda_kl
        self.lambda_vq = lambda_vq

        self.energy_margin = energy_margin
        self.consistency_temperature = consistency_temperature

    def jepa_prediction_loss(
        self,
        predicted_state,
        target_state,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        JEPA-style latent prediction loss.

        Predict future latent state without reconstructing molecules.

        Args:
            predicted_state: Predicted next state
            target_state: Ground truth next state
            uncertainty: Predicted uncertainty (optional)

        Returns:
            Loss value
        """
        pred_vec = predicted_state.concatenate()
        target_vec = target_state.concatenate()

        mse = (pred_vec - target_vec) ** 2

        if uncertainty is not None:
            # Heteroscedastic loss (NLL under Gaussian)
            nll = 0.5 * (mse / (uncertainty ** 2 + 1e-6) + torch.log(uncertainty ** 2 + 1e-6))
            loss = nll.mean()
        else:
            loss = mse.mean()

        return loss

    def energy_contrastive_loss(
        self,
        energy_pos: torch.Tensor,
        energy_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss for energy model.

        Positive pairs should have low energy, negatives high energy.

        Args:
            energy_pos: Energy for positive (compatible) pairs [B]
            energy_neg: Energy for negative (incompatible) pairs [B]

        Returns:
            Loss value
        """
        # Positive energies should be low (near 0)
        loss_pos = torch.mean((energy_pos - 0) ** 2)

        # Negative energies should be high (above margin)
        loss_neg = torch.mean(torch.relu(self.energy_margin - energy_neg) ** 2)

        return loss_pos + loss_neg

    def consistency_regularization(
        self,
        state1,
        state2,
    ) -> torch.Tensor:
        """
        Consistency regularization: similar molecules should have similar latents.

        Uses InfoNCE-style contrastive loss.

        Args:
            state1: Original states
            state2: Augmented states (e.g., with small perturbations)

        Returns:
            Loss value
        """
        z1 = state1.concatenate()
        z2 = state2.concatenate()

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.t()) / self.consistency_temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(z1.size(0), device=z1.device)

        # InfoNCE loss
        loss = F.cross_entropy(similarity, labels)

        return loss

    def property_prediction_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-task property prediction loss.

        Args:
            predictions: Predicted properties [B, num_properties]
            targets: Ground truth properties [B, num_properties]
            mask: Mask for which properties are available [B, num_properties]

        Returns:
            Loss value
        """
        mse = (predictions - targets) ** 2

        if mask is not None:
            mse = mse * mask
            loss = mse.sum() / (mask.sum() + 1e-6)
        else:
            loss = mse.mean()

        return loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        phase: str = "jepa",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss based on training phase.

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            phase: Training phase ("jepa", "property", "planning")

        Returns:
            Dictionary with loss components
        """
        losses = {}
        total_loss = 0.0

        # Phase 1: JEPA Pre-training
        if phase == "jepa":
            # JEPA prediction loss
            if "next_state" in outputs and "target_state" in targets:
                loss_jepa = self.jepa_prediction_loss(
                    outputs["next_state"],
                    targets["target_state"],
                    outputs.get("uncertainty", None),
                )
                losses["jepa"] = loss_jepa
                total_loss = total_loss + self.lambda_jepa * loss_jepa

            # Energy contrastive loss
            if "energy_pos" in outputs and "energy_neg" in outputs:
                loss_energy = self.energy_contrastive_loss(
                    outputs["energy_pos"],
                    outputs["energy_neg"],
                )
                losses["energy"] = loss_energy
                total_loss = total_loss + self.lambda_energy * loss_energy

            # Consistency regularization
            if "state_orig" in outputs and "state_aug" in outputs:
                loss_consistency = self.consistency_regularization(
                    outputs["state_orig"],
                    outputs["state_aug"],
                )
                losses["consistency"] = loss_consistency
                total_loss = total_loss + self.lambda_consistency * loss_consistency

            # VAE regularization
            if "vae_params" in outputs:
                from ..models.latent import HierarchicalLatentWorldState
                dummy_model = HierarchicalLatentWorldState()
                loss_kl = dummy_model.compute_kl_loss(outputs["vae_params"])
                losses["kl"] = loss_kl
                total_loss = total_loss + self.lambda_kl * loss_kl

            # VQ loss
            if "vq_loss" in outputs:
                losses["vq"] = outputs["vq_loss"]
                total_loss = total_loss + self.lambda_vq * outputs["vq_loss"]

        # Phase 2: Property Prediction
        elif phase == "property":
            if "property_predictions" in outputs and "property_targets" in targets:
                loss_property = self.property_prediction_loss(
                    outputs["property_predictions"],
                    targets["property_targets"],
                    targets.get("property_mask", None),
                )
                losses["property"] = loss_property
                total_loss = total_loss + self.lambda_property * loss_property

            # Still include VAE regularization
            if "vae_params" in outputs:
                from ..models.latent import HierarchicalLatentWorldState
                dummy_model = HierarchicalLatentWorldState()
                loss_kl = dummy_model.compute_kl_loss(outputs["vae_params"])
                losses["kl"] = loss_kl
                total_loss = total_loss + self.lambda_kl * loss_kl

        # Phase 3: Planning & RL
        elif phase == "planning":
            # Goal-conditioned success
            if "final_energy" in outputs and "target_energy" in targets:
                loss_goal = F.mse_loss(
                    outputs["final_energy"],
                    targets["target_energy"],
                )
                losses["goal"] = loss_goal
                total_loss = total_loss + loss_goal

        losses["total"] = total_loss
        return losses


class ContrastiveEnergyLoss(nn.Module):
    """
    Standalone contrastive loss for energy model training.

    Uses in-batch negatives and hard negative mining.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        latent_states,
        z_targets,
        z_envs,
        p_targets,
        energy_model,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent_states: Batch of latent states
            z_targets: Target embeddings [B, target_dim]
            z_envs: Environment embeddings [B, env_dim]
            p_targets: Property targets [B, property_dim]
            energy_model: Energy model
            labels: Binary labels [B] (1 = positive, 0 = negative)

        Returns:
            Loss value
        """
        B = z_targets.size(0)

        # Compute energies
        energies = []
        for i in range(B):
            state = latent_states[i]
            energy_output = energy_model(
                state,
                z_targets[i:i+1],
                z_envs[i:i+1],
                p_targets[i:i+1],
            )
            energies.append(energy_output["total_energy"])

        energies = torch.cat(energies, dim=0)  # [B]

        if labels is not None:
            # Supervised contrastive
            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                loss_pos = (energies[pos_mask] ** 2).mean()
                loss_neg = torch.relu(self.margin - energies[neg_mask]).mean()
                loss = loss_pos + loss_neg
            else:
                loss = torch.tensor(0.0, device=energies.device)
        else:
            # Unsupervised: minimize overall energy
            loss = energies.mean()

        return loss
