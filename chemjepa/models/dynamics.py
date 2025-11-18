"""
Latent Dynamics Predictor

Transformer-based sequence model for predicting state transitions:
f_θ(z_t, a_t) → z_{t+1}

Key features:
- Factored transition: z_{t+1} = z_t + Δz_rxn + Δz_env
- Heteroscedastic uncertainty estimation
- Counterfactual reasoning support
- Action = reaction operator from learned codebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .latent import LatentState


class ReactionCodebook(nn.Module):
    """
    Learned discrete codebook of reaction operators.

    Instead of pre-defining reactions, learns ~1000 common reaction patterns
    from data using vector quantization.
    """

    def __init__(
        self,
        num_reactions: int = 1000,
        reaction_dim: int = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.num_reactions = num_reactions
        self.reaction_dim = reaction_dim
        self.commitment_cost = commitment_cost

        # Learnable codebook
        self.embedding = nn.Embedding(num_reactions, reaction_dim)
        self.embedding.weight.data.uniform_(-1/num_reactions, 1/num_reactions)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vector quantization: map continuous z to nearest codebook entry.

        Args:
            z: Continuous reaction encoding [B, reaction_dim]

        Returns:
            z_q: Quantized (discretized) reaction [B, reaction_dim]
            indices: Codebook indices [B]
            vq_loss: Vector quantization loss
        """
        # Flatten if needed
        original_shape = z.shape
        z_flat = z.view(-1, self.reaction_dim)

        # Calculate distances to codebook entries
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Get nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)

        # VQ loss
        e_latent_loss = F.mse_loss(z_q.detach(), z_flat)
        q_latent_loss = F.mse_loss(z_q, z_flat.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape back
        z_q = z_q.view(original_shape)

        return z_q, indices.view(original_shape[:-1]), vq_loss

    def get_reaction(self, indices: torch.Tensor) -> torch.Tensor:
        """Get reaction embeddings from indices."""
        return self.embedding(indices)


class FactoredTransitionModel(nn.Module):
    """
    Factored state transition model.

    z_{t+1} = z_t + Δz_rxn(z_mol, a_t) + Δz_env(z_context)

    This factorization enables counterfactual reasoning:
    "What if we ran the same reaction at different conditions?"
    """

    def __init__(
        self,
        mol_dim: int = 768,
        rxn_dim: int = 384,
        context_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # Reaction-specific change
        self.reaction_net = nn.Sequential(
            nn.Linear(mol_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mol_dim + rxn_dim),
        )

        # Environment drift
        self.environment_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, mol_dim + rxn_dim),
        )

        # Uncertainty prediction (heteroscedastic)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(mol_dim + action_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mol_dim + rxn_dim),
            nn.Softplus(),  # Ensure positive
        )

    def forward(
        self,
        latent_state: LatentState,
        action: torch.Tensor,
        predict_uncertainty: bool = True,
    ) -> Tuple[LatentState, Optional[torch.Tensor]]:
        """
        Predict next state given current state and action.

        Args:
            latent_state: Current hierarchical state
            action: Reaction operator embedding [B, action_dim]
            predict_uncertainty: Whether to predict uncertainty

        Returns:
            next_state: Predicted next state
            uncertainty: Predicted standard deviation (if predict_uncertainty)
        """
        z_mol = latent_state.z_mol
        z_rxn = latent_state.z_rxn
        z_context = latent_state.z_context

        # Reaction-specific change
        delta_rxn = self.reaction_net(torch.cat([z_mol, action], dim=-1))

        # Environment drift
        delta_env = self.environment_net(z_context)

        # Total change
        delta = delta_rxn + delta_env

        # Split into mol and rxn components
        delta_mol = delta[:, :z_mol.shape[1]]
        delta_rxn_state = delta[:, z_mol.shape[1]:]

        # Update state (context remains the same - causal masking)
        next_z_mol = z_mol + delta_mol
        next_z_rxn = z_rxn + delta_rxn_state

        next_state = LatentState(
            z_mol=next_z_mol,
            z_rxn=next_z_rxn,
            z_context=z_context,  # Context doesn't change
        )

        # Predict uncertainty
        uncertainty = None
        if predict_uncertainty:
            uncertainty_input = torch.cat([z_mol, action, z_context], dim=-1)
            uncertainty = self.uncertainty_net(uncertainty_input)

        return next_state, uncertainty


class DynamicsPredictor(nn.Module):
    """
    Latent dynamics predictor with learned reaction codebook.

    Predicts state transitions in latent space:
    - Forward rollouts (predict future states)
    - Counterfactual reasoning (swap environment)
    - Uncertainty quantification

    Args:
        mol_dim: Molecular embedding dimension (default: 768)
        rxn_dim: Reaction state dimension (default: 384)
        context_dim: Context dimension (default: 256)
        num_reactions: Size of reaction codebook (default: 1000)
        action_dim: Reaction operator dimension (default: 256)
        hidden_dim: Hidden dimension (default: 512)
        num_transformer_layers: Number of transformer layers (default: 4)
    """

    def __init__(
        self,
        mol_dim: int = 768,
        rxn_dim: int = 384,
        context_dim: int = 256,
        num_reactions: int = 1000,
        action_dim: int = 256,
        hidden_dim: int = 512,
        num_transformer_layers: int = 4,
    ):
        super().__init__()

        self.mol_dim = mol_dim
        self.rxn_dim = rxn_dim
        self.context_dim = context_dim
        self.action_dim = action_dim

        # Reaction codebook
        self.codebook = ReactionCodebook(
            num_reactions=num_reactions,
            reaction_dim=action_dim,
        )

        # Action encoder (maps raw action features to codebook space)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # State encoder
        state_dim = mol_dim + rxn_dim + context_dim
        self.state_encoder = nn.Linear(state_dim, hidden_dim)

        # Transformer for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # Factored transition model
        self.transition_model = FactoredTransitionModel(
            mol_dim=mol_dim,
            rxn_dim=rxn_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def encode_action(self, action: torch.Tensor, quantize: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode action and optionally quantize to codebook.

        Args:
            action: Raw action features [B, action_dim]
            quantize: Whether to quantize to discrete codebook

        Returns:
            action_emb: Action embedding [B, action_dim]
            vq_loss: Vector quantization loss (if quantize=True)
        """
        action_emb = self.action_encoder(action)

        vq_loss = None
        if quantize:
            action_emb, _, vq_loss = self.codebook(action_emb)

        return action_emb, vq_loss

    def forward(
        self,
        latent_state: LatentState,
        action: torch.Tensor,
        quantize_action: bool = True,
        predict_uncertainty: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state given current state and action.

        Args:
            latent_state: Current hierarchical state
            action: Reaction operator [B, action_dim]
            quantize_action: Whether to quantize action to codebook
            predict_uncertainty: Whether to predict uncertainty

        Returns:
            Dictionary with:
                - next_state: Predicted next state
                - uncertainty: Predicted uncertainty (optional)
                - vq_loss: Vector quantization loss (optional)
                - action_embedding: Encoded action
        """
        # Encode action
        action_emb, vq_loss = self.encode_action(action, quantize=quantize_action)

        # Encode state
        state_vec = latent_state.concatenate()
        state_emb = self.state_encoder(state_vec).unsqueeze(1)  # [B, 1, hidden_dim]

        # Process with transformer (for multi-step sequences, this is useful)
        # For single-step, it still provides useful representation learning
        state_emb = self.transformer(state_emb).squeeze(1)  # [B, hidden_dim]

        # Predict transition
        next_state, uncertainty = self.transition_model(
            latent_state,
            action_emb,
            predict_uncertainty=predict_uncertainty,
        )

        output = {
            "next_state": next_state,
            "action_embedding": action_emb,
        }

        if uncertainty is not None:
            output["uncertainty"] = uncertainty

        if vq_loss is not None:
            output["vq_loss"] = vq_loss

        return output

    def rollout(
        self,
        initial_state: LatentState,
        actions: torch.Tensor,
        quantize_actions: bool = True,
    ) -> Tuple[list, list]:
        """
        Perform multi-step rollout.

        Args:
            initial_state: Starting state
            actions: Sequence of actions [B, T, action_dim]
            quantize_actions: Whether to quantize actions

        Returns:
            states: List of predicted states (length T+1, includes initial)
            uncertainties: List of predicted uncertainties (length T)
        """
        B, T, _ = actions.shape

        states = [initial_state]
        uncertainties = []

        current_state = initial_state

        for t in range(T):
            action_t = actions[:, t, :]

            output = self.forward(
                current_state,
                action_t,
                quantize_action=quantize_actions,
                predict_uncertainty=True,
            )

            current_state = output["next_state"]
            states.append(current_state)
            uncertainties.append(output["uncertainty"])

        return states, uncertainties

    def counterfactual_rollout(
        self,
        initial_state: LatentState,
        actions: torch.Tensor,
        alternative_context: torch.Tensor,
    ) -> Tuple[list, list]:
        """
        Perform rollout with alternative context (counterfactual).

        Useful for: "What if we ran the same reactions at different pH?"

        Args:
            initial_state: Starting state
            actions: Sequence of actions [B, T, action_dim]
            alternative_context: Alternative z_context [B, context_dim]

        Returns:
            states: List of predicted states
            uncertainties: List of uncertainties
        """
        # Replace context
        counterfactual_state = LatentState(
            z_mol=initial_state.z_mol,
            z_rxn=initial_state.z_rxn,
            z_context=alternative_context,
        )

        return self.rollout(counterfactual_state, actions)

    def predict_loss(
        self,
        current_state: LatentState,
        action: torch.Tensor,
        target_state: LatentState,
        reduction: str = 'mean',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction loss for training.

        Args:
            current_state: Current state
            action: Action taken
            target_state: Ground truth next state
            reduction: Loss reduction ('mean' or 'sum')

        Returns:
            Dictionary with losses
        """
        output = self.forward(current_state, action, quantize_action=True, predict_uncertainty=True)

        next_state = output["next_state"]
        uncertainty = output["uncertainty"]

        # Prediction loss (MSE weighted by uncertainty)
        # More uncertain predictions get lower weight
        target_vec = target_state.concatenate()
        pred_vec = next_state.concatenate()

        mse = (target_vec - pred_vec) ** 2

        if uncertainty is not None:
            # Negative log likelihood under Gaussian
            nll = 0.5 * (mse / (uncertainty ** 2 + 1e-6) + torch.log(uncertainty ** 2 + 1e-6))
            loss_pred = nll.mean() if reduction == 'mean' else nll.sum()
        else:
            loss_pred = mse.mean() if reduction == 'mean' else mse.sum()

        losses = {
            "prediction_loss": loss_pred,
        }

        if "vq_loss" in output:
            losses["vq_loss"] = output["vq_loss"]

        return losses
