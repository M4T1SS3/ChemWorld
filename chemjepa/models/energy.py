"""
Energy-Based Compatibility Function

Learns an energy landscape where low energy = high compatibility.

E(z_mol, z_target, z_env, p_target) = weighted sum of:
  - E_binding: Binding affinity to target
  - E_stability: Chemical stability in environment
  - E_property: Match to desired properties
  - E_feasibility: Synthetic accessibility
  - E_density: Latent density (regularization)

Key innovation: Learned energy decomposition via meta-network.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .latent import LatentState


class BindingAffinityEnergy(nn.Module):
    """
    Energy term for molecule-target binding affinity.

    Predicts interaction strength using learned dot product + MLP.
    """

    def __init__(self, mol_dim: int = 768, target_dim: int = 256, hidden_dim: int = 512):
        super().__init__()

        # Bilinear interaction
        self.bilinear = nn.Bilinear(mol_dim, target_dim, hidden_dim)

        # Refinement network
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_mol: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_mol: Molecular embedding [B, mol_dim]
            z_target: Target protein embedding [B, target_dim]

        Returns:
            Binding energy [B, 1] (lower = stronger binding)
        """
        interaction = self.bilinear(z_mol, z_target)
        energy = self.net(interaction)
        return energy


class StabilityEnergy(nn.Module):
    """
    Energy term for chemical stability in given environment.

    Considers:
    - pH stability
    - Temperature stability
    - Solvent compatibility
    """

    def __init__(self, mol_dim: int = 768, env_dim: int = 128, hidden_dim: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(mol_dim + env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_mol: torch.Tensor, z_env: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_mol: Molecular embedding [B, mol_dim]
            z_env: Environment embedding [B, env_dim]

        Returns:
            Stability energy [B, 1] (lower = more stable)
        """
        combined = torch.cat([z_mol, z_env], dim=-1)
        energy = self.net(combined)
        return energy


class PropertyMatchEnergy(nn.Module):
    """
    Energy term for matching target properties (ADMET, etc.).

    Multi-task prediction of properties with learned property weights.
    """

    def __init__(
        self,
        mol_dim: int = 768,
        property_dim: int = 64,
        num_properties: int = 10,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.num_properties = num_properties

        # Shared molecular representation
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Property-specific heads
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_properties)
        ])

        # Property target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(property_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_properties),
        )

    def forward(
        self,
        z_mol: torch.Tensor,
        p_target: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_mol: Molecular embedding [B, mol_dim]
            p_target: Target property vector [B, property_dim]
            property_mask: Mask for which properties to consider [B, num_properties]

        Returns:
            energy: Property mismatch energy [B, 1]
            predictions: Individual property predictions [B, num_properties]
        """
        # Encode molecule
        h = self.mol_encoder(z_mol)

        # Predict properties
        predictions = torch.stack([head(h).squeeze(-1) for head in self.property_heads], dim=-1)  # [B, num_properties]

        # Encode targets
        targets = self.target_encoder(p_target)  # [B, num_properties]

        # Compute mismatch (lower = better match)
        mismatch = (predictions - targets) ** 2

        # Apply mask if provided
        if property_mask is not None:
            mismatch = mismatch * property_mask

        # Aggregate
        energy = mismatch.sum(dim=-1, keepdim=True)

        return energy, predictions


class FeasibilityEnergy(nn.Module):
    """
    Energy term for synthetic feasibility.

    Estimates:
    - Synthetic accessibility (SA score)
    - Number of synthesis steps
    - Availability of starting materials
    """

    def __init__(self, mol_dim: int = 768, rxn_dim: int = 384, hidden_dim: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(mol_dim + rxn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_mol: torch.Tensor, z_rxn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_mol: Molecular embedding [B, mol_dim]
            z_rxn: Reaction state [B, rxn_dim]

        Returns:
            Feasibility energy [B, 1] (lower = more feasible)
        """
        combined = torch.cat([z_mol, z_rxn], dim=-1)
        energy = self.net(combined)
        return energy


class MetaWeightingNetwork(nn.Module):
    """
    Learns to dynamically weight different energy terms based on task context.

    Allows flexible multi-objective optimization without retraining:
    - For drug discovery: prioritize binding + ADMET
    - For materials: prioritize stability + properties
    - For synthesis: prioritize feasibility
    """

    def __init__(
        self,
        target_dim: int = 256,
        property_dim: int = 64,
        num_energy_terms: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.num_energy_terms = num_energy_terms

        self.net = nn.Sequential(
            nn.Linear(target_dim + property_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_energy_terms),
            nn.Softmax(dim=-1),  # Weights sum to 1
        )

    def forward(self, z_target: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_target: Target embedding [B, target_dim]
            p_target: Target properties [B, property_dim]

        Returns:
            Energy weights [B, num_energy_terms]
        """
        context = torch.cat([z_target, p_target], dim=-1)
        weights = self.net(context)
        return weights


class EnergyModel(nn.Module):
    """
    Energy-based compatibility function with learned decomposition.

    E(z, c) = Σᵢ αᵢ · Eᵢ(z, c)

    where α are learned via meta-network based on task context.

    Args:
        mol_dim: Molecular embedding dimension (default: 768)
        rxn_dim: Reaction state dimension (default: 384)
        context_dim: Context dimension (default: 256)
        target_dim: Target protein dimension (default: 256)
        env_dim: Environment dimension (default: 128)
        property_dim: Property vector dimension (default: 64)
        num_properties: Number of property prediction heads (default: 10)
        use_density_term: Whether to include latent density regularization (default: True)
    """

    def __init__(
        self,
        mol_dim: int = 768,
        rxn_dim: int = 384,
        context_dim: int = 256,
        target_dim: int = 256,
        env_dim: int = 128,
        property_dim: int = 64,
        num_properties: int = 10,
        use_density_term: bool = True,
    ):
        super().__init__()

        self.mol_dim = mol_dim
        self.use_density_term = use_density_term

        # Energy components
        self.binding_energy = BindingAffinityEnergy(mol_dim, target_dim)
        self.stability_energy = StabilityEnergy(mol_dim, env_dim)
        self.property_energy = PropertyMatchEnergy(mol_dim, property_dim, num_properties)
        self.feasibility_energy = FeasibilityEnergy(mol_dim, rxn_dim)

        num_terms = 4
        if use_density_term:
            num_terms = 5
            # Density energy (simple MLP)
            self.density_energy = nn.Sequential(
                nn.Linear(mol_dim, 512),
                nn.GELU(),
                nn.Linear(512, 1),
            )

        # Meta-weighting network
        self.meta_network = MetaWeightingNetwork(
            target_dim=target_dim,
            property_dim=property_dim,
            num_energy_terms=num_terms,
        )

    def forward(
        self,
        latent_state: LatentState,
        z_target: torch.Tensor,
        z_env: torch.Tensor,
        p_target: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total energy and optionally individual components.

        Args:
            latent_state: Hierarchical latent state
            z_target: Target protein embedding [B, target_dim]
            z_env: Environment embedding [B, env_dim]
            p_target: Target property vector [B, property_dim]
            property_mask: Mask for which properties matter [B, num_properties]
            return_components: Whether to return individual energy terms

        Returns:
            Dictionary with:
                - total_energy: Weighted sum [B, 1]
                - weights: Energy term weights [B, num_terms]
                - (optional) individual energy terms
                - (optional) property predictions
        """
        z_mol = latent_state.z_mol
        z_rxn = latent_state.z_rxn

        # Compute individual energy terms
        e_binding = self.binding_energy(z_mol, z_target)
        e_stability = self.stability_energy(z_mol, z_env)
        e_property, property_preds = self.property_energy(z_mol, p_target, property_mask)
        e_feasibility = self.feasibility_energy(z_mol, z_rxn)

        # Stack energies
        energies = [e_binding, e_stability, e_property, e_feasibility]

        if self.use_density_term:
            e_density = self.density_energy(z_mol)
            energies.append(e_density)

        energy_stack = torch.cat(energies, dim=-1)  # [B, num_terms]

        # Compute adaptive weights
        weights = self.meta_network(z_target, p_target)  # [B, num_terms]

        # Weighted sum
        total_energy = (energy_stack * weights).sum(dim=-1, keepdim=True)  # [B, 1]

        # Build output
        output = {
            "total_energy": total_energy,
            "weights": weights,
            "property_predictions": property_preds,
        }

        if return_components:
            output["e_binding"] = e_binding
            output["e_stability"] = e_stability
            output["e_property"] = e_property
            output["e_feasibility"] = e_feasibility
            if self.use_density_term:
                output["e_density"] = e_density

        return output

    def score_compatibility(
        self,
        latent_state: LatentState,
        z_target: torch.Tensor,
        z_env: torch.Tensor,
        p_target: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute compatibility score (negative energy).

        Higher score = better compatibility.

        Args:
            (same as forward)

        Returns:
            Compatibility score [B, 1]
        """
        output = self.forward(latent_state, z_target, z_env, p_target, property_mask)
        return -output["total_energy"]

    def contrastive_loss(
        self,
        latent_state_pos: LatentState,
        latent_state_neg: LatentState,
        z_target: torch.Tensor,
        z_env: torch.Tensor,
        p_target: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Contrastive loss for training.

        Positive pairs should have low energy, negatives should have high energy.

        Args:
            latent_state_pos: Positive (compatible) molecules
            latent_state_neg: Negative (incompatible) molecules
            z_target, z_env, p_target: Context
            margin: Margin for contrastive loss

        Returns:
            Loss value
        """
        e_pos = self.forward(latent_state_pos, z_target, z_env, p_target)["total_energy"]
        e_neg = self.forward(latent_state_neg, z_target, z_env, p_target)["total_energy"]

        # Positive energies should be low, negative energies should be high
        loss_pos = torch.mean((e_pos - 0) ** 2)
        loss_neg = torch.mean(torch.relu(margin - e_neg) ** 2)

        return loss_pos + loss_neg
