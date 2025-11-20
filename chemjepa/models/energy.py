#!/usr/bin/env python3
"""
Energy-Based Compatibility Model for ChemJEPA

Learns a decomposable energy function that scores molecular candidates against
multiple objectives without retraining. Enables flexible multi-objective optimization.

Architecture:
    E(z_mol, objectives) = Σ_i w_i * E_i(z_mol, θ_i)

Where:
    - E_binding: Binding affinity to target protein
    - E_stability: Molecular stability and synthesizability
    - E_properties: Match to desired property ranges (LogP, MW, etc.)
    - E_novelty: Distance from known molecules (exploration bonus)

Key Features:
    - Dynamic objective weighting without retraining
    - Learned energy decomposition (not hard-coded)
    - Uncertainty-aware scoring via ensemble
    - Compatible with Phase 1 frozen embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EnergyComponent(nn.Module):
    """
    Single energy component (e.g., binding, stability, properties).

    Maps from molecular latent (768-dim) to scalar energy value.
    Uses deep residual architecture for expressiveness.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Output head (energy is scalar)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z_mol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_mol: Molecular latent [batch, 768]

        Returns:
            energy: Scalar energy [batch, 1]
        """
        x = self.input_proj(z_mol)

        # Residual blocks
        for block in self.blocks:
            x = x + block(x)  # Residual connection

        energy = self.output_head(x)
        return energy


class PropertyEnergyComponent(nn.Module):
    """
    Energy component for property matching (LogP, MW, TPSA, etc.).

    Learns to score molecules based on deviation from target property ranges.
    Uses learned distance metrics rather than hard-coded thresholds.
    """

    def __init__(self, input_dim: int = 768, num_properties: int = 5, hidden_dim: int = 256):
        super().__init__()

        self.num_properties = num_properties

        # Property prediction head (same as linear probe in evaluation)
        self.property_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_properties)
        )

        # Learned distance metric for each property
        # Maps (predicted, target) -> energy penalty
        self.distance_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 64),  # [predicted, target]
                nn.SiLU(),
                nn.Linear(64, 1)
            )
            for _ in range(num_properties)
        ])

    def forward(
        self,
        z_mol: torch.Tensor,
        target_properties: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_mol: Molecular latent [batch, 768]
            target_properties: Target property values [batch, num_properties]
                              (None during training, provided during optimization)

        Returns:
            energy: Property matching energy [batch, 1]
            predicted_properties: Predicted property values [batch, num_properties]
        """
        # Predict properties
        predicted = self.property_predictor(z_mol)

        if target_properties is None:
            # Training mode: return zero energy and predictions
            return torch.zeros(z_mol.shape[0], 1, device=z_mol.device), predicted

        # Compute learned distance for each property
        energies = []
        for i, distance_net in enumerate(self.distance_nets):
            pred_i = predicted[:, i:i+1]
            target_i = target_properties[:, i:i+1]

            # Concatenate predicted and target
            pair = torch.cat([pred_i, target_i], dim=-1)  # [batch, 2]

            # Compute distance via learned network
            dist = distance_net(pair)  # [batch, 1]
            energies.append(dist)

        # Sum property distances
        total_energy = sum(energies)

        return total_energy, predicted


class ChemJEPAEnergyModel(nn.Module):
    """
    Complete energy-based compatibility model for ChemJEPA.

    Decomposes total energy into learnable components:
        E_total = w_binding * E_binding + w_stability * E_stability +
                  w_properties * E_properties + w_novelty * E_novelty

    Key features:
        - Component weights can be adjusted at inference time
        - All components trained jointly with contrastive learning
        - Compatible with frozen Phase 1 molecular encoder
    """

    def __init__(
        self,
        mol_dim: int = 768,
        hidden_dim: int = 512,
        num_properties: int = 5,
        use_ensemble: bool = True,
        ensemble_size: int = 3
    ):
        super().__init__()

        self.mol_dim = mol_dim
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size

        # Energy components
        if use_ensemble:
            # Ensemble for uncertainty estimation
            self.binding_components = nn.ModuleList([
                EnergyComponent(mol_dim, hidden_dim) for _ in range(ensemble_size)
            ])
            self.stability_components = nn.ModuleList([
                EnergyComponent(mol_dim, hidden_dim) for _ in range(ensemble_size)
            ])
        else:
            self.binding_component = EnergyComponent(mol_dim, hidden_dim)
            self.stability_component = EnergyComponent(mol_dim, hidden_dim)

        # Property matching (shared across ensemble for efficiency)
        self.property_component = PropertyEnergyComponent(
            mol_dim,
            num_properties=num_properties,
            hidden_dim=hidden_dim // 2
        )

        # Novelty component (distance to training distribution)
        self.novelty_component = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Default weights (can be overridden at inference)
        self.register_buffer('default_weights', torch.tensor([
            1.0,  # binding
            0.5,  # stability
            0.3,  # properties
            0.1   # novelty
        ]))

    def forward(
        self,
        z_mol: torch.Tensor,
        target_properties: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total energy and components.

        Args:
            z_mol: Molecular latent from Phase 1 encoder [batch, 768]
            target_properties: Target property values [batch, num_properties]
            weights: Component weights [4] (binding, stability, properties, novelty)
            return_components: Return individual energy components

        Returns:
            Dictionary with:
                - 'energy': Total energy [batch, 1]
                - 'uncertainty': Energy uncertainty from ensemble [batch, 1]
                - 'components': Dict of individual energies (if return_components=True)
                - 'predicted_properties': Predicted properties [batch, num_properties]
        """
        batch_size = z_mol.shape[0]
        device = z_mol.device

        if weights is None:
            weights = self.default_weights

        # Compute ensemble components
        if self.use_ensemble:
            # Binding affinity (ensemble mean + std)
            binding_energies = torch.stack([
                comp(z_mol) for comp in self.binding_components
            ], dim=0)  # [ensemble_size, batch, 1]
            binding_mean = binding_energies.mean(dim=0)
            binding_std = binding_energies.std(dim=0)

            # Stability (ensemble mean + std)
            stability_energies = torch.stack([
                comp(z_mol) for comp in self.stability_components
            ], dim=0)
            stability_mean = stability_energies.mean(dim=0)
            stability_std = stability_energies.std(dim=0)

            # Total uncertainty
            uncertainty = binding_std + stability_std
        else:
            binding_mean = self.binding_component(z_mol)
            stability_mean = self.stability_component(z_mol)
            uncertainty = torch.zeros_like(binding_mean)

        # Property matching
        property_energy, predicted_props = self.property_component(z_mol, target_properties)

        # Novelty (exploration bonus)
        novelty_energy = self.novelty_component(z_mol)

        # Weighted sum
        total_energy = (
            weights[0] * binding_mean +
            weights[1] * stability_mean +
            weights[2] * property_energy +
            weights[3] * novelty_energy
        )

        # Build output
        output = {
            'energy': total_energy,
            'uncertainty': uncertainty,
            'predicted_properties': predicted_props
        }

        if return_components:
            output['components'] = {
                'binding': binding_mean,
                'stability': stability_mean,
                'properties': property_energy,
                'novelty': novelty_energy
            }

        return output

    def optimize_molecule(
        self,
        z_mol_init: torch.Tensor,
        target_properties: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        lr: float = 0.01
    ) -> Tuple[torch.Tensor, float]:
        """
        Optimize molecular latent to minimize energy.

        This is the core of latent space planning - gradient descent in z_mol space
        to find molecules that minimize the energy function.

        Args:
            z_mol_init: Initial molecular latent [1, 768]
            target_properties: Target property values [1, num_properties]
            weights: Energy component weights
            num_steps: Optimization steps
            lr: Learning rate

        Returns:
            z_mol_optimized: Optimized latent [1, 768]
            final_energy: Final energy value
        """
        z_mol = z_mol_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_mol], lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            output = self.forward(z_mol, target_properties, weights)
            energy = output['energy']

            # Minimize energy
            energy.backward()
            optimizer.step()

        with torch.no_grad():
            final_output = self.forward(z_mol, target_properties, weights)
            final_energy = final_output['energy'].item()

        return z_mol.detach(), final_energy


class EnergyContrastiveLoss(nn.Module):
    """
    Contrastive loss for training energy model.

    Key idea: Molecules with better properties should have lower energy.
    Uses ranking loss to enforce this ordering.
    """

    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        energies: torch.Tensor,
        property_values: torch.Tensor,
        property_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            energies: Predicted energies [batch, 1]
            property_values: True property values [batch, num_properties]
            property_targets: Target property values [num_properties]

        Returns:
            loss: Contrastive ranking loss
        """
        batch_size = energies.shape[0]

        # Compute distance to target for each molecule
        distances = torch.norm(property_values - property_targets.unsqueeze(0), dim=-1)  # [batch]

        # Molecules closer to target should have lower energy
        # Use pairwise ranking loss
        loss = 0.0
        count = 0

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # If molecule i is closer to target than j, E(i) should be lower than E(j)
                    if distances[i] < distances[j]:
                        # E(i) should be < E(j), penalize if E(i) >= E(j) - margin
                        ranking_loss = F.relu(energies[i] - energies[j] + self.margin)
                        loss = loss + ranking_loss
                        count += 1

        return loss / max(count, 1)


if __name__ == '__main__':
    """Quick test of energy model"""
    print("Testing ChemJEPA Energy Model...")

    # Create model
    model = ChemJEPAEnergyModel(
        mol_dim=768,
        hidden_dim=512,
        num_properties=5,
        use_ensemble=True,
        ensemble_size=3
    )

    # Test forward pass
    batch_size = 8
    z_mol = torch.randn(batch_size, 768)
    target_props = torch.randn(batch_size, 5)

    output = model(z_mol, target_props, return_components=True)

    print(f"\nOutput shapes:")
    print(f"  Energy: {output['energy'].shape}")
    print(f"  Uncertainty: {output['uncertainty'].shape}")
    print(f"  Predicted properties: {output['predicted_properties'].shape}")
    print(f"  Components: {list(output['components'].keys())}")

    print(f"\nEnergy statistics:")
    print(f"  Mean: {output['energy'].mean().item():.4f}")
    print(f"  Std:  {output['energy'].std().item():.4f}")
    print(f"  Uncertainty mean: {output['uncertainty'].mean().item():.4f}")

    # Test optimization
    print(f"\nTesting latent optimization...")
    z_init = torch.randn(1, 768)
    target = torch.randn(1, 5)

    z_opt, final_energy = model.optimize_molecule(z_init, target, num_steps=50)
    print(f"  Initial energy: {model(z_init, target)['energy'].item():.4f}")
    print(f"  Final energy:   {final_energy:.4f}")

    print("\n✓ Energy model test passed!")
