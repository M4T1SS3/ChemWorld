"""
Hierarchical Latent World State (H-LWS)

Three-tier latent space with explicit hierarchy:
- z_mol: Molecular structure (bonds, atoms, topology)
- z_rxn: Reaction mechanism state (transformations, intermediates)
- z_context: Environment + target + properties (conditions, objectives)

Key features:
- Information bottleneck between levels
- Causal masking (context cannot depend on future reactions)
- Modular sub-worlds for different chemistry domains
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class LatentState:
    """Container for hierarchical latent state."""
    z_mol: torch.Tensor      # [B, mol_dim] - molecular structure
    z_rxn: torch.Tensor      # [B, rxn_dim] - reaction state
    z_context: torch.Tensor  # [B, context_dim] - environment/target/properties

    def concatenate(self) -> torch.Tensor:
        """Concatenate all levels into single vector."""
        return torch.cat([self.z_mol, self.z_rxn, self.z_context], dim=-1)

    def to(self, device):
        """Move all tensors to device."""
        return LatentState(
            z_mol=self.z_mol.to(device),
            z_rxn=self.z_rxn.to(device),
            z_context=self.z_context.to(device),
        )


class AttentionGate(nn.Module):
    """
    Gated attention mechanism for information flow control.

    Controls how much information flows from z_mol to z_rxn,
    implementing the information bottleneck.
    """

    def __init__(self, mol_dim: int, rxn_dim: int):
        super().__init__()

        self.query = nn.Linear(rxn_dim, mol_dim)
        self.key = nn.Linear(mol_dim, mol_dim)
        self.value = nn.Linear(mol_dim, rxn_dim)

        self.gate = nn.Sequential(
            nn.Linear(mol_dim + rxn_dim, rxn_dim),
            nn.Sigmoid(),
        )

        self.scale = mol_dim ** -0.5

    def forward(self, z_mol: torch.Tensor, z_rxn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_mol: Molecular embedding [B, mol_dim]
            z_rxn: Reaction state [B, rxn_dim]

        Returns:
            Gated information flow [B, rxn_dim]
        """
        # Attention
        q = self.query(z_rxn)  # [B, mol_dim]
        k = self.key(z_mol)    # [B, mol_dim]
        v = self.value(z_mol)  # [B, rxn_dim]

        attn = torch.sum(q * k, dim=-1, keepdim=True) * self.scale  # [B, 1]
        attn = torch.softmax(attn, dim=-1)

        attended = attn * v  # [B, rxn_dim]

        # Gate
        gate_input = torch.cat([z_mol, z_rxn], dim=-1)
        gate_value = self.gate(gate_input)

        return gate_value * attended


class ReactionSubspace(nn.Module):
    """
    Domain-specific reaction subspace.

    Each chemistry domain (organic, biochemistry, polymer, etc.) has its own
    reaction subspace with shared molecular embeddings.
    """

    def __init__(
        self,
        domain_name: str,
        mol_dim: int,
        rxn_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.domain_name = domain_name
        self.mol_dim = mol_dim
        self.rxn_dim = rxn_dim

        # Domain-specific processing
        self.attention_gate = AttentionGate(mol_dim, rxn_dim)

        self.rxn_net = nn.Sequential(
            nn.Linear(rxn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, rxn_dim),
        )

    def forward(
        self,
        z_mol: torch.Tensor,
        z_rxn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_mol: Molecular embedding [B, mol_dim]
            z_rxn: Current reaction state [B, rxn_dim]

        Returns:
            Updated reaction state [B, rxn_dim]
        """
        # Gated information flow from molecule
        mol_info = self.attention_gate(z_mol, z_rxn)

        # Process reaction state
        z_rxn_new = self.rxn_net(z_rxn + mol_info)

        # Residual connection
        return z_rxn + z_rxn_new


class HierarchicalLatentWorldState(nn.Module):
    """
    Hierarchical latent world state with three tiers.

    Implements compositional structure with:
    - Information bottleneck (z_mol -> z_rxn gated)
    - Causal masking (z_context independent of future)
    - Modular domains (separate z_rxn subspaces)

    Args:
        mol_dim: Dimension of molecular embeddings (default: 768)
        rxn_dim: Dimension of reaction state (default: 384)
        context_dim: Dimension of context (env + target + properties) (default: 256)
        domains: List of chemistry domains to support (default: ["organic"])
        use_vae: Whether to use VAE-style regularization (default: True)
    """

    def __init__(
        self,
        mol_dim: int = 768,
        rxn_dim: int = 384,
        context_dim: int = 256,
        domains: list = None,
        use_vae: bool = True,
    ):
        super().__init__()

        self.mol_dim = mol_dim
        self.rxn_dim = rxn_dim
        self.context_dim = context_dim
        self.use_vae = use_vae

        # Default to organic chemistry domain
        self.domains = domains or ["organic"]

        # Reaction subspaces (one per domain)
        self.reaction_subspaces = nn.ModuleDict({
            domain: ReactionSubspace(
                domain_name=domain,
                mol_dim=mol_dim,
                rxn_dim=rxn_dim,
            )
            for domain in self.domains
        })

        # VAE components (optional)
        if use_vae:
            # Molecular VAE
            self.mol_mu = nn.Linear(mol_dim, mol_dim)
            self.mol_logvar = nn.Linear(mol_dim, mol_dim)

            # Reaction VAE
            self.rxn_mu = nn.Linear(rxn_dim, rxn_dim)
            self.rxn_logvar = nn.Linear(rxn_dim, rxn_dim)

            # Context VAE
            self.context_mu = nn.Linear(context_dim, context_dim)
            self.context_logvar = nn.Linear(context_dim, context_dim)

        # Context fusion (combines env + target + properties)
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim * 2, context_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_molecular(self, z_mol: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode molecular state with optional VAE.

        Returns:
            z_mol: (Potentially stochastic) molecular embedding
            mu: Mean (if VAE)
            logvar: Log variance (if VAE)
        """
        if self.use_vae and self.training:
            mu = self.mol_mu(z_mol)
            logvar = self.mol_logvar(z_mol)
            z_mol = self.reparameterize(mu, logvar)
            return z_mol, mu, logvar
        else:
            return z_mol, None, None

    def encode_reaction(
        self,
        z_mol: torch.Tensor,
        z_rxn: torch.Tensor,
        domain: str = "organic",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode reaction state with domain-specific processing.

        Args:
            z_mol: Molecular embedding [B, mol_dim]
            z_rxn: Reaction state [B, rxn_dim]
            domain: Chemistry domain (e.g., "organic", "biochemistry")

        Returns:
            z_rxn: Updated reaction state
            mu: Mean (if VAE)
            logvar: Log variance (if VAE)
        """
        # Apply domain-specific reaction subspace
        if domain not in self.reaction_subspaces:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.reaction_subspaces.keys())}")

        z_rxn = self.reaction_subspaces[domain](z_mol, z_rxn)

        if self.use_vae and self.training:
            mu = self.rxn_mu(z_rxn)
            logvar = self.rxn_logvar(z_rxn)
            z_rxn = self.reparameterize(mu, logvar)
            return z_rxn, mu, logvar
        else:
            return z_rxn, None, None

    def encode_context(self, z_context: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode context (environment + target + properties).

        Returns:
            z_context: Processed context embedding
            mu: Mean (if VAE)
            logvar: Log variance (if VAE)
        """
        z_context = self.context_fusion(z_context)

        if self.use_vae and self.training:
            mu = self.context_mu(z_context)
            logvar = self.context_logvar(z_context)
            z_context = self.reparameterize(mu, logvar)
            return z_context, mu, logvar
        else:
            return z_context, None, None

    def forward(
        self,
        z_mol: torch.Tensor,
        z_rxn: torch.Tensor,
        z_context: torch.Tensor,
        domain: str = "organic",
    ) -> Tuple[LatentState, Dict[str, torch.Tensor]]:
        """
        Process hierarchical latent state.

        Args:
            z_mol: Molecular embeddings [B, mol_dim]
            z_rxn: Reaction state [B, rxn_dim]
            z_context: Context (env + target + properties) [B, context_dim]
            domain: Chemistry domain

        Returns:
            latent_state: Processed hierarchical state
            vae_params: Dictionary with VAE parameters (mu, logvar) for each level
        """
        vae_params = {}

        # Encode each level
        z_mol, mol_mu, mol_logvar = self.encode_molecular(z_mol)
        if mol_mu is not None:
            vae_params["mol_mu"] = mol_mu
            vae_params["mol_logvar"] = mol_logvar

        z_rxn, rxn_mu, rxn_logvar = self.encode_reaction(z_mol, z_rxn, domain)
        if rxn_mu is not None:
            vae_params["rxn_mu"] = rxn_mu
            vae_params["rxn_logvar"] = rxn_logvar

        z_context, ctx_mu, ctx_logvar = self.encode_context(z_context)
        if ctx_mu is not None:
            vae_params["context_mu"] = ctx_mu
            vae_params["context_logvar"] = ctx_logvar

        # Create latent state
        latent_state = LatentState(
            z_mol=z_mol,
            z_rxn=z_rxn,
            z_context=z_context,
        )

        return latent_state, vae_params

    def add_domain(self, domain_name: str):
        """
        Add new chemistry domain without catastrophic forgetting.

        Args:
            domain_name: Name of new domain (e.g., "polymer_chemistry")
        """
        if domain_name not in self.reaction_subspaces:
            self.reaction_subspaces[domain_name] = ReactionSubspace(
                domain_name=domain_name,
                mol_dim=self.mol_dim,
                rxn_dim=self.rxn_dim,
            )
            self.domains.append(domain_name)
            print(f"Added new domain: {domain_name}")

    @property
    def total_dim(self) -> int:
        """Total dimension of concatenated latent state."""
        return self.mol_dim + self.rxn_dim + self.context_dim

    def compute_kl_loss(self, vae_params: Dict[str, torch.Tensor], beta: float = 1.0) -> torch.Tensor:
        """
        Compute KL divergence loss for VAE regularization.

        Args:
            vae_params: Dictionary with mu and logvar for each level
            beta: Beta-VAE weight (default: 1.0)

        Returns:
            Total KL loss
        """
        kl_loss = 0.0

        for level in ["mol", "rxn", "context"]:
            mu_key = f"{level}_mu"
            logvar_key = f"{level}_logvar"

            if mu_key in vae_params and logvar_key in vae_params:
                mu = vae_params[mu_key]
                logvar = vae_params[logvar_key]

                # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                kl_loss = kl_loss + kl.mean()

        return beta * kl_loss
