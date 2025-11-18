"""
Environment/Condition Encoder with domain prototype learning.

Encodes reaction conditions (pH, temperature, solvent, etc.) into a latent space
where common condition clusters are represented as learnable prototypes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class DomainPrototypeLayer(nn.Module):
    """
    Learns prototypes for common experimental condition domains.

    Examples of prototypes:
    - "aqueous basic" (pH > 10, water solvent)
    - "organic aprotic" (DMF/DMSO, RT)
    - "acidic reflux" (pH < 3, elevated temp)
    """

    def __init__(self, input_dim: int, num_prototypes: int = 32, prototype_dim: int = 128):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim

        # Learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))

        # Input projection
        self.input_proj = nn.Linear(input_dim, prototype_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim),
            nn.Tanh(),
            nn.Linear(prototype_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, input_dim]

        Returns:
            Prototype-based embedding [B, prototype_dim]
        """
        # Project input
        x_proj = self.input_proj(x)  # [B, prototype_dim]

        # Compute similarity to each prototype
        # Using cosine similarity
        x_norm = torch.nn.functional.normalize(x_proj, dim=-1)
        proto_norm = torch.nn.functional.normalize(self.prototypes, dim=-1)

        similarity = torch.matmul(x_norm, proto_norm.t())  # [B, num_prototypes]

        # Compute attention weights
        attn_input = similarity.unsqueeze(-1) * self.prototypes.unsqueeze(0)  # [B, num_prototypes, prototype_dim]
        attn_scores = self.attention(attn_input).squeeze(-1)  # [B, num_prototypes]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_prototypes]

        # Weighted combination of prototypes
        output = torch.matmul(attn_weights, self.prototypes)  # [B, prototype_dim]

        # Residual connection
        output = output + x_proj

        return output


class EnvironmentEncoder(nn.Module):
    """
    Encodes reaction conditions and environment into latent space.

    Handles both categorical (solvent type, reaction type) and continuous
    (pH, temperature, pressure) features using domain prototype learning.

    Args:
        categorical_dims: Dictionary mapping categorical feature names to vocabulary sizes
        continuous_features: List of continuous feature names
        output_dim: Output embedding dimension (default: 128)
        num_prototypes: Number of domain prototypes to learn (default: 32)
        hidden_dim: Hidden dimension (default: 256)
    """

    def __init__(
        self,
        categorical_dims: Optional[Dict[str, int]] = None,
        continuous_features: Optional[List[str]] = None,
        output_dim: int = 128,
        num_prototypes: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.categorical_dims = categorical_dims or {}
        self.continuous_features = continuous_features or []
        self.output_dim = output_dim
        self.num_prototypes = num_prototypes

        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, hidden_dim // len(categorical_dims))
            for name, vocab_size in self.categorical_dims.items()
        })

        # Continuous feature processing
        if len(self.continuous_features) > 0:
            self.continuous_net = nn.Sequential(
                nn.Linear(len(self.continuous_features), hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.continuous_net = None

        # Compute input dimension for prototype layer
        cat_dim = sum([hidden_dim // len(categorical_dims) for _ in self.categorical_dims]) if self.categorical_dims else 0
        cont_dim = hidden_dim if self.continuous_net is not None else 0
        total_dim = cat_dim + cont_dim

        # Domain prototype learning
        self.prototype_layer = DomainPrototypeLayer(
            input_dim=total_dim,
            num_prototypes=num_prototypes,
            prototype_dim=hidden_dim,
        )

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            categorical: Dictionary of categorical features [B] for each feature
            continuous: Continuous features [B, num_continuous_features]

        Returns:
            Environment embedding [B, output_dim]
        """
        features = []

        # Process categorical features
        if categorical is not None:
            for name, values in categorical.items():
                if name in self.categorical_embeddings:
                    emb = self.categorical_embeddings[name](values)
                    features.append(emb)

        # Process continuous features
        if continuous is not None and self.continuous_net is not None:
            cont_emb = self.continuous_net(continuous)
            features.append(cont_emb)

        # Concatenate all features
        if len(features) == 0:
            raise ValueError("At least one of categorical or continuous features must be provided")

        x = torch.cat(features, dim=-1)

        # Apply domain prototype learning
        x = self.prototype_layer(x)

        # Project to output dimension
        z_env = self.output_net(x)

        return z_env

    @staticmethod
    def create_default(output_dim: int = 128) -> "EnvironmentEncoder":
        """
        Create encoder with default chemistry-relevant features.

        Categorical:
        - solvent_type: water, DMSO, DMF, THF, DCM, etc. (20 common solvents)
        - reaction_type: Suzuki, Buchwald, amide_coupling, etc. (50 common reactions)
        - atmosphere: air, N2, Ar, vacuum (4 options)

        Continuous:
        - pH: 0-14
        - temperature: -80 to 200°C (normalized)
        - pressure: 0-10 bar (normalized)
        - concentration: 0-10 M (normalized)
        - time: 0-48 hours (normalized)
        """
        categorical_dims = {
            "solvent": 25,  # Including "unknown"
            "reaction_type": 55,  # Including "unknown"
            "atmosphere": 5,  # Including "unknown"
        }

        continuous_features = [
            "pH",
            "temperature",
            "pressure",
            "concentration",
            "time",
        ]

        return EnvironmentEncoder(
            categorical_dims=categorical_dims,
            continuous_features=continuous_features,
            output_dim=output_dim,
        )


# Useful constants for encoding
SOLVENT_VOCAB = {
    "unknown": 0,
    "water": 1,
    "DMSO": 2,
    "DMF": 3,
    "THF": 4,
    "DCM": 5,
    "chloroform": 6,
    "acetone": 7,
    "methanol": 8,
    "ethanol": 9,
    "acetonitrile": 10,
    "toluene": 11,
    "benzene": 12,
    "hexane": 13,
    "ethyl_acetate": 14,
    "diethyl_ether": 15,
    "dioxane": 16,
    "pyridine": 17,
    "acetic_acid": 18,
    "NMP": 19,
    "DMA": 20,
    # ... add more as needed
}

REACTION_TYPE_VOCAB = {
    "unknown": 0,
    "Suzuki_coupling": 1,
    "Buchwald_Hartwig": 2,
    "amide_coupling": 3,
    "reductive_amination": 4,
    "Grignard": 5,
    "Wittig": 6,
    "Diels_Alder": 7,
    "click_chemistry": 8,
    "Sonogashira": 9,
    "Heck": 10,
    "Negishi": 11,
    "Stille": 12,
    "esterification": 13,
    "hydrolysis": 14,
    "oxidation": 15,
    "reduction": 16,
    "alkylation": 17,
    "acylation": 18,
    "nitration": 19,
    "halogenation": 20,
    # ... add more as needed
}

ATMOSPHERE_VOCAB = {
    "unknown": 0,
    "air": 1,
    "N2": 2,
    "Ar": 3,
    "vacuum": 4,
}
