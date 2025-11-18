"""
Protein/Target Encoder using pre-trained ESM-2 + geometric networks.

Integrates sequence information from ESM-2 with structural information
when available, using binding site attention to focus on relevant regions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_geometric.nn import GATv2Conv, global_mean_pool


class BindingSiteAttention(nn.Module):
    """
    Learns to weight ESM-2 residue embeddings by predicted binding pocket importance.

    Can be trained via:
    - Weak supervision from docking scores
    - Known binding site annotations
    - Contrastive learning (binders vs non-binders)
    """

    def __init__(self, esm_dim: int = 1280, hidden_dim: int = 256):
        super().__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            residue_embeddings: ESM-2 residue embeddings [B, L, esm_dim]
            attention_mask: Mask for padding [B, L] (1 = real, 0 = padding)

        Returns:
            weighted_embeddings: Attention-weighted embeddings [B, esm_dim]
            attention_weights: Binding site importance scores [B, L]
        """
        # Compute attention scores
        scores = self.attention_net(residue_embeddings).squeeze(-1)  # [B, L]

        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), -1e9)

        # Softmax to get weights
        weights = torch.softmax(scores, dim=-1)  # [B, L]

        # Weighted sum
        weighted = torch.sum(residue_embeddings * weights.unsqueeze(-1), dim=1)  # [B, esm_dim]

        return weighted, weights


class ProteinStructureEncoder(nn.Module):
    """
    Encodes protein structure using graph attention networks.

    Represents protein as graph where:
    - Nodes = amino acids (or atoms in full-atom mode)
    - Edges = spatial proximity (< 10Å) or sequence neighbors
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 16,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, edge_dim)

        # Graph attention layers
        self.conv_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 8,
                heads=8,
                edge_dim=edge_dim,
                concat=True,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, node_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            batch: Batch assignment [N]

        Returns:
            Protein structure embedding [B, output_dim]
        """
        h = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for conv in self.conv_layers:
            h_new = conv(h, edge_index, edge_attr)
            h = h + h_new  # Residual

        # Global pooling
        h_global = global_mean_pool(h, batch)

        return self.output_proj(h_global)


class ProteinEncoder(nn.Module):
    """
    Unified protein encoder combining sequence (ESM-2) and structure information.

    Args:
        use_esm: Whether to use ESM-2 sequence embeddings (default: True)
        use_structure: Whether to use structural information (default: False)
        esm_dim: ESM-2 embedding dimension (1280 for ESM-2)
        structure_dim: Structure encoder output dimension
        output_dim: Final output dimension (default: 256)
        freeze_esm: Whether to freeze ESM-2 weights (default: True)
    """

    def __init__(
        self,
        use_esm: bool = True,
        use_structure: bool = False,
        esm_dim: int = 1280,
        structure_dim: int = 256,
        output_dim: int = 256,
        freeze_esm: bool = True,
    ):
        super().__init__()

        self.use_esm = use_esm
        self.use_structure = use_structure
        self.output_dim = output_dim

        # ESM-2 model (loaded lazily)
        self.esm_model = None
        self.esm_dim = esm_dim
        self.freeze_esm = freeze_esm

        if use_esm:
            self.binding_site_attention = BindingSiteAttention(
                esm_dim=esm_dim,
                hidden_dim=256,
            )

        # Structure encoder
        if use_structure:
            self.structure_encoder = ProteinStructureEncoder(
                output_dim=structure_dim,
            )
            self.structure_dim = structure_dim
        else:
            self.structure_dim = 0

        # Fusion network
        fusion_input_dim = 0
        if use_esm:
            fusion_input_dim += esm_dim
        if use_structure:
            fusion_input_dim += structure_dim

        if fusion_input_dim == 0:
            raise ValueError("At least one of use_esm or use_structure must be True")

        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _load_esm_model(self):
        """Lazy loading of ESM-2 model."""
        if self.esm_model is None and self.use_esm:
            try:
                import esm
                # Load smaller ESM-2 model (esm2_t33_650M_UR50D)
                # For production, use esm2_t36_3B_UR50D
                self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

                if self.freeze_esm:
                    for param in self.esm_model.parameters():
                        param.requires_grad = False

                self.esm_model.eval()

            except ImportError:
                raise ImportError(
                    "ESM is not installed. Install with: pip install fair-esm"
                )

    def forward(
        self,
        sequence: Optional[str] = None,
        esm_embeddings: Optional[torch.Tensor] = None,
        structure_graph: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            sequence: Amino acid sequence (single letter codes)
            esm_embeddings: Pre-computed ESM-2 embeddings [B, L, esm_dim] (optional)
            structure_graph: Tuple of (x, edge_index, edge_attr, batch) for structure
            return_attention: Whether to return binding site attention weights

        Returns:
            Protein embedding [B, output_dim]
            (optionally) attention_weights [B, L] if return_attention=True
        """
        features = []
        attention_weights = None

        # Process sequence with ESM-2
        if self.use_esm:
            if esm_embeddings is None:
                if sequence is None:
                    raise ValueError("Either sequence or esm_embeddings must be provided")

                self._load_esm_model()

                # Tokenize and embed
                # This is a simplified version - in practice, handle batching properly
                data = [(f"protein", sequence)]
                batch_labels, batch_strs, batch_tokens = self.esm_alphabet.get_batch_converter()(data)

                with torch.no_grad():
                    results = self.esm_model(batch_tokens, repr_layers=[33])
                    esm_embeddings = results["representations"][33][:, 1:-1, :]  # Remove BOS/EOS

            # Apply binding site attention
            seq_features, attention_weights = self.binding_site_attention(esm_embeddings)
            features.append(seq_features)

        # Process structure
        if self.use_structure:
            if structure_graph is None:
                raise ValueError("structure_graph must be provided when use_structure=True")

            x, edge_index, edge_attr, batch = structure_graph
            struct_features = self.structure_encoder(x, edge_index, edge_attr, batch)
            features.append(struct_features)

        # Fuse features
        fused = torch.cat(features, dim=-1)
        z_target = self.fusion_net(fused)

        if return_attention:
            return z_target, attention_weights
        return z_target

    @staticmethod
    def create_sequence_only(output_dim: int = 256, freeze_esm: bool = True) -> "ProteinEncoder":
        """Create encoder that only uses sequence information (ESM-2)."""
        return ProteinEncoder(
            use_esm=True,
            use_structure=False,
            output_dim=output_dim,
            freeze_esm=freeze_esm,
        )

    @staticmethod
    def create_structure_only(output_dim: int = 256) -> "ProteinEncoder":
        """Create encoder that only uses structural information."""
        return ProteinEncoder(
            use_esm=False,
            use_structure=True,
            output_dim=output_dim,
        )

    @staticmethod
    def create_multimodal(output_dim: int = 256, freeze_esm: bool = True) -> "ProteinEncoder":
        """Create encoder that uses both sequence and structure."""
        return ProteinEncoder(
            use_esm=True,
            use_structure=True,
            output_dim=output_dim,
            freeze_esm=freeze_esm,
        )
