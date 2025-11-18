"""
Simple example demonstrating ChemJEPA usage.

This is a minimal example showing how to:
1. Initialize the model
2. Encode a molecule
3. Compute energy/compatibility
4. Run imagination engine
"""

import torch
from chemjepa import ChemJEPA


def main():
    print("=" * 80)
    print("ChemJEPA Simple Example")
    print("=" * 80)

    # Initialize model
    print("\n1. Initializing ChemJEPA model...")
    model = ChemJEPA(
        mol_dim=768,
        rxn_dim=384,
        context_dim=256,
        device='cpu',  # Use CPU for demo
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model initialized with {num_params:,} parameters")

    # Create dummy molecular graph
    # In practice, this would come from RDKit
    print("\n2. Creating dummy molecular data...")

    # Simple graph: 5 atoms, 4 bonds
    num_atoms = 5
    atom_features = torch.randn(num_atoms, 128)  # Dummy atom features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],  # Source nodes
        [1, 0, 2, 1, 3, 2, 4, 3],  # Target nodes
    ])
    batch = torch.zeros(num_atoms, dtype=torch.long)  # Single molecule
    edge_attr = torch.randn(edge_index.shape[1], 32)  # Dummy edge features
    pos = torch.randn(num_atoms, 3)  # Dummy 3D coordinates

    mol_graph = (atom_features, edge_index, batch, edge_attr, pos)

    # Create dummy environment
    print("\n3. Creating dummy environment...")
    env_categorical = {
        "solvent": torch.tensor([1]),  # Water
        "reaction_type": torch.tensor([0]),  # Unknown
        "atmosphere": torch.tensor([2]),  # N2
    }
    env_continuous = torch.tensor([[7.0, 25.0, 1.0, 0.1, 2.0]])  # pH, temp, pressure, conc, time

    env_features = (env_categorical, env_continuous)

    # Create dummy protein (using embedding instead of sequence)
    print("\n4. Creating dummy protein target...")
    protein_embedding = torch.randn(1, 1280)  # ESM-2 embedding dimension

    # Target properties
    print("\n5. Setting target properties...")
    p_target = torch.randn(1, 64)  # Random property target

    # Forward pass
    print("\n6. Running forward pass...")
    with torch.no_grad():
        outputs = model(
            mol_graph=mol_graph,
            env_features=env_features,
            protein_features=protein_embedding,
            p_target=p_target,
            domain="organic",
        )

    print(f"   Latent state dimensions:")
    print(f"     z_mol: {outputs['z_mol'].shape}")
    print(f"     z_rxn: {outputs['z_rxn'].shape}")
    print(f"     z_context: {outputs['z_context'].shape}")
    print(f"   Total energy: {outputs['energy'].item():.4f}")
    print(f"   Property predictions shape: {outputs['property_predictions'].shape}")

    # Imagination example (simplified)
    print("\n7. Running imagination engine (simplified)...")
    print("   Note: This requires proper initialization - see full examples")

    # The full workflow would be:
    # results = model.imagine(
    #     target_properties={"IC50": "<10nM", "bioavailability": ">50%"},
    #     protein_target="MKLRTV...",  # Protein sequence
    #     num_candidates=10,
    # )

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Prepare real molecular dataset (see data/README.md)")
    print("  2. Pre-train with phase 1 config: python train.py --config configs/phase1_jepa.yaml")
    print("  3. Fine-tune with phase 2 config: python train.py --config configs/phase2_properties.yaml")
    print("  4. Use trained model for molecular discovery")


if __name__ == "__main__":
    main()
