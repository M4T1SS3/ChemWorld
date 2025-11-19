"""
Chemistry utilities using RDKit.

Helper functions for:
- SMILES processing
- Molecular featurization
- 3D conformer generation
- Property calculation
"""

import torch
from typing import Optional, Tuple, List
import warnings


def smiles_to_graph(smiles: str, use_3d: bool = True) -> Tuple:
    """
    Convert SMILES string to molecular graph.

    Args:
        smiles: SMILES string
        use_3d: Whether to generate 3D coordinates

    Returns:
        Tuple of (x, edge_index, edge_attr, pos)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit is required. Install with: conda install -c conda-forge rdkit")

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens if using 3D
    if use_3d:
        mol = Chem.AddHs(mol)

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = get_atom_features(atom)
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index and features
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])

        # Bond features
        features = get_bond_features(bond)
        edge_features.append(features)
        edge_features.append(features)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # 3D coordinates
    pos = None
    if use_3d:
        pos = None
        # Try multiple random seeds for conformer generation
        for seed in [42, 123, 456, 789]:
            try:
                # Generate conformer with this seed
                result = AllChem.EmbedMolecule(mol, randomSeed=seed)
                if result == 0:  # Success
                    AllChem.MMFFOptimizeMolecule(mol)

                    conformer = mol.GetConformer()
                    pos = torch.tensor([
                        [conformer.GetAtomPosition(i).x,
                         conformer.GetAtomPosition(i).y,
                         conformer.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())
                    ], dtype=torch.float)

                    # Validate coordinates are not degenerate
                    if not torch.all(pos == 0.0):
                        # Check that positions have reasonable variance
                        pos_std = pos.std()
                        if pos_std > 0.1:  # Reasonable 3D structure
                            break
            except Exception as e:
                continue

        # If all attempts failed, add small gaussian noise instead of zeros
        if pos is None or torch.all(pos == 0.0):
            warnings.warn(f"Failed to generate valid 3D coordinates for {smiles}, using noisy positions")
            # Create random positions with small variance instead of all zeros
            pos = torch.randn(mol.GetNumAtoms(), 3) * 0.1

    return x, edge_index, edge_attr, pos


def get_atom_features(atom) -> List[float]:
    """
    Compute atom features.

    Features:
    - Atomic number (one-hot up to 100)
    - Degree (one-hot up to 6)
    - Formal charge
    - Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2)
    - Aromaticity (binary)
    - Total H count (one-hot up to 5)
    """
    from rdkit import Chem

    features = []

    # Atomic number (one-hot for common atoms, else "other")
    atomic_num = atom.GetAtomicNum()
    common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
    one_hot = [1.0 if atomic_num == x else 0.0 for x in common_atoms]
    one_hot.append(1.0 if atomic_num not in common_atoms else 0.0)  # "other"
    features.extend(one_hot)

    # Degree (one-hot)
    degree = atom.GetDegree()
    features.extend([1.0 if degree == i else 0.0 for i in range(7)])

    # Formal charge
    features.append(float(atom.GetFormalCharge()))

    # Hybridization (one-hot)
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb = atom.GetHybridization()
    features.extend([1.0 if hyb == h else 0.0 for h in hybridizations])
    features.append(1.0 if hyb not in hybridizations else 0.0)  # "other"

    # Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Total H count (one-hot)
    h_count = atom.GetTotalNumHs()
    features.extend([1.0 if h_count == i else 0.0 for i in range(6)])

    return features


def get_bond_features(bond) -> List[float]:
    """
    Compute bond features.

    Features:
    - Bond type (one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC)
    - Conjugated (binary)
    - In ring (binary)
    - Stereo (one-hot: STEREONONE, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS, STEREOANY)
    """
    from rdkit import Chem

    features = []

    # Bond type (one-hot)
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bt = bond.GetBondType()
    features.extend([1.0 if bt == t else 0.0 for t in bond_types])

    # Conjugated
    features.append(1.0 if bond.GetIsConjugated() else 0.0)

    # In ring
    features.append(1.0 if bond.IsInRing() else 0.0)

    # Stereo (one-hot)
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY,
    ]
    stereo = bond.GetStereo()
    features.extend([1.0 if stereo == s else 0.0 for s in stereo_types])

    return features


def calculate_properties(smiles: str) -> dict:
    """
    Calculate molecular properties.

    Returns dictionary with:
    - LogP
    - TPSA
    - MolWt
    - NumHDonors
    - NumHAcceptors
    - NumRotatableBonds
    - QED
    - SA_score (if available)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED
    except ImportError:
        raise ImportError("RDKit is required")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    properties = {
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolWt": Descriptors.MolWt(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "QED": QED.qed(mol),
    }

    # Try to compute SA score
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        properties["SA_score"] = sascorer.calculateScore(mol)
    except:
        pass

    return properties
