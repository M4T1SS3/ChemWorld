"""
Script to download and prepare training datasets.

Downloads:
1. ZINC250k subset (for quick testing)
2. QM9 (for property prediction)
3. USPTO-50k (for reaction prediction)
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import zipfile


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_zinc250k(data_dir: Path):
    """
    Download and prepare ZINC250k dataset.

    ZINC250k: 250,000 drug-like molecules from ZINC database.
    """
    print("\n" + "="*80)
    print("Preparing ZINC250k dataset")
    print("="*80)

    zinc_dir = data_dir / "zinc250k"
    zinc_dir.mkdir(parents=True, exist_ok=True)

    # Download
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    output_file = zinc_dir / "zinc250k.csv"

    if output_file.exists():
        print(f"✓ ZINC250k already downloaded: {output_file}")
    else:
        print(f"Downloading ZINC250k...")
        download_url(url, output_file)
        print(f"✓ Downloaded to {output_file}")

    # Load and inspect
    df = pd.read_csv(output_file)
    print(f"\nDataset info:")
    print(f"  Total molecules: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Create train/val/test splits
    print("\nCreating train/val/test splits (80/10/10)...")
    n = len(df)
    indices = list(range(n))

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(indices)

    # Split
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    df.iloc[train_indices].to_csv(zinc_dir / "train.csv", index=False)
    df.iloc[val_indices].to_csv(zinc_dir / "val.csv", index=False)
    df.iloc[test_indices].to_csv(zinc_dir / "test.csv", index=False)

    print(f"  Train: {len(train_indices)} molecules")
    print(f"  Val: {len(val_indices)} molecules")
    print(f"  Test: {len(test_indices)} molecules")
    print(f"✓ Splits saved to {zinc_dir}")


def prepare_qm9(data_dir: Path):
    """
    Download and prepare QM9 dataset.

    QM9: ~134k molecules with quantum mechanical properties.
    """
    print("\n" + "="*80)
    print("Preparing QM9 dataset")
    print("="*80)

    qm9_dir = data_dir / "qm9"
    qm9_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use PyTorch Geometric's QM9 loader
        from torch_geometric.datasets import QM9

        print("Downloading QM9 via PyTorch Geometric...")
        dataset = QM9(root=str(qm9_dir))

        print(f"\nDataset info:")
        print(f"  Total molecules: {len(dataset)}")
        print(f"  Example data: {dataset[0]}")

        # Convert to CSV for easier use
        print("\nConverting to CSV format...")
        data_list = []

        for i in tqdm(range(min(len(dataset), 10000))):  # Limit to 10k for quick demo
            data = dataset[i]

            # Extract SMILES (if available)
            # Note: QM9 doesn't directly provide SMILES, would need RDKit conversion
            # For now, just save indices

            row = {
                "idx": i,
                "num_atoms": data.num_nodes,
                "num_bonds": data.num_edges // 2,
            }

            # Add target properties
            target_names = [
                "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
                "U0", "U", "H", "G", "Cv"
            ]
            for j, name in enumerate(target_names):
                if j < data.y.shape[1]:
                    row[name] = data.y[0, j].item()

            data_list.append(row)

        df = pd.DataFrame(data_list)
        df.to_csv(qm9_dir / "qm9_sample.csv", index=False)

        print(f"✓ Saved {len(df)} molecules to {qm9_dir / 'qm9_sample.csv'}")

    except ImportError:
        print("⚠️  PyTorch Geometric not installed, skipping QM9")
        print("   Install with: pip install torch-geometric")


def create_dummy_dataset(data_dir: Path):
    """
    Create a small dummy dataset for quick testing.
    """
    print("\n" + "="*80)
    print("Creating dummy dataset for testing")
    print("="*80)

    dummy_dir = data_dir / "dummy"
    dummy_dir.mkdir(parents=True, exist_ok=True)

    # Some example drug-like molecules
    smiles_list = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)NCC(COc1ccccc1)O",  # Propranolol
        "c1ccc2c(c1)c(c[nH]2)CCN",  # Serotonin
        "CC(C)(C)NCC(c1ccc(c(c1)CO)O)O",  # Salbutamol
        "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",  # Morphine
        "COc1cc2c(cc1OC)C(C(=O)OC)C(c3ccc4c(c3)OCO4)CC2=O",  # Podophyllotoxin
        "C1CC1NC(=O)C2=C(C=CC(=C2)Cl)Cl",  # Ciprofibrate base structure
        "CC(C)c1ccc(cc1)C(C)C(=O)O",  # Related to ibuprofen
    ]

    # Generate dummy properties
    import random
    random.seed(42)

    data = []
    for smiles in smiles_list:
        data.append({
            "smiles": smiles,
            "LogP": random.uniform(0, 5),
            "TPSA": random.uniform(20, 140),
            "MolWt": random.uniform(150, 500),
            "QED": random.uniform(0, 1),
            "SA_score": random.uniform(1, 10),
        })

    df = pd.DataFrame(data)

    # Create train/val/test
    df.iloc[:6].to_csv(dummy_dir / "train.csv", index=False)
    df.iloc[6:8].to_csv(dummy_dir / "val.csv", index=False)
    df.iloc[8:].to_csv(dummy_dir / "test.csv", index=False)

    print(f"✓ Created dummy dataset with {len(df)} molecules")
    print(f"  Train: 6, Val: 2, Test: 2")
    print(f"  Saved to {dummy_dir}")


def main():
    """Main data preparation script."""
    # Setup directories
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print("ChemJEPA Data Preparation")
    print("="*80)
    print(f"Data directory: {data_dir}")

    # Always create dummy dataset
    create_dummy_dataset(data_dir)

    # Ask user what to download
    print("\n" + "="*80)
    print("Which datasets would you like to download?")
    print("="*80)
    print("1. Dummy dataset only (for quick testing) - DONE")
    print("2. ZINC250k (~250k molecules, ~50MB)")
    print("3. QM9 (~134k molecules with properties)")
    print("4. All of the above")
    print("5. Skip downloads")

    choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"

    if choice in ["2", "4"]:
        prepare_zinc250k(data_dir)

    if choice in ["3", "4"]:
        prepare_qm9(data_dir)

    print("\n" + "="*80)
    print("✓ Data preparation complete!")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print("\nNext steps:")
    print("  1. Review the datasets in the data/ directory")
    print("  2. Update configs to point to your chosen dataset")
    print("  3. Run training: python train.py --config configs/phase1_jepa.yaml")


if __name__ == "__main__":
    main()
