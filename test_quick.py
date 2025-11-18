#!/usr/bin/env python3
"""
Quick Test for ChemJEPA Installation
Runs in ~30 seconds to verify everything works.
"""

import sys
import torch
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("ChemJEPA Quick Test")
print("=" * 80)

# Test 1: Check device
print("\n[1/5] Checking device...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✓ Using MPS (Apple GPU acceleration)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print(f"✓ Using CPU")

# Test 2: Import ChemJEPA
print("\n[2/5] Importing ChemJEPA...")
try:
    from chemjepa import ChemJEPA
    from chemjepa.data.loaders import MolecularDataset, collate_molecular_batch
    from chemjepa.training.trainer import ChemJEPATrainer
    from chemjepa.training.objectives import ChemJEPALoss
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Load dummy data
print("\n[3/5] Loading dummy dataset...")
data_dir = Path(__file__).parent / "data" / "dummy"
train_csv = data_dir / "train.csv"
val_csv = data_dir / "val.csv"

if not train_csv.exists():
    print(f"✗ Data not found at {train_csv}")
    print("Run: python3 scripts/prepare_data.py")
    sys.exit(1)

try:
    train_dataset = MolecularDataset(
        data_path=str(train_csv),
        smiles_column='smiles',
        use_3d=True
    )
    val_dataset = MolecularDataset(
        data_path=str(val_csv),
        smiles_column='smiles',
        use_3d=True
    )
    print(f"✓ Loaded {len(train_dataset)} train, {len(val_dataset)} val molecules")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 4: Initialize model
print("\n[4/5] Initializing model...")
try:
    model = ChemJEPA(
        mol_dim=768,
        rxn_dim=384,
        context_dim=256,
        device=device
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {total_params:,} parameters")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

# Test 5: Quick training test
print("\n[5/5] Running quick training test...")
try:
    from torch.utils.data import DataLoader

    # Create tiny loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_molecular_batch
    )

    # Initialize training components
    criterion = ChemJEPALoss(
        lambda_jepa=1.0,
        lambda_energy=0.5,
        lambda_consistency=0.3
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    trainer = ChemJEPATrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        log_dir="checkpoints/test",
        use_wandb=False
    )

    print("  Running 1 epoch...")

    # Run 1 epoch
    trainer.train(
        train_loader=train_loader,
        val_loader=train_loader,  # Use train as val for quick test
        num_epochs=1,
        phase="jepa"
    )

    print("✓ Training test passed")

except Exception as e:
    print(f"✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print("\nNext steps:")
print("  1. Run production training: python3 train_production.py")
print("  2. Or train on your own data - see README.md")
print("\nChemJEPA is ready! 🎉")
print("=" * 80)
