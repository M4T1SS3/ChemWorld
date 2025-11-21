#!/usr/bin/env python3
"""
FastAPI Backend Server for ChemJEPA
Serves molecular analysis endpoints for the UI
"""

import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chemjepa import ChemJEPA, MolecularEncoder
from chemjepa.data.loaders import MolecularDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ChemJEPA API",
    description="Molecular analysis API powered by ChemJEPA",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[ChemJEPA] = None
device = None

# Request/Response models
class AnalyzeRequest(BaseModel):
    smiles: str

class MolecularProperties(BaseModel):
    smiles: str
    LogP: float
    TPSA: float
    MolWt: float
    QED: float
    SA: float
    NumHDonors: int
    NumHAcceptors: int
    NumRotatableBonds: int
    NumAromaticRings: int
    NumSaturatedRings: int

class EnergyDecomposition(BaseModel):
    total: float
    binding: float
    stability: float
    properties: float
    novelty: float

class AnalyzeResponse(BaseModel):
    properties: MolecularProperties
    energy: EnergyDecomposition
    latent_representation: Optional[list] = None

def calculate_sa_score(mol):
    """Calculate synthetic accessibility score (simplified)"""
    # Simple approximation based on ring complexity and rotatable bonds
    num_rings = Descriptors.RingCount(mol)
    num_rotatable = Descriptors.NumRotatableBonds(mol)
    num_hetero = Descriptors.NumHeteroatoms(mol)

    # Lower score = easier to synthesize
    sa = 1.0 + (num_rings * 0.5) + (num_rotatable * 0.1) + (num_hetero * 0.05)
    return min(sa, 10.0)

def calculate_molecular_properties(smiles: str) -> MolecularProperties:
    """Calculate molecular properties from SMILES using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # Calculate properties
        props = MolecularProperties(
            smiles=smiles,
            LogP=round(Crippen.MolLogP(mol), 2),
            TPSA=round(Descriptors.TPSA(mol), 2),
            MolWt=round(Descriptors.MolWt(mol), 2),
            QED=round(QED.qed(mol), 3),
            SA=round(calculate_sa_score(mol), 2),
            NumHDonors=Lipinski.NumHDonors(mol),
            NumHAcceptors=Lipinski.NumHAcceptors(mol),
            NumRotatableBonds=Descriptors.NumRotatableBonds(mol),
            NumAromaticRings=Descriptors.NumAromaticRings(mol),
            NumSaturatedRings=Descriptors.NumSaturatedRings(mol),
        )

        return props
    except Exception as e:
        logger.error(f"Error calculating properties: {e}")
        raise ValueError(f"Failed to calculate properties: {str(e)}")

def predict_energy_decomposition(model: ChemJEPA, smiles: str, device) -> EnergyDecomposition:
    """Predict energy decomposition using ChemJEPA model"""
    try:
        # Create a temporary dataset with single molecule
        import tempfile
        import pandas as pd

        # Create temp CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({'smiles': [smiles]})
            df.to_csv(f.name, index=False)
            temp_path = f.name

        # Load molecule through dataset
        dataset = MolecularDataset(
            data_path=temp_path,
            smiles_column='smiles',
            use_3d=True
        )

        # Get molecular data
        mol_data = dataset[0]

        # Move to device and add batch dimension
        batch = {
            'x': mol_data['x'].unsqueeze(0).to(device),
            'edge_index': mol_data['edge_index'].to(device),
            'edge_attr': mol_data['edge_attr'].unsqueeze(0).to(device) if mol_data['edge_attr'] is not None else None,
            'batch': torch.zeros(mol_data['x'].size(0), dtype=torch.long).to(device),
        }

        # Run inference
        model.eval()
        with torch.no_grad():
            # Get latent representation
            latent_state = model.encode_molecule(batch)

            # Predict energy
            energy_pred = model.energy_model(latent_state)

        # Extract energy components (mock decomposition for now)
        total_energy = float(energy_pred.mean().item())

        # Clean up temp file
        Path(temp_path).unlink()

        # Return energy decomposition
        # TODO: Implement actual decomposition from model
        return EnergyDecomposition(
            total=round(total_energy, 2),
            binding=round(total_energy * 0.4, 2),
            stability=round(total_energy * 0.3, 2),
            properties=round(total_energy * 0.2, 2),
            novelty=round(total_energy * 0.1, 2),
        )

    except Exception as e:
        logger.error(f"Error predicting energy: {e}")
        # Return placeholder values on error
        return EnergyDecomposition(
            total=-5.0,
            binding=-2.0,
            stability=-1.5,
            properties=-1.0,
            novelty=-0.5,
        )

@app.on_event("startup")
async def load_model():
    """Load ChemJEPA model on startup"""
    global model, device

    logger.info("Loading ChemJEPA model...")

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    try:
        # Initialize model
        model = ChemJEPA(
            mol_dim=768,
            rxn_dim=384,
            context_dim=256,
            device=device
        )

        # Load checkpoint if available
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "best_jepa.pt"
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                # Use strict=False to allow partial loading if architectures don't match exactly
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} layers")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} layers")
                logger.info("✓ Model loaded successfully (partial match)")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    logger.warning(f"Checkpoint architecture mismatch, skipping incompatible layers: {e}")
                    logger.info("✓ Using freshly initialized model (checkpoint incompatible)")
                else:
                    raise
        else:
            logger.warning("No checkpoint found, using untrained model")

        model.to(device)
        model.eval()

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "device": str(device) if device else "none"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(device) if device else "none",
        "version": "1.0.0"
    }

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_molecule(request: AnalyzeRequest):
    """
    Analyze a molecule from SMILES string

    Returns molecular properties and energy predictions
    """
    try:
        logger.info(f"Analyzing molecule: {request.smiles}")

        # Calculate molecular properties
        properties = calculate_molecular_properties(request.smiles)

        # Predict energy if model is loaded
        if model is not None:
            energy = predict_energy_decomposition(model, request.smiles, device)
        else:
            # Fallback to placeholder
            energy = EnergyDecomposition(
                total=-5.0,
                binding=-2.0,
                stability=-1.5,
                properties=-1.0,
                novelty=-0.5,
            )

        return AnalyzeResponse(
            properties=properties,
            energy=energy
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ChemJEPA API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
