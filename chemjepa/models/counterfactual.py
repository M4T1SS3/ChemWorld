"""
Counterfactual Planning for Molecular Discovery

Core innovation: Use factored dynamics to answer counterfactual questions
without expensive oracle queries.

Key Idea:
---------
Standard approach:
    - Want to test reaction R at conditions C1, C2, C3
    - Need 3 oracle queries (expensive!)

Our approach:
    - Factorization: Δz = Δz_rxn(R) + Δz_env(C)
    - Compute Δz_rxn once (oracle)
    - Reuse for different C1, C2, C3 (cheap!)
    - 3x fewer oracle queries

This enables sample-efficient multi-objective optimization.

Example:
--------
>>> cf_planner = CounterfactualPlanner(dynamics_model, energy_model)
>>>
>>> # Factual: Run reaction at pH 7
>>> factual_state = cf_planner.rollout(state, action, conditions={'pH': 7})
>>>
>>> # Counterfactual: "What if pH 3 instead?" (no oracle needed!)
>>> cf_state = cf_planner.counterfactual_rollout(
>>>     state, action,
>>>     factual_conditions={'pH': 7},
>>>     counterfactual_conditions={'pH': 3}
>>> )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .latent import LatentState
from .dynamics import DynamicsPredictor
from .energy import ChemJEPAEnergyModel


@dataclass
class CounterfactualResult:
    """Result of counterfactual prediction."""
    factual_state: LatentState
    counterfactual_state: LatentState
    factual_energy: float
    counterfactual_energy: float
    oracle_calls: int  # How many oracle calls were made

    @property
    def energy_delta(self) -> float:
        """Energy difference: counterfactual - factual."""
        return self.counterfactual_energy - self.factual_energy

    @property
    def is_better(self) -> bool:
        """Is counterfactual better than factual? (lower energy)"""
        return self.counterfactual_energy < self.factual_energy


class CounterfactualPlanner:
    """
    Counterfactual planning using factored dynamics.

    Enables answering "what if" questions without oracle queries:
    - "What if different reaction conditions?"
    - "What if different catalyst?"
    - "What if optimized for stability instead of binding?"

    Key advantage: Reuse reaction computation across different conditions.
    """

    def __init__(
        self,
        dynamics_model: DynamicsPredictor,
        energy_model: ChemJEPAEnergyModel,
        device: torch.device = None,
    ):
        """
        Args:
            dynamics_model: Factored dynamics predictor
            energy_model: Energy scoring function
            device: Compute device
        """
        self.dynamics_model = dynamics_model
        self.energy_model = energy_model
        self.device = device or torch.device('cpu')

        # Statistics
        self.oracle_calls = 0
        self.counterfactual_calls = 0

    def compute_reaction_delta(
        self,
        state: LatentState,
        action: torch.Tensor,
    ) -> LatentState:
        """
        Compute reaction-only change: Δz_rxn(state, action).

        This is the EXPENSIVE part (requires oracle or dynamics model).
        We want to compute this ONCE and reuse.

        Args:
            state: Current state
            action: Reaction to apply

        Returns:
            Δz_rxn: Change due to reaction only
        """
        # Use dynamics model's forward method to predict next state
        with torch.no_grad():
            # Add batch dimension if needed
            if action.dim() == 1:
                action = action.unsqueeze(0)

            # Predict next state
            prediction = self.dynamics_model.forward(
                state,
                action,
                quantize_action=True,
                predict_uncertainty=False,
            )

            next_state = prediction['next_state']

            # Extract reaction delta (difference from current state)
            delta_rxn = LatentState(
                z_mol=next_state.z_mol - state.z_mol,
                z_rxn=next_state.z_rxn - state.z_rxn,
                z_context=next_state.z_context - state.z_context,
            )

        self.oracle_calls += 1
        return delta_rxn

    def compute_environment_delta(
        self,
        state: LatentState,
        conditions: Dict[str, float],
    ) -> LatentState:
        """
        Compute environment-only change: Δz_env(conditions).

        This is CHEAP - just encode conditions to latent.
        No oracle needed!

        Args:
            state: Current state
            conditions: Environmental conditions (pH, temp, solvent, etc.)

        Returns:
            Δz_env: Change due to environment only
        """
        # Encode conditions to context embedding
        # For now, simple encoding (can be learned)
        context_features = []

        # Standard conditions with defaults
        ph = conditions.get('pH', 7.0)
        temp = conditions.get('temp', 298.0)
        pressure = conditions.get('pressure', 1.0)

        # Normalize to ~[-1, 1] range
        ph_norm = (ph - 7.0) / 7.0  # pH 0-14
        temp_norm = (temp - 298.0) / 100.0  # ~200-400K
        pressure_norm = (pressure - 1.0) / 10.0  # ~0-10 atm

        # Create simple context vector
        # In production, this would be learned
        context_vec = torch.tensor(
            [ph_norm, temp_norm, pressure_norm],
            device=state.z_context.device,
            dtype=state.z_context.dtype,
        )

        # Expand to full context dimension (256)
        # Simple approach: repeat + add noise for diversity
        context_full = context_vec.repeat(256 // 3 + 1)[:256]

        # Create delta (for now, just use the context directly)
        # More sophisticated: learn mapping from conditions → Δz_env
        delta_env = LatentState(
            z_mol=torch.zeros_like(state.z_mol),
            z_rxn=torch.zeros_like(state.z_rxn),
            z_context=context_full.unsqueeze(0) if context_full.dim() == 1 else context_full,
        )

        return delta_env

    def counterfactual_rollout(
        self,
        state: LatentState,
        action: torch.Tensor,
        factual_conditions: Dict[str, float],
        counterfactual_conditions: Dict[str, float],
    ) -> CounterfactualResult:
        """
        Predict both factual and counterfactual outcomes.

        Key insight: Δz_rxn is SAME for both, only Δz_env differs.
        This means we only call oracle ONCE instead of TWICE!

        Args:
            state: Initial state
            action: Reaction to apply
            factual_conditions: Actual conditions
            counterfactual_conditions: "What if" conditions

        Returns:
            CounterfactualResult with both outcomes
        """
        # Step 1: Compute reaction delta (EXPENSIVE - oracle call)
        delta_rxn = self.compute_reaction_delta(state, action)

        # Step 2: Compute environment deltas (CHEAP - no oracle)
        delta_env_factual = self.compute_environment_delta(state, factual_conditions)
        delta_env_cf = self.compute_environment_delta(state, counterfactual_conditions)

        # Step 3: Combine to get final states
        # Factorization: z_next = z_current + Δz_rxn + Δz_env

        factual_state = LatentState(
            z_mol=state.z_mol + delta_rxn.z_mol + delta_env_factual.z_mol,
            z_rxn=state.z_rxn + delta_rxn.z_rxn + delta_env_factual.z_rxn,
            z_context=state.z_context + delta_rxn.z_context + delta_env_factual.z_context,
        )

        cf_state = LatentState(
            z_mol=state.z_mol + delta_rxn.z_mol + delta_env_cf.z_mol,
            z_rxn=state.z_rxn + delta_rxn.z_rxn + delta_env_cf.z_rxn,
            z_context=state.z_context + delta_rxn.z_context + delta_env_cf.z_context,
        )

        # Step 4: Evaluate energies
        with torch.no_grad():
            factual_energy = self.energy_model(factual_state.z_mol)['energy'].item()
            cf_energy = self.energy_model(cf_state.z_mol)['energy'].item()

        self.counterfactual_calls += 1

        return CounterfactualResult(
            factual_state=factual_state,
            counterfactual_state=cf_state,
            factual_energy=factual_energy,
            counterfactual_energy=cf_energy,
            oracle_calls=1,  # Only 1 oracle call for 2 predictions!
        )

    def multi_counterfactual_rollout(
        self,
        state: LatentState,
        action: torch.Tensor,
        factual_conditions: Dict[str, float],
        counterfactual_conditions_list: List[Dict[str, float]],
    ) -> List[CounterfactualResult]:
        """
        Predict multiple counterfactuals efficiently.

        Example: Test reaction at pH 3, 5, 7, 9, 11
        Standard: 5 oracle calls
        Our method: 1 oracle call (reuse Δz_rxn)
        Speedup: 5x

        Args:
            state: Initial state
            action: Reaction to apply
            factual_conditions: Actual conditions
            counterfactual_conditions_list: List of "what if" conditions

        Returns:
            List of CounterfactualResults
        """
        # Compute reaction delta ONCE
        delta_rxn = self.compute_reaction_delta(state, action)

        results = []
        delta_env_factual = self.compute_environment_delta(state, factual_conditions)

        # Compute factual state
        factual_state = LatentState(
            z_mol=state.z_mol + delta_rxn.z_mol + delta_env_factual.z_mol,
            z_rxn=state.z_rxn + delta_rxn.z_rxn + delta_env_factual.z_rxn,
            z_context=state.z_context + delta_rxn.z_context + delta_env_factual.z_context,
        )

        with torch.no_grad():
            factual_energy = self.energy_model(factual_state.z_mol)['energy'].item()

        # Compute all counterfactuals (reusing delta_rxn!)
        for cf_conditions in counterfactual_conditions_list:
            delta_env_cf = self.compute_environment_delta(state, cf_conditions)

            cf_state = LatentState(
                z_mol=state.z_mol + delta_rxn.z_mol + delta_env_cf.z_mol,
                z_rxn=state.z_rxn + delta_rxn.z_rxn + delta_env_cf.z_rxn,
                z_context=state.z_context + delta_rxn.z_context + delta_env_cf.z_context,
            )

            with torch.no_grad():
                cf_energy = self.energy_model(cf_state.z_mol)['energy'].item()

            results.append(CounterfactualResult(
                factual_state=factual_state,
                counterfactual_state=cf_state,
                factual_energy=factual_energy,
                counterfactual_energy=cf_energy,
                oracle_calls=1 // len(counterfactual_conditions_list),  # Amortized
            ))

        self.counterfactual_calls += len(counterfactual_conditions_list)
        return results

    def get_statistics(self) -> Dict[str, int]:
        """
        Get oracle call statistics.

        Returns:
            Dictionary with oracle_calls, counterfactual_calls, and speedup
        """
        # Without counterfactuals, would need oracle call for each prediction
        oracle_calls_without_cf = self.oracle_calls + self.counterfactual_calls
        speedup = oracle_calls_without_cf / max(self.oracle_calls, 1)

        return {
            'oracle_calls': self.oracle_calls,
            'counterfactual_predictions': self.counterfactual_calls,
            'baseline_oracle_calls': oracle_calls_without_cf,
            'speedup': speedup,
        }

    def reset_statistics(self):
        """Reset oracle call counters."""
        self.oracle_calls = 0
        self.counterfactual_calls = 0
