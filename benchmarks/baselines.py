"""
Baseline Methods for Molecular Optimization

Provides simple baselines to compare against counterfactual planning:
1. RandomSearch - Sample random molecules, pick best
2. GreedyOptimization - Hill climbing in latent space
3. StandardMCTS - MCTS without counterfactual branching

These establish lower bounds on performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chemjepa.models.latent import LatentState
from chemjepa.models.energy import ChemJEPAEnergyModel
from chemjepa.models.dynamics import DynamicsPredictor


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    best_state: LatentState
    best_energy: float
    oracle_calls: int
    states_explored: List[LatentState]
    energies: List[float]

    @property
    def sample_efficiency(self) -> float:
        """How quickly did we find the best state?"""
        if self.oracle_calls == 0:
            return 0.0
        return abs(self.best_energy) / self.oracle_calls


class RandomSearch:
    """
    Baseline 1: Random Search

    Simplest possible approach:
    1. Sample N random molecules
    2. Evaluate all
    3. Return best

    Pros: No assumptions, easy to implement
    Cons: Very sample-inefficient
    """

    def __init__(
        self,
        energy_model: ChemJEPAEnergyModel,
        device: torch.device = None,
    ):
        self.energy_model = energy_model
        self.device = device or torch.device('cpu')

    def optimize(
        self,
        initial_state: LatentState,
        num_samples: int = 100,
        latent_dim: int = 768,
    ) -> OptimizationResult:
        """
        Random search optimization.

        Args:
            initial_state: Starting point (ignored)
            num_samples: Number of random samples
            latent_dim: Dimension of latent space

        Returns:
            OptimizationResult with best found state
        """
        states = []
        energies = []

        # Sample random latent states
        for _ in range(num_samples):
            z_mol = torch.randn(1, latent_dim, device=self.device)

            state = LatentState(
                z_mol=z_mol,
                z_rxn=torch.randn(1, 384, device=self.device),
                z_context=torch.randn(1, 256, device=self.device),
            )

            with torch.no_grad():
                energy = self.energy_model(z_mol)['energy'].item()

            states.append(state)
            energies.append(energy)

        # Find best
        best_idx = np.argmin(energies)

        return OptimizationResult(
            best_state=states[best_idx],
            best_energy=energies[best_idx],
            oracle_calls=num_samples,
            states_explored=states,
            energies=energies,
        )


class GreedyOptimization:
    """
    Baseline 2: Greedy Hill Climbing

    Local optimization:
    1. Start from initial state
    2. Sample N neighbors
    3. Pick best neighbor
    4. Repeat until no improvement

    Pros: Sample efficient if landscape is smooth
    Cons: Gets stuck in local minima
    """

    def __init__(
        self,
        energy_model: ChemJEPAEnergyModel,
        device: torch.device = None,
    ):
        self.energy_model = energy_model
        self.device = device or torch.device('cpu')

    def optimize(
        self,
        initial_state: LatentState,
        num_steps: int = 20,
        num_neighbors: int = 10,
        step_size: float = 0.1,
    ) -> OptimizationResult:
        """
        Greedy hill climbing.

        Args:
            initial_state: Starting point
            num_steps: Max number of steps
            num_neighbors: Neighbors to sample per step
            step_size: Perturbation magnitude

        Returns:
            OptimizationResult with best found state
        """
        current_state = initial_state
        states = [initial_state]
        energies = []

        with torch.no_grad():
            current_energy = self.energy_model(current_state.z_mol)['energy'].item()
        energies.append(current_energy)

        oracle_calls = 1

        for step in range(num_steps):
            # Sample neighbors
            neighbor_states = []
            neighbor_energies = []

            for _ in range(num_neighbors):
                # Perturb current state
                noise = torch.randn_like(current_state.z_mol) * step_size

                neighbor = LatentState(
                    z_mol=current_state.z_mol + noise,
                    z_rxn=current_state.z_rxn + torch.randn_like(current_state.z_rxn) * step_size,
                    z_context=current_state.z_context + torch.randn_like(current_state.z_context) * step_size,
                )

                with torch.no_grad():
                    energy = self.energy_model(neighbor.z_mol)['energy'].item()

                neighbor_states.append(neighbor)
                neighbor_energies.append(energy)
                oracle_calls += 1

            # Pick best neighbor
            best_neighbor_idx = np.argmin(neighbor_energies)
            best_neighbor = neighbor_states[best_neighbor_idx]
            best_neighbor_energy = neighbor_energies[best_neighbor_idx]

            # Accept if better
            if best_neighbor_energy < current_energy:
                current_state = best_neighbor
                current_energy = best_neighbor_energy
                states.append(current_state)
                energies.append(current_energy)
            else:
                # No improvement, stop
                break

        # Find overall best
        best_idx = np.argmin(energies)

        return OptimizationResult(
            best_state=states[best_idx],
            best_energy=energies[best_idx],
            oracle_calls=oracle_calls,
            states_explored=states,
            energies=energies,
        )


class StandardMCTS:
    """
    Baseline 3: Standard MCTS (no counterfactuals)

    Monte Carlo Tree Search without factorization:
    1. Build search tree
    2. Expand nodes with UCB
    3. Evaluate all transitions with oracle
    4. No counterfactual branching

    Pros: Better than greedy, explores broadly
    Cons: Requires oracle for every transition (expensive!)

    This is the KEY baseline - we want to beat this with counterfactuals.
    """

    def __init__(
        self,
        energy_model: ChemJEPAEnergyModel,
        dynamics_model: DynamicsPredictor,
        device: torch.device = None,
    ):
        self.energy_model = energy_model
        self.dynamics_model = dynamics_model
        self.device = device or torch.device('cpu')

    def optimize(
        self,
        initial_state: LatentState,
        num_iterations: int = 50,
        num_actions_per_state: int = 5,
        horizon: int = 3,
    ) -> OptimizationResult:
        """
        Standard MCTS optimization.

        Args:
            initial_state: Starting state
            num_iterations: Number of MCTS iterations
            num_actions_per_state: Actions to try per state
            horizon: Planning horizon

        Returns:
            OptimizationResult with best found state
        """
        states_explored = [initial_state]
        energies = []

        with torch.no_grad():
            initial_energy = self.energy_model(initial_state.z_mol)['energy'].item()
        energies.append(initial_energy)

        oracle_calls = 1
        best_state = initial_state
        best_energy = initial_energy

        # Simple beam search (simplified MCTS)
        current_beam = [initial_state]

        for iteration in range(num_iterations):
            next_beam = []
            beam_energies = []

            for state in current_beam:
                # Try multiple actions
                for _ in range(num_actions_per_state):
                    # Sample random action
                    action = torch.randn(
                        self.dynamics_model.action_dim,
                        device=self.device
                    )

                    # Predict next state (ORACLE CALL!)
                    with torch.no_grad():
                        prediction = self.dynamics_model.forward(
                            state,
                            action.unsqueeze(0),
                            quantize_action=True,
                            predict_uncertainty=False,
                        )

                    next_state = prediction['next_state']

                    # Evaluate (ORACLE CALL!)
                    with torch.no_grad():
                        energy = self.energy_model(next_state.z_mol)['energy'].item()

                    oracle_calls += 2  # Dynamics + energy

                    next_beam.append(next_state)
                    beam_energies.append(energy)

                    states_explored.append(next_state)
                    energies.append(energy)

                    # Track best
                    if energy < best_energy:
                        best_state = next_state
                        best_energy = energy

            # Keep top-k states for next iteration
            if len(next_beam) > 10:
                top_indices = np.argsort(beam_energies)[:10]
                current_beam = [next_beam[i] for i in top_indices]
            else:
                current_beam = next_beam

            # Early stopping if no improvement
            if iteration > 10 and len(energies) > 10:
                recent_improvement = min(energies[-10:]) - best_energy
                if recent_improvement > -0.01:
                    break

        return OptimizationResult(
            best_state=best_state,
            best_energy=best_energy,
            oracle_calls=oracle_calls,
            states_explored=states_explored,
            energies=energies,
        )


def compare_methods(
    initial_state: LatentState,
    energy_model: ChemJEPAEnergyModel,
    dynamics_model: Optional[DynamicsPredictor] = None,
    oracle_budget: int = 100,
) -> Dict[str, OptimizationResult]:
    """
    Compare all baseline methods with same oracle budget.

    Args:
        initial_state: Starting point
        energy_model: Energy function
        dynamics_model: Dynamics model (for MCTS)
        oracle_budget: Max oracle calls per method

    Returns:
        Dictionary mapping method name to result
    """
    results = {}

    # Random search
    random_search = RandomSearch(energy_model)
    results['random'] = random_search.optimize(
        initial_state,
        num_samples=oracle_budget,
    )

    # Greedy
    greedy = GreedyOptimization(energy_model)
    results['greedy'] = greedy.optimize(
        initial_state,
        num_steps=oracle_budget // 10,
        num_neighbors=10,
    )

    # Standard MCTS (if dynamics available)
    if dynamics_model is not None:
        mcts = StandardMCTS(energy_model, dynamics_model)
        results['mcts'] = mcts.optimize(
            initial_state,
            num_iterations=oracle_budget // 10,
            num_actions_per_state=5,
        )

    return results
