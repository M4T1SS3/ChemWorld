#!/usr/bin/env python3
"""
Quick test to verify counterfactual planning works.

Tests:
1. Counterfactual module loads
2. Baselines run
3. Count oracle calls correctly
"""

import torch
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chemjepa.models.latent import LatentState
from chemjepa.models.energy import ChemJEPAEnergyModel
from chemjepa.models.dynamics import DynamicsPredictor
from chemjepa.models.counterfactual import CounterfactualPlanner
from benchmarks.baselines import RandomSearch, GreedyOptimization, StandardMCTS


def test_counterfactual_planner():
    """Test counterfactual planning module."""
    print("=" * 60)
    print("Testing Counterfactual Planner")
    print("=" * 60)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create simple models for testing
    energy_model = ChemJEPAEnergyModel(
        mol_dim=768,
        hidden_dim=512,
        num_properties=5,
    ).to(device)

    dynamics_model = DynamicsPredictor(
        mol_dim=768,
        rxn_dim=384,
        context_dim=256,
        num_reactions=1000,
        action_dim=256,
    ).to(device)

    # Create counterfactual planner
    cf_planner = CounterfactualPlanner(dynamics_model, energy_model, device)

    # Create test state
    state = LatentState(
        z_mol=torch.randn(1, 768, device=device),
        z_rxn=torch.randn(1, 384, device=device),
        z_context=torch.randn(1, 256, device=device),
    )

    action = torch.randn(256, device=device)

    # Test single counterfactual
    print("Test 1: Single counterfactual rollout")
    result = cf_planner.counterfactual_rollout(
        state,
        action,
        factual_conditions={'pH': 7.0, 'temp': 298.0},
        counterfactual_conditions={'pH': 3.0, 'temp': 350.0},
    )

    print(f"✓ Factual energy: {result.factual_energy:.4f}")
    print(f"✓ Counterfactual energy: {result.counterfactual_energy:.4f}")
    print(f"✓ Energy delta: {result.energy_delta:.4f}")
    print(f"✓ Oracle calls: {result.oracle_calls}")
    print(f"✓ Is better: {result.is_better}\n")

    # Test multiple counterfactuals
    print("Test 2: Multiple counterfactual rollouts")
    cf_conditions_list = [
        {'pH': 3.0, 'temp': 298.0},
        {'pH': 5.0, 'temp': 298.0},
        {'pH': 9.0, 'temp': 298.0},
        {'pH': 11.0, 'temp': 298.0},
    ]

    results = cf_planner.multi_counterfactual_rollout(
        state,
        action,
        factual_conditions={'pH': 7.0, 'temp': 298.0},
        counterfactual_conditions_list=cf_conditions_list,
    )

    print(f"✓ Tested {len(results)} conditions")
    for i, res in enumerate(results):
        pH = cf_conditions_list[i]['pH']
        print(f"  pH={pH}: energy={res.counterfactual_energy:.4f}")

    # Test statistics
    print("\nTest 3: Oracle call statistics")
    stats = cf_planner.get_statistics()
    print(f"✓ Oracle calls: {stats['oracle_calls']}")
    print(f"✓ Counterfactual predictions: {stats['counterfactual_predictions']}")
    print(f"✓ Baseline oracle calls: {stats['baseline_oracle_calls']}")
    print(f"✓ Speedup: {stats['speedup']:.2f}x")

    print("\n✅ All counterfactual planner tests passed!\n")


def test_baselines():
    """Test baseline methods."""
    print("=" * 60)
    print("Testing Baseline Methods")
    print("=" * 60)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    energy_model = ChemJEPAEnergyModel(
        mol_dim=768,
        hidden_dim=512,
        num_properties=5,
    ).to(device)

    dynamics_model = DynamicsPredictor(
        mol_dim=768,
        rxn_dim=384,
        context_dim=256,
        num_reactions=1000,
        action_dim=256,
    ).to(device)

    initial_state = LatentState(
        z_mol=torch.randn(1, 768, device=device),
        z_rxn=torch.randn(1, 384, device=device),
        z_context=torch.randn(1, 256, device=device),
    )

    # Test Random Search
    print("\nTest 1: Random Search")
    random_search = RandomSearch(energy_model, device)
    result_random = random_search.optimize(initial_state, num_samples=20)
    print(f"✓ Best energy: {result_random.best_energy:.4f}")
    print(f"✓ Oracle calls: {result_random.oracle_calls}")
    print(f"✓ Sample efficiency: {result_random.sample_efficiency:.6f}")

    # Test Greedy
    print("\nTest 2: Greedy Optimization")
    greedy = GreedyOptimization(energy_model, device)
    result_greedy = greedy.optimize(initial_state, num_steps=5, num_neighbors=5)
    print(f"✓ Best energy: {result_greedy.best_energy:.4f}")
    print(f"✓ Oracle calls: {result_greedy.oracle_calls}")
    print(f"✓ Sample efficiency: {result_greedy.sample_efficiency:.6f}")

    # Test Standard MCTS
    print("\nTest 3: Standard MCTS")
    mcts = StandardMCTS(energy_model, dynamics_model, device)
    result_mcts = mcts.optimize(initial_state, num_iterations=5, num_actions_per_state=3)
    print(f"✓ Best energy: {result_mcts.best_energy:.4f}")
    print(f"✓ Oracle calls: {result_mcts.oracle_calls}")
    print(f"✓ Sample efficiency: {result_mcts.sample_efficiency:.6f}")

    print("\n✅ All baseline tests passed!\n")


def main():
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL PLANNING TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_counterfactual_planner()
        test_baselines()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run full benchmark: python benchmarks/multi_objective_qm9.py")
        print("  2. Compare methods: Check sample efficiency")
        print("  3. Generate figures: python scripts/generate_paper_figures.py")
        print()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
