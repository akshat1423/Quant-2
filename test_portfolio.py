"""
Test script for Advanced Portfolio Optimization components
"""

import numpy as np
from portfolio_optimization import (
    MultiArmedBandit, CVaROptimizer, MarkovDecisionProcess,
    MarkovModulatedModel, PartiallyObservableMDP, AdvancedPortfolioOptimizer
)

def test_multi_armed_bandit():
    """Test Multi-Armed Bandit implementation"""
    print("Testing Multi-Armed Bandit (UCB)...")
    
    mab = MultiArmedBandit(n_arms=3, confidence_level=2.0)
    
    # Simulate some rounds
    for i in range(10):
        arm = mab.select_arm()
        reward = np.random.normal(0.05, 0.1)  # Simulate return
        mab.update(arm, reward)
    
    mean_regret = mab.get_mean_regret()
    print(f"  ✓ MAB test completed. Mean regret: {mean_regret:.4f}")
    
    return mean_regret < 1.0  # Should be reasonable

def test_cvar_optimizer():
    """Test CVaR optimizer"""
    print("Testing CVaR Optimizer...")
    
    cvar_opt = CVaROptimizer(confidence_level=0.05)
    
    # Generate synthetic returns
    returns = np.random.normal(0.01, 0.05, (100, 3))
    expected_returns = np.mean(returns, axis=0)
    
    # Test CVaR calculation
    portfolio_returns = np.sum(returns * [0.33, 0.33, 0.34], axis=1)
    cvar = cvar_opt.calculate_cvar(portfolio_returns)
    
    # Test optimization
    weights = cvar_opt.optimize_portfolio_cvar(expected_returns, returns)
    
    print(f"  ✓ CVaR test completed. CVaR: {cvar:.4f}, Weights sum: {np.sum(weights):.3f}")
    
    return abs(np.sum(weights) - 1.0) < 0.01  # Weights should sum to 1

def test_markov_decision_process():
    """Test MDP implementation"""
    print("Testing Markov Decision Process...")
    
    mdp = MarkovDecisionProcess(n_states=3, n_actions=2, discount_factor=0.9)
    
    # Run value iteration
    mdp.value_iteration(max_iterations=100)
    
    # Test action selection
    action = mdp.get_optimal_action(state=0)
    
    # Test model update
    mdp.update_model(state=0, action=0, next_state=1, reward=0.1)
    
    print(f"  ✓ MDP test completed. Optimal action for state 0: {action}")
    
    return 0 <= action < mdp.n_actions

def test_markov_modulated_model():
    """Test Markov-modulated models"""
    print("Testing Markov-Modulated Models...")
    
    # Test observable model
    obs_model = MarkovModulatedModel(n_regimes=3, observable=True)
    obs_model.simulate_regime_change()
    mean, vol = obs_model.get_current_parameters()
    
    # Test unobservable model
    hidden_model = MarkovModulatedModel(n_regimes=3, observable=False)
    hidden_model.update_beliefs(observation=2)
    hidden_mean, hidden_vol = hidden_model.get_current_parameters()
    
    print(f"  ✓ Markov models test completed. Observable: μ={mean:.4f}, σ={vol:.4f}")
    
    return vol > 0 and hidden_vol > 0

def test_pomdp():
    """Test Partially Observable MDP"""
    print("Testing Partially Observable MDP...")
    
    pomdp = PartiallyObservableMDP(n_states=3, n_actions=2, n_observations=2)
    
    # Test belief update
    pomdp.update_belief(action=0, observation=1)
    
    # Test action selection
    action = pomdp.select_action()
    
    # Test jump process modeling
    market_data = np.random.normal(0, 0.02, 50)
    market_data[10] += 0.05  # Add a jump
    jumps = pomdp.model_jump_process(market_data)
    
    print(f"  ✓ POMDP test completed. Selected action: {action}, Jumps detected: {np.sum(jumps)}")
    
    return 0 <= action < pomdp.n_actions

def test_advanced_portfolio_optimizer():
    """Test the main portfolio optimizer"""
    print("Testing Advanced Portfolio Optimizer...")
    
    optimizer = AdvancedPortfolioOptimizer(n_assets=3, target_regret=0.4)
    
    # Generate synthetic data
    data = optimizer.generate_synthetic_data(n_periods=50)
    
    # Run optimization
    results = optimizer.optimize_portfolio(data, rebalance_frequency=10)
    
    # Check results
    checks = [
        results['total_return'] is not None,
        results['volatility'] > 0,
        len(results['portfolio_returns']) == 50,
        results['mean_regret'] >= 0
    ]
    
    print(f"  ✓ Portfolio optimizer test completed.")
    print(f"    - Total return: {results['total_return']:.2%}")
    print(f"    - Volatility: {results['volatility']:.2%}")
    print(f"    - Mean regret: {results['mean_regret']:.4f}")
    print(f"    - Target regret achieved: {results['mean_regret'] <= 0.4}")
    
    return all(checks)

def run_all_tests():
    """Run all component tests"""
    print("Advanced Portfolio Optimization - Component Tests")
    print("=" * 60)
    
    tests = [
        ("Multi-Armed Bandit", test_multi_armed_bandit),
        ("CVaR Optimizer", test_cvar_optimizer),
        ("Markov Decision Process", test_markov_decision_process),
        ("Markov-Modulated Models", test_markov_modulated_model),
        ("Partially Observable MDP", test_pomdp),
        ("Advanced Portfolio Optimizer", test_advanced_portfolio_optimizer)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "PASS" if result else "FAIL"
            print(f"  {status}")
        except Exception as e:
            print(f"  FAIL - Error: {e}")
            results.append(False)
        print()
    
    print("Test Summary:")
    print("-" * 30)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Portfolio optimization system is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()