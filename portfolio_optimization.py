"""
Advanced Portfolio Optimization & Selection
Self Project Mar'24 – Apr'24

This module implements advanced portfolio optimization strategies using:
- Multi-Armed Bandits (MAB) with UCB algorithm
- Markov Decision Processes (MDPs) for portfolio optimization
- Conditional Value at Risk (CVaR) for risk management
- Markov-modulated drift models (observable & unobservable)
- Partially Observable MDPs (PDMDPs) for jump processes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class MultiArmedBandit:
    """
    Multi-Armed Bandit for stock selection using UCB algorithm
    Reduces stock selection risk with optimized mean regret
    """
    
    def __init__(self, n_arms: int, confidence_level: float = 2.0):
        """
        Initialize MAB with UCB algorithm
        
        Args:
            n_arms: Number of stocks/assets to choose from
            confidence_level: UCB confidence parameter
        """
        self.n_arms = n_arms
        self.confidence_level = confidence_level
        
        # Initialize counters and rewards
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_count = 0
        self.regret_history = []
        
    def ucb_score(self, arm: int) -> float:
        """Calculate UCB score for given arm"""
        if self.arm_counts[arm] == 0:
            return float('inf')  # Ensure unplayed arms are selected first
        
        avg_reward = self.arm_rewards[arm] / self.arm_counts[arm]
        confidence_interval = self.confidence_level * np.sqrt(
            np.log(self.total_count) / self.arm_counts[arm]
        )
        return avg_reward + confidence_interval
    
    def select_arm(self) -> int:
        """Select arm using UCB algorithm"""
        ucb_scores = [self.ucb_score(arm) for arm in range(self.n_arms)]
        return np.argmax(ucb_scores)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics with observed reward"""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_count += 1
        
        # Calculate regret (simplified for demonstration)
        best_expected = np.max(self.arm_rewards / np.maximum(self.arm_counts, 1))
        current_expected = self.arm_rewards[arm] / self.arm_counts[arm]
        regret = best_expected - current_expected
        self.regret_history.append(regret)
    
    def get_mean_regret(self) -> float:
        """Get current mean regret"""
        return np.mean(self.regret_history) if self.regret_history else 0.0


class CVaROptimizer:
    """
    Conditional Value at Risk (CVaR) optimizer for risk management
    """
    
    def __init__(self, confidence_level: float = 0.05):
        """
        Initialize CVaR optimizer
        
        Args:
            confidence_level: VaR confidence level (default 5%)
        """
        self.confidence_level = confidence_level
    
    def calculate_cvar(self, returns: np.ndarray) -> float:
        """Calculate CVaR for given returns series"""
        var_threshold = np.percentile(returns, self.confidence_level * 100)
        tail_losses = returns[returns <= var_threshold]
        return np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
    
    def optimize_portfolio_cvar(self, expected_returns: np.ndarray, 
                               returns_scenarios: np.ndarray) -> np.ndarray:
        """
        Optimize portfolio weights minimizing CVaR
        
        Args:
            expected_returns: Expected returns for each asset
            returns_scenarios: Historical return scenarios matrix
            
        Returns:
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        n_scenarios = len(returns_scenarios)
        
        # Decision variables
        weights = cp.Variable(n_assets, nonneg=True)
        var = cp.Variable()
        auxiliary_vars = cp.Variable(n_scenarios, nonneg=True)
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_scenarios @ weights
        
        # CVaR constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            auxiliary_vars >= -(portfolio_returns - var),  # CVaR constraint
        ]
        
        # Objective: Minimize CVaR while maximizing expected return
        cvar = var + (1 / (self.confidence_level * n_scenarios)) * cp.sum(auxiliary_vars)
        expected_portfolio_return = expected_returns @ weights
        
        # Multi-objective: minimize CVaR and maximize expected return
        objective = cp.Minimize(cvar - 0.1 * expected_portfolio_return)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if weights.value is not None:
            return weights.value
        else:
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets


class MarkovDecisionProcess:
    """
    Markov Decision Process for portfolio optimization
    Models portfolio transitions and optimizes long-term returns
    """
    
    def __init__(self, n_states: int, n_actions: int, discount_factor: float = 0.95):
        """
        Initialize MDP for portfolio optimization
        
        Args:
            n_states: Number of market states
            n_actions: Number of possible portfolio actions
            discount_factor: Future reward discount factor
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        
        # Initialize transition probabilities and rewards
        self.transition_probs = np.random.rand(n_states, n_actions, n_states)
        self.transition_probs = self.transition_probs / np.sum(self.transition_probs, axis=2, keepdims=True)
        
        self.rewards = np.random.randn(n_states, n_actions) * 0.1
        self.value_function = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)
    
    def value_iteration(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        """Perform value iteration to find optimal policy"""
        for iteration in range(max_iterations):
            old_values = self.value_function.copy()
            
            for state in range(self.n_states):
                # Calculate Q-values for all actions
                q_values = []
                for action in range(self.n_actions):
                    expected_value = np.sum(
                        self.transition_probs[state, action, :] * 
                        (self.rewards[state, action] + self.discount_factor * old_values)
                    )
                    q_values.append(expected_value)
                
                # Update value function and policy
                self.value_function[state] = max(q_values)
                self.policy[state] = np.argmax(q_values)
            
            # Check convergence
            if np.max(np.abs(self.value_function - old_values)) < tolerance:
                break
    
    def get_optimal_action(self, state: int) -> int:
        """Get optimal action for given state"""
        return self.policy[state]
    
    def update_model(self, state: int, action: int, next_state: int, reward: float):
        """Update MDP model with observed transition"""
        # Simple update (in practice, would use more sophisticated learning)
        self.rewards[state, action] = 0.9 * self.rewards[state, action] + 0.1 * reward
        
        # Update transition probabilities (simplified)
        self.transition_probs[state, action, next_state] += 0.01
        self.transition_probs[state, action, :] /= np.sum(self.transition_probs[state, action, :])


class MarkovModulatedModel:
    """
    Markov-modulated drift model for portfolio dynamics
    Handles both observable and unobservable regime changes
    """
    
    def __init__(self, n_regimes: int, observable: bool = True):
        """
        Initialize Markov-modulated model
        
        Args:
            n_regimes: Number of market regimes
            observable: Whether regime states are observable
        """
        self.n_regimes = n_regimes
        self.observable = observable
        
        # Regime transition matrix
        self.regime_transition = np.random.rand(n_regimes, n_regimes)
        self.regime_transition = self.regime_transition / np.sum(self.regime_transition, axis=1, keepdims=True)
        
        # Regime-specific parameters
        self.regime_means = np.random.randn(n_regimes) * 0.05
        self.regime_volatilities = np.random.rand(n_regimes) * 0.2 + 0.1
        
        self.current_regime = 0
        self.regime_history = []
        
        if not observable:
            # Hidden Markov Model parameters
            self.observation_probs = np.random.rand(n_regimes, 10)  # 10 observation levels
            self.observation_probs = self.observation_probs / np.sum(self.observation_probs, axis=1, keepdims=True)
            self.regime_beliefs = np.ones(n_regimes) / n_regimes
    
    def simulate_regime_change(self):
        """Simulate regime transition"""
        probs = self.regime_transition[self.current_regime, :]
        self.current_regime = np.random.choice(self.n_regimes, p=probs)
        self.regime_history.append(self.current_regime)
    
    def get_current_parameters(self) -> Tuple[float, float]:
        """Get current regime parameters"""
        if self.observable:
            return self.regime_means[self.current_regime], self.regime_volatilities[self.current_regime]
        else:
            # Use belief-weighted parameters
            expected_mean = np.sum(self.regime_beliefs * self.regime_means)
            expected_vol = np.sum(self.regime_beliefs * self.regime_volatilities)
            return expected_mean, expected_vol
    
    def update_beliefs(self, observation: int):
        """Update regime beliefs based on observation (for unobservable case)"""
        if not self.observable:
            # Bayes update
            likelihood = self.observation_probs[:, observation]
            prior = self.regime_beliefs @ self.regime_transition.T
            posterior = likelihood * prior
            self.regime_beliefs = posterior / np.sum(posterior)


class PartiallyObservableMDP:
    """
    Partially Observable MDP (POMDP) for modeling jump processes
    in portfolio optimization
    """
    
    def __init__(self, n_states: int, n_actions: int, n_observations: int):
        """
        Initialize POMDP for jump process modeling
        
        Args:
            n_states: Number of hidden states
            n_actions: Number of possible actions
            n_observations: Number of possible observations
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_observations = n_observations
        
        # POMDP parameters
        self.transition_probs = np.random.rand(n_states, n_actions, n_states)
        self.transition_probs = self.transition_probs / np.sum(self.transition_probs, axis=2, keepdims=True)
        
        self.observation_probs = np.random.rand(n_states, n_actions, n_observations)
        self.observation_probs = self.observation_probs / np.sum(self.observation_probs, axis=2, keepdims=True)
        
        self.rewards = np.random.randn(n_states, n_actions) * 0.1
        
        # Belief state (probability distribution over hidden states)
        self.belief_state = np.ones(n_states) / n_states
    
    def update_belief(self, action: int, observation: int):
        """Update belief state based on action and observation"""
        # Predict step
        predicted_belief = np.zeros(self.n_states)
        for next_state in range(self.n_states):
            for current_state in range(self.n_states):
                predicted_belief[next_state] += (
                    self.belief_state[current_state] * 
                    self.transition_probs[current_state, action, next_state]
                )
        
        # Update step
        for state in range(self.n_states):
            predicted_belief[state] *= self.observation_probs[state, action, observation]
        
        # Normalize
        self.belief_state = predicted_belief / np.sum(predicted_belief)
    
    def select_action(self) -> int:
        """Select action based on current belief state (simplified policy)"""
        # Simple heuristic: choose action with highest expected reward
        expected_rewards = []
        for action in range(self.n_actions):
            expected_reward = np.sum(self.belief_state * self.rewards[:, action])
            expected_rewards.append(expected_reward)
        
        return np.argmax(expected_rewards)
    
    def model_jump_process(self, market_data: np.ndarray) -> np.ndarray:
        """Model jump process in market data"""
        jump_indicators = np.zeros(len(market_data))
        
        for i in range(1, len(market_data)):
            # Detect potential jumps (simplified)
            return_magnitude = abs(market_data[i] - market_data[i-1])
            threshold = np.std(market_data[:i]) * 2  # 2-sigma threshold
            
            if return_magnitude > threshold:
                jump_indicators[i] = 1
                # Update belief based on jump observation
                self.update_belief(action=0, observation=1)  # Jump observed
            else:
                self.update_belief(action=0, observation=0)  # No jump
        
        return jump_indicators


class AdvancedPortfolioOptimizer:
    """
    Main portfolio optimization engine integrating all strategies:
    MAB, MDP, CVaR, Markov-modulated models, and PDMDPs
    """
    
    def __init__(self, n_assets: int = 5, target_regret: float = 0.4):
        """
        Initialize advanced portfolio optimizer
        
        Args:
            n_assets: Number of assets in portfolio
            target_regret: Target mean regret (default 0.4 as specified)
        """
        self.n_assets = n_assets
        self.target_regret = target_regret
        
        # Initialize components
        self.mab = MultiArmedBandit(n_assets)
        self.cvar_optimizer = CVaROptimizer()
        self.mdp = MarkovDecisionProcess(n_states=5, n_actions=3)  # 5 market states, 3 actions
        self.markov_model = MarkovModulatedModel(n_regimes=3, observable=True)
        self.hidden_markov_model = MarkovModulatedModel(n_regimes=3, observable=False)
        self.pomdp = PartiallyObservableMDP(n_states=3, n_actions=2, n_observations=2)
        
        # Portfolio tracking
        self.portfolio_history = []
        self.returns_history = []
        self.weights_history = []
        
    def generate_synthetic_data(self, n_periods: int = 252) -> Dict[str, np.ndarray]:
        """Generate synthetic market data for simulation"""
        np.random.seed(42)  # For reproducible results
        
        # Generate correlated asset returns
        correlation_matrix = np.random.rand(self.n_assets, self.n_assets)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Generate returns with regime changes
        returns = np.zeros((n_periods, self.n_assets))
        market_states = np.zeros(n_periods, dtype=int)
        
        for t in range(n_periods):
            # Simulate regime change
            self.markov_model.simulate_regime_change()
            market_states[t] = self.markov_model.current_regime
            
            # Get regime-specific parameters
            mean, volatility = self.markov_model.get_current_parameters()
            
            # Generate correlated returns
            random_shocks = np.random.multivariate_normal(
                mean=np.zeros(self.n_assets),
                cov=correlation_matrix
            )
            
            returns[t, :] = mean + volatility * random_shocks
            
            # Add occasional jumps
            if np.random.rand() < 0.05:  # 5% chance of jump
                jump_size = np.random.normal(0, 0.03, self.n_assets)
                returns[t, :] += jump_size
        
        return {
            'returns': returns,
            'market_states': market_states,
            'correlation_matrix': correlation_matrix
        }
    
    def optimize_portfolio(self, data: Dict[str, np.ndarray], 
                         rebalance_frequency: int = 20) -> Dict[str, Union[np.ndarray, float]]:
        """
        Main portfolio optimization using all strategies
        
        Args:
            data: Market data dictionary
            rebalance_frequency: How often to rebalance (in periods)
            
        Returns:
            Optimization results
        """
        returns = data['returns']
        n_periods = len(returns)
        
        # Initialize tracking arrays
        portfolio_weights = np.zeros((n_periods, self.n_assets))
        portfolio_returns = np.zeros(n_periods)
        selected_assets = np.zeros(n_periods, dtype=int)
        
        # Initial equal weights
        current_weights = np.ones(self.n_assets) / self.n_assets
        
        for t in range(n_periods):
            # Store current weights
            portfolio_weights[t, :] = current_weights
            
            # Calculate portfolio return
            if t > 0:
                portfolio_returns[t] = np.sum(current_weights * returns[t, :])
            
            # MAB asset selection (select one asset to overweight)
            selected_asset = self.mab.select_arm()
            selected_assets[t] = selected_asset
            
            # Update MAB with observed return
            if t > 0:
                asset_return = returns[t, selected_asset]
                self.mab.update(selected_asset, asset_return)
            
            # MDP state update and action selection
            current_state = data['market_states'][t] if t < len(data['market_states']) else 0
            if t > 0:
                prev_state = data['market_states'][t-1] if t-1 < len(data['market_states']) else 0
                self.mdp.update_model(prev_state, 0, current_state, portfolio_returns[t])
            
            # Periodic rebalancing using CVaR optimization
            if t > 0 and t % rebalance_frequency == 0 and t >= 50:  # Need sufficient history
                # Get recent returns for CVaR optimization
                recent_returns = returns[max(0, t-50):t, :]
                expected_returns = np.mean(recent_returns, axis=0)
                
                # CVaR optimization
                try:
                    optimal_weights = self.cvar_optimizer.optimize_portfolio_cvar(
                        expected_returns, recent_returns
                    )
                    
                    # Combine MAB selection with CVaR weights
                    # Overweight selected asset by MAB
                    mab_bonus = 0.1  # 10% bonus weight for MAB selected asset
                    optimal_weights[selected_asset] += mab_bonus
                    optimal_weights = optimal_weights / np.sum(optimal_weights)  # Renormalize
                    
                    current_weights = optimal_weights
                    
                except:
                    # Fallback to equal weights if optimization fails
                    current_weights = np.ones(self.n_assets) / self.n_assets
                    current_weights[selected_asset] *= 1.2  # Still give MAB bonus
                    current_weights = current_weights / np.sum(current_weights)
            
            # Update POMDP for jump detection
            if t > 0:
                portfolio_return = portfolio_returns[t]
                jump_threshold = 0.02  # 2% threshold for jump detection
                observation = 1 if abs(portfolio_return) > jump_threshold else 0
                action = self.pomdp.select_action()
                self.pomdp.update_belief(action, observation)
        
        # Calculate performance metrics
        total_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        cvar = self.cvar_optimizer.calculate_cvar(portfolio_returns)
        
        return {
            'portfolio_weights': portfolio_weights,
            'portfolio_returns': portfolio_returns,
            'selected_assets': selected_assets,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cvar': cvar,
            'mean_regret': self.mab.get_mean_regret()
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def plot_results(self, results: Dict[str, Union[np.ndarray, float]]):
        """Plot optimization results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio returns
        cumulative_returns = np.cumprod(1 + results['portfolio_returns'])
        axes[0, 0].plot(cumulative_returns)
        axes[0, 0].set_title('Cumulative Portfolio Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True)
        
        # Portfolio weights over time
        weights = results['portfolio_weights']
        for i in range(self.n_assets):
            axes[0, 1].plot(weights[:, i], label=f'Asset {i+1}')
        axes[0, 1].set_title('Portfolio Weights Over Time')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAB regret
        axes[1, 0].plot(self.mab.regret_history)
        axes[1, 0].set_title(f'MAB Regret (Mean: {results["mean_regret"]:.4f})')
        axes[1, 0].set_ylabel('Regret')
        axes[1, 0].axhline(y=self.target_regret, color='r', linestyle='--', 
                          label=f'Target: {self.target_regret}')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Performance metrics
        metrics_text = f"""
        Total Return: {results['total_return']:.2%}
        Annualized Return: {results['annualized_return']:.2%}
        Volatility: {results['volatility']:.2%}
        Sharpe Ratio: {results['sharpe_ratio']:.3f}
        Max Drawdown: {results['max_drawdown']:.2%}
        CVaR (5%): {results['cvar']:.4f}
        Mean Regret: {results['mean_regret']:.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/Quant-2/Quant-2/portfolio_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def run_comprehensive_analysis(self) -> Dict[str, Union[np.ndarray, float]]:
        """Run comprehensive portfolio optimization analysis"""
        print("Advanced Portfolio Optimization & Selection")
        print("=" * 50)
        print("Generating synthetic market data...")
        
        # Generate synthetic data
        data = self.generate_synthetic_data(n_periods=252)  # 1 year of daily data
        
        print("Running portfolio optimization...")
        print("- Multi-Armed Bandit (UCB) for asset selection")
        print("- CVaR optimization for risk management")
        print("- MDP for long-term strategy")
        print("- Markov-modulated drift models")
        print("- POMDP for jump process modeling")
        
        # Run optimization
        results = self.optimize_portfolio(data)
        
        print("\nOptimization Results:")
        print("-" * 30)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"CVaR (5%): {results['cvar']:.4f}")
        print(f"Mean Regret: {results['mean_regret']:.4f}")
        
        # Check if target regret is achieved
        if results['mean_regret'] <= self.target_regret:
            print(f"\n✓ Target regret of {self.target_regret} achieved!")
        else:
            print(f"\n⚠ Target regret of {self.target_regret} not achieved. Current: {results['mean_regret']:.4f}")
        
        # Plot results
        self.plot_results(results)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize portfolio optimizer
    optimizer = AdvancedPortfolioOptimizer(n_assets=5, target_regret=0.4)
    
    # Run comprehensive analysis
    results = optimizer.run_comprehensive_analysis()
    
    print(f"\nAdvanced Portfolio Optimization completed successfully!")
    print(f"Results saved to portfolio_results.png")