"""
Advanced Portfolio Optimization & Selection - Strategy Implementation
"""

from portfolio_optimization import (
    MultiArmedBandit, CVaROptimizer, MarkovDecisionProcess,
    MarkovModulatedModel, PartiallyObservableMDP, AdvancedPortfolioOptimizer
)
import numpy as np

def main_strategy():
    """
    Main strategy function that demonstrates the Advanced Portfolio Optimization
    """
    print("Advanced Portfolio Optimization & Selection")
    print("Self Project Mar'24 – Apr'24")
    print("=" * 60)
    
    # Initialize the advanced portfolio optimizer
    optimizer = AdvancedPortfolioOptimizer(n_assets=5, target_regret=0.4)
    
    # Run comprehensive analysis
    results = optimizer.run_comprehensive_analysis()
    
    return results

if __name__ == "__main__":
    main_strategy()