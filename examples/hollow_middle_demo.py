#!/usr/bin/env python
"""
Example: The "Hollow Middle" Demo

This example demonstrates the "Hollow Middle" thesis where:
1. Asset prices (Market) rise, driving up Cost of Living (CoL).
2. Wages stagnate relative to productivity/inflation.
3. The gap is filled by debt, increasing default risk.
"""

import numpy as np
from monte_carlo_finance import MonteCarloSimulation, SimulationConfig

def main():
    print("=" * 60)
    print("The 'Hollow Middle' Simulation Demo")
    print("=" * 60)
    
    # Create simulation with custom configuration
    config = SimulationConfig(
        num_simulations=500,    # Fewer simulations for clarity
        num_steps=252 * 5,      # 5 years to let the divergence play out
        time_horizon=5.0,
        random_seed=42,
    )
    
    # 1. Market Setup (The Asset Economy)
    # High returns drive the "wealth effect" but also inflation
    config.market.initial_value = 100.0
    config.market.drift = 0.08          # 8% annual return
    config.market.volatility = 0.15     # Moderate volatility
    
    # 2. Consumer Setup (The Real Economy)
    # Wages grow slower than the market/inflation
    config.consumer.initial_wage_index = 100.0
    config.consumer.initial_col_index = 98.0  # Starts with tight surplus (2% gap)
    
    # The "Decoupling": Wages grow at 1% (Productivity 4% * 75% Lag)
    config.consumer.productivity_growth = 0.04
    config.consumer.wage_lag = 0.75
    
    # The "Bifurcation": CoL rises with Market (Housing/Assets)
    # Base inflation is hardcoded to 2% in model, plus this weight * market_return
    config.consumer.asset_inflation_weight = 0.6
    
    # The "Debt Trap"
    config.consumer.max_leverage_ratio = 0.5 # Default starts when Debt > 50% of Income
    config.consumer.savings_rate = 0.1 # Pay down debt slowly if surplus exists
    
    # Create and run simulation
    sim = MonteCarloSimulation(config)
    
    print(f"\nRunning simulation over {config.time_horizon} years...")
    print(f"Scenario: Wages grow @ 3%, CoL @ 5% + 0.3*MarketReturn")
    
    results = sim.run(progress_callback=lambda c, t: print(f"  Progress: {c}/{t}", end="\r"))
    print("\n\nSimulation complete!")
    
    # Analyze Results
    print("\n" + "-" * 60)
    print("CONSUMER ECONOMY RESULTS (Average across simulations)")
    print("-" * 60)
    
    # Extract paths
    wages = results.consumer_wage_paths
    col = results.consumer_col_paths
    debt = results.consumer_debt_paths
    defaults = results.consumer_default_prob_paths
    
    # Calculate averages over time
    avg_wage = np.mean(wages, axis=0)
    avg_col = np.mean(col, axis=0)
    avg_debt = np.mean(debt, axis=0)
    avg_default = np.mean(defaults, axis=0)
    
    # Initial vs Final
    print(f"{'Metric':<20} | {'Start':<12} | {'End (Year 5)':<12} | {'Change':<10}")
    print("-" * 60)
    print(f"{'Avg Wage Index':<20} | {avg_wage[0]:.2f}         | {avg_wage[-1]:.2f}         | {((avg_wage[-1]/avg_wage[0])-1)*100:+.1f}%")
    print(f"{'Avg CoL Index':<20} | {avg_col[0]:.2f}         | {avg_col[-1]:.2f}         | {((avg_col[-1]/avg_col[0])-1)*100:+.1f}%")
    print(f"{'Avg Debt Index':<20} | {avg_debt[0]:.2f}         | {avg_debt[-1]:.2f}         | {((avg_debt[-1]/avg_debt[0])-1)*100:+.1f}%" if avg_debt[0] > 0 else f"{'Avg Debt Index':<20} | {avg_debt[0]:.2f}         | {avg_debt[-1]:.2f}         | N/A")
    print(f"{'Default Prob':<20} | {avg_default[0]:.2%}       | {avg_default[-1]:.2%}       | {((avg_default[-1]-avg_default[0])*100):+.2f} pts")
    
    print("\n" + "-" * 60)
    print("THE HOLLOW MIDDLE ANALYSIS")
    print("-" * 60)
    
    # Calculate the "Gap"
    initial_surplus = avg_wage[0] - avg_col[0]
    final_surplus = avg_wage[-1] - avg_col[-1]
    
    print(f"Initial Surplus (Index): {initial_surplus:.2f}")
    print(f"Final Surplus (Index):   {final_surplus:.2f}")
    
    if final_surplus < initial_surplus:
        print("\n>>> RESULT: The Middle Class has been squeezed.")
        print(f"    Surplus dropped by {initial_surplus - final_surplus:.2f} points")
        print(f"    Debt increased to cover the gap.")
    else:
        print("\n>>> RESULT: The Middle Class is thriving (Unexpected in this scenario).")

if __name__ == "__main__":
    main()
