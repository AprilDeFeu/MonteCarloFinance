#!/usr/bin/env python
"""
Example: Stress Testing with Shockwave Events

This example demonstrates how to run stress tests with various
shockwave scenarios, including building defaults and market crashes.
"""

import numpy as np
from monte_carlo_finance import (
    MonteCarloSimulation,
    SimulationConfig,
    ShockwaveEvent,
    ShockType,
)


def main():
    """Run stress test scenarios."""
    print("=" * 70)
    print("Stress Testing: Shockwave Events and Market Crashes")
    print("=" * 70)
    
    # Base configuration
    config = SimulationConfig(
        num_simulations=500,
        num_steps=252,          # One year daily
        time_horizon=1.0,
        random_seed=42,
    )
    
    # Set up realistic parameters
    config.market.initial_value = 100.0
    config.market.drift = 0.05
    config.market.volatility = 0.18
    
    config.bond.face_value = 1000.0
    config.bond.coupon_rate = 0.045
    config.bond.maturity_years = 10.0
    config.bond.initial_yield = 0.045
    config.bond.credit_spread = 0.015
    
    # Define stress scenarios
    scenarios = [
        {
            "name": "Baseline (No Events)",
            "events": [],
            "description": "Normal market conditions",
        },
        {
            "name": "Single Building Default",
            "events": [(126, ShockwaveEvent(  # Mid-year
                shock_type=ShockType.DEFAULT,
                timestamp=126,
                yield_impact=0.025,         # 2.5% yield spike
                market_impact=0.12,         # 12% market drop
                duration=15,                # Effects last 15 days
                propagation_factor=0.6,
                name="Commercial Building Default",
                metadata={"asset": "Office Tower A", "loan_value": 50_000_000},
            ))],
            "description": "Single major commercial real estate default",
        },
        {
            "name": "Multiple Defaults (Cascade)",
            "events": [
                (100, ShockwaveEvent(
                    shock_type=ShockType.DEFAULT,
                    timestamp=100,
                    yield_impact=0.02,
                    market_impact=0.08,
                    duration=10,
                    propagation_factor=0.7,
                    name="Building Default 1",
                )),
                (115, ShockwaveEvent(  # Second default triggered by first
                    shock_type=ShockType.DEFAULT,
                    timestamp=115,
                    yield_impact=0.025,
                    market_impact=0.10,
                    duration=12,
                    propagation_factor=0.65,
                    name="Building Default 2",
                )),
                (130, ShockwaveEvent(  # Third default in cascade
                    shock_type=ShockType.DEFAULT,
                    timestamp=130,
                    yield_impact=0.03,
                    market_impact=0.12,
                    duration=15,
                    propagation_factor=0.6,
                    name="Building Default 3",
                )),
            ],
            "description": "Cascade of defaults in commercial real estate",
        },
        {
            "name": "Liquidity Crisis",
            "events": [(150, ShockwaveEvent(
                shock_type=ShockType.LIQUIDITY,
                timestamp=150,
                yield_impact=0.04,          # Major yield spike
                market_impact=0.18,         # 18% market drop
                duration=20,
                propagation_factor=0.75,
                name="Systemic Liquidity Crisis",
            ))],
            "config_overrides": {
                "panic.panic_sensitivity": 0.8,  # Higher panic sensitivity
            },
            "description": "Systemic liquidity event with elevated panic",
        },
        {
            "name": "Full Market Crash",
            "events": [(100, ShockwaveEvent(
                shock_type=ShockType.MARKET_CRASH,
                timestamp=100,
                yield_impact=0.06,          # Major flight to quality
                market_impact=0.30,         # 30% market crash
                duration=30,
                propagation_factor=0.8,
                name="2008-Style Market Crash",
            ))],
            "config_overrides": {
                "panic.base_panic_level": 0.4,
                "panic.volatility_multiplier": 3.0,
            },
            "description": "Severe market crash with high panic",
        },
    ]
    
    # Create simulation
    sim = MonteCarloSimulation(config)
    
    print("\nRunning stress test scenarios...\n")
    
    # Run stress tests
    results = sim.stress_test(scenarios)
    
    # Display results comparison
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS COMPARISON")
    print("=" * 70)
    
    # Header
    print(f"\n{'Scenario':<30} {'Market':<12} {'Bond':<12} {'Default':<10} {'Events':<8}")
    print(f"{'':30} {'Final':<12} {'Price':<12} {'Rate':<10} {'Count':<8}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        scenario_name = result.statistics["scenario_name"]
        market_final = result.market_paths[:, -1].mean()
        bond_final = result.bond_prices[:, -1].mean()
        default_rate = result.statistics["default_rate"]["rate"]
        total_events = result.statistics["shockwave"]["total_events"]
        
        print(f"{scenario_name:<30} {market_final:<12.2f} {bond_final:<12.2f} "
              f"{default_rate:<10.2%} {total_events:<8}")
    
    # Detailed analysis for each scenario
    print("\n" + "=" * 70)
    print("DETAILED SCENARIO ANALYSIS")
    print("=" * 70)
    
    for i, result in enumerate(results):
        scenario = scenarios[i]
        print(f"\n--- {result.statistics['scenario_name']} ---")
        print(f"Description: {scenario['description']}")
        
        # Market statistics
        market_stats = result.statistics["market_final"]
        print(f"\nMarket Value:")
        print(f"  Final Mean:    {market_stats['mean']:.2f}")
        print(f"  Final Std:     {market_stats['std']:.2f}")
        print(f"  VaR (5%):      {market_stats['percentile_5']:.2f}")
        print(f"  Worst case:    {market_stats['min']:.2f}")
        
        # Risk metrics
        return_stats = result.statistics["market_returns"]
        print(f"\nRisk Metrics:")
        print(f"  Volatility:    {return_stats['std']*np.sqrt(252)*100:.1f}% (annualized)")
        print(f"  Max drawdown potential: {100 - market_stats['min']:.1f}%")
        
        # Panic statistics
        panic_stats = result.statistics["panic_max"]
        print(f"\nPanic Dynamics:")
        print(f"  Average max panic: {panic_stats['mean']:.4f}")
        print(f"  Peak panic:        {panic_stats['max']:.4f}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Impact of Stress Events")
    print("=" * 70)
    
    baseline_market = results[0].market_paths[:, -1].mean()
    baseline_bond = results[0].bond_prices[:, -1].mean()
    
    print(f"\n{'Scenario':<30} {'Market Impact':<15} {'Bond Impact':<15}")
    print("-" * 60)
    
    for result in results:
        name = result.statistics["scenario_name"]
        market_final = result.market_paths[:, -1].mean()
        bond_final = result.bond_prices[:, -1].mean()
        
        market_impact = (market_final - baseline_market) / baseline_market * 100
        bond_impact = (bond_final - baseline_bond) / baseline_bond * 100
        
        print(f"{name:<30} {market_impact:>+14.2f}% {bond_impact:>+14.2f}%")
    
    print("\n" + "=" * 70)
    print("Stress testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
