#!/usr/bin/env python
"""
Example: Basic Monte Carlo Simulation

This example demonstrates how to run a basic Monte Carlo simulation
for financial markets using the default configuration.
"""

from monte_carlo_finance import MonteCarloSimulation, SimulationConfig


def main():
    """Run a basic Monte Carlo simulation."""
    print("=" * 60)
    print("Basic Monte Carlo Financial Simulation")
    print("=" * 60)
    
    # Create simulation with custom configuration
    config = SimulationConfig(
        num_simulations=1000,
        num_steps=252,          # Daily steps for one year
        time_horizon=1.0,       # 1 year horizon
        random_seed=42,         # For reproducibility
    )
    
    # Customize market parameters
    config.market.initial_value = 100.0
    config.market.drift = 0.07          # 7% expected return
    config.market.volatility = 0.20     # 20% volatility
    
    # Customize bond parameters
    config.bond.face_value = 1000.0
    config.bond.coupon_rate = 0.05
    config.bond.maturity_years = 10.0
    
    # Create and run simulation
    sim = MonteCarloSimulation(config)
    
    print("\nRunning simulation...")
    results = sim.run(progress_callback=lambda c, t: print(f"  Progress: {c}/{t}", end="\r"))
    print("\n\nSimulation complete!")
    
    # Display results
    print("\n" + "-" * 60)
    print("MARKET RESULTS")
    print("-" * 60)
    
    market_stats = results.statistics["market_final"]
    print(f"Final Market Value Statistics:")
    print(f"  Mean:      {market_stats['mean']:.2f}")
    print(f"  Std Dev:   {market_stats['std']:.2f}")
    print(f"  Median:    {market_stats['median']:.2f}")
    print(f"  Min:       {market_stats['min']:.2f}")
    print(f"  Max:       {market_stats['max']:.2f}")
    print(f"  5th %%ile:  {market_stats['percentile_5']:.2f}")
    print(f"  95th %%ile: {market_stats['percentile_95']:.2f}")
    
    print("\n" + "-" * 60)
    print("BOND RESULTS")
    print("-" * 60)
    
    bond_stats = results.statistics["bond_price_final"]
    print(f"Final Bond Price Statistics:")
    print(f"  Mean:      {bond_stats['mean']:.2f}")
    print(f"  Std Dev:   {bond_stats['std']:.2f}")
    print(f"  Median:    {bond_stats['median']:.2f}")
    
    yield_stats = results.statistics["bond_yield_final"]
    print(f"\nFinal Bond Yield Statistics:")
    print(f"  Mean:      {yield_stats['mean']:.4f} ({yield_stats['mean']*100:.2f}%)")
    print(f"  Std Dev:   {yield_stats['std']:.4f}")
    
    print("\n" + "-" * 60)
    print("RISK METRICS")
    print("-" * 60)
    
    return_stats = results.statistics["market_returns"]
    print(f"Market Return Statistics:")
    print(f"  Mean (daily):  {return_stats['mean']:.6f}")
    print(f"  Std (daily):   {return_stats['std']:.6f}")
    print(f"  VaR (95%):     {return_stats['var']:.4f} ({return_stats['var']*100:.2f}%)")
    print(f"  CVaR:          {return_stats['cvar']:.4f} ({return_stats['cvar']*100:.2f}%)")
    print(f"  Skewness:      {return_stats['skewness']:.4f}")
    print(f"  Kurtosis:      {return_stats['kurtosis']:.4f}")
    
    print("\n" + "-" * 60)
    print("DEFAULT AND EVENT STATISTICS")
    print("-" * 60)
    
    default_stats = results.statistics["default_rate"]
    print(f"Default Rate: {default_stats['rate']:.2%}")
    print(f"  Defaults: {default_stats['count']} / {default_stats['total']}")
    
    shock_stats = results.statistics["shockwave"]
    print(f"\nShockwave Events:")
    print(f"  Total events: {shock_stats['total_events']}")
    print(f"  Avg per sim:  {shock_stats['avg_events_per_sim']:.2f}")
    
    panic_stats = results.statistics["panic_max"]
    print(f"\nPanic Level (maximum reached):")
    print(f"  Mean max:  {panic_stats['mean']:.4f}")
    print(f"  Max max:   {panic_stats['max']:.4f}")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
