#!/usr/bin/env python
"""
Example: Custom Scenario - Building Default Cascade

This example demonstrates a custom scenario where a default on a building
leads to rising bond yields, triggering automated selloffs and market panic.
This is the specific use case mentioned in the requirements.
"""

import numpy as np
from monte_carlo_finance import (
    MonteCarloSimulation,
    SimulationConfig,
    ShockwaveEvent,
    ShockType,
)
from monte_carlo_finance.triggers.panic import PanicConfig


def main():
    """Simulate building default cascade scenario."""
    print("=" * 70)
    print("Building Default Cascade Scenario")
    print("=" * 70)
    print("""
This simulation models a scenario where:
1. A major commercial building defaults on its loan
2. This triggers a spike in bond yields
3. Automated selloff triggers are activated
4. Market panic ensues, causing further volatility and selling
5. Cascade effects propagate through the system
""")
    
    # Configure simulation for detailed analysis
    config = SimulationConfig(
        num_simulations=500,
        num_steps=252,          # One year daily
        time_horizon=1.0,
        random_seed=42,
    )
    
    # Market configuration - moderate risk environment
    config.market.initial_value = 100.0
    config.market.drift = 0.05           # 5% expected return
    config.market.volatility = 0.18      # 18% base volatility
    config.market.jump_intensity = 0.2   # Some jump risk
    config.market.jump_mean = -0.03      # Average jump is -3%
    config.market.jump_std = 0.05
    
    # Bond configuration - investment grade with some credit risk
    config.bond.face_value = 1000.0
    config.bond.coupon_rate = 0.05       # 5% coupon
    config.bond.maturity_years = 10.0
    config.bond.initial_yield = 0.04     # Starting at 4% yield
    config.bond.yield_volatility = 0.008 # Yield volatility
    config.bond.credit_spread = 0.015    # 1.5% credit spread
    config.bond.recovery_rate = 0.35     # 35% recovery rate
    
    # Shockwave configuration - sensitive to cascades
    config.shockwave.base_probability = 0.005  # 0.5% base shock probability
    config.shockwave.yield_impact = 0.015      # 1.5% yield impact per shock
    config.shockwave.market_impact = 0.08      # 8% market impact
    config.shockwave.cascade_decay = 0.65      # 65% cascade propagation
    config.shockwave.selloff_threshold = 0.03  # 3% yield increase triggers selloff
    config.shockwave.selloff_intensity = 0.25  # Sell 25% of position
    
    # Panic configuration - realistic panic dynamics
    config.panic.base_panic_level = 0.05       # Slight background anxiety
    config.panic.panic_sensitivity = 0.6       # Moderate sensitivity
    config.panic.panic_decay = 0.08            # Slow decay (8% per day)
    config.panic.max_panic_level = 1.0
    config.panic.volatility_multiplier = 2.5   # 2.5x volatility at max panic
    config.panic.drift_impact = -0.15          # -15% drift at max panic
    config.panic.correlation_boost = 0.4       # +40% correlation at max panic
    config.panic.herd_behavior_factor = 0.35   # Strong herding
    config.panic.recovery_threshold = 0.25     # Below 25% = not panicking
    
    # Create simulation
    sim = MonteCarloSimulation(config)
    
    # =========================================================================
    # Scenario 1: Building Default Event
    # =========================================================================
    print("\n" + "-" * 70)
    print("SCENARIO 1: Building Default on Day 126 (Mid-Year)")
    print("-" * 70)
    
    # Create the default event
    building_default = ShockwaveEvent(
        shock_type=ShockType.DEFAULT,
        timestamp=126,              # Day 126 (mid-year)
        yield_impact=0.035,         # 3.5% immediate yield spike
        market_impact=0.15,         # 15% initial market impact
        duration=25,                # Effects persist for 25 trading days
        propagation_factor=0.70,    # 70% propagation strength
        name="Commercial Tower Default",
        metadata={
            "asset_type": "Commercial Real Estate",
            "asset_name": "Downtown Office Tower",
            "loan_value": 150_000_000,
            "occupancy_at_default": 0.35,
            "cause": "Major tenant bankruptcy + remote work shift",
        }
    )
    
    print("\nDefault Event Details:")
    print(f"  Asset: {building_default.metadata['asset_name']}")
    print(f"  Loan Value: ${building_default.metadata['loan_value']:,}")
    print(f"  Cause: {building_default.metadata['cause']}")
    print(f"  Day of Default: {building_default.timestamp}")
    print(f"  Yield Impact: {building_default.yield_impact*100:.1f}%")
    print(f"  Market Impact: {building_default.market_impact*100:.1f}%")
    print(f"  Duration: {building_default.duration} days")
    
    # Run the scenario
    print("\nRunning simulation...")
    results_default = sim.run_scenario([(126, building_default)], num_simulations=500)
    
    # =========================================================================
    # Scenario 2: Baseline (No Default)
    # =========================================================================
    print("\n" + "-" * 70)
    print("SCENARIO 2: Baseline (No Default Event)")
    print("-" * 70)
    
    print("\nRunning baseline simulation...")
    results_baseline = sim.run_scenario([], num_simulations=500)
    
    # =========================================================================
    # Comparison Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Market value comparison
    print("\n--- Market Value Analysis ---")
    
    baseline_market = results_baseline.market_paths[:, -1]
    default_market = results_default.market_paths[:, -1]
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'With Default':<15} {'Difference':<15}")
    print("-" * 70)
    
    metrics = [
        ("Mean Final Value", baseline_market.mean(), default_market.mean()),
        ("Median Final Value", np.median(baseline_market), np.median(default_market)),
        ("Std Dev", baseline_market.std(), default_market.std()),
        ("5th Percentile", np.percentile(baseline_market, 5), np.percentile(default_market, 5)),
        ("1st Percentile", np.percentile(baseline_market, 1), np.percentile(default_market, 1)),
        ("Minimum", baseline_market.min(), default_market.min()),
    ]
    
    for name, base_val, def_val in metrics:
        diff = def_val - base_val
        diff_pct = diff / base_val * 100 if base_val != 0 else 0
        print(f"{name:<25} {base_val:<15.2f} {def_val:<15.2f} {diff:>+.2f} ({diff_pct:>+.1f}%)")
    
    # Bond analysis
    print("\n--- Bond Analysis ---")
    
    baseline_bonds = results_baseline.bond_prices[:, -1]
    default_bonds = results_default.bond_prices[:, -1]
    baseline_yields = results_baseline.bond_yields[:, -1]
    default_yields = results_default.bond_yields[:, -1]
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'With Default':<15} {'Difference':<15}")
    print("-" * 70)
    
    bond_metrics = [
        ("Mean Bond Price", baseline_bonds.mean(), default_bonds.mean()),
        ("Mean Yield", baseline_yields.mean()*100, default_yields.mean()*100),
        ("Max Yield", baseline_yields.max()*100, default_yields.max()*100),
    ]
    
    for name, base_val, def_val in bond_metrics:
        diff = def_val - base_val
        unit = "%" if "Yield" in name else ""
        print(f"{name:<25} {base_val:<15.2f}{unit} {def_val:<15.2f}{unit} {diff:>+.2f}{unit}")
    
    # Default rate comparison
    print("\n--- Credit Risk Impact ---")
    
    baseline_defaults = results_baseline.statistics["default_rate"]["rate"]
    default_defaults = results_default.statistics["default_rate"]["rate"]
    
    print(f"Baseline default rate: {baseline_defaults:.2%}")
    print(f"With shock default rate: {default_defaults:.2%}")
    print(f"Increase in defaults: {(default_defaults - baseline_defaults)*100:.2f} percentage points")
    
    # Panic analysis
    print("\n--- Panic Dynamics ---")
    
    baseline_panic = np.max(results_baseline.panic_levels, axis=1)
    default_panic = np.max(results_default.panic_levels, axis=1)
    
    print(f"\nMaximum Panic Levels Reached:")
    print(f"  Baseline mean: {baseline_panic.mean():.4f}")
    print(f"  With default mean: {default_panic.mean():.4f}")
    print(f"  Baseline max: {baseline_panic.max():.4f}")
    print(f"  With default max: {default_panic.max():.4f}")
    
    # Selloff analysis
    print("\n--- Automated Selloff Activity ---")
    
    baseline_selloffs = sum(len(s) for s in results_baseline.selloff_events)
    default_selloffs = sum(len(s) for s in results_default.selloff_events)
    
    print(f"Total selloff triggers (baseline): {baseline_selloffs}")
    print(f"Total selloff triggers (with default): {default_selloffs}")
    print(f"Increase: {default_selloffs - baseline_selloffs} additional selloffs")
    
    # Event cascade analysis
    print("\n--- Event Cascade Analysis ---")
    
    baseline_events = sum(len(e) for e in results_baseline.shockwave_events)
    default_events = sum(len(e) for e in results_default.shockwave_events)
    
    print(f"Total shockwave events (baseline): {baseline_events}")
    print(f"Total shockwave events (with default): {default_events}")
    print(f"Average events per simulation (baseline): {baseline_events/500:.2f}")
    print(f"Average events per simulation (with default): {default_events/500:.2f}")
    
    # =========================================================================
    # Risk Metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("RISK METRICS")
    print("=" * 70)
    
    # Calculate returns
    baseline_returns = np.diff(np.log(results_baseline.market_paths), axis=1)
    default_returns = np.diff(np.log(results_default.market_paths), axis=1)
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'With Default':<15}")
    print("-" * 60)
    
    # Annualized volatility
    base_vol = baseline_returns.std() * np.sqrt(252) * 100
    def_vol = default_returns.std() * np.sqrt(252) * 100
    print(f"{'Annualized Volatility':<30} {base_vol:<15.2f}% {def_vol:<15.2f}%")
    
    # Value at Risk (95%)
    base_var = np.percentile(baseline_returns.flatten(), 5)
    def_var = np.percentile(default_returns.flatten(), 5)
    print(f"{'Daily VaR (95%)':<30} {base_var*100:<15.4f}% {def_var*100:<15.4f}%")
    
    # Conditional VaR
    base_cvar = baseline_returns[baseline_returns <= base_var].mean()
    def_cvar = default_returns[default_returns <= def_var].mean()
    print(f"{'Daily CVaR':<30} {base_cvar*100:<15.4f}% {def_cvar*100:<15.4f}%")
    
    # Maximum drawdown potential
    base_max_dd = (100 - baseline_market.min()) / 100
    def_max_dd = (100 - default_market.min()) / 100
    print(f"{'Max Drawdown Potential':<30} {base_max_dd*100:<15.2f}% {def_max_dd*100:<15.2f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    market_impact = (default_market.mean() - baseline_market.mean()) / baseline_market.mean() * 100
    vol_increase = (def_vol - base_vol) / base_vol * 100
    
    print(f"""
The building default scenario demonstrates the cascade effects of a 
credit event on the financial system:

1. DIRECT IMPACT
   - Market value decreased by {abs(market_impact):.1f}% on average
   - Bond prices fell as yields spiked

2. VOLATILITY AMPLIFICATION  
   - Annualized volatility increased by {vol_increase:.1f}%
   - Panic levels elevated throughout the system

3. CONTAGION EFFECTS
   - {default_events - baseline_events:,} additional shockwave events triggered
   - {default_selloffs - baseline_selloffs} additional automated selloffs

4. TAIL RISK
   - 5th percentile market value: {np.percentile(default_market, 5):.2f} vs {np.percentile(baseline_market, 5):.2f}
   - Worst case market value: {default_market.min():.2f} vs {baseline_market.min():.2f}

This illustrates how a single default can trigger a chain reaction
through yield spikes, automated selloffs, and market panic.
""")
    
    print("=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
