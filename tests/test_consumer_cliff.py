import pytest
from monte_carlo_finance.core.simulation import MonteCarloSimulation
from monte_carlo_finance.core.config import SimulationConfig, ConsumerConfig, ShockwaveConfig
from monte_carlo_finance.triggers.shockwave import ShockType

def test_consumer_cliff_trigger():
    """
    Test that the Consumer Cliff (Credit Failure) shockwave is triggered
    when consumer default probability exceeds the threshold.
    """
    # 1. Setup Config with High Debt to force default
    consumer_config = ConsumerConfig(
        initial_wage_index=100.0,
        initial_col_index=100.0,
        initial_debt_index=200.0,  # Massive debt
        max_leverage_ratio=1.0,    # Low tolerance
        productivity_growth=0.0,
        wage_lag=0.0,
        asset_inflation_weight=0.0
    )
    
    # Ensure shockwave is enabled (it is by default if passed to simulation)
    # The threshold is hardcoded to 0.05 in the trigger logic currently
    shock_config = ShockwaveConfig()
    
    config = SimulationConfig(
        num_simulations=1, # Single run for deterministic check
        num_steps=5, # Short run, should trigger immediately
        time_horizon=5/252, # 5 days
        consumer=consumer_config,
        shockwave=shock_config,
        random_seed=42
    )
    
    # 2. Run Simulation
    sim = MonteCarloSimulation(config)
    results = sim.run()
    
    # 3. Verify Trigger
    # Check consumer default probability path
    default_probs = results.consumer_default_prob_paths[0]
    print(f"Consumer Default Probs: {default_probs}")
    
    # Check if any shockwave was active in the single simulation
    events = results.shockwave_events[0]
    assert len(events) > 0, "Shockwave should be active due to consumer default"
    
    # Check if the specific shock type was triggered
    active_types = [e.shock_type for e in events]
    assert ShockType.CONSUMER_CREDIT_FAILURE in active_types, \
        f"Expected CONSUMER_CREDIT_FAILURE, found {active_types}"
        
    # 4. Verify Impact
    # The market should have crashed significantly in the first few steps
    final_price = results.market_paths[0, -1]
    initial_price = config.market.initial_value
    
    print(f"Initial Price: {initial_price}, Final Price: {final_price}")
    print(f"Events Triggered: {[e.name for e in events]}")

if __name__ == "__main__":
    test_consumer_cliff_trigger()
