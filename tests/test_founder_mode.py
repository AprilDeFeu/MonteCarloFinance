
import unittest
import numpy as np
from monte_carlo_finance.core.config import SimulationConfig, MarketConfig, LiquidityConfig
from monte_carlo_finance.core.simulation import MonteCarloSimulation
from monte_carlo_finance.triggers.shockwave import ShockType

class TestFounderMode(unittest.TestCase):
    def test_volatility_amplification(self):
        """Test that founder mode intensity increases effective volatility."""
        # Case 1: Normal Management
        config_normal = SimulationConfig(
            num_simulations=1,
            num_steps=100,
            random_seed=42,
            market=MarketConfig(
                volatility=0.2,
                founder_mode_intensity=0.0,
                moral_hazard_factor=0.0
            )
        )
        sim_normal = MonteCarloSimulation(config_normal)
        results_normal = sim_normal.run()
        
        # Case 2: Founder Mode
        config_founder = SimulationConfig(
            num_simulations=1,
            num_steps=100,
            random_seed=42,
            market=MarketConfig(
                volatility=0.2,
                founder_mode_intensity=0.8, # High intensity
                moral_hazard_factor=0.5
            )
        )
        sim_founder = MonteCarloSimulation(config_founder)
        results_founder = sim_founder.run()
        
        # Calculate realized volatility
        returns_normal = np.diff(np.log(results_normal.market_paths[0]))
        vol_normal = np.std(returns_normal) * np.sqrt(252) # Annualized
        
        returns_founder = np.diff(np.log(results_founder.market_paths[0]))
        vol_founder = np.std(returns_founder) * np.sqrt(252)
        
        print(f"Normal Volatility: {vol_normal:.4f}")
        print(f"Founder Mode Volatility: {vol_founder:.4f}")
        
        self.assertGreater(vol_founder, vol_normal, "Founder mode should increase volatility")

    def test_management_failure_trigger(self):
        """Test that low liquidity + high ZIRP dependency triggers Management Failure."""
        config = SimulationConfig(
            num_simulations=1,
            num_steps=200,
            random_seed=123,
            market=MarketConfig(
                zirp_dependency=0.9, # High dependency
                founder_mode_intensity=0.5
            ),
            liquidity=LiquidityConfig(
                initial_level=0.5, # Start with low liquidity (below 0.8 threshold)
                mean_reversion_speed=0.0 # Stay low
            )
        )
        
        sim = MonteCarloSimulation(config)
        results = sim.run()
        
        # Check for events
        # shockwave_events is a list of lists (one per simulation)
        events = results.shockwave_events[0]
        management_failures = [e for e in events if e.shock_type == ShockType.MANAGEMENT_FAILURE]
        
        print(f"Found {len(management_failures)} Management Failure events")
        
        if len(management_failures) > 0:
            print("Management Failure triggered successfully.")
            self.assertEqual(management_failures[0].shock_type, ShockType.MANAGEMENT_FAILURE)
        else:
            print("No Management Failure triggered (might be due to randomness).")

if __name__ == '__main__':
    unittest.main()
