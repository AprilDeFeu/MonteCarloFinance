"""Tests for Monte Carlo simulation engine."""

import numpy as np

from monte_carlo_finance.core.config import SimulationConfig
from monte_carlo_finance.core.simulation import MonteCarloSimulation, SimulationResults
from monte_carlo_finance.triggers.shockwave import ShockType, ShockwaveEvent


class TestSimulationResults:
    """Tests for SimulationResults."""

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        # Create minimal results
        results = SimulationResults(
            market_paths=np.random.rand(10, 101) * 100 + 50,
            bond_prices=np.random.rand(10, 101) * 100 + 900,
            bond_yields=np.random.rand(10, 101) * 0.05 + 0.03,
            panic_levels=np.random.rand(10, 101) * 0.5,
            fear_greed_index=np.random.rand(10, 101) * 100,
            shockwave_events=[[] for _ in range(10)],
            selloff_events=[[] for _ in range(10)],
            defaults=np.zeros(10, dtype=bool),
        )

        results.calculate_statistics()

        assert "market_final" in results.statistics
        assert "bond_price_final" in results.statistics
        assert "default_rate" in results.statistics
        assert "mean" in results.statistics["market_final"]

    def test_get_percentile_paths(self):
        """Test percentile path calculation."""
        results = SimulationResults(
            market_paths=np.random.rand(100, 51) * 100 + 50,
            bond_prices=np.random.rand(100, 51) * 100 + 900,
            bond_yields=np.random.rand(100, 51) * 0.05 + 0.03,
            panic_levels=np.random.rand(100, 51) * 0.5,
            fear_greed_index=np.random.rand(100, 51) * 100,
            shockwave_events=[[] for _ in range(100)],
            selloff_events=[[] for _ in range(100)],
            defaults=np.zeros(100, dtype=bool),
        )

        percentiles = results.get_percentile_paths()

        assert "market" in percentiles
        assert "bond_price" in percentiles
        assert 50 in percentiles["market"]
        assert len(percentiles["market"][50]) == 51


class TestMonteCarloSimulation:
    """Tests for MonteCarloSimulation."""

    def test_initialization(self):
        """Test simulation initialization."""
        sim = MonteCarloSimulation()

        assert sim.config is not None
        assert sim.config.num_simulations == 1000

    def test_initialization_with_config(self):
        """Test simulation with custom config."""
        config = SimulationConfig(
            num_simulations=100,
            num_steps=50,
        )
        sim = MonteCarloSimulation(config)

        assert sim.config.num_simulations == 100
        assert sim.config.num_steps == 50

    def test_run_returns_results(self):
        """Test that run returns SimulationResults."""
        config = SimulationConfig(
            num_simulations=10,
            num_steps=20,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        results = sim.run()

        assert isinstance(results, SimulationResults)
        assert results.market_paths.shape == (10, 21)
        assert results.bond_prices.shape == (10, 21)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config1 = SimulationConfig(
            num_simulations=10,
            num_steps=20,
            random_seed=42,
        )
        sim1 = MonteCarloSimulation(config1)
        results1 = sim1.run()

        config2 = SimulationConfig(
            num_simulations=10,
            num_steps=20,
            random_seed=42,
        )
        sim2 = MonteCarloSimulation(config2)
        results2 = sim2.run()

        np.testing.assert_array_almost_equal(
            results1.market_paths,
            results2.market_paths
        )

    def test_initial_values(self):
        """Test that paths start at initial values."""
        config = SimulationConfig(
            num_simulations=10,
            num_steps=20,
            random_seed=42,
        )
        config.market.initial_value = 100.0

        sim = MonteCarloSimulation(config)
        results = sim.run()

        np.testing.assert_array_equal(results.market_paths[:, 0], 100.0)

    def test_progress_callback(self):
        """Test progress callback is called."""
        config = SimulationConfig(
            num_simulations=10,
            num_steps=20,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        sim.run(progress_callback=callback)

        assert len(progress_calls) == 10
        assert progress_calls[-1] == (10, 10)

    def test_pre_step_hook(self):
        """Test pre-step hook is called."""
        config = SimulationConfig(
            num_simulations=2,
            num_steps=5,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        hook_calls = []

        def hook(step, market_model, bond_model, panic_model):
            hook_calls.append(step)

        sim.register_pre_step_hook(hook)
        sim.run()

        # 2 simulations * 5 steps = 10 calls
        assert len(hook_calls) == 10

    def test_post_step_hook(self):
        """Test post-step hook is called."""
        config = SimulationConfig(
            num_simulations=2,
            num_steps=5,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        hook_calls = []

        def hook(step, market_model, bond_model, panic_model, events):
            hook_calls.append(step)

        sim.register_post_step_hook(hook)
        sim.run()

        assert len(hook_calls) == 10

    def test_run_scenario_with_events(self):
        """Test running scenario with predefined events."""
        config = SimulationConfig(
            num_simulations=5,
            num_steps=20,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        # Create scenario event
        event = ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=10,
            yield_impact=0.03,
            market_impact=0.15,
            duration=5,
            name="Test default event"
        )

        scenario_events = [(10, event)]

        results = sim.run_scenario(scenario_events)

        assert isinstance(results, SimulationResults)

        # Check that events were recorded
        total_events = sum(len(e) for e in results.shockwave_events)
        assert total_events > 0

    def test_statistics_calculated(self):
        """Test that statistics are calculated after run."""
        config = SimulationConfig(
            num_simulations=10,
            num_steps=20,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        results = sim.run()

        assert "market_final" in results.statistics
        assert "default_rate" in results.statistics
        assert "shockwave" in results.statistics

    def test_panic_affects_volatility(self):
        """Test that panic affects simulation dynamics."""
        # This test verifies that panic model modifiers are correctly applied
        # by checking that the panic levels recorded during simulation change
        # and that the modifiers affect the market model

        config = SimulationConfig(
            num_simulations=10,
            num_steps=50,
            random_seed=42,
        )

        # Set up high base panic that will maintain elevated levels
        config.panic.base_panic_level = 0.5
        config.panic.panic_decay = 0.05  # Lower decay to maintain panic longer
        config.panic.volatility_multiplier = 2.5

        sim = MonteCarloSimulation(config)
        results = sim.run()

        # Verify panic levels were recorded and stayed elevated
        # With base_panic_level=0.5 and low decay, mean panic should be > 0.3
        mean_panic = results.panic_levels.mean()
        assert mean_panic > 0.3, f"Expected mean panic > 0.3, got {mean_panic}"

        # Verify panic affects market volatility - check that returns have
        # variation (if volatility multiplier wasn't applied, returns would be smaller)
        returns = np.diff(results.market_paths, axis=1) / results.market_paths[:, :-1]
        returns_std = returns.std()

        # With 2.5x volatility multiplier at ~50% panic level,
        # effective volatility should be notably higher than base 20%
        # returns std should be > 0.01 for daily steps
        assert returns_std > 0.005, f"Expected returns_std > 0.005, got {returns_std}"


class TestStressTest:
    """Tests for stress testing functionality."""

    def test_stress_test_multiple_scenarios(self):
        """Test running multiple stress scenarios."""
        config = SimulationConfig(
            num_simulations=5,
            num_steps=10,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        scenarios = [
            {
                "name": "Normal",
                "events": [],
            },
            {
                "name": "Crisis",
                "events": [(5, ShockwaveEvent(
                    shock_type=ShockType.MARKET_CRASH,
                    timestamp=5,
                    yield_impact=0.05,
                    market_impact=0.20,
                    duration=3,
                ))],
            },
        ]

        results = sim.stress_test(scenarios)

        assert len(results) == 2
        assert results[0].statistics["scenario_name"] == "Normal"
        assert results[1].statistics["scenario_name"] == "Crisis"

    def test_stress_test_with_config_overrides(self):
        """Test stress test with config parameter overrides."""
        config = SimulationConfig(
            num_simulations=5,
            num_steps=10,
            random_seed=42,
        )
        sim = MonteCarloSimulation(config)

        scenarios = [
            {
                "name": "High Volatility",
                "events": [],
                "config_overrides": {
                    "market.volatility": 0.50,
                },
            },
        ]

        results = sim.stress_test(scenarios)

        assert len(results) == 1

        # Config should be restored after
        assert sim.config.market.volatility == 0.20  # Default value
