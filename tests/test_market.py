"""Tests for market model."""

import numpy as np

from monte_carlo_finance.core.config import MarketConfig
from monte_carlo_finance.models.market import MarketModel, simulate_multiple_markets
from monte_carlo_finance.utils.random import RandomGenerator


class TestMarketModel:
    """Tests for MarketModel."""

    def test_initialization(self):
        """Test model initialization."""
        config = MarketConfig(initial_value=100.0)
        model = MarketModel(config)

        assert model.current_value == 100.0
        assert len(model.history) == 1

    def test_step_updates_value(self):
        """Test that step updates the current value."""
        config = MarketConfig(initial_value=100.0)
        rng = RandomGenerator(seed=42)
        model = MarketModel(config, rng)

        initial_value = model.current_value
        model.step(dt=1/252)

        assert model.current_value != initial_value
        assert len(model.history) == 2

    def test_simulate_path_length(self):
        """Test that simulate_path returns correct length."""
        config = MarketConfig(initial_value=100.0)
        model = MarketModel(config, RandomGenerator(seed=42))

        num_steps = 100
        path = model.simulate_path(num_steps, dt=1/252)

        assert len(path) == num_steps + 1  # Initial + num_steps
        assert path[0] == 100.0  # First value is initial

    def test_reset(self):
        """Test that reset restores initial state."""
        config = MarketConfig(initial_value=100.0)
        model = MarketModel(config, RandomGenerator(seed=42))

        model.simulate_path(50, dt=1/252)
        assert model.current_value != 100.0

        model.reset()
        assert model.current_value == 100.0
        assert len(model.history) == 1

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config = MarketConfig(initial_value=100.0)

        model1 = MarketModel(config, RandomGenerator(seed=42))
        path1 = model1.simulate_path(100, dt=1/252)

        model2 = MarketModel(config, RandomGenerator(seed=42))
        path2 = model2.simulate_path(100, dt=1/252)

        np.testing.assert_array_almost_equal(path1, path2)

    def test_shock_impact(self):
        """Test that shock impact reduces value."""
        config = MarketConfig(initial_value=100.0, volatility=0.0, drift=0.0)
        model = MarketModel(config, RandomGenerator(seed=42))

        # With no volatility and drift, shock should be the main driver
        model.step(dt=1/252, shock_impact=0.1)  # 10% shock

        # Value should decrease due to shock
        assert model.current_value < 100.0

    def test_positive_values(self):
        """Test that market values remain non-negative."""
        config = MarketConfig(
            initial_value=100.0,
            drift=-0.5,  # Strong negative drift
            volatility=0.5,  # High volatility
        )
        model = MarketModel(config, RandomGenerator(seed=42))

        # Run many steps
        path = model.simulate_path(500, dt=1/252)

        # All values should be non-negative
        assert np.all(path >= 0)

    def test_panic_modifiers(self):
        """Test that panic modifiers affect behavior."""
        config = MarketConfig(initial_value=100.0, volatility=0.2, drift=0.05)

        # Normal run using step-by-step (not simulate_path which resets)
        model1 = MarketModel(config, RandomGenerator(seed=42))
        for _ in range(100):
            model1.step(1/252)
        path1 = model1.history.copy()

        # Run with panic (high volatility, negative drift)
        model2 = MarketModel(config, RandomGenerator(seed=42))
        model2.set_panic_modifiers(volatility_modifier=2.0, drift_modifier=-0.10)
        for _ in range(100):
            model2.step(1/252)
        path2 = model2.history

        # Paths should be different due to different volatility and drift
        assert not np.allclose(path1, path2)

    def test_get_returns(self):
        """Test return calculation."""
        config = MarketConfig(initial_value=100.0)
        model = MarketModel(config, RandomGenerator(seed=42))

        model.simulate_path(100, dt=1/252)
        returns = model.get_returns()

        assert len(returns) == 100  # One less than path length

        # Verify calculation
        history = model.history
        expected_returns = np.diff(np.log(history))
        np.testing.assert_array_almost_equal(returns, expected_returns)

    def test_get_simple_returns(self):
        """Test simple return calculation."""
        config = MarketConfig(initial_value=100.0)
        model = MarketModel(config, RandomGenerator(seed=42))

        model.simulate_path(100, dt=1/252)
        returns = model.get_simple_returns()

        assert len(returns) == 100

        # Verify calculation
        history = model.history
        expected_returns = np.diff(history) / history[:-1]
        np.testing.assert_array_almost_equal(returns, expected_returns)


class TestSimulateMultipleMarkets:
    """Tests for simulate_multiple_markets function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        config = MarketConfig(initial_value=100.0)
        results = simulate_multiple_markets(
            config,
            num_simulations=10,
            num_steps=50,
            dt=1/252,
            seed=42
        )

        assert results.shape == (10, 51)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config = MarketConfig(initial_value=100.0)

        results1 = simulate_multiple_markets(
            config,
            num_simulations=10,
            num_steps=50,
            dt=1/252,
            seed=42
        )

        results2 = simulate_multiple_markets(
            config,
            num_simulations=10,
            num_steps=50,
            dt=1/252,
            seed=42
        )

        np.testing.assert_array_almost_equal(results1, results2)

    def test_initial_values(self):
        """Test that all paths start at initial value."""
        config = MarketConfig(initial_value=100.0)
        results = simulate_multiple_markets(
            config,
            num_simulations=10,
            num_steps=50,
            dt=1/252,
            seed=42
        )

        np.testing.assert_array_equal(results[:, 0], 100.0)
