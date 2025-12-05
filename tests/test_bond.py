"""Tests for bond model."""

import numpy as np

from monte_carlo_finance.core.config import BondConfig
from monte_carlo_finance.models.bond import BondModel, simulate_multiple_bonds
from monte_carlo_finance.utils.random import RandomGenerator


class TestBondModel:
    """Tests for BondModel."""

    def test_initialization(self):
        """Test model initialization."""
        config = BondConfig(
            face_value=1000.0,
            coupon_rate=0.05,
            initial_yield=0.05,
        )
        model = BondModel(config)

        assert model.current_yield == 0.05
        assert model.remaining_maturity == config.maturity_years
        assert not model.is_defaulted

    def test_initial_price_calculation(self):
        """Test that initial price is calculated correctly."""
        config = BondConfig(
            face_value=1000.0,
            coupon_rate=0.05,
            initial_yield=0.05,
            credit_spread=0.0,  # No spread for simple test
            maturity_years=10.0,
        )
        model = BondModel(config)

        # With yield = coupon rate and no spread, price should be near par
        # (Not exactly par due to discrete coupon payments)
        assert 900 < model.current_price < 1100

    def test_step_updates_state(self):
        """Test that step updates the state."""
        config = BondConfig()
        rng = RandomGenerator(seed=42)
        model = BondModel(config, rng)

        initial_maturity = model.remaining_maturity

        model.step(dt=1/252)

        # State should change
        assert model.remaining_maturity < initial_maturity
        assert len(model.yield_history) == 2
        assert len(model.price_history) == 2

    def test_simulate_path_length(self):
        """Test that simulate_path returns correct length."""
        config = BondConfig()
        model = BondModel(config, RandomGenerator(seed=42))

        num_steps = 100
        prices, yields = model.simulate_path(num_steps, dt=1/252)

        assert len(prices) == num_steps + 1
        assert len(yields) == num_steps + 1

    def test_reset(self):
        """Test that reset restores initial state."""
        config = BondConfig()
        model = BondModel(config, RandomGenerator(seed=42))

        model.simulate_path(50, dt=1/252)

        model.reset()

        assert model.current_yield == config.initial_yield
        assert model.remaining_maturity == config.maturity_years
        assert not model.is_defaulted
        assert len(model.yield_history) == 1

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config = BondConfig()

        model1 = BondModel(config, RandomGenerator(seed=42))
        prices1, yields1 = model1.simulate_path(100, dt=1/252)

        model2 = BondModel(config, RandomGenerator(seed=42))
        prices2, yields2 = model2.simulate_path(100, dt=1/252)

        np.testing.assert_array_almost_equal(prices1, prices2)
        np.testing.assert_array_almost_equal(yields1, yields2)

    def test_yield_shock_impact(self):
        """Test that yield shock affects the yield."""
        config = BondConfig(yield_volatility=0.0)  # No random volatility
        model = BondModel(config, RandomGenerator(seed=42))

        initial_yield = model.current_yield
        model.step(dt=1/252, yield_shock=0.01, check_default=False)

        # Yield should increase due to shock (plus mean reversion)
        # The exact change depends on mean reversion
        assert model.current_yield != initial_yield

    def test_maturity_decreases(self):
        """Test that remaining maturity decreases over time."""
        config = BondConfig(maturity_years=10.0)
        model = BondModel(config, RandomGenerator(seed=42))

        dt = 1/252
        for _ in range(252):  # One year
            model.step(dt, check_default=False)

        # After one year, maturity should decrease by ~1 year
        assert abs(model.remaining_maturity - 9.0) < 0.01

    def test_duration_calculation(self):
        """Test duration calculation."""
        config = BondConfig(
            face_value=1000.0,
            coupon_rate=0.05,
            maturity_years=10.0,
            initial_yield=0.05,
            credit_spread=0.0,
        )
        model = BondModel(config)

        duration = model.get_duration()

        # Duration should be positive and less than maturity
        assert 0 < duration < config.maturity_years

    def test_modified_duration(self):
        """Test modified duration calculation."""
        config = BondConfig()
        model = BondModel(config)

        mac_duration = model.get_duration()
        mod_duration = model.get_modified_duration()

        # Modified duration should be less than Macaulay duration
        assert mod_duration < mac_duration

    def test_convexity(self):
        """Test convexity calculation."""
        config = BondConfig()
        model = BondModel(config)

        convexity = model.get_convexity()

        # Convexity should be positive
        assert convexity > 0

    def test_panic_modifiers(self):
        """Test that panic modifiers affect behavior."""
        config = BondConfig()

        # Normal run using step-by-step (not simulate_path which resets)
        model1 = BondModel(config, RandomGenerator(seed=42))
        for _ in range(100):
            model1.step(1/252, check_default=False)
        prices1 = model1.price_history.copy()

        # Run with panic modifiers
        model2 = BondModel(config, RandomGenerator(seed=42))
        model2.set_panic_modifiers(
            yield_volatility_modifier=2.0,
            spread_modifier=0.02
        )
        for _ in range(100):
            model2.step(1/252, check_default=False)
        prices2 = model2.price_history

        # Paths should be different
        assert not np.allclose(prices1, prices2)


class TestSimulateMultipleBonds:
    """Tests for simulate_multiple_bonds function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        config = BondConfig()
        prices, yields = simulate_multiple_bonds(
            config,
            num_simulations=10,
            num_steps=50,
            dt=1/252,
            seed=42
        )

        assert prices.shape == (10, 51)
        assert yields.shape == (10, 51)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config = BondConfig()

        prices1, yields1 = simulate_multiple_bonds(
            config,
            num_simulations=10,
            num_steps=50,
            dt=1/252,
            seed=42
        )

        prices2, yields2 = simulate_multiple_bonds(
            config,
            num_simulations=10,
            num_steps=50,
            dt=1/252,
            seed=42
        )

        np.testing.assert_array_almost_equal(prices1, prices2)
        np.testing.assert_array_almost_equal(yields1, yields2)
