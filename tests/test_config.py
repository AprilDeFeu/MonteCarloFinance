"""Tests for configuration module."""

from monte_carlo_finance.core.config import (
    BondConfig,
    MarketConfig,
    PanicConfig,
    ShockwaveConfig,
    SimulationConfig,
)


class TestMarketConfig:
    """Tests for MarketConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = MarketConfig()
        assert config.initial_value == 100.0
        assert config.drift == 0.05
        assert config.volatility == 0.20
        assert config.jump_intensity == 0.1
        assert config.jump_mean == -0.05
        assert config.jump_std == 0.10

    def test_custom_values(self):
        """Test custom value assignment."""
        config = MarketConfig(
            initial_value=200.0,
            drift=0.08,
            volatility=0.30,
        )
        assert config.initial_value == 200.0
        assert config.drift == 0.08
        assert config.volatility == 0.30


class TestBondConfig:
    """Tests for BondConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = BondConfig()
        assert config.face_value == 1000.0
        assert config.coupon_rate == 0.05
        assert config.maturity_years == 10.0
        assert config.recovery_rate == 0.40

    def test_custom_values(self):
        """Test custom value assignment."""
        config = BondConfig(
            face_value=5000.0,
            coupon_rate=0.03,
            maturity_years=5.0,
        )
        assert config.face_value == 5000.0
        assert config.coupon_rate == 0.03
        assert config.maturity_years == 5.0


class TestShockwaveConfig:
    """Tests for ShockwaveConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ShockwaveConfig()
        assert config.base_probability == 0.01
        assert config.yield_impact == 0.02
        assert config.market_impact == 0.10
        assert config.selloff_threshold == 0.05

    def test_custom_values(self):
        """Test custom value assignment."""
        config = ShockwaveConfig(
            base_probability=0.05,
            yield_impact=0.04,
        )
        assert config.base_probability == 0.05
        assert config.yield_impact == 0.04


class TestPanicConfig:
    """Tests for PanicConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = PanicConfig()
        assert config.base_panic_level == 0.0
        assert config.panic_sensitivity == 0.5
        assert config.max_panic_level == 1.0
        assert config.volatility_multiplier == 2.0

    def test_custom_values(self):
        """Test custom value assignment."""
        config = PanicConfig(
            base_panic_level=0.1,
            panic_sensitivity=0.8,
            volatility_multiplier=3.0,
        )
        assert config.base_panic_level == 0.1
        assert config.panic_sensitivity == 0.8
        assert config.volatility_multiplier == 3.0


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SimulationConfig()
        assert config.num_simulations == 1000
        assert config.num_steps == 252
        assert config.time_horizon == 1.0
        assert config.random_seed is None

    def test_dt_calculation(self):
        """Test time step calculation."""
        config = SimulationConfig(num_steps=252, time_horizon=1.0)
        assert abs(config.dt - 1.0 / 252) < 1e-10

        config = SimulationConfig(num_steps=52, time_horizon=2.0)
        assert abs(config.dt - 2.0 / 52) < 1e-10

    def test_nested_configs(self):
        """Test that nested configs are properly initialized."""
        config = SimulationConfig()
        assert isinstance(config.market, MarketConfig)
        assert isinstance(config.bond, BondConfig)
        assert isinstance(config.shockwave, ShockwaveConfig)
        assert isinstance(config.panic, PanicConfig)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimulationConfig()
        d = config.to_dict()

        assert d["num_simulations"] == 1000
        assert d["num_steps"] == 252
        assert "market" in d
        assert "bond" in d
        assert "shockwave" in d
        assert "panic" in d
        assert d["market"]["initial_value"] == 100.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "num_simulations": 500,
            "num_steps": 100,
            "market": {"initial_value": 150.0},
            "bond": {"face_value": 2000.0},
        }
        config = SimulationConfig.from_dict(d)

        assert config.num_simulations == 500
        assert config.num_steps == 100
        assert config.market.initial_value == 150.0
        assert config.bond.face_value == 2000.0

    def test_round_trip(self):
        """Test that to_dict and from_dict are consistent."""
        original = SimulationConfig(
            num_simulations=500,
            time_horizon=2.0,
        )
        d = original.to_dict()
        restored = SimulationConfig.from_dict(d)

        assert restored.num_simulations == original.num_simulations
        assert restored.time_horizon == original.time_horizon
        assert restored.market.initial_value == original.market.initial_value
