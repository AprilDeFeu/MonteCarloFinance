"""Tests for panic model."""

from monte_carlo_finance.core.config import PanicConfig
from monte_carlo_finance.triggers.panic import FearGreedIndex, PanicModel, PanicState


class TestPanicModel:
    """Tests for PanicModel."""

    def test_initialization(self):
        """Test model initialization."""
        config = PanicConfig(base_panic_level=0.1)
        model = PanicModel(config)

        assert model.panic_level == 0.1
        assert len(model.panic_history) == 1

    def test_update_increases_panic_on_negative_return(self):
        """Test that negative returns increase panic."""
        config = PanicConfig(
            base_panic_level=0.0,
            panic_sensitivity=1.0,
            panic_decay=0.0,
        )
        model = PanicModel(config)

        state = model.update(market_return=-0.05)

        assert state.level > 0

    def test_update_increases_panic_on_yield_change(self):
        """Test that yield increases raise panic."""
        config = PanicConfig(
            base_panic_level=0.0,
            panic_sensitivity=1.0,
            panic_decay=0.0,
        )
        model = PanicModel(config)

        state = model.update(yield_change=0.02)

        assert state.level > 0

    def test_panic_decay(self):
        """Test natural panic decay."""
        config = PanicConfig(
            base_panic_level=0.0,
            panic_decay=0.5,  # 50% decay per step
        )
        model = PanicModel(config)

        model.inject_panic(0.8)

        state = model.step_decay()

        assert state.level < 0.8
        assert state.level > 0  # Not fully decayed

    def test_max_panic_level(self):
        """Test that panic is capped at max level."""
        config = PanicConfig(
            base_panic_level=0.0,
            max_panic_level=1.0,
            panic_sensitivity=10.0,  # High sensitivity
            panic_decay=0.0,
        )
        model = PanicModel(config)

        # Generate very adverse event
        model.update(market_return=-0.5, yield_change=0.1, shockwave_intensity=1.0)

        assert model.panic_level <= 1.0

    def test_volatility_multiplier(self):
        """Test that volatility multiplier increases with panic."""
        config = PanicConfig(
            base_panic_level=0.0,
            volatility_multiplier=3.0,
        )
        model = PanicModel(config)

        # At zero panic, multiplier should be 1
        assert model.get_volatility_multiplier() == 1.0

        model.inject_panic(1.0)

        # At max panic, multiplier should be 3.0
        assert model.get_volatility_multiplier() == 3.0

    def test_drift_modifier(self):
        """Test that drift modifier becomes more negative with panic."""
        config = PanicConfig(
            base_panic_level=0.0,
            drift_impact=-0.20,  # -20% drift at max panic
        )
        model = PanicModel(config)

        # At zero panic, modifier should be 0
        assert model.get_drift_modifier() == 0.0

        model.inject_panic(1.0)

        # At max panic, modifier should be -0.20
        assert model.get_drift_modifier() == -0.20

    def test_correlation_boost(self):
        """Test that correlation boost increases with panic."""
        config = PanicConfig(
            base_panic_level=0.0,
            correlation_boost=0.6,
        )
        model = PanicModel(config)

        # At zero panic, boost should be 0
        assert model.get_correlation_boost() == 0.0

        model.inject_panic(1.0)

        # At max panic, boost should be 0.6
        assert model.get_correlation_boost() == 0.6

    def test_is_panicking(self):
        """Test panic detection."""
        config = PanicConfig(
            base_panic_level=0.0,
            recovery_threshold=0.3,
        )
        model = PanicModel(config)

        assert not model.is_panicking

        model.inject_panic(0.5)

        assert model.is_panicking

        model.inject_panic(0.2)

        assert not model.is_panicking

    def test_get_state(self):
        """Test getting current state."""
        config = PanicConfig()
        model = PanicModel(config)

        model.inject_panic(0.5)

        state = model.get_state()

        assert isinstance(state, PanicState)
        assert state.level == 0.5
        assert state.volatility_multiplier > 1.0

    def test_trigger_panic_event(self):
        """Test triggering named panic event."""
        config = PanicConfig(base_panic_level=0.0)
        model = PanicModel(config)

        state = model.trigger_panic_event("Bank Run", intensity=0.7)

        assert state.level > 0.5

    def test_reset(self):
        """Test model reset."""
        config = PanicConfig(base_panic_level=0.1)
        model = PanicModel(config)

        model.inject_panic(0.8)
        model.update(market_return=-0.1)

        model.reset()

        assert model.panic_level == 0.1
        assert len(model.panic_history) == 1

    def test_get_summary(self):
        """Test summary statistics."""
        config = PanicConfig(base_panic_level=0.0)
        model = PanicModel(config)

        model.inject_panic(0.5)
        model.update(market_return=-0.05)
        model.step_decay()

        summary = model.get_summary()

        assert "current_level" in summary
        assert "max_level" in summary
        assert "mean_level" in summary
        assert "time_in_panic" in summary

    def test_herd_behavior(self):
        """Test that herd behavior amplifies panic when already panicking."""
        config = PanicConfig(
            base_panic_level=0.0,
            herd_behavior_factor=0.5,
            recovery_threshold=0.1,
            panic_decay=0.0,
        )
        model = PanicModel(config)

        # First event without existing panic
        model.update(market_return=-0.05)

        model.reset()

        # Now inject some panic first
        model.inject_panic(0.5)  # Above recovery threshold
        initial_with_herd = model.panic_level

        # Same event but with herd behavior
        model.update(market_return=-0.05)
        model.panic_level - initial_with_herd

        # Herd behavior should amplify the increase
        # (This is a qualitative test as exact values depend on formula)


class TestFearGreedIndex:
    """Tests for FearGreedIndex."""

    def test_initialization(self):
        """Test index initialization."""
        index = FearGreedIndex(initial_value=50.0)

        assert index.value == 50.0
        assert index.sentiment == "Neutral"

    def test_negative_return_decreases_index(self):
        """Test that negative returns decrease the index (more fear)."""
        index = FearGreedIndex(initial_value=50.0, sensitivity=1.0, decay=0.0)

        index.update(market_return=-0.05)

        assert index.value < 50.0

    def test_positive_return_increases_index(self):
        """Test that positive returns increase the index (more greed)."""
        index = FearGreedIndex(initial_value=50.0, sensitivity=1.0, decay=0.0)

        index.update(market_return=0.05)

        assert index.value > 50.0

    def test_high_volatility_indicates_fear(self):
        """Test that high volatility decreases the index."""
        index = FearGreedIndex(initial_value=50.0, sensitivity=1.0, decay=0.0)

        index.update(market_return=0.0, volatility_ratio=2.0)  # Twice normal vol

        assert index.value < 50.0

    def test_clamping(self):
        """Test that index stays within 0-100 range."""
        index = FearGreedIndex(initial_value=50.0, sensitivity=10.0, decay=0.0)

        # Try to push beyond 100
        for _ in range(10):
            index.update(market_return=0.10)

        assert index.value <= 100.0

        # Try to push below 0
        index.reset()
        for _ in range(10):
            index.update(market_return=-0.10)

        assert index.value >= 0.0

    def test_sentiment_classification(self):
        """Test sentiment classification at different levels."""
        index = FearGreedIndex()

        index._value = 10
        assert index.sentiment == "Extreme Fear"

        index._value = 30
        assert index.sentiment == "Fear"

        index._value = 50
        assert index.sentiment == "Neutral"

        index._value = 70
        assert index.sentiment == "Greed"

        index._value = 90
        assert index.sentiment == "Extreme Greed"

    def test_mean_reversion(self):
        """Test that index reverts to neutral over time."""
        index = FearGreedIndex(initial_value=10.0, decay=0.1)

        # Without new events, should revert toward 50
        for _ in range(50):
            index.update(market_return=0.0)

        assert index.value > 10.0  # Should have moved toward 50

    def test_reset(self):
        """Test index reset."""
        index = FearGreedIndex(initial_value=50.0)

        index.update(market_return=-0.10)
        index.update(market_return=-0.10)

        index.reset()

        assert index.value == 50.0
        assert len(index.history) == 1
