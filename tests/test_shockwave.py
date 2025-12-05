"""Tests for shockwave trigger system."""

from monte_carlo_finance.core.config import ShockwaveConfig
from monte_carlo_finance.triggers.shockwave import (
    ShockType,
    ShockwaveEvent,
    ShockwaveTrigger,
)
from monte_carlo_finance.utils.random import RandomGenerator


class TestShockwaveEvent:
    """Tests for ShockwaveEvent."""

    def test_initialization(self):
        """Test event initialization."""
        event = ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=10,
            yield_impact=0.02,
            market_impact=0.10,
            duration=5,
        )

        assert event.shock_type == ShockType.DEFAULT
        assert event.timestamp == 10
        assert event.yield_impact == 0.02
        assert event.market_impact == 0.10
        assert event.is_active

    def test_decay(self):
        """Test event decay over time."""
        event = ShockwaveEvent(
            shock_type=ShockType.MARKET_CRASH,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=3,
        )

        initial_yield_impact = event.current_yield_impact

        event.decay(decay_factor=0.5)

        assert event.current_yield_impact == initial_yield_impact * 0.5
        assert event._remaining_duration == 2
        assert event.is_active

    def test_becomes_inactive(self):
        """Test that event becomes inactive after duration."""
        event = ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=2,
        )

        event.decay()
        assert event.is_active

        event.decay()
        assert not event.is_active

    def test_reset(self):
        """Test event reset."""
        event = ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=5,
        )

        event.decay()
        event.decay()

        event.reset()

        assert event._remaining_duration == 5
        assert event._current_impact == 1.0


class TestShockwaveTrigger:
    """Tests for ShockwaveTrigger."""

    def test_initialization(self):
        """Test trigger system initialization."""
        config = ShockwaveConfig()
        trigger = ShockwaveTrigger(config, RandomGenerator(seed=42))

        assert len(trigger.active_events) == 0
        assert len(trigger.event_history) == 0
        assert len(trigger.selloff_orders) == 0

    def test_inject_event(self):
        """Test manual event injection."""
        config = ShockwaveConfig()
        trigger = ShockwaveTrigger(config)

        event = ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=5,
        )

        trigger.inject_event(event)

        assert len(trigger.active_events) == 1
        assert len(trigger.event_history) == 1

    def test_step_processes_events(self):
        """Test that step processes active events."""
        config = ShockwaveConfig(cascade_decay=0.5)
        trigger = ShockwaveTrigger(config)

        event = ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=3,
        )

        trigger.inject_event(event)

        yield_impact, market_impact = trigger.step()

        assert yield_impact > 0
        assert market_impact > 0

    def test_event_expires_after_duration(self):
        """Test that events expire after their duration."""
        config = ShockwaveConfig()
        trigger = ShockwaveTrigger(config)

        event = ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=2,
        )

        trigger.inject_event(event)

        trigger.step()
        assert len(trigger.active_events) == 1

        trigger.step()
        assert len(trigger.active_events) == 0

    def test_large_market_drop_triggers_event(self):
        """Test that large market drops trigger events."""
        config = ShockwaveConfig(base_probability=0.0)  # No random events
        trigger = ShockwaveTrigger(config, RandomGenerator(seed=42))

        # Trigger with large negative return
        events = trigger.check_triggers(
            market_value=100,
            market_return=-0.10,  # 10% drop
            bond_yield=0.05,
            yield_change=0.0,
            panic_level=0.0
        )

        assert len(events) > 0
        assert any(e.shock_type == ShockType.MARKET_CRASH for e in events)

    def test_yield_spike_triggers_event(self):
        """Test that yield spikes trigger events."""
        config = ShockwaveConfig(base_probability=0.0)  # No random events
        trigger = ShockwaveTrigger(config, RandomGenerator(seed=42))

        # Trigger with yield spike
        events = trigger.check_triggers(
            market_value=100,
            market_return=0.0,
            bond_yield=0.07,
            yield_change=0.03,  # 3% yield spike
            panic_level=0.0
        )

        assert len(events) > 0
        assert any(e.shock_type == ShockType.DEFAULT for e in events)

    def test_selloff_threshold_trigger(self):
        """Test that accumulated yield changes trigger selloff."""
        config = ShockwaveConfig(
            base_probability=0.0,
            selloff_threshold=0.03,
        )
        trigger = ShockwaveTrigger(config, RandomGenerator(seed=42))

        # Accumulate yield changes
        trigger.check_triggers(100, 0.0, 0.05, 0.015, 0.0)
        assert len(trigger.selloff_orders) == 0

        trigger.check_triggers(100, 0.0, 0.065, 0.016, 0.0)
        assert len(trigger.selloff_orders) == 1

    def test_create_default_event(self):
        """Test creating a default event."""
        config = ShockwaveConfig()
        trigger = ShockwaveTrigger(config)

        event = trigger.create_default_event(
            asset_name="Commercial Building",
            severity=1.5
        )

        assert event.shock_type == ShockType.DEFAULT
        assert "Commercial Building" in event.name
        assert event.metadata["severity"] == 1.5

    def test_custom_trigger(self):
        """Test registering and using custom triggers."""
        config = ShockwaveConfig(base_probability=0.0)
        trigger = ShockwaveTrigger(config, RandomGenerator(seed=42))

        # Create custom trigger
        def custom_trigger(state):
            if state["market_value"] < 90:
                return ShockwaveEvent(
                    shock_type=ShockType.CUSTOM,
                    timestamp=state["step"],
                    yield_impact=0.01,
                    market_impact=0.05,
                    duration=2,
                    name="Custom threshold trigger"
                )
            return None

        trigger.register_custom_trigger(custom_trigger)

        # Should not trigger at normal value
        events = trigger.check_triggers(100, 0.0, 0.05, 0.0, 0.0)
        custom_events = [e for e in events if e.shock_type == ShockType.CUSTOM]
        assert len(custom_events) == 0

        # Should trigger at low value
        events = trigger.check_triggers(85, 0.0, 0.05, 0.0, 0.0)
        custom_events = [e for e in events if e.shock_type == ShockType.CUSTOM]
        assert len(custom_events) == 1

    def test_reset(self):
        """Test trigger system reset."""
        config = ShockwaveConfig()
        trigger = ShockwaveTrigger(config)

        trigger.inject_event(ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=5,
        ))

        trigger.step()
        trigger.step()

        trigger.reset()

        assert len(trigger.active_events) == 0
        assert len(trigger.event_history) == 0
        assert len(trigger.selloff_orders) == 0

    def test_get_summary(self):
        """Test summary statistics."""
        config = ShockwaveConfig()
        trigger = ShockwaveTrigger(config)

        trigger.inject_event(ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=0,
            yield_impact=0.02,
            market_impact=0.10,
            duration=5,
        ))
        trigger.inject_event(ShockwaveEvent(
            shock_type=ShockType.MARKET_CRASH,
            timestamp=0,
            yield_impact=0.01,
            market_impact=0.05,
            duration=3,
        ))

        summary = trigger.get_summary()

        assert summary["total_events"] == 2
        assert summary["active_events"] == 2
        assert "default" in summary["events_by_type"]
        assert "market_crash" in summary["events_by_type"]
