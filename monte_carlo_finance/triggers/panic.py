"""
Market panic modeling system.

Implements quantifiable and adjustable market panic dynamics that
affect market behavior, including increased volatility, negative drift,
and heightened correlations during stress periods.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from monte_carlo_finance.core.config import PanicConfig
from monte_carlo_finance.utils.random import RandomGenerator


@dataclass
class PanicState:
    """Current state of market panic.

    Attributes:
        level: Current panic level (0-1)
        volatility_multiplier: Current volatility multiplier
        drift_modifier: Current drift modifier
        correlation_boost: Current correlation boost
        is_panicking: Whether market is in panic mode
    """
    level: float
    volatility_multiplier: float
    drift_modifier: float
    correlation_boost: float
    is_panicking: bool


class PanicModel:
    """Model for market panic dynamics.

    This model captures the psychological aspects of market behavior during
    stress periods, including:

    - Panic level that rises with adverse events
    - Increased volatility during panic
    - Negative drift effects (flight to safety)
    - Increased correlations (all assets moving together)
    - Herd behavior amplification
    """

    def __init__(
        self,
        config: PanicConfig,
        rng: Optional[RandomGenerator] = None
    ):
        """Initialize the panic model.

        Args:
            config: Panic configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng or RandomGenerator()

        # Current state
        self._panic_level = config.base_panic_level
        self._volatility_multiplier = 1.0
        self._drift_modifier = 0.0
        self._correlation_boost = 0.0

        # History
        self._panic_history: List[float] = [config.base_panic_level]

        # Event tracking
        self._adverse_events: List[dict] = []

    @property
    def panic_level(self) -> float:
        """Get current panic level."""
        return self._panic_level

    @property
    def panic_history(self) -> np.ndarray:
        """Get panic level history."""
        return np.array(self._panic_history)

    @property
    def is_panicking(self) -> bool:
        """Check if market is currently in panic mode."""
        return self._panic_level > self.config.recovery_threshold

    def get_state(self) -> PanicState:
        """Get current panic state.

        Returns:
            PanicState object with current values
        """
        return PanicState(
            level=self._panic_level,
            volatility_multiplier=self._volatility_multiplier,
            drift_modifier=self._drift_modifier,
            correlation_boost=self._correlation_boost,
            is_panicking=self.is_panicking
        )

    def update(
        self,
        market_return: float = 0.0,
        yield_change: float = 0.0,
        shockwave_intensity: float = 0.0,
        external_fear: float = 0.0
    ) -> PanicState:
        """Update panic level based on market events.

        Args:
            market_return: Recent market return (negative = bad)
            yield_change: Recent yield change (positive = bad for bonds)
            shockwave_intensity: Intensity of any shockwave events
            external_fear: External fear factor (0-1)

        Returns:
            Updated PanicState
        """
        # Calculate adverse event impact
        adverse_impact = 0.0

        # Negative market returns increase panic
        if market_return < 0:
            # Non-linear response to large losses
            adverse_impact += abs(market_return) ** 0.8 * 2.0

        # Large yield increases indicate stress
        if yield_change > 0:
            adverse_impact += yield_change * 5.0

        # Shockwave events directly increase panic
        adverse_impact += shockwave_intensity

        # External fear factors
        adverse_impact += external_fear

        # Apply sensitivity
        # Fragility factor amplifies the sensitivity
        sensitivity = self.config.panic_sensitivity
        sensitivity *= getattr(self.config, 'fragility_factor', 1.0)
        
        panic_increase = adverse_impact * sensitivity

        # Herd behavior amplification when panic is high
        if self.is_panicking:
            panic_increase *= (1 + self.config.herd_behavior_factor * self._panic_level)

        # Natural decay
        panic_decay = self._panic_level * self.config.panic_decay

        # Update panic level
        new_panic = self._panic_level + panic_increase - panic_decay

        # Clamp to valid range
        new_panic = max(0.0, min(new_panic, self.config.max_panic_level))

        self._panic_level = new_panic
        self._panic_history.append(new_panic)

        # Update derived values
        self._update_modifiers()

        # Record adverse event if significant
        if adverse_impact > 0.01:
            self._adverse_events.append({
                "panic_level": new_panic,
                "market_return": market_return,
                "yield_change": yield_change,
                "shockwave_intensity": shockwave_intensity,
            })

        return self.get_state()

    def _update_modifiers(self) -> None:
        """Update volatility, drift, and correlation modifiers based on panic level."""
        p = self._panic_level

        # Volatility multiplier: increases with panic
        # At max panic, volatility is multiplied by config.volatility_multiplier
        self._volatility_multiplier = 1.0 + (self.config.volatility_multiplier - 1.0) * p

        # Drift modifier: becomes more negative with panic
        # At max panic, drift is reduced by config.drift_impact
        self._drift_modifier = self.config.drift_impact * p

        # Correlation boost: increases with panic
        # At max panic, correlation increases by config.correlation_boost
        self._correlation_boost = self.config.correlation_boost * p

    def get_volatility_multiplier(self) -> float:
        """Get current volatility multiplier.

        Returns:
            Multiplier for base volatility (1.0 = no change)
        """
        return self._volatility_multiplier

    def get_drift_modifier(self) -> float:
        """Get current drift modifier.

        Returns:
            Additive modifier for drift (usually negative during panic)
        """
        return self._drift_modifier

    def get_correlation_boost(self) -> float:
        """Get current correlation boost.

        Returns:
            Boost to apply to asset correlations
        """
        return self._correlation_boost

    def inject_panic(self, level: float) -> None:
        """Directly set panic level (for scenario testing).

        Args:
            level: Panic level to set (0-1)
        """
        self._panic_level = max(0.0, min(level, self.config.max_panic_level))
        self._update_modifiers()
        self._panic_history.append(self._panic_level)

    def trigger_panic_event(
        self,
        name: str,
        intensity: float = 0.5
    ) -> PanicState:
        """Trigger a named panic event.

        Args:
            name: Name of the panic event
            intensity: Intensity of the event (0-1)

        Returns:
            Updated PanicState
        """
        # Panic events cause immediate jump
        jump = intensity * (1 - self._panic_level)  # Diminishing returns
        self._panic_level = min(
            self._panic_level + jump,
            self.config.max_panic_level
        )

        self._update_modifiers()
        self._panic_history.append(self._panic_level)

        self._adverse_events.append({
            "event_name": name,
            "intensity": intensity,
            "resulting_panic": self._panic_level,
        })

        return self.get_state()

    def step_decay(self) -> PanicState:
        """Apply one step of natural panic decay.

        Use this for time steps without significant events.

        Returns:
            Updated PanicState
        """
        # Natural decay
        self._panic_level *= (1 - self.config.panic_decay)

        # Don't go below base level
        self._panic_level = max(self._panic_level, self.config.base_panic_level)

        self._update_modifiers()
        self._panic_history.append(self._panic_level)

        return self.get_state()

    def reset(self) -> None:
        """Reset panic model to initial state."""
        self._panic_level = self.config.base_panic_level
        self._volatility_multiplier = 1.0
        self._drift_modifier = 0.0
        self._correlation_boost = 0.0
        self._panic_history = [self.config.base_panic_level]
        self._adverse_events = []

    def get_summary(self) -> dict:
        """Get summary statistics of panic dynamics.

        Returns:
            Dictionary with panic statistics
        """
        history = self.panic_history
        return {
            "current_level": self._panic_level,
            "max_level": float(np.max(history)),
            "mean_level": float(np.mean(history)),
            "time_in_panic": float(np.mean(history > self.config.recovery_threshold)),
            "adverse_events_count": len(self._adverse_events),
            "is_panicking": self.is_panicking,
        }


class FearGreedIndex:
    """Fear and Greed index for market sentiment tracking.

    This complements the panic model by tracking both fear (negative)
    and greed (positive) sentiment in the market.
    """

    def __init__(
        self,
        initial_value: float = 50.0,  # 0 = Extreme Fear, 100 = Extreme Greed
        sensitivity: float = 0.3,
        decay: float = 0.05
    ):
        """Initialize the Fear and Greed index.

        Args:
            initial_value: Starting index value (0-100)
            sensitivity: Sensitivity to market events
            decay: Rate of return to neutral (50)
        """
        self._value = initial_value
        self._sensitivity = sensitivity
        self._decay = decay
        self._history: List[float] = [initial_value]

    @property
    def value(self) -> float:
        """Get current index value."""
        return self._value

    @property
    def history(self) -> np.ndarray:
        """Get index history."""
        return np.array(self._history)

    @property
    def sentiment(self) -> str:
        """Get sentiment classification."""
        if self._value < 20:
            return "Extreme Fear"
        elif self._value < 40:
            return "Fear"
        elif self._value < 60:
            return "Neutral"
        elif self._value < 80:
            return "Greed"
        else:
            return "Extreme Greed"

    def update(
        self,
        market_return: float = 0.0,
        volatility_ratio: float = 1.0,  # Current vol / average vol
        volume_ratio: float = 1.0  # Current volume / average volume
    ) -> float:
        """Update the index based on market conditions.

        Args:
            market_return: Recent market return
            volatility_ratio: Current vs average volatility
            volume_ratio: Current vs average volume

        Returns:
            Updated index value
        """
        # Market return impact
        return_impact = market_return * 100 * self._sensitivity

        # High volatility indicates fear
        vol_impact = -(volatility_ratio - 1.0) * 10 * self._sensitivity

        # High volume with negative returns indicates fear
        if market_return < 0:
            volume_impact = -(volume_ratio - 1.0) * 5 * self._sensitivity
        else:
            volume_impact = (volume_ratio - 1.0) * 2 * self._sensitivity

        # Total impact
        total_impact = return_impact + vol_impact + volume_impact

        # Mean reversion to neutral
        mean_reversion = (50 - self._value) * self._decay

        # Update value
        self._value += total_impact + mean_reversion

        # Clamp to valid range
        self._value = max(0.0, min(100.0, self._value))

        self._history.append(self._value)

        return self._value

    def reset(self) -> None:
        """Reset to initial state."""
        self._value = 50.0
        self._history = [50.0]
