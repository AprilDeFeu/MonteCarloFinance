"""
Shockwave event and trigger system.

Implements cascade effects where one event (e.g., a default on a building loan)
can trigger a chain reaction affecting bond yields and causing automated selloffs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from monte_carlo_finance.core.config import ShockwaveConfig
from monte_carlo_finance.utils.random import RandomGenerator


class ShockType(Enum):
    """Types of shockwave events."""
    DEFAULT = "default"  # Credit default event
    LIQUIDITY = "liquidity"  # Liquidity crisis
    REGULATORY = "regulatory"  # Regulatory change
    GEOPOLITICAL = "geopolitical"  # Geopolitical event
    MARKET_CRASH = "market_crash"  # General market crash
    SECTOR_SPECIFIC = "sector_specific"  # Sector-specific shock
    CUSTOM = "custom"  # User-defined shock


@dataclass
class ShockwaveEvent:
    """Represents a single shockwave event.

    Attributes:
        shock_type: Type of shock event
        timestamp: When the shock occurred (time step)
        yield_impact: Impact on bond yields (additive)
        market_impact: Impact on market values (multiplicative)
        duration: How long the shock persists (time steps)
        propagation_factor: How much the shock propagates to connected assets
        name: Optional descriptive name
        metadata: Additional event metadata
    """
    shock_type: ShockType
    timestamp: int
    yield_impact: float
    market_impact: float
    duration: int = 1
    propagation_factor: float = 0.5
    name: str = ""
    metadata: dict = field(default_factory=dict)

    # Internal tracking
    _remaining_duration: int = field(default=-1, init=False)
    _current_impact: float = field(default=1.0, init=False)

    def __post_init__(self):
        self._remaining_duration = self.duration
        self._current_impact = 1.0

    def decay(self, decay_factor: float = 0.7) -> None:
        """Apply decay to the shock impact.

        Args:
            decay_factor: Factor by which impact decays each step
        """
        self._remaining_duration -= 1
        self._current_impact *= decay_factor

    @property
    def is_active(self) -> bool:
        """Check if shock is still active."""
        return self._remaining_duration > 0

    @property
    def current_yield_impact(self) -> float:
        """Get current yield impact after decay."""
        return self.yield_impact * self._current_impact

    @property
    def current_market_impact(self) -> float:
        """Get current market impact after decay."""
        return self.market_impact * self._current_impact

    def reset(self) -> None:
        """Reset shock to initial state."""
        self._remaining_duration = self.duration
        self._current_impact = 1.0


@dataclass
class SelloffOrder:
    """Represents an automated selloff order.

    Attributes:
        triggered_at: Time step when triggered
        asset_type: Type of asset being sold
        intensity: Fraction of position to sell
        reason: Trigger reason
    """
    triggered_at: int
    asset_type: str
    intensity: float
    reason: str


class ShockwaveTrigger:
    """System for managing shockwave events and cascade effects.

    This class monitors market conditions and triggers shockwave events
    based on configurable thresholds and probabilities.
    """

    def __init__(
        self,
        config: ShockwaveConfig,
        rng: Optional[RandomGenerator] = None
    ):
        """Initialize the trigger system.

        Args:
            config: Shockwave configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng or RandomGenerator()

        # Active events
        self._active_events: List[ShockwaveEvent] = []

        # Event history
        self._event_history: List[ShockwaveEvent] = []

        # Selloff orders
        self._selloff_orders: List[SelloffOrder] = []

        # Custom trigger functions
        self._custom_triggers: List[Callable[..., Optional[ShockwaveEvent]]] = []

        # State tracking
        self._current_step = 0
        self._cumulative_yield_change = 0.0

    @property
    def active_events(self) -> List[ShockwaveEvent]:
        """Get list of currently active shockwave events."""
        return self._active_events

    @property
    def event_history(self) -> List[ShockwaveEvent]:
        """Get history of all events."""
        return self._event_history

    @property
    def selloff_orders(self) -> List[SelloffOrder]:
        """Get list of triggered selloff orders."""
        return self._selloff_orders

    def register_custom_trigger(
        self,
        trigger_func: Callable[..., Optional[ShockwaveEvent]]
    ) -> None:
        """Register a custom trigger function.

        The function should accept (step, market_state, bond_state) and
        return a ShockwaveEvent if triggered, or None otherwise.

        Args:
            trigger_func: Custom trigger function
        """
        self._custom_triggers.append(trigger_func)

    def check_triggers(
        self,
        market_value: float,
        market_return: float,
        bond_yield: float,
        yield_change: float,
        panic_level: float = 0.0
    ) -> List[ShockwaveEvent]:
        """Check all triggers and generate events.

        Args:
            market_value: Current market value
            market_return: Recent market return
            bond_yield: Current bond yield
            yield_change: Recent yield change
            panic_level: Current panic level (0-1)

        Returns:
            List of newly triggered events
        """
        new_events = []

        # Track cumulative yield change
        self._cumulative_yield_change += yield_change

        # Random shock based on base probability
        # Probability increases with panic level
        adjusted_prob = self.config.base_probability * (1 + panic_level)

        if self.rng.uniform() < adjusted_prob:
            event = ShockwaveEvent(
                shock_type=ShockType.MARKET_CRASH,
                timestamp=self._current_step,
                yield_impact=self.config.yield_impact,
                market_impact=self.config.market_impact,
                duration=self.config.propagation_delay + 3,
                propagation_factor=self.config.cascade_decay,
                name=f"Random shock at step {self._current_step}"
            )
            new_events.append(event)

        # Large negative return trigger
        if market_return < -0.05:  # 5% drop
            intensity = min(abs(market_return) / 0.10, 1.0)  # Scale up to 10% drop
            event = ShockwaveEvent(
                shock_type=ShockType.MARKET_CRASH,
                timestamp=self._current_step,
                yield_impact=self.config.yield_impact * intensity,
                market_impact=self.config.market_impact * intensity * 0.5,
                duration=5,
                propagation_factor=self.config.cascade_decay,
                name=f"Market drop trigger ({market_return:.2%})"
            )
            new_events.append(event)

        # Yield spike trigger (potential default signal)
        if yield_change > 0.02:  # 2% yield spike
            event = ShockwaveEvent(
                shock_type=ShockType.DEFAULT,
                timestamp=self._current_step,
                yield_impact=yield_change * 0.5,  # Amplify yield increase
                market_impact=self.config.market_impact * 0.3,
                duration=7,
                propagation_factor=self.config.cascade_decay,
                name=f"Yield spike trigger ({yield_change:.2%})"
            )
            new_events.append(event)

        # Check selloff threshold
        if self._cumulative_yield_change > self.config.selloff_threshold:
            selloff = SelloffOrder(
                triggered_at=self._current_step,
                asset_type="bond",
                intensity=self.config.selloff_intensity,
                reason=f"Yield threshold exceeded ({self._cumulative_yield_change:.2%})"
            )
            self._selloff_orders.append(selloff)
            self._cumulative_yield_change = 0.0  # Reset after trigger

            # Selloff causes additional market impact
            event = ShockwaveEvent(
                shock_type=ShockType.LIQUIDITY,
                timestamp=self._current_step,
                yield_impact=0.005,  # Small yield increase from selling
                market_impact=self.config.selloff_intensity * 0.1,
                duration=3,
                propagation_factor=0.5,
                name="Automated selloff cascade"
            )
            new_events.append(event)

        # Custom triggers
        state = {
            "market_value": market_value,
            "market_return": market_return,
            "bond_yield": bond_yield,
            "yield_change": yield_change,
            "panic_level": panic_level,
            "step": self._current_step,
        }

        for trigger_func in self._custom_triggers:
            result = trigger_func(state)
            if result is not None:
                new_events.append(result)

        # Add new events
        for event in new_events:
            self._active_events.append(event)
            self._event_history.append(event)

        return new_events

    def step(self) -> tuple:
        """Advance by one time step and calculate cumulative impacts.

        Returns:
            Tuple of (total_yield_impact, total_market_impact)
        """
        self._current_step += 1

        total_yield_impact = 0.0
        total_market_impact = 0.0

        # Process active events
        still_active = []
        for event in self._active_events:
            if event.is_active:
                # Scale impact by duration to spread the effect over the event lifetime
                # This prevents the full impact from being applied every single step
                duration_factor = 1.0 / max(event.duration, 1)
                total_yield_impact += event.current_yield_impact * duration_factor
                total_market_impact += event.current_market_impact * duration_factor
                event.decay(self.config.cascade_decay)
                if event.is_active:
                    still_active.append(event)

        self._active_events = still_active

        return (total_yield_impact, total_market_impact)

    def inject_event(self, event: ShockwaveEvent) -> None:
        """Manually inject a shockwave event.

        Args:
            event: Event to inject
        """
        event.timestamp = self._current_step
        self._active_events.append(event)
        self._event_history.append(event)

    def create_default_event(
        self,
        asset_name: str = "Building",
        severity: float = 1.0
    ) -> ShockwaveEvent:
        """Create a default event (e.g., building loan default).

        Args:
            asset_name: Name of the defaulting asset
            severity: Severity multiplier (1.0 = normal)

        Returns:
            Configured ShockwaveEvent
        """
        return ShockwaveEvent(
            shock_type=ShockType.DEFAULT,
            timestamp=self._current_step,
            yield_impact=self.config.yield_impact * severity * 1.5,
            market_impact=self.config.market_impact * severity,
            duration=10,
            propagation_factor=self.config.cascade_decay,
            name=f"{asset_name} default event",
            metadata={"asset": asset_name, "severity": severity}
        )

    def reset(self) -> None:
        """Reset the trigger system."""
        self._active_events = []
        self._event_history = []
        self._selloff_orders = []
        self._current_step = 0
        self._cumulative_yield_change = 0.0

    def get_summary(self) -> dict:
        """Get summary statistics of shockwave events.

        Returns:
            Dictionary with event statistics
        """
        return {
            "total_events": len(self._event_history),
            "active_events": len(self._active_events),
            "selloff_orders": len(self._selloff_orders),
            "events_by_type": self._count_by_type(),
        }

    def _count_by_type(self) -> dict:
        """Count events by type."""
        counts = {}
        for event in self._event_history:
            type_name = event.shock_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
