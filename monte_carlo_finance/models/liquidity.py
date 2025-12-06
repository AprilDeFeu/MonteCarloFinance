"""
Global Liquidity Model.

Simulates the "tide" of global liquidity (Fed Balance Sheet, M2, etc.)
that drives asset correlations and the "Everything Bubble".
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from monte_carlo_finance.core.config import LiquidityConfig
from monte_carlo_finance.utils.random import RandomGenerator


@dataclass
class LiquidityState:
    level: float
    change: float
    regime: str  # "easing", "tightening", "neutral"

class LiquidityModel:
    """Simulates Global Liquidity (The "Water" in the bathtub)."""

    def __init__(
        self,
        config: LiquidityConfig,
        rng: Optional[RandomGenerator] = None
    ):
        self.config = config
        self.rng = rng or RandomGenerator()
        
        self._current_level = config.initial_level
        self._history: List[float] = [config.initial_level]
        self._regime = "neutral"

    @property
    def current_level(self) -> float:
        return self._current_level

    def step(self, dt: float, market_return: float, inflation_pressure: float = 0.0) -> float:
        """Evolve liquidity based on trend, noise, and Fed reaction function.
        
        Args:
            dt: Time step
            market_return: Recent market performance (triggers Fed Put if negative)
            inflation_pressure: Proxy for inflation risk (triggers tightening)
            
        Returns:
            Change in liquidity level
        """
        prev_level = self._current_level
        
        # 1. Base Dynamics (OU Process + Drift)
        # dL = theta * (mu - L) * dt + sigma * dW
        drift_term = self.config.drift * dt
        mean_rev = self.config.mean_reversion_speed * (1.0 - self._current_level) * dt
        noise = self.config.volatility * np.sqrt(dt) * self.rng.normal()
        
        # 2. Fed Reaction Function (The "Fed Put" vs "Fed Dilemma")
        fed_action = 0.0
        
        # Scenario A: Market Crash (Fed Put)
        if market_return < -0.05:  # 5% drop
            # Fed adds liquidity, BUT constrained by inflation
            intervention = abs(market_return) * self.config.fed_put_sensitivity
            
            # The Dilemma: If inflation/liquidity is already high, they can't print as much
            if self._current_level > self.config.inflation_constraint:
                intervention *= 0.2  # Handcuffed by inflation
                self._regime = "constrained_put"
            else:
                self._regime = "easing"
            
            fed_action += intervention

        # Scenario B: Overheating (Tightening)
        elif self._current_level > 1.2 or inflation_pressure > 0.05:
            fed_action -= 0.05 * dt  # QT (Quantitative Tightening)
            self._regime = "tightening"
        else:
            self._regime = "neutral"

        # Update Level
        self._current_level += drift_term + mean_rev + noise + fed_action
        
        # Ensure strictly positive (can't have negative money supply... theoretically)
        self._current_level = max(0.1, self._current_level)
        
        self._history.append(self._current_level)
        
        return self._current_level - prev_level

    def reset(self) -> None:
        """Reset the model to initial state."""
        self._current_level = self.config.initial_level
        self._history = [self.config.initial_level]
        self._regime = "neutral"
