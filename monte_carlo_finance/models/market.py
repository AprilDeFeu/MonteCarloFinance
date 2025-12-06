"""
Market value simulation model.

Implements Geometric Brownian Motion (GBM) with jump diffusion for
realistic market value simulation, including fat-tail events.
"""

from typing import Optional

import numpy as np

from monte_carlo_finance.core.config import MarketConfig
from monte_carlo_finance.utils.random import RandomGenerator


class MarketModel:
    """Market value simulation using jump-diffusion process.

    This model combines:
    - Geometric Brownian Motion (GBM) for continuous price evolution
    - Jump-diffusion (Merton model) for sudden price movements

    The model can be parameterized to simulate various market conditions,
    from calm periods to high-volatility crisis scenarios.
    """

    def __init__(
        self,
        config: MarketConfig,
        rng: Optional[RandomGenerator] = None
    ):
        """Initialize the market model.

        Args:
            config: Market configuration parameters
            rng: Random number generator (creates new one if not provided)
        """
        self.config = config
        self.rng = rng or RandomGenerator()

        # Current state
        self._current_value = config.initial_value
        self._history: list = [config.initial_value]

        # Panic modifiers (can be updated externally)
        self._volatility_modifier = 1.0
        self._drift_modifier = 0.0

    @property
    def current_value(self) -> float:
        """Get the current market value."""
        return self._current_value

    @property
    def history(self) -> np.ndarray:
        """Get the value history."""
        return np.array(self._history)

    def set_panic_modifiers(
        self,
        volatility_modifier: float = 1.0,
        drift_modifier: float = 0.0
    ) -> None:
        """Set panic-related modifiers for market dynamics.

        Args:
            volatility_modifier: Multiplier for volatility (1.0 = normal)
            drift_modifier: Additive modifier for drift
        """
        self._volatility_modifier = volatility_modifier
        self._drift_modifier = drift_modifier

    def step(
        self,
        dt: float,
        shock_impact: float = 0.0,
        liquidity_return: float = 0.0
    ) -> float:
        """Advance the market by one time step.

        Uses the jump-diffusion model:
        dS = (mu - lambda*k)S*dt + sigma*S*dW + S*dJ + beta*S*dL/L

        where:
        - mu is drift
        - sigma is volatility
        - lambda is jump intensity
        - k is expected jump size
        - dW is Wiener process increment
        - dJ is jump process
        - dL/L is liquidity return

        Args:
            dt: Time step in years
            shock_impact: External shock impact (multiplicative, e.g., 0.1 = 10% drop)
            liquidity_return: Percentage change in global liquidity

        Returns:
            New market value
        """
        S = self._current_value

        # Founder Mode & Moral Hazard Logic
        # Founder Mode increases volatility (impulsiveness)
        # Moral Hazard increases volatility (risk-taking)
        founder_impact = self.config.founder_mode_intensity
        moral_hazard = self.config.moral_hazard_factor
        
        # PE Dominance Logic (Asset Stripping)
        # PE increases drift (short-term efficiency) but increases volatility (leverage)
        pe_impact = self.config.pe_dominance
        
        # Base volatility multiplier from behavioral factors
        behavioral_vol_multiplier = 1.0 + founder_impact + (0.5 * moral_hazard) + (0.3 * pe_impact)

        # Apply modifiers
        sigma = self.config.volatility * self._volatility_modifier * behavioral_vol_multiplier
        
        # PE boosts drift (short term extraction)
        pe_drift_boost = 0.02 * pe_impact # Up to 2% extra drift
        mu = self.config.drift + self._drift_modifier + pe_drift_boost

        # GBM component
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * np.sqrt(dt) * self.rng.normal()

        # Liquidity component (The "Everything Bubble" factor)
        # If beta > 1, market amplifies liquidity moves
        liquidity_term = self.config.liquidity_beta * liquidity_return

        # Jump component (compound Poisson process)
        jump_term = 0.0
        # PE increases jump intensity (bankruptcy risk)
        effective_jump_intensity = self.config.jump_intensity * (1.0 + pe_impact)

        # Ising Model Phase Transition Logic
        # High Coupling (J) + Low Volatility (T) = High Probability of Spontaneous Symmetry Breaking (Crash)
        # We model this as an amplifier to jump intensity
        ising_coupling = self.config.market_coupling_strength
        if ising_coupling > 0:
            # "Temperature" is volatility. Lower T = Higher Criticality.
            # We normalize T around 0.20 (20% vol).
            temperature = max(0.05, sigma) # Floor at 5% to prevent division by zero
            criticality = ising_coupling / temperature
            # Exponential amplification of jump risk
            ising_multiplier = np.exp(criticality * 0.1) # Scaling factor
            effective_jump_intensity *= ising_multiplier

        num_jumps = self.rng.poisson(effective_jump_intensity * dt)

        if num_jumps > 0:
            # Generate jump sizes
            jump_sizes = self.rng.normal(
                self.config.jump_mean,
                self.config.jump_std,
                size=int(num_jumps)
            )
            jump_term = np.sum(np.log(1 + jump_sizes))

        # External shock impact
        shock_term = 0.0
        if shock_impact > 0:
            shock_term = -shock_impact  # Negative impact on price

        # Update value (log-normal evolution)
        log_return = drift_term + diffusion_term + jump_term + shock_term + liquidity_term
        new_value = S * np.exp(log_return)

        # Ensure non-negative
        new_value = max(new_value, 0.0)

        self._current_value = new_value
        self._history.append(new_value)

        return new_value

    def simulate_path(
        self,
        num_steps: int,
        dt: float,
        shock_schedule: Optional[np.ndarray] = None,
        liquidity_returns: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate a full path of market values.

        Args:
            num_steps: Number of time steps
            dt: Time step size in years
            shock_schedule: Array of shock impacts for each time step
            liquidity_returns: Array of liquidity returns for each time step

        Returns:
            Array of market values (length num_steps + 1)
        """
        self.reset()

        if shock_schedule is None:
            shock_schedule = np.zeros(num_steps)
            
        if liquidity_returns is None:
            liquidity_returns = np.zeros(num_steps)

        for i in range(num_steps):
            shock = shock_schedule[i] if i < len(shock_schedule) else 0.0
            liq_ret = liquidity_returns[i] if i < len(liquidity_returns) else 0.0
            self.step(dt, shock, liq_ret)

        return self.history

    def reset(self) -> None:
        """Reset the model to initial state."""
        self._current_value = self.config.initial_value
        self._history = [self.config.initial_value]
        self._volatility_modifier = 1.0
        self._drift_modifier = 0.0

    def get_returns(self) -> np.ndarray:
        """Calculate log returns from history.

        Returns:
            Array of log returns
        """
        history = self.history
        if len(history) < 2:
            return np.array([])

        return np.diff(np.log(history))

    def get_simple_returns(self) -> np.ndarray:
        """Calculate simple returns from history.

        Returns:
            Array of simple returns
        """
        history = self.history
        if len(history) < 2:
            return np.array([])

        return np.diff(history) / history[:-1]


def simulate_multiple_markets(
    config: MarketConfig,
    num_simulations: int,
    num_steps: int,
    dt: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """Run multiple market simulations in parallel.

    Args:
        config: Market configuration
        num_simulations: Number of independent simulations
        num_steps: Number of time steps per simulation
        dt: Time step size in years
        seed: Random seed for reproducibility

    Returns:
        2D array of shape (num_simulations, num_steps + 1)
    """
    rng = RandomGenerator(seed)
    results = np.zeros((num_simulations, num_steps + 1))

    for i in range(num_simulations):
        model = MarketModel(config, rng)
        results[i] = model.simulate_path(num_steps, dt)

    return results
