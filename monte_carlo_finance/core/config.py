"""
Configuration module for Monte Carlo simulations.

Provides highly parameterizable configuration options for all aspects of
financial market simulation, including market dynamics, bond behavior,
shockwave events, and panic scenarios.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketConfig:
    """Configuration for market value simulation.

    Attributes:
        initial_value: Starting market value (e.g., index level)
        drift: Expected annual return (mu) - annualized drift rate
        volatility: Annual volatility (sigma) - standard deviation of returns
        jump_intensity: Lambda parameter for jump frequency (jumps per year)
        jump_mean: Mean jump size (as proportion of value)
        jump_std: Standard deviation of jump size
    """
    initial_value: float = 100.0
    drift: float = 0.05  # 5% annual expected return
    volatility: float = 0.20  # 20% annual volatility
    jump_intensity: float = 0.1  # Average 0.1 jumps per year
    jump_mean: float = -0.05  # Average jump is -5%
    jump_std: float = 0.10  # Jump size std dev


@dataclass
class BondConfig:
    """Configuration for bond value simulation.

    Attributes:
        face_value: Bond face/par value
        coupon_rate: Annual coupon rate
        maturity_years: Time to maturity in years
        initial_yield: Starting yield to maturity
        yield_volatility: Volatility of yield changes
        mean_reversion_speed: Speed of mean reversion for yields (Vasicek model)
        long_term_yield: Long-term mean yield for mean reversion
        credit_spread: Additional spread over risk-free rate
        recovery_rate: Expected recovery rate in case of default
    """
    face_value: float = 1000.0
    coupon_rate: float = 0.05  # 5% coupon
    maturity_years: float = 10.0
    initial_yield: float = 0.05  # 5% yield
    yield_volatility: float = 0.01  # 1% yield volatility
    mean_reversion_speed: float = 0.3  # Mean reversion speed
    long_term_yield: float = 0.04  # 4% long-term yield
    credit_spread: float = 0.02  # 2% credit spread
    recovery_rate: float = 0.40  # 40% recovery on default


@dataclass
class ShockwaveConfig:
    """Configuration for shockwave/cascade events.

    Attributes:
        base_probability: Base probability of a shockwave event per time step
        yield_impact: Impact on bond yields when shock occurs (additive)
        market_impact: Impact on market values (multiplicative, e.g., 0.1 = 10% drop)
        propagation_delay: Time steps before cascade effects propagate
        cascade_decay: Decay factor for cascade propagation (0-1)
        selloff_threshold: Yield increase threshold that triggers automated selloffs
        selloff_intensity: Intensity of automated selloff (fraction of position sold)
    """
    base_probability: float = 0.01  # 1% chance per time step
    yield_impact: float = 0.02  # 2% yield increase on shock
    market_impact: float = 0.10  # 10% market drop on shock
    propagation_delay: int = 1  # 1 time step delay
    cascade_decay: float = 0.7  # 70% decay per propagation step
    selloff_threshold: float = 0.05  # 5% yield increase triggers selloff
    selloff_intensity: float = 0.20  # Sell 20% of position


@dataclass
class PanicConfig:
    """Configuration for market panic modeling.

    Attributes:
        base_panic_level: Baseline panic level (0-1 scale)
        panic_sensitivity: How sensitive panic is to adverse events
        panic_decay: Rate at which panic decays over time (per time step)
        max_panic_level: Maximum panic level cap
        volatility_multiplier: How much panic multiplies base volatility
        drift_impact: How much panic affects drift (negative impact)
        correlation_boost: How much panic increases asset correlations
        herd_behavior_factor: Strength of herding behavior during panic
        recovery_threshold: Panic level below which normal behavior resumes
    """
    base_panic_level: float = 0.0
    panic_sensitivity: float = 0.5  # Sensitivity to adverse events
    panic_decay: float = 0.1  # 10% decay per time step
    max_panic_level: float = 1.0
    volatility_multiplier: float = 2.0  # Volatility doubles at max panic
    drift_impact: float = -0.10  # -10% drift at max panic
    correlation_boost: float = 0.5  # Correlation increases by 0.5 at max panic
    herd_behavior_factor: float = 0.3  # Herding strength
    recovery_threshold: float = 0.2  # Below this, normal behavior resumes


@dataclass
class SimulationConfig:
    """Main configuration for Monte Carlo simulation.

    Attributes:
        num_simulations: Number of simulation paths to generate
        num_steps: Number of time steps per simulation
        time_horizon: Total simulation time in years
        random_seed: Seed for reproducibility (None for random)
        market: Market configuration
        bond: Bond configuration
        shockwave: Shockwave configuration
        panic: Panic configuration
    """
    num_simulations: int = 1000
    num_steps: int = 252  # Daily steps for one year
    time_horizon: float = 1.0  # 1 year
    random_seed: Optional[int] = None

    market: MarketConfig = field(default_factory=MarketConfig)
    bond: BondConfig = field(default_factory=BondConfig)
    shockwave: ShockwaveConfig = field(default_factory=ShockwaveConfig)
    panic: PanicConfig = field(default_factory=PanicConfig)

    @property
    def dt(self) -> float:
        """Time step size in years."""
        return self.time_horizon / self.num_steps

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "num_simulations": self.num_simulations,
            "num_steps": self.num_steps,
            "time_horizon": self.time_horizon,
            "random_seed": self.random_seed,
            "market": {
                "initial_value": self.market.initial_value,
                "drift": self.market.drift,
                "volatility": self.market.volatility,
                "jump_intensity": self.market.jump_intensity,
                "jump_mean": self.market.jump_mean,
                "jump_std": self.market.jump_std,
            },
            "bond": {
                "face_value": self.bond.face_value,
                "coupon_rate": self.bond.coupon_rate,
                "maturity_years": self.bond.maturity_years,
                "initial_yield": self.bond.initial_yield,
                "yield_volatility": self.bond.yield_volatility,
                "mean_reversion_speed": self.bond.mean_reversion_speed,
                "long_term_yield": self.bond.long_term_yield,
                "credit_spread": self.bond.credit_spread,
                "recovery_rate": self.bond.recovery_rate,
            },
            "shockwave": {
                "base_probability": self.shockwave.base_probability,
                "yield_impact": self.shockwave.yield_impact,
                "market_impact": self.shockwave.market_impact,
                "propagation_delay": self.shockwave.propagation_delay,
                "cascade_decay": self.shockwave.cascade_decay,
                "selloff_threshold": self.shockwave.selloff_threshold,
                "selloff_intensity": self.shockwave.selloff_intensity,
            },
            "panic": {
                "base_panic_level": self.panic.base_panic_level,
                "panic_sensitivity": self.panic.panic_sensitivity,
                "panic_decay": self.panic.panic_decay,
                "max_panic_level": self.panic.max_panic_level,
                "volatility_multiplier": self.panic.volatility_multiplier,
                "drift_impact": self.panic.drift_impact,
                "correlation_boost": self.panic.correlation_boost,
                "herd_behavior_factor": self.panic.herd_behavior_factor,
                "recovery_threshold": self.panic.recovery_threshold,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Create configuration from dictionary."""
        return cls(
            num_simulations=data.get("num_simulations", 1000),
            num_steps=data.get("num_steps", 252),
            time_horizon=data.get("time_horizon", 1.0),
            random_seed=data.get("random_seed"),
            market=MarketConfig(**data.get("market", {})),
            bond=BondConfig(**data.get("bond", {})),
            shockwave=ShockwaveConfig(**data.get("shockwave", {})),
            panic=PanicConfig(**data.get("panic", {})),
        )
