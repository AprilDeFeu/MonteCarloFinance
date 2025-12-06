"""
Configuration module for Monte Carlo simulations.

Provides highly parameterizable configuration options for all aspects of
financial market simulation, including market dynamics, bond behavior,
shockwave events, and panic scenarios.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LiquidityConfig:
    """Configuration for Global Liquidity simulation.
    
    Attributes:
        initial_level: Starting liquidity index (1.0 = neutral)
        drift: Natural trend of liquidity (e.g., secular expansion)
        volatility: Volatility of liquidity injections/withdrawals
        mean_reversion_speed: How fast it returns to trend
        fed_put_sensitivity: How much liquidity increases when market crashes
        inflation_constraint: Threshold where liquidity MUST tighten (Fed Dilemma)
    """
    initial_level: float = 1.0
    drift: float = 0.02  # 2% annual growth
    volatility: float = 0.10  # 10% volatility
    mean_reversion_speed: float = 0.5
    fed_put_sensitivity: float = 1.5  # High sensitivity to crashes
    inflation_constraint: float = 0.8  # If liquidity > 1.2, risk of inflation spike
    qe_impact: float = 0.05  # Impact of QE event
    qt_impact: float = -0.05  # Impact of QT event


@dataclass
class ConsumerConfig:
    """Configuration for Consumer/Labor simulation.
    
    Attributes:
        initial_wage_index: Starting real wage level (100.0)
        initial_col_index: Starting cost of living index (100.0)
        productivity_growth: Annual growth in productivity (potential wage growth)
        wage_lag: Percentage of productivity NOT passed to wages (The "1971" wedge)
        asset_inflation_weight: How much asset prices (Housing) impact Cost of Living
        max_leverage_ratio: Debt-to-Income ratio where defaults begin (sigmoid midpoint)
        default_steepness: Steepness of sigmoid default curve (higher = sharper transition)
        savings_rate: Base savings rate (negative implies debt accumulation)
    """
    initial_wage_index: float = 100.0
    initial_col_index: float = 100.0
    initial_debt_index: float = 0.0
    productivity_growth: float = 0.02  # 2% annual productivity growth
    wage_lag: float = 0.5  # Only 50% of productivity goes to wages
    asset_inflation_weight: float = 0.4  # 40% of CoL driven by asset prices (Housing/Ed)
    max_leverage_ratio: float = 3.0  # 3x Income (sigmoid midpoint)
    default_steepness: float = 3.0  # Sigmoid steepness (higher = sharper transition)
    savings_rate: float = 0.05  # 5% savings rate (historically low)
    wage_volatility: float = 0.03  # Annual volatility of wage growth
    col_volatility: float = 0.04  # Annual volatility of Cost of Living
    spending_volatility: float = 0.10  # Annual volatility of discretionary spending
    financial_nihilism: float = 0.0  # 0.0 = Prudent, 1.0 = "YOLO" (High Risk/Debt)
    bnpl_usage: float = 0.0  # 0.0 = Cash, 1.0 = Everything Financed
    wealth_distribution_pareto: float = 0.1  # Top 10% follow Pareto
    saving_propensity_gap: float = 0.8  # Difference in lambda between rich (0.9) and poor (0.1)
    scarcity_tunneling: float = 0.0  # 0.0 = No tax, 1.0 = Full IQ drop effect
    risk_aversion_curve: float = 0.0  # 0.0 = Constant, 1.0 = DRRA (Rich buy dip, Poor buy lotto)
    herd_behavior: float = 0.3  # Non-linear amplification near default threshold (Ising-style)


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
        narrative_premium: Portion of value driven by narrative (AI bubble)
        liquidity_beta: Sensitivity to global liquidity changes (1.0 = 1:1 correlation)
        founder_mode_intensity: Degree of impulsive/micromanagement behavior (0-1)
        zirp_dependency: Reliance on cheap capital (0-1)
        moral_hazard_factor: Lack of accountability/consequences (0-1)
    """
    initial_value: float = 100.0
    drift: float = 0.05  # 5% annual expected return
    volatility: float = 0.20  # 20% annual volatility
    jump_intensity: float = 0.1  # Average 0.1 jumps per year
    jump_mean: float = -0.05  # Average jump is -5%
    jump_std: float = 0.10  # Jump size std dev
    narrative_premium: float = 0.20  # 20% of value is "narrative" (AI bubble)
    liquidity_beta: float = 1.2  # High sensitivity to liquidity (Everything Bubble)
    founder_mode_intensity: float = 0.0  # 0.0 = Professional Management, 1.0 = "Founder Mode"
    zirp_dependency: float = 0.0  # 0.0 = Profitable, 1.0 = Needs free money to survive
    moral_hazard_factor: float = 0.0  # 0.0 = Accountable, 1.0 = Golden Parachute guaranteed
    pe_dominance: float = 0.0  # 0.0 = Public Co, 1.0 = PE Owned (Asset Stripping)
    insulation_factor: float = 0.0  # 0.0 = Skin in Game, 1.0 = "I'll be gone" (Exit Strategy)
    market_coupling_strength: float = 0.0  # Ising Model J parameter (Herding)
    external_field: float = 0.0  # Ising Model H parameter (News/Fed)


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
    maturity_wall_2025: float = 957.0  # Billions maturing in 2025
    maturity_wall_2026: float = 539.0  # Billions maturing in 2026
    maturity_wall_2027: float = 550.0  # Billions maturing in 2027


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
        consumer_default_threshold: Consumer default probability threshold for triggering crash
    """
    base_probability: float = 0.01  # 1% chance per time step
    yield_impact: float = 0.02  # 2% yield increase on shock
    market_impact: float = 0.10  # 10% market drop on shock
    propagation_delay: int = 1  # 1 time step delay
    cascade_decay: float = 0.7  # 70% decay per propagation step
    selloff_threshold: float = 0.05  # 5% yield increase triggers selloff
    selloff_intensity: float = 0.20  # Sell 20% of position
    consumer_default_threshold: float = 0.05  # 5% consumer default probability triggers crash


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
    fragility_factor: float = 1.5  # Multiplier for panic sensitivity due to systemic fragility


@dataclass
class PGREConfig:
    """Configuration for Paramount Group (PGRE) / One Market Plaza simulation."""
    
    # Loan Details (Source: S&P Global, March 2024)
    loan_balance: float = 850_000_000.0
    interest_rate: float = 0.0408
    
    # Liquidity Position (Source: Q3 2025 10-Q)
    initial_cash: float = 330_000_000.0
    # Negative Unlevered Free Cash Flow (2025E/2026E) due to TI/LC costs
    annual_cash_burn: float = 40_000_000.0  # Averaging 2025/2026 deficits
    
    # Extension Costs (Feb 6, 2026)
    extension_fee_pct: float = 0.005  # 0.50% conservative estimate
    rate_cap_cost_est: float = 15_000_000.0 # Estimated cost for new cap
    
    # Covenant Thresholds (Feb 5, 2027)
    debt_yield_threshold_2027: float = 0.085
    
    # NCF Scenarios
    ncf_servicer: float = 92_000_000.0   # Optimistic (Servicer Adjusted)
    ncf_stressed: float = 60_500_000.0   # Pessimistic (S&P Stressed)
    ncf_volatility: float = 0.25  # Annual volatility of NCF (Leasing risk)
    cash_burn_volatility: float = 0.40  # Annual volatility of Cash Burn (TI/LC lumpiness)
    
    # Market Dynamics (Source: Q3 2025 Leasing Data)
    sf_rent_decline_rate: float = 0.114  # -11.4% renewal spreads
    ti_lc_cost_psf: float = 173.36       # Cost to buy tenants
    expiring_sqft_2026: float = 1_180_364.0


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
    start_date: str = "2025-12-05"  # Simulation start date
    random_seed: Optional[int] = None

    market: MarketConfig = field(default_factory=MarketConfig)
    bond: BondConfig = field(default_factory=BondConfig)
    shockwave: ShockwaveConfig = field(default_factory=ShockwaveConfig)
    panic: PanicConfig = field(default_factory=PanicConfig)
    pgre: PGREConfig = field(default_factory=PGREConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    consumer: ConsumerConfig = field(default_factory=ConsumerConfig)

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
            "start_date": self.start_date,
            "random_seed": self.random_seed,
            "market": {
                "initial_value": self.market.initial_value,
                "drift": self.market.drift,
                "volatility": self.market.volatility,
                "jump_intensity": self.market.jump_intensity,
                "jump_mean": self.market.jump_mean,
                "jump_std": self.market.jump_std,
                "liquidity_beta": self.market.liquidity_beta,
                "founder_mode_intensity": self.market.founder_mode_intensity,
                "zirp_dependency": self.market.zirp_dependency,
                "moral_hazard_factor": self.market.moral_hazard_factor,
                "narrative_premium": self.market.narrative_premium,
                "pe_dominance": self.market.pe_dominance,
                "insulation_factor": self.market.insulation_factor,
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
                "fragility_factor": self.panic.fragility_factor,
            },
            "pgre": {
                "loan_balance": self.pgre.loan_balance,
                "interest_rate": self.pgre.interest_rate,
                "initial_cash": self.pgre.initial_cash,
                "annual_cash_burn": self.pgre.annual_cash_burn,
                "extension_fee_pct": self.pgre.extension_fee_pct,
                "rate_cap_cost_est": self.pgre.rate_cap_cost_est,
                "debt_yield_threshold_2027": self.pgre.debt_yield_threshold_2027,
                "ncf_servicer": self.pgre.ncf_servicer,
                "ncf_stressed": self.pgre.ncf_stressed,
                "sf_rent_decline_rate": self.pgre.sf_rent_decline_rate,
                "ti_lc_cost_psf": self.pgre.ti_lc_cost_psf,
                "expiring_sqft_2026": self.pgre.expiring_sqft_2026,
            },
            "liquidity": {
                "initial_level": self.liquidity.initial_level,
                "mean_reversion_speed": self.liquidity.mean_reversion_speed,
                "volatility": self.liquidity.volatility,
                "fed_put_sensitivity": self.liquidity.fed_put_sensitivity,
                "inflation_constraint": self.liquidity.inflation_constraint,
                "qe_impact": self.liquidity.qe_impact,
                "qt_impact": self.liquidity.qt_impact,
            },
            "consumer": {
                "initial_wage_index": self.consumer.initial_wage_index,
                "initial_col_index": self.consumer.initial_col_index,
                "productivity_growth": self.consumer.productivity_growth,
                "wage_lag": self.consumer.wage_lag,
                "asset_inflation_weight": self.consumer.asset_inflation_weight,
                "max_leverage_ratio": self.consumer.max_leverage_ratio,
                "savings_rate": self.consumer.savings_rate,
                "financial_nihilism": self.consumer.financial_nihilism,
                "bnpl_usage": self.consumer.bnpl_usage,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Create configuration from dictionary."""
        return cls(
            num_simulations=data.get("num_simulations", 1000),
            num_steps=data.get("num_steps", 252),
            time_horizon=data.get("time_horizon", 1.0),
            start_date=data.get("start_date", "2025-12-05"),
            random_seed=data.get("random_seed"),
            market=MarketConfig(**data.get("market", {})),
            bond=BondConfig(**data.get("bond", {})),
            shockwave=ShockwaveConfig(**data.get("shockwave", {})),
            panic=PanicConfig(**data.get("panic", {})),
            pgre=PGREConfig(**data.get("pgre", {})),
            liquidity=LiquidityConfig(**data.get("liquidity", {})),
            consumer=ConsumerConfig(**data.get("consumer", {})),
        )
