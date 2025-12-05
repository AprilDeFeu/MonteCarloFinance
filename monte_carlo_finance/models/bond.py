"""
Bond value simulation model.

Implements bond pricing with stochastic interest rate dynamics
using the Vasicek model, including credit risk considerations.
"""

from typing import List, Optional, Tuple

import numpy as np

from monte_carlo_finance.core.config import BondConfig
from monte_carlo_finance.utils.random import RandomGenerator


class BondModel:
    """Bond value simulation using Vasicek interest rate model.

    This model includes:
    - Mean-reverting interest rate dynamics (Vasicek model)
    - Credit spread modeling
    - Default probability assessment
    - Bond price calculation using yield curve
    """

    def __init__(
        self,
        config: BondConfig,
        rng: Optional[RandomGenerator] = None
    ):
        """Initialize the bond model.

        Args:
            config: Bond configuration parameters
            rng: Random number generator (creates new one if not provided)
        """
        self.config = config
        self.rng = rng or RandomGenerator()

        # Current state
        self._current_yield = config.initial_yield
        self._current_spread = config.credit_spread
        self._remaining_maturity = config.maturity_years
        self._is_defaulted = False

        # Calculate initial price
        self._current_price = self._calculate_price()

        # History tracking
        self._yield_history: List[float] = [self._current_yield]
        self._price_history: List[float] = [self._current_price]
        self._spread_history: List[float] = [self._current_spread]

        # Panic modifiers
        self._yield_volatility_modifier = 1.0
        self._spread_modifier = 0.0

    @property
    def current_yield(self) -> float:
        """Get current yield to maturity."""
        return self._current_yield

    @property
    def current_price(self) -> float:
        """Get current bond price."""
        return self._current_price

    @property
    def current_spread(self) -> float:
        """Get current credit spread."""
        return self._current_spread

    @property
    def remaining_maturity(self) -> float:
        """Get remaining time to maturity."""
        return self._remaining_maturity

    @property
    def is_defaulted(self) -> bool:
        """Check if bond has defaulted."""
        return self._is_defaulted

    @property
    def yield_history(self) -> np.ndarray:
        """Get yield history."""
        return np.array(self._yield_history)

    @property
    def price_history(self) -> np.ndarray:
        """Get price history."""
        return np.array(self._price_history)

    @property
    def spread_history(self) -> np.ndarray:
        """Get spread history."""
        return np.array(self._spread_history)

    def set_panic_modifiers(
        self,
        yield_volatility_modifier: float = 1.0,
        spread_modifier: float = 0.0
    ) -> None:
        """Set panic-related modifiers for bond dynamics.

        Args:
            yield_volatility_modifier: Multiplier for yield volatility
            spread_modifier: Additive modifier for credit spread
        """
        self._yield_volatility_modifier = yield_volatility_modifier
        self._spread_modifier = spread_modifier

    def step(
        self,
        dt: float,
        yield_shock: float = 0.0,
        check_default: bool = True
    ) -> Tuple[float, float]:
        """Advance the bond model by one time step.

        Uses Vasicek model for interest rate:
        dr = kappa * (theta - r) * dt + sigma * dW

        where:
        - kappa is mean reversion speed
        - theta is long-term mean
        - sigma is volatility
        - dW is Wiener process increment

        Args:
            dt: Time step in years
            yield_shock: External yield shock (additive)
            check_default: Whether to check for default

        Returns:
            Tuple of (new price, new yield)
        """
        if self._is_defaulted:
            return (self._current_price, self._current_yield)

        # Update remaining maturity
        self._remaining_maturity -= dt
        if self._remaining_maturity <= 0:
            # Bond has matured
            self._remaining_maturity = 0
            self._current_price = self.config.face_value
            return (self._current_price, self._current_yield)

        # Vasicek model parameters
        kappa = self.config.mean_reversion_speed
        theta = self.config.long_term_yield
        sigma = self.config.yield_volatility * self._yield_volatility_modifier

        # Mean reversion term
        mean_reversion = kappa * (theta - self._current_yield) * dt

        # Diffusion term
        diffusion = sigma * np.sqrt(dt) * self.rng.normal()

        # Update yield
        new_yield = self._current_yield + mean_reversion + diffusion + yield_shock

        # Ensure non-negative yield (can go negative in some models but we cap it)
        new_yield = max(new_yield, -0.05)  # Allow slight negative yields

        self._current_yield = new_yield

        # Update credit spread
        self._current_spread = self.config.credit_spread + self._spread_modifier

        # Check for default
        if check_default:
            default_prob = self._calculate_default_probability(dt)
            if self.rng.uniform() < default_prob:
                self._is_defaulted = True
                self._current_price = self.config.face_value * self.config.recovery_rate
                self._yield_history.append(self._current_yield)
                self._price_history.append(self._current_price)
                self._spread_history.append(self._current_spread)
                return (self._current_price, self._current_yield)

        # Calculate new price
        self._current_price = self._calculate_price()

        # Update history
        self._yield_history.append(self._current_yield)
        self._price_history.append(self._current_price)
        self._spread_history.append(self._current_spread)

        return (self._current_price, self._current_yield)

    def _calculate_price(self) -> float:
        """Calculate bond price using yield to maturity.

        Uses present value of future cash flows.

        Returns:
            Bond price
        """
        if self._remaining_maturity <= 0:
            return self.config.face_value

        # Total yield including credit spread
        total_yield = self._current_yield + self._current_spread

        # Number of remaining coupon payments (assume annual)
        remaining_payments = int(np.ceil(self._remaining_maturity))

        if remaining_payments == 0:
            # Less than one period to maturity
            return self.config.face_value / (1 + total_yield * self._remaining_maturity)

        # Present value of coupon payments
        coupon = self.config.face_value * self.config.coupon_rate
        pv_coupons = 0.0

        for t in range(1, remaining_payments + 1):
            time_to_payment = min(t, self._remaining_maturity)
            pv_coupons += coupon / (1 + total_yield) ** time_to_payment

        # Present value of face value
        pv_face = self.config.face_value / (1 + total_yield) ** self._remaining_maturity

        return pv_coupons + pv_face

    def _calculate_default_probability(self, dt: float) -> float:
        """Calculate probability of default in the next time step.

        Uses spread-based default probability estimation.

        Args:
            dt: Time step in years

        Returns:
            Probability of default
        """
        # Simple model: default probability based on spread
        # Higher spread implies higher default risk
        # Using hazard rate approximation
        spread = self._current_spread + self._spread_modifier
        recovery = self.config.recovery_rate

        # Hazard rate = spread / (1 - recovery)
        if recovery < 1.0:
            hazard_rate = spread / (1 - recovery)
        else:
            hazard_rate = 0.0

        # Probability of default in time dt
        default_prob = 1 - np.exp(-hazard_rate * dt)

        return default_prob

    def simulate_path(
        self,
        num_steps: int,
        dt: float,
        yield_shocks: Optional[np.ndarray] = None,
        check_default: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a full path of bond prices and yields.

        Args:
            num_steps: Number of time steps
            dt: Time step size in years
            yield_shocks: Array of yield shocks for each time step
            check_default: Whether to check for default

        Returns:
            Tuple of (price array, yield array)
        """
        self.reset()

        if yield_shocks is None:
            yield_shocks = np.zeros(num_steps)

        for i in range(num_steps):
            shock = yield_shocks[i] if i < len(yield_shocks) else 0.0
            self.step(dt, shock, check_default)

        return (self.price_history, self.yield_history)

    def reset(self) -> None:
        """Reset the model to initial state."""
        self._current_yield = self.config.initial_yield
        self._current_spread = self.config.credit_spread
        self._remaining_maturity = self.config.maturity_years
        self._is_defaulted = False
        self._current_price = self._calculate_price()
        self._yield_history = [self._current_yield]
        self._price_history = [self._current_price]
        self._spread_history = [self._current_spread]
        self._yield_volatility_modifier = 1.0
        self._spread_modifier = 0.0

    def get_duration(self) -> float:
        """Calculate Macaulay duration of the bond.

        Returns:
            Duration in years
        """
        if self._remaining_maturity <= 0 or self._is_defaulted:
            return 0.0

        total_yield = self._current_yield + self._current_spread
        coupon = self.config.face_value * self.config.coupon_rate
        remaining_payments = int(np.ceil(self._remaining_maturity))

        weighted_cf = 0.0
        total_pv = 0.0

        for t in range(1, remaining_payments + 1):
            time_to_payment = min(t, self._remaining_maturity)
            cf = coupon if t < remaining_payments else coupon + self.config.face_value
            pv = cf / (1 + total_yield) ** time_to_payment
            weighted_cf += time_to_payment * pv
            total_pv += pv

        if total_pv == 0:
            return 0.0

        return weighted_cf / total_pv

    def get_modified_duration(self) -> float:
        """Calculate modified duration of the bond.

        Returns:
            Modified duration
        """
        mac_duration = self.get_duration()
        total_yield = self._current_yield + self._current_spread
        return mac_duration / (1 + total_yield)

    def get_convexity(self) -> float:
        """Calculate convexity of the bond.

        Returns:
            Convexity measure
        """
        if self._remaining_maturity <= 0 or self._is_defaulted:
            return 0.0

        total_yield = self._current_yield + self._current_spread
        coupon = self.config.face_value * self.config.coupon_rate
        remaining_payments = int(np.ceil(self._remaining_maturity))

        weighted_cf = 0.0
        total_pv = 0.0

        for t in range(1, remaining_payments + 1):
            time_to_payment = min(t, self._remaining_maturity)
            cf = coupon if t < remaining_payments else coupon + self.config.face_value
            pv = cf / (1 + total_yield) ** time_to_payment
            weighted_cf += time_to_payment * (time_to_payment + 1) * pv
            total_pv += pv

        if total_pv == 0:
            return 0.0

        return weighted_cf / (total_pv * (1 + total_yield) ** 2)


def simulate_multiple_bonds(
    config: BondConfig,
    num_simulations: int,
    num_steps: int,
    dt: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Run multiple bond simulations.

    Args:
        config: Bond configuration
        num_simulations: Number of independent simulations
        num_steps: Number of time steps per simulation
        dt: Time step size in years
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prices array, yields array) each of shape (num_simulations, num_steps + 1)
    """
    rng = RandomGenerator(seed)
    prices = np.zeros((num_simulations, num_steps + 1))
    yields = np.zeros((num_simulations, num_steps + 1))

    for i in range(num_simulations):
        model = BondModel(config, rng)
        p, y = model.simulate_path(num_steps, dt)
        prices[i] = p
        yields[i] = y

    return (prices, yields)
