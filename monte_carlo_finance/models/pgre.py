from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from monte_carlo_finance.core.config import PGREConfig
from monte_carlo_finance.utils.random import RandomGenerator

@dataclass
class PGREState:
    """Current state of the PGRE/OMP asset."""
    current_cash: float
    current_ncf: float
    debt_yield: float
    is_defaulted: bool = False
    default_reason: Optional[str] = None
    extended_2026: bool = False
    survived_2027: bool = False
    history: List[dict] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

class PGREModel:
    """
    Models the specific credit risks of One Market Plaza (OMP).
    
    Key Events:
    1. Continuous: Cash Burn due to TI/LC costs and negative FCF.
    2. Feb 6, 2026: Extension Hurdle (Liquidity Test).
    3. Feb 5, 2027: Forbearance Hurdle (Debt Yield Test).
    
    Monte Carlo: Adds stochastic volatility to cash burn and NCF.
    """
    
    def __init__(self, config: PGREConfig, rng: Optional[RandomGenerator] = None):
        self.config = config
        self.rng = rng or RandomGenerator()
        self.state = PGREState(
            current_cash=config.initial_cash,
            current_ncf=config.ncf_servicer,
            debt_yield=config.ncf_servicer / config.loan_balance
        )
        
    def step(self, dt: float, market_stress_factor: float, current_date_str: str):
        """
        Advance the model by one time step.
        
        Args:
            dt: Time step in years.
            market_stress_factor: 0.0 to 1.0 (1.0 = Max Panic/Crash).
            current_date_str: Date string (YYYY-MM-DD) to check triggers.
        """
        if self.state.is_defaulted:
            return

        # 1. Apply Cash Burn with stochastic component
        # Base burn increases if market stress is high (harder to lease, higher concessions)
        burn_multiplier = 1.0 + (market_stress_factor * 0.5)
        base_burn = self.config.annual_cash_burn * burn_multiplier * dt
        
        # Stochastic volatility: Cash burn varies based on config
        # Models uncertainty in TI/LC costs, unexpected expenses, lease timing
        cash_burn_vol = getattr(self.config, 'cash_burn_volatility', 0.40)
        stochastic_burn = base_burn * (1.0 + cash_burn_vol * float(self.rng.normal()))
        stochastic_burn = max(0.0, stochastic_burn)  # Burn cannot be negative
        
        self.state.current_cash -= stochastic_burn
        
        # 2. Degrade NCF based on Market Stress with stochastic component
        # Target NCF: High stress -> S&P Stressed (55M), Low stress -> Servicer (92M)
        target_ncf = self.config.ncf_servicer * (1.0 - market_stress_factor) + \
                     self.config.ncf_stressed * market_stress_factor
        
        # Stochastic volatility: NCF varies based on tenant decisions, market
        ncf_vol = getattr(self.config, 'ncf_volatility', 0.25)
        ncf_shock = target_ncf * ncf_vol * float(self.rng.normal()) * np.sqrt(dt)
        
        # Mean-revert towards target with noise
        reversion_speed = 0.5  # How fast NCF moves to target
        self.state.current_ncf += reversion_speed * (target_ncf - self.state.current_ncf) * dt + ncf_shock
        self.state.current_ncf = max(0.0, self.state.current_ncf)  # NCF cannot be negative
        
        self.state.debt_yield = self.state.current_ncf / self.config.loan_balance

        # 3. Check Triggers
        self._check_dates(current_date_str)
        
        # Record history
        self.state.history.append({
            'date': current_date_str,
            'cash': self.state.current_cash,
            'ncf': self.state.current_ncf,
            'debt_yield': self.state.debt_yield,
            'defaulted': self.state.is_defaulted
        })

    def reset(self):
        """Reset the model to initial state."""
        self.state = PGREState(
            current_cash=self.config.initial_cash,
            current_ncf=self.config.ncf_servicer,
            debt_yield=self.config.ncf_servicer / self.config.loan_balance
        )

    def _check_dates(self, date_str: str):
        """
        Check for key trigger dates.
        
        Uses date ranges instead of exact matches to handle discrete time steps
        that might skip over exact trigger dates.
        """
        from datetime import datetime
        
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Feb 6, 2026: Extension Test
        # Trigger window: Feb 1-10, 2026
        extension_start = datetime(2026, 2, 1)
        extension_end = datetime(2026, 2, 10)
        
        if extension_start <= current_date <= extension_end and not self.state.extended_2026:
            cost = (self.config.loan_balance * self.config.extension_fee_pct) + \
                   self.config.rate_cap_cost_est
            
            if self.state.current_cash >= cost:
                self.state.current_cash -= cost
                self.state.extended_2026 = True
            else:
                self.state.is_defaulted = True
                self.state.default_reason = "LIQUIDITY_FAILURE_2026"

        # Feb 5, 2027: Forbearance Test
        # Trigger window: Feb 1-10, 2027
        forbearance_start = datetime(2027, 2, 1)
        forbearance_end = datetime(2027, 2, 10)
        
        if forbearance_start <= current_date <= forbearance_end and not self.state.survived_2027:
            if self.state.debt_yield >= self.config.debt_yield_threshold_2027:
                self.state.survived_2027 = True
            else:
                self.state.is_defaulted = True
                self.state.default_reason = "DEBT_YIELD_FAILURE_2027"
        
        # Emergency default: Cash exhausted completely at any time
        if self.state.current_cash <= 0 and not self.state.is_defaulted:
            self.state.is_defaulted = True
            self.state.default_reason = "CASH_EXHAUSTION"
