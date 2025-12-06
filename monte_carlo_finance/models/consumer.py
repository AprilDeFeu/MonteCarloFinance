"""
Consumer/Labor Model.

Simulates the "Real Economy" of the middle class, tracking the divergence
between wages and cost of living, and the resulting accumulation of debt.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from monte_carlo_finance.utils.random import RandomGenerator
from monte_carlo_finance.core.config import ConsumerConfig


@dataclass
class ConsumerState:
    """Current state of the Consumer."""
    wage_index: float
    col_index: float
    debt_index: float
    disposable_income: float
    default_probability: float
    history: List[dict] = field(default_factory=list)


class ConsumerModel:
    """
    Models the financial health of the consumer class.
    
    Key Dynamics:
    1. Wages grow slower than Productivity (Decoupling).
    2. Cost of Living grows with CPI + Asset Prices (Bifurcation).
    3. The gap is filled by Debt.
    4. High Debt + Shock = Default Cascade.
    """
    
    def __init__(self, config: ConsumerConfig, rng: Optional[RandomGenerator] = None):
        self.config = config
        self.rng = rng or RandomGenerator()
        self.state = ConsumerState(
            wage_index=config.initial_wage_index,
            col_index=config.initial_col_index,
            debt_index=config.initial_debt_index,
            disposable_income=0.0,
            default_probability=0.0
        )
        
    def step(self, dt: float, market_return: float, liquidity_level: float):
        """
        Advance the consumer model.
        
        Args:
            dt: Time step.
            market_return: Return of the asset market (impacts CoL via housing).
            liquidity_level: Global liquidity (impacts credit availability).
        """
        # 1. Wage Growth (Decoupled from Productivity) with stochastic component
        # Wages grow at (Productivity * (1 - Lag)) + noise
        real_wage_growth = self.config.productivity_growth * (1.0 - self.config.wage_lag)
        wage_vol = getattr(self.config, 'wage_volatility', 0.03)
        stochastic_wage_growth = real_wage_growth + wage_vol * float(self.rng.normal()) * np.sqrt(dt)
        self.state.wage_index *= (1.0 + stochastic_wage_growth * dt)
        
        # 2. Cost of Living (Bifurcated Inflation) with stochastic component
        # Base CPI (2%) + Asset Inflation (Housing/Tuition driven by Market)
        base_inflation = 0.02
        asset_inflation = max(0, market_return) * self.config.asset_inflation_weight
        col_vol = getattr(self.config, 'col_volatility', 0.04)
        total_inflation = base_inflation + asset_inflation + col_vol * float(self.rng.normal()) * np.sqrt(dt)
        self.state.col_index *= (1.0 + total_inflation * dt)
        
        # 3. Disposable Income & Debt (with stochastic debt accumulation)
        # If CoL > Wages, we burn savings or take debt
        
        # BNPL Logic: Masks the true cost of living temporarily
        # Effective CoL is reduced by BNPL usage (kicking the can)
        effective_col = self.state.col_index * (1.0 - self.config.bnpl_usage * 0.1)
        
        # Normalized gap with stochastic component (Unexpected expenses/windfalls)
        # This makes the debt accumulation non-linear
        spending_vol = getattr(self.config, 'spending_volatility', 0.10)
        unexpected_expense = self.state.wage_index * spending_vol * float(self.rng.normal()) * np.sqrt(dt)
        
        gap = self.state.wage_index - effective_col - unexpected_expense
        self.state.disposable_income = gap
        
        # Financial Nihilism: Increases spending/debt regardless of gap
        nihilism_spend = self.state.wage_index * self.config.financial_nihilism * 0.05 # Up to 5% extra spend
        
        # BNPL Debt Accumulation (Always happens if BNPL is used)
        deferred_col = self.state.col_index - effective_col
        self.state.debt_index += deferred_col * dt
        
        # Nihilism Spending (Always adds to debt/burn) with stochastic component
        # Spending behavior varies: sometimes splurge, sometimes restraint
        stochastic_nihilism = nihilism_spend * (1.0 + spending_vol * float(self.rng.normal()))
        self.state.debt_index += max(0, stochastic_nihilism) * dt

        # Bandwidth Tax & Scarcity Tunneling
        # If CoL consumes a large portion of wages, cognitive bandwidth drops
        scarcity_ratio = effective_col / self.state.wage_index if self.state.wage_index > 0 else 1.0
        scarcity_intensity = max(0.0, (scarcity_ratio - 0.8) / 0.2) # 0 to 1 as ratio goes 0.8 to 1.0
        
        # Reduce savings rate due to tunneling (inability to plan)
        effective_savings_rate = self.config.savings_rate
        if self.config.scarcity_tunneling > 0:
            effective_savings_rate *= (1.0 - self.config.scarcity_tunneling * scarcity_intensity)
            
        # DRRA: Poor take negative-EV risks (Lottery ticket effect)
        if self.config.risk_aversion_curve > 0 and scarcity_intensity > 0:
            lottery_spend = self.state.wage_index * self.config.risk_aversion_curve * scarcity_intensity * 0.02
            self.state.debt_index += lottery_spend * dt

        if gap < 0:
            # Cash Deficit funded by additional debt
            self.state.debt_index += abs(gap) * dt
        else:
            # Cash Surplus pays down debt or saves
            # Note: Even with surplus, we might not pay down the full BNPL amount immediately
            repayment = gap * effective_savings_rate * dt
            if self.state.debt_index > 0:
                self.state.debt_index = max(0, self.state.debt_index - repayment)
                
        # 4. Default Probability (Non-linear Phase Transition)
        self.state.default_probability = self._calculate_default_probability(
            liquidity_level, market_return
        )
        
        self.state.history.append({
            'wage': self.state.wage_index,
            'col': self.state.col_index,
            'debt': self.state.debt_index,
            'default_prob': self.state.default_probability
        })
    
    def _calculate_default_probability(
        self, liquidity_level: float, market_return: float
    ) -> float:
        """
        Calculate default probability using Ising-model phase transition dynamics.
        
        Args:
            liquidity_level: Global liquidity (impacts credit availability).
            market_return: Return of the asset market (impacts stress).
            
        Returns:
            Default probability between 0 and 1.
        """
        # Based on Ising Model concept: Small stress below threshold, rapid cascade above
        leverage = (self.state.debt_index / self.state.wage_index 
                    if self.state.wage_index > 0 else 100.0)
        herd_factor = getattr(self.config, "herd_behavior", 0.3)
        
        # Liquidity acts as a buffer. High liquidity = easy to roll debt.
        # Low liquidity = hard to roll debt -> shifts threshold left.
        liquidity_stress = max(0.1, 2.0 - liquidity_level)  # 1.0 is neutral
        effective_threshold = self.config.max_leverage_ratio / liquidity_stress
        
        # Phase transition dynamics (Ising-inspired) using sigmoid backbone
        k = self.config.default_steepness
        exponent = -k * (leverage - effective_threshold)
        base_sigmoid = 1.0 / (1.0 + np.exp(np.clip(exponent, -50, 50)))
        
        # Neighbor/contagion effect: strongest near threshold, fades away
        distance_from_critical = abs(leverage - effective_threshold)
        neighbor_effect = herd_factor * (1 - np.exp(-distance_from_critical**2))
        neighbor_effect *= base_sigmoid * (1 - base_sigmoid)
        
        default_prob = min(1.0, base_sigmoid + neighbor_effect)
        
        # Apply market stress amplifier (panic increases default correlation)
        market_stress = max(0, -market_return * 10)
        stress_amplifier = 1.0 + market_stress * 0.5
        
        return min(1.0, default_prob * stress_amplifier)
        
    def reset(self):
        self.state = ConsumerState(
            wage_index=self.config.initial_wage_index,
            col_index=self.config.initial_col_index,
            debt_index=self.config.initial_debt_index,
            disposable_income=0.0,
            default_probability=0.0
        )
