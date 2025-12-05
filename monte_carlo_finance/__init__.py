"""
Monte Carlo Finance - A comprehensive Monte Carlo simulation framework for financial markets.

This package provides tools for simulating:
- Market value dynamics
- Bond pricing and yield movements
- Shockwave events and cascade triggers
- Market panic scenarios

All components are highly parameterizable to allow for custom scenario modeling.
"""

from monte_carlo_finance.core.config import SimulationConfig
from monte_carlo_finance.core.simulation import MonteCarloSimulation
from monte_carlo_finance.models.bond import BondModel
from monte_carlo_finance.models.market import MarketModel
from monte_carlo_finance.triggers.panic import PanicModel
from monte_carlo_finance.triggers.shockwave import (
    ShockType,
    ShockwaveEvent,
    ShockwaveTrigger,
)

__version__ = "0.1.0"
__all__ = [
    "MonteCarloSimulation",
    "SimulationConfig",
    "MarketModel",
    "BondModel",
    "ShockwaveEvent",
    "ShockwaveTrigger",
    "ShockType",
    "PanicModel",
]
