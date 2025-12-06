"""
Tests for the Consumer/Labor Model.
"""

import pytest
import numpy as np
from monte_carlo_finance.core.config import ConsumerConfig
from monte_carlo_finance.models.consumer import ConsumerModel
from monte_carlo_finance.utils.random import RandomGenerator


@pytest.fixture
def consumer_config():
    return ConsumerConfig(
        initial_wage_index=100.0,
        initial_col_index=100.0,
        productivity_growth=0.02,
        wage_lag=0.5,  # 50% lag
        asset_inflation_weight=0.4,
        max_leverage_ratio=3.0,
        savings_rate=0.05,
        wage_volatility=0.0,  # Zero volatility for deterministic tests
        col_volatility=0.0,
        spending_volatility=0.0
    )


@pytest.fixture
def consumer_model(consumer_config):
    # Use fixed seed for reproducibility
    rng = RandomGenerator(seed=42)
    return ConsumerModel(consumer_config, rng)


def test_initialization(consumer_model, consumer_config):
    """Test that the model initializes correctly."""
    assert consumer_model.state.wage_index == consumer_config.initial_wage_index
    assert consumer_model.state.col_index == consumer_config.initial_col_index
    assert consumer_model.state.debt_index == 0.0
    assert consumer_model.state.default_probability == 0.0


def test_wage_lag_mechanics(consumer_model):
    """Test that wages grow slower than productivity."""
    dt = 1.0
    market_return = 0.0
    liquidity_level = 1.0
    
    # Step 1 year
    consumer_model.step(dt, market_return, liquidity_level)
    
    # Expected wage growth: productivity * (1 - wage_lag)
    # 0.02 * (1 - 0.5) = 0.01 (1%)
    # With zero volatility configured, this should be deterministic
    expected_wage = 100.0 * (1 + 0.01)
    
    assert np.isclose(consumer_model.state.wage_index, expected_wage, rtol=1e-4)


def test_asset_inflation_impact():
    """Test that market returns drive Cost of Living."""
    # Create config with zero volatility for deterministic behavior
    config = ConsumerConfig(
        initial_wage_index=100.0,
        initial_col_index=100.0,
        productivity_growth=0.02,
        wage_lag=0.5,
        asset_inflation_weight=0.4,
        max_leverage_ratio=3.0,
        savings_rate=0.05,
        wage_volatility=0.0,
        col_volatility=0.0,
        spending_volatility=0.0
    )
    rng = RandomGenerator(seed=42)
    consumer_model = ConsumerModel(config, rng)
    
    dt = 1.0
    market_return = 0.10  # 10% market rally
    liquidity_level = 1.0
    
    consumer_model.step(dt, market_return, liquidity_level)
    
    # Expected CoL growth:
    # Base CPI (assumed 2%) + Asset Inflation
    # Asset Inflation = market_return * asset_inflation_weight = 0.10 * 0.4 = 0.04
    # Total CoL growth approx 0.02 + 0.04 = 0.06
    
    # Let's check if CoL > Wage Growth (which is 1%)
    assert consumer_model.state.col_index > consumer_model.state.wage_index
    
    # Check debt accumulation
    # Since CoL > Wages, debt should increase (assuming savings rate isn't huge)
    assert consumer_model.state.debt_index > 0.0


def test_debt_spiral_and_default():
    """Test that high debt leads to default probability."""
    # Create config with some stochastic behavior for realism
    config = ConsumerConfig(
        initial_wage_index=100.0,
        initial_col_index=100.0,
        productivity_growth=0.02,
        wage_lag=0.5,
        asset_inflation_weight=0.4,
        max_leverage_ratio=3.0,
        savings_rate=0.05
    )
    rng = RandomGenerator(seed=42)
    consumer_model = ConsumerModel(config, rng)
    
    dt = 1.0
    market_return = 0.20  # Massive asset bubble
    liquidity_level = 0.8  # Tight liquidity (makes debt harder to service)
    
    # Force debt accumulation over multiple steps
    for _ in range(10):
        consumer_model.step(dt, market_return, liquidity_level)
        
    assert consumer_model.state.debt_index > 0
    assert consumer_model.state.default_probability > 0


def test_reset():
    """Test that reset restores initial state."""
    config = ConsumerConfig(
        initial_wage_index=100.0,
        initial_col_index=100.0,
        productivity_growth=0.02,
        wage_lag=0.5,
        asset_inflation_weight=0.4,
        max_leverage_ratio=3.0,
        savings_rate=0.05
    )
    rng = RandomGenerator(seed=42)
    consumer_model = ConsumerModel(config, rng)
    
    consumer_model.step(1.0, 0.1, 1.0)
    consumer_model.reset()
    
    assert consumer_model.state.wage_index == 100.0
    assert consumer_model.state.col_index == 100.0
    assert consumer_model.state.debt_index == 0.0
    assert len(consumer_model.state.history) == 0
