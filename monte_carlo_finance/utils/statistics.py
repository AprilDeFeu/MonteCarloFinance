"""
Statistical utility functions for Monte Carlo simulation analysis.
"""

from typing import Dict

import numpy as np


def calculate_statistics(
    data: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate comprehensive statistics for simulation results.

    Args:
        data: 1D or 2D array of simulation results
        confidence_level: Confidence level for VaR and CVaR calculations

    Returns:
        Dictionary containing statistical measures
    """
    # Flatten if needed for overall statistics
    flat_data = data.flatten() if data.ndim > 1 else data

    alpha = 1 - confidence_level

    return {
        "mean": float(np.mean(flat_data)),
        "std": float(np.std(flat_data)),
        "min": float(np.min(flat_data)),
        "max": float(np.max(flat_data)),
        "median": float(np.median(flat_data)),
        "skewness": float(calculate_skewness(flat_data)),
        "kurtosis": float(calculate_kurtosis(flat_data)),
        "var": float(np.percentile(flat_data, alpha * 100)),  # Value at Risk
        "cvar": float(calculate_cvar(flat_data, alpha)),  # Conditional VaR
        "percentile_5": float(np.percentile(flat_data, 5)),
        "percentile_25": float(np.percentile(flat_data, 25)),
        "percentile_75": float(np.percentile(flat_data, 75)),
        "percentile_95": float(np.percentile(flat_data, 95)),
    }


def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data.

    Args:
        data: 1D array of values

    Returns:
        Skewness value
    """
    n = len(data)
    if n < 3:
        return 0.0

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return 0.0

    return float(np.mean(((data - mean) / std) ** 3))


def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis of data.

    Args:
        data: 1D array of values

    Returns:
        Excess kurtosis value (normal distribution has kurtosis of 0)
    """
    n = len(data)
    if n < 4:
        return 0.0

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return 0.0

    return float(np.mean(((data - mean) / std) ** 4) - 3)


def calculate_cvar(
    data: np.ndarray,
    alpha: float = 0.05
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.

    Args:
        data: 1D array of values (e.g., returns or P&L)
        alpha: Tail probability (e.g., 0.05 for 5% tail)

    Returns:
        CVaR value
    """
    var = np.percentile(data, alpha * 100)
    return float(np.mean(data[data <= var]))


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Array of periodic returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year

    if np.std(excess_returns) == 0:
        return 0.0

    return float(
        np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    )


def calculate_max_drawdown(values: np.ndarray) -> float:
    """Calculate maximum drawdown from peak.

    Args:
        values: Array of portfolio/asset values

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.2 for 20% drawdown)
    """
    if len(values) == 0:
        return 0.0

    # Calculate running maximum
    running_max = np.maximum.accumulate(values)

    # Calculate drawdowns
    drawdowns = (running_max - values) / running_max

    return float(np.max(drawdowns))


def calculate_correlation_matrix(
    data: np.ndarray
) -> np.ndarray:
    """Calculate correlation matrix for multiple assets/simulations.

    Args:
        data: 2D array where rows are time periods and columns are assets

    Returns:
        Correlation matrix
    """
    return np.corrcoef(data, rowvar=False)


def calculate_rolling_statistics(
    data: np.ndarray,
    window: int,
    statistic: str = "mean"
) -> np.ndarray:
    """Calculate rolling statistics.

    Args:
        data: 1D array of values
        window: Rolling window size
        statistic: Type of statistic ("mean", "std", "min", "max")

    Returns:
        Array of rolling statistics (with NaN for initial values)
    """
    n = len(data)
    result = np.full(n, np.nan)

    stat_func = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }.get(statistic, np.mean)

    for i in range(window - 1, n):
        result[i] = stat_func(data[i - window + 1:i + 1])

    return result
