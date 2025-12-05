"""
Random number generation utilities for Monte Carlo simulations.

Provides a consistent interface for random number generation with
support for reproducibility through seeding.
"""

from typing import Optional, Tuple

import numpy as np


class RandomGenerator:
    """Random number generator for Monte Carlo simulations.

    This class wraps numpy's random number generation to provide
    consistent random number generation across all simulation components.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the random generator.

        Args:
            seed: Random seed for reproducibility. If None, uses random state.
        """
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def seed(self) -> Optional[int]:
        """Get the seed used for this generator."""
        return self._seed

    def normal(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        size: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Generate normal (Gaussian) random numbers.

        Args:
            mean: Mean of the distribution
            std: Standard deviation of the distribution
            size: Output shape

        Returns:
            Array of random numbers
        """
        return self._rng.normal(mean, std, size)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Generate uniform random numbers.

        Args:
            low: Lower bound
            high: Upper bound
            size: Output shape

        Returns:
            Array of random numbers
        """
        return self._rng.uniform(low, high, size)

    def poisson(
        self,
        lam: float = 1.0,
        size: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Generate Poisson random numbers.

        Args:
            lam: Expected number of events (lambda)
            size: Output shape

        Returns:
            Array of random integers
        """
        return self._rng.poisson(lam, size)

    def exponential(
        self,
        scale: float = 1.0,
        size: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Generate exponential random numbers.

        Args:
            scale: Scale parameter (1/rate)
            size: Output shape

        Returns:
            Array of random numbers
        """
        return self._rng.exponential(scale, size)

    def bernoulli(
        self,
        p: float = 0.5,
        size: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Generate Bernoulli random numbers (0 or 1).

        Args:
            p: Probability of success (1)
            size: Output shape

        Returns:
            Array of 0s and 1s
        """
        return (self._rng.uniform(size=size) < p).astype(int)

    def correlated_normal(
        self,
        correlation: float,
        size: Optional[Tuple[int, ...]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two correlated normal random variables.

        Uses Cholesky decomposition to generate correlated normals.

        Args:
            correlation: Correlation coefficient (-1 to 1)
            size: Output shape for each variable

        Returns:
            Tuple of two correlated normal arrays
        """
        z1 = self.normal(size=size)
        z2 = self.normal(size=size)

        # Apply correlation using Cholesky decomposition
        x1 = z1
        x2 = correlation * z1 + np.sqrt(1 - correlation**2) * z2

        return x1, x2

    def multivariate_normal(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        size: Optional[int] = None
    ) -> np.ndarray:
        """Generate multivariate normal random numbers.

        Args:
            mean: Mean vector
            cov: Covariance matrix
            size: Number of samples

        Returns:
            Array of random vectors
        """
        return self._rng.multivariate_normal(mean, cov, size)
