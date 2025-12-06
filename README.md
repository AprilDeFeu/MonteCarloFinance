# MonteCarloFinance

## Overview

**MonteCarloFinance** is a high-fidelity stochastic simulation library designed to model complex financial market dynamics, credit risk, and macroeconomic interactions. Unlike traditional Black-Scholes models, this framework integrates behavioral economics, liquidity cycles, and discrete event triggers to simulate "fat-tailed" risks and systemic fragility.

The core engine employs a multi-agent approach where **Market**, **Bond**, **Consumer**, and **Liquidity** models interact dynamically, driven by a central **Monte Carlo** orchestrator.

## Mathematical Framework

The simulation is built upon a system of coupled Stochastic Differential Equations (SDEs) and discrete state transitions.

### 1. Market Dynamics (Jump-Diffusion with Regime Switching)

The asset price $S_t$ evolves according to a Geometric Brownian Motion (GBM) augmented with Poisson jumps and a regime-dependent drift:

$$
\frac{dS_t}{S_t} = \mu(L_t, P_t) dt + \sigma(P_t) dW_t + J dN_t
$$

Where:
- $\mu(L_t, P_t)$: Drift term, modulated by Global Liquidity $L_t$ and Panic Level $P_t$.
- $\sigma(P_t)$: Volatility, amplified by Panic $P_t$ (Regime Switching).
- $dW_t$: Standard Wiener process (Brownian motion).
- $dN_t$: Poisson process with intensity $\lambda$, representing discrete shock events.
- $J$: Jump size distribution (typically $\mathcal{N}(\mu_J, \sigma_J)$).

### 2. Global Liquidity Model (Ornstein-Uhlenbeck)

Liquidity $L_t$ acts as a mean-reverting force ("The Tide") that influences asset correlations and drift:

$$
dL_t = \theta (\bar{L} - L_t) dt + \sigma_L dW_t^L
$$

Where:
- $\theta$: Speed of mean reversion.
- $\bar{L}$: Long-term liquidity equilibrium.
- $\sigma_L$: Volatility of central bank interventions.

### 3. Consumer Credit & The "Hollow Middle" (Ising Model)

Consumer default probability is modeled as a phase transition using an Ising Model framework, capturing the non-linear nature of social contagion and financial stress.

The default probability $P(\text{Default})$ is a function of the "Energy" state $E$ of the consumer balance sheet:

$$
P(\text{Default}) \propto \frac{1}{1 + e^{-\beta (H + J \sum s_i)}}
$$

Where:
- $H$: External field (Macroeconomic stress: Unemployment, Inflation).
- $J$: Coupling constant (Social contagion / Herding behavior).
- $\beta$: Inverse "temperature" (Market volatility).

This allows the model to simulate spontaneous symmetry breaking—where a stable consumer base suddenly cascades into default.

### 4. Bond Yields (Vasicek Model)

Interest rates $r_t$ follow a mean-reverting stochastic process:

$$
dr_t = a(b - r_t) dt + \sigma_r dW_t^r
$$

Where:
- $a$: Speed of reversion.
- $b$: Long-term mean rate.
- $\sigma_r$: Volatility of rates.

### 5. Idiosyncratic Risk: One Market Plaza (PGRE)

Specific assets like One Market Plaza are modeled with discrete event triggers and stochastic cash flows:

- **Cash Burn**: Modeled as a stochastic process with high volatility ($\sigma_{burn} \approx 40\%$) to capture TI/LC cost uncertainty.
- **Net Cash Flow (NCF)**: Evolves with mean reversion towards a stress-dependent target (Servicer vs. S&P Scenarios).
- **Triggers**:
  - **Feb 6, 2026**: Extension Hurdle (Liquidity Test).
  - **Feb 5, 2027**: Hard Maturity / Forbearance Test (Debt Yield $\ge 8.5\%$).

## Directory Structure

```
MonteCarloFinance/
├── data/                   # Raw financial data (CSV, PDF, HTML)
├── examples/               # Simulation runners and diagnostic scripts
├── graphs/                 # Generated simulation plots (PNG)
├── logs/                   # Execution logs
├── monte_carlo_finance/    # Core library package
│   ├── core/               # Orchestrator and Configuration
│   ├── models/             # Mathematical models (Market, Bond, Consumer, PGRE)
│   ├── triggers/           # Event triggers (Panic, Shockwaves)
│   └── utils/              # Math and statistical utilities
└── tests/                  # Unit tests
```

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running a Simulation

To run the multi-index simulation (Russell 3000, S&P 500, Dow, TSX) with the latest stochastic models:

```bash
python examples/multi_index_simulation.py
```

Results will be saved to the `graphs/` directory, and execution details will be logged to `logs/`.

## Configuration

All simulation parameters are centralized in `monte_carlo_finance/core/config.py`. Key parameters include:

- `SimulationConfig`: Time horizon, steps, seed.
- `MarketConfig`: Drift, volatility, jump intensity, Ising coupling.
- `ConsumerConfig`: Wage lag, asset inflation weight, scarcity tunneling.
- `PGREConfig`: Loan balance, cash burn rate, covenant thresholds.

## License

Proprietary and Confidential.
