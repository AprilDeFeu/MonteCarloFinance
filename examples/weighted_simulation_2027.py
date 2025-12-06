"""
Weighted Monte Carlo Simulation (Target: June 2027).

This script runs a simulation configured with the "Economic Forensic Analysis"
parameters, targeting a horizon of June 2027.

Outputs:
  - simulation_results_2027.png: Visualization with percentile bands
  - logs/simulation_validation.log: Numerical validation log for model diagnostics
"""

import os
import sys
import json
import logging
from datetime import datetime, date

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from monte_carlo_finance.core.config import (
    SimulationConfig, MarketConfig, BondConfig,
    LiquidityConfig, ConsumerConfig, PGREConfig,
    ShockwaveConfig, PanicConfig
)
from monte_carlo_finance.core.simulation import MonteCarloSimulation


def _setup_logging():
    """Setup logging for numerical validation."""
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "simulation_validation.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def _load_regression_params():
    """Load regression parameters if available."""
    artifact_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "russell_regression_2025.json")
    if os.path.exists(artifact_path):
        with open(artifact_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _compute_percentile_bands(data: np.ndarray) -> dict:
    """Compute median, 1-sigma, and 5/95 percentile bands.
    
    Args:
        data: Shape (num_sims, num_steps+1)
        
    Returns:
        dict with keys: median, mean, p5, p25, p75, p95, std
    """
    return {
        'median': np.percentile(data, 50, axis=0),
        'mean': np.mean(data, axis=0),
        'p5': np.percentile(data, 5, axis=0),
        'p25': np.percentile(data, 25, axis=0),
        'p75': np.percentile(data, 75, axis=0),
        'p95': np.percentile(data, 95, axis=0),
        'std': np.std(data, axis=0)
    }


def _log_validation_metrics(logger, name: str, data: np.ndarray, bands: dict, 
                            expected_range=None, check_linearity=False):
    """Log numerical validation metrics for a data series.
    
    Args:
        logger: Logger instance
        name: Name of the metric
        data: Raw data array (num_sims, num_steps+1)
        bands: Pre-computed percentile bands
        expected_range: Optional (min, max) expected range for final values
        check_linearity: If True, perform detailed linearity analysis
    """
    final_values = data[:, -1]
    initial_values = data[:, 0]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATION: {name}")
    logger.info(f"{'='*60}")
    
    # Distribution statistics
    logger.info(f"Initial Value: {initial_values[0]:.4f}")
    logger.info(f"Final Mean: {np.mean(final_values):.4f}")
    logger.info(f"Final Median: {np.median(final_values):.4f}")
    logger.info(f"Final Std Dev: {np.std(final_values):.4f}")
    logger.info(f"Final Skewness: {sp_stats.skew(final_values):.4f}")
    logger.info(f"Final Kurtosis: {sp_stats.kurtosis(final_values):.4f}")
    
    # Percentile ranges
    logger.info(f"5th Percentile (Final): {bands['p5'][-1]:.4f}")
    logger.info(f"25th Percentile (Final): {bands['p25'][-1]:.4f}")
    logger.info(f"75th Percentile (Final): {bands['p75'][-1]:.4f}")
    logger.info(f"95th Percentile (Final): {bands['p95'][-1]:.4f}")
    
    # Check for anomalies
    pct_zero = np.mean(final_values <= 0) * 100
    pct_extreme = np.mean(np.abs(final_values) > 1e6) * 100
    logger.info(f"Pct at/below zero: {pct_zero:.2f}%")
    logger.info(f"Pct extreme (>1e6): {pct_extreme:.2f}%")
    
    # Trend analysis (linear regression on median path)
    x = np.arange(len(bands['median']))
    y = bands['median']
    slope, intercept, r_val, _, _ = sp_stats.linregress(x, y)
    r2_linear = float(r_val) ** 2
    logger.info(f"Median Trend Slope: {slope:.6f}")
    logger.info(f"Linear R²: {r2_linear:.4f}")
    
    # Linearity analysis - compare linear vs polynomial fit
    if check_linearity:
        logger.info("--- LINEARITY ANALYSIS ---")
        
        # Linear fit residuals
        y_linear_pred = float(intercept) + float(slope) * x
        ss_res_linear = np.sum((y - y_linear_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Quadratic fit (ax² + bx + c)
        coeffs_quad = np.polyfit(x, y, 2)
        y_quad_pred = np.polyval(coeffs_quad, x)
        ss_res_quad = np.sum((y - y_quad_pred) ** 2)
        r2_quadratic = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0
        
        # Cubic fit (ax³ + bx² + cx + d)
        coeffs_cubic = np.polyfit(x, y, 3)
        y_cubic_pred = np.polyval(coeffs_cubic, x)
        ss_res_cubic = np.sum((y - y_cubic_pred) ** 2)
        r2_cubic = 1 - (ss_res_cubic / ss_tot) if ss_tot > 0 else 0
        
        logger.info(f"Quadratic R²: {r2_quadratic:.4f}")
        logger.info(f"Cubic R²: {r2_cubic:.4f}")
        
        # Improvement from linear
        quad_improvement = (r2_quadratic - r2_linear) * 100
        cubic_improvement = (r2_cubic - r2_linear) * 100
        logger.info(f"Quadratic R² improvement over Linear: {quad_improvement:.2f}%")
        logger.info(f"Cubic R² improvement over Linear: {cubic_improvement:.2f}%")
        
        # Detect non-linearity
        if quad_improvement > 1.0 or cubic_improvement > 2.0:
            logger.info(">>> NON-LINEAR BEHAVIOR DETECTED <<<")
            # Check for S-curve pattern (cubic with sign changes)
            if len(coeffs_cubic) >= 4 and abs(coeffs_cubic[0]) > 1e-10:
                logger.info("Cubic term significant - possible S-curve/phase transition")
        else:
            logger.warning(">>> WARNING: PATH IS PREDOMINANTLY LINEAR <<<")
            logger.warning("Expected sigmoid/phase-transition behavior not detected.")
        
        # Rate of change analysis (first derivative)
        dy = np.diff(y)
        early_rate = np.mean(np.abs(dy[:len(dy)//3]))
        mid_rate = np.mean(np.abs(dy[len(dy)//3:2*len(dy)//3]))
        late_rate = np.mean(np.abs(dy[2*len(dy)//3:]))
        logger.info(f"Avg Rate of Change - Early: {early_rate:.6f}, Mid: {mid_rate:.6f}, Late: {late_rate:.6f}")
        
        # S-curve detection: mid-section should have different rate than early/late
        rate_ratio_early_mid = mid_rate / early_rate if early_rate > 0 else 1.0
        rate_ratio_mid_late = late_rate / mid_rate if mid_rate > 0 else 1.0
        logger.info(f"Rate Ratio (Mid/Early): {rate_ratio_early_mid:.2f}")
        logger.info(f"Rate Ratio (Late/Mid): {rate_ratio_mid_late:.2f}")
    
    # Expected range check
    if expected_range:
        in_range = np.mean((final_values >= expected_range[0]) & 
                          (final_values <= expected_range[1])) * 100
        logger.info(f"Pct within expected range {expected_range}: {in_range:.2f}%")
    
    # Stationarity check (variance ratio)
    early_var = np.var(data[:, :len(data[0])//4])
    late_var = np.var(data[:, -len(data[0])//4:])
    variance_ratio = late_var / early_var if early_var > 0 else float('inf')
    logger.info(f"Variance Ratio (late/early): {variance_ratio:.4f}")


def _plot_with_bands(ax, date_axis, bands: dict, color: str, title: str, 
                     ylabel: str, subset_paths=None, reference_line=None):
    """Plot data with 1-sigma and 5/95 percentile bands using date X-axis.
    
    Args:
        ax: Matplotlib axis
        date_axis: X-axis date values
        bands: Dict with median, p5, p25, p75, p95, std
        color: Primary color
        title: Plot title
        ylabel: Y-axis label
        subset_paths: Optional array of individual paths to overlay
        reference_line: Optional horizontal reference value
    """
    # Ensure date axis is numeric for matplotlib and keeps labels correct
    date_axis_num = mdates.date2num(date_axis)
    
    # Plot subset of individual paths (very light)
    if subset_paths is not None:
        ax.plot(date_axis_num, subset_paths[:30].T, alpha=0.03, color=color, linewidth=0.5)
    
    # 5th-95th percentile band (outer)
    ax.fill_between(date_axis_num, bands['p5'], bands['p95'], 
                    alpha=0.15, color=color, label='5th-95th Percentile')
    
    # 25th-75th percentile band (1-sigma approximation)
    ax.fill_between(date_axis_num, bands['p25'], bands['p75'], 
                    alpha=0.30, color=color, label='25th-75th Percentile (≈1σ)')
    
    # Median line
    ax.plot(date_axis_num, bands['median'], color=color, linewidth=2, 
            label='Median', linestyle='-')
    
    # Mean line (dashed)
    ax.plot(date_axis_num, bands['mean'], color=color, linewidth=1.5, 
            label='Mean', linestyle='--', alpha=0.8)
    
    # Reference line
    if reference_line is not None:
        ax.axhline(reference_line, color='black', linestyle=':', 
                   linewidth=1.5, label=f'Reference: {reference_line:.0f}')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dates', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Format date axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def _log_axis_metadata(logger, fig, axes):
    """Log axis labels and tick samples to verify plot formatting."""
    for i, axis in enumerate(axes.flatten()):
        xticks = axis.get_xticklabels()
        yticks = axis.get_yticklabels()
        xtick_text = [t.get_text() for t in xticks[:3]]
        ytick_text = [t.get_text() for t in yticks[:3]]
        logger.info(f"AXIS[{i}] title='{axis.get_title()}', xlabel='{axis.get_xlabel()}', ylabel='{axis.get_ylabel()}'")
        logger.info(f"AXIS[{i}] sample x-ticks={xtick_text}, sample y-ticks={ytick_text}")


def run_weighted_simulation():
    logger = _setup_logging()
    logger.info("="*60)
    logger.info("WEIGHTED MONTE CARLO SIMULATION - TARGET: JUNE 2027")
    logger.info("="*60)
    
    print("Initializing Weighted Simulation (Target: June 2027)...")

    # 1. Calibrate from regression (Jan 1 2025 to Dec 5 2025)
    params = _load_regression_params()
    inferred_drift = 0.06  # fallback annual drift
    inferred_vol = 0.18    # fallback annual vol
    if params:
        inferred_drift = params.get("annualized_drift", inferred_drift)
        inferred_vol = params.get("annualized_vol", inferred_vol)
        logger.info(f"Loaded regression params: drift={inferred_drift:.4f}, vol={inferred_vol:.4f}")

    # Current level (Dec 5, 2025)
    current_level = 22807.90

    # Timeline: today to Jun 1 2027
    start_date = date(2025, 12, 5)
    end_date = date(2027, 6, 1)
    horizon_days = (end_date - start_date).days
    time_horizon_years = horizon_days / 365.0
    num_steps = int(252 * time_horizon_years)
    
    logger.info(f"Simulation Period: {start_date} to {end_date}")
    logger.info(f"Horizon: {time_horizon_years:.2f} years, {num_steps} steps")

    config = SimulationConfig(
        start_date=str(start_date),
        time_horizon=time_horizon_years,
        num_steps=num_steps,
        num_simulations=5000,
        random_seed=42,

        # Market Configuration (calibrated)
        market=MarketConfig(
            initial_value=current_level,
            drift=inferred_drift,
            volatility=inferred_vol,
            jump_intensity=0.15,
            jump_mean=-0.06,
            jump_std=0.04,
            narrative_premium=0.20,
            zirp_dependency=0.50,
            founder_mode_intensity=0.40,
            market_coupling_strength=0.40,
            insulation_factor=0.50
        ),

        # Bond Configuration (Maturity Walls, scaled probabilities)
        bond=BondConfig(
            initial_yield=0.045,
            long_term_yield=0.055,
            yield_volatility=0.08,
            credit_spread=0.015,
            maturity_wall_2025=0.60,
            maturity_wall_2026=0.85,
            maturity_wall_2027=0.70
        ),

        # Consumer Configuration (phase-transition default, tuned for Ising dynamics)
        # Start with leverage BELOW threshold to observe S-curve phase transition
        consumer=ConsumerConfig(
            initial_wage_index=100.0,
            initial_col_index=110.0,  # CoL slightly above wages
            initial_debt_index=120.0,  # Leverage = 1.2 (below 1.5 threshold)
            productivity_growth=0.025,
            wage_lag=0.40,
            asset_inflation_weight=0.8,
            savings_rate=0.0,
            scarcity_tunneling=0.65,
            financial_nihilism=0.80,
            bnpl_usage=0.70,
            max_leverage_ratio=1.5,  # Phase transition threshold
            default_steepness=7.0,   # Sharper transition to reduce linear drift appearance
            wealth_distribution_pareto=0.90,
            herd_behavior=0.45      # Amplify contagion near threshold (Ising coupling)
        ),

        # Liquidity Configuration
        liquidity=LiquidityConfig(
            initial_level=1.0,
            mean_reversion_speed=0.5,
            volatility=0.08,
            fed_put_sensitivity=1.2,
            inflation_constraint=0.9
        ),

        # Shockwave/Panic tuning to avoid immediate collapse
        shockwave=ShockwaveConfig(
            base_probability=0.004,
            yield_impact=0.01,
            market_impact=0.05,
            propagation_delay=1,
            cascade_decay=0.75,
            selloff_threshold=0.05,
            selloff_intensity=0.15
        ),

        panic=PanicConfig(
            base_panic_level=0.05,
            panic_sensitivity=0.40,
            panic_decay=0.15,
            max_panic_level=1.0,
            volatility_multiplier=1.8,
            drift_impact=-0.08,
            correlation_boost=0.4,
            herd_behavior_factor=0.25,
            recovery_threshold=0.25,
            fragility_factor=1.2
        ),

        # PGRE Specifics (One Market Plaza)
        pgre=PGREConfig(
            initial_cash=150_000_000.0,
            annual_cash_burn=80_000_000.0,
            extension_fee_pct=0.007,
            rate_cap_cost_est=25_000_000.0,
            ncf_servicer=88_000_000.0,
            ncf_stressed=55_000_000.0,
            debt_yield_threshold_2027=0.085
        )
    )
    
    # Log configuration
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Market: drift={config.market.drift:.4f}, vol={config.market.volatility:.4f}")
    logger.info(f"Consumer: leverage_thresh={config.consumer.max_leverage_ratio:.2f}, steepness={config.consumer.default_steepness:.2f}")
    logger.info(f"PGRE: cash=${config.pgre.initial_cash/1e6:.1f}M, burn=${config.pgre.annual_cash_burn/1e6:.1f}M/yr")
    
    # 2. Run Simulation
    sim = MonteCarloSimulation(config)
    results = sim.run()
    
    # 3. Calculate percentile bands for all series
    # Create date axis for plotting
    from datetime import timedelta
    date_axis = [start_date + timedelta(days=i * horizon_days / num_steps) 
                 for i in range(num_steps + 1)]
    logger.info(f"Date axis start={date_axis[0]}, end={date_axis[-1]}, total_points={len(date_axis)}")
    
    market_bands = _compute_percentile_bands(results.market_paths)
    consumer_default_bands = _compute_percentile_bands(results.consumer_default_prob_paths)
    bond_yield_bands = _compute_percentile_bands(results.bond_yields)
    liquidity_bands = _compute_percentile_bands(results.liquidity_paths)
    consumer_debt_bands = _compute_percentile_bands(results.consumer_debt_paths)
    pgre_cash_bands = _compute_percentile_bands(results.pgre_cash_paths)
    
    # 4. Log validation metrics
    _log_validation_metrics(logger, "MARKET VALUE (Russell 3000)", 
                           results.market_paths, market_bands,
                           expected_range=(current_level * 0.5, current_level * 2.0))
    _log_validation_metrics(logger, "CONSUMER DEFAULT PROBABILITY",
                           results.consumer_default_prob_paths, consumer_default_bands,
                           expected_range=(0.0, 1.0), check_linearity=True)
    _log_validation_metrics(logger, "BOND YIELD",
                           results.bond_yields, bond_yield_bands,
                           expected_range=(0.01, 0.15))
    _log_validation_metrics(logger, "LIQUIDITY INDEX",
                           results.liquidity_paths, liquidity_bands,
                           expected_range=(0.5, 2.0))
    _log_validation_metrics(logger, "CONSUMER DEBT INDEX",
                           results.consumer_debt_paths, consumer_debt_bands)
    _log_validation_metrics(logger, "PGRE CASH",
                           results.pgre_cash_paths, pgre_cash_bands)
    
    # 5. Summary Statistics
    final_values = results.market_paths[:, -1]
    mean_return = np.mean(final_values / current_level - 1.0)
    crash_prob = np.mean(final_values < current_level * 0.7)
    default_prob_final = np.mean(results.consumer_default_prob_paths[:, -1])
    pgre_default_rate = np.mean(results.pgre_defaults)
    total_shocks = results.statistics["shockwave"]["total_events"]
    
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Mean Market Return: {mean_return:.2%}")
    logger.info(f"Crash Probability (>30% drop): {crash_prob:.2%}")
    logger.info(f"Avg Consumer Default Prob (June 2027): {default_prob_final:.2%}")
    logger.info(f"PGRE Default Probability: {pgre_default_rate:.2%}")
    logger.info(f"Total Shockwave Events: {total_shocks}")
    
    print("\nSimulation Complete.")
    print("-" * 50)
    print(f"Mean Market Return: {mean_return:.2%}")
    print(f"Crash Probability (>30% drop): {crash_prob:.2%}")
    print(f"Avg Consumer Default Prob (June 2027): {default_prob_final:.2%}")
    print(f"PGRE Default Probability: {pgre_default_rate:.2%}")
    print(f"Total Shockwave Events: {total_shocks}")
    
    # 6. Visualization with proper bands and labels
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Monte Carlo Simulation: Dec 2025 → June 2027\n(Russell 3000 Calibrated)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Market Paths
    _plot_with_bands(
        axes[0, 0], date_axis, market_bands, 'steelblue',
        'Russell 3000 Index', 'Index Level',
        subset_paths=results.market_paths, reference_line=current_level
    )
    
    # Plot 2: Consumer Default Probability
    _plot_with_bands(
        axes[0, 1], date_axis, consumer_default_bands, 'crimson',
        'Consumer Default Probability', 'Probability'
    )
    
    # Plot 3: Bond Yields
    _plot_with_bands(
        axes[0, 2], date_axis, bond_yield_bands, 'forestgreen',
        'Bond Yields', 'Yield (%)',
        subset_paths=results.bond_yields
    )
    
    # Plot 4: Liquidity Index
    _plot_with_bands(
        axes[1, 0], date_axis, liquidity_bands, 'purple',
        'Liquidity Index', 'Index Level',
        subset_paths=results.liquidity_paths, reference_line=1.0
    )
    
    # Plot 5: Consumer Debt Index
    _plot_with_bands(
        axes[1, 1], date_axis, consumer_debt_bands, 'darkorange',
        'Consumer Debt Index', 'Debt Index'
    )
    
    # Plot 6: PGRE Cash Position
    _plot_with_bands(
        axes[1, 2], date_axis, pgre_cash_bands, 'darkred',
        'PGRE (One Market Plaza) Cash', 'Cash ($)',
        reference_line=0
    )
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    _log_axis_metadata(logger, fig, axes)
    output_path = os.path.join(os.path.dirname(__file__), "simulation_results_2027.png")
    plt.savefig(output_path, dpi=150)
    logger.info(f"\nPlots saved to {output_path}")
    print(f"\nPlots saved to {output_path}")
    print("Validation log saved to logs/simulation_validation.log")


if __name__ == "__main__":
    run_weighted_simulation()
