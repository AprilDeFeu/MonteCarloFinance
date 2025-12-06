"""
Multi-Index Monte Carlo Simulation (Target: June 2027).

Runs simulations for:
1. Russell 3000 (Baseline)
2. S&P 500
3. Dow Jones
4. S&P/TSX

Includes specific maturity markers for One Market Plaza.
Note: Simulation starts Dec 5, 2025. Historical dates before this cannot be shown.
"""

import os
import sys
import json
import logging
from datetime import datetime, date, timedelta

import numpy as np
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

# --- Configuration ---

# Maturity Dates to Mark (Only dates AFTER simulation start: Dec 5, 2025)
# Note: Feb 17, 2025 is BEFORE the simulation starts and cannot be shown.
MATURITY_EVENTS = [
    {"date": date(2026, 2, 6), "label": "OMP Extension (Feb '26)", "color": "orange", "style": "--"},
    {"date": date(2027, 2, 5), "label": "OMP Hard Maturity (Feb '27)", "color": "darkred", "style": "-."}
]

# Index Parameters (Calibrated from analyze_indices.py)
INDEX_CONFIGS = {
    "Russell 3000": {
        "type": "baseline", # Use default/artifact logic
        "color": "steelblue"
    },
    "S&P 500": {
        "type": "explicit",
        "initial_value": 6870.4,
        "drift": 0.1301,
        "volatility": 0.1415,
        "color": "navy"
    },
    "Dow Jones": {
        "type": "explicit",
        "initial_value": 47954.99,
        "drift": 0.1622,
        "volatility": 0.1400,
        "color": "darkgreen"
    },
    "S&P/TSX": {
        "type": "explicit",
        "initial_value": 132621.88,
        "drift": 0.4726,
        "volatility": 0.1446,
        "color": "maroon"
    }
}

def _setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "multi_index_simulation.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def _load_russell_params():
    """Load regression parameters for Russell 3000 if available."""
    artifact_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "russell_regression_2025.json")
    if os.path.exists(artifact_path):
        with open(artifact_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _compute_percentile_bands(data: np.ndarray) -> dict:
    return {
        'median': np.percentile(data, 50, axis=0),
        'mean': np.mean(data, axis=0),
        'p5': np.percentile(data, 5, axis=0),
        'p25': np.percentile(data, 25, axis=0),
        'p75': np.percentile(data, 75, axis=0),
        'p95': np.percentile(data, 95, axis=0),
        'std': np.std(data, axis=0)
    }

def _plot_with_bands(ax, x_axis, bands, color, title, ylabel, subset_paths=None, reference_line=None):
    """Helper to plot bands and paths."""
    # Plot subset of individual paths (spaghetti)
    if subset_paths is not None:
        # Plot first 50 paths for visibility
        num_paths_to_plot = min(50, subset_paths.shape[0])
        for i in range(num_paths_to_plot):
            ax.plot(x_axis, subset_paths[i, :], color=color, alpha=0.05, linewidth=0.5)
            
    # Plot bands
    ax.fill_between(x_axis, bands['p5'], bands['p95'], color=color, alpha=0.1, label='5-95%')
    ax.fill_between(x_axis, bands['p25'], bands['p75'], color=color, alpha=0.2, label='25-75%')
    ax.plot(x_axis, bands['median'], color=color, linewidth=2, label='Median')
    
    if reference_line is not None:
        ax.axhline(y=reference_line, color='black', linestyle=':', alpha=0.5)
        
    # Add Maturity Markers
    for event in MATURITY_EVENTS:
        # Only plot if date is within range
        if x_axis[0] <= event['date'] <= x_axis[-1]:
            ax.axvline(x=event['date'], color=event['color'], linestyle=event['style'], 
                      alpha=0.7, linewidth=1.5, label=event['label'] if 'label' not in [l.get_label() for l in ax.get_lines()] else "")
            # Add text label near top
            ylim = ax.get_ylim()
            # ax.text(event['date'], ylim[1], event['label'], rotation=90, verticalalignment='top', fontsize=8)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

def run_multi_index_simulation():
    logger = _setup_logging()
    logger.info("Starting Multi-Index Simulation...")
    
    # Common Parameters
    start_date = date(2025, 12, 5) # From data
    target_date = date(2027, 6, 30)
    horizon_days = (target_date - start_date).days
    num_steps = horizon_days # Daily steps
    num_sims = 1000 # Reduced from 5000 for speed in multi-run, can increase
    
    # Date axis
    date_axis = [start_date + timedelta(days=i) for i in range(num_steps + 1)]
    
    # Loop through indices
    for index_name, index_params in INDEX_CONFIGS.items():
        logger.info(f"\nRunning Simulation for: {index_name}")
        print(f"Running Simulation for: {index_name}...")
        
        # 1. Configure
        # Load base Russell params for defaults
        russell_params = _load_russell_params()
        
        # Determine Market Params
        if index_params["type"] == "baseline":
            if russell_params:
                initial_val = russell_params.get("current_price", 2800.0)
                drift = russell_params.get("drift", 0.08)
                vol = russell_params.get("volatility", 0.15)
            else:
                initial_val = 2800.0
                drift = 0.08
                vol = 0.15
        else:
            initial_val = index_params["initial_value"]
            drift = index_params["drift"]
            vol = index_params["volatility"]
            
        logger.info(f"Params: Initial={initial_val}, Drift={drift:.2%}, Vol={vol:.2%}")

        # Calculate time horizon in years for the config
        time_horizon_years = num_steps / 365.0

        # Build Config
        config = SimulationConfig(
            random_seed=42,
            num_simulations=num_sims,
            num_steps=num_steps,
            time_horizon=time_horizon_years,
            
            market=MarketConfig(
                initial_value=initial_val,
                drift=drift,
                volatility=vol,
                jump_intensity=0.15,
                jump_mean=-0.06,
                jump_std=0.04,
                narrative_premium=0.20,
                zirp_dependency=0.50,
                founder_mode_intensity=0.20,  # Reduced from 0.40
                market_coupling_strength=0.20,  # Reduced from 0.40 to lower Ising criticality
                insulation_factor=0.50
            ),
            
            # Shared Configs (Same for all indices)
            bond=BondConfig(
                initial_yield=0.045,
                long_term_yield=0.055,
                yield_volatility=0.08,
                credit_spread=0.015,
                maturity_wall_2025=0.60,
                maturity_wall_2026=0.85,
                maturity_wall_2027=0.70
            ),
            consumer=ConsumerConfig(
                initial_wage_index=100.0,
                initial_col_index=110.0,
                initial_debt_index=120.0,
                productivity_growth=0.025,
                wage_lag=0.40,
                asset_inflation_weight=0.8,
                savings_rate=0.0,
                scarcity_tunneling=0.65,
                financial_nihilism=0.80,
                bnpl_usage=0.70,
                max_leverage_ratio=1.5,
                default_steepness=7.0,
                wealth_distribution_pareto=0.90,
                herd_behavior=0.45,
                wage_volatility=0.03,
                col_volatility=0.04,
                spending_volatility=0.10
            ),
            liquidity=LiquidityConfig(
                initial_level=1.0,
                mean_reversion_speed=0.5,
                volatility=0.08,
                fed_put_sensitivity=1.2,
                inflation_constraint=0.9
            ),
            shockwave=ShockwaveConfig(
                base_probability=0.004,
                yield_impact=0.01,
                market_impact=0.03,  # Reduced from 0.05
                propagation_delay=1,
                cascade_decay=0.75,
                selloff_threshold=0.05,
                selloff_intensity=0.10  # Reduced from 0.15
            ),
            panic=PanicConfig(
                base_panic_level=0.02,  # Reduced from 0.05
                panic_sensitivity=0.30,  # Reduced from 0.40
                panic_decay=0.20,  # Increased from 0.15 (faster recovery)
                max_panic_level=1.0,
                volatility_multiplier=1.5,  # Reduced from 1.8
                drift_impact=-0.05,  # Reduced from -0.08 (less negative drag)
                correlation_boost=0.3,  # Reduced from 0.4
                herd_behavior_factor=0.20,  # Reduced from 0.25
                recovery_threshold=0.25,
                fragility_factor=1.0  # Reduced from 1.2
            ),
            # PGRE: Using more accurate values from Q3 2025 10-Q
            # Initial cash: $330M, Annual burn: ~$40M (average of 2025/2026 projections)
            pgre=PGREConfig(
                initial_cash=330_000_000.0,  # Corrected from 150M
                annual_cash_burn=40_000_000.0,  # Corrected from 80M
                extension_fee_pct=0.005,  # 0.5% (conservative)
                rate_cap_cost_est=15_000_000.0,  # Corrected from 25M
                ncf_servicer=92_000_000.0,  # Optimistic (Servicer adjusted)
                ncf_stressed=60_500_000.0,  # S&P Global stressed
                debt_yield_threshold_2027=0.085,
                ncf_volatility=0.25,
                cash_burn_volatility=0.40
            )
        )
        
        # 2. Run
        sim = MonteCarloSimulation(config)
        results = sim.run()
        
        # 3. Process Results
        market_bands = _compute_percentile_bands(results.market_paths)
        consumer_default_bands = _compute_percentile_bands(results.consumer_default_prob_paths)
        bond_yield_bands = _compute_percentile_bands(results.bond_yields)
        liquidity_bands = _compute_percentile_bands(results.liquidity_paths)
        consumer_debt_bands = _compute_percentile_bands(results.consumer_debt_paths)
        pgre_cash_bands = _compute_percentile_bands(results.pgre_cash_paths)
        
        # 4. Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Monte Carlo Simulation: {index_name}\n(Drift={drift:.1%}, Vol={vol:.1%})', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Market Paths
        _plot_with_bands(
            axes[0, 0], date_axis, market_bands, index_params['color'],
            f'{index_name} Level', 'Index Level',
            subset_paths=results.market_paths, reference_line=initial_val
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
        
        # Add Legend for Maturity Lines (only once per figure)
        handles, labels = axes[0,0].get_legend_handles_labels()
        # Filter duplicates
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
        
        plt.tight_layout(rect=(0, 0.05, 1, 0.95))
        
        # Save
        safe_name = index_name.replace(" ", "_").replace("/", "_")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"simulation_results_{safe_name}.png")
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved plot to {output_path}")
        print(f"Saved plot to {output_path}")
        plt.close(fig)

    print("\nAll simulations complete.")

if __name__ == "__main__":
    run_multi_index_simulation()
