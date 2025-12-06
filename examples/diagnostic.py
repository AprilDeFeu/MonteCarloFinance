"""
Diagnostic script to understand the simulation behavior.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datetime import date, timedelta
import numpy as np

# Check key calculation
print("=" * 60)
print("DIAGNOSTIC: Understanding the Simulation Behavior")
print("=" * 60)

# 1. PGRE Cash Analysis
print("\n--- PGRE CASH ANALYSIS ---")
initial_cash = 150_000_000  # From multi_index script
annual_burn = 80_000_000    # From multi_index script
start_date = date(2025, 12, 5)
target_date = date(2027, 6, 30)
horizon_days = (target_date - start_date).days
num_steps = horizon_days
time_horizon_years = num_steps / 365.0

print(f"Start Date: {start_date}")
print(f"Target Date: {target_date}")
print(f"Horizon Days: {horizon_days}")
print(f"Time Horizon (years): {time_horizon_years:.2f}")
print(f"Initial Cash: ${initial_cash:,.0f}")
print(f"Annual Burn: ${annual_burn:,.0f}")
print(f"Expected Cash at End (no stress): ${initial_cash - annual_burn * time_horizon_years:,.0f}")

# Days to deplete
days_to_deplete = (initial_cash / annual_burn) * 365
deplete_date = start_date + timedelta(days=days_to_deplete)
print(f"Days to Deplete Cash: {days_to_deplete:.0f}")
print(f"Approximate Depletion Date: {deplete_date}")

# Feb 17 2025 is BEFORE the simulation start!
print(f"\n!!! CRITICAL: Feb 17, 2025 is BEFORE simulation start (Dec 5, 2025) !!!")
print(f"That date cannot be marked as it's in the past relative to the simulation.")

# 2. Market Drift Analysis
print("\n--- MARKET DRIFT ANALYSIS ---")
drift = 0.1301  # S&P 500 from data
vol = 0.1415
dt = time_horizon_years / num_steps

print(f"Drift (annualized): {drift:.2%}")
print(f"Volatility (annualized): {vol:.2%}")
print(f"dt (time step): {dt:.6f}")

# Panic Model Impact
panic_drift_impact = -0.08  # From config
panic_vol_multiplier = 1.8
print(f"\nPanic Config:")
print(f"  drift_impact: {panic_drift_impact}")
print(f"  volatility_multiplier: {panic_vol_multiplier}")

# Jump Impact
jump_intensity = 0.15
jump_mean = -0.06
print(f"\nJump Config:")
print(f"  jump_intensity: {jump_intensity} (jumps/year)")
print(f"  jump_mean: {jump_mean} (per jump)")
print(f"  Expected jump drag: {jump_intensity * jump_mean:.2%} per year")

# Net Expected Return
# Drift term = (mu - 0.5*sigma^2)
base_drift_term = drift - 0.5 * vol**2
print(f"\nExpected Log Drift (base): {base_drift_term:.4f}")
print(f"Expected Log Drift (with jump drag): {base_drift_term + jump_intensity * jump_mean:.4f}")

# If panic builds up to 50% on average
avg_panic = 0.5
effective_drift = drift + panic_drift_impact * avg_panic
print(f"\nWith 50% avg panic:")
print(f"  Effective Drift: {effective_drift:.4f}")
print(f"  Effective Log Drift: {effective_drift - 0.5 * (vol * panic_vol_multiplier * avg_panic)**2:.4f}")

# Ising Model Amplification
print("\n--- ISING MODEL AMPLIFICATION ---")
coupling = 0.40  # market_coupling_strength
temperature = vol  # ~0.14
criticality = coupling / max(0.05, temperature)
ising_multiplier = np.exp(criticality * 0.1)
print(f"Coupling (J): {coupling}")
print(f"Temperature (sigma): {temperature}")
print(f"Criticality (J/T): {criticality:.2f}")
print(f"Ising Multiplier on Jump Intensity: {ising_multiplier:.2f}")
print(f"Effective Jump Intensity: {jump_intensity * ising_multiplier:.3f}")
print(f"Effective Jump Drag: {jump_intensity * ising_multiplier * jump_mean:.2%} per year")

print("\n" + "=" * 60)
print("CONCLUSION: The downward drift is caused by:")
print("1. Negative jump_mean (-6%) amplified by Ising model")
print("2. Panic model adding negative drift_modifier")
print("3. Behavioral factors (founder_mode, pe_dominance) increasing volatility")
print("=" * 60)

print("\n\nRECOMMENDATIONS:")
print("1. Reduce jump_intensity or make jump_mean less negative")
print("2. Reduce panic_drift_impact severity")
print("3. Reduce market_coupling_strength (Ising criticality)")
print("4. Add volatility/spread to Consumer Debt and PGRE Cash models")
print("5. Fix the date: Feb 17 2025 is PAST. The data already starts Dec 2025.")
