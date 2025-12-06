"""
Russell 3000 Regression Analysis Script.

This script performs a multi-factor regression analysis to determine the
sensitivity (betas) of the Russell 3000 index (or proxy) to various
macroeconomic and financial factors.

The results are used to parameterize the MarketModel in the Monte Carlo simulation.
"""

import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm


# Define data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FRED_DIR = os.path.join(DATA_DIR, "FRED")
GOLD_DIR = os.path.join(DATA_DIR, "Gold")
HISTORICAL_DIR = os.path.join(DATA_DIR, "Historical")

def load_fred_data(filename, column_name):
    """Load and clean FRED data."""
    path = os.path.join(FRED_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # FRED often uses '.' for missing data
    df = df.replace('.', np.nan).astype(float)
    df = df.dropna()
    df.columns = [column_name]
    return df


def load_russell_data():
    """Load Russell 3000 total return data from Historical folder.

    Returns a DataFrame indexed by date with a `RUSSELL` column of prices.
    """
    path = os.path.join(HISTORICAL_DIR, "Russell3000TotalReturn_04041997-05122025.csv")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None

    df = pd.read_csv(path)
    # Normalize columns
    df.columns = [c.strip().replace('"', '') for c in df.columns]

    # Parse dates and numeric prices (strip commas)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = df["Price"].str.replace(",", "", regex=False).astype(float)
    df = df.sort_values("Date").set_index("Date")
    df = df[["Price"]].rename(columns={"Price": "RUSSELL"})
    return df

def load_gold_data():
    """Load Gold price data."""
    path = os.path.join(GOLD_DIR, "AUX-USD.csv")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
        
    # Assuming format needs parsing as per instructions
    try:
        df = pd.read_csv(path)
        # Instructions say: Parse 'DD-MMM-YYYY' format
        # We need to identify the date column. Let's assume it's the first one.
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], format='%d-%b-%Y')
        df = df.set_index(date_col)
        df = df.sort_index()
        # Keep closing price, assume it's the second column or named 'Close'
        # For now, let's take the first numeric column after date
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df = df[[numeric_cols[0]]]
            df.columns = ['Gold']
            return df
    except Exception as e:
        print(f"Error loading gold data: {e}")
        return None
    return None

def run_regression():
    """Run the regression analysis."""
    print("Loading data...")

    # 1. Dependent Variable: Russell 3000 Total Return
    russell = load_russell_data()
    if russell is None:
        print("Russell 3000 data not found. Cannot run regression.")
        return

    # Focus on Jan 1 2025 through Dec 5 2025
    start_window = pd.Timestamp("2025-01-01")
    end_window = pd.Timestamp("2025-12-05")
    russell = russell.loc[(russell.index >= start_window) & (russell.index <= end_window)]
    if russell.empty:
        print("Russell data window is empty.")
        return

    # 2. Load Independent Variables
    # Yield Curve
    t10y2y = load_fred_data("T10Y2Y.csv", "T10Y2Y")

    # Reverse Repo
    rrp = load_fred_data("RRPONTSYD.csv", "RRP")

    # Mortgage Rates
    mortgage = load_fred_data("MORTGAGE30US.csv", "Mortgage30")

    # Unemployment
    u6 = load_fred_data("U6RATE.csv", "U6")

    # Gold
    gold = load_gold_data()

    # Combine all dataframes
    data_frames = [russell, t10y2y, rrp, mortgage, u6, gold]
    data_frames = [df for df in data_frames if df is not None]
    
    if not data_frames:
        print("No data loaded.")
        return

    # Merge on index (Date)
    df = pd.concat(data_frames, axis=1)
    
    # Forward fill missing data (e.g. daily vs weekly) then drop remaining NaNs
    df = df.ffill().dropna()
    
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head())
    
    # Calculate Returns / Changes
    # For regression, we often use log returns for prices and raw changes for rates
    
    df['RUSSELL_Ret'] = np.log(df['RUSSELL'] / df['RUSSELL'].shift(1))
    if 'Gold' in df.columns:
        df['Gold_Ret'] = np.log(df['Gold'] / df['Gold'].shift(1))
    
    # For rates, we might use changes
    if 'T10Y2Y' in df.columns:
        df['T10Y2Y_Chg'] = df['T10Y2Y'].diff()
    if 'Mortgage30' in df.columns:
        df['Mortgage30_Chg'] = df['Mortgage30'].diff()
    if 'U6' in df.columns:
        df['U6_Chg'] = df['U6'].diff()
        
    # Drop NaNs created by differencing
    df = df.dropna()
    
    # Define Regression Model
    # Y = RUSSELL_Ret
    # X = [T10Y2Y_Chg, Mortgage30_Chg, U6_Chg, Gold_Ret, ...]

    y = df['RUSSELL_Ret']
    
    features = []
    if 'T10Y2Y_Chg' in df.columns: features.append('T10Y2Y_Chg')
    if 'Mortgage30_Chg' in df.columns: features.append('Mortgage30_Chg')
    if 'U6_Chg' in df.columns: features.append('U6_Chg')
    if 'Gold_Ret' in df.columns: features.append('Gold_Ret')
    
    if not features:
        print("Not enough features for regression.")
        return
        
    X = df[features]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()

    print("\nRegression Results:")
    print(model.summary())

    annualized_vol = model.resid.std() * np.sqrt(252)
    annualized_drift = y.mean() * 252

    print("\nDerived Parameters for Simulation:")
    print(f"Annualized Drift (mean return): {annualized_drift:.4f}")
    print(f"Base Volatility (Std Dev of Residuals): {annualized_vol:.4f} (Annualized)")

    for feature in features:
        print(f"Beta for {feature}: {model.params[feature]:.4f}")

    # Persist parameters for downstream Monte Carlo usage
    out = {
        "window_start": str(start_window.date()),
        "window_end": str(end_window.date()),
        "annualized_drift": float(annualized_drift),
        "annualized_vol": float(annualized_vol),
        "betas": {feature: float(model.params[feature]) for feature in features},
        "residual_std": float(model.resid.std()),
    }

    artifact_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "russell_regression_2025.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved regression parameters to artifacts/russell_regression_2025.json")

if __name__ == "__main__":
    run_regression()
