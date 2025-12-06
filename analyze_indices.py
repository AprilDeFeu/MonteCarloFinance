
import pandas as pd
import numpy as np
import os

def analyze_csv(filepath, name):
    try:
        # Read CSV
        # Format seems to be: "Date","Price","Open","High","Low","Vol.","Change %"
        # Dates are YYYY-MM-DD
        # Prices have commas
        df = pd.read_csv(filepath)
        
        # Clean Price column
        if 'Price' not in df.columns:
            print(f"Error: 'Price' column not found in {name}")
            return None
            
        df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        print(f"DEBUG: {name} - Price head: {df['Price'].head()}")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        print(f"DEBUG: {name} - Rows loaded: {len(df)}")
        if len(df) < 2:
            print(f"Error: Not enough data in {name}")
            return None

        # Calculate Log Returns
        df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1))
        df = df.dropna(subset=['LogReturn'])
        print(f"DEBUG: {name} - Rows after dropna: {len(df)}")
        
        # Calculate Annualized Metrics (assuming 252 trading days)
        # Use last 1 year for current regime, or full history?
        # User said "regressions... according to what you assess to be the best implementation"
        # Given the "regime shift" thesis, recent history (1-2 years) might be more relevant for drift/vol,
        # but long history is good for jump parameters.
        # Let's look at last 1 year (approx 252 days) for drift/vol calibration.
        
        recent_df = df.tail(252)
        
        daily_mean = recent_df['LogReturn'].mean()
        daily_std = recent_df['LogReturn'].std()
        
        annualized_drift = daily_mean * 252
        annualized_vol = daily_std * np.sqrt(252)
        
        last_price = df['Price'].iloc[-1]
        
        print(f"--- {name} ---")
        print(f"Last Date: {df['Date'].iloc[-1].date()}")
        print(f"Last Price: {last_price}")
        print(f"Annualized Drift (1yr): {annualized_drift:.4f}")
        print(f"Annualized Volatility (1yr): {annualized_vol:.4f}")
        
        return {
            "name": name,
            "initial_value": last_price,
            "drift": annualized_drift,
            "volatility": annualized_vol
        }
        
    except Exception as e:
        print(f"Failed to analyze {name}: {e}")
        return None

def main():
    data_dir = r"f:\April\Stuff\Finance\MonteCarloFinance\data\Historical"
    files = [
        ("S&P 500", "SP500_04041997-05122025.csv"),
        ("Dow Jones", "DowJones_04041997-05122025.csv"),
        ("S&P/TSX", "S&P_TSX_04041997-05122025.csv")
    ]
    
    results = []
    for name, filename in files:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            res = analyze_csv(path, name)
            if res:
                results.append(res)
        else:
            print(f"File not found: {path}")

if __name__ == "__main__":
    main()
