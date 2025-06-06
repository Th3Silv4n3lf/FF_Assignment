import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import SpreadForm as sf
import Analysis_Inter as ai
def calculate_hurst(series):
    series = series.dropna()
    if len(series) < 100:
        return 0.5
    H, _, _ = compute_Hc(series)
    return H

def strategy_condition_check(series, adf_significance=0.05, hurst_threshold=0.45):
    """
    Returns True if the series passes both:
    - ADF test (p < 0.05)
    - Hurst exponent test (H < 0.45 => mean reverting)
    """
    adf_pass = adf_test(series, significance=adf_significance)
    hurst_val = calculate_hurst(series)
    hurst_pass = hurst_val < hurst_threshold
    return adf_pass and hurst_pass, hurst_val

def zscore(series, window=60):
    """Calculate Z-score for a given series and rolling window."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std

def adf_test(series, significance=0.05):
    """Check for stationarity using Augmented Dickey-Fuller test."""
    series = series.dropna()
    if len(series) < 20:
        return False
    p_value = adfuller(series)[1]
    return p_value < significance

def generate_spread_signals(df, spread_col, z_window=60, entry_threshold=2, exit_threshold=1):
    """
    Generates +1 (long), -1 (short), 0 (hold) signals for a spread column.
    Only one position at a time is allowed.
    """
    signals = pd.Series(0, index=df.index)
    position = 0
    z = zscore(df[spread_col], window=z_window)

    for i in range(1, len(df)):
        if position == 0:
            # Entry logic
            if z[i] > entry_threshold:
                signals[i] = -1  # Go short
                position = -1
            elif z[i] < -entry_threshold:
                signals[i] = 1  # Go long
                position = 1
        elif position == 1:
            # Exit long
            if z[i] >= -exit_threshold:
                signals[i] = 0
                position = 0
            else:
                signals[i] = 1  # Hold long
        elif position == -1:
            # Exit short
            if z[i] <= exit_threshold:
                signals[i] = 0
                position = 0
            else:
                signals[i] = -1  # Hold short

    df[f'{spread_col}_signal'] = signals
    return df

# Example: user specifies pairs as a list of tuples [('FEIc2', 'FEIc1'), ('FEIc3', 'FEIc2')]
user_pairs = [('FEIc6', 'FEIc5'), ('FEIc3', 'FEIc2'), ('FEIc5', 'FEIc4'), ('FEIc4', 'FEIc3'), ('FEIc2', 'FEIc1')]


df = ai.get_combined_spreads_dataframe('./Assignment1/Cleaned_Data/resampled_1d/spreads_only.csv', nan_threshold=0.2)

results = []
for x, y in user_pairs:
    spread_col = f"{x}-{y}_s"
    if spread_col not in df.columns:
        print(f"Spread column {spread_col} not found in dataframe.")
        continue

    condition_pass, hurst_value = strategy_condition_check(df[spread_col])

    if condition_pass:
        df = generate_spread_signals(df, spread_col)
        results.append((x, y, True, hurst_value))
    else:
        print(f"Spread {spread_col} not suitable: Hurst={hurst_value:.2f}")
        results.append((x, y, False, hurst_value))

# Save only once, with all signals included
signal_cols = [f"{x}-{y}_s_signal" for x, y, ok, _ in results if ok]
if signal_cols:
    out_cols = ['Date-Time'] + [f"{x}-{y}_s" for x, y, ok, _ in results if ok] + signal_cols
    df[out_cols].to_csv('spread_with_signals_combined.csv', index=False)
