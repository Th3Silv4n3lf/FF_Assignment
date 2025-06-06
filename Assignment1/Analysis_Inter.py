import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import os
import matplotlib.pyplot as plt
import SpreadForm as sf
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf
from hurst import compute_Hc
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import seaborn as sns

# --------------------------------------------
# Correlation Matrix
# --------------------------------------------
def plot_correlation_matrix(df, title="Correlation Matrix"):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return corr

# --------------------------------------------
# Rolling Correlation Between Two Contracts
# --------------------------------------------
def plot_rolling_correlation(df, contract1, contract2, window=60):
    rolling_corr = df[contract1].rolling(window).corr(df[contract2])
    plt.figure(figsize=(12, 5))
    plt.plot(rolling_corr, label=f"{contract1}-{contract2} {window}-Day Rolling Corr")
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Rolling Correlation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spread_and_butterfly_from_csv(csv_path, save_dir=None, show=False):
    """
    Plots all columns from a spreads_only or butterfly_only CSV (with 'Date-Time'), drops rows with missing values,
    and saves PNGs for each column. Only plots columns that are not 'Date-Time'.
    """
    df = pd.read_csv(csv_path)
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    value_cols = [col for col in df.columns if col != 'Date-Time']
    if not value_cols:
        print(f'No spread or butterfly columns found in {csv_path}')
        return
    # df = df.dropna(subset=value_cols)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    for col in value_cols:
        plt.figure(figsize=(14, 5))
        plt.plot(df['Date-Time'], df[col], label=col)
        plt.title(f'{col} vs Time')
        plt.xlabel('Date-Time')
        plt.ylabel(col)
        plt.tight_layout()
        plt.legend()
        plot_filename = f'{base_name}_{col}_vs_time.png'
        plt.savefig(os.path.join(save_dir, plot_filename))
        if show:
            plt.show()
        plt.close()
def get_combined_spreads_dataframe(csv_path, nan_threshold=0.4):
    """
    Reads a spreads_only or butterfly_only CSV (with 'Date-Time'), drops columns with more than nan_threshold fraction of NaN values,
    and returns a DataFrame with only the spread/butterfly columns and 'Date-Time'.
    nan_threshold: float, e.g. 0.4 means drop columns with >40% NaN.
    """
    df = pd.read_csv(csv_path)
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    value_cols = [col for col in df.columns if col != 'Date-Time']
    if not value_cols:
        print(f'No spread or butterfly columns found in {csv_path}')
        return None
    # Drop columns with >nan_threshold NaN
    n_rows = len(df)
    cols_to_keep = [col for col in value_cols if df[col].isna().sum() / n_rows <= nan_threshold]
    if not cols_to_keep:
        print('No columns meet the NaN threshold requirement.')
        return df[['Date-Time']]
    return df[['Date-Time'] + cols_to_keep]

def compute_z_score(series, window=60):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std
def show_z_score_plot(df, column, window=60):
    """
    Plots the z-score of a given column in the DataFrame.
    """
    z_score = compute_z_score(df[column], window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date-Time'], y=z_score, mode='lines', name=f'Z-Score of {column}'))
    fig.update_layout(title=f'Z-Score of {column}', xaxis_title='Date-Time', yaxis_title='Z-Score')
    pio.show(fig)
def plot_autocorrelation(df, column, lags=40):
    """
    Plots the autocorrelation of a given column in the DataFrame.
    """
    plt.figure(figsize=(10, 5))
    plot_acf(df[column], lags=lags)
    plt.title(f'Autocorrelation of {column}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.grid()
    plt.show()

def cointegration_test(df):
    #Check cointegration between all contracts to find more favourable spreads
    for i in range(len(df.columns) - 1):
        for j in range(i + 1, len(df.columns)):
            col1 = df.columns[i]
            col2 = df.columns[j]
            if col1 != 'Date-Time' and col2 != 'Date-Time':
                score, p_value, _ = coint(df[col1], df[col2])
                # print(f'Cointegration test between {col1} and {col2}: p-value = {p_value}')
                if p_value < 0.001:
                    print(f'{col1} and {col2} are highly cointegrated (p-value < 0.001) with p-value {p_value:.6f}')
def hurst_exponent(df):
    #Check husrst exponent for each column to find mean-reverting behaviour
    hurst_results = {}
    for col in df.columns:
        if col != 'Date-Time':
            H, c, data_reg = compute_Hc(df[col].dropna())
            hurst_results[col] = H
            print(f'Hurst Exponent for {col}: {H}')
    return hurst_results

def compute_half_life_for_spreads(df):
    """
    Computes the half-life of mean reversion for all columns in the DataFrame ending with '_s'.
    Returns a dictionary with column names as keys and half-life values as values.
    """
    half_life_results = {}
    for col in df.columns:
        if col.endswith('_s'):
            series = df[col].dropna()
            lagged = series.shift(1).dropna()
            delta = series - lagged
            delta = delta.dropna()
            lagged = lagged.loc[delta.index]
            if len(lagged) > 0 and len(delta) > 0:
                model = OLS(delta, sm.add_constant(lagged)).fit()
                lambda_val = model.params[1]
                if lambda_val != 0:
                    half_life = -np.log(2) / lambda_val
                    half_life_results[col] = half_life
                else:
                    half_life_results[col] = np.nan
            else:
                half_life_results[col] = np.nan
    return half_life_results

def adf_test(df):
    """ Performs Augmented Dickey-Fuller test on each column in the DataFrame (except 'Date-Time').
    Returns a dictionary with column names as keys and p-values as values.
    """
    adf_results = {}
    for col in df.columns:
        if col != 'Date-Time':
            result = adfuller(df[col].dropna())
            adf_results[col] = result[1]  # Store p-value
            print(f'ADF Statistic for {col}: {result[0]}')
            print(f'p-value for {col}: {result[1]}')
    return adf_results










# Example usage:
combined_spreads_df = get_combined_spreads_dataframe('./Assignment1/Cleaned_Data/resampled_1d/spreads_only.csv', nan_threshold=0.2)
# combined_butterfly_df = get_combined_spreads_dataframe('./Assignment1/Cleaned_Data/resampled_1d/butterfly_only.csv', nan_threshold=0.2)
# print(combined_spreads_df.head())
#Calculating and Plotting Z-Score to analyse possible outliers for mean reversion
# if combined_spreads_df is not None:
#     # Show z-score plot for the first spread column
#     first_spread_col = combined_spreads_df.columns[1]  # Assuming the first column is 'Date-Time'
#     show_z_score_plot(combined_spreads_df, "FEIc3-FEIc2_s", window=60)

df = sf.combine_sofr_csvs_from_dir('./Assignment1/Cleaned_Data/resampled_1d/')

# #ADF Test to find stationary over past data(p-value <0.05 suggests stationary and thus mean reversion)
# #Store p-values for each spread column such that it can be sorted later
# ADF_vals = adf_test(combined_spreads_df)
# # Sort the ADF values by p-value
# ADF_vals = dict(sorted(ADF_vals.items(), key=lambda item: item[1]))
# print("Sorted ADF p-values (lower is more stationary):" , ADF_vals)

# # Cointegration test to find cointegrated pairs
# cointegration_test(combined_spreads_df)

# # Check if mean-reversion exists using autocorrelation
# plot_autocorrelation(combined_spreads_df, "FEIc2-FEIc1_s", lags=100)

# hurst_r = hurst_exponent(combined_spreads_df)
# # Sort the Hurst exponent results
# hurst_r = dict(sorted(hurst_r.items(), key=lambda item: item[1]))
# print("Sorted Hurst Exponent (lower is more mean-reverting):", hurst_r)

# half_life = compute_half_life_for_spreads(combined_spreads_df)
# # Sort the half-life results
# half_life = dict(sorted(half_life.items(), key=lambda item: item[1]))
# print("Sorted Half-Life (lower is more mean-reverting):", half_life)

plot_correlation_matrix(df,title="Correlation Matrix of Outrights")