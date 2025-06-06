import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os
import matplotlib.pyplot as plt

def plot_price_and_volume(file_path, save_dir=None, show=True):
    """
    Plot price (Close) vs time and volume vs time for a given OHLCV CSV file.
    Saves the plots as PNG if save_dir is provided.
    """
    df = pd.read_csv(file_path)
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    title = os.path.basename(file_path)

    # Price vs Time
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date-Time'], df['Close'], label='Close Price', color='blue')
    plt.title(f'Close Price vs Time - {title}')
    plt.xlabel('Date-Time')
    plt.ylabel('Close Price')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title}_close_vs_time.png'))
    if show:
        plt.show()
    plt.close()

    # Volume vs Time
    plt.figure(figsize=(14, 4))
    plt.plot(df['Date-Time'], df['Volume'], label='Volume', color='orange')
    plt.title(f'Volume vs Time - {title}')
    plt.xlabel('Date-Time')
    plt.ylabel('Volume')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title}_volume_vs_time.png'))
    if show:
        plt.show()
    plt.close()

# Example usage:
# plot_price_and_volume('./resampled_1d/FEIc1_ohlcv_1d.csv')

#Create a directory in the base_path to save plots then apply the function in loop to all csv files in the directory
def plot_all_contracts_in_directory(data_dir, save_dir=None):
    """
    Plot price and volume for all contract CSV files in the directory.
    Only plots files that contain 'Close' and 'Volume' columns (i.e., outright contract files).
    Saves the plots as PNG if save_dir is provided.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        print('No CSV files found in', data_dir)
        return
    if save_dir is None:
        save_dir = data_dir
    os.makedirs(save_dir, exist_ok=True)

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        # Only plot if file has 'Close' and 'Volume' columns
        try:
            df = pd.read_csv(fpath, nrows=2)
            if 'Close' in df.columns and 'Volume' in df.columns:
                plot_price_and_volume(fpath, save_dir=save_dir, show=False)
                print(f'Plots saved for {fname} in {save_dir}')
        except Exception as e:
            print(f'Skipped {fname}: {e}')

# # Example usage:
# plot_all_contracts_in_directory('./Assignment1/Cleaned_Data/resampled_1d', save_dir='./Assignment1/Cleaned_Data/resampled_1d/plots_outright')
# plot_all_contracts_in_directory('./Assignment1/Cleaned_Data/resampled_1min', save_dir='./Assignment1/Cleaned_Data/resampled_1min/plots_outright')
# plot_all_contracts_in_directory('./Assignment1/Cleaned_Data/resampled_1h', save_dir='./Assignment1/Cleaned_Data/resampled_1h/plots_outright')
# plot_all_contracts_in_directory('./Assignment1/Cleaned_Data/resampled_15min', save_dir='./Assignment1/Cleaned_Data/resampled_15min/plots_outright')


# The function removes all rows with missing values in any column and thus removes all rows, 
# NEED TO FIX

def plot_spread_and_butterfly_from_csv(csv_path, save_dir=None, show=False):
    """
    Plots all columns from a spreads_only or butterfly_only CSV (with 'Date-Time'),
    and saves PNGs for each column. Also plots 10-period and 30-period moving averages and their ±2SD Bollinger Bands for each column.
    The Bollinger Band region is filled with a semi-transparent color for distinction.
    Only plots columns that are not 'Date-Time'.
    """
    df = pd.read_csv(csv_path)
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    value_cols = [col for col in df.columns if col != 'Date-Time']
    if not value_cols:
        print(f'No spread or butterfly columns found in {csv_path}')
        return
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    for col in value_cols:
        plt.figure(figsize=(14, 5))
        plt.plot(df['Date-Time'], df[col], label=col)
        # 10-period MA and Bollinger Bands
        ma10 = df[col].rolling(window=10, min_periods=1).mean()
        std10 = df[col].rolling(window=10, min_periods=1).std()
        upper10 = ma10 + 2 * std10
        lower10 = ma10 - 2 * std10
        plt.plot(df['Date-Time'], ma10, label=f'{col} 10MA', linestyle='--', color='orange')
        plt.fill_between(df['Date-Time'], lower10, upper10, color='orange', alpha=0.18, label=f'{col} 10MA ±2SD')
        # 30-period MA and Bollinger Bands
        ma30 = df[col].rolling(window=30, min_periods=1).mean()
        std30 = df[col].rolling(window=30, min_periods=1).std()
        upper30 = ma30 + 2 * std30
        lower30 = ma30 - 2 * std30
        plt.plot(df['Date-Time'], ma30, label=f'{col} 30MA', linestyle='--', color='green')
        plt.fill_between(df['Date-Time'], lower30, upper30, color='green', alpha=0.13, label=f'{col} 30MA ±2SD')
        plt.title(f'{col} vs Time (with 10MA, 30MA & ±2SD Bands)')
        plt.xlabel('Date-Time')
        plt.ylabel(col)
        plt.tight_layout()
        plt.legend()
        plot_filename = f'{base_name}_{col}_vs_time.png'
        plt.savefig(os.path.join(save_dir, plot_filename))
        if show:
            plt.show()
        plt.close()

# Example usage:
plot_spread_and_butterfly_from_csv('./Assignment1/Cleaned_Data/resampled_1d/spreads_only.csv', save_dir='./Assignment1/Cleaned_Data/resampled_1d/plots')
plot_spread_and_butterfly_from_csv('./Assignment1/Cleaned_Data/resampled_1d/butterfly_only.csv', save_dir='./Assignment1/Cleaned_Data/resampled_1d/plots')