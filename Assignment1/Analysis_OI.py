import os
import pandas as pd
from resampling import trim_years as resamp_trim_years
def merge_oi_with_resampled(oi_excel_path, cleaned_data_dir, save_dir):
    """
    Merges OI data from Excel with resampled FEIcm data by contract and Date-Time.
    Saves merged DataFrames as CSVs in save_dir.
    """
    # Read OI data
    oi_df = pd.read_excel(oi_excel_path)
    oi_df['Date-Time'] = pd.to_datetime(oi_df['Date-Time']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f'Original OI DataFrame shape: {oi_df.shape}')
    print(f'Original OI DataFrame datetime head:\n', oi_df['Date-Time'].head())
    resamp_trim_years(oi_df, 2006, 2009)  # Trim OI data to the same years as FEIcm data

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=False)

    # Loop through all FEIcm resampled files
    for i in range(1, 17):
        feicm_file = os.path.join(cleaned_data_dir, f'FEIcm{i}_ohlcv_1d.csv')
        if not os.path.exists(feicm_file):
            print(f'File not found: {feicm_file}. Skipping...')
            continue
        
        feicm_df = pd.read_csv(feicm_file)
        feicm_df['Date-Time'] = pd.to_datetime(feicm_df['Date-Time'])

        # Filter OI for this contract
        contract_name = f'FEIcm{i}'

        # Merge on date
        merged_df = pd.merge(feicm_df, oi_df[['Date-Time', contract_name]], on='Date-Time', how='left')
        
        #Show merged DataFrame
        print(f'Merged DataFrame for {contract_name}:\n', merged_df.head())
        print(f'Merged DataFrame shape: {merged_df.shape}')
        # Save merged DataFrame if no. of rows is greater than 800
        if merged_df.shape[0] > 800:
            print(f'Saving merged DataFrame for {contract_name} with shape {merged_df.shape}')
            save_path = os.path.join(save_dir, f'{contract_name}_ohlcv_oi.csv')
            merged_df.to_csv(save_path, index=False)
        else:
            print(f'Skipping save for {contract_name} due to insufficient rows: {merged_df.shape[0]}')

def save_df_to_csv(df, save_dir, filename):
    """
    Saves a DataFrame to CSV in save_dir, creating the directory if needed.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    df.to_csv(save_path, index=False)

def plot_oi_volume_price_vs_time(csv_path, save_dir, filename='oi_volume_price_plot.png'):
    """
    Reads a merged DataFrame from CSV, filters Date-Time, OI, Volume, and Close Price columns,
    and saves a static plot (OI & Volume as overlaid bars, Close Price as a line) to the given directory.
    """
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv(csv_path)
    date_col = 'Date-Time' if 'Date-Time' in df.columns else 'date'
    oi_col = 'OI' if 'OI' in df.columns else df.columns[-1]
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
    close_col = 'Close' if 'Close' in df.columns else 'close'

    df[date_col] = pd.to_datetime(df[date_col])

    fig, ax1 = plt.subplots(figsize=(18, 7))  # Increased width for more x-axis space
    ax1.plot(df[date_col], df[close_col], label='Close Price', color='green', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    width = 2  # days, for bar width
    ax2.bar(df[date_col], df[oi_col], label='Open Interest (OI)', color='blue', alpha=0.5, width=width)
    ax2.bar(df[date_col], df[volume_col], label='Volume', color='orange', alpha=0.5, width=width)
    ax2.set_ylabel('OI / Volume', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Set x-axis major ticks to every month
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title('OI & Volume (right, overlaid bars) and Close Price (left, line) vs Time')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    
# merge_oi_with_resampled('./Assignment1/OI/FEIcm_OI.xlsx','./Assignment1/Cleaned_DataM/resampled_1d', './Assignment1/Merged_FEI_OI_Data')
# give contract number to plot_oi_volume_price_vs_time
for i in range(1, 14):
    plot_oi_volume_price_vs_time(
        f'./Assignment1/Merged_FEI_OI_Data/FEIcm{i}_ohlcv_oi.csv',
        './Assignment1/Plots',
        f'FEIcm{i}_oi_volume_price_plot.png'
    )