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

def plot_oi_volume_price_vs_time(csv_path):
    """
    Reads a merged DataFrame from CSV, filters Date-Time, OI, Volume, and Close Price columns,
    and plots OI & Volume on right y-axis, Close Price on left y-axis vs Time. Interactive plot.
    Close Price is a line, OI is a bar, Volume is a bar (stacked, but with different colors for clarity).
    """
    import plotly.graph_objs as go
    import plotly.offline as pyo
    import pandas as pd

    df = pd.read_csv(csv_path)
    date_col = 'Date-Time' if 'Date-Time' in df.columns else 'date'
    oi_col = 'OI' if 'OI' in df.columns else df.columns[-1]
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
    close_col = 'Close' if 'Close' in df.columns else 'close'

    df[date_col] = pd.to_datetime(df[date_col])

    trace_close = go.Scatter(x=df[date_col], y=df[close_col], name='Close Price', yaxis='y1', line=dict(color='green', width=2), mode='lines')
    # OI and Volume as overlaid bars, both starting from y=0, but with different colors and some transparency
    trace_oi = go.Bar(x=df[date_col], y=df[oi_col], name='Open Interest (OI)', yaxis='y2', marker_color='blue', opacity=0.5, offsetgroup=0, width=50000000)
    trace_volume = go.Bar(x=df[date_col], y=df[volume_col], name='Volume', yaxis='y2', marker_color='orange', opacity=0.8, offsetgroup=0, width=50000000)

    layout = go.Layout(
        title='OI & Volume (right, overlaid bars) and Close Price (left, line) vs Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Close Price', titlefont=dict(color='green'), tickfont=dict(color='green')),
        yaxis2=dict(title='OI / Volume', titlefont=dict(color='blue'), tickfont=dict(color='blue'),
                    overlaying='y', side='right', showgrid=False, rangemode='tozero'),
        barmode='overlay',
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode='x unified',
    )

    fig = go.Figure(data=[trace_close, trace_oi, trace_volume], layout=layout)
    pyo.plot(fig)
    
# merge_oi_with_resampled('./Assignment1/OI/FEIcm_OI.xlsx','./Assignment1/Cleaned_DataM/resampled_1d', './Assignment1/Merged_FEI_OI_Data')
plot_oi_volume_price_vs_time('./Assignment1/Merged_FEI_OI_Data/FEIcm1_ohlcv_oi.csv')