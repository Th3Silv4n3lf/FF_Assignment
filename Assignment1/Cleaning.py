import pandas as pd
import os

def missing_OHLC_check(df, filename):
    """
    Check for missing values in the OHLCV columns and print the results.
    If there are missing values, forward fill the OHLC data.
    Returns the processed DataFrame.
    """
    missing = df[['Date-Time', 'Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum()
    print("Missing values per column:")
    print(f'{filename} :  {missing}')
    if missing.sum() > 0:
        print(f'Warning: {filename} has missing values.')
        # Forward fill only OHLC data, keep Volume as it is
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()
    return df

def stats_check(df, filename):
    """
    Print statistics of the OHLCV data with normal (non-scientific) numbers.
    """
    print(f'Statistics for {filename}:')
    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(df.describe())

def garbage_value_check(df, filename):
    """
    Remove rows where any OHLC value is outside [80, 110].
    """
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    garbage_mask = ~((df[ohlc_cols] >= 80) & (df[ohlc_cols] <= 110)).all(axis=1)
    garbage_count = garbage_mask.sum()
    if garbage_count > 0:
        print(f'Garbage values found in {filename}: {garbage_count} rows will be dropped.')
        print(df.loc[garbage_mask, ['Date-Time'] + ohlc_cols])
        df = df.loc[~garbage_mask].copy()
    else:
        print(f'No garbage values in {filename}')
    return df

def missing_volume_check(df, filename):
    """
    Check for missing values and drop rows where Volume is missing.
    Returns the cleaned DataFrame.
    """
    if df['Volume'].isnull().any():
        print(f'Warning: {filename} has missing Volume values. These rows will be dropped.')
        df = df.dropna(subset=['Volume']).copy()
    else:
        print(f'No missing Volume values in {filename}')
    return df

def convert_tradebook_to_ohlcv(input_csv):
    """
    Convert a trade book CSV to OHLCV format, assuming 'Date-Time' is already in UTC (ends with Z).
    Ignores 'GMT Offset'. Only rows with missing timestamps are dropped.
    Returns the processed DataFrame.
    """
    df = pd.read_csv(input_csv)
    # Parse 'Date-Time' as UTC (ISO8601 with Z)
    df['Date-Time'] = pd.to_datetime(df['Date-Time'], utc=True)
    # Rename 'Last' to 'Close' for standard OHLCV naming
    if 'Last' in df.columns:
        df = df.rename(columns={'Last': 'Close'})
    # Reorder columns to standard OHLCV order
    ohlcv_cols = ['Date-Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[ohlcv_cols]
    # Check for missing values in timestamp column only
    missing = df['Date-Time'].isnull().sum()
    print("Missing timestamp values:")
    print(missing)
    # Drop rows with missing timestamp only
    df = df.dropna(subset=['Date-Time'])
    return df

def save_final_ohlcv(df, output_csv):
    """
    Save the final processed DataFrame to the output path in ISO format for Date-Time.
    """
    df = df.copy()
    df['Date-Time'] = pd.to_datetime(df['Date-Time']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df.to_csv(output_csv, index=False)
    print(f'Final OHLCV data saved to {output_csv}')

# Main processing loop
base_path = 'c:/Users/ishan.ostwal/OneDrive - hertshtengroup.com/Documents/FF_DataAnalysis/Assignment1/FEI_DataM/'
output_path = 'c:/Users/ishan.ostwal/OneDrive - hertshtengroup.com/Documents/FF_DataAnalysis/Assignment1/Cleaned_DataM/'
output_end_path = '_ohlcv.csv'
input_end_path = '.csv'
for i in range(1, 17):
    filename = f'FEIcm{i}'
    input_filepath = f'{base_path}{filename}{input_end_path}'
    output_filepath = f'{output_path}{filename}{output_end_path}'
    if os.path.isfile(input_filepath):
        df = convert_tradebook_to_ohlcv(input_filepath)
        df = missing_OHLC_check(df, filename)
        df = garbage_value_check(df, filename)
        df = missing_volume_check(df, filename)
        stats_check(df, filename)
        save_final_ohlcv(df, output_filepath)
    else:
        print(f'Not found: {filename}')

