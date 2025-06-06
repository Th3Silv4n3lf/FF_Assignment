import pandas as pd
import os

def resample_ohlcv(df, rule):
    """
    Resample OHLCV DataFrame to the given rule (e.g., '1T', '15T', '1H', '1D').
    """
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    df = df.set_index('Date-Time')
    resampled = df.resample(rule).agg(ohlc_dict)
    resampled = resampled.dropna(subset=['Open'])
    resampled = resampled.reset_index()
    #drop any row with volume =0 or na
    resampled = resampled[resampled['Volume'] > 0].copy()
    # Format Date-Time as ISO string with Z for UTC
    resampled['Date-Time'] = resampled['Date-Time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return resampled

def get_rule_from_param(param):
    """
    Map user parameter to pandas resample rule.
    """
    param = param.lower()
    if param in ['1min', '1minute', '1m', '1-min']:
        return '1T'
    elif param in ['15min', '15minute', '15m', '15-min']:
        return '15T'
    elif param in ['1h', '1hour', '1-hr', '1hr']:
        return '1H'
    elif param in ['1d', '1day', '1-day', '1d']:
        return '1D'
    else:
        raise ValueError(f'Unknown resampling parameter: {param}')

def trim_years(df, start_year=2006, end_year=2009):
    """
    Trim the DataFrame to only include rows where the year is between start_year and end_year (inclusive).
    Assumes 'Date-Time' is a datetime or string in ISO format.
    """
    #first check if date-time is already in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['Date-Time']):
        # Convert to datetime if not already
        df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    # Filter rows based on year
    mask = (df['Date-Time'].dt.year >= start_year) & (df['Date-Time'].dt.year <= end_year)
    return df.loc[mask].copy()

def batch_resample_ohlcv(resample_param):
    """
    Resample all FEIcm*_ohlcv.csv files in Cleaned_Data to the given interval.
    """
    cleaned_dir = 'c:/Users/ishan.ostwal/OneDrive - hertshtengroup.com/Documents/FF_DataAnalysis/Assignment1/Cleaned_DataM/'
    out_dir = os.path.join(cleaned_dir, f'resampled_{resample_param}')
    os.makedirs(out_dir, exist_ok=True)
    rule = get_rule_from_param(resample_param)
    for i in range(1, 17):
        fname = f'FEIcm{i}_ohlcv.csv'
        in_path = os.path.join(cleaned_dir, fname)
        if os.path.isfile(in_path):
            df = pd.read_csv(in_path)
            df = trim_years(df, 2006, 2009)
            resampled = resample_ohlcv(df, rule)
            out_path = os.path.join(out_dir, f'FEIcm{i}_ohlcv_{resample_param}.csv')
            resampled.to_csv(out_path, index=False)
            print(f'Resampled {fname} to {resample_param} (2006-2009) and saved as {out_path}')
        else:
            print(f'File not found: {in_path}')

if __name__ == "__main__":
    # Example usage:
    # batch_resample_ohlcv('1min')
    # batch_resample_ohlcv('15min')
    # batch_resample_ohlcv('1h')
    # batch_resample_ohlcv('1d')

    batch_resample_ohlcv('1min')
    batch_resample_ohlcv('15min')
    batch_resample_ohlcv('1h')
    batch_resample_ohlcv('1d')
    # The above code will resample the OHLCV data for each of the specified intervals and save them in a new directory.