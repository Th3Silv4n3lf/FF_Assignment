import os
import pandas as pd

def combine_sofr_csvs_from_dir(directory):
    """
    Combines all CSV files in the given directory into a single DataFrame with closing prices.
    Each file is assumed to have a 'Date-Time' and 'Close' column.
    The contract name is inferred from the filename (without extension).
    """
    combined_df = pd.DataFrame()
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('d.csv')]
    for file_path in file_paths:
        contract_name = os.path.splitext(os.path.basename(file_path))[0].replace('_ohlcv_1d', '')
        df = pd.read_csv(file_path, parse_dates=['Date-Time'])
        df = df[['Date-Time', 'Close']].rename(columns={'Close': contract_name})
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='Date-Time', how='outer')
    combined_df.sort_values('Date-Time', inplace=True)
    combined_df.ffill(inplace=True)
    #backfill for intial missing data
    combined_df.bfill(inplace=True)
    return combined_df


def add_spread_and_butterfly_columns(df):
    """
    Adds single and double difference spreads and butterfly (fly) spreads to the combined SOFR DataFrame.
    Assumes columns are named by contract (e.g., 'SRAcm1', 'SRAcm2', ..., 'SRAcm16')
    Rounds all spread and butterfly columns to 5 decimal places for CSV compatibility.
    """
    contracts = [col for col in df.columns if col != 'Date-Time']
    contracts.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # Ensure correct order

    # Add spreads
    for i in range(len(contracts) - 1):
        spread_col = f'{contracts[i+1]}-{contracts[i]}_s'
        df[spread_col] = (df[contracts[i+1]] - df[contracts[i]]).round(5)

    # Add butterfly (fly) spreads
    for i in range(len(contracts) - 2):
        butterfly_col = f'{contracts[i]}_{contracts[i+1]}_{contracts[i+2]}_b'
        df[butterfly_col] = (((df[contracts[i]] + df[contracts[i+2]]) / 2 - df[contracts[i+1]]).round(5))
    return df


def save_spread_dataframe(df, output_file=None, directory=None):
    """
    Saves the modified DataFrame with spread and butterfly calculations to CSV.
    If directory is provided and output_file is None, saves as 'combined_spreads.csv' in that directory.
    If output_file is provided, saves to that path.
    """
    if output_file is None and directory is not None:
        output_file = os.path.join(directory, 'combined_spreads.csv')
    elif output_file is None:
        raise ValueError('Either output_file or directory must be provided')
    df.to_csv(output_file, index=False)


def save_spread_and_butterfly_only_csvs(df, directory=None):
    """
    Saves two additional CSVs: one with only spread columns, one with only butterfly columns (plus 'Date-Time').
    Output files are named 'spreads_only.csv' and 'butterfly_only.csv' in the given directory.
    """
    if directory is None:
        raise ValueError('directory must be provided')
    spread_cols = [col for col in df.columns if col.endswith('_s')]
    butterfly_cols = [col for col in df.columns if col.endswith('_b')]
    # Always include 'Date-Time' as the first column
    spread_df = df[['Date-Time'] + spread_cols]
    butterfly_df = df[['Date-Time'] + butterfly_cols]
    spread_df.to_csv(os.path.join(directory, 'spreads_only.csv'), index=False)
    butterfly_df.to_csv(os.path.join(directory, 'butterfly_only.csv'), index=False)


# Example usage:
df = combine_sofr_csvs_from_dir('./Assignment1/Cleaned_Data/resampled_1d/')
df = add_spread_and_butterfly_columns(df)
# save_spread_dataframe(df, directory='./Assignment1/Cleaned_Data/resampled_1d/')
save_spread_and_butterfly_only_csvs(df, directory='./Assignment1/Cleaned_Data/resampled_1d/')
