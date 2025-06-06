import pandas as pd
import numpy as np

def generate_signals(df, rule_func, **rule_kwargs):
    """
    Given a DataFrame and a rule function, generate a 'Signal' column (1=Buy, 0=NoOp, -1=Sell).
    rule_func: function that takes df and **rule_kwargs, returns a Series of signals.
    """
    signals = rule_func(df, **rule_kwargs)
    df = df.copy()
    df['Signal'] = signals
    return df

def save_signals_to_csv(input_csv, output_csv, rule_func, **rule_kwargs):
    """
    Loads a CSV, applies the rule, saves the DataFrame with the 'Signal' column to output_csv.
    """
    df = pd.read_csv(input_csv, parse_dates=['Date-Time'])
    df_with_signals = generate_signals(df, rule_func, **rule_kwargs)
    df_with_signals.to_csv(output_csv, index=False)
    return output_csv

def rule_ma_crossover(df, short_window=10, long_window=50):
    """
    Example rule: Moving Average Crossover.
    Buy (1) when short MA crosses above long MA, Sell (-1) when short MA crosses below long MA, else 0.
    """
    short_ma = df['Close'].rolling(window=short_window, min_periods=1).mean()
    long_ma = df['Close'].rolling(window=long_window, min_periods=1).mean()
    signal = np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0))
    # Only signal on cross, not on every bar
    prev_signal = pd.Series(signal).shift(1, fill_value=0)
    cross_signal = np.where(signal != prev_signal, signal, 0)
    return cross_signal

def rule_bollinger_band(df, window=20, num_std=2):
    """
    Example rule: Bollinger Bands.
    Buy (1) if Close crosses above lower band, Sell (-1) if Close crosses below upper band, else 0.
    """
    ma = df['Close'].rolling(window=window, min_periods=1).mean()
    std = df['Close'].rolling(window=window, min_periods=1).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    prev_close = df['Close'].shift(1)
    buy = (prev_close < lower) & (df['Close'] > lower)
    sell = (prev_close > upper) & (df['Close'] < upper)
    signal = np.where(buy, 1, np.where(sell, -1, 0))
    return signal

def rule_with_month_filter(df, base_rule_func, low_volume_months=[1, 4, 7, 10], **kwargs):
    """
    Example rule: Filter signals based on the month.
    Sets signal to 0 for months in low_volume_months.
    """
    # Get base signals
    signals = base_rule_func(df, **kwargs)
    # Set signal to 0 for low volume months
    months = df['Date-Time'].dt.month if 'Date-Time' in df else pd.to_datetime(df['Date']).dt.month
    signals = np.where(months.isin(low_volume_months), 0, signals)
    return signals




# Example usage:
# save_signals_to_csv('Cleaned_Data/resampled_1d/FEIc1_ohlcv_1d.csv', 'output_signals.csv', rule_ma_crossover, short_window=10, long_window=50)
# save_signals_to_csv('Cleaned_Data/resampled_1d/FEIc1_ohlcv_1d.csv', 'output_signals.csv', rule_bollinger_band, window=20, num_std=2)


#Main function to run the example
# if __name__ == "__main__":
    
    #First Stategy: Check trend(through MA and volume) and ending date, if date is the first date of the new trading week, then check the
    # trend and close price difference between new week date and last date of the last week  and correspondingly buy, sell, or hold
    
    
    #Second Strategy: If month = low volume month + 1, then check the trend and correspondingly buy, sell, or hold
    
    
    #Third Strategy: