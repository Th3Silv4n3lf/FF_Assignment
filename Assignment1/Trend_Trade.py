# Re-import necessary libraries after kernel reset
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import os

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
    minus_di = 100 * (abs(minus_dm.rolling(window=period).sum()) / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

def calculate_hurst(series):
    series = series.dropna()
    if len(series) < 100:
        return 0.5
    H, _, _ = compute_Hc(series, kind='price', simplified=True)
    return H

def calculate_autocorrelation(series, lag=1):
    return series.autocorr(lag=lag)

def apply_exit_logic_tick(df, price_col='Close',
                          target_ticks=6, stop_ticks=3,
                          tick_size=0.005,
                          use_signal_exit=True):
    """
    Apply exit logic using tick-based stop-loss and take-profit.
    Optionally closes based on signal reversal.
    """
    df = df.copy()
    df['position'] = 0
    df['exit_price'] = np.nan

    position = 0
    entry_price = 0

    for i in range(1, len(df)):
        signal = df.loc[df.index[i], 'trend_signal']
        price = df.loc[df.index[i], price_col]

        if position == 0:
            if signal == 1 or signal == -1:
                position = signal
                entry_price = price
                df.loc[df.index[i], 'position'] = position
        else:
            tick_move = (price - entry_price) / tick_size if position == 1 else (entry_price - price) / tick_size

            exit_signal = (
                tick_move >= target_ticks or
                tick_move >= stop_ticks or
                (use_signal_exit and signal != position)
            )

            if exit_signal:
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'exit_price'] = price
                position = 0
                entry_price = 0
            else:
                df.loc[df.index[i], 'position'] = position

    return df

def save_signal_dataframe(df, output_path='trend_signal_output.csv'):
    """
    Saves the signal-appended DataFrame to a CSV file.
    If output_path includes directories that don't exist, they are created.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    df.to_csv(output_path, index=True)
    print(f"Saved DataFrame with signals to {output_path}")

def trend_following_signals_strict(df, price_col='Close', short_window=14, long_window=42,
                                   adx_threshold=25, rsi_thresh=(45, 55),
                                   autocorr_lag=1, autocorr_thresh=0.3, hurst_thresh=0.6):
    """
    Enhanced trend-following strategy with stricter filters to reduce noise and increase trade quality.
    Generates a 'trend_signal' column: 1 for long, -1 for short, 0 for no position.
    """
    signals = pd.Series(0, index=df.index)

    short_ma = df[price_col].rolling(window=short_window).mean()
    long_ma = df[price_col].rolling(window=long_window).mean()
    adx = calculate_adx(df['High'], df['Low'], df[price_col])
    rsi = calculate_rsi(df[price_col])
    autocorr = calculate_autocorrelation(df[price_col], lag=autocorr_lag)
    hurst_exp = calculate_hurst(df[price_col])

    position = 0

    for i in range(max(short_window, long_window), len(df)):
        # Apply all strict conditions before generating signal
        adx_ok = adx[i] >= adx_threshold
        rsi_ok = rsi_thresh[0] <= rsi[i] <= rsi_thresh[1]
        autocorr_ok = autocorr > autocorr_thresh
        hurst_ok = hurst_exp > hurst_thresh
        ma_crossover_long = short_ma[i] > long_ma[i]
        ma_crossover_short = short_ma[i] < long_ma[i]

        if adx_ok and rsi_ok and autocorr_ok and hurst_ok:
            if ma_crossover_long and position <= 0:
                signals[i] = 1
                position = 1
            elif ma_crossover_short and position >= 0:
                signals[i] = -1
                position = -1
            else:
                signals[i] = position
        else:
            signals[i] = 0
            position = 0

    df['trend_signal'] = signals
    return df

def backtest_strategy(df, price_col='Close', tick_size=0.005, tick_value=12.5):
    trades = []
    position = 0
    entry_price = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        if position == 0 and row['position'] != 0:
            position = row['position']
            entry_price = row[price_col]
            entry_time = row['Date-Time'] if 'Date-Time' in row else df.index[i]
        elif position != 0 and row['position'] == 0 and not pd.isna(row['exit_price']):
            exit_price = row['exit_price']
            exit_time = row['Date-Time'] if 'Date-Time' in row else df.index[i]
            tick_move = (exit_price - entry_price) / tick_size if position == 1 else (entry_price - exit_price) / tick_size
            pnl = tick_move * tick_value
            # Determine reason for squaring off
            if abs(tick_move) >= 6:
                exit_reason = 'target'
            elif abs(tick_move) >= 3:
                exit_reason = 'stop_loss'
            else:
                exit_reason = 'signal_reversal'
            trades.append({
                'entry_index': df.index[i-1],
                'exit_index': df.index[i],
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'Long' if position == 1 else 'Short',
                'tick_move': tick_move,
                'pnl': pnl,
                'exit_reason': exit_reason
            })
            position = 0
            entry_price = 0

    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean() if total_trades > 0 else 0
    expectancy = avg_pnl

    summary = {
        'Total Trades': total_trades,
        'Winning Trades': len(wins),
        'Losing Trades': len(losses),
        'Win Rate': win_rate,
        'Total PnL': total_pnl,
        'Average PnL per Trade': avg_pnl,
        'Expectancy': expectancy
    }

    return trades_df, summary

def save_backtest_results(trades_df, summary_dict, trade_path='backtest_trades.csv', summary_path='backtest_summary.txt'):
    trades_df.to_csv(trade_path, index=False)
    with open(summary_path, 'w') as f:
        for k, v in summary_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved trades to {trade_path} and summary to {summary_path}")

if __name__ == "__main__":
    df = pd.read_csv('./Assignment1/Cleaned_Data/resampled_1d/FEIc1_ohlcv_1d.csv')
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    df = trend_following_signals_strict(df)
    save_signal_dataframe(df, output_path='./Assignment1/Cleaned_Data/FEIc1_1d_signals.csv')
    df = apply_exit_logic_tick(df)
    trades_df, summary = backtest_strategy(df)
    save_backtest_results(trades_df, summary, trade_path='./Assignment1/Cleaned_Data/FEIc1_1d_tradepath.csv', summary_path='./Assignment1/Cleaned_Data/FEIc1_1d_summary.txt')
