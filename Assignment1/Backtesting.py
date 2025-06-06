import pandas as pd

def backtest_spread_signals(csv_path, target=.04, stop_loss=.02, trade_log_path=None):
    """
    Backtest all spread signal columns in the given CSV.
    For each signal column, calculate win/lose ratio and cumulative P&L using a risk-reward system (target/stop_loss in spread units).
    Returns a summary DataFrame with results for each spread.
    Also outputs a trade log CSV if trade_log_path is provided.
    """
    df = pd.read_csv(csv_path)
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    results = []
    trade_log = []
    for col in df.columns:
        if col.endswith('_signal'):
            spread_col = col[:-7]  # Remove '_signal'
            if spread_col not in df.columns:
                continue
            signals = df[col]
            spread = df[spread_col]
            entry_idx = None
            entry_price = None
            direction = 0
            for i in range(1, len(df)):
                if signals[i-1] == 0 and signals[i] != 0:
                    # Entry
                    entry_idx = i
                    entry_price = spread[i]
                    direction = signals[i]
                    entry_time = df['Date-Time'][i]
                elif signals[i-1] != 0 and signals[i] == 0 and entry_idx is not None:
                    # Exit (forced by signal)
                    exit_price = spread[i]
                    profit = direction * (exit_price - entry_price)
                    trade_log.append({
                        'spread': spread_col,
                        'entry_time': entry_time,
                        'exit_time': df['Date-Time'][i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_reason': 'signal'
                    })
                    entry_idx = None
                    entry_price = None
                    direction = 0
                elif entry_idx is not None:
                    # Check for target/stop
                    move = direction * (spread[i] - entry_price)
                    if move >= target:
                        trade_log.append({
                            'spread': spread_col,
                            'entry_time': entry_time,
                            'exit_time': df['Date-Time'][i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': spread[i],
                            'profit': target,
                            'exit_reason': 'target'
                        })
                        entry_idx = None
                        entry_price = None
                        direction = 0
                    elif move <= -stop_loss:
                        trade_log.append({
                            'spread': spread_col,
                            'entry_time': entry_time,
                            'exit_time': df['Date-Time'][i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': spread[i],
                            'profit': -stop_loss,
                            'exit_reason': 'stop_loss'
                        })
                        entry_idx = None
                        entry_price = None
                        direction = 0
            # Summary stats
            trade_profits = [t['profit'] for t in trade_log if t['spread'] == spread_col]
            wins = sum(1 for t in trade_profits if t > 0)
            losses = sum(1 for t in trade_profits if t < 0)
            win_ratio = wins / (wins + losses) if (wins + losses) > 0 else 0
            cum_pnl = sum(trade_profits)
            results.append({
                'spread': spread_col,
                'n_trades': len(trade_profits),
                'wins': wins,
                'losses': losses,
                'win_ratio': win_ratio,
                'cum_pnl': cum_pnl
            })
    if trade_log_path:
        pd.DataFrame(trade_log).to_csv(trade_log_path, index=False)
    return pd.DataFrame(results)

# Example usage:
if __name__ == "__main__":
    summary = backtest_spread_signals('spread_with_signals_combined.csv', target=.035, stop_loss=.015, trade_log_path='spread_trade_log.csv')
    print(summary)
