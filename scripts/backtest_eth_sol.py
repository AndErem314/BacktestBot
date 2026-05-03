#!/usr/bin/env python3
"""
Backtest Ichimoku + PSAR strategy for ETH and SOL.
Reuses logic from BTC backtest (same parameters).
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.ichimoku_strategy import UnifiedIchimokuAnalyzer, IchimokuParameters
from backtesting.ichimoku_backtester import IchimokuBacktester

def load_data(db_path, timeframe='4h'):
    """Load OHLCV data from local SQL DB."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data WHERE timeframe = ? ORDER BY timestamp",
        conn, params=(timeframe,)
    )
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def run_backtest(symbol):
    """Run Ichimoku + PSAR backtest for a symbol."""
    print(f'\n{"="*60}')
    print(f'Backtesting Ichimoku + PSAR for {symbol}')
    print(f'{"="*60}')
    
    db_path = f'data/trading_data_{symbol}.db'
    if not os.path.exists(db_path):
        print(f'ERROR: Database not found: {db_path}')
        return None
    
    # Load data
    df = load_data(db_path, timeframe='4h')
    print(f'Loaded {len(df)} 4h candles for {symbol}')
    
    # Initialize analyzer with BTC-tested parameters
    analyzer = UnifiedIchimokuAnalyzer()
    params = IchimokuParameters(
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52,
        senkou_offset=26,
        chikou_offset=26
    )
    
    # Calculate Ichimoku components
    df = analyzer.calculate_ichimoku_components(df, parameters=params)
    
    # Initialize backtester
    backtester = IchimokuBacktester(
        initial_capital=10000,
        commission=0.001,
        use_psar=True,
        psar_step=0.02,
        psar_max=0.2
    )
    
    # Run backtest
    results = backtester.run(df)
    
    # Print metrics
    print(f'\nResults for {symbol}:')
    print(f'Total Return: {results.get("total_return_pct", 0):.2f}%')
    print(f'Win Rate: {results.get("win_rate", 0):.2f}%')
    print(f'Max Drawdown: {results.get("max_drawdown_pct", 0):.2f}%')
    print(f'Sharpe Ratio: {results.get("sharpe_ratio", 0):.2f}')
    print(f'Total Trades: {results.get("total_trades", 0)}')
    
    return results

def main():
    symbols = ['ETH', 'SOL']
    all_results = {}
    
    for symbol in symbols:
        results = run_backtest(symbol)
        if results:
            all_results[symbol] = results
    
    # Save results for comparison PDF
    import json
    with open('scripts/eth_sol_backtest_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'\nResults saved to scripts/eth_sol_backtest_results.json')
    return all_results

if __name__ == '__main__':
    main()
