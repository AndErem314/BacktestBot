#!/usr/bin/env python3
"""
Lightweight: Baseline + Quick Monte Carlo (10 runs) for HMA RSI LR.
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.hma_rsi_lr_strategy import HmaRsiLrCryptoStrategy, create_hma_rsi_lr_config

def load_data(db_path, timeframe):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data WHERE timeframe = ? ORDER BY timestamp",
        conn, params=(timeframe,)
    )
    conn.close()
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def backtest(strategy, df_4h, df_1d):
    df = strategy.calculate_indicators(df_4h, df_1d)
    entry_signals = strategy.generate_entry_signals(df)
    
    equity = 100000
    position = None
    trades = []
    equity_curve = []
    
    for i in range(len(df)):
        price = df.iloc[i]['close']
        
        if position is not None:
            exit_reason = strategy.generate_exit_signals(df.iloc[:i+1], position)
            if exit_reason:
                pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                equity += position['size'] * pnl_pct / 100
                trades.append(pnl_pct)
                position = None
        
        if position is None and entry_signals.iloc[i] == 1:
            position = {'entry_price': price, 'size': equity}
        
        cur_eq = equity
        if position is not None:
            cur_eq += (price - position['entry_price']) / position['entry_price'] * position['size']
        equity_curve.append(cur_eq)
    
    if position is not None:
        last_price = df.iloc[-1]['close']
        pnl_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
        equity += position['size'] * pnl_pct / 100
        trades.append(pnl_pct)
    
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0}
    
    trades_arr = np.array(trades)
    winning = (trades_arr > 0).sum()
    total_return = (equity - 100000) / 100000 * 100
    
    eq_curve = pd.Series(equity_curve)
    peak = eq_curve.cummax()
    dd = (eq_curve - peak) / peak * 100
    max_dd = abs(dd.min())
    
    returns = eq_curve.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(8760) if len(returns) > 1 and returns.std() > 0 else 0
    
    return {
        'total_trades': len(trades),
        'win_rate': winning / len(trades) * 100,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("LIGHTWEIGHT: BASELINE + QUICK MONTE CARLO (10 RUNS)")
    print("="*60)
    
    db_path = str(ROOT / 'data' / 'trading_data_BTC.db')
    df_4h = load_data(db_path, '4h')
    df_1d = load_data(db_path, '1d')
    
    if df_4h.empty or df_1d.empty:
        print("❌ Insufficient data!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df_4h)} 4h, {len(df_1d)} 1d candles")
    
    # Baseline
    print("\n" + "="*60)
    print("BASELINE (Original Parameters)")
    print("="*60)
    config = create_hma_rsi_lr_config('BTC')
    strategy = HmaRsiLrCryptoStrategy(config)
    baseline = backtest(strategy, df_4h, df_1d)
    print(f"  Trades: {baseline['total_trades']}")
    print(f"  Win Rate: {baseline['win_rate']:.2f}%")
    print(f"  Return: {baseline['total_return_pct']:.2f}%")
    print(f"  Max DD: {baseline['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe: {baseline['sharpe_ratio']:.2f}")
    
    # Quick Monte Carlo (10 runs)
    print("\n" + "="*60)
    print("QUICK MONTE CARLO (10 RUNS)")
    print("="*60)
    
    random.seed(42)
    returns = []
    sharpes = []
    drawdowns = []
    
    for i in range(10):
        fast = random.randint(14, 18)
        slow = random.randint(60, 70)
        rsi_th = random.randint(48, 55)
        lr = random.randint(40, 60)
        
        if fast >= slow:
            slow = fast + random.randint(1, 10)
        
        config = create_hma_rsi_lr_config('BTC')
        config.strategy_parameters = {
            'fast_hma_period': fast,
            'slow_hma_period': slow,
            'rsi_period': 14,
            'rsi_threshold': rsi_th,
            'lr_period': lr
        }
        
        strategy = HmaRsiLrCryptoStrategy(config)
        result = backtest(strategy, df_4h, df_1d)
        
        if result['total_trades'] > 0:
            returns.append(result['total_return_pct'])
            sharpes.append(result['sharpe_ratio'])
            drawdowns.append(result['max_drawdown_pct'])
            print(f"  Run {i+1}: Return={result['total_return_pct']:.1f}%, Trades={result['total_trades']}, Sharpe={result['sharpe_ratio']:.2f}")
    
    if returns:
        print(f"\nMonte Carlo Results (10 runs):")
        print(f"  Profitable: {len([r for r in returns if r > 0])}/10")
        print(f"  Median Return: {np.percentile(returns, 50):.2f}%")
        print(f"  Median Sharpe: {np.percentile(sharpes, 50):.2f}")
        print(f"  Median Max DD: {np.percentile(drawdowns, 50):.2f}%")
    
    print("\n" + "="*60)
    print("LIGHTWEIGHT ANALYSIS COMPLETE")
    print("="*60)
