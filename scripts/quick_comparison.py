#!/usr/bin/env python3
"""
Quick comparison: HMA RSI LR vs Ichimoku for BTC using updated data.
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.hma_rsi_lr_strategy import HmaRsiLrCryptoStrategy, create_hma_rsi_lr_config
from backtesting.ichimoku_backtester import IchimokuBacktester
from strategy.ichimoku_strategy import UnifiedIchimokuAnalyzer, IchimokuParameters

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

def run_hma_rsi_lr(df_4h, df_1d):
    config = create_hma_rsi_lr_config('BTC')
    strategy = HmaRsiLrCryptoStrategy(config)
    df = strategy.calculate_indicators(df_4h, df_1d)
    entry_signals = strategy.generate_entry_signals(df)
    
    equity = 100000
    position = None
    trades = []
    equity_curve = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        
        if position is not None:
            exit_reason = strategy.generate_exit_signals(df.iloc[:i+1], position)
            if exit_reason:
                pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                equity += position['size'] * pnl_pct / 100
                trades.append({'pnl_pct': pnl_pct})
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
        trades.append({'pnl_pct': pnl_pct})
    
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0}
    
    df_t = pd.DataFrame(trades)
    winning = (df_t['pnl_pct'] > 0).sum()
    total_return = (equity - 100000) / 100000 * 100
    
    df_eq = pd.DataFrame({'equity': equity_curve})
    peak = df_eq['equity'].cummax()
    dd = (df_eq['equity'] - peak) / peak * 100
    max_dd = abs(dd.min())
    
    returns = df_eq['equity'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(8760) if len(returns) > 1 and returns.std() > 0 else 0
    
    return {
        'total_trades': len(trades),
        'win_rate': winning / len(trades) * 100,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe
    }

def run_ichimoku(df_4h):
    # Create Ichimoku parameters object
    params = IchimokuParameters(
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52,
        senkou_offset=26,
        chikou_offset=26
    )
    
    # Compute Ichimoku indicators
    analyzer = UnifiedIchimokuAnalyzer()
    df = analyzer.calculate_ichimoku_components(df_4h, parameters=params)
    
    # Initialize backtester (not used in this simplified version)
    # backtester = IchimokuBacktester(asset_class='crypto')
    
    # Entry conditions: Price above cloud, Tenkan > Kijun, SpanA > SpanB, Chikou > price
    df['bullish'] = (
        (df['close'] > df['cloud_top']) &  # Price above cloud
        (df['tenkan_sen'] > df['kijun_sen']) &  # Tenkan above Kijun
        (df['senkou_span_a'] > df['senkou_span_b']) &  # SpanA above SpanB
        (df['chikou_span'] > df['close'])  # Chikou above price (simplified)
    )
    
    # Exit: Tenkan < Kijun
    df['bearish'] = (df['tenkan_sen'] < df['kijun_sen'])
    
    equity = 100000
    position = None
    trades = []
    equity_curve = []
    
    for i in range(len(df)):
        price = df.iloc[i]['close']
        
        if position is not None:
            if df.iloc[i]['bearish']:
                pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                equity += position['size'] * pnl_pct / 100
                trades.append({'pnl_pct': pnl_pct})
                position = None
        
        if position is None and df.iloc[i]['bullish']:
            position = {'entry_price': price, 'size': equity}
        
        cur_eq = equity
        if position is not None:
            cur_eq += (price - position['entry_price']) / position['entry_price'] * position['size']
        equity_curve.append(cur_eq)
    
    if position is not None:
        last_price = df.iloc[-1]['close']
        pnl_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
        equity += position['size'] * pnl_pct / 100
        trades.append({'pnl_pct': pnl_pct})
    
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0}
    
    df_t = pd.DataFrame(trades)
    winning = (df_t['pnl_pct'] > 0).sum()
    total_return = (equity - 100000) / 100000 * 100
    
    df_eq = pd.DataFrame({'equity': equity_curve})
    peak = df_eq['equity'].cummax()
    dd = (df_eq['equity'] - peak) / peak * 100
    max_dd = abs(dd.min())
    
    returns = df_eq['equity'].pct_change().dropna()
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
    print("QUICK COMPARISON: HMA RSI LR vs Ichimoku (BTC)")
    print("="*60)
    
    db_path = str(ROOT / 'data' / 'trading_data_BTC.db')
    print(f"\nLoading data from: {db_path}")
    
    df_4h = load_data(db_path, '4h')
    df_1d = load_data(db_path, '1d')
    
    if df_4h.empty or df_1d.empty:
        print("❌ Insufficient data!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df_4h)} 4h candles, {len(df_1d)} 1d candles")
    
    # HMA RSI LR
    print("\n" + "="*60)
    print("HMA RSI LR Strategy (Original Parameters)")
    print("="*60)
    hma_results = run_hma_rsi_lr(df_4h, df_1d)
    for k, v in hma_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # Ichimoku
    print("\n" + "="*60)
    print("Ichimoku Strategy (Default Parameters)")
    print("="*60)
    ichimoku_results = run_ichimoku(df_4h)
    for k, v in ichimoku_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'HMA RSI LR':>15} {'Ichimoku':>15}")
    print("-"*50)
    for metric in ['total_trades', 'win_rate', 'total_return_pct', 'max_drawdown_pct', 'sharpe_ratio']:
        hma_val = hma_results.get(metric, 0)
        ich_val = ichimoku_results.get(metric, 0)
        if metric == 'win_rate':
            print(f"{metric:<20} {hma_val:>14.2f}% {ich_val:>14.2f}%")
        elif metric == 'total_return_pct':
            print(f"{metric:<20} {hma_val:>14.2f}% {ich_val:>14.2f}%")
        elif metric == 'max_drawdown_pct':
            print(f"{metric:<20} {hma_val:>14.2f}% {ich_val:>14.2f}%")
        elif metric == 'sharpe_ratio':
            print(f"{metric:<20} {hma_val:>14.2f} {ich_val:>14.2f}")
        else:
            print(f"{metric:<20} {hma_val:>15} {ich_val:>15}")
    
    print("="*60)
    print("QUICK COMPARISON COMPLETE")
    print("="*60)
