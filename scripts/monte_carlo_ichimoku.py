#!/usr/bin/env python3
"""
Monte Carlo Robustness Test for Ichimoku Strategy (BTC).
Jitters Ichimoku parameters and analyzes distribution of results.
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

def run_ichimoku_backtest(df_4h, params):
    """Run Ichimoku backtest with given parameters."""
    # Calculate Ichimoku indicators
    analyzer = UnifiedIchimokuAnalyzer()
    df = analyzer.calculate_ichimoku_components(df_4h, parameters=params)
    
    # Entry conditions (same as before)
    df['bullish'] = (
        (df['close'] > df['cloud_top']) &  # Price above cloud
        (df['tenkan_sen'] > df['kijun_sen']) &  # Tenkan above Kijun
        (df['senkou_span_a'] > df['senkou_span_b']) &  # SpanA above SpanB
        (df['chikou_span'] > df['close'])  # Chikou above price
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
                trades.append(pnl_pct)
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
        'sharpe_ratio': sharpe,
        'params': f"TK:{params.tenkan_period} KJ:{params.kijun_period} SB:{params.senkou_b_period}"
    }

def monte_carlo_ichimoku(df_4h, n_runs=200):
    """Run Monte Carlo simulation for Ichimoku."""
    print(f"\n{'='*60}")
    print(f"ICHIMOKU MONTE CARLO SIMULATION ({n_runs} runs)")
    print(f"{'='*60}")
    
    random.seed(42)
    results = []
    
    for i in range(n_runs):
        # Jitter parameters (around defaults: TK=9, KJ=26, SB=52)
        tenkan = random.randint(7, 11)
        kijun = random.randint(22, 30)
        senkou_b = random.randint(48, 56)
        senkou_offset = 26  # Usually fixed
        chikou_offset = 26  # Usually fixed
        
        params = IchimokuParameters(
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_b_period=senkou_b,
            senkou_offset=senkou_offset,
            chikou_offset=chikou_offset
        )
        
        result = run_ichimoku_backtest(df_4h, params)
        result['params'] = f"TK:{tenkan} KJ:{kijun} SB:{senkou_b}"
        results.append(result)
        
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_runs}")
    
    # Filter valid results
    valid_results = [r for r in results if r['total_trades'] > 0]
    
    if not valid_results:
        print("❌ No valid runs!")
        return
    
    returns = [r['total_return_pct'] for r in valid_results]
    sharpes = [r['sharpe_ratio'] for r in valid_results]
    drawdowns = [r['max_drawdown_pct'] for r in valid_results]
    win_rates = [r['win_rate'] for r in valid_results]
    trades_counts = [r['total_trades'] for r in valid_results]
    
    print(f"\nMonte Carlo Results Distribution ({len(valid_results)} valid runs):")
    print(f"\nReturn %:")
    print(f"  Min: {min(returns):.2f}")
    print(f"  5th percentile: {np.percentile(returns, 5):.2f}")
    print(f"  Median: {np.percentile(returns, 50):.2f}")
    print(f"  95th percentile: {np.percentile(returns, 95):.2f}")
    print(f"  Max: {max(returns):.2f}")
    print(f"  Mean: {np.mean(returns):.2f}")
    
    print(f"\nSharpe Ratio:")
    print(f"  Median: {np.percentile(sharpes, 50):.2f}")
    print(f"  % with Sharpe >0.5: {len([s for s in sharpes if s > 0.5])/len(sharpes)*100:.1f}%")
    print(f"  % with Sharpe >1.0: {len([s for s in sharpes if s > 1.0])/len(sharpes)*100:.1f}%")
    
    print(f"\nMax Drawdown %:")
    print(f"  Median: {np.percentile(drawdowns, 50):.2f}")
    print(f"  Worst: {max(drawdowns):.2f}")
    
    print(f"\nWin Rate %:")
    print(f"  Median: {np.percentile(win_rates, 50):.2f}")
    print(f"  Range: {min(win_rates):.1f}% - {max(win_rates):.1f}%")
    
    print(f"\nTrades:")
    print(f"  Median: {np.percentile(trades_counts, 50):.0f}")
    print(f"  Range: {min(trades_counts)} - {max(trades_counts)}")
    
    # Robustness confidence
    profitable_pct = len([r for r in returns if r > 0]) / len(returns) * 100
    sharpe_good_pct = len([s for s in sharpes if s > 0.5]) / len(sharpes) * 100
    print(f"\nRobustness Confidence:")
    print(f"  {profitable_pct:.0f}% profitable")
    print(f"  {sharpe_good_pct:.0f}% with Sharpe >0.5")
    
    # Best and worst
    sorted_by_return = sorted(valid_results, key=lambda x: x['total_return_pct'], reverse=True)
    print(f"\nBest Run: {sorted_by_return[0]['params']}")
    print(f"  Return: {sorted_by_return[0]['total_return_pct']:.2f}%")
    print(f"  Win Rate: {sorted_by_return[0]['win_rate']:.2f}%")
    print(f"  Sharpe: {sorted_by_return[0]['sharpe_ratio']:.2f}")
    print(f"  Trades: {sorted_by_return[0]['total_trades']}")
    
    print(f"\nWorst Run: {sorted_by_return[-1]['params']}")
    print(f"  Return: {sorted_by_return[-1]['total_return_pct']:.2f}%")
    print(f"  Win Rate: {sorted_by_return[-1]['win_rate']:.2f}%")
    print(f"  Sharpe: {sorted_by_return[-1]['sharpe_ratio']:.2f}")
    print(f"  Trades: {sorted_by_return[-1]['total_trades']}")
    
    return {
        'profitable_pct': profitable_pct,
        'sharpe_good_pct': sharpe_good_pct,
        'median_return': np.percentile(returns, 50),
        'median_sharpe': np.percentile(sharpes, 50),
        'best_return': sorted_by_return[0]['total_return_pct'],
        'worst_return': sorted_by_return[-1]['total_return_pct']
    }

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("ICHIMOKU MONTE CARLO ROBUSTNESS TEST")
    print(f"{'='*60}")
    
    db_path = str(ROOT / 'data' / 'trading_data_BTC.db')
    print(f"\nLoading data from: {db_path}")
    
    df_4h = load_data(db_path, '4h')
    
    if df_4h.empty:
        print("❌ Insufficient data!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df_4h)} 4h candles")
    
    # Run Monte Carlo
    results = monte_carlo_ichimoku(df_4h, n_runs=200)
    
    print(f"\n{'='*60}")
    print("ICHIMOKU MONTE CARLO COMPLETE")
    print(f"{'='*60}")
