#!/usr/bin/env python3
"""
Parameter tuning, Monte Carlo robustness test, and comparison with Ichimoku for BTC.
Uses updated data from trading_data_BTC.db (through 2026-05-03).
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import itertools
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.hma_rsi_lr_strategy import HmaRsiLrCryptoStrategy, create_hma_rsi_lr_config
from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner
import yaml

def load_data_from_sql(db_path, timeframe):
    """Load OHLCV data from SQL."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE timeframe = ?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(timeframe,))
    conn.close()
    if len(df) == 0:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def run_hma_rsi_lr_backtest(df_4h, df_1d, config):
    """Run backtest for HMA RSI LR strategy."""
    strategy = HmaRsiLrCryptoStrategy(config)
    df = strategy.calculate_indicators(df_4h, df_1d)
    entry_signals = strategy.generate_entry_signals(df)
    
    equity = 100000
    position = None
    trades = []
    equity_curve = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['timestamp']
        current_price = row['close']
        
        if position is not None:
            exit_reason = strategy.generate_exit_signals(df.iloc[:i+1], position)
            if exit_reason is not None:
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price * 100
                pnl_amount = position['size'] * pnl_pct / 100
                equity += pnl_amount
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'equity': equity
                })
                position = None
        
        if position is None and entry_signals.iloc[i] == 1:
            position = {
                'entry_price': current_price,
                'entry_time': current_time,
                'size': equity
            }
        
        cur_eq = equity
        if position is not None:
            unrealized = (current_price - position['entry_price']) / position['entry_price'] * position['size']
            cur_eq = equity + unrealized
        equity_curve.append({'timestamp': current_time, 'equity': cur_eq})
    
    if position is not None:
        last_price = df.iloc[-1]['close']
        last_time = df.iloc[-1]['timestamp']
        pnl_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
        pnl_amount = position['size'] * pnl_pct / 100
        equity += pnl_amount
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': last_time,
            'entry_price': position['entry_price'],
            'exit_price': last_price,
            'pnl_pct': pnl_pct,
            'equity': equity
        })
    
    # Calculate metrics
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0}
    
    df_trades = pd.DataFrame(trades)
    winning = len(df_trades[df_trades['pnl_pct'] > 0])
    total_return_pct = (equity - 100000) / 100000 * 100
    
    df_eq = pd.DataFrame(equity_curve)
    df_eq['peak'] = df_eq['equity'].cummax()
    df_eq['drawdown'] = (df_eq['equity'] - df_eq['peak']) / df_eq['peak'] * 100
    max_dd = abs(df_eq['drawdown'].min())
    
    returns = df_eq['equity'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(8760) if len(returns) > 1 and returns.std() > 0 else 0
    
    return {
        'total_trades': len(trades),
        'winning_trades': winning,
        'win_rate': winning / len(trades) * 100,
        'total_return_pct': total_return_pct,
        'final_equity': equity,
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe
    }

def parameter_tuning(df_4h, df_1d):
    """Grid search over HMA RSI LR parameters."""
    print(f"\n{'='*60}")
    print("PARAMETER TUNING - HMA RSI LR")
    print(f"{'='*60}")
    
    # Define parameter ranges (as per video Monte Carlo)
    fast_hma_range = range(14, 19)  # 14-18
    slow_hma_range = range(60, 71)   # 60-70
    rsi_threshold_range = range(48, 56)  # 48-55
    lr_period_range = range(40, 61)   # 40-60
    
    best_result = None
    best_return = -999999
    results = []
    
    # Limit combinations for speed (sample)
    combos = list(itertools.product(fast_hma_range, slow_hma_range, rsi_threshold_range, lr_period_range))
    # Sample 50 random combinations for speed
    random.seed(42)
    sampled_combos = random.sample(combos, min(50, len(combos)))
    
    print(f"Testing {len(sampled_combos)} parameter combinations...")
    
    for i, (fast, slow, rsi_th, lr) in enumerate(sampled_combos):
        if fast >= slow:
            continue
        config = create_hma_rsi_lr_config('BTC')
        config.strategy_parameters = {
            'fast_hma_period': fast,
            'slow_hma_period': slow,
            'rsi_period': 14,
            'rsi_threshold': rsi_th,
            'lr_period': lr
        }
        result = run_hma_rsi_lr_backtest(df_4h, df_1d, config)
        result['params'] = f"HMA({fast},{slow}) RSI>{rsi_th} LR({lr})"
        results.append(result)
        
        if result['total_return_pct'] > best_return:
            best_return = result['total_return_pct']
            best_result = result.copy()
        
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(sampled_combos)}")
    
    # Sort by return
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    print(f"\nTop 5 Parameter Combinations:")
    print(results_df.head(5).to_string(index=False))
    
    print(f"\nBest Parameters:")
    for key, val in best_result.items():
        if key != 'params':
            print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")
    
    return best_result

def monte_carlo_simulation(df_4h, df_1d, base_config, n_runs=500):
    """Run Monte Carlo simulation with jittered parameters."""
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION ({n_runs} runs)")
    print(f"{'='*60}")
    
    results = []
    base_params = base_config.strategy_parameters
    
    for i in range(n_runs):
        # Jitter parameters (as per video)
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
        
        result = run_hma_rsi_lr_backtest(df_4h, df_1d, config)
        result['params'] = f"HMA({fast},{slow}) RSI>{rsi_th} LR({lr})"
        results.append(result)
        
        if (i+1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_runs}")
    
    # Analyze distribution
    returns = [r['total_return_pct'] for r in results if r['total_trades'] > 0]
    sharpes = [r['sharpe_ratio'] for r in results if r['total_trades'] > 0]
    drawdowns = [r['max_drawdown_pct'] for r in results if r['total_trades'] > 0]
    win_rates = [r['win_rate'] for r in results if r['total_trades'] > 0]
    
    print(f"\nMonte Carlo Results Distribution:")
    print(f"  Profitable runs: {len([r for r in returns if r > 0])}/{len(returns)} ({len([r for r in returns if r > 0])/len(returns)*100:.1f}%)")
    print(f"\nReturn %:")
    print(f"  Min: {min(returns):.2f}")
    print(f"  5th percentile: {np.percentile(returns, 5):.2f}")
    print(f"  Median: {np.percentile(returns, 50):.2f}")
    print(f"  95th percentile: {np.percentile(returns, 95):.2f}")
    print(f"  Max: {max(returns):.2f}")
    print(f"  Mean: {np.mean(returns):.2f}")
    print(f"\nSharpe Ratio:")
    print(f"  Median: {np.percentile(sharpes, 50):.2f}")
    print(f"  25% > 0.5: {len([s for s in sharpes if s > 0.5])/len(sharpes)*100:.1f}%")
    print(f"\nMax Drawdown %:")
    print(f"  Median: {np.percentile(drawdowns, 50):.2f}")
    print(f"  Worst: {max(drawdowns):.2f}")
    print(f"\nWin Rate %:")
    print(f"  Median: {np.percentile(win_rates, 50):.2f}")
    
    # Robustness confidence
    profitable_pct = len([r for r in returns if r > 0]) / len(returns) * 100
    sharpe_good_pct = len([s for s in sharpes if s > 0.5]) / len(sharpes) * 100
    print(f"\nRobustness Confidence: {profitable_pct:.0f}% profitable, {sharpe_good_pct:.0f}% Sharpe >0.5")
    
    return results

def run_ichimoku_backtest():
    """Run Ichimoku backtest for BTC using existing config."""
    print(f"\n{'='*60}")
    print("ICHIMOKU CRYPTO BACKTEST")
    print(f"{'='*60}")
    
    try:
        with open(ROOT / 'config' / 'strategies.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        strategy_key = 'strategy_01'  # Bitcoin Crypto Strategy
        if strategy_key not in config['strategies']:
            print(f"Strategy {strategy_key} not found!")
            return None
        
        strategy_config = config['strategies'][strategy_key]
        
        runner = StrategyBacktestRunner()
        symbol = strategy_config['symbols'][0].replace('/', '')  # BTCUSDT -> BTC
        timeframe = strategy_config['timeframes'][0]
        
        print(f"Loading data for {strategy_config['name']}...")
        data = runner.fetch_sql_data_with_signals(
            symbol_short=symbol,
            timeframe=timeframe,
            ichimoku_params=strategy_config['ichimoku_parameters']
        )
        
        if data.empty:
            print("No data returned! Trying alternative approach...")
            # Load directly from SQL
            db_path = str(ROOT / 'data' / f'trading_data_BTC.db')
            df = load_data_from_sql(db_path, '4h')
            if df.empty:
                print("No 4h data found!")
                return None
            # Compute Ichimoku indicators
            from strategy.ichimoku_strategy import UnifiedIchimokuAnalyzer
            analyzer = UnifiedIchimokuAnalyzer()
            df = analyzer.calculate_ichimoku(df, **strategy_config['ichimoku_parameters'])
            data = df
        
        print(f"Data loaded: {len(data)} rows")
        
        # Run backtest
        result = runner.run_strategy_backtest(strategy_config, data)
        
        if result is None:
            print("Backtest returned None!")
            return None
        
        metrics = {
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'total_return_pct': result.total_return_pct,
            'max_drawdown_pct': result.max_drawdown_pct,
            'sharpe_ratio': result.sharpe_ratio
        }
        
        print(f"\nIchimoku Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        
        return metrics
        
    except Exception as e:
        print(f"Error running Ichimoku backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("COMPREHENSIVE BACKTEST: HMA RSI LR vs ICHIMOKU")
    print(f"{'='*60}")
    
    # Load data
    db_path = str(ROOT / 'data' / 'trading_data_BTC.db')
    print(f"\nLoading data from: {db_path}")
    
    df_4h = load_data_from_sql(db_path, '4h')
    df_1d = load_data_from_sql(db_path, '1d')
    
    if len(df_4h) == 0 or len(df_1d) == 0:
        print("❌ Insufficient data!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df_4h)} 4h candles and {len(df_1d)} 1d candles")
    
    # 1. Baseline HMA RSI LR
    print(f"\n{'='*60}")
    print("BASELINE HMA RSI LR (Original Parameters)")
    print(f"{'='*60}")
    config = create_hma_rsi_lr_config('BTC')
    baseline = run_hma_rsi_lr_backtest(df_4h, df_1d, config)
    print(f"\nBaseline Results:")
    for k, v in baseline.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # 2. Parameter Tuning
    best_params = parameter_tuning(df_4h, df_1d)
    
    # 3. Monte Carlo Simulation
    monte_carlo_simulation(df_4h, df_1d, config, n_runs=300)  # Reduced for speed
    
    # 4. Ichimoku Comparison
    ichimoku_results = run_ichimoku_backtest()
    
    # 5. Final Comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"\nHMA RSI LR (Baseline):")
    for k, v in baseline.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    if ichimoku_results:
        print(f"\nIchimoku Strategy:")
        for k, v in ichimoku_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
