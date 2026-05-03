#!/usr/bin/env python3
"""
Ichimoku Backtest with PSAR enabled vs disabled.
Uses the proper IchimokuBacktester from your framework.
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner
from strategy.ichimoku_strategy import UnifiedIchimokuAnalyzer, IchimokuParameters

def load_data_with_indicators(db_path, timeframe, ichimoku_params, psar_enabled=False, psar_params=None):
    """Load data and compute all indicators (Ichimoku + optional PSAR)."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data WHERE timeframe = ? ORDER BY timestamp",
        conn, params=(timeframe,)
    )
    conn.close()
    
    if df.empty:
        return df
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate Ichimoku
    analyzer = UnifiedIchimokuAnalyzer()
    params = IchimokuParameters(
        tenkan_period=ichimoku_params.get('tenkan_period', 9),
        kijun_period=ichimoku_params.get('kijun_period', 26),
        senkou_b_period=ichimoku_params.get('senkou_b_period', 52),
        senkou_offset=ichimoku_params.get('senkou_offset', 26),
        chikou_offset=ichimoku_params.get('chikou_offset', 26)
    )
    df = analyzer.calculate_ichimoku_components(df, parameters=params)
    
    # Calculate signals
    from strategy.ichimoku_strategy import SignalConditions, SignalType
    
    # Define signal conditions (same as strategy_01)
    buy_conditions = [
        SignalType.PRICE_ABOVE_CLOUD,
        SignalType.TENKAN_ABOVE_KIJUN,
        SignalType.SPAN_A_ABOVE_SPAN_B,
        SignalType.CHIKOU_ABOVE_PRICE
    ]
    sell_conditions = [
        SignalType.TENKAN_BELOW_KIJUN
    ]
    
    signal_conditions = SignalConditions(
        buy_conditions=buy_conditions,
        sell_conditions=sell_conditions,
        buy_logic="AND",
        sell_logic="AND"
    )
    
    df = analyzer.detect_boolean_signals(df, parameters=params)
    
    # Calculate PSAR if enabled
    if psar_enabled and psar_params:
        from strategy.psar_indicator import compute_psar
        psar_result = compute_psar(
            df,
            step=psar_params.get('step', 0.02),
            max_step=psar_params.get('max_step', 0.2)
        )
        df['psar'] = psar_result['psar']
        df['psar_trend'] = psar_result['psar_trend']
        df['psar_uptrend'] = psar_result['psar_trend'] == 1
        df['psar_downtrend'] = psar_result['psar_trend'] == -1
        
        # Add PSAR conditions to buy/sell
        if psar_params.get('use_for_buy', False):
            buy_conditions.append(SignalType.PSAR_TREND_UP)
        if psar_params.get('use_for_sell', False):
            sell_conditions.append(SignalType.PSAR_TREND_DOWN)
    
    return df, signal_conditions

def run_backtest_with_conditions(df, signal_conditions, use_psar_for_buy=False, use_psar_for_sell=False):
    """Run backtest using the signal conditions."""
    equity = 100000
    position = None
    trades = []
    equity_curve = []
    
    # Get signal columns based on configuration
    buy_signal_cols = [
        'price_above_cloud',
        'tenkan_above_kijun',
        'SpanAaboveSpanB',
        'chikou_above_price'
    ]
    
    # Add PSAR if enabled
    if use_psar_for_buy and 'psar_uptrend' in df.columns:
        buy_signal_cols.append('psar_uptrend')
    
    sell_signal_cols = [
        'tenkan_below_kijun'
    ]
    
    if use_psar_for_sell and 'psar_downtrend' in df.columns:
        sell_signal_cols.append('psar_downtrend')
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Check buy conditions (AND logic)
        buy_signal = True
        for col in buy_signal_cols:
            if col in df.columns:
                if not row[col]:
                    buy_signal = False
                    break
        
        # Check sell conditions (AND logic)
        sell_signal = True
        for col in sell_signal_cols:
            if col in df.columns:
                if not row[col]:
                    sell_signal = False
                    break
        
        # Exit if in position and sell signal
        if position is not None and sell_signal:
            price = row['close']
            pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
            equity += position['size'] * pnl_pct / 100
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_time': position['entry_time'],
                'exit_time': row['timestamp'],
                'entry_price': position['entry_price'],
                'exit_price': price
            })
            position = None
        
        # Enter if not in position and buy signal
        if position is None and buy_signal:
            position = {
                'entry_price': row['close'],
                'entry_time': row['timestamp'],
                'size': equity
            }
        
        # Track equity
        cur_eq = equity
        if position is not None:
            price = row['close']
            cur_eq += (price - position['entry_price']) / position['entry_price'] * position['size']
        equity_curve.append(cur_eq)
    
    # Close open position
    if position is not None:
        price = df.iloc[-1]['close']
        pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
        equity += position['size'] * pnl_pct / 100
        trades.append({
            'pnl_pct': pnl_pct,
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'entry_price': position['entry_price'],
            'exit_price': price
        })
    
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0}
    
    trades_arr = np.array([t['pnl_pct'] for t in trades])
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
        'winning_trades': winning,
        'win_rate': winning / len(trades) * 100,
        'total_return_pct': total_return,
        'final_equity': equity,
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ICHIMOKU BACKTEST: WITH vs WITHOUT PSAR")
    print("="*60)
    
    # Load strategy config
    with open(ROOT / 'config' / 'strategies.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    strategy_key = 'strategy_01'
    if strategy_key not in config['strategies']:
        print(f"❌ Strategy {strategy_key} not found!")
        sys.exit(1)
    
    strategy_config = config['strategies'][strategy_key]
    ichimoku_params = strategy_config['ichimoku_parameters']
    psar_params = strategy_config.get('psar_parameters', {})
    
    db_path = str(ROOT / 'data' / 'trading_data_BTC.db')
    print(f"\nLoading data from: {db_path}")
    print(f"Strategy: {strategy_config['name']}")
    print(f"Ichimoku Params: TK={ichimoku_params['tenkan_period']}, KJ={ichimoku_params['kijun_period']}, SB={ichimoku_params['senkou_b_period']}")
    print(f"PSAR Params: enabled={psar_params.get('enabled', False)}")
    
    # Test 1: WITHOUT PSAR (original)
    print("\n" + "="*60)
    print("TEST 1: WITHOUT PSAR (Original)")
    print("="*60)
    
    df1, signal_conditions = load_data_with_indicators(
        db_path, '4h', ichimoku_params, 
        psar_enabled=False
    )
    
    if df1.empty:
        print("❌ No data loaded!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df1)} candles with Ichimoku indicators")
    
    results_no_psar = run_backtest_with_conditions(df1, signal_conditions, use_psar_for_buy=False, use_psar_for_sell=False)
    
    print(f"\nResults WITHOUT PSAR:")
    print(f"  Trades: {results_no_psar['total_trades']}")
    print(f"  Winning: {results_no_psar['winning_trades']}")
    print(f"  Win Rate: {results_no_psar['win_rate']:.2f}%")
    print(f"  Return: {results_no_psar['total_return_pct']:.2f}%")
    print(f"  Final Equity: ${results_no_psar['final_equity']:,.2f}")
    print(f"  Max DD: {results_no_psar['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe: {results_no_psar['sharpe_ratio']:.2f}")
    
    # Test 2: WITH PSAR enabled
    print("\n" + "="*60)
    print("TEST 2: WITH PSAR ENABLED")
    print("="*60)
    
    psar_params_enabled = psar_params.copy()
    psar_params_enabled['enabled'] = True
    psar_params_enabled['use_for_buy'] = True   # Use PSAR uptrend for buy
    psar_params_enabled['use_for_sell'] = True # Use PSAR downtrend for sell
    
    df2, signal_conditions2 = load_data_with_indicators(
        db_path, '4h', ichimoku_params,
        psar_enabled=True,
        psar_params=psar_params_enabled
    )
    
    print(f"✅ Loaded {len(df2)} candles with Ichimoku + PSAR indicators")
    
    results_with_psar = run_backtest_with_conditions(df2, signal_conditions2, 
                                                  use_psar_for_buy=psar_params_enabled.get('use_for_buy', False),
                                                  use_psar_for_sell=psar_params_enabled.get('use_for_sell', False))
    
    print(f"\nResults WITH PSAR:")
    print(f"  Trades: {results_with_psar['total_trades']}")
    print(f"  Winning: {results_with_psar['winning_trades']}")
    print(f"  Win Rate: {results_with_psar['win_rate']:.2f}%")
    print(f"  Return: {results_with_psar['total_return_pct']:.2f}%")
    print(f"  Final Equity: ${results_with_psar['final_equity']:,.2f}")
    print(f"  Max DD: {results_with_psar['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe: {results_with_psar['sharpe_ratio']:.2f}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON: WITH vs WITHOUT PSAR")
    print("="*60)
    print(f"{'Metric':<25} {'Without PSAR':>15} {'With PSAR':>15} {'Diff':>15}")
    print("-"*70)
    
    metrics = ['total_trades', 'win_rate', 'total_return_pct', 'max_drawdown_pct', 'sharpe_ratio']
    labels = {
        'total_trades': 'Trades',
        'win_rate': 'Win Rate %',
        'total_return_pct': 'Return %',
        'max_drawdown_pct': 'Max DD %',
        'sharpe_ratio': 'Sharpe'
    }
    
    for metric in metrics:
        val1 = results_no_psar[metric]
        val2 = results_with_psar[metric]
        diff = val2 - val1
        
        if metric == 'win_rate':
            print(f"{labels[metric]:<25} {val1:>14.2f}% {val2:>14.2f}% {diff:>+14.2f}%")
        elif metric == 'total_return_pct':
            print(f"{labels[metric]:<25} {val1:>14.2f}% {val2:>14.2f}% {diff:>+14.2f}%")
        elif metric == 'max_drawdown_pct':
            print(f"{labels[metric]:<25} {val1:>14.2f}% {val2:>14.2f}% {diff:>+14.2f}%")
        elif metric == 'sharpe_ratio':
            print(f"{labels[metric]:<25} {val1:>15.2f} {val2:>15.2f} {diff:>+15.2f}")
        else:
            print(f"{labels[metric]:<25} {val1:>15} {val2:>15} {diff:>+15}")
    
    print("="*60)
    print("ICHIMOKU PSAR COMPARISON COMPLETE")
    print("="*60)
