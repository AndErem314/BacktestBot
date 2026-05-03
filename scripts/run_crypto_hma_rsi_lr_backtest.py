#!/usr/bin/env python3
"""
Backtest script for HMA RSI LR Crypto Strategy using local SQL data.
Uses BTC/USDT data from trading_data_BTC.db
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.hma_rsi_lr_strategy import HmaRsiLrCryptoStrategy, create_hma_rsi_lr_config


def load_data_from_sql(db_path: str, timeframe: str) -> pd.DataFrame:
    """
    Load OHLCV data from local SQL database.
    
    Args:
        db_path: Path to SQLite database
        timeframe: '4h' or '1d'
        
    Returns:
        DataFrame with OHLCV data
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE timeframe = ?
        ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, params=(timeframe,))
    conn.close()
    
    if len(df) == 0:
        print(f"❌ No data found for timeframe {timeframe}")
        return pd.DataFrame()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Loaded {len(df)} {timeframe} candles")
    print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    return df


def run_backtest(strategy, df_4h, df_1d, initial_equity=100000):
    """
    Run backtest using the strategy signals.
    
    Args:
        strategy: Strategy instance
        df_4h: 4h OHLCV data with indicators
        df_1d: Daily OHLCV data for LR filter
        initial_equity: Starting equity
        
    Returns:
        Dict with backtest results
    """
    print(f"\n{'='*60}")
    print(f"Running Backtest: {strategy.get_name()}")
    print(f"{'='*60}")
    
    # Calculate indicators
    print(f"\nCalculating indicators...")
    df_with_indicators = strategy.calculate_indicators(df_4h, df_1d)
    
    # Generate entry signals
    print(f"Generating signals...")
    entry_signals = strategy.generate_entry_signals(df_with_indicators)
    
    # Initialize backtest variables
    equity = initial_equity
    position = None
    trades = []
    equity_curve = []
    
    print(f"\nProcessing {len(df_with_indicators)} candles...")
    
    for i in range(len(df_with_indicators)):
        row = df_with_indicators.iloc[i]
        current_time = row['timestamp']
        current_price = row['close']
        
        # Check for exit if in position
        if position is not None:
            exit_reason = strategy.generate_exit_signals(
                df_with_indicators.iloc[:i+1], 
                position
            )
            
            if exit_reason is not None:
                # Close position
                entry_price = position['entry_price']
                entry_time = position['entry_time']
                
                # Calculate PnL
                pnl_pct = (current_price - entry_price) / entry_price * 100
                pnl_amount = position['size'] * pnl_pct / 100
                
                equity += pnl_amount
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_amount,
                    'equity': equity,
                    'exit_reason': exit_reason
                })
                
                position = None
        
        # Check for entry if not in position
        if position is None and entry_signals.iloc[i] == 1:
            # Open long position (100% equity)
            position = {
                'entry_price': current_price,
                'entry_time': current_time,
                'size': equity  # 100% of current equity
            }
        
        # Record equity
        current_equity = equity
        if position is not None:
            # Mark-to-market
            unrealized_pnl = (current_price - position['entry_price']) / position['entry_price'] * position['size']
            current_equity = equity + unrealized_pnl
        
        equity_curve.append({
            'timestamp': current_time,
            'equity': current_equity,
            'price': current_price
        })
    
    # Close any open position at the end
    if position is not None:
        last_price = df_with_indicators.iloc[-1]['close']
        last_time = df_with_indicators.iloc[-1]['timestamp']
        
        pnl_pct = (last_price - position['entry_price']) / position['entry_price'] * 100
        pnl_amount = position['size'] * pnl_pct / 100
        equity += pnl_amount
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': last_time,
            'entry_price': position['entry_price'],
            'exit_price': last_price,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'equity': equity,
            'exit_reason': 'end_of_data'
        })
    
    # Calculate results
    results = calculate_results(trades, equity_curve, initial_equity)
    
    return results


def calculate_results(trades, equity_curve, initial_equity):
    """Calculate backtest performance metrics."""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0
        }
    
    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)
    
    # Basic metrics
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    final_equity = df_equity['equity'].iloc[-1]
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100
    
    # Max drawdown
    df_equity['peak'] = df_equity['equity'].cummax()
    df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak'] * 100
    max_drawdown_pct = abs(df_equity['drawdown'].min())
    
    # Sharpe ratio (simplified)
    if len(df_equity) > 1:
        returns = df_equity['equity'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(8760) if returns.std() > 0 else 0  # 8760 = 24*365 for 4h bars
    else:
        sharpe = 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': total_trades - winning_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return_pct,
        'final_equity': final_equity,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe,
        'avg_win_pct': df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0,
        'avg_loss_pct': df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].mean() if (total_trades - winning_trades) > 0 else 0
    }


def print_results(results):
    """Print backtest results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"\nReturn: {results['total_return_pct']:.2f}%")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"\nAvg Win: {results['avg_win_pct']:.2f}%")
    print(f"Avg Loss: {results['avg_loss_pct']:.2f}%")
    print(f"{'='*60}")
    
    # Compare with video metrics
    print(f"\n{'='*60}")
    print(f"COMPARISON TO VIDEO METRICS")
    print(f"{'='*60}")
    print(f"Video claims: ~1,660% return, ~19% max DD, ~47% win rate, ~184 trades")
    print(f"")
    print(f"Your results:")
    print(f"  Return: {results['total_return_pct']:.2f}% {'✅' if results['total_return_pct'] > 1000 else '⚠️'}")
    print(f"  Max DD: {results['max_drawdown_pct']:.2f}% {'✅' if results['max_drawdown_pct'] < 25 else '⚠️'}")
    print(f"  Win Rate: {results['win_rate']:.2f}% {'✅' if 40 < results['win_rate'] < 55 else '⚠️'}")
    print(f"  Trades: {results['total_trades']} {'✅' if 150 < results['total_trades'] < 220 else '⚠️'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"HMA RSI LR Crypto Strategy Backtest")
    print(f"{'='*60}")
    
    # Create strategy
    config = create_hma_rsi_lr_config('BTC')
    strategy = HmaRsiLrCryptoStrategy(config)
    
    # Validate config
    is_valid, errors = strategy.validate_config()
    if not is_valid:
        print(f"❌ Invalid config:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    
    # Load data from local SQL
    db_path = str(ROOT / "data" / "trading_data_BTC.db")
    print(f"\nLoading data from: {db_path}")
    
    df_4h = load_data_from_sql(db_path, '4h')
    df_1d = load_data_from_sql(db_path, '1d')
    
    if len(df_4h) == 0 or len(df_1d) == 0:
        print(f"❌ Insufficient data")
        sys.exit(1)
    
    # Run backtest
    results = run_backtest(strategy, df_4h, df_1d, initial_equity=100000)
    
    # Print results
    print_results(results)
