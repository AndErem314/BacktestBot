"""
Test script to verify MACD/PSAR computation in UnifiedIchimokuAnalyzer.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project to path
sys.path.append(str(Path(__file__).parent))

from strategy.ichimoku_strategy import UnifiedIchimokuAnalyzer, IchimokuParameters, SignalType, SignalConditions

def test_macd_psar_computation():
    """Test that MACD/PSAR indicators are computed and signals generated.""" 
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='4h')
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.normal(0, 100, len(dates)))
    
    sample_data = pd.DataFrame({
        'close': close_prices,
        'high': close_prices + np.random.uniform(0, 200, len(dates)),
        'low': close_prices - np.random.uniform(0, 200, len(dates)),
        'open': close_prices + np.random.uniform(-100, 100, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}\n")
    
    # Initialize analyzer
    analyzer = UnifiedIchimokuAnalyzer()
    
    # Create Ichimoku parameters
    params = IchimokuParameters(
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52,
        chikou_offset=26,
        senkou_offset=26
    )
    
    # Test 1: Compute all indicators (Ichimoku + MACD + PSAR)
    print("=" * 60)
    print("TEST 1: compute_all_indicators() with MACD and PSAR")
    print("=" * 60)
    
    result_df = analyzer.compute_all_indicators(
        sample_data, 
        ichimoku_params=params,
        compute_macd=True,
        compute_psar=True
    )
    
    print(f"Result DataFrame shape: {result_df.shape}")
    print(f"Columns after computation: {list(result_df.columns)}\n")
    
    # Check MACD columns
    macd_cols = ['macd_line', 'signal_line', 'macd_histogram']
    print("MACD Columns Check:")
    for col in macd_cols:
        if col in result_df.columns:
            print(f"  ✓ {col} - EXISTS (sample: {result_df[col].iloc[-1]:.2f})")
        else:
            print(f"  ✗ {col} - MISSING")
    
    # Check PSAR columns
    psar_cols = ['psar', 'psar_trend', 'psar_reversal']
    print("\nPSAR Columns Check:")
    for col in psar_cols:
        if col in result_df.columns:
            print(f"  ✓ {col} - EXISTS (sample: {result_df[col].iloc[-1]})")
        else:
            print(f"  ✗ {col} - MISSING")
    
    # Check Ichimoku columns
    ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
    print("\nIchimoku Columns Check:")
    for col in ichimoku_cols:
        if col in result_df.columns:
            print(f"  ✓ {col} - EXISTS")
        else:
            print(f"  ✗ {col} - MISSING")
    
    # Test 2: detect_boolean_signals with MACD/PSAR
    print("\n" + "=" * 60)
    print("TEST 2: detect_boolean_signals() with MACD/PSAR signals")
    print("=" * 60)
    
    # Need to add Ichimoku boolean signals first
    result_df = analyzer.detect_boolean_signals(result_df, params)
    
    # Check MACD boolean signals
    macd_signal_cols = ['macd_above_signal', 'macd_below_signal']
    print("\nMACD Boolean Signals Check:")
    for col in macd_signal_cols:
        if col in result_df.columns:
            count = result_df[col].sum()
            print(f"  ✓ {col} - EXISTS ({count} True signals)")
        else:
            print(f"  ✗ {col} - MISSING")
    
    # Check PSAR boolean signals
    psar_signal_cols = ['psar_trend_up', 'psar_trend_down']
    print("\nPSAR Boolean Signals Check:")
    for col in psar_signal_cols:
        if col in result_df.columns:
            count = result_df[col].sum()
            print(f"  ✓ {col} - EXISTS ({count} True signals)")
        else:
            print(f"  ✗ {col} - MISSING")
    
    # Test 3: Create strategy config with MACD/PSAR signals and check
    print("\n" + "=" * 60)
    print("TEST 3: Strategy with MACD/PSAR signals")
    print("=" * 60)
    
    buy_conditions = [
        SignalType.PRICE_ABOVE_CLOUD,
        SignalType.TENKAN_ABOVE_KIJUN,
        SignalType.MACD_ABOVE_SIGNAL,
        SignalType.PSAR_TREND_UP
    ]
    sell_conditions = [
        SignalType.TENKAN_BELOW_KIJUN,
        SignalType.MACD_BELOW_SIGNAL,
        SignalType.PSAR_TREND_DOWN
    ]
    
    signal_conditions = SignalConditions(
        buy_conditions=buy_conditions,
        sell_conditions=sell_conditions,
        buy_logic="AND",
        sell_logic="OR"
    )
    
    # Check signals
    analysis = analyzer.check_strategy_signals(result_df, signal_conditions)
    
    print(f"\nBuy signal: {analysis['buy_signal']}")
    print(f"Sell signal: {analysis['sell_signal']}")
    print(f"Buy conditions met: {analysis['buy_conditions_met']}")
    print(f"Sell conditions met: {analysis['sell_conditions_met']}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    checks = [
        ('macd_line' in result_df.columns, "MACD computation"),
        ('psar' in result_df.columns, "PSAR computation"),
        ('macd_above_signal' in result_df.columns, "MACD boolean signals"),
        ('psar_trend_up' in result_df.columns, "PSAR boolean signals"),
    ]
    
    for check, name in checks:
        status = "✓ PASS" if check else "✗ FAIL"
        print(f"{status}: {name}")
        if not check:
            all_passed = False
    
    print(f"\n{'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
    
    return all_passed

if __name__ == "__main__":
    success = test_macd_psar_computation()
    sys.exit(0 if success else 1)