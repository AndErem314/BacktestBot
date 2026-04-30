"""
Quick test script to verify commodity strategy backtesting works.
"""
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner
import yaml

def test_commodity_strategy():
    """Test running a commodity strategy from strategies.yaml"""
    
    # Load strategy config
    with open('config/strategies.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get commodity strategy
    strategy_key = 'commodity_gold_winning'
    if strategy_key not in config['strategies']:
        print(f"Strategy {strategy_key} not found!")
        return
    
    strategy_config = config['strategies'][strategy_key]
    print(f"Testing strategy: {strategy_config['name']}")
    print(f"Asset class: {strategy_config.get('asset_class', 'crypto')}")
    print(f"Symbols: {strategy_config['symbols']}")
    print(f"Timeframes: {strategy_config['timeframes']}")
    
    # Create backtester with correct asset class
    asset_class = strategy_config.get('asset_class', 'crypto')
    backtester = IchimokuBacktester(asset_class=asset_class)
    
    # Create runner
    runner = StrategyBacktestRunner(backtester)
    
    # Fetch data for symbol
    symbol = strategy_config['symbols'][0]  # 'GOLD'
    timeframe = strategy_config['timeframes'][0]  # '1d'
    
    print(f"\nFetching data for {symbol} on {timeframe} timeframe...")
    
    try:
        data = runner.fetch_sql_data_with_signals(
            symbol_short=symbol,
            timeframe=timeframe,
            ichimoku_params=strategy_config['ichimoku_parameters']
        )
        
        if data.empty:
            print("No data returned! Check database.")
            return
        
        print(f"Data loaded: {len(data)} rows")
        print(f"Columns: {list(data.columns)}")
        
        # Run backtest
        print("\nRunning backtest...")
        result = runner.run_strategy_backtest(strategy_config, data)
        
        print(f"\nBacktest Results:")
        print(f"Total trades: {result.total_trades}")
        print(f"Win rate: {result.win_rate:.2%}")
        print(f"Total return: {result.total_return_pct:.2%}")
        print(f"Max drawdown: {result.max_drawdown_pct:.2%}")
        print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        
        # Debug: Check why no trades
        if result.total_trades == 0:
            print("\nDebug: No trades executed. Checking data...")
            print(f"Data shape: {data.shape}")
            print(f"Price above cloud count: {data['price_above_cloud'].sum()}")
            print(f"Tenkan above kijun count: {data['tenkan_above_kijun'].sum()}")
            print(f"SpanA above SpanB count: {data['senkou_span_a'].notna().sum()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_commodity_strategy()
