"""
Test script for Step 6: Testing & Validation of multi-asset backtesting.
Tests both crypto and commodity strategies with PDF report generation.
"""
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent))

from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner
from reporting.report_generator import ReportGenerator

def test_with_report(strategy_key, description):
    """Test a strategy and generate PDF report."""
    print(f"\n{'='*60}")
    print(f"TESTING: {description}")
    print(f"{'='*60}")
    
    # Load strategy
    with open('config/strategies.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if strategy_key not in config['strategies']:
        print(f"Strategy {strategy_key} not found!")
        return
    
    strategy_config = config['strategies'][strategy_key]
    asset_class = strategy_config.get('asset_class', 'crypto')
    symbol = strategy_config['symbols'][0]
    timeframe = strategy_config['timeframes'][0]
    
    print(f"Strategy: {strategy_config['name']}")
    print(f"Asset Class: {asset_class.upper()}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    
    # Create runner
    runner = StrategyBacktestRunner()
    
    # Fetch data and run backtest
    try:
        data = runner.fetch_sql_data_with_signals(
            symbol_short=symbol,
            timeframe=timeframe,
            ichimoku_params=strategy_config['ichimoku_parameters']
        )
        
        if data.empty:
            print("No data returned!")
            return
        
        print(f"Data loaded: {len(data)} rows")
        
        # Run backtest
        result = runner.run_strategy_backtest(strategy_config, data)
        
        if result is None:
            print("Backtest returned None!")
            return
        
        print(f"\nBacktest Results:")
        print(f"  Total trades: {result.total_trades}")
        print(f"  Win rate: {result.win_rate:.2%}")
        print(f"  Total return: {result.total_return_pct:.2f}%")
        print(f"  Max drawdown: {result.max_drawdown_pct:.2%}")
        print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
        
        # Generate PDF report
        if result.total_trades > 0:
            print(f"\nGenerating PDF report...")
            report_gen = ReportGenerator(output_dir='reports')
            
            # Prepare results dict for report generator
            results_for_report = {
                'strategy_config': strategy_config,
                'metrics': {
                    'performance_metrics': {
                        'total_return': result.total_return_pct / 100,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown_pct / 100,
                        'win_rate': result.win_rate,
                        'profit_factor': result.profit_factor,
                        'total_trades': result.total_trades,
                    }
                },
                'trades': result.trades,
                'equity_curve': result.equity_curve,
            }
            
            try:
                reports = report_gen.generate_backtest_report(
                    results=results_for_report,
                    format='pdf',
                    filename_prefix=f"{strategy_key}_test"
                )
                print(f"PDF report generated: {reports.get('pdf', 'N/A')}")
            except Exception as e:
                print(f"Error generating PDF: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Test commodity strategy
    test_with_report('commodity_gold_winning', 'Gold Commodity Strategy')
    
    # Test crypto strategy
    test_with_report('strategy_01', 'Bitcoin Crypto Strategy')
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")
