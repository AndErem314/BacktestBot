"""
Script to collect all historical data from August 1st, 2020 for selected trading pairs.

Uses per-symbol SQLite databases (data/trading_data_BTC.db, etc.) with simplified schema.
"""
from .data_fetcher import DataCollector
from .database_init import DatabaseInitializer
from typing import List
import os


def collect_all_historical_data_for_all_pairs(start_date: str = '2020-08-01', symbols: List[str] = None):
    """
    Collect all historical data for all trading pairs since the specified start date.
    Uses per-symbol databases for streamlined data management.
    
    Args:
        start_date: Start date for historical data collection
        symbols: List of symbol shorts to collect data for (e.g., ['BTC', 'ETH', 'SOL'])
    """
    # Symbol shorts and timeframes
    if symbols is None:
        symbols = ['BTC', 'ETH', 'SOL']  # All supported symbols
    timeframes = ['4h', '1d']

    print(f"📊 Starting historical data collection from {start_date}")
    print(f"🎯 Symbols: {symbols}")
    print(f"⏰ Timeframes: {timeframes}")
    
    # Initialize databases if they don't exist
    initializer = DatabaseInitializer()
    print("\n🔧 Checking database initialization...")
    init_results = initializer.initialize_all_databases(force_reinit=False)
    
    # Process each symbol with its own database
    for symbol in symbols:
        symbol_pair = f"{symbol}/USDT"
        print(f"\n🗄️ Processing {symbol_pair}")
        
        # Skip if database initialization failed
        if not init_results.get(symbol, False):
            print(f"❌ Skipping {symbol} - database initialization failed")
            continue
        
        # Initialize data collector with symbol
        collector = DataCollector(symbol=symbol)
        
        # Collect historical data for this symbol
        collector.collect_historical_data(timeframes, start_date)
    
    print("\n✅ Historical data collection completed for all symbols!")
    
    # Show final summary
    print("\n📊 Final Database Summary:")
    initializer.print_summary()


if __name__ == '__main__':
    collect_all_historical_data_for_all_pairs()
