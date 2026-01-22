"""
Data Fetcher for OHLCV market data.
Integrates with the DataManager for streamlined data operations using per-symbol SQLite databases.
"""
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
from .data_manager import DataManager


class DataFetcher:
    """Fetches OHLCV data and stores it using the per-symbol database schema."""
    
    def __init__(self, exchange_name: str = 'binance', symbol: str = None, db_path: str = None):
        """
        Initialize data fetcher for a specific symbol.
        
        Args:
            exchange_name: Name of the exchange (default: binance)
            symbol: Cryptocurrency symbol (BTC, ETH, or SOL)
            db_path: Direct path to database (optional, uses symbol if not provided)
        """
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'enableRateLimit': True,
            'sandbox': False,
        })
        
        # Initialize data manager with symbol or path
        if symbol:
            self.symbol_short = symbol.upper()
            self.symbol = f"{symbol.upper()}/USDT"
            self.data_manager = DataManager(symbol=symbol)
        elif db_path:
            self.data_manager = DataManager(db_path=db_path)
            self.symbol = self.data_manager.symbol_pair
            self.symbol_short = self.data_manager.symbol
        else:
            raise ValueError("Either symbol or db_path must be provided")
        
        self.db_path = self.data_manager.db_path
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def fetch_ohlcv_batch(self, symbol: str, timeframe: str, since: int, limit: int = 1000) -> pd.DataFrame:
        """Fetch a single batch of OHLCV data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV batch for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_ohlcv(self, symbol: str, timeframe: str, start_time: int, 
                              end_time: Optional[int] = None, batch_size: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data in batches.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds (optional, defaults to now)
            batch_size: Number of records per API call
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        since = start_time
        batch_count = 0
        
        self.logger.info(f"Fetching historical data for {symbol} ({timeframe}) from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
        
        while since < end_time:
            df_batch = self.fetch_ohlcv_batch(symbol, timeframe, since, batch_size)
            
            if df_batch.empty:
                self.logger.warning(f"No data received for batch starting at {datetime.fromtimestamp(since/1000)}; advancing by one interval")
                since += self._get_timeframe_ms(timeframe)
                continue
            
            # Filter data to not exceed end_time
            df_batch = df_batch[df_batch.index <= pd.to_datetime(end_time, unit='ms')]

            # If filtering removed all rows, advance by one interval to avoid stalling
            if df_batch.empty:
                since += self._get_timeframe_ms(timeframe)
                continue
            
            all_data.append(df_batch)
            batch_count += 1
            
            # Update since to the next timestamp after the last received
            last_timestamp = int(df_batch.index[-1].timestamp() * 1000)
            since = last_timestamp + self._get_timeframe_ms(timeframe)
            
            # Rate limiting
            time.sleep(self.exchange.rateLimit / 1000)
            
            # Progress logging
            if batch_count % 10 == 0:
                self.logger.info(f"Fetched {batch_count} batches for {symbol} ({timeframe})")
        
        if all_data:
            full_df = pd.concat(all_data).drop_duplicates()
            full_df = full_df.sort_index()
            self.logger.info(f"Fetched {len(full_df)} total records for {symbol} ({timeframe})")
            return full_df
        
        return pd.DataFrame()
    
    def fetch_recent_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, 
                          limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent OHLCV data.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            since: Start timestamp (optional)
            limit: Maximum number of records
            
        Returns:
            DataFrame with recent OHLCV data
        """
        self.logger.info(f"Fetching recent data for {symbol} ({timeframe})")
        return self.fetch_ohlcv_batch(symbol, timeframe, since, limit)
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds."""
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000
        }
        
        if timeframe[-1] in multipliers:
            return int(timeframe[:-1]) * multipliers[timeframe[-1]]
        return 60 * 60 * 1000  # Default 1 hour
    
    def collect_and_store_data(self, symbol: str = None, timeframe: str = None, start_time: Optional[int] = None, 
                              force_full_refresh: bool = False) -> Dict[str, int]:
        """
        Collect OHLCV data and store it in the per-symbol database.
        
        Args:
            symbol: Trading pair (optional, uses instance symbol if not provided)
            timeframe: Timeframe (4h or 1d)
            start_time: Start timestamp (optional)
            force_full_refresh: If True, fetch all data from start_time
            
        Returns:
            Dictionary with operation statistics
        """
        try:
            # Use instance symbol if not provided
            if symbol is None:
                symbol = self.symbol
            
            # Validate timeframe
            if timeframe not in ['4h', '1d']:
                raise ValueError(f"Invalid timeframe: {timeframe}. Must be 4h or 1d")
            
            if force_full_refresh and start_time:
                # Fetch all historical data from start_time
                df = self.fetch_historical_ohlcv(symbol, timeframe, start_time)
                operation_type = "full_refresh"
            else:
                # Incremental update: fetch from last known timestamp
                last_timestamp = self.data_manager.get_last_timestamp(symbol, timeframe)
                
                if last_timestamp is None and start_time:
                    # No existing data, fetch from start_time
                    df = self.fetch_historical_ohlcv(symbol, timeframe, start_time)
                    operation_type = "initial_load"
                elif last_timestamp:
                    # Fetch recent data from last timestamp
                    df = self.fetch_recent_ohlcv(symbol, timeframe, since=last_timestamp)
                    operation_type = "incremental_update"
                else:
                    # No start_time provided and no existing data
                    self.logger.warning(f"No start_time provided and no existing data for {symbol} ({timeframe})")
                    return {'inserted': 0, 'duplicates': 0, 'errors': 0, 'operation_type': 'skipped'}
            
            if df.empty:
                self.logger.info(f"No new data to store for {symbol} ({timeframe})")
                return {'inserted': 0, 'duplicates': 0, 'errors': 0, 'operation_type': operation_type}
            
            # Store data using DataManager (symbol param optional for per-symbol db)
            result = self.data_manager.save_ohlcv_data(df, timeframe=timeframe)
            result['operation_type'] = operation_type
            
            self.logger.info(f"Data collection completed for {symbol} ({timeframe}): {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to collect and store data for {symbol} ({timeframe}): {e}")
            return {'inserted': 0, 'duplicates': 0, 'errors': 1, 'operation_type': 'error'}
    
    def fill_data_gaps(self, symbol: str = None, timeframe: str = None, start_time: int = None, end_time: int = None) -> Dict[str, int]:
        """
        Identify and fill gaps in existing data.
        
        Args:
            symbol: Trading pair (optional, uses instance symbol)
            timeframe: Timeframe (4h or 1d)
            start_time: Range start timestamp
            end_time: Range end timestamp
            
        Returns:
            Dictionary with gap filling statistics
        """
        # Use instance symbol if not provided
        if symbol is None:
            symbol = self.symbol
        
        self.logger.info(f"Checking for data gaps in {symbol} ({timeframe})")
        
        gaps = self.data_manager.get_data_gaps(symbol, timeframe, start_time, end_time)
        
        if not gaps:
            self.logger.info(f"No gaps found for {symbol} ({timeframe})")
            return {'inserted': 0, 'duplicates': 0, 'errors': 0, 'gaps_filled': 0}
        
        self.logger.info(f"Found {len(gaps)} gaps for {symbol} ({timeframe})")
        
        total_stats = {'inserted': 0, 'duplicates': 0, 'errors': 0, 'gaps_filled': 0}
        
        for gap_start, gap_end in gaps:
            self.logger.info(f"Filling gap: {datetime.fromtimestamp(gap_start/1000)} to {datetime.fromtimestamp(gap_end/1000)}")
            
            df_gap = self.fetch_historical_ohlcv(symbol, timeframe, gap_start, gap_end)
            
            if not df_gap.empty:
                result = self.data_manager.save_ohlcv_data(df_gap, timeframe=timeframe)
                total_stats['inserted'] += result['inserted']
                total_stats['duplicates'] += result['duplicates']
                total_stats['errors'] += result['errors']
                total_stats['gaps_filled'] += 1
            
            # Rate limiting between gap fills
            time.sleep(1)
        
        self.logger.info(f"Gap filling completed for {symbol} ({timeframe}): {total_stats}")
        return total_stats
    
    def bulk_collect_data(self, symbols: List[str], timeframes: List[str], 
                         start_time: Optional[int] = None, force_refresh: bool = False) -> Dict[str, Dict[str, Dict]]:
        """
        Collect data for multiple symbols and timeframes.
        
        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes
            start_time: Start timestamp (optional)
            force_refresh: Force full data refresh
            
        Returns:
            Nested dictionary with collection results
        """
        results = {}
        
        total_combinations = len(symbols) * len(timeframes)
        current_combination = 0
        
        self.logger.info(f"Starting bulk data collection for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        for symbol in symbols:
            results[symbol] = {}
            
            for timeframe in timeframes:
                current_combination += 1
                
                self.logger.info(f"Processing {symbol} ({timeframe}) - {current_combination}/{total_combinations}")
                
                try:
                    result = self.collect_and_store_data(
                        symbol, timeframe, start_time, force_refresh
                    )
                    results[symbol][timeframe] = result
                    
                    # Brief pause between symbol/timeframe combinations
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {symbol} ({timeframe}): {e}")
                    results[symbol][timeframe] = {
                        'inserted': 0, 'duplicates': 0, 'errors': 1, 'operation_type': 'error'
                    }
        
        # Generate summary
        total_inserted = sum(result.get('inserted', 0) for symbol_results in results.values() 
                            for result in symbol_results.values())
        total_duplicates = sum(result.get('duplicates', 0) for symbol_results in results.values() 
                              for result in symbol_results.values())
        total_errors = sum(result.get('errors', 0) for symbol_results in results.values() 
                          for result in symbol_results.values())
        
        self.logger.info(f"Bulk collection completed: {total_inserted} inserted, {total_duplicates} duplicates, {total_errors} errors")
        
        return results


class DataCollector:
    """High-level interface for data collection operations."""
    
    def __init__(self, exchange_name: str = 'binance', symbol: str = None, db_path: str = None):
        """
        Initialize data collector.
        
        Args:
            exchange_name: Exchange to use
            symbol: Cryptocurrency symbol (BTC, ETH, or SOL)
            db_path: Direct database path (alternative to symbol)
        """
        if symbol:
            self.symbol = symbol.upper()
            self.symbol_pair = f"{symbol.upper()}/USDT"
            self.fetcher = DataFetcher(exchange_name, symbol=symbol)
            self.data_manager = DataManager(symbol=symbol)
        elif db_path:
            # Extract symbol from db_path for compatibility
            if 'trading_data_' in db_path:
                symbol = db_path.split('trading_data_')[1].split('.')[0]
                self.symbol = symbol.upper()
                self.symbol_pair = f"{symbol.upper()}/USDT"
            else:
                self.symbol = 'UNKNOWN'
                self.symbol_pair = 'UNKNOWN/USDT'
            self.fetcher = DataFetcher(exchange_name, db_path=db_path)
            self.data_manager = DataManager(db_path=db_path)
        else:
            raise ValueError("Either symbol or db_path must be provided")
        
        self.logger = logging.getLogger(__name__)
    
    def update_all_data(self, timeframes: List[str]) -> None:
        """Update all existing data with latest values for this symbol."""
        self.logger.info(f"Starting incremental data update for {self.symbol_pair}")
        
        results = {}
        for timeframe in timeframes:
            result = self.fetcher.collect_and_store_data(timeframe=timeframe)
            results[timeframe] = result
        
        # Display summary
        print(f"\n=== DATA UPDATE SUMMARY FOR {self.symbol_pair} ===")
        for timeframe, result in results.items():
            print(f"{timeframe}: {result['inserted']} new records, "
                  f"{result['duplicates']} duplicates, {result['errors']} errors")
        
        print(f"\n=== DATABASE SUMMARY ===")
        summary_df = self.data_manager.get_data_summary()
        print(summary_df.to_string(index=False))
    
    def collect_historical_data(self, timeframes: List[str], start_date: str) -> None:
        """Collect historical data from a specific start date for this symbol."""
        start_time = int(pd.to_datetime(start_date).timestamp() * 1000)
        
        self.logger.info(f"Starting historical data collection for {self.symbol_pair} from {start_date}")
        
        results = {}
        for timeframe in timeframes:
            result = self.fetcher.collect_and_store_data(
                timeframe=timeframe, 
                start_time=start_time, 
                force_full_refresh=False
            )
            results[timeframe] = result
        
        # Display results
        print(f"\n=== HISTORICAL DATA COLLECTION SUMMARY FOR {self.symbol_pair} ===")
        for timeframe, result in results.items():
            print(f"{timeframe}: {result['inserted']} records collected (" 
                  f"{result['duplicates']} duplicates, {result['errors']} errors)")
    
    def validate_and_repair_data(self, timeframes: List[str], start_date: str) -> None:
        """Validate data integrity and fill any gaps for this symbol."""
        start_time = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        self.logger.info(f"Starting data validation and gap filling for {self.symbol_pair}")
        
        # Check data integrity
        integrity_issues = self.data_manager.validate_data_integrity()
        print(f"\n=== DATA INTEGRITY REPORT FOR {self.symbol_pair} ===")
        for issue_type, count in integrity_issues.items():
            if count > 0:
                print(f"⚠️  {issue_type}: {count} records")
            else:
                print(f"✅ {issue_type}: {count} records")
        
        # Fill gaps for each timeframe
        print(f"\n=== GAP FILLING REPORT ===")
        for timeframe in timeframes:
            gap_result = self.fetcher.fill_data_gaps(
                timeframe=timeframe, 
                start_time=start_time, 
                end_time=end_time
            )
            if gap_result['gaps_filled'] > 0:
                print(f"{timeframe}: {gap_result['gaps_filled']} gaps filled, "
                      f"{gap_result['inserted']} new records")
            else:
                print(f"{timeframe}: No gaps found")
