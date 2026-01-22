"""
Data Manager for OHLCV data operations.
Handles all database operations for the simplified per-symbol SQLite schema.
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
import logging
import numpy as np


class DataManager:
    """Manages OHLCV and Ichimoku data operations for per-symbol trading databases."""
    
    def __init__(self, symbol: str = None, db_path: str = None):
        """
        Initialize the data manager for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol (BTC, ETH, or SOL). If None, must provide db_path
            db_path: Direct path to database file. If None, uses symbol to construct path
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Determine database path
        if db_path is None and symbol is None:
            raise ValueError("Either symbol or db_path must be provided")
        
        if db_path:
            # Direct path provided (for backward compatibility)
            if not os.path.isabs(db_path):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
                db_path = os.path.abspath(os.path.join(project_root, db_path))
            self.db_path = db_path
            # Extract symbol from path if possible
            if 'trading_data_' in os.path.basename(db_path):
                self.symbol = os.path.basename(db_path).replace('trading_data_', '').replace('.db', '').upper()
            else:
                self.symbol = 'UNKNOWN'
        else:
            # Symbol provided - construct path
            self.symbol = symbol.upper()
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            data_dir = os.path.join(project_root, 'data')
            self.db_path = os.path.join(data_dir, f'trading_data_{self.symbol}.db')
        
        self.symbol_pair = f"{self.symbol}/USDT"
        self.ensure_data_directory()
        
        # Verify database exists
        if not os.path.exists(self.db_path):
            self.logger.warning(f"Database not found: {self.db_path}. Please run database_init.py first.")
        
        # Connection management
        self._conn = None
        self._transaction_active = False
    
    def get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
            self._conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes
        return self._conn
    
    def close_connection(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def begin_transaction(self):
        """Begin a database transaction."""
        conn = self.get_connection()
        conn.execute("BEGIN TRANSACTION")
        self._transaction_active = True
    
    def commit_transaction(self):
        """Commit the current transaction."""
        if self._transaction_active:
            self.get_connection().commit()
            self._transaction_active = False
    
    def rollback_transaction(self):
        """Rollback the current transaction."""
        if self._transaction_active:
            self.get_connection().rollback()
            self._transaction_active = False

    def ensure_data_directory(self):
        """Ensure data directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def save_ohlcv_data(self, df: pd.DataFrame, symbol: str = None, timeframe: str = None) -> Dict[str, int]:
        """
        Save OHLCV data to the database.
        
        Args:
            df: DataFrame with OHLCV data (timestamp as index)
            symbol: Trading pair symbol (optional, uses instance symbol if not provided)
            timeframe: Timeframe (4h or 1d)
            
        Returns:
            Dict with 'inserted', 'duplicates', 'errors' counts
        """
        if df.empty:
            return {'inserted': 0, 'duplicates': 0, 'errors': 0}
        
        # Use instance symbol if not provided (for per-symbol databases)
        if symbol is None:
            symbol = self.symbol_pair
        
        # Validate timeframe
        if timeframe not in ['4h', '1d']:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be 4h or 1d")
        
        # Prepare data
        df_copy = df.copy()
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['timestamp'] = df_copy.index
        else:
            df_copy['timestamp'] = pd.to_datetime(df_copy.index)
        
        # Ensure we have all required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df_copy.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Prepare records for insertion
        records = []
        for _, row in df_copy.iterrows():
            # Validate data
            if any(pd.isna([row['open'], row['high'], row['low'], row['close'], row['volume']])):
                self.logger.warning(f"Skipping record with NaN values at {row['timestamp']}")
                continue
            
            if row['high'] < row['low'] or any(val < 0 for val in [row['open'], row['high'], row['low'], row['close']]):
                self.logger.warning(f"Skipping invalid OHLCV data at {row['timestamp']}")
                continue
            
            records.append((
                row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                timeframe
            ))
        
        if not records:
            return {'inserted': 0, 'duplicates': 0, 'errors': 0}
        
        # Batch insert
        inserted_count = 0
        duplicate_count = 0
        error_count = 0
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for record in records:
                try:
                    cursor.execute('''
                        INSERT INTO ohlcv_data 
                        (timestamp, open, high, low, close, volume, timeframe)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', record)
                    inserted_count += 1
                except sqlite3.IntegrityError:
                    duplicate_count += 1
                except sqlite3.Error as e:
                    self.logger.error(f"Error inserting record: {e}")
                    error_count += 1
            
            if not self._transaction_active:
                conn.commit()
            
            self.logger.info(
                f"OHLCV data saved for {self.symbol} ({timeframe}): "
                f"{inserted_count} inserted, {duplicate_count} duplicates, {error_count} errors"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save OHLCV data: {e}")
            if not self._transaction_active:
                conn.rollback()
            raise
        
        return {
            'inserted': inserted_count,
            'duplicates': duplicate_count,
            'errors': error_count
        }
    
    def get_last_timestamp(self, symbol: str = None, timeframe: str = None) -> Optional[int]:
        """Get the last timestamp for a specific timeframe in milliseconds."""
        try:
            conn = self.get_connection()
            cursor = conn.execute(
                "SELECT MAX(timestamp) FROM ohlcv_data WHERE timeframe = ?",
                (timeframe,)
            )
            result = cursor.fetchone()[0]
            
            if result:
                return int(pd.to_datetime(result).timestamp() * 1000)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get last timestamp: {e}")
            return None
    
    def get_ohlcv_data(self, timeframe: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data from the database.
        
        Args:
            timeframe: Timeframe to retrieve (1h, 4h, or 1d)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            limit: Maximum number of records (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        query = "SELECT * FROM ohlcv_data WHERE timeframe = ?"
        params = [timeframe]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat() if isinstance(start_date, datetime) else start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat() if isinstance(end_date, datetime) else end_date)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            conn = self.get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve OHLCV data: {e}")
            return pd.DataFrame()
    
    def save_ichimoku_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Save Ichimoku indicator data to the database.
        
        Args:
            df: DataFrame with Ichimoku data and ohlcv_id
            
        Returns:
            Dict with operation statistics
        """
        if df.empty:
            return {'inserted': 0, 'updated': 0, 'errors': 0}
        
        required_columns = ['ohlcv_id', 'tenkan_sen', 'kijun_sen', 
                          'senkou_span_a', 'senkou_span_b', 'chikou_span']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        inserted_count = 0
        updated_count = 0
        error_count = 0
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    # Calculate additional fields
                    cloud_thickness = abs(row.get('senkou_span_a', 0) - row.get('senkou_span_b', 0))
                    cloud_color = 'green' if row.get('senkou_span_a', 0) > row.get('senkou_span_b', 0) else 'red'
                    
                    # Try to insert or update
                    cursor.execute('''
                        INSERT OR REPLACE INTO ichimoku_data 
                        (ohlcv_id, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, 
                         chikou_span, cloud_color, cloud_thickness, price_position, 
                         trend_strength, tk_cross)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        int(row['ohlcv_id']),
                        float(row['tenkan_sen']) if pd.notna(row['tenkan_sen']) else None,
                        float(row['kijun_sen']) if pd.notna(row['kijun_sen']) else None,
                        float(row['senkou_span_a']) if pd.notna(row['senkou_span_a']) else None,
                        float(row['senkou_span_b']) if pd.notna(row['senkou_span_b']) else None,
                        float(row['chikou_span']) if pd.notna(row['chikou_span']) else None,
                        cloud_color,
                        float(cloud_thickness) if pd.notna(cloud_thickness) else None,
                        row.get('price_position'),
                        row.get('trend_strength'),
                        row.get('tk_cross')
                    ))
                    
                    if cursor.rowcount > 0:
                        inserted_count += 1
                    
                except sqlite3.Error as e:
                    self.logger.error(f"Error saving Ichimoku data: {e}")
                    error_count += 1
            
            if not self._transaction_active:
                conn.commit()
            
            self.logger.info(
                f"Ichimoku data saved for {self.symbol}: "
                f"{inserted_count} records, {error_count} errors"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save Ichimoku data: {e}")
            if not self._transaction_active:
                conn.rollback()
            raise
        
        return {
            'inserted': inserted_count,
            'updated': updated_count,
            'errors': error_count
        }
    
    def get_ichimoku_data(self, timeframe: str, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data with Ichimoku indicators and PSAR if available.
        
        Args:
            timeframe: Timeframe to retrieve
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with OHLCV and Ichimoku data
        """
        # Prefer dynamic join to include PSAR data when table exists; fallback to view otherwise
        params = [timeframe]
        try:
            conn = self.get_connection()
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='psar_data'")
            has_psar = cur.fetchone() is not None
        except Exception:
            has_psar = False
        # Build WHERE clauses safely, then add ORDER BY at the very end
        where_clauses = []
        if has_psar:
            base_select = (
                "SELECT o.id, o.timestamp, o.open, o.high, o.low, o.close, o.volume, o.timeframe, "
                "i.tenkan_sen, i.kijun_sen, i.senkou_span_a, i.senkou_span_b, i.chikou_span, "
                "i.cloud_color, i.cloud_thickness, i.price_position, i.trend_strength, i.tk_cross, "
                "p.psar, p.psar_trend, p.psar_reversal "
                "FROM ohlcv_data o "
                "LEFT JOIN ichimoku_data i ON o.id = i.ohlcv_id "
                "LEFT JOIN psar_data p ON o.id = p.ohlcv_id "
            )
            where_clauses.append("o.timeframe = ?")
        else:
            base_select = "SELECT * FROM ohlcv_ichimoku_view "
            where_clauses.append("timeframe = ?")

        if start_date:
            where_clauses.append("timestamp >= ?")
            params.append(start_date.isoformat() if isinstance(start_date, datetime) else start_date)
        if end_date:
            where_clauses.append("timestamp <= ?")
            params.append(end_date.isoformat() if isinstance(end_date, datetime) else end_date)

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        order_sql = " ORDER BY o.timestamp" if has_psar else " ORDER BY timestamp"
        query = base_select + where_sql + order_sql
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat() if isinstance(start_date, datetime) else start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat() if isinstance(end_date, datetime) else end_date)
        
        try:
            conn = self.get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve Ichimoku data: {e}")
            return pd.DataFrame()
    
    def save_psar_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """Save PSAR indicator data to the database.

        Expects columns: ohlcv_id, psar, psar_trend, psar_reversal, optional: step, max_step
        """
        if df.empty:
            return {'inserted': 0, 'updated': 0, 'errors': 0}
        required_columns = ['ohlcv_id', 'psar']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for PSAR save: {missing}")
        inserted = 0
        updated = 0
        errors = 0
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            for _, row in df.iterrows():
                try:
                    cur.execute('''
                        INSERT OR REPLACE INTO psar_data
                        (ohlcv_id, psar, psar_trend, psar_reversal, step, max_step)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        int(row['ohlcv_id']),
                        float(row['psar']) if pd.notna(row['psar']) else None,
                        int(row['psar_trend']) if pd.notna(row.get('psar_trend')) else None,
                        int(bool(row.get('psar_reversal'))) if row.get('psar_reversal') is not None else None,
                        float(row.get('step')) if row.get('step') is not None else None,
                        float(row.get('max_step')) if row.get('max_step') is not None else None,
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    self.logger.error(f"Error saving PSAR data: {e}")
                    errors += 1
            if not self._transaction_active:
                conn.commit()
            self.logger.info(f"PSAR data saved for {self.symbol}: {inserted} records, {errors} errors")
        except Exception as e:
            self.logger.error(f"Failed to save PSAR data: {e}")
            if not self._transaction_active:
                conn.rollback()
            raise
        return {'inserted': inserted, 'updated': updated, 'errors': errors}

    def save_macd_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """Save MACD indicator data to the database.

        Expects columns: ohlcv_id, macd_line, signal_line, macd_histogram, 
        optional: fast_period, slow_period, signal_period
        """
        if df.empty:
            return {'inserted': 0, 'updated': 0, 'errors': 0}
        required_columns = ['ohlcv_id', 'macd_line', 'signal_line', 'macd_histogram']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for MACD save: {missing}")
        inserted = 0
        updated = 0
        errors = 0
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            for _, row in df.iterrows():
                try:
                    cur.execute('''
                        INSERT OR REPLACE INTO macd_data
                        (ohlcv_id, macd_line, signal_line, macd_histogram, fast_period, slow_period, signal_period)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        int(row['ohlcv_id']),
                        float(row['macd_line']) if pd.notna(row['macd_line']) else None,
                        float(row['signal_line']) if pd.notna(row['signal_line']) else None,
                        float(row['macd_histogram']) if pd.notna(row['macd_histogram']) else None,
                        int(row.get('fast_period')) if row.get('fast_period') is not None else None,
                        int(row.get('slow_period')) if row.get('slow_period') is not None else None,
                        int(row.get('signal_period')) if row.get('signal_period') is not None else None,
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    self.logger.error(f"Error saving MACD data: {e}")
                    errors += 1
            if not self._transaction_active:
                conn.commit()
            self.logger.info(f"MACD data saved for {self.symbol}: {inserted} records, {errors} errors")
        except Exception as e:
            self.logger.error(f"Failed to save MACD data: {e}")
            if not self._transaction_active:
                conn.rollback()
            raise
        return {'inserted': inserted, 'updated': updated, 'errors': errors}

    def get_latest_signals(self, timeframe: str = None, limit: int = 10) -> pd.DataFrame:
        """
        Get the latest Ichimoku trading signals.
        
        Args:
            timeframe: Specific timeframe or None for all
            limit: Number of recent signals to retrieve
            
        Returns:
            DataFrame with recent signals
        """
        query = "SELECT * FROM ichimoku_signals_view"
        params = []
        
        if timeframe:
            query += " WHERE timeframe = ?"
            params.append(timeframe)
        
        query += f" LIMIT {limit}"
        
        try:
            conn = self.get_connection()
            return pd.read_sql_query(query, conn, params=params)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve signals: {e}")
            return pd.DataFrame()
    
    def get_data_gaps(self, symbol: str = None, timeframe: str = None, start_time: int = None, end_time: int = None) -> List[Tuple[int, int]]:
        """
        Identify gaps in data for a given timeframe.
        
        Args:
            symbol: Not used in per-symbol database
            timeframe: Timeframe to check
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            List of (gap_start, gap_end) tuples in milliseconds
        """
        try:
            conn = self.get_connection()
            cursor = conn.execute('''
                SELECT timestamp FROM ohlcv_data 
                WHERE timeframe = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (timeframe, 
                  pd.to_datetime(start_time, unit='ms').isoformat(),
                  pd.to_datetime(end_time, unit='ms').isoformat()))
            
            timestamps = [int(pd.to_datetime(row[0]).timestamp() * 1000) for row in cursor.fetchall()]
            
            if not timestamps:
                return [(start_time, end_time)]
            
            # Calculate expected interval
            timeframe_ms = self._timeframe_to_ms(timeframe)
            gaps = []
            
            # Check gap before first timestamp
            if timestamps[0] > start_time + timeframe_ms:
                gaps.append((start_time, timestamps[0] - timeframe_ms))
            
            # Check gaps between timestamps
            for i in range(len(timestamps) - 1):
                expected_next = timestamps[i] + timeframe_ms
                if timestamps[i + 1] > expected_next + timeframe_ms:
                    gaps.append((expected_next, timestamps[i + 1] - timeframe_ms))
            
            # Check gap after last timestamp
            if timestamps[-1] < end_time - timeframe_ms:
                gaps.append((timestamps[-1] + timeframe_ms, end_time))
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Failed to identify data gaps: {e}")
            return [(start_time, end_time)]
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        mapping = {
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return mapping.get(timeframe, 60 * 60 * 1000)
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get a summary of all OHLCV data in the database."""
        try:
            conn = self.get_connection()
            query = '''
                SELECT 
                    timeframe,
                    COUNT(*) as record_count,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM ohlcv_data
                GROUP BY timeframe
                ORDER BY timeframe
            '''
            
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['symbol'] = self.symbol_pair
                df = df[['symbol', 'timeframe', 'record_count', 'earliest', 'latest']]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get data summary: {e}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        try:
            conn = self.get_connection()
            stats = {
                'symbol': self.symbol_pair,
                'database_path': self.db_path,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            }
            
            # Get record counts
            cursor = conn.execute("SELECT COUNT(*) FROM ohlcv_data")
            stats['total_ohlcv_records'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM ichimoku_data")
            stats['total_ichimoku_records'] = cursor.fetchone()[0]
            
            # Get timeframe breakdown
            cursor = conn.execute("""
                SELECT timeframe, 
                       COUNT(*) as count,
                       MIN(timestamp) as earliest,
                       MAX(timestamp) as latest
                FROM ohlcv_data 
                GROUP BY timeframe
            """)
            
            stats['timeframes'] = {}
            for row in cursor.fetchall():
                stats['timeframes'][row[0]] = {
                    'count': row[1],
                    'earliest': row[2],
                    'latest': row[3],
                    'days_covered': (pd.to_datetime(row[3]) - pd.to_datetime(row[2])).days if row[2] and row[3] else 0
                }
            
            # Get metadata
            cursor = conn.execute("SELECT key, value FROM metadata")
            stats['metadata'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}
    
    def validate_data_integrity(self) -> Dict[str, int]:
        """Validate OHLCV data integrity and return statistics."""
        issues = {
            'invalid_ohlc': 0,
            'negative_prices': 0,
            'zero_volume': 0
        }
        
        try:
            conn = self.get_connection()
            
            # Check for invalid OHLC relationships
            cursor = conn.execute('''
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE high < low OR open < 0 OR high < 0 OR low < 0 OR close < 0
            ''')
            issues['invalid_ohlc'] = cursor.fetchone()[0]
            
            # Check for negative prices
            cursor = conn.execute('''
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE open < 0 OR high < 0 OR low < 0 OR close < 0
            ''')
            issues['negative_prices'] = cursor.fetchone()[0]
            
            # Check for zero volume
            cursor = conn.execute('''
                SELECT COUNT(*) FROM ohlcv_data WHERE volume = 0
            ''')
            issues['zero_volume'] = cursor.fetchone()[0]
            
            self.logger.info(f"Data integrity check completed: {issues}")
            
        except Exception as e:
            self.logger.error(f"Data integrity validation failed: {e}")
        
        return issues
    
    def optimize_database(self):
        """Optimize the database for better performance."""
        try:
            conn = self.get_connection()
            
            # Analyze tables for query optimization
            conn.execute("ANALYZE ohlcv_data")
            conn.execute("ANALYZE ichimoku_data")
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
            self.logger.info(f"Database optimization completed for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize database: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
