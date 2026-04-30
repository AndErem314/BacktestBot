"""
Commodity Data Handler - Implements BaseDataHandler for commodity data.
Works with SQLite databases using the unified schema (same as crypto).
Supports Yahoo Finance data fetching and persistence.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from base.base_data_handler import BaseDataHandler

class CommodityDataHandler(BaseDataHandler):
    """
    Data handler for commodity trading data.
    Inherits from BaseDataHandler abstract class.
    
    Works with SQLite databases:
    - trading_data_GOLD.db (Gold)
    - trading_data_CLOIL.db (Crude Oil)
    
    Uses unified schema (same as crypto):
    - ohlcv_data table
    - ichimoku_data table
    - psar_data table (optional for commodities)
    - macd_data table (optional for commodities)
    """
    
    # Commodity symbol mapping
    COMMODITY_MAP = {
        'GOLD': ('trading_data_GOLD.db', 'GC=F'),
        'CLOIL': ('trading_data_CLOIL.db', 'CL=F'),
    }
    
    def __init__(self, symbol: str, db_path: Optional[str] = None):
        """
        Initialize commodity data handler.
        
        Args:
            symbol: Commodity symbol ('GOLD' or 'CLOIL')
            db_path: Optional explicit path to database (overrides auto-detection)
        """
        if symbol not in self.COMMODITY_MAP:
            raise ValueError(f"Unsupported commodity: {symbol}. Supported: {list(self.COMMODITY_MAP.keys())}")
        
        db_name, yahoo_symbol = self.COMMODITY_MAP[symbol]
        
        if db_path is None:
            # Auto-detect path relative to BacktestBot data directory
            project_root = Path(__file__).parent.parent
            db_path = str(project_root / 'data' / db_name)
        
        super().__init__(symbol, db_path)
        self.yahoo_symbol = yahoo_symbol
        self._conn = None
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn
    
    def initialize_database(self) -> None:
        """Create SQLite database with unified schema if not exists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create OHLCV table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                open DECIMAL(20, 8) NOT NULL CHECK(open > 0),
                high DECIMAL(20, 8) NOT NULL CHECK(high > 0),
                low DECIMAL(20, 8) NOT NULL CHECK(low > 0),
                close DECIMAL(20, 8) NOT NULL CHECK(close > 0),
                volume DECIMAL(20, 8) NOT NULL CHECK(volume >= 0),
                timeframe TEXT NOT NULL CHECK(timeframe IN ('4h', '1d')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CHECK(high >= low),
                CHECK(high >= open),
                CHECK(high >= close),
                CHECK(low <= open),
                CHECK(low <= close),
                UNIQUE(timestamp, timeframe)
            )
        """)
        
        # Create Ichimoku table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ichimoku_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ohlcv_id INTEGER NOT NULL,
                tenkan_sen DECIMAL(20, 8),
                kijun_sen DECIMAL(20, 8),
                senkou_span_a DECIMAL(20, 8),
                senkou_span_b DECIMAL(20, 8),
                chikou_span DECIMAL(20, 8),
                cloud_color TEXT CHECK(cloud_color IN ('green', 'red', NULL)),
                price_position TEXT CHECK(price_position IN ('above_cloud', 'in_cloud', 'below_cloud', NULL)),
                trend_strength TEXT CHECK(trend_strength IN ('strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish', NULL)),
                tk_cross TEXT CHECK(tk_cross IN ('bullish_cross', 'bearish_cross', 'no_cross', NULL)),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                UNIQUE(ohlcv_id)
            )
        """)
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert metadata
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                     ('symbol', self.symbol))
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                     ('yahoo_symbol', self.yahoo_symbol))
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                     ('asset_class', 'commodity'))
        
        conn.commit()
        print(f"Initialized commodity database: {self.db_path}")
    
    def load_ohlcv(self, 
                   timeframe: str, 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """Load OHLCV data for specified timeframe."""
        conn = self._get_connection()
        
        query = """
            SELECT timestamp, open, high, low, close, volume, timeframe
            FROM ohlcv_data
            WHERE timeframe = ?
        """
        params = [timeframe]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def save_ohlcv(self, data: pd.DataFrame, timeframe: str) -> None:
        """Persist OHLCV data to database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Clear existing data for this timeframe
        cursor.execute("DELETE FROM ohlcv_data WHERE timeframe = ?", (timeframe,))
        
        # Insert new data
        for _, row in data.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv_data 
                (timestamp, open, high, low, close, volume, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['timestamp'], pd.Timestamp) else row['timestamp'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                timeframe
            ))
        
        conn.commit()
        print(f"Saved {len(data)} {timeframe} candles to {self.db_path}")
    
    def load_indicators(self, indicator_type: str, timeframe: str) -> pd.DataFrame:
        """Load calculated indicators from database."""
        conn = self._get_connection()
        
        if indicator_type == 'ichimoku':
            query = """
                SELECT o.timestamp, o.open, o.high, o.low, o.close, o.volume,
                       i.tenkan_sen, i.kijun_sen, i.senkou_span_a, i.senkou_span_b,
                       i.chikou_span, i.cloud_color, i.price_position
                FROM ohlcv_data o
                LEFT JOIN ichimoku_data i ON o.id = i.ohlcv_id
                WHERE o.timeframe = ?
                ORDER BY o.timestamp
            """
        else:
            raise ValueError(f"Unsupported indicator type: {indicator_type}")
        
        df = pd.read_sql_query(query, conn, params=[timeframe])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def save_indicators(self, indicator_type: str, data: pd.DataFrame) -> None:
        """Persist calculated indicators to database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if indicator_type == 'ichimoku':
            # Get OHLCV IDs
            cursor.execute("SELECT id, timestamp FROM ohlcv_data WHERE timeframe = ?", ('1d',))
            ohlcv_ids = {row[1]: row[0] for row in cursor.fetchall()}
            
            # Clear existing data
            cursor.execute("""
                DELETE FROM ichimoku_data 
                WHERE ohlcv_id IN (SELECT id FROM ohlcv_data WHERE timeframe = ?)
            """, ('1d',))
            
            # Insert new data
            for _, row in data.iterrows():
                timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                ohlcv_id = ohlcv_ids.get(timestamp_str)
                if not ohlcv_id:
                    continue
                
                cursor.execute("""
                    INSERT OR REPLACE INTO ichimoku_data
                    (ohlcv_id, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b,
                     chikou_span, cloud_color, price_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ohlcv_id,
                    float(row['tenkan_sen']) if pd.notnull(row['tenkan_sen']) else None,
                    float(row['kijun_sen']) if pd.notnull(row['kijun_sen']) else None,
                    float(row['senkou_span_a']) if pd.notnull(row['senkou_span_a']) else None,
                    float(row['senkou_span_b']) if pd.notnull(row['senkou_span_b']) else None,
                    float(row['chikou_span']) if pd.notnull(row['chikou_span']) else None,
                    row['cloud_color'],
                    row['price_position']
                ))
        
        conn.commit()
        print(f"Saved {indicator_type} indicators to {self.db_path}")
    
    def get_latest_timestamp(self, timeframe: str) -> Optional[pd.Timestamp]:
        """Get most recent timestamp for specified timeframe."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MAX(timestamp) FROM ohlcv_data WHERE timeframe = ?
        """, (timeframe,))
        
        result = cursor.fetchone()[0]
        if result:
            return pd.Timestamp(result)
        return None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data per timeframe."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        summary = {}
        for tf in ['1d', '4h']:
            cursor.execute("""
                SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM ohlcv_data
                WHERE timeframe = ?
            """, (tf,))
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                count, min_ts, max_ts = result
                min_date = pd.Timestamp(min_ts)
                max_date = pd.Timestamp(max_ts)
                days = (max_date - min_date).days
                
                summary[tf] = {
                    'total_records': count,
                    'earliest_timestamp': min_ts,
                    'latest_timestamp': max_ts,
                    'days_of_data': days
                }
            else:
                summary[tf] = {
                    'total_records': 0,
                    'earliest_timestamp': None,
                    'latest_timestamp': None,
                    'days_of_data': 0
                }
        
        return summary
    
    def fetch_fresh_data(self, timeframe: str, period: str = '3y') -> pd.DataFrame:
        """
        Fetch fresh data from Yahoo Finance.
        
        Args:
            timeframe: '1d' or '4h'
            period: Yahoo Finance period string (e.g., '3y' for 3 years)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Install with: pip install yfinance")
        
        # Map timeframe to Yahoo interval
        interval_map = {'1d': '1d', '4h': '4h'}
        if timeframe not in interval_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        print(f"Fetching {self.yahoo_symbol} {timeframe} data from Yahoo Finance...")
        ticker = yf.Ticker(self.yahoo_symbol)
        data = ticker.history(period=period, interval=interval_map[timeframe])
        
        # Reset index to make timestamp a column
        data.reset_index(inplace=True)
        
        # Rename columns to match our schema
        data.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        print(f"Fetched {len(data)} {timeframe} candles")
        return data
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
