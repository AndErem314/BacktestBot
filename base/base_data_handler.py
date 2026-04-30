"""
Abstract Base Class for data loading and persistence.
Unified interface for both crypto (CCXT/SQLite) and commodities (Yahoo Finance/SQLite).
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any, List

class BaseDataHandler(ABC):
    """
    Abstract interface for data loading/persistence.
    All data handlers (crypto, commodity) must implement this interface.
    """
    
    def __init__(self, symbol: str, db_path: str):
        """
        Initialize data handler.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'GOLD/USDC')
            db_path: Path to SQLite database file
        """
        self.symbol = symbol
        self.db_path = db_path
    
    @abstractmethod
    def initialize_database(self) -> None:
        """
        Create SQLite database with predefined schema.
        Uses the unified schema from data/symbol_schema.sql
        """
        pass
    
    @abstractmethod
    def load_ohlcv(self, 
                   timeframe: str, 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data for specified timeframe.
        
        Args:
            timeframe: '1d' or '4h'
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def save_ohlcv(self, data: pd.DataFrame, timeframe: str) -> None:
        """
        Persist OHLCV data to database.
        
        Args:
            data: DataFrame with OHLCV data
            timeframe: '1d' or '4h'
        """
        pass
    
    @abstractmethod
    def load_indicators(self, indicator_type: str, timeframe: str) -> pd.DataFrame:
        """
        Load calculated indicators from database.
        
        Args:
            indicator_type: 'ichimoku', 'psar', or 'macd'
            timeframe: '1d' or '4h'
            
        Returns:
            DataFrame with indicator values joined to OHLCV data
        """
        pass
    
    @abstractmethod
    def save_indicators(self, indicator_type: str, data: pd.DataFrame) -> None:
        """
        Persist calculated indicators to database.
        
        Args:
            indicator_type: 'ichimoku', 'psar', or 'macd'
            data: DataFrame with indicator values and ohlcv_id column
        """
        pass
    
    @abstractmethod
    def get_latest_timestamp(self, timeframe: str) -> Optional[pd.Timestamp]:
        """
        Get most recent timestamp for specified timeframe.
        
        Returns:
            Timestamp of latest candle, or None if no data
        """
        pass
    
    @abstractmethod
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data per timeframe.
        
        Returns:
            Dict with keys '1d' and '4h', each containing:
            - total_records: int
            - earliest_timestamp: str
            - latest_timestamp: str
            - days_of_data: float
        """
        pass
