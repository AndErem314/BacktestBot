#!/usr/bin/env python3
"""
HMA RSI Linear Regression Crypto Strategy
Implements the strategy from: https://www.youtube.com/watch?v=aOIRo4Q7qZE

Strategy Logic (multi-timeframe):
- 4h timeframe: HMA(16) crosses above HMA(65) + RSI(14) > 52
- Daily timeframe filter: Close > Linear Regression(50) line
- Exit: HMA(16) crosses below HMA(65)

No fixed stop loss or take profit - signal-driven only.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import pandas_ta as ta

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.base_strategy import BaseStrategy, StrategyConfig
from base.base_data_handler import BaseDataHandler


class HmaRsiLrCryptoStrategy(BaseStrategy):
    """
    HMA RSI Linear Regression strategy for crypto trading.
    
    Entry (all 3 must align):
    1. HMA(16) crosses above HMA(65) on 4h timeframe
    2. RSI(14) > 52 on 4h timeframe
    3. Daily close > Linear Regression(50) line (daily timeframe filter)
    
    Exit:
    - HMA(16) crosses below HMA(65) on 4h timeframe
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.fast_hma_period = config.strategy_parameters.get('fast_hma_period', 16)
        self.slow_hma_period = config.strategy_parameters.get('slow_hma_period', 65)
        self.rsi_period = config.strategy_parameters.get('rsi_period', 14)
        self.rsi_threshold = config.strategy_parameters.get('rsi_threshold', 52)
        self.lr_period = config.strategy_parameters.get('lr_period', 50)
        
    def calculate_indicators(self, ohlcv_data: pd.DataFrame, daily_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate all required indicators for this strategy.
        
        Args:
            ohlcv_data: DataFrame with 4h OHLCV data
            daily_data: DataFrame with daily OHLCV data (for LR filter)
            
        Returns:
            DataFrame with original data plus indicator columns
        """
        df = ohlcv_data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate HMA (Hull Moving Average)
        df['hma_fast'] = ta.hma(df['close'], length=self.fast_hma_period)
        df['hma_slow'] = ta.hma(df['close'], length=self.slow_hma_period)
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        
        # Calculate HMA crossover signals
        df['hma_fast_prev'] = df['hma_fast'].shift(1)
        df['hma_slow_prev'] = df['hma_slow'].shift(1)
        
        # Bullish cross: previous fast <= slow, current fast > slow
        df['hma_bullish_cross'] = (
            (df['hma_fast_prev'].le(df['hma_slow_prev'])) & 
            (df['hma_fast'].gt(df['hma_slow']))
        )
        
        # Bearish cross: previous fast >= slow, current fast < slow
        df['hma_bearish_cross'] = (
            (df['hma_fast_prev'].ge(df['hma_slow_prev'])) & 
            (df['hma_fast'].lt(df['hma_slow']))
        )
        
        # Add daily Linear Regression filter if daily data provided
        if daily_data is not None and len(daily_data) > 0:
            daily_lr = self._calculate_linear_regression(daily_data, self.lr_period)
            
            # Forward-fill daily LR values to 4h timestamps
            df['daily_close'] = np.nan
            df['daily_lr'] = np.nan
            
            for idx, row in df.iterrows():
                # Find the most recent daily candle on or before this 4h candle
                daily_candle = daily_data[daily_data['timestamp'] <= row['timestamp']].iloc[-1:]
                if not daily_candle.empty:
                    daily_ts = daily_candle.iloc[0]['timestamp']
                    if daily_ts in daily_lr.index:
                        df.loc[idx, 'daily_lr'] = daily_lr.loc[daily_ts]
                        df.loc[idx, 'daily_close'] = daily_candle.iloc[0]['close']
            
            # Forward fill NaN values
            df['daily_lr'] = df['daily_lr'].ffill()
            df['daily_close'] = df['daily_close'].ffill()
            
            # Daily trend filter: close > LR line
            df['daily_trend_bullish'] = df['daily_close'] > df['daily_lr']
        else:
            # If no daily data, assume trend is bullish (will be handled by caller)
            df['daily_trend_bullish'] = True
        
        return df
    
    def _calculate_linear_regression(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Linear Regression line for given period.
        
        Args:
            df: DataFrame with OHLCV data
            period: Lookback period for linear regression
            
        Returns:
            Series with LR values indexed by timestamp
        """
        lr_values = {}
        
        for i in range(period, len(df) + 1):
            window = df.iloc[i-period:i]
            x = np.arange(period)
            y = window['close'].values
            
            # Calculate linear regression
            coeffs = np.polyfit(x, y, 1)
            lr_value = np.polyval(coeffs, period - 1)  # Value at end of window
            
            timestamp = df.iloc[i-1]['timestamp']
            lr_values[timestamp] = lr_value
        
        return pd.Series(lr_values)
    
    def generate_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate entry signals based on strategy logic.
        
        Args:
            data: DataFrame with OHLCV + indicator data
            
        Returns:
            Series with values: 1 (long), -1 (short), 0 (no signal)
        """
        # All 3 conditions must align for entry
        conditions = (
            data['hma_bullish_cross'] &           # HMA bullish cross
            data['rsi'].gt(self.rsi_threshold) &  # RSI > threshold
            data['daily_trend_bullish']            # Daily trend filter
        )
        
        signals = pd.Series(0, index=data.index)
        signals.loc[conditions] = 1  # Long signal
        
        return signals
    
    def generate_exit_signals(self, data: pd.DataFrame, position: Dict) -> Optional[str]:
        """
        Check exit conditions for current position.
        
        Args:
            data: DataFrame with OHLCV + indicator data (current candle)
            position: Dict with position info (entry_price, entry_time, etc.)
            
        Returns:
            Exit reason string or None if no exit signal
        """
        if len(data) == 0:
            return None
        
        # Check latest candle for bearish HMA cross
        latest = data.iloc[-1]
        
        if latest['hma_bearish_cross']:
            return 'hma_bearish_cross'
        
        return None
    
    def get_name(self) -> str:
        """Return human-readable strategy name."""
        return f"HMA_RSI_LR_Crypto_{self.fast_hma_period}_{self.slow_hma_period}"
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate strategy configuration."""
        errors = []
        
        if self.fast_hma_period <= 0:
            errors.append("fast_hma_period must be positive")
        
        if self.slow_hma_period <= self.fast_hma_period:
            errors.append("slow_hma_period must be greater than fast_hma_period")
        
        if self.rsi_period <= 0:
            errors.append("rsi_period must be positive")
        
        if not (0 <= self.rsi_threshold <= 100):
            errors.append("rsi_threshold must be between 0 and 100")
        
        if self.lr_period <= 0:
            errors.append("lr_period must be positive")
        
        return (len(errors) == 0, errors)


def create_hma_rsi_lr_config(symbol: str = 'BTC', timeframe: str = '4h') -> StrategyConfig:
    """
    Create a StrategyConfig for the HMA RSI LR strategy.
    
    Args:
        symbol: Trading symbol (BTC, ETH, SOL)
        timeframe: Primary timeframe for signals
        
    Returns:
        StrategyConfig object
    """
    return StrategyConfig(
        name=f"HMA_RSI_LR_{symbol}",
        asset_class='crypto',
        timeframe='both',  # Uses both 4h and daily
        strategy_parameters={
            'fast_hma_period': 16,
            'slow_hma_period': 65,
            'rsi_period': 14,
            'rsi_threshold': 52,
            'lr_period': 50
        },
        signal_conditions={
            'entry': 'HMA(16) crosses above HMA(65) AND RSI(14) > 52 AND Daily close > LR(50)',
            'exit': 'HMA(16) crosses below HMA(65)'
        },
        risk_management={
            'stop_loss_pct': None,  # Signal-driven
            'take_profit_pct': None,  # Signal-driven
        },
        position_sizing={
            'method': 'full_equity',
            'size': 1.0  # 100% of equity
        },
        description="Multi-timeframe strategy using HMA crossover, RSI momentum, and daily LR trend filter",
        enabled=True,
        symbols=[f"{symbol}/USDT"]
    )


if __name__ == "__main__":
    # Quick test
    config = create_hma_rsi_lr_config('BTC')
    strategy = HmaRsiLrCryptoStrategy(config)
    print(f"Strategy: {strategy.get_name()}")
    print(f"Valid config: {strategy.validate_config()}")
