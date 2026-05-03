"""
Abstract Base Class for trading strategies.
Unified interface for both crypto and commodity strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    asset_class: str  # 'crypto' or 'commodity'
    timeframe: str  # '1d' or '4h' or 'both'
    ichimoku_parameters: Dict[str, int] = field(default_factory=dict)  # tenkan_period, kijun_period, etc.
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)  # Generic strategy params (HMA, RSI, LR, etc.)
    signal_conditions: Dict[str, Any] = field(default_factory=dict)  # buy_conditions, sell_conditions, logic
    risk_management: Dict[str, Any] = field(default_factory=dict)  # stop_loss_pct, take_profit_pct, etc.
    position_sizing: Dict[str, Any] = field(default_factory=dict)  # method, fixed_size, etc.
    description: str = ""
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)


class BaseStrategy(ABC):
    """
    Abstract interface for trading strategies.
    All strategies (crypto YAML-based, commodity class-based) must implement this.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.
        
        Args:
            config: StrategyConfig object with all parameters
        """
        self.config = config
        self.name = config.name
        self.asset_class = config.asset_class
    
    @abstractmethod
    def calculate_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators for this strategy.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus indicator columns
        """
        pass
    
    @abstractmethod
    def generate_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate entry signals based on strategy logic.
        
        Args:
            data: DataFrame with OHLCV + indicator data
            
        Returns:
            Series with values: 1 (long), -1 (short), 0 (no signal)
        """
        pass
    
    @abstractmethod
    def generate_exit_signals(self, data: pd.DataFrame, position: Dict) -> Optional[str]:
        """
        Check exit conditions for current position.
        
        Args:
            data: DataFrame with OHLCV + indicator data (current candle)
            position: Dict with position info (entry_price, entry_time, etc.)
            
        Returns:
            Exit reason string (e.g., 'stoploss', 'take_profit', 'sell_signal')
            or None if no exit signal
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return human-readable strategy name.
        
        Returns:
            Strategy name string
        """
        pass
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate strategy configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required ichimoku parameters
        required_params = ['tenkan_period', 'kijun_period', 'senkou_b_period']
        for param in required_params:
            if param not in self.config.ichimoku_parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Check risk management
        if 'stop_loss_pct' not in self.config.risk_management:
            errors.append("Missing stop_loss_pct in risk_management")
        
        return (len(errors) == 0, errors)
