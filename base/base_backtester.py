"""
Abstract Base Class for backtesting engines.
Unified interface for both crypto and commodity backtesting.
"""

from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    commission: float
    slippage: float
    net_pnl: float
    return_pct: float
    bars_held: int
    entry_reason: str
    exit_reason: str

@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]
    psar_stats: Optional[Dict[str, Any]] = None

class BaseBacktester(ABC):
    """
    Abstract interface for backtesting engines.
    All backtesters must implement this interface for modularity.
    """
    
    def __init__(self, 
                 commission_rate: float = 0.001,  # 0.1%
                 slippage_rate: float = 0.0003,  # 0.03%
                 pyramiding: int = 1):
        """
        Initialize backtesting engine.
        
        Args:
            commission_rate: Commission as decimal (0.001 = 0.1%)
            slippage_rate: Slippage as decimal
            pyramiding: Max simultaneous positions (1 = no pyramiding)
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.pyramiding = pyramiding
        
        # Trading state
        self.initial_capital = 0.0
        self.cash = 0.0
        self.positions: List[Dict] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Trade counter
        self._trade_counter = 0
    
    @abstractmethod
    def run_backtest(self, 
                     strategy: 'BaseStrategy', 
                     data_handler: 'BaseDataHandler', 
                     initial_capital: float = 10000.0) -> BacktestResult:
        """
        Run a complete backtest.
        
        Args:
            strategy: Strategy instance implementing BaseStrategy
            data_handler: Data handler implementing BaseDataHandler
            initial_capital: Starting portfolio value
            
        Returns:
            BacktestResult with comprehensive results
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate performance metrics from trades.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Dict with metrics (win_rate, profit_factor, max_drawdown, etc.)
        """
        pass
    
    def _apply_slippage_and_commission(self, price: float, is_entry: bool) -> float:
        """
        Apply slippage and commission to price.
        
        Args:
            price: Original price
            is_entry: True if entering position, False if exiting
            
        Returns:
            Adjusted price
        """
        if is_entry:
            return price * (1 + self.slippage_rate) * (1 + self.commission_rate)
        else:
            return price * (1 - self.slippage_rate) * (1 - self.commission_rate)
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"trade_{self._trade_counter:04d}"
    
    def _calculate_position_size(self, 
                                capital: float, 
                                price: float, 
                                strategy_config: Dict) -> float:
        """
        Calculate position size based on strategy config.
        
        Args:
            capital: Available capital
            price: Current price
            strategy_config: Strategy configuration dict
            
        Returns:
            Number of units to buy/sell
        """
        method = strategy_config.get('position_sizing', {}).get('method', 'fixed')
        
        if method == 'fixed':
            fixed_size = strategy_config.get('position_sizing', {}).get('fixed_size', 1000)
            return fixed_size / price
        elif method == 'percent':
            max_pct = strategy_config.get('risk_management', {}).get('max_position_size_pct', 100)
            available = capital * (max_pct / 100)
            return available / price
        else:
            # Default: use all capital
            return capital / price
