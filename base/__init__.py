"""
Base package for BacktestBot refactoring.

Provides Abstract Base Classes (ABCs) for unified crypto & commodity backtesting:
- BaseDataHandler: Data loading/persistence interface
- BaseStrategy: Trading strategy interface
- BaseBacktester: Core backtesting engine interface
"""

from .base_data_handler import BaseDataHandler
from .base_strategy import BaseStrategy, StrategyConfig
from .base_backtester import BaseBacktester, Trade, BacktestResult

__all__ = ['BaseDataHandler', 'BaseStrategy', 'StrategyConfig', 'BaseBacktester', 'Trade', 'BacktestResult']
