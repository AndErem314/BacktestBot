"""
Ichimoku Strategy Backtester - Refactored to inherit from BaseBacktester.

Provides a modular backtesting engine for Ichimoku strategies with:
- Support for both crypto and commodity asset classes
- Inherits unified commission/slippage/position sizing from BaseBacktester
- Backward compatible with existing dictionary-based strategy configs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum
import logging
import json
import yaml
from pathlib import Path

# Local imports
from base.base_backtester import BaseBacktester, Trade, BacktestResult
from base.base_strategy import BaseStrategy, StrategyConfig
from base.base_data_handler import BaseDataHandler
from data_fetching.data_manager import DataManager
from strategy.ichimoku_strategy import (
    UnifiedIchimokuAnalyzer,
    IchimokuStrategyConfig,
    IchimokuParameters,
)
from reporting.report_generator import ReportGenerator

# Configure logging
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class IchimokuBacktester(BaseBacktester):
    """
    Refactored Ichimoku backtesting engine inheriting from BaseBacktester.
    
    Features:
    - Fixed position sizing (100% of equity)
    - Pyramiding = 1 (only one position at a time)
    - Commission = 0.1%, Slippage = 0.03%
    - PSAR confirmation (crypto) or simplified (commodity)
    - Asset class aware (crypto/commodity)
    """
    
    def __init__(self,
                 commission_rate: float = 0.001,  # 0.1%
                 slippage_rate: float = 0.0003,  # 0.03%
                 pyramiding: int = 1,
                 asset_class: str = 'crypto'):  # 'crypto' or 'commodity'
        """
        Initialize the backtesting engine.
        
        Args:
            commission_rate: Commission as decimal (0.001 = 0.1%)
            slippage_rate: Slippage as decimal
            pyramiding: Max simultaneous positions (1 = no pyramiding)
            asset_class: 'crypto' or 'commodity' - affects default behavior
        """
        # Call parent init
        super().__init__(commission_rate, slippage_rate, pyramiding)
        
        # Asset class specific settings
        self.asset_class = asset_class
        
        # Strategy reference (for backward compatibility)
        self.strategy_config = None
        
        # PSAR confirmation stats (crypto-specific)
        self._psar_stats = {
            'psar_available': False,
            'raw_long': 0,
            'raw_short': 0,
            'confirmed_long': 0,
            'confirmed_short': 0,
            'filtered_long': 0,
            'filtered_short': 0,
        }
        
        logger.info(f"Ichimoku Backtester initialized for {asset_class} - fixed position sizing")
    
    def run_backtest(self,
                     strategy: Union[BaseStrategy, Dict],
                     data_handler: Optional[BaseDataHandler] = None,
                     initial_capital: float = 10000.0) -> BacktestResult:
        """
        Run a complete backtest - implements BaseBacktester abstract method.
        
        Args:
            strategy: BaseStrategy object or dict (for backward compatibility)
            data_handler: BaseDataHandler for loading data
            initial_capital: Starting portfolio value
            
        Returns:
            BacktestResult with comprehensive results
        """
        # Handle backward compatibility: if strategy is a dict, convert to appropriate format
        if isinstance(strategy, dict):
            return self._run_backtest_from_config(strategy, data_handler, initial_capital)
        
        # Load data from handler if provided
        if data_handler:
            timeframe = '1d'  # Default, should be from strategy config
            data = data_handler.load_ohlcv(timeframe)
            # Load indicators
            data = data_handler.load_indicators('ichimoku', timeframe)
        else:
            raise ValueError("data_handler is required for new interface")
        
        # Convert BaseStrategy to internal format and run
        # TODO: Implement proper BaseStrategy to internal format conversion
        raise NotImplementedError("BaseStrategy interface not yet fully implemented")
    
    def _run_backtest_from_config(self,
                                strategy_config: Dict,
                                data_handler: Optional[BaseDataHandler] = None,
                                initial_capital: float = 10000.0) -> BacktestResult:
        """
        Backward-compatible backtest runner using dict config.
        
        Args:
            strategy_config: Strategy configuration dictionary
            data_handler: Optional data handler (if None, expects data DataFrame)
            initial_capital: Starting capital
            
        Returns:
            BacktestResult
        """
        self.strategy_config = strategy_config
        
        # If data_handler provided, load data from it
        if data_handler:
            timeframe = strategy_config.get('timeframes', ['1d'])[0]
            data = data_handler.load_ohlcv(timeframe)
            if 'ichimoku' in strategy_config.get('signal_conditions', {}).get('buy_conditions', []):
                data = data_handler.load_indicators('ichimoku', timeframe)
        else:
            # Original behavior: data should be passed separately
            raise ValueError("For backward compatibility, use run_backtest_with_data()")
        
        return self._execute_backtest_with_data(strategy_config, data, initial_capital)
    
    def run_backtest_with_data(self,
                              strategy_config: Dict,
                              data: pd.DataFrame,
                              initial_capital: float = 10000.0) -> BacktestResult:
        """
        Run backtest with provided DataFrame (original interface).
        
        Args:
            strategy_config: Strategy configuration dictionary
            data: Market data DataFrame with OHLCV and indicators
            initial_capital: Starting capital
            
        Returns:
            BacktestResult
        """
        self.strategy_config = strategy_config
        
        # Validate data
        if not self._validate_data(data):
            raise ValueError("Data validation failed - missing required columns")
        
        # Reset state
        self._reset()
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Run backtest
        self._execute_trading(data)
        
        # Calculate results
        results = self._calculate_results()
        
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        return results
    
    def _reset(self):
        """Reset all state variables."""
        self.initial_capital = 0
        self.cash = 0
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._trade_counter = 0
        # Reset PSAR stats
        for key in self._psar_stats:
            self._psar_stats[key] = 0
        self._psar_stats['psar_available'] = False
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Ensure price columns
        if not all(col in data.columns for col in required_columns):
            return False
        
        # If signal columns are missing, they will be computed upstream
        return True
    
    def _execute_trading(self, data: pd.DataFrame):
        """Execute trading logic based on strategy signals."""
        in_position = False
        current_position = None
        
        # Determine if PSAR columns are available for confirmation stats
        self._psar_stats['psar_available'] = ('psar_trend' in data.columns) or ('psar_uptrend' in data.columns)
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Update equity curve
            self._update_equity_curve(timestamp, row['close'], current_position)
            
            # Check for signals
            if not in_position:
                # Evaluate LONG
                base_long, confirmed_long = self._evaluate_entry_signal(row, PositionSide.LONG)
                if base_long:
                    self._psar_stats['raw_long'] += 1
                if confirmed_long:
                    self._psar_stats['confirmed_long'] += 1
                    entry_price = self._calculate_fill_price(PositionSide.LONG, is_entry=True, row=row)
                    position_size = self._calculate_position_size(entry_price)
                    if position_size > 0:
                        current_position = self._enter_long(
                            symbol=self.strategy_config['symbols'][0],
                            timestamp=timestamp,
                            price=entry_price,
                            quantity=position_size,
                            reason="long_entry"
                        )
                        in_position = True
                        logger.debug(f"Entered LONG at {entry_price}")
                elif base_long and self._psar_stats['psar_available']:
                    self._psar_stats['filtered_long'] += 1
                
                # If not LONG, evaluate SHORT
                if not in_position:
                    base_short, confirmed_short = self._evaluate_entry_signal(row, PositionSide.SHORT)
                    if base_short:
                        self._psar_stats['raw_short'] += 1
                    if confirmed_short:
                        self._psar_stats['confirmed_short'] += 1
                        entry_price = self._calculate_fill_price(PositionSide.SHORT, is_entry=True, row=row)
                        position_size = self._calculate_position_size(entry_price)
                        if position_size > 0:
                            current_position = self._enter_short(
                                symbol=self.strategy_config['symbols'][0],
                                timestamp=timestamp,
                                price=entry_price,
                                quantity=position_size,
                                reason="short_entry"
                            )
                            in_position = True
                            logger.debug(f"Entered SHORT at {entry_price}")
                    elif base_short and self._psar_stats['psar_available']:
                        self._psar_stats['filtered_short'] += 1
            
            else:
                # Check for exit signal or stop loss
                exit_signal = False
                exit_reason = ""
                
                # Exit on configured signal
                if self._check_exit_signal(row, PositionSide(current_position['side'])):
                    exit_signal = True
                    exit_reason = "signal_exit"
                # Check stop loss
                elif self._check_stop_loss(row, current_position):
                    exit_signal = True
                    exit_reason = "stop_loss"
                
                if exit_signal:
                    side = PositionSide(current_position['side'])
                    exit_price = self._calculate_fill_price(side, is_entry=False, row=row)
                    self._exit_position(
                        timestamp=timestamp,
                        price=exit_price,
                        reason=exit_reason
                    )
                    in_position = False
                    current_position = None
                    logger.debug(f"Exited position at {exit_price} ({exit_reason})")
    
    def _evaluate_entry_signal(self, row: pd.Series, side: PositionSide) -> Tuple[bool, bool]:
        """
        Return (base_ok, confirmed_ok) for entry on this bar.
        base_ok: only configured signal conditions
        confirmed_ok: base_ok AND PSAR confirmation if PSAR columns present (crypto only)
        """
        sc = self.strategy_config.get('signal_conditions', {})
        if side == PositionSide.LONG:
            conditions = sc.get('long_entry_conditions') or sc.get('buy_conditions') or []
            logic = sc.get('long_entry_logic') or sc.get('buy_logic', 'AND')
        else:
            conditions = sc.get('short_entry_conditions')
            if not conditions:
                base = sc.get('long_entry_conditions') or sc.get('buy_conditions') or []
                conditions = self._mirror_conditions(base)
            logic = sc.get('short_entry_logic') or sc.get('long_entry_logic') or sc.get('buy_logic', 'AND')
        
        base_ok = self._check_conditions(row, conditions, logic)
        
        # PSAR confirmation (crypto-specific only)
        if self.asset_class == 'crypto':
            if side == PositionSide.LONG:
                psar_ok = ('psar_uptrend' in row and bool(row['psar_uptrend'])) or ('psar_trend' in row and row['psar_trend'] == 1)
            else:
                psar_ok = ('psar_downtrend' in row and bool(row['psar_downtrend'])) or ('psar_trend' in row and row['psar_trend'] == -1)
            
            if ('psar_uptrend' in row) or ('psar_trend' in row):
                return base_ok, (base_ok and psar_ok)
        
        # For commodities or when PSAR not available: confirmed = base
        return base_ok, base_ok
    
    def _get_signal_mapping(self) -> Dict[str, str]:
        """
        Map config condition names to DataFrame columns.
        For SpanA/ SpanB conditions, we compute them dynamically.
        """
        return {
            'PriceAboveCloud': 'price_above_cloud',
            'PriceBelowCloud': 'price_below_cloud',
            'TenkanAboveKijun': 'tenkan_above_kijun',
            'TenkanBelowKijun': 'tenkan_below_kijun',
            'SpanAaboveSpanB': 'span_a_above_span_b',  # Computed dynamically
            'SpanAbelowSpanB': 'span_a_below_span_b',  # Computed dynamically
            'ChikouAbovePrice': 'chikou_above_price',
            'ChikouBelowPrice': 'chikou_below_price',
            'ChikouAboveCloud': 'chikou_above_cloud',
            'ChikouBelowCloud': 'chikou_below_cloud',
            'PSARUptrend': 'psar_uptrend',
            'PSARDowntrend': 'psar_downtrend',
        }
    
    def _check_conditions(self, row: pd.Series, conditions: List[str], logic: str) -> bool:
        """Generic condition checker with AND/OR logic."""
        if not conditions:
            return False
        
        # Ensure span_a columns exist in the dataframe (accessed via row.index)
        self._ensure_span_columns(row)
        
        mapping = self._get_signal_mapping()
        flags: List[bool] = []
        for cond in conditions:
            col = mapping.get(cond)
            if col and col in row.index:
                flags.append(bool(row[col]))
        
        if not flags:
            return False
        if (logic or 'AND').upper() == 'OR':
            return any(flags)
        return all(flags)
    
    def _ensure_span_columns(self, row: pd.Series):
        """
        Ensure span_a_above_span_b and span_a_below_span_b columns exist.
        These are computed dynamically from senkou_span_a and senkou_span_b.
        Note: This is a simplified version - in reality, we need access to the full DataFrame.
        """
        # This is a placeholder - the actual computation should happen in the data preparation step
        # For now, we'll compute it in the data fetching method
        pass
    
    def _check_exit_signal(self, row: pd.Series, side: PositionSide) -> bool:
        """Check if exit signal conditions are met."""
        sc = self.strategy_config.get('signal_conditions', {})
        if side == PositionSide.LONG:
            conditions = sc.get('long_exit_conditions') or sc.get('sell_conditions') or []
            logic = sc.get('long_exit_logic') or sc.get('sell_logic', 'AND')
        else:
            conditions = sc.get('short_exit_conditions')
            if not conditions:
                base = sc.get('long_exit_conditions') or sc.get('sell_conditions') or []
                conditions = self._mirror_conditions(base)
            logic = sc.get('short_exit_logic') or sc.get('long_exit_logic') or sc.get('sell_logic', 'AND')
        return self._check_conditions(row, conditions, logic)
    
    def _mirror_condition(self, cond: str) -> Optional[str]:
        """Return the opposite condition name for mirroring LONG<->SHORT."""
        mirror_map = {
            'PriceAboveCloud': 'PriceBelowCloud',
            'PriceBelowCloud': 'PriceAboveCloud',
            'TenkanAboveKijun': 'TenkanBelowKijun',
            'TenkanBelowKijun': 'TenkanAboveKijun',
            'SpanAaboveSpanB': 'SpanAbelowSpanB',
            'SpanAbelowSpanB': 'SpanAaboveSpanB',
            'ChikouAbovePrice': 'ChikouBelowPrice',
            'ChikouBelowPrice': 'ChikouAbovePrice',
            'ChikouAboveCloud': 'ChikouBelowCloud',
            'ChikouBelowCloud': 'ChikouAboveCloud',
        }
        return mirror_map.get(cond)
    
    def _mirror_conditions(self, conditions: List[str]) -> List[str]:
        return [self._mirror_condition(c) for c in conditions if self._mirror_condition(c)]
    
    def _check_stop_loss(self, row: pd.Series, position: Dict) -> bool:
        """Check if stop loss condition is met (based on closed bars)."""
        if not position:
            return False
        
        stop_loss_pct = self.strategy_config['risk_management'].get('stop_loss_pct')
        if stop_loss_pct is None:
            return False
        
        current_price = row['close']
        entry_price = position['entry_price']
        side = PositionSide(position.get('side', PositionSide.LONG.value))
        
        if side == PositionSide.LONG:
            stop_price = entry_price * (1 - stop_loss_pct / 100)
            return current_price <= stop_price
        else:
            stop_price = entry_price * (1 + stop_loss_pct / 100)
            return current_price >= stop_price
    
    def _calculate_fill_price(self, side: PositionSide, is_entry: bool, row: pd.Series) -> float:
        """Calculate trade fill price with slippage for LONG/SHORT entries and exits."""
        base_price = row['close']
        slippage = base_price * self.slippage_rate
        if side == PositionSide.LONG:
            return base_price + slippage if is_entry else base_price - slippage
        elif side == PositionSide.SHORT:
            return base_price - slippage if is_entry else base_price + slippage
        else:
            return base_price
    
    def _calculate_position_size(self, entry_price: float) -> float:
        """Calculate position size based on strategy config."""
        if self.cash <= 0:
            return 0
        
        method = self.strategy_config.get('position_sizing', {}).get('method', 'fixed')
        
        if method == 'fixed':
            fixed_size = self.strategy_config.get('position_sizing', {}).get('fixed_size', 1000)
            return fixed_size / entry_price
        else:
            # Default: use all capital
            return self.cash / entry_price
    
    def _enter_long(self, symbol: str, timestamp: datetime, price: float,
                    quantity: float, reason: str) -> Dict:
        """Enter a long position."""
        trade_value = price * quantity
        commission = trade_value * self.commission_rate
        self.cash -= (trade_value + commission)
        position = {
            'symbol': symbol,
            'side': PositionSide.LONG.value,
            'entry_time': timestamp,
            'entry_price': price,
            'quantity': quantity,
            'commission_paid': commission,
            'entry_reason': reason
        }
        self.positions[symbol] = position
        return position
    
    def _enter_short(self, symbol: str, timestamp: datetime, price: float,
                     quantity: float, reason: str) -> Dict:
        """Enter a short position."""
        trade_value = price * quantity
        commission = trade_value * self.commission_rate
        # Receive proceeds from short sale minus commission
        self.cash += (trade_value - commission)
        position = {
            'symbol': symbol,
            'side': PositionSide.SHORT.value,
            'entry_time': timestamp,
            'entry_price': price,
            'quantity': quantity,
            'commission_paid': commission,
            'entry_reason': reason
        }
        self.positions[symbol] = position
        return position
    
    def _exit_position(self, timestamp: datetime, price: float, reason: str):
        """Exit current position."""
        if not self.positions:
            return
        
        symbol = list(self.positions.keys())[0]
        position = self.positions[symbol]
        side = PositionSide(position.get('side', PositionSide.LONG.value))
        
        # Calculate exit values
        exit_value = price * position['quantity']
        commission_exit = exit_value * self.commission_rate
        
        # Update cash and compute P&L based on side
        entry_value = position['entry_price'] * position['quantity']
        commission_entry = position['commission_paid']
        
        if side == PositionSide.LONG:
            net_proceeds = exit_value - commission_exit
            self.cash += net_proceeds
            gross_pnl = exit_value - entry_value
        else:  # SHORT
            # Buy-to-cover reduces cash
            self.cash -= (exit_value + commission_exit)
            gross_pnl = entry_value - exit_value
        
        total_commission = commission_entry + commission_exit
        net_pnl = gross_pnl - total_commission
        return_pct = (net_pnl / entry_value) * 100 if entry_value != 0 else 0.0
        
        # Calculate slippage (approximate, symmetric)
        base_price = (position['entry_price'] + price) / 2
        slippage_amount = base_price * self.slippage_rate * position['quantity']
        
        # Calculate bars held
        bars_held = len(self.equity_curve) - self._find_entry_bar_index(position['entry_time'])
        
        # Record trade using base class Trade dataclass
        trade = Trade(
            trade_id=self._generate_trade_id(),
            symbol=symbol,
            entry_time=position['entry_time'],
            exit_time=timestamp,
            side=side,
            entry_price=position['entry_price'],
            exit_price=price,
            quantity=position['quantity'],
            commission=total_commission,
            slippage=slippage_amount,
            net_pnl=net_pnl,
            return_pct=return_pct,
            bars_held=bars_held,
            entry_reason=position['entry_reason'],
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # Remove position
        self.positions.clear()
        
        logger.info(
            f"Trade closed: {symbol} | PnL: ${net_pnl:.2f} ({return_pct:.2f}%) | "
            f"Reason: {reason}"
        )
    
    def _find_entry_bar_index(self, entry_time: datetime) -> int:
        """Find the equity curve index for entry time."""
        for i, point in enumerate(self.equity_curve):
            if point['timestamp'] == entry_time:
                return i
        return 0
    
    def _update_equity_curve(self, timestamp: datetime, price: float, position: Optional[Dict]):
        """Update equity curve with current portfolio value."""
        position_value = 0.0
        if position:
            qty = position['quantity']
            side = PositionSide(position.get('side', PositionSide.LONG.value))
            if side == PositionSide.LONG:
                position_value = price * qty
            else:  # SHORT
                position_value = -price * qty
        
        total_value = self.cash + position_value
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'price': price
        })
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        if not self.trades:
            return self._empty_results()
        
        # Get metrics dict from calculate_metrics (call on self, not super())
        metrics = self.calculate_metrics(self.trades)
        
        if not metrics:
            return self._empty_results()
        
        # Create BacktestResult from metrics
        return BacktestResult(
            total_trades=metrics.get('total_trades', 0),
            winning_trades=metrics.get('winning_trades', 0),
            losing_trades=metrics.get('losing_trades', 0),
            total_return_pct=metrics.get('total_return_pct', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            max_drawdown_pct=metrics.get('max_drawdown_pct', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            trades=self.trades,
            equity_curve=pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame(),
            metrics=metrics,
            psar_stats=self._psar_stats if hasattr(self, '_psar_stats') else None
        )
    
    def calculate_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """Implement abstract method from BaseBacktester."""
        if not trades:
            return {}
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0  # Returns decimal (0-1)
        
        # Calculate profit factor
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate total return
        final_equity = self.equity_curve[-1]['total_value'] if self.equity_curve else self.initial_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate max drawdown
        equity_values = [point['total_value'] for point in self.equity_curve]
        max_drawdown_pct = self._calculate_max_drawdown(equity_values)
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(trades_df)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,  # Already decimal (0-1), don't multiply by 100
            'profit_factor': profit_factor,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve. Returns decimal (0-1)."""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd  # Return decimal (0-1), NOT percentage
    
    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from trade returns."""
        if len(trades_df) < 2:
            return 0.0
        
        # Use trade returns to calculate Sharpe
        returns = trades_df['return_pct'] / 100  # Convert to decimal
        
        if returns.std() == 0:
            return 0.0
        
        # Annualize assuming 252 trading days
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        return sharpe
    
    def _empty_results(self) -> BacktestResult:
        """Return empty results when no trades occurred."""
        final_equity = self.equity_curve[-1]['total_value'] if self.equity_curve else self.initial_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_return_pct=total_return_pct,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            trades=[],
            equity_curve=pd.DataFrame(),
            metrics={
                'total_return_pct': total_return_pct,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'net_profit': 0.0
            }
        )
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"TRD_{self._trade_counter:06d}"


# Strategy JSON Integration and utilities
class StrategyBacktestRunner:
    """
    Helper class to run backtests from strategy JSON/YAML configurations.
    Updated to work with refactored IchimokuBacktester.
    Handles both crypto and commodity asset classes.
    """
    
    def __init__(self, backtester: Optional[IchimokuBacktester] = None):
        """
        Initialize runner. If backtester is not provided, it will be created
        dynamically based on the strategy's asset_class.
        """
        self.backtester = backtester
    
    def run_strategy_backtest(self, strategy_config: Dict,
                              data: pd.DataFrame,
                              initial_capital: float = 10000.0) -> BacktestResult:
        """
        Run backtest for a strategy from JSON/YAML configuration.
        
        Args:
            strategy_config: Strategy configuration from JSON/YAML
            data: Market data with Ichimoku signals
            initial_capital: Initial capital for backtest
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest for: {strategy_config['name']}")
        
        # Validate strategy configuration
        if not self._validate_strategy_config(strategy_config):
            raise ValueError("Invalid strategy configuration")
        
        # Determine asset class from strategy config (default to 'crypto')
        asset_class = strategy_config.get('asset_class', 'crypto')
        
        # Create or update backtester based on asset_class
        if self.backtester is None or self.backtester.asset_class != asset_class:
            logger.info(f"Creating new backtester for asset_class: {asset_class}")
            self.backtester = IchimokuBacktester(asset_class=asset_class)
        
        # Run backtest using refactored interface
        result = self.backtester.run_backtest_with_data(
            strategy_config=strategy_config,
            data=data,
            initial_capital=initial_capital
        )
        
        return result
    
    def _validate_strategy_config(self, config: Dict) -> bool:
        """Validate strategy configuration."""
        required_fields = [
            'name', 'symbols', 'signal_conditions',
            'ichimoku_parameters', 'risk_management', 'position_sizing'
        ]
        
        if not all(field in config for field in required_fields):
            logger.error("Missing required fields in strategy configuration")
            return False
        
        # Validate signal conditions (support legacy buy/sell and new long/short)
        signal_conditions = config['signal_conditions']
        has_legacy = 'buy_conditions' in signal_conditions and 'sell_conditions' in signal_conditions
        has_directional = ('long_entry_conditions' in signal_conditions and 
                           'long_exit_conditions' in signal_conditions)
        if not (has_legacy or has_directional):
            logger.error("Missing entry/exit conditions in signal_conditions")
            return False
        
        return True
    
    # ---------- Convenience high-level helpers ----------
    def load_strategy_from_json(self, strategy_key: str,
                                json_path: Optional[Union[str, Path]] = None) -> Dict:
        """Load a single strategy configuration by key from strategies.yaml."""
        candidates: List[Path] = []
        if json_path:
            candidates.append(Path(json_path))
        # Preferred path as per user instruction
        base = Path(__file__).resolve().parents[1]
        candidates.extend([
            # Prefer YAML first
            base / 'config' / 'strategies.yaml',
            base / 'config' / 'strategies.json',
            base / 'strategy' / 'config' / 'strategies.yaml',
            base / 'strategy' / 'config' / 'strategies.json',
        ])
        
        data: Dict[str, Any] = {}
        file_found = None
        for p in candidates:
            if p.exists():
                try:
                    with open(p, 'r') as f:
                        if p.suffix == '.json':
                            data = json.load(f)
                        else:
                            data = yaml.safe_load(f)
                    file_found = p
                    break
                except Exception:
                    continue
        if not data:
            raise FileNotFoundError("No valid strategies file found in expected locations (json or yaml)")
        
        strategies = data.get('strategies', {})
        if strategy_key not in strategies:
            raise KeyError(f"Strategy key '{strategy_key}' not found in {file_found}")
        return strategies[strategy_key]
    
    def fetch_sql_data_with_signals(self, symbol_short: str, timeframe: str,
                                    start: Optional[str] = None,
                                    end: Optional[str] = None,
                                    ichimoku_params: Optional[Dict[str, Any]] = None,
                                    force_recompute: bool = False) -> pd.DataFrame:
        """Fetch OHLCV+Ichimoku and add boolean signals."""
        dm = DataManager(symbol=symbol_short)
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) if end else None
        
        analyzer = UnifiedIchimokuAnalyzer()
        params = IchimokuStrategyConfig.create_parameters(**(ichimoku_params or {}))
        
        if force_recompute:
            # Load raw OHLCV only and recompute all components with strategy params
            df = dm.get_ohlcv_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
            dm.close_connection()
            if df.empty:
                return df
            df = analyzer.calculate_ichimoku_components(df, params)
            df = analyzer.detect_boolean_signals(df, params)
            # Compute PSAR in-memory for confirmation
            try:
                from strategy.psar_indicator import compute_psar
                psar_df = compute_psar(df[['high','low','close']])
                df['psar'] = psar_df['psar']
                df['psar_trend'] = psar_df['psar_trend']
                df['psar_reversal'] = psar_df['psar_reversal']
            except Exception:
                pass
            # Derive boolean PSAR signals on closed bars
            if 'psar_trend' in df.columns:
                closed_mask = df.index.to_series().notna()
                if len(df) > 0:
                    closed_mask.iloc[-1] = False
                df['psar_uptrend'] = (df['psar_trend'] == 1) & closed_mask
                df['psar_downtrend'] = (df['psar_trend'] == -1) & closed_mask
            return df
        
        # Default path: use SQL ichimoku view if present, otherwise OHLCV and compute missing
        try:
            df = dm.get_ichimoku_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
        except Exception:
            df = dm.get_ohlcv_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
        dm.close_connection()
        if df.empty:
            return df
        
        # If Ichimoku components are not present, compute them from price data using analyzer
        if not set(['tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b','chikou_span']).issubset(df.columns):
            df = analyzer.calculate_ichimoku_components(df, params)
        # Add boolean signals
        df = analyzer.detect_boolean_signals(df, params)
        
        # Ensure PSAR columns present; if not, compute in-memory
        psar_present = 'psar' in df.columns
        if not psar_present:
            try:
                from strategy.psar_indicator import compute_psar
                psar_df = compute_psar(df[['high','low','close']])
                df['psar'] = psar_df['psar']
                df['psar_trend'] = psar_df['psar_trend']
                df['psar_reversal'] = psar_df['psar_reversal']
            except Exception:
                pass
        # Derive boolean PSAR signals on closed bars
        if 'psar_trend' in df.columns:
            closed_mask = df.index.to_series().notna()
            if len(df) > 0:
                closed_mask.iloc[-1] = False
            df['psar_uptrend'] = (df['psar_trend'] == 1) & closed_mask
            df['psar_downtrend'] = (df['psar_trend'] == -1) & closed_mask
        
        # Compute SpanA vs SpanB boolean columns for signal evaluation
        if 'senkou_span_a' in df.columns and 'senkou_span_b' in df.columns:
            df['span_a_above_span_b'] = df['senkou_span_a'] > df['senkou_span_b']
            df['span_a_below_span_b'] = df['senkou_span_a'] < df['senkou_span_b']
        
        return df
    
    def run_from_json(self, strategy_key: str, symbol_short: str, timeframe: str,
                       start: Optional[str] = None, end: Optional[str] = None,
                       initial_capital: float = 10000.0,
                       report_formats: str = 'pdf',
                       output_dir: str = 'results',
                       with_llm_optimization: bool = False,
                       llm_provider: Optional[str] = None,
                       analysis_start: Optional[str] = None,
                       analysis_end: Optional[str] = None,
                       llm_model_override: Optional[str] = None,
                       prompt_variant: str = 'analyst',
                       force_recompute_ichimoku: bool = False) -> Dict[str, Any]:
        """Load strategy by key, fetch data, run backtest, and generate report."""
        strategy_config = self.load_strategy_from_json(strategy_key)
        data = self.fetch_sql_data_with_signals(symbol_short, timeframe,
                                                start, end,
                                                strategy_config.get('ichimoku_parameters'),
                                                force_recompute=force_recompute_ichimoku)
        if data.empty:
            raise ValueError("No data available for backtest")
        
        result = self.run_strategy_backtest(strategy_config, data, initial_capital)
        
        # Prepare structures for reporting
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades]) if result.trades else pd.DataFrame()
        # Normalize columns for reporting
        if not trades_df.empty:
            if 'net_pnl' in trades_df.columns:
                trades_df['pnl'] = trades_df['net_pnl']
            if 'side' in trades_df.columns:
                trades_df['direction'] = trades_df['side'].apply(lambda s: s.value if hasattr(s, 'value') else str(s).lower())
            if 'entry_reason' in trades_df.columns:
                trades_df['entry_signal'] = trades_df['entry_reason']
        
        equity_df = result.equity_curve.copy() if isinstance(result.equity_curve, pd.DataFrame) else pd.DataFrame(result.equity_curve)
        
        # Map backtester metrics into the format expected by ReportGenerator
        perf_metrics = {
            'total_return': (result.metrics.get('total_return_pct', 0) or 0) / 100.0,
            'sharpe_ratio': result.metrics.get('sharpe_ratio', 0.0) or 0.0,
            'max_drawdown': (result.metrics.get('max_drawdown_pct', 0) or 0) / 100.0,
            'win_rate': (result.metrics.get('win_rate', 0) or 0) / 100.0,
            'profit_factor': result.metrics.get('profit_factor', 0.0) if isinstance(result.metrics.get('profit_factor'), (int, float)) and result.metrics.get('profit_factor') > 0 else 0.0,
            'total_trades': result.metrics.get('total_trades', len(result.trades) if isinstance(result.trades, list) else 0),
        }
        
        return {
            'result': result,
            'trades_df': trades_df,
            'equity_df': equity_df,
            'perf_metrics': perf_metrics
        }
    
    def generate_llm_optimization_report(self, *,
                                         result: BacktestResult,
                                         data_df: pd.DataFrame,
                                         trades_df: pd.DataFrame,
                                         equity_df: pd.DataFrame,
                                         strategy_config: Dict[str, Any],
                                         output_dir: str,
                                         symbol_short: str,
                                         timeframe: str,
                                         analysis_start: Optional[str] = None,
                                         analysis_end: Optional[str] = None,
                                         llm_provider: Optional[str] = None,
                                         llm_model_override: Optional[str] = None,
                                         prompt_variant: str = 'analyst') -> Optional[str]:
        """Generate an LLM-only optimization PDF using existing backtest artifacts."""
        try:
            from llm_analysis import (
                load_llm_config, LLMClient, build_llm_payload, build_prompt,
                parse_llm_output, build_final_text, write_llm_pdf
            )
            payload = build_llm_payload(
                result_metrics=result.metrics,
                trades_df=trades_df,
                equity_df=equity_df,
                strategy_config=strategy_config,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
                budget='standard'
            )
            prompt = build_prompt(payload, variant=prompt_variant)
            cfg = load_llm_config()
            client = LLMClient(cfg)
            
            # Determine effective provider/model
            effective_provider = (llm_provider or cfg.provider or 'openai').lower()
            if effective_provider not in ('openai', 'gemini'):
                effective_provider = 'openai' if cfg.openai_api_key else 'gemini'
            if effective_provider == 'openai':
                effective_model = llm_model_override or cfg.openai_model or 'gpt-4o-mini'
            else:
                effective_model = llm_model_override or cfg.gemini_model or 'gemini-2.5-pro'
            
            # Token counts for prompt and (later) output
            try:
                from llm_analysis.token_utils import count_tokens as _count_tokens
                prompt_tokens = _count_tokens(prompt, effective_provider, effective_model)
            except Exception:
                prompt_tokens = 0
            
            raw = client.generate(prompt, provider=llm_provider, model_override=llm_model_override)
            
            try:
                from llm_analysis.token_utils import count_tokens as _count_tokens
                output_tokens = _count_tokens(raw or '', effective_provider, effective_model)
            except Exception:
                output_tokens = 0
            
            json_obj, memo = parse_llm_output(raw)
            title = 'Strategy Settings Optimization — Executive Summary' if prompt_variant == 'analyst' else 'Risk-Focused Optimization'
            final_text = build_final_text(title, json_obj, memo)
            
            # Prepend usage header
            usage_header = (
                f"Provider: {effective_provider} | Model: {effective_model} | "
                f"Prompt tokens: {prompt_tokens} | Output tokens: {output_tokens} | Total: {prompt_tokens + output_tokens}"
            )
            final_text = usage_header + "\n\n" + final_text
            
            # Optionally write optimized YAML config
            try:
                opt_yaml = self._write_llm_optimized_yaml(
                    base_strategy_config=strategy_config,
                    llm_json=json_obj,
                    symbol_short=symbol_short,
                    timeframe=timeframe,
                    output_dir=output_dir
                )
            except Exception:
                opt_yaml = None
            
            pdf_path = write_llm_pdf(
                output_dir=output_dir,
                filename_prefix=f"{symbol_short}_{timeframe}",
                title=title,
                text_body=final_text
            )
            return pdf_path
        except Exception as e:
            logger.error(f"LLM optimization generation failed: {e}")
            return None
    
    def _write_llm_optimized_yaml(self, *, base_strategy_config: Dict[str, Any], llm_json: Dict[str, Any], symbol_short: str, timeframe: str, output_dir: Union[str, Path]) -> Optional[str]:
        """Build and write an optimized strategy YAML based on LLM JSON suggestions."""
        try:
            import yaml
        except Exception:
            return None
        from pathlib import Path as _P
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        key_base = base_strategy_config.get('name', 'strategy').lower().replace(' ', '_')
        strat_key = f"{key_base}_llm_{symbol_short.lower()}_{timeframe}"
        
        # Start from the original strategy config
        sc = json.loads(json.dumps(base_strategy_config)) if isinstance(base_strategy_config, dict) else {}
        
        pc = (llm_json or {}).get('parameter_changes', {}) if isinstance(llm_json, dict) else {}
        # Ichimoku params
        ichi = pc.get('ichimoku', {}) or {}
        if 'ichimoku_parameters' not in sc:
            sc['ichimoku_parameters'] = {}
        for p in ('tenkan_period','kijun_period','senkou_b_period','chikou_offset','senkou_offset'):
            if p in ichi and isinstance(ichi[p], dict) and 'suggested' in ichi[p] and isinstance(ichi[p]['suggested'], (int, float)):
                sc['ichimoku_parameters'][p] = int(ichi[p]['suggested']) if isinstance(ichi[p]['suggested'], float) and p.endswith('period') else ichi[p]['suggested']
        
        # Signal logic and conditions
        sl = pc.get('signal_logic', {}) or {}
        if 'signal_conditions' not in sc:
            sc['signal_conditions'] = {'buy_conditions': [], 'sell_conditions': [], 'buy_logic': 'AND', 'sell_logic': 'AND'}
        if 'buy_logic' in sl and isinstance(sl['buy_logic'], dict):
            sc['signal_conditions']['buy_logic'] = sl['buy_logic'].get('suggested', sc['signal_conditions'].get('buy_logic', 'AND'))
        if 'sell_logic' in sl and isinstance(sl['sell_logic'], dict):
            sc['signal_conditions']['sell_logic'] = sl['sell_logic'].get('suggested', sc['signal_conditions'].get('sell_logic', 'AND'))
        add_conditions = sl.get('add_conditions', []) or []
        remove_conditions = set(sl.get('remove_conditions', []) or [])
        if isinstance(add_conditions, list):
            bc = list(sc['signal_conditions'].get('buy_conditions', []))
            for c in add_conditions:
                if c not in bc:
                    bc.append(c)
            sc['signal_conditions']['buy_conditions'] = bc
        if remove_conditions:
            for list_name in ('buy_conditions','sell_conditions'):
                cur = [c for c in sc['signal_conditions'].get(list_name, []) if c not in remove_conditions]
                sc['signal_conditions'][list_name] = cur
        
        # Risk management
        rm = pc.get('risk_management', {}) or {}
        if 'risk_management' not in sc:
            sc['risk_management'] = {}
        for k in ('stop_loss_pct','take_profit_pct'):
            v = rm.get(k)
            if isinstance(v, dict) and 'suggested' in v and isinstance(v['suggested'], (int, float)):
                sc['risk_management'][k] = float(v['suggested'])
        
        # Symbols and timeframe harmonization
        sc['symbols'] = sc.get('symbols') or [f"{symbol_short}/USDT"]
        if isinstance(sc.get('timeframes'), list):
            if timeframe not in sc['timeframes']:
                sc['timeframes'].append(timeframe)
        else:
            sc['timeframes'] = [timeframe]
        
        # Build final YAML
        out = {'strategies': {strat_key: sc}}
        
        out_dir = _P(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{strat_key}_{ts}.yaml"
        with open(out_path, 'w') as f:
            yaml.safe_dump(out, f, sort_keys=False)
        return str(out_path)
