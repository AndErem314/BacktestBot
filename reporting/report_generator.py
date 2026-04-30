"""
Professional Reporting System for Trading Strategy Backtests

This module provides comprehensive report generation in multiple formats:
- PDF reports with charts and analysis
- JSON/CSV data exports
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Template
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    print("Warning: pdfkit not installed. PDF generation will be disabled.")
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ReportGenerator:
    """
    Professional report generator for trading strategy backtests.
Supports PDF and data export formats.
    """
    
    def __init__(self, output_dir: str = "frontend/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report metadata
        self.report_metadata = {
            'generated_at': datetime.now(),
            'generator_version': '1.0.0',
            'company': 'Trading Bot Analytics'
        }
        
    def generate_backtest_report(self, 
                               results: Dict[str, Any],
                               format: str = 'all',
                               filename_prefix: str = 'backtest_report') -> Dict[str, str]:
        """
        Generate comprehensive backtest report in specified format(s).
        
        Args:
            results: Backtest results dictionary containing:
                - data: Price data DataFrame
                - trades: Trades DataFrame
                - equity_curve: Equity curve DataFrame
                - metrics: Performance metrics
                - strategy_config: Strategy configuration
            format: Report format ('pdf', 'json', 'csv', 'all')
            filename_prefix: Prefix for output files
            
        Returns:
            Dictionary with paths to generated reports
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        generated_reports = {}
        
        # Generate reports based on format
        if format in ['pdf', 'all']:
            pdf_path = self._generate_pdf_report(results, base_filename)
            generated_reports['pdf'] = pdf_path
            
            
        if format in ['json', 'all']:
            json_path = self._export_json(results, base_filename)
            generated_reports['json'] = json_path
            
        if format in ['csv', 'all']:
            csv_paths = self._export_csv(results, base_filename)
            generated_reports['csv'] = csv_paths
            
        return generated_reports
        
    def create_executive_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create executive summary with key insights and recommendations.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Executive summary dictionary
        """
        # Extract key metrics
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        total_trades = metrics.get('total_trades', 0)
        
        # Performance assessment
        performance_rating = self._assess_performance(metrics)
        
        # Risk assessment
        risk_assessment = self._assess_risk(metrics)
        
        # Generate insights
        insights = self._generate_insights(metrics)
        
        # Create summary
        executive_summary = {
            'overview': {
                'total_return': f"{total_return:.2%}",
                'sharpe_ratio': f"{sharpe_ratio:.2f}",
                'max_drawdown': f"{max_drawdown:.2%}",
                'win_rate': f"{win_rate:.2%}",
                'profit_factor': f"{profit_factor:.2f}",
                'total_trades': total_trades
            },
            'performance_rating': performance_rating,
            'risk_assessment': risk_assessment,
            'key_insights': insights,
            'recommendations': self._generate_recommendations(metrics, performance_rating, risk_assessment)
        }
        
        return executive_summary
        
    def _assess_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall strategy performance"""
        sharpe = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Performance scoring
        if sharpe > 2 and total_return > 0.5 and win_rate > 0.6:
            rating = "Excellent"
            score = 5
        elif sharpe > 1.5 and total_return > 0.3 and win_rate > 0.55:
            rating = "Good"
            score = 4
        elif sharpe > 1 and total_return > 0.15 and win_rate > 0.5:
            rating = "Satisfactory"
            score = 3
        elif sharpe > 0.5 and total_return > 0:
            rating = "Below Average"
            score = 2
        else:
            rating = "Poor"
            score = 1
            
        return {
            'rating': rating,
            'score': score,
            'details': {
                'sharpe_rating': 'Excellent' if sharpe > 2 else 'Good' if sharpe > 1 else 'Poor',
                'return_rating': 'Excellent' if total_return > 0.5 else 'Good' if total_return > 0.2 else 'Poor',
                'consistency_rating': 'Excellent' if win_rate > 0.6 else 'Good' if win_rate > 0.5 else 'Poor'
            }
        }
        
    def _assess_risk(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess strategy risk profile"""
        max_dd = abs(metrics.get('max_drawdown', 0))
        avg_loss = abs(metrics.get('avg_loss', 0))
        loss_std = metrics.get('loss_std', 0)
        var_95 = abs(metrics.get('var_95', 0))
        
        # Risk scoring
        if max_dd < 0.1 and var_95 < 0.02:
            risk_level = "Low"
            risk_score = 1
        elif max_dd < 0.2 and var_95 < 0.03:
            risk_level = "Moderate"
            risk_score = 2
        elif max_dd < 0.3 and var_95 < 0.05:
            risk_level = "High"
            risk_score = 3
        else:
            risk_level = "Very High"
            risk_score = 4
            
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'max_drawdown': f"{max_dd:.2%}",
            'value_at_risk_95': f"{var_95:.2%}",
            'risk_metrics': {
                'drawdown_severity': 'Severe' if max_dd > 0.3 else 'Moderate' if max_dd > 0.15 else 'Mild',
                'loss_consistency': 'High Variance' if loss_std > avg_loss * 2 else 'Moderate Variance' if loss_std > avg_loss else 'Low Variance',
                'tail_risk': 'High' if var_95 > 0.05 else 'Moderate' if var_95 > 0.03 else 'Low'
            }
        }
        
    def _generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate key insights from metrics"""
        insights = []
        
        # Performance insights
        if metrics.get('sharpe_ratio', 0) > 1.5:
            insights.append("Strategy shows strong risk-adjusted returns with Sharpe ratio > 1.5")
        elif metrics.get('sharpe_ratio', 0) < 0.5:
            insights.append("Low Sharpe ratio indicates poor risk-adjusted performance")
            
        # Win rate insights
        win_rate = metrics.get('win_rate', 0)
        if win_rate > 0.6:
            insights.append(f"High win rate of {win_rate:.1%} suggests consistent strategy")
        elif win_rate < 0.4:
            insights.append(f"Low win rate of {win_rate:.1%} indicates need for entry signal refinement")
            
        # Profit factor insights
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor > 2:
            insights.append(f"Excellent profit factor of {profit_factor:.2f} shows strong edge")
        elif profit_factor < 1.2:
            insights.append("Profit factor near 1.0 suggests marginal profitability")
            
        # Drawdown insights
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd > 0.25:
            insights.append(f"High maximum drawdown of {max_dd:.1%} poses significant risk")
        elif max_dd < 0.1:
            insights.append("Low drawdown indicates good capital preservation")
            
        # Trading frequency
        total_trades = metrics.get('total_trades', 0)
        avg_trades_per_month = total_trades / max(1, metrics.get('months_traded', 12))
        if avg_trades_per_month < 2:
            insights.append("Low trading frequency may limit profit potential")
        elif avg_trades_per_month > 20:
            insights.append("High trading frequency increases transaction cost impact")

        # PSAR confirmation insights (if present)
        if 'psar_confirmation_rate_long' in metrics or 'psar_confirmation_rate_short' in metrics:
            rl = metrics.get('psar_signals_raw_long') or 0
            cl = metrics.get('psar_signals_confirmed_long') or 0
            rs = metrics.get('psar_signals_raw_short') or 0
            cs = metrics.get('psar_signals_confirmed_short') or 0
            rll = metrics.get('psar_confirmation_rate_long')
            rls = metrics.get('psar_confirmation_rate_short')
            msg = f"PSAR confirmation: long {cl}/{rl} ({rll:.1f}% if rll else 0%), short {cs}/{rs} ({rls:.1f}% if rls else 0%)"
            try:
                insights.append(msg)
            except Exception:
                pass
            
        return insights
        
    def _generate_recommendations(self, 
                                metrics: Dict[str, Any],
                                performance: Dict[str, Any],
                                risk: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if performance['score'] < 3:
            recommendations.append("Consider optimizing entry signals for better performance")
            recommendations.append("Review and adjust Ichimoku parameters using walk-forward analysis")
            
        # Risk-based recommendations
        if risk['risk_score'] > 2:
            recommendations.append("Implement tighter stop-loss levels to reduce drawdown")
            recommendations.append("Consider position sizing based on volatility")
            
        # Win rate recommendations
        if metrics.get('win_rate', 0) < 0.45:
            recommendations.append("Add confirmation signals to improve entry accuracy")
            recommendations.append("Consider trend strength filters to avoid false signals")
            
        # Profit factor recommendations
        if metrics.get('profit_factor', 0) < 1.5:
            recommendations.append("Optimize take-profit levels to improve risk/reward ratio")
            recommendations.append("Consider trailing stops to capture larger trends")
            
        # General recommendations
        recommendations.append("Validate strategy on out-of-sample data before live trading")
        recommendations.append("Monitor strategy performance and adjust for changing market conditions")
        
        return recommendations
        
    def _generate_pdf_report(self, results: Dict[str, Any], base_filename: str) -> str:
        """Generate PDF report with charts and analysis"""
        pdf_path = self.output_dir / f"{base_filename}.pdf"
        
        # Create PDF with multiple pages
        with PdfPages(pdf_path) as pdf:
            # Page 1: Title and Executive Summary
            self._create_title_page(pdf, results)
            
            # Page 2: Performance Overview
            self._create_performance_overview_page(pdf, results)
            
            # Page 3: Trading Analysis (includes Underwater Chart)
            self._create_trading_analysis_page(pdf, results)

            # Pages 4-5+: Trade Details (Chronological by Entry Time, per side)
            # This function will emit one or two pages depending on available directions.
            self._create_trade_details_page(pdf, results)
            
        # Save PDF metadata
        d = pdf.infodict()
        strategy_config = results.get('strategy_config', {})
        strategy_name = strategy_config.get('name', 'Strategy')
        asset_class = strategy_config.get('asset_class', 'crypto').upper()
        symbol = strategy_config.get('symbol', 'N/A')
        
        d['Title'] = f'Backtest Report - {strategy_name} ({asset_class})'
        d['Author'] = 'Trading Bot Analytics'
        d['Subject'] = f'Trading Strategy Backtest Report - {asset_class} - {symbol}'
        d['Keywords'] = f'Trading, Backtest, Ichimoku, Performance, {asset_class}'
        d['CreationDate'] = datetime.now()
            
        return str(pdf_path)
        
    def _create_title_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create title page with executive summary"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Trading Strategy Backtest Report', fontsize=24, y=0.95)
        
        # Get executive summary
        metrics = results.get('metrics', {}).get('performance_metrics', {})
        summary = self.create_executive_summary(metrics)
        
        # Strategy info
        strategy_config = results.get('strategy_config', {})
        strategy_name = strategy_config.get('name', 'Ichimoku Strategy')
        trading_pair = strategy_config.get('symbol', 'N/A')
        timeframe = strategy_config.get('timeframe', 'N/A')
        asset_class = strategy_config.get('asset_class', 'crypto').upper()
        
        # Create text content
        text_content = f"""
Strategy: {strategy_name}
Asset Class: {asset_class}
Trading Pair: {trading_pair}
Timeframe: {timeframe}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
{'='*50}

Performance Overview:
• Total Return: {summary['overview']['total_return']}
• Sharpe Ratio: {summary['overview']['sharpe_ratio']}
• Maximum Drawdown: {summary['overview']['max_drawdown']}
• Win Rate: {summary['overview']['win_rate']}
• Profit Factor: {summary['overview']['profit_factor']}
• Total Trades: {summary['overview']['total_trades']}

Performance Rating: {summary['performance_rating']['rating']} ({summary['performance_rating']['score']}/5)
Risk Level: {summary['risk_assessment']['risk_level']}

Key Insights:
"""
        for i, insight in enumerate(summary['key_insights'][:5], 1):
            text_content += f"{i}. {insight}\n"
            
        # Add text to figure
        plt.text(0.1, 0.85, text_content, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_performance_overview_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create performance overview page with charts"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Performance Overview', fontsize=18, y=0.98)
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        equity_curve = results.get('equity_curve', pd.DataFrame())
        if not equity_curve.empty:
            ax1.plot(equity_curve.index, equity_curve['equity'], 'b-', linewidth=2)
            ax1.fill_between(equity_curve.index, equity_curve['equity'], 
                           equity_curve['equity'].iloc[0], alpha=0.3)
            ax1.set_title('Portfolio Equity Curve')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            
        # 2. Monthly returns
        ax2 = fig.add_subplot(gs[1, 0])
        if 'returns' in equity_curve.columns:
            monthly_returns = equity_curve['returns'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            colors = ['g' if r > 0 else 'r' for r in monthly_returns]
            ax2.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
            ax2.set_title('Monthly Returns')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
            
        # 3. Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        if 'drawdown' in equity_curve.columns:
            ax3.fill_between(equity_curve.index, 0, equity_curve['drawdown'], 
                           color='red', alpha=0.5)
            ax3.set_title('Drawdown Chart')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
            
        # 4. Win/Loss distribution
        ax4 = fig.add_subplot(gs[2, :])
        trades = results.get('trades', pd.DataFrame())
        if not trades.empty and 'pnl' in trades.columns:
            wins = trades[trades['pnl'] > 0]['pnl']
            losses = trades[trades['pnl'] <= 0]['pnl']
            
            bins = 30
            ax4.hist(wins, bins=bins, alpha=0.6, label='Wins', color='green')
            ax4.hist(losses, bins=bins, alpha=0.6, label='Losses', color='red')
            ax4.set_title('Trade P&L Distribution')
            ax4.set_xlabel('P&L ($)')
            ax4.set_ylabel('Frequency')

        # Overlay PSAR confirmation summary as figure text if present
        m = results.get('metrics', {}).get('performance_metrics', {})
        if 'psar_signals_raw_long' in m or 'psar_signals_raw_short' in m:
            rl = m.get('psar_signals_raw_long', 0)
            cl = m.get('psar_signals_confirmed_long', 0)
            rll = m.get('psar_confirmation_rate_long') or 0
            rs = m.get('psar_signals_raw_short', 0)
            cs = m.get('psar_signals_confirmed_short', 0)
            rls = m.get('psar_confirmation_rate_short') or 0
            txt = f"PSAR Confirmation — Long: {cl}/{rl} ({rll:.1f}%) | Short: {cs}/{rs} ({rls:.1f}%)"
            fig.text(0.5, 0.02, txt, ha='center', fontsize=9)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_trading_analysis_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create trading analysis page (with underwater chart)"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Trading Analysis', fontsize=18, y=0.98)
        
        trades = results.get('trades', pd.DataFrame())
        equity_curve = results.get('equity_curve', pd.DataFrame())
        
        # Layout: 2 rows, 2 columns. Top row has two charts side-by-side.
        # Bottom row is the Underwater Chart spanning both columns.
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        # Top-left: P&L by Day of Week
        if not trades.empty and 'entry_time' in trades.columns and 'pnl' in trades.columns:
            ax_tl = fig.add_subplot(gs[0, 0])
            trades['dow'] = pd.to_datetime(trades['entry_time']).dt.dayofweek
            dow_pnl = trades.groupby('dow')['pnl'].sum()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax_tl.bar(range(len(dow_pnl)), dow_pnl.values, alpha=0.7, color='tab:pink')
            ax_tl.set_xticks(range(len(dow_pnl)))
            ax_tl.set_xticklabels([days[i] for i in dow_pnl.index])
            ax_tl.set_title('P&L by Day of Week')
            ax_tl.set_ylabel('Total P&L ($)')
            ax_tl.grid(True, alpha=0.3)
        
        # Top-right: Exit Reason Distribution
        if not trades.empty and 'exit_reason' in trades.columns:
            ax_tr = fig.add_subplot(gs[0, 1])
            exit_counts = trades['exit_reason'].value_counts()
            colors = ['red' if 'stop_loss' in reason else 'green' for reason in exit_counts.index]
            ax_tr.bar(exit_counts.index, exit_counts.values, alpha=0.7, color=colors)
            ax_tr.set_title('Exit Reason Distribution')
            ax_tr.set_xlabel('Exit Reason')
            ax_tr.set_ylabel('Count')
            ax_tr.tick_params(axis='x', rotation=45)
            ax_tr.grid(True, alpha=0.3)
        
        # Bottom: Underwater chart (Drawdown from Peak)
        if not equity_curve.empty and 'drawdown' in equity_curve.columns:
            ax_bottom = fig.add_subplot(gs[1, :])
            ax_bottom.fill_between(equity_curve.index, equity_curve['drawdown'], 0, 
                                color='red', alpha=0.5)
            ax_bottom.set_title('Underwater Chart (Drawdown from Peak)')
            ax_bottom.set_xlabel('Date')
            ax_bottom.set_ylabel('Drawdown (%)')
            ax_bottom.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_trade_details_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create trade detail pages sorted by Entry Time ascending (per side).
        Emits one page for LONG and another for SHORT if available. Paginates automatically.
        """
        trades = results.get('trades', pd.DataFrame())
        if trades.empty:
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Trade Details — LONG (Chronological)', fontsize=18, y=0.98)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.set_title('No trades', pad=12)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            return

        # Ensure direction and datetimes
        if 'direction' not in trades.columns and 'side' in trades.columns:
            trades['direction'] = trades['side'].apply(lambda s: s.value if hasattr(s, 'value') else str(s).lower())
        if 'entry_time' in trades.columns:
            trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        if 'exit_time' in trades.columns:
            trades['exit_time'] = pd.to_datetime(trades['exit_time'])

        def emit_side(side_label: str, side_filter: str):
            side_df = trades[trades.get('direction', '').str.lower() == side_filter]
            side_df = side_df.sort_values(by='entry_time', ascending=True)
            headers = ['Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price', 'P&L', 'Return %', 'Signal']
            if side_df.empty:
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle(f'Trade Details — {side_label} (Chronological)', fontsize=18, y=0.98)
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_title('No trades', pad=12)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                return

            max_rows_per_page = 40
            total = len(side_df)
            pages = (total + max_rows_per_page - 1) // max_rows_per_page
            for p in range(pages):
                chunk = side_df.iloc[p*max_rows_per_page:(p+1)*max_rows_per_page]
                table_data = [headers]
                for _, tr in chunk.iterrows():
                    entry_time = tr['entry_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(tr.get('entry_time', None)) else ''
                    exit_time = tr['exit_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(tr.get('exit_time', None)) else ''
                    direction = tr.get('direction', side_filter)
                    entry_price = f"${tr.get('entry_price', 0):.2f}"
                    exit_price = f"${tr.get('exit_price', 0):.2f}"
                    pnl = tr.get('pnl', 0)
                    pnl_str = f"${pnl:.2f}"
                    return_pct = tr.get('return_pct', 0) * 100
                    return_str = f"{return_pct:.2f}%"
                    signal = tr.get('entry_signal', tr.get('entry_reason', 'N/A'))
                    table_data.append([entry_time, exit_time, direction, entry_price, exit_price, pnl_str, return_str, signal])

                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle(f'Trade Details — {side_label} (Chronological) — Page {p+1}/{pages}', fontsize=18, y=0.98)
                ax = fig.add_subplot(111)
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=table_data, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 2)
                for i in range(len(table_data[0])):
                    table[(0, i)].set_facecolor('#40466e')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                for i in range(1, len(table_data)):
                    pnl_cell = table_data[i][5]
                    pnl_val = float(pnl_cell.replace('$', '')) if isinstance(pnl_cell, str) else 0
                    table[(i, 5)].set_facecolor('#90EE90' if pnl_val > 0 else '#FFB6C1')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        emit_side('LONG', 'long')
        emit_side('SHORT', 'short')

    def _create_trade_details_short_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Deprecated: handled by _create_trade_details_page; kept for backward compatibility."""
        self._create_trade_details_page(pdf, results)

    def _create_trade_logs_pages(self, pdf: PdfPages, results: Dict[str, Any], max_rows_per_page: int = 40):
        """Create chronological trade logs per side (LONG and SHORT), chunked across pages."""
        trades = results.get('trades', pd.DataFrame())
        if trades.empty:
            return
        # Ensure direction and clean types
        if 'direction' not in trades.columns and 'side' in trades.columns:
            trades['direction'] = trades['side'].apply(lambda s: s.value if hasattr(s, 'value') else str(s).lower())
        # Normalize datetime
        if 'entry_time' in trades.columns:
            trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        if 'exit_time' in trades.columns:
            trades['exit_time'] = pd.to_datetime(trades['exit_time'])

        sides = [('LONG', 'long'), ('SHORT', 'short')]
        for side_title, side_val in sides:
            side_df = trades[trades.get('direction', '').str.lower() == side_val]
            side_df = side_df.sort_values(by='entry_time', ascending=True)
            if side_df.empty:
                # Emit empty page for this side for consistency
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle(f'Trade Log — {side_title} (Chronological)', fontsize=18, y=0.98)
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_title(f'Trade Log — {side_title} (No trades)', pad=12)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                continue

            headers = ['Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price', 'P&L', 'Return %', 'Signal']
            total = len(side_df)
            pages = (total + max_rows_per_page - 1) // max_rows_per_page
            for p in range(pages):
                chunk = side_df.iloc[p*max_rows_per_page:(p+1)*max_rows_per_page]
                table_data = [headers]
                for _, tr in chunk.iterrows():
                    entry_time = tr['entry_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(tr.get('entry_time', None)) else ''
                    exit_time = tr['exit_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(tr.get('exit_time', None)) else ''
                    direction = tr.get('direction', side_val)
                    entry_price = f"${tr.get('entry_price', 0):.2f}"
                    exit_price = f"${tr.get('exit_price', 0):.2f}"
                    pnl = tr.get('pnl', 0)
                    pnl_str = f"${pnl:.2f}"
                    return_pct = tr.get('return_pct', 0) * 100
                    return_str = f"{return_pct:.2f}%"
                    signal = tr.get('entry_signal', tr.get('entry_reason', 'N/A'))
                    table_data.append([entry_time, exit_time, direction, entry_price, exit_price, pnl_str, return_str, signal])

                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle(f'Trade Log — {side_title} (Chronological) — Page {p+1}/{pages}', fontsize=18, y=0.98)
                ax = fig.add_subplot(111)
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=table_data, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 2)
                for i in range(len(table_data[0])):
                    table[(0, i)].set_facecolor('#40466e')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                for i in range(1, len(table_data)):
                    pnl_cell = table_data[i][5]
                    pnl_val = float(pnl_cell.replace('$', '')) if isinstance(pnl_cell, str) else 0
                    table[(i, 5)].set_facecolor('#90EE90' if pnl_val > 0 else '#FFB6C1')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
    def _create_recommendations_page(self, pdf: PdfPages, results: Dict[str, Any]):
        """Create optimization recommendations page"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Optimization Recommendations', fontsize=18, y=0.95)
        
        # Get recommendations
        metrics = results.get('metrics', {}).get('performance_metrics', {})
        recommendations = self.generate_optimization_recommendations(metrics, results)
        
        # Create text content
        text_content = f"""
OPTIMIZATION RECOMMENDATIONS
{'='*50}

Based on the backtest analysis, here are specific recommendations
for improving strategy performance:

"""
        
        # Add parameter optimization suggestions
        text_content += f"""
1. PARAMETER OPTIMIZATION
   {'-'*40}
"""
        for rec in recommendations['parameter_optimization']:
            text_content += f"   • {rec}\n"
            
        # Add signal improvement suggestions
        text_content += f"""

2. SIGNAL IMPROVEMENTS
   {'-'*40}
"""
        for rec in recommendations['signal_improvements']:
            text_content += f"   • {rec}\n"
            
        # Add risk management suggestions
        text_content += f"""

3. RISK MANAGEMENT
   {'-'*40}
"""
        for rec in recommendations['risk_management']:
            text_content += f"   • {rec}\n"
            
        # Add next steps
        text_content += f"""

4. NEXT STEPS
   {'-'*40}
"""
        for rec in recommendations['next_steps']:
            text_content += f"   • {rec}\n"
            
        # Add text to figure
        plt.text(0.1, 0.9, text_content, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def generate_optimization_recommendations(self, 
                                           metrics: Dict[str, Any],
                                           results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate detailed optimization recommendations based on backtest results.
        
        Args:
            metrics: Performance metrics
            results: Full backtest results
            
        Returns:
            Dictionary of categorized recommendations
        """
        recommendations = {
            'parameter_optimization': [],
            'signal_improvements': [],
            'risk_management': [],
            'next_steps': []
        }
        
        # Analyze current parameters
        strategy_config = results.get('strategy_config', {})
        ichimoku_params = strategy_config.get('parameters', {}).get('ichimoku_params', {})
        
        # Parameter optimization recommendations
        if metrics.get('sharpe_ratio', 0) < 1:
            recommendations['parameter_optimization'].append(
                "Run walk-forward optimization to find more robust Ichimoku parameters"
            )
            recommendations['parameter_optimization'].append(
                f"Current Tenkan period ({ichimoku_params.get('tenkan_period', 9)}) "
                "may benefit from testing range 7-12"
            )
            
        if metrics.get('total_trades', 0) < 50:
            recommendations['parameter_optimization'].append(
                "Consider shorter Ichimoku periods to increase trading frequency"
            )
            
        # Signal improvement recommendations
        if metrics.get('win_rate', 0) < 0.45:
            recommendations['signal_improvements'].append(
                "Add momentum confirmation (RSI > 50) to filter false breakouts"
            )
            recommendations['signal_improvements'].append(
                "Implement volume confirmation for cloud breakout signals"
            )
            
        if metrics.get('avg_win', 0) < metrics.get('avg_loss', 0) * 1.5:
            recommendations['signal_improvements'].append(
                "Consider trend strength filters to capture larger moves"
            )
            recommendations['signal_improvements'].append(
                "Test adding Chikou span confirmation for stronger signals"
            )
            
        # Risk management recommendations
        if abs(metrics.get('max_drawdown', 0)) > 0.2:
            recommendations['risk_management'].append(
                f"Implement dynamic position sizing to reduce drawdown below 20%"
            )
            recommendations['risk_management'].append(
                "Consider volatility-based stop losses instead of fixed percentage"
            )
            
        if metrics.get('largest_loss', 0) > metrics.get('avg_loss', 0) * 3:
            recommendations['risk_management'].append(
                "Add maximum loss limits to prevent catastrophic trades"
            )
            recommendations['risk_management'].append(
                "Implement correlated asset filters during high volatility"
            )
            
        # Next steps
        recommendations['next_steps'].append(
            "Run parameter optimization with suggested ranges"
        )
        recommendations['next_steps'].append(
            "Validate improvements on out-of-sample data (last 3-6 months)"
        )
        recommendations['next_steps'].append(
            "Implement Monte Carlo simulation for robustness testing"
        )
        recommendations['next_steps'].append(
            "Consider market regime filters for adaptive parameter selection"
        )
        
        return recommendations
        
    def _generate_html_report(self, results: Dict[str, Any], base_filename: str) -> str:
        """Generate interactive HTML report"""
        html_path = self.output_dir / f"{base_filename}.html"
        
        # Create HTML template
        html_template = self._get_html_template()
        
        # Prepare data for template
        metrics = results.get('metrics', {}).get('performance_metrics', {})
        executive_summary = self.create_executive_summary(metrics)
        
        # Create interactive charts
        charts = {
            'equity_curve': self._create_plotly_equity_curve(results),
            'drawdown_chart': self._create_plotly_drawdown_chart(results),
            'monthly_returns': self._create_plotly_monthly_returns(results),
            'trade_analysis': self._create_plotly_trade_analysis(results),
            'performance_metrics': self._create_metrics_table(metrics)
        }
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            title=f"Backtest Report - {results.get('strategy_config', {}).get('name', 'Strategy')}",
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            executive_summary=executive_summary,
            charts=charts,
            strategy_config=results.get('strategy_config', {})
        )
        
        # Save HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        return str(html_path)
        
    def _get_html_template(self) -> str:
        """Get HTML report template"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .section {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .chart-container {
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #2c3e50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .recommendation-list {
            list-style-type: none;
            padding-left: 0;
        }
        .recommendation-list li {
            padding: 10px 0;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin: 10px 0;
        }
        .performance-rating {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .rating-excellent { background-color: #27ae60; color: white; }
        .rating-good { background-color: #3498db; color: white; }
        .rating-satisfactory { background-color: #f39c12; color: white; }
        .rating-poor { background-color: #e74c3c; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p><strong>Trading Pair:</strong> {{ strategy_config.symbol }} | <strong>Timeframe:</strong> {{ strategy_config.timeframe }}</p>
            <p>Generated: {{ generated_at }}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ executive_summary.overview.total_return }}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ executive_summary.overview.sharpe_ratio }}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ executive_summary.overview.max_drawdown }}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ executive_summary.overview.win_rate }}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>
            
            <p><strong>Performance Rating:</strong> 
                <span class="performance-rating rating-{{ executive_summary.performance_rating.rating.lower() }}">
                    {{ executive_summary.performance_rating.rating }}
                </span>
            </p>
            <p><strong>Risk Level:</strong> {{ executive_summary.risk_assessment.risk_level }}</p>
            
            <h3>Key Insights</h3>
            <ul>
                {% for insight in executive_summary.key_insights %}
                <li>{{ insight }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="section">
            <h2>Performance Analysis</h2>
            <div class="chart-container" id="equity-curve"></div>
            <div class="chart-container" id="drawdown-chart"></div>
            <div class="chart-container" id="monthly-returns"></div>
        </div>
        
        <div class="section">
            <h2>Trading Analysis</h2>
            <div class="chart-container" id="trade-analysis"></div>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            {{ charts.performance_metrics | safe }}
        </div>
    </div>
    
    <script>
        // Render Plotly charts
        {{ charts.equity_curve | safe }}
        {{ charts.drawdown_chart | safe }}
        {{ charts.monthly_returns | safe }}
        {{ charts.trade_analysis | safe }}
    </script>
</body>
</html>
'''
        
    def _create_plotly_equity_curve(self, results: Dict[str, Any]) -> str:
        """Create Plotly equity curve chart"""
        equity_curve = results.get('equity_curve', pd.DataFrame())
        
        if equity_curve.empty:
            return ""
            
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2),
            fill='tonexty',
            fillcolor='rgba(0, 100, 255, 0.1)'
        ))
        
        # Add baseline
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=[equity_curve['equity'].iloc[0]] * len(equity_curve),
            mode='lines',
            name='Initial Capital',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            hovermode='x unified',
            height=400
        )
        
        # Convert to JavaScript code
        return f"Plotly.newPlot('equity-curve', {fig.to_json()});"
        
    def _create_plotly_drawdown_chart(self, results: Dict[str, Any]) -> str:
        """Create Plotly drawdown chart"""
        equity_curve = results.get('equity_curve', pd.DataFrame())
        
        if equity_curve.empty or 'drawdown' not in equity_curve.columns:
            return ""
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title='Drawdown Chart',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            height=300
        )
        
        return f"Plotly.newPlot('drawdown-chart', {fig.to_json()});"
        
    def _create_plotly_monthly_returns(self, results: Dict[str, Any]) -> str:
        """Create Plotly monthly returns heatmap"""
        equity_curve = results.get('equity_curve', pd.DataFrame())
        
        if equity_curve.empty or 'returns' not in equity_curve.columns:
            return ""
            
        # Calculate monthly returns
        monthly_returns = equity_curve['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create year-month matrix
        returns_df = pd.DataFrame(monthly_returns)
        returns_df['Year'] = returns_df.index.year
        returns_df['Month'] = returns_df.index.month
        
        # Pivot for heatmap
        heatmap_data = returns_df.pivot(index='Year', columns='Month', values='returns')
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values * 100,  # Convert to percentage
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values * 100,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title='Return %')
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400
        )
        
        return f"Plotly.newPlot('monthly-returns', {fig.to_json()});"
        
    def _create_plotly_trade_analysis(self, results: Dict[str, Any]) -> str:
        """Create Plotly trade analysis charts"""
        trades = results.get('trades', pd.DataFrame())
        
        if trades.empty:
            return ""
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Trade P&L Distribution', 'Win/Loss Bar Chart', 
                          'Trade Duration', 'P&L by Signal Type')
        )
        
        # 1. P&L distribution
        fig.add_trace(
            go.Histogram(x=trades['pnl'], nbinsx=30, name='P&L Distribution'),
            row=1, col=1
        )
        
        # 2. Win/Loss bar chart
        wins = len(trades[trades['pnl'] > 0])
        losses = len(trades[trades['pnl'] <= 0])
        fig.add_trace(
            go.Bar(x=['Wins', 'Losses'], y=[wins, losses], 
                  marker_color=['green', 'red'], name='Win/Loss'),
            row=1, col=2
        )
        
        # 3. Trade duration
        if 'duration_hours' in trades.columns:
            fig.add_trace(
                go.Box(y=trades['duration_hours'], name='Duration'),
                row=2, col=1
            )
            
        # 4. P&L by signal type
        if 'entry_signal' in trades.columns:
            signal_pnl = trades.groupby('entry_signal')['pnl'].sum()
            fig.add_trace(
                go.Bar(x=signal_pnl.index, y=signal_pnl.values, name='Signal P&L'),
                row=2, col=2
            )
            
        fig.update_layout(height=600, showlegend=False)
        
        return f"Plotly.newPlot('trade-analysis', {fig.to_json()});"
        
    def _create_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Create HTML metrics table"""
        # Define metrics to display
        metric_rows = [
            ('Total Return', f"{metrics.get('total_return', 0):.2%}"),
            ('Annualized Return', f"{metrics.get('annual_return', 0):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.2%}"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
            ('Average Win', f"${metrics.get('avg_win', 0):.2f}"),
            ('Average Loss', f"${metrics.get('avg_loss', 0):.2f}"),
            ('Total Trades', f"{metrics.get('total_trades', 0)}"),
            ('Value at Risk (95%)', f"{metrics.get('var_95', 0):.2%}"),
            ('Expected Shortfall (95%)', f"{metrics.get('cvar_95', 0):.2%}")
        ]
        
        # PSAR rows if available
        if 'psar_signals_raw_long' in metrics or 'psar_signals_raw_short' in metrics:
            rl = metrics.get('psar_signals_raw_long', 0)
            cl = metrics.get('psar_signals_confirmed_long', 0)
            rll = metrics.get('psar_confirmation_rate_long')
            rs = metrics.get('psar_signals_raw_short', 0)
            cs = metrics.get('psar_signals_confirmed_short', 0)
            rls = metrics.get('psar_confirmation_rate_short')
            metric_rows.extend([
                ('PSAR Long Confirmed/Raw', f"{cl}/{rl} ({(rll or 0):.1f}%)"),
                ('PSAR Short Confirmed/Raw', f"{cs}/{rs} ({(rls or 0):.1f}%)"),
            ])

        # Create HTML table
        html = '<table><tr><th>Metric</th><th>Value</th></tr>'
        for metric, value in metric_rows:
            html += f'<tr><td>{metric}</td><td>{value}</td></tr>'
        html += '</table>'
        
        return html
        
    def _export_json(self, results: Dict[str, Any], base_filename: str) -> str:
        """Export results as JSON"""
        json_path = self.output_dir / f"{base_filename}.json"
        
        # Prepare data for JSON serialization
        json_data = {
            'metadata': self.report_metadata,
            'strategy_config': results.get('strategy_config', {}),
            'performance_metrics': results.get('metrics', {}).get('performance_metrics', {}),
            'executive_summary': self.create_executive_summary(
                results.get('metrics', {}).get('performance_metrics', {})
            ),
            'trades_summary': {
                'total_trades': len(results.get('trades', [])),
                'winning_trades': len(results.get('trades', pd.DataFrame())[
                    results.get('trades', pd.DataFrame()).get('pnl', pd.Series()) > 0
                ]) if not results.get('trades', pd.DataFrame()).empty else 0,
                'losing_trades': len(results.get('trades', pd.DataFrame())[
                    results.get('trades', pd.DataFrame()).get('pnl', pd.Series()) <= 0
                ]) if not results.get('trades', pd.DataFrame()).empty else 0
            }
        }
        
        # Convert datetime objects to strings
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
            
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=serialize_datetime)
            
        return str(json_path)
        
    def _prepare_streamlined_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Prepare streamlined trades DataFrame with focused columns.
        
        Args:
            trades: Full trades DataFrame
            
        Returns:
            Streamlined DataFrame with essential columns only
        """
        streamlined = pd.DataFrame()
        
        # Trade ID
        streamlined['trade_id'] = trades['trade_id']
        
        # Entry and exit times (formatted)
        streamlined['entry_time'] = pd.to_datetime(trades['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        streamlined['exit_time'] = pd.to_datetime(trades['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Calculate duration in days
        entry_dt = pd.to_datetime(trades['entry_time'])
        exit_dt = pd.to_datetime(trades['exit_time'])
        duration_days = (exit_dt - entry_dt).dt.total_seconds() / 86400  # Convert seconds to days
        streamlined['duration_days'] = duration_days.round(2)
        
        # Side (LONG or SHORT) - clean up the enum format
        if 'side' in trades.columns:
            streamlined['side'] = trades['side'].astype(str).str.replace('PositionSide.', '')
        else:
            streamlined['side'] = 'LONG'  # Default for current implementation
        
        # Entry and exit prices (formatted to 2 decimals)
        streamlined['entry_price'] = trades['entry_price'].round(2)
        streamlined['exit_price'] = trades['exit_price'].round(2)
        
        # P&L (formatted to 2 decimals)
        streamlined['pnl'] = trades['pnl'].round(2)
        
        # Return percentage (formatted with % sign)
        streamlined['return_pct'] = trades['return_pct'].round(2).astype(str) + '%'
        
        # Trade result (WIN or LOSS)
        streamlined['trade_result'] = trades['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')
        
        # Exit reason
        streamlined['exit_reason'] = trades['exit_reason']
        
        return streamlined
    
    def _export_csv(self, results: Dict[str, Any], base_filename: str) -> Dict[str, str]:
        """Export results as CSV files"""
        csv_paths = {}
        
        # Export trades with streamlined format
        trades = results.get('trades', pd.DataFrame())
        if not trades.empty:
            # Create streamlined trades DataFrame with focused columns
            streamlined_trades = self._prepare_streamlined_trades(trades)
            trades_path = self.output_dir / f"{base_filename}_trades.csv"
            streamlined_trades.to_csv(trades_path, index=False)
            csv_paths['trades'] = str(trades_path)
            
        # Export equity curve
        equity_curve = results.get('equity_curve', pd.DataFrame())
        if not equity_curve.empty:
            equity_path = self.output_dir / f"{base_filename}_equity_curve.csv"
            equity_curve.to_csv(equity_path)
            csv_paths['equity_curve'] = str(equity_path)
            
        # Export metrics
        metrics = results.get('metrics', {}).get('performance_metrics', {})
        if metrics:
            metrics_df = pd.DataFrame([metrics])
            metrics_path = self.output_dir / f"{base_filename}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            csv_paths['metrics'] = str(metrics_path)
            
        return csv_paths
        
    def export_results(self, results: Dict[str, Any], format: str = 'pdf') -> str:
        """
        Export results in specified format.
        
        Args:
            results: Backtest results
            format: Export format ('pdf', 'json', 'csv')
            
        Returns:
            Path to exported file(s)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"export_{timestamp}"
        
        if format == 'pdf':
            return self._generate_pdf_report(results, base_filename)
        elif format == 'json':
            return self._export_json(results, base_filename)
        elif format == 'csv':
            return self._export_csv(results, base_filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")