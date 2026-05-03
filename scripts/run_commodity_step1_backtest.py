#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import itertools

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner
from data_fetching.commodity_data_handler import CommodityDataHandler

# === Configuration ===
COMMODITIES = ["GOLD", "CLOIL"]  # GC=F, CL=F
TIMEFRAME = "1d"  # 4+ years of data available (2022-04-01 to 2026-04-30)
START_DATE = "2022-04-01"  # Use maximum available history for MORE trades
END_DATE = "2026-04-30"
INITIAL_CAPITAL = 10000.0
REPORTS_DIR = ROOT / "reports" / "step1_results"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline strategy keys (from strategies.yaml)
BASELINE_GOLD = "commodity_gold_winning"
BASELINE_OIL = "commodity_oil_winning"

# === Step 1 Parameter Grids ===
# 1. Ichimoku Parameters (single-indicator, no MACD/PSAR)
ichimoku_tenkan = [9, 12, 15]
ichimoku_kijun = [26, 30, 34]
ichimoku_senkou_b = [52, 60, 65]
ichimoku_adx = [15, 20, 25, None]  # None = no ADX filter

# 2. MACD Parameters (single-indicator, no Ichimoku/PSAR)
macd_fast = [10, 12, 14]
macd_slow = [24, 26, 28]
macd_signal = [7, 9, 11]

# 3. PSAR Parameters (single-indicator, no Ichimoku/MACD)
psar_af = [0.01, 0.02, 0.03]
psar_max_af = [0.15, 0.2, 0.25]

# === Naming Conventions ===
def get_ichimoku_name(t, k, sb, adx):
    adx_str = f"ADX{adx}" if adx is not None else "ADXNone"
    return f"Ichimoku_T{t}_K{k}_SB{sb}_{adx_str}"

def get_macid_name(f, s, sig):
    return f"MACD_F{f}_S{s}_Sig{sig}"

def get_psar_name(af, max_af):
    return f"PSAR_AF{af}_Max{max_af}"

# === Generate Strategy Configs ===
def create_ichimoku_strategy(key, name, tenkan, kijun, senkou_b, adx):
    """Create Ichimoku-only strategy config (no MACD/PSAR)"""
    buy_conditions = ["PriceAboveCloud", "TenkanAboveKijun", "SpanAaboveSpanB"]
    return {
        "name": name,
        "description": f"Ichimoku single-indicator test: {name}",
        "enabled": True,
        "asset_class": "commodity",
        "timeframes": [TIMEFRAME],
        "symbols": ["{symbol}"],  # Placeholder, replaced per commodity
        "signal_conditions": {
            "buy_conditions": buy_conditions,
            "sell_conditions": ["TenkanBelowKijun"],
            "buy_logic": "AND",
            "sell_logic": "AND"
        },
        "ichimoku_parameters": {
            "tenkan_period": tenkan,
            "kijun_period": kijun,
            "senkou_b_period": senkou_b,
            "chikou_offset": 26,
            "senkou_offset": 26
        },
        "risk_management": {
            "stop_loss_pct": 4.0,
            "take_profit_pct": 8.0,
            "close_on_sell_signal": True,
            "trailing_stop": False,
            "max_position_size_pct": 100.0,
            "risk_per_trade_pct": 4.0
        },
        "position_sizing": {
            "method": "fixed",
            "fixed_size": 1000
        },
        "macd_parameters": {"enabled": False},
        "psar_parameters": {"enabled": False}
    }

def create_macid_strategy(key, name, fast, slow, signal):
    """Create MACD-only strategy config"""
    return {
        "name": name,
        "description": f"MACD single-indicator test: {name}",
        "enabled": True,
        "asset_class": "commodity",
        "timeframes": [TIMEFRAME],
        "symbols": ["{symbol}"],
        "signal_conditions": {
            "buy_conditions": ["MACDAboveSignal"],
            "sell_conditions": ["MACDBelowSignal"],
            "buy_logic": "AND",
            "sell_logic": "AND"
        },
        "ichimoku_parameters": {
            "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
            "chikou_offset": 26, "senkou_offset": 26
        },
        "risk_management": {
            "stop_loss_pct": 4.0, "take_profit_pct": 8.0,
            "close_on_sell_signal": True, "trailing_stop": False,
            "max_position_size_pct": 100.0, "risk_per_trade_pct": 4.0
        },
        "position_sizing": {"method": "fixed", "fixed_size": 1000},
        "macd_parameters": {"enabled": True, "fast": fast, "slow": slow, "signal": signal},
        "psar_parameters": {"enabled": False}
    }

def create_psar_strategy(key, name, af, max_af):
    """Create PSAR-only strategy config"""
    return {
        "name": name,
        "description": f"PSAR single-indicator test: {name}",
        "enabled": True,
        "asset_class": "commodity",
        "timeframes": [TIMEFRAME],
        "symbols": ["{symbol}"],
        "signal_conditions": {
            "buy_conditions": ["PSARTrendUp"],
            "sell_conditions": ["PSARTrendDown"],
            "buy_logic": "AND",
            "sell_logic": "AND"
        },
        "ichimoku_parameters": {
            "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
            "chikou_offset": 26, "senkou_offset": 26
        },
        "risk_management": {
            "stop_loss_pct": 4.0, "take_profit_pct": 8.0,
            "close_on_sell_signal": True, "trailing_stop": False,
            "max_position_size_pct": 100.0, "risk_per_trade_pct": 4.0
        },
        "position_sizing": {"method": "fixed", "fixed_size": 1000},
        "macd_parameters": {"enabled": False},
        "psar_parameters": {"enabled": True, "step": af, "max_step": max_af}
    }

# === Main Backtest Runner ===
def main():
    # Load existing strategies config
    strategies_yaml = ROOT / "config" / "strategies.yaml"
    with open(strategies_yaml, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Add baseline strategies if not present
    if BASELINE_GOLD not in config.get("strategies", {}):
        config.setdefault("strategies", {})[BASELINE_GOLD] = {
            "name": "Gold Winning Strategy - 4% SL / 8% TP",
            "enabled": True,
            "asset_class": "commodity",
            "timeframes": [TIMEFRAME],
            "symbols": ["GOLD"],
            "signal_conditions": {
                "buy_conditions": ["PriceAboveCloud", "TenkanAboveKijun", "SpanAaboveSpanB"],
                "sell_conditions": ["TenkanBelowKijun"],
                "buy_logic": "AND", "sell_logic": "AND"
            },
            "ichimoku_parameters": {
                "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
                "chikou_offset": 26, "senkou_offset": 26
            },
            "risk_management": {
                "stop_loss_pct": 4.0, "take_profit_pct": 8.0,
                "close_on_sell_signal": True, "trailing_stop": False,
                "max_position_size_pct": 100.0, "risk_per_trade_pct": 4.0
            },
            "position_sizing": {"method": "fixed", "fixed_size": 1000},
            "macd_parameters": {"enabled": False},
            "psar_parameters": {"enabled": False}
        }
    
    if BASELINE_OIL not in config.get("strategies", {}):
        config.setdefault("strategies", {})[BASELINE_OIL] = {
            "name": "Crude Oil Winning Strategy - 4% SL / 8% TP",
            "enabled": True,
            "asset_class": "commodity",
            "timeframes": [TIMEFRAME],
            "symbols": ["CLOIL"],
            "signal_conditions": {
                "buy_conditions": ["PriceAboveCloud", "TenkanAboveKijun", "SpanAaboveSpanB"],
                "sell_conditions": ["TenkanBelowKijun"],
                "buy_logic": "AND", "sell_logic": "AND"
            },
            "ichimoku_parameters": {
                "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
                "chikou_offset": 26, "senkou_offset": 26
            },
            "risk_management": {
                "stop_loss_pct": 4.0, "take_profit_pct": 8.0,
                "close_on_sell_signal": True, "trailing_stop": False,
                "max_position_size_pct": 100.0, "risk_per_trade_pct": 4.0
            },
            "position_sizing": {"method": "fixed", "fixed_size": 1000},
            "macd_parameters": {"enabled": False},
            "psar_parameters": {"enabled": False}
        }
    
    # Generate all Step 1 strategy combinations
    step1_strategies = {}
    
    # 1. Ichimoku combinations (108 total)
    for t, k, sb, adx in itertools.product(ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_b, ichimoku_adx):
        name = get_ichimoku_name(t, k, sb, adx)
        key = f"step1_ichimoku_{name}"
        step1_strategies[key] = create_ichimoku_strategy(key, name, t, k, sb, adx)
    
    # 2. MACD combinations (27 total)
    for f, s, sig in itertools.product(macd_fast, macd_slow, macd_signal):
        name = get_macid_name(f, s, sig)
        key = f"step1_macd_{name}"
        step1_strategies[key] = create_macid_strategy(key, name, f, s, sig)
    
    # 3. PSAR combinations (9 total)
    for af, max_af in itertools.product(psar_af, psar_max_af):
        name = get_psar_name(af, max_af)
        key = f"step1_psar_{name}"
        step1_strategies[key] = create_psar_strategy(key, name, af, max_af)
    
    # Add Step 1 strategies to config
    config["strategies"].update(step1_strategies)
    
    # Save updated config to BOTH temp and original location
    temp_yaml = REPORTS_DIR / "step1_strategies.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # CRITICAL: Also save to original config file so StrategyBacktestRunner can find it
    with open(strategies_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Generated {len(step1_strategies)} Step 1 strategy combinations")
    print(f"Updated config saved to {strategies_yaml}")
    
    # Run backtests for each commodity
    backtester = IchimokuBacktester()
    runner = StrategyBacktestRunner(backtester)
    
    results = []
    
    for commodity in COMMODITIES:
        print(f"\n{'='*60}")
        print(f"Running backtests for {commodity} ({'GC=F' if commodity == 'GOLD' else 'CL=F'})")
        print(f"{'='*60}")
        
        # Run baseline first
        baseline_key = BASELINE_GOLD if commodity == "GOLD" else BASELINE_OIL
        print(f"\nRunning baseline: {baseline_key}")
        try:
            outcome = runner.run_from_json(
                strategy_key=baseline_key,
                symbol_short=commodity,
                timeframe=TIMEFRAME,
                start=START_DATE,
                end=END_DATE,
                initial_capital=INITIAL_CAPITAL,
                report_formats="csv",
                output_dir=str(REPORTS_DIR),
                force_recompute_ichimoku=True
            )
            metrics = outcome["result"].metrics or {}
            results.append({
                "strategy_key": baseline_key,
                "name": config["strategies"][baseline_key]["name"],
                "commodity": commodity,
                "total_trades": metrics.get("total_trades"),
                "total_return_pct": metrics.get("total_return_pct"),
                "win_rate_pct": metrics.get("win_rate_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "is_baseline": True
            })
            print(f"✓ Baseline {baseline_key}: {metrics.get('total_return_pct'):.2f}% return, {metrics.get('total_trades')} trades")
        except Exception as e:
            print(f"✗ Baseline {baseline_key} failed: {e}")
            results.append({"strategy_key": baseline_key, "commodity": commodity, "error": str(e), "is_baseline": True})
        
        # Run Step 1 strategies
        for key in step1_strategies.keys():
            strategy = step1_strategies[key]
            # Update symbol for current commodity
            strategy["symbols"] = [commodity]
            print(f"Running {key} for {commodity}...")
            try:
                outcome = runner.run_from_json(
                    strategy_key=key,
                    symbol_short=commodity,
                    timeframe=TIMEFRAME,
                    start=START_DATE,
                    end=END_DATE,
                    initial_capital=INITIAL_CAPITAL,
                    report_formats="csv",
                    output_dir=str(REPORTS_DIR),
                    force_recompute_ichimoku=True
                )
                metrics = outcome["result"].metrics or {}
                results.append({
                    "strategy_key": key,
                    "name": strategy["name"],
                    "commodity": commodity,
                    "total_trades": metrics.get("total_trades"),
                    "total_return_pct": metrics.get("total_return_pct"),
                    "win_rate_pct": metrics.get("win_rate_pct"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "is_baseline": False
                })
                print(f"✓ {key}: {metrics.get('total_return_pct'):.2f}% return, {metrics.get('total_trades')} trades")
            except Exception as e:
                print(f"✗ {key} failed: {e}")
                results.append({"strategy_key": key, "commodity": commodity, "error": str(e), "is_baseline": False})
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = REPORTS_DIR / f"step1_results_{timestamp}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nAll results saved to {results_csv}")
    
    # Generate comparison Markdown
    md_path = REPORTS_DIR / f"step1_comparison_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Step 1 Commodity Backtest Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Baseline: {BASELINE_GOLD} (Gold), {BASELINE_OIL} (Oil)\n\n")
        f.write(f"## Summary Statistics\n\n")
        for commodity in COMMODITIES:
            commodity_results = results_df[results_df["commodity"] == commodity]
            baseline = commodity_results[commodity_results["is_baseline"] == True].iloc[0]
            f.write(f"### {commodity}\n\n")
            f.write(f"**Baseline Return**: {baseline['total_return_pct']:.2f}%\n")
            f.write(f"**Baseline Trades**: {baseline['total_trades']}\n\n")
            f.write(f"#### Top 5 Strategies (by Return)\n\n")
            top5 = commodity_results.sort_values("total_return_pct", ascending=False).head(5)
            f.write(top5[["strategy_key", "name", "total_return_pct", "total_trades"]].to_markdown(index=False))
            f.write("\n\n")
    print(f"Comparison Markdown saved to {md_path}")

if __name__ == "__main__":
    main()
