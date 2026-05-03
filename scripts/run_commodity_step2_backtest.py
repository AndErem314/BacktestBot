#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner

COMMODITIES = ["GOLD", "CLOIL"]
TIMEFRAME = "1d"
START_DATE = "2022-04-01"
END_DATE = "2026-04-30"
INITIAL_CAPITAL = 10000.0
REPORTS_DIR = ROOT / "reports" / "step2_results"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_PARAMS = {
    "GOLD": {
        "ichimoku": {"tenkan": 15, "kijun": 30, "senkou_b": 65},
        "macd": {"fast": 14, "slow": 28, "signal": 11},
        "psar": {"step": 0.03, "max_step": 0.15}
    },
    "CLOIL": {
        "ichimoku": {"tenkan": 9, "kijun": 34, "senkou_b": 52},
        "macd": {"fast": 14, "slow": 28, "signal": 11},
        "psar": {"step": 0.03, "max_step": 0.15}
    }
}

def create_confluence_strategy(key, name, commodity, indicators, sell_logic="AND"):
    params = TOP_PARAMS[commodity]
    buy_conditions = []
    sell_conditions = []
    use_ichimoku = indicators.get('ichimoku', False)
    use_macd = indicators.get('macd', False)
    use_psar = indicators.get('psar', False)
    
    if use_ichimoku:
        buy_conditions.extend(["PriceAboveCloud", "TenkanAboveKijun", "SpanAaboveSpanB"])
        sell_conditions.append("TenkanBelowKijun")
    if use_macd:
        buy_conditions.append("MACDAboveSignal")
        sell_conditions.append("MACDBelowSignal")
    if use_psar:
        buy_conditions.append("PSARTrendUp")
        sell_conditions.append("PSARTrendDown")
    
    return {
        "name": name,
        "description": f"Confluence strategy: {name} for {commodity}",
        "enabled": True,
        "asset_class": "commodity",
        "timeframes": [TIMEFRAME],
        "symbols": [commodity],
        "signal_conditions": {
            "buy_conditions": buy_conditions,
            "sell_conditions": sell_conditions,
            "buy_logic": "AND",
            "sell_logic": sell_logic
        },
        "ichimoku_parameters": {
            "tenkan_period": params["ichimoku"]["tenkan"] if use_ichimoku else 9,
            "kijun_period": params["ichimoku"]["kijun"] if use_ichimoku else 26,
            "senkou_b_period": params["ichimoku"]["senkou_b"] if use_ichimoku else 52,
            "chikou_offset": 26, "senkou_offset": 26
        },
        "macd_parameters": {
            "enabled": use_macd,
            "fast": params["macd"]["fast"] if use_macd else 12,
            "slow": params["macd"]["slow"] if use_macd else 26,
            "signal": params["macd"]["signal"] if use_macd else 9
        },
        "psar_parameters": {
            "enabled": use_psar,
            "step": params["psar"]["step"] if use_psar else 0.02,
            "max_step": params["psar"]["max_step"] if use_psar else 0.2
        },
        "risk_management": {
            "stop_loss_pct": 4.0, "take_profit_pct": 8.0,
            "close_on_sell_signal": True, "trailing_stop": False,
            "max_position_size_pct": 100.0, "risk_per_trade_pct": 4.0
        },
        "position_sizing": {"method": "fixed", "fixed_size": 1000}
    }

def main():
    strategies_yaml = ROOT / "config" / "strategies.yaml"
    with open(strategies_yaml, "r") as f:
        config = yaml.safe_load(f) or {}
    config.setdefault("strategies", {})
    
    combinations = [
        ("Ichimoku+MACD", {"ichimoku": True, "macd": True, "psar": False}, "AND"),
        ("Ichimoku+PSAR", {"ichimoku": True, "macd": False, "psar": True}, "AND"),
        ("MACD+PSAR", {"ichimoku": False, "macd": True, "psar": True}, "AND"),
        ("AllThree", {"ichimoku": True, "macd": True, "psar": True}, "OR"),
    ]
    
    step2_strategies = {}
    for commodity in COMMODITIES:
        for suffix, indicators, sell_logic in combinations:
            key = f"step2_{commodity}_{suffix.replace('+', '_')}"
            name = f"{suffix} ({commodity})"
            step2_strategies[key] = create_confluence_strategy(key, name, commodity, indicators, sell_logic)
    
    config["strategies"].update(step2_strategies)
    with open(strategies_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Generated {len(step2_strategies)} Step 2 confluence strategies")
    
    backtester = IchimokuBacktester()
    runner = StrategyBacktestRunner(backtester)
    results = []
    
    for key, strategy in step2_strategies.items():
        commodity = strategy["symbols"][0]
        print(f"Running {key} for {commodity}...")
        try:
            outcome = runner.run_from_json(
                strategy_key=key, symbol_short=commodity, timeframe=TIMEFRAME,
                start=START_DATE, end=END_DATE, initial_capital=INITIAL_CAPITAL,
                report_formats="csv", output_dir=str(REPORTS_DIR), force_recompute_ichimoku=True
            )
            metrics = outcome["result"].metrics or {}
            results.append({
                "strategy_key": key, "name": strategy["name"], "commodity": commodity,
                "total_trades": metrics.get("total_trades"),
                "total_return_pct": metrics.get("total_return_pct"),
                "win_rate_pct": metrics.get("win_rate_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sell_logic": strategy["signal_conditions"]["sell_logic"]
            })
            print(f"  ✓ {key}: {metrics.get('total_return_pct', 0):.2f}%, {metrics.get('total_trades', 0)} trades")
        except Exception as e:
            print(f"  ✗ {key} failed: {e}")
            results.append({"strategy_key": key, "commodity": commodity, "error": str(e)})
    
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = REPORTS_DIR / f"step2_results_{timestamp}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")
    
    md_path = REPORTS_DIR / f"step2_comparison_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# Step 2 Confluence Backtest Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        for commodity in COMMODITIES:
            comm_results = results_df[results_df["commodity"] == commodity]
            f.write(f"## {commodity}\n\n")
            f.write("### Top Strategies (by Return)\n\n")
            top = comm_results.sort_values("total_return_pct", ascending=False)
            if not top.empty:
                f.write(top[["strategy_key", "name", "total_return_pct", "total_trades", "sell_logic"]].to_markdown(index=False))
                f.write("\n\n")
    print(f"Comparison saved to {md_path}")

if __name__ == "__main__":
    main()
