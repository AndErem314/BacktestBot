"""
Compute and persist MACD from SQL OHLCV data for BTC/USDT, ETH/USDT, and SOL/USDT.

- Loads OHLCV from per-symbol SQLite databases
- Computes MACD components using strategy/macd_indicator.py
- Saves results into macd_data via DataManager.save_macd_data
"""
from typing import Dict
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_fetching.data_manager import DataManager
from strategy.macd_indicator import compute_macd

SYMBOLS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["1h", "4h", "1d"]


def compute_for_symbol(symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}

    dm = DataManager(symbol=symbol)

    for tf in TIMEFRAMES:
        ohlcv = dm.get_ohlcv_data(timeframe=tf)
        if ohlcv.empty:
            stats[tf] = {"inserted": 0, "updated": 0, "errors": 0}
            continue
        required_cols = {"id", "close"}
        if not required_cols.issubset(ohlcv.columns):
            raise ValueError(f"{symbol} {tf}: Missing columns in OHLCV: {required_cols - set(ohlcv.columns)}")

        macd_df = compute_macd(ohlcv[["close"]], fast=fast, slow=slow, signal=signal)
        payload = pd.DataFrame({
            "ohlcv_id": ohlcv["id"].values,
            "macd_line": macd_df["macd_line"].values,
            "signal_line": macd_df["signal_line"].values,
            "macd_histogram": macd_df["macd_histogram"].values,
            "fast_period": fast,
            "slow_period": slow,
            "signal_period": signal,
        }, index=ohlcv.index)

        res = dm.save_macd_data(payload)
        stats[tf] = res

    dm.close_connection()
    return stats


def main():
    overall: Dict[str, Dict[str, Dict[str, int]]] = {}
    for sym in SYMBOLS:
        try:
            overall[sym] = compute_for_symbol(sym)
            print(f"{sym}: {overall[sym]}")
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    print("\nSummary:")
    for sym, sym_stats in overall.items():
        print(sym, sym_stats)


if __name__ == "__main__":
    main()
