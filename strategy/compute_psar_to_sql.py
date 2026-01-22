"""
Compute and persist Parabolic SAR (PSAR) from SQL OHLCV data for BTC/USDT, ETH/USDT, and SOL/USDT.

- Loads OHLCV from per-symbol SQLite databases
- Computes PSAR components using strategy/psar_indicator.py
- Saves results into psar_data via DataManager.save_psar_data
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
from strategy.psar_indicator import compute_psar

SYMBOLS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["4h", "1d"]


def compute_for_symbol(symbol: str, step: float = 0.02, max_step: float = 0.2) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}

    dm = DataManager(symbol=symbol)

    for tf in TIMEFRAMES:
        ohlcv = dm.get_ohlcv_data(timeframe=tf)
        if ohlcv.empty:
            stats[tf] = {"inserted": 0, "updated": 0, "errors": 0}
            continue
        required_cols = {"id", "high", "low", "close"}
        if not required_cols.issubset(ohlcv.columns):
            raise ValueError(f"{symbol} {tf}: Missing columns in OHLCV: {required_cols - set(ohlcv.columns)}")

        psar_df = compute_psar(ohlcv[["high","low","close"]], step=step, max_step=max_step)
        payload = pd.DataFrame({
            "ohlcv_id": ohlcv["id"].values,
            "psar": psar_df["psar"].values,
            "psar_trend": psar_df["psar_trend"].values,
            "psar_reversal": psar_df["psar_reversal"].values,
            "step": step,
            "max_step": max_step,
        }, index=ohlcv.index)

        res = dm.save_psar_data(payload)
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
