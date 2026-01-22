"""
Compute and persist Ichimoku indicators from SQL OHLCV data for BTC/USDT, ETH/USDT, and SOL/USDT.

- Loads OHLCV from per-symbol SQLite databases created by streamline_workflow
- Computes Ichimoku components using strategy/ichimoku_strategy.py
- Derives basic attributes (price_position, trend_strength, tk_cross)
- Saves results into ichimoku_data via DataManager.save_ichimoku_data

This prepares data for later backtesting using strategy configurations.
"""
import sys
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

# Import from the correct path
from data_fetching.data_manager import DataManager
from strategy.ichimoku_strategy import (
    UnifiedIchimokuAnalyzer,
    IchimokuStrategyConfig,
)

SYMBOLS = ['BTC', 'ETH', 'SOL']
TIMEFRAMES = ['4h', '1d']


def derive_additional_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derive price_position, trend_strength, and tk_cross from Ichimoku components.

    Assumptions:
    - price_position compares close to current cloud (max/min of senkou spans)
    - trend_strength is a simple classification based on price position and cloud color
    - tk_cross detects state of cross; it is not an event detector, but last state
    """
    out = df.copy()

    # Cloud top/bottom (already present if strategy added them, but recompute to be safe)
    cloud_top = out[["senkou_span_a", "senkou_span_b"]].max(axis=1)
    cloud_bottom = out[["senkou_span_a", "senkou_span_b"]].min(axis=1)

    # price_position
    out["price_position"] = np.where(
        out["close"] > cloud_top,
        "above_cloud",
        np.where(out["close"] < cloud_bottom, "below_cloud", "in_cloud"),
    )

    # cloud_color
    out["cloud_color"] = np.where(
        out["senkou_span_a"] >= out["senkou_span_b"], "green", "red"
    )

    # trend_strength (coarse)
    def classify(row) -> str:
        if pd.isna(row["tenkan_sen"]) or pd.isna(row["kijun_sen"]):
            return None
        price_pos = row["price_position"]
        color = row["cloud_color"]
        tk_above = row["tenkan_sen"] >= row["kijun_sen"]
        if price_pos == "above_cloud" and color == "green" and tk_above:
            return "strong_bullish"
        if price_pos == "above_cloud" and (color == "green" or tk_above):
            return "bullish"
        if price_pos == "below_cloud" and color == "red" and not tk_above:
            return "strong_bearish"
        if price_pos == "below_cloud" and (color == "red" or not tk_above):
            return "bearish"
        return "neutral"

    out["trend_strength"] = out.apply(classify, axis=1)

    # tk_cross state (not discrete event). If Tenkan above Kijun -> bullish_cross, else bearish_cross.
    # If any is NaN, leave as None.
    cond_valid = out[["tenkan_sen", "kijun_sen"]].notna().all(axis=1)
    tk_state = np.where(
        cond_valid & (out["tenkan_sen"] > out["kijun_sen"]),
        "bullish_cross",
        np.where(cond_valid & (out["tenkan_sen"] < out["kijun_sen"]), "bearish_cross", None),
    )
    out["tk_cross"] = tk_state

    return out


def compute_for_symbol(symbol: str) -> Dict[str, Dict[str, int]]:
    """Compute and save Ichimoku for a single symbol across timeframes."""
    stats: Dict[str, Dict[str, int]] = {}

    dm = DataManager(symbol=symbol)
    analyzer = UnifiedIchimokuAnalyzer()
    params = IchimokuStrategyConfig.create_parameters()  # defaults 9/26/52

    for tf in TIMEFRAMES:
        # Fetch OHLCV with id and required cols
        ohlcv = dm.get_ohlcv_data(timeframe=tf)
        if ohlcv.empty:
            stats[tf] = {"inserted": 0, "updated": 0, "errors": 0}
            continue

        # Ensure required columns
        required_cols = {"id", "high", "low", "close"}
        missing = required_cols - set(ohlcv.columns)
        if missing:
            raise ValueError(f"{symbol} {tf}: Missing columns in OHLCV: {missing}")

        # Compute Ichimoku components
        ichimoku = analyzer.calculate_ichimoku_components(ohlcv, params)

        # Derive additional attributes and signals context for SQL
        enriched = derive_additional_fields(ichimoku)

        # Prepare payload for SQL save
        # Keep only rows that map cleanly to ohlcv rows and include ohlcv_id
        payload = pd.DataFrame({
            "ohlcv_id": ohlcv["id"].values,
            "tenkan_sen": enriched["tenkan_sen"].values,
            "kijun_sen": enriched["kijun_sen"].values,
            "senkou_span_a": enriched["senkou_span_a"].values,
            "senkou_span_b": enriched["senkou_span_b"].values,
            "chikou_span": enriched["chikou_span"].values,
            "price_position": enriched.get("price_position"),
            "trend_strength": enriched.get("trend_strength"),
            "tk_cross": enriched.get("tk_cross"),
        }, index=ohlcv.index)

        # Save to SQL
        res = dm.save_ichimoku_data(payload)
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
