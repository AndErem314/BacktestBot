#!/usr/bin/env python3
"""
Workaround for 4h data limitation:
Resample 1d candles to create synthetic 4h-like data for backtesting.
This gives us 4+ years of "4h" data instead of just 1 year.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def create_synthetic_4h_from_1d(symbol_short):
    """
    Create synthetic 4h data by resampling 1d candles.
    Each 1d candle is split into 6x 4h candles (approximately).
    """
    db_name = f"trading_data_{symbol_short}.db"
    db_path = ROOT / "data" / db_name
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Creating synthetic 4h data for {symbol_short}")
    print(f"{'='*60}")
    
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    # Load 1d data
    query = """
        SELECT id, timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE timeframe = '1d'
        ORDER BY timestamp
    """
    df_1d = pd.read_sql_query(query, conn)
    
    if len(df_1d) == 0:
        print("  No 1d data found")
        conn.close()
        return
    
    print(f"  Loaded {len(df_1d)} 1d candles")
    
    # Create synthetic 4h data
    synthetic_4h = []
    
    for idx, row in df_1d.iterrows():
        base_timestamp = pd.to_datetime(row['timestamp'])
        open_price = float(row['open'])
        high_price = float(row['high'])
        low_price = float(row['low'])
        close_price = float(row['close'])
        volume = float(row['volume'])
        
        # Split into 6x 4h candles (24h = 6 * 4h)
        # This is an approximation - real 4h data would have different OHLCV
        for i in range(6):
            candle_time = base_timestamp + pd.Timedelta(hours=4*i)
            
            # Approximate OHLCV for synthetic 4h candle
            # Use small random variations around the 1d values
            import random
            random.seed(int(candle_time.timestamp()))  # Deterministic
            
            synthetic_4h.append({
                'timestamp': candle_time.isoformat(),
                'open': open_price * (1 + random.uniform(-0.005, 0.005)),
                'high': high_price * (1 + random.uniform(0, 0.01)),
                'low': low_price * (1 + random.uniform(-0.01, 0)),
                'close': close_price * (1 + random.uniform(-0.005, 0.005)),
                'volume': volume / 6 * (1 + random.uniform(-0.2, 0.2)),
                'timeframe': '4h_synthetic'
            })
    
    df_4h = pd.DataFrame(synthetic_4h)
    print(f"  Created {len(df_4h)} synthetic 4h candles")
    
    # Clear existing synthetic data
    cursor.execute("DELETE FROM ohlcv_data WHERE timeframe = '4h_synthetic'")
    
    # Insert synthetic 4h data
    for _, row in df_4h.iterrows():
        cursor.execute("""
            INSERT INTO ohlcv_data (timestamp, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row['timestamp'],
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume']),
            '4h_synthetic'
        ))
    
    conn.commit()
    print(f"  ✅ Stored {len(df_4h)} synthetic 4h candles")
    
    # Compute indicators for synthetic 4h data
    print(f"\n  Computing indicators for synthetic 4h data...")
    
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_4h.set_index('timestamp', inplace=True)
    
    # Compute MACD
    from strategy.macd_indicator import compute_macd
    macd_df = compute_macd(df_4h, fast=12, slow=26, signal=9)
    
    # Compute PSAR
    from strategy.psar_indicator import compute_psar
    psar_df = compute_psar(df_4h, step=0.02, max_step=0.2)
    
    # Store MACD
    cursor.execute("SELECT id FROM ohlcv_data WHERE timeframe = '4h_synthetic' ORDER BY timestamp")
    ohlcv_ids = [row[0] for row in cursor.fetchall()]
    
    macd_stored = 0
    for i, ohlcv_id in enumerate(ohlcv_ids):
        cursor.execute("""
            INSERT OR REPLACE INTO macd_data 
            (ohlcv_id, macd_line, signal_line, macd_histogram, fast_period, slow_period, signal_period)
            VALUES (?, ?, ?, ?, 12, 26, 9)
        """, (
            ohlcv_id,
            float(macd_df.iloc[i]['macd_line']) if i < len(macd_df) else None,
            float(macd_df.iloc[i]['signal_line']) if i < len(macd_df) else None,
            float(macd_df.iloc[i]['macd_histogram']) if i < len(macd_df) else None,
        ))
        macd_stored += 1
    
    # Store PSAR
    psar_stored = 0
    for i, ohlcv_id in enumerate(ohlcv_ids):
        cursor.execute("""
            INSERT OR REPLACE INTO psar_data 
            (ohlcv_id, psar, psar_trend, psar_reversal, step, max_step)
            VALUES (?, ?, ?, ?, 0.02, 0.2)
        """, (
            ohlcv_id,
            float(psar_df.iloc[i]['psar']) if i < len(psar_df) else None,
            int(psar_df.iloc[i]['psar_trend']) if i < len(psar_df) else None,
            int(psar_df.iloc[i]['psar_reversal']) if i < len(psar_df) else None,
        ))
        psar_stored += 1
    
    conn.commit()
    print(f"  ✅ Stored {macd_stored} MACD records")
    print(f"  ✅ Stored {psar_stored} PSAR records")
    
    conn.close()
    print(f"\n✅ Completed synthetic 4h data creation for {symbol_short}")

if __name__ == "__main__":
    for symbol in ["GOLD", "CLOIL"]:
        create_synthetic_4h_from_1d(symbol)
    
    print("\n" + "="*60)
    print("✅ Synthetic 4h data created with 4+ years of history")
    print("   Use timeframe='4h_synthetic' in strategy configs")
    print("="*60)
