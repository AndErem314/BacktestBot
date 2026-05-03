#!/usr/bin/env python3
"""
Compute MACD and PSAR indicators for commodities and store in SQL database.
This fixes the issue where MACD/PSAR strategies had 0 trades.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import sqlite3

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.macd_indicator import compute_macd
from strategy.psar_indicator import compute_psar

COMMODITIES = ["GOLD", "CLOIL"]
DB_DIR = ROOT / "data"

def compute_and_store_indicators(symbol_short):
    """Compute MACD and PSAR for all timeframes and store in SQL."""
    db_name = f"trading_data_{symbol_short}.db"
    db_path = DB_DIR / db_name
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {symbol_short} ({'GC=F' if symbol_short == 'GOLD' else 'CL=F'})")
    print(f"{'='*60}")
    
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    for timeframe in ['1d', '4h']:
        print(f"\nProcessing {timeframe}...")
        
        # Load OHLCV data WITH id column
        query = """
            SELECT id, timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE timeframe = ?
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=[timeframe])
        
        if len(df) == 0:
            print(f"  No data for {timeframe}, skipping")
            continue
        
        print(f"  Loaded {len(df)} candles")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # === Compute MACD ===
        print(f"  Computing MACD...")
        macd_df = compute_macd(df, fast=12, slow=26, signal=9)
        
        # Store MACD in macd_data table
        macd_stored = 0
        for timestamp, row in df.iterrows():
            ohlcv_id = int(row['id'])
            
            # Check if MACD already exists
            cursor.execute("SELECT id FROM macd_data WHERE ohlcv_id = ?", (ohlcv_id,))
            exists = cursor.fetchone()
            
            macd_line = float(macd_df.loc[timestamp, 'macd_line']) if timestamp in macd_df.index else None
            signal_line = float(macd_df.loc[timestamp, 'signal_line']) if timestamp in macd_df.index else None
            histogram = float(macd_df.loc[timestamp, 'macd_histogram']) if timestamp in macd_df.index else None
            
            if exists:
                cursor.execute("""
                    UPDATE macd_data 
                    SET macd_line = ?, signal_line = ?, macd_histogram = ?, 
                        fast_period = 12, slow_period = 26, signal_period = 9,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE ohlcv_id = ?
                """, (macd_line, signal_line, histogram, ohlcv_id))
            else:
                cursor.execute("""
                    INSERT INTO macd_data 
                    (ohlcv_id, macd_line, signal_line, macd_histogram, 
                     fast_period, slow_period, signal_period)
                    VALUES (?, ?, ?, ?, 12, 26, 9)
                """, (ohlcv_id, macd_line, signal_line, histogram))
            
            macd_stored += 1
        
        conn.commit()
        print(f"  ✅ Stored {macd_stored} MACD records")
        
        # === Compute PSAR ===
        print(f"  Computing PSAR...")
        psar_df = compute_psar(df, step=0.02, max_step=0.2)
        
        # Store PSAR in psar_data table
        psar_stored = 0
        for timestamp, row in df.iterrows():
            ohlcv_id = int(row['id'])
            
            # Check if PSAR already exists
            cursor.execute("SELECT id FROM psar_data WHERE ohlcv_id = ?", (ohlcv_id,))
            exists = cursor.fetchone()
            
            psar_val = float(psar_df.loc[timestamp, 'psar']) if timestamp in psar_df.index else None
            psar_trend = int(psar_df.loc[timestamp, 'psar_trend']) if timestamp in psar_df.index else None
            psar_reversal = int(psar_df.loc[timestamp, 'psar_reversal']) if timestamp in psar_df.index else None
            
            if exists:
                cursor.execute("""
                    UPDATE psar_data 
                    SET psar = ?, psar_trend = ?, psar_reversal = ?, 
                        step = 0.02, max_step = 0.2,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE ohlcv_id = ?
                """, (psar_val, psar_trend, psar_reversal, ohlcv_id))
            else:
                cursor.execute("""
                    INSERT INTO psar_data 
                    (ohlcv_id, psar, psar_trend, psar_reversal, step, max_step)
                    VALUES (?, ?, ?, ?, 0.02, 0.2)
                """, (ohlcv_id, psar_val, psar_trend, psar_reversal))
            
            psar_stored += 1
        
        conn.commit()
        print(f"  ✅ Stored {psar_stored} PSAR records")
    
    conn.close()
    print(f"\n✅ Completed {symbol_short}")

def verify_indicators(symbol_short):
    """Verify that indicators were stored correctly."""
    db_name = f"trading_data_{symbol_short}.db"
    db_path = DB_DIR / db_name
    
    conn = sqlite3.connect(str(db_path))
    
    print(f"\n--- Verification for {symbol_short} ---")
    
    for timeframe in ['1d', '4h']:
        # Check OHLCV count
        cursor = conn.execute("SELECT COUNT(*) FROM ohlcv_data WHERE timeframe = ?", (timeframe,))
        ohlcv_count = cursor.fetchone()[0]
        
        # Check MACD count
        cursor = conn.execute("""
            SELECT COUNT(*) FROM macd_data md
            JOIN ohlcv_data od ON md.ohlcv_id = od.id
            WHERE od.timeframe = ?
        """, (timeframe,))
        macd_count = cursor.fetchone()[0]
        
        # Check PSAR count
        cursor = conn.execute("""
            SELECT COUNT(*) FROM psar_data pd
            JOIN ohlcv_data od ON pd.ohlcv_id = od.id
            WHERE od.timeframe = ?
        """, (timeframe,))
        psar_count = cursor.fetchone()[0]
        
        print(f"\n{timeframe}:")
        print(f"  OHLCV: {ohlcv_count}")
        print(f"  MACD: {macd_count} ({macd_count/ohlcv_count*100:.1f}%)")
        print(f"  PSAR: {psar_count} ({psar_count/ohlcv_count*100:.1f}%)")
    
    conn.close()

if __name__ == "__main__":
    for symbol in COMMODITIES:
        compute_and_store_indicators(symbol)
        verify_indicators(symbol)
    
    print("\n" + "="*60)
    print("✅ All indicators computed and stored in SQL databases")
    print("="*60)
