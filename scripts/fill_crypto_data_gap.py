#!/usr/bin/env python3
"""
Fill missing crypto data gap from Binance using CCXT.
Fetches 4h and 1d data from 2026-01-23 to present and updates local SQL.
"""
import sys
from pathlib import Path
import pandas as pd
import sqlite3
from datetime import datetime, timezone

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_fetching.data_fetcher import DataFetcher

def fill_data_gap(symbol_short='BTC', timeframe='4h'):
    """
    Fetch missing data from Binance and update the local SQL database.
    
    Args:
        symbol_short: Symbol short name (BTC, ETH, SOL)
        timeframe: Timeframe to fetch ('4h' or '1d')
    """
    print(f"\n{'='*60}")
    print(f"Filling data gap for {symbol_short}/USDT ({timeframe})")
    print(f"{'='*60}")
    
    # Initialize data fetcher
    fetcher = DataFetcher(exchange_name='binance', symbol=symbol_short)
    
    # Get database path
    db_path = fetcher.db_path
    print(f"Database: {db_path}")
    
    # Get latest timestamp from database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    query = """
        SELECT MAX(timestamp) as latest
        FROM ohlcv_data
        WHERE timeframe = ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (timeframe,))
    result = cursor.fetchone()
    
    if result and result[0]:
        latest_timestamp = pd.to_datetime(result[0])
        print(f"Latest timestamp in DB: {latest_timestamp}")
        
        # Start from the next period
        if timeframe == '4h':
            since_dt = latest_timestamp + pd.Timedelta(hours=4)
        elif timeframe == '1d':
            since_dt = latest_timestamp + pd.Timedelta(days=1)
        else:
            since_dt = latest_timestamp + pd.Timedelta(hours=1)
    else:
        print(f"No existing data found for {timeframe}, will fetch from 2026-01-23")
        since_dt = pd.to_datetime('2026-01-23')
    
    since_ms = int(since_dt.timestamp() * 1000)
    print(f"Fetching from: {since_dt} (since_ms={since_ms})")
    
    # Fetch data from Binance
    symbol_pair = f"{symbol_short}/USDT"
    print(f"\nFetching {symbol_pair} {timeframe} data from Binance...")
    
    try:
        # Fetch data in batches
        all_data = []
        current_since = since_ms
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        while current_since < now_ms:
            print(f"  Fetching batch from {pd.to_datetime(current_since, unit='ms')}...")
            ohlcv = fetcher.exchange.fetch_ohlcv(symbol_pair, timeframe, current_since, limit=1000)
            
            if not ohlcv:
                print(f"  No more data returned")
                break
            
            df_batch = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')
            
            # Filter to only get data after our latest timestamp
            df_batch = df_batch[df_batch['timestamp'] > latest_timestamp]
            
            if df_batch.empty:
                print(f"  No new data in this batch")
                break
            
            all_data.append(df_batch)
            print(f"  Got {len(df_batch)} new candles")
            
            # Update since to last timestamp + 1 interval
            last_ts = int(df_batch['timestamp'].iloc[-1].timestamp() * 1000)
            
            if timeframe == '4h':
                current_since = last_ts + (4 * 60 * 60 * 1000)
            elif timeframe == '1d':
                current_since = last_ts + (24 * 60 * 60 * 1000)
            else:
                current_since = last_ts + (60 * 60 * 1000)
            
            # Rate limiting
            import time
            time.sleep(fetcher.exchange.rateLimit / 1000)
        
        if not all_data:
            print(f"\n✅ No new data to add for {timeframe}")
            conn.close()
            return
        
        # Combine all data
        df_new = pd.concat(all_data, ignore_index=True)
        df_new = df_new.drop_duplicates(subset=['timestamp'])
        df_new = df_new.sort_values('timestamp')
        
        print(f"\nTotal new candles: {len(df_new)}")
        print(f"Date range: {df_new['timestamp'].iloc[0]} to {df_new['timestamp'].iloc[-1]}")
        
        # Insert into database
        print(f"\nInserting into database...")
        inserted = 0
        
        for _, row in df_new.iterrows():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO ohlcv_data (timestamp, open, high, low, close, volume, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['timestamp'].isoformat(),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    timeframe
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                print(f"  Error inserting row: {e}")
                continue
        
        conn.commit()
        print(f"✅ Inserted {inserted} new records")
        
        # Show updated stats
        cursor.execute("""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM ohlcv_data
            WHERE timeframe = ?
        """, (timeframe,))
        stats = cursor.fetchone()
        print(f"\nUpdated {timeframe} stats:")
        print(f"  Total records: {stats[0]}")
        print(f"  Date range: {stats[1]} to {stats[2]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.close()

if __name__ == "__main__":
    print("Starting data gap fill for BTC/USDT...")
    
    # Fill 4h data
    fill_data_gap('BTC', '4h')
    
    # Fill 1d data
    fill_data_gap('BTC', '1d')
    
    print(f"\n{'='*60}")
    print("✅ Data gap fill complete!")
    print(f"{'='*60}")
