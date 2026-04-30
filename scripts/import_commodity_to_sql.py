#!/usr/bin/env python3
"""
Import commodity data from Yahoo Finance JSON to SQLite.
Creates database with unified schema matching crypto DBs.
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path

# Configuration
COMMODITY_DATA_DIR = Path("/Users/andrey/Documents/commodity_backtest/user_data/data/yahoo_commodities")
BACKTESTBOT_DATA_DIR = Path("/Users/andrey/GitHub_projects/BacktestBot/data")

# Commodity mapping: folder name -> database name
COMMODITIES = {
    "GOLD_USDC": "trading_data_GOLD.db",
    "CLOIL_USDC": "trading_data_CLOIL.db",
}

def init_database(db_path: Path):
    """Initialize SQLite database with unified schema."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create OHLCV table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            open DECIMAL(20, 8) NOT NULL CHECK(open > 0),
            high DECIMAL(20, 8) NOT NULL CHECK(high > 0),
            low DECIMAL(20, 8) NOT NULL CHECK(low > 0),
            close DECIMAL(20, 8) NOT NULL CHECK(close > 0),
            volume DECIMAL(20, 8) NOT NULL CHECK(volume >= 0),
            timeframe TEXT NOT NULL CHECK(timeframe IN ('4h', '1d')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            CHECK(high >= low),
            CHECK(high >= open),
            CHECK(high >= close),
            CHECK(low <= open),
            CHECK(low <= close),
            UNIQUE(timestamp, timeframe)
        )
    """)
    
    # Create Ichimoku table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ichimoku_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ohlcv_id INTEGER NOT NULL,
            tenkan_sen DECIMAL(20, 8),
            kijun_sen DECIMAL(20, 8),
            senkou_span_a DECIMAL(20, 8),
            senkou_span_b DECIMAL(20, 8),
            chikou_span DECIMAL(20, 8),
            cloud_color TEXT CHECK(cloud_color IN ('green', 'red', NULL)),
            price_position TEXT CHECK(price_position IN ('above_cloud', 'in_cloud', 'below_cloud', NULL)),
            trend_strength TEXT CHECK(trend_strength IN ('strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish', NULL)),
            tk_cross TEXT CHECK(tk_cross IN ('bullish_cross', 'bearish_cross', 'no_cross', NULL)),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
            UNIQUE(ohlcv_id)
        )
    """)
    
    # Create PSAR table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS psar_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ohlcv_id INTEGER NOT NULL,
            psar DECIMAL(20, 8),
            psar_trend INTEGER CHECK(psar_trend IN (-1, 0, 1)),
            psar_reversal INTEGER CHECK(psar_reversal IN (0,1)),
            step DECIMAL(10, 6),
            max_step DECIMAL(10, 6),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
            UNIQUE(ohlcv_id)
        )
    """)
    
    # Create MACD table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS macd_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ohlcv_id INTEGER NOT NULL,
            macd_line DECIMAL(20, 8),
            signal_line DECIMAL(20, 8),
            macd_histogram DECIMAL(20, 8),
            fast_period INTEGER,
            slow_period INTEGER,
            signal_period INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
            UNIQUE(ohlcv_id)
        )
    """)
    
    # Create metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON ohlcv_data(timeframe)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ichimoku_ohlcv ON ichimoku_data(ohlcv_id)")
    
    conn.commit()
    print(f"Initialized database: {db_path}")
    return conn

def load_json_data(json_path: Path) -> pd.DataFrame:
    """Load OHLCV data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)
    return df

def import_timeframe(conn, df: pd.DataFrame, timeframe: str):
    """Import OHLCV data for a specific timeframe."""
    cursor = conn.cursor()
    
    # Clear existing data for this timeframe
    cursor.execute("DELETE FROM ohlcv_data WHERE timeframe = ?", (timeframe,))
    
    # Insert new data
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO ohlcv_data 
            (timestamp, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row['date'].strftime('%Y-%m-%d %H:%M:%S'),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume']),
            timeframe
        ))
    
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM ohlcv_data WHERE timeframe = ?", (timeframe,))
    count = cursor.fetchone()[0]
    print(f"  Imported {count} {timeframe} candles")

def calculate_ichimoku(df: pd.DataFrame, tenkan_period=9, kijun_period=26, 
                       senkou_b_period=52, displacement=26) -> pd.DataFrame:
    """Calculate Ichimoku indicators."""
    # Tenkan-sen
    df['tenkan_sen'] = (df['high'].rolling(tenkan_period).max() + 
                         df['low'].rolling(tenkan_period).min()) / 2
    
    # Kijun-sen
    df['kijun_sen'] = (df['high'].rolling(kijun_period).max() + 
                        df['low'].rolling(kijun_period).min()) / 2
    
    # Senkou Span A
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
    
    # Senkou Span B
    df['senkou_span_b'] = ((df['high'].rolling(senkou_b_period).max() + 
                             df['low'].rolling(senkou_b_period).min()) / 2).shift(displacement)
    
    # Chikou Span
    df['chikou_span'] = df['close'].shift(-displacement)
    
    # Cloud color
    df['cloud_color'] = df.apply(
        lambda row: 'green' if pd.notnull(row['senkou_span_a']) and pd.notnull(row['senkou_span_b']) and
        row['senkou_span_a'] > row['senkou_span_b'] else 
        'red' if pd.notnull(row['senkou_span_a']) and pd.notnull(row['senkou_span_b']) and
        row['senkou_span_a'] < row['senkou_span_b'] else None, axis=1)
    
    # Price position
    df['price_position'] = df.apply(
        lambda row: 'above_cloud' if pd.notnull(row['senkou_span_a']) and pd.notnull(row['senkou_span_b']) and
        row['close'] > max(row['senkou_span_a'], row['senkou_span_b'])
        else 'below_cloud' if pd.notnull(row['senkou_span_a']) and pd.notnull(row['senkou_span_b']) and
        row['close'] < min(row['senkou_span_a'], row['senkou_span_b'])
        else 'in_cloud' if pd.notnull(row['senkou_span_a']) and pd.notnull(row['senkou_span_b'])
        else None, axis=1)
    
    return df

def save_ichimoku_to_db(conn, df: pd.DataFrame, timeframe: str):
    """Save calculated Ichimoku indicators to database."""
    cursor = conn.cursor()
    
    # Get OHLCV IDs
    cursor.execute("SELECT id, timestamp FROM ohlcv_data WHERE timeframe = ? ORDER BY timestamp", (timeframe,))
    ohlcv_ids = {row[1]: row[0] for row in cursor.fetchall()}
    
    # Clear existing Ichimoku data for this timeframe
    cursor.execute("""
        DELETE FROM ichimoku_data 
        WHERE ohlcv_id IN (SELECT id FROM ohlcv_data WHERE timeframe = ?)
    """, (timeframe,))
    
    # Insert Ichimoku data
    for _, row in df.iterrows():
        timestamp_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
        ohlcv_id = ohlcv_ids.get(timestamp_str)
        if not ohlcv_id:
            continue
        
        cursor.execute("""
            INSERT OR REPLACE INTO ichimoku_data
            (ohlcv_id, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, 
             chikou_span, cloud_color, price_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ohlcv_id,
            float(row['tenkan_sen']) if pd.notnull(row['tenkan_sen']) else None,
            float(row['kijun_sen']) if pd.notnull(row['kijun_sen']) else None,
            float(row['senkou_span_a']) if pd.notnull(row['senkou_span_a']) else None,
            float(row['senkou_span_b']) if pd.notnull(row['senkou_span_b']) else None,
            float(row['chikou_span']) if pd.notnull(row['chikou_span']) else None,
            row['cloud_color'],
            row['price_position']
        ))
    
    conn.commit()
    cursor.execute("""
        SELECT COUNT(*) FROM ichimoku_data 
        WHERE ohlcv_id IN (SELECT id FROM ohlcv_data WHERE timeframe = ?)
    """, (timeframe,))
    count = cursor.fetchone()[0]
    print(f"  Saved {count} Ichimoku records for {timeframe}")

def main():
    """Main import function."""
    print("="*60)
    print("Commodity Data Import to SQLite (Unified Schema)")
    print("="*60)
    
    for folder_name, db_name in COMMODITIES.items():
        print(f"\nProcessing {folder_name}...")
        
        # Paths
        commodity_dir = COMMODITY_DATA_DIR / folder_name
        db_path = BACKTESTBOT_DATA_DIR / db_name
        
        # Initialize database
        conn = init_database(db_path)
        
        # Import 1d data
        json_1d = commodity_dir / "1d.json"
        if json_1d.exists():
            print(f"  Loading 1d data from {json_1d}...")
            df_1d = load_json_data(json_1d)
            print(f"  Loaded {len(df_1d)} 1d candles")
            import_timeframe(conn, df_1d, '1d')
            
            # Calculate and save Ichimoku for 1d
            print(f"  Calculating Ichimoku for 1d...")
            df_1d = calculate_ichimoku(df_1d)
            save_ichimoku_to_db(conn, df_1d, '1d')
        else:
            print(f"  WARNING: {json_1d} not found!")
        
        # Import 4h data
        json_4h = commodity_dir / "4h.json"
        if json_4h.exists():
            print(f"  Loading 4h data from {json_4h}...")
            df_4h = load_json_data(json_4h)
            print(f"  Loaded {len(df_4h)} 4h candles")
            import_timeframe(conn, df_4h, '4h')
            
            # Calculate and save Ichimoku for 4h
            print(f"  Calculating Ichimoku for 4h...")
            df_4h = calculate_ichimoku(df_4h)
            save_ichimoku_to_db(conn, df_4h, '4h')
        else:
            print(f"  WARNING: {json_4h} not found!")
        
        # Update metadata
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ('symbol', folder_name))
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ('asset_class', 'commodity'))
        conn.commit()
        
        conn.close()
        print(f"  ✓ Completed {folder_name}")
    
    print("\n" + "="*60)
    print("Import completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
