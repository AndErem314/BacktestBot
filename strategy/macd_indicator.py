"""
MACD (Moving Average Convergence Divergence) indicator computation utilities.

Returns MACD line, signal line, and histogram per bar.
Default parameters: fast=12, slow=26, signal=9
"""
from typing import Tuple
import pandas as pd
import numpy as np


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD indicator for a price DataFrame.

    Args:
        df: DataFrame with column ['close'] indexed by timestamp
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        DataFrame with columns ['macd_line','signal_line','macd_histogram'] aligned to df index
    """
    if 'close' not in df.columns:
        raise ValueError("compute_macd requires 'close' column")
    if len(df) == 0:
        return pd.DataFrame(index=df.index, columns=['macd_line', 'signal_line', 'macd_histogram'])

    close = df['close'].astype(float)
    
    # Calculate EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # MACD line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal line = EMA of MACD line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram = MACD line - Signal line
    macd_histogram = macd_line - signal_line
    
    out = pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_histogram': macd_histogram
    }, index=df.index)
    
    return out
