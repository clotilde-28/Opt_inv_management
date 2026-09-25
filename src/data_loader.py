"""
Data loading and preprocessing module for Tardis.dev high-frequency order book and trade datasets.
Ensures strict causal alignment using pandas.merge_asof without look-ahead bias.
"""

from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np


class TardisDataLoader:
    """
    Parser and causal merger for Tardis.dev tick-level order book and trades data.
    """

    def __init__(self, book_ticker_path: str, trades_path: str):
        """
        Initialize the loader with filepaths to the book_ticker and trades files.
        """
        if not os.path.exists(book_ticker_path):
            raise FileNotFoundError(f"Book ticker file not found: {book_ticker_path}")
        if not os.path.exists(trades_path):
            raise FileNotFoundError(f"Trades file not found: {trades_path}")
            
        self.book_ticker_path = book_ticker_path
        self.trades_path = trades_path

    def load_raw_data(
        self, 
        nrows_trades: Optional[int] = None, 
        nrows_book: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw trades and book ticker datasets selecting strictly relevant columns.
        
        Book ticker columns: timestamp, ask_price, ask_amount, bid_price, bid_amount
        Trades columns: timestamp, price, amount, side
        """
        book_cols = ["timestamp", "ask_price", "ask_amount", "bid_price", "bid_amount"]
        trades_cols = ["timestamp", "price", "amount", "side"]
        
        trades_df = pd.read_csv(
            self.trades_path,
            usecols=trades_cols,
            nrows=nrows_trades,
            dtype={
                "timestamp": np.int64,
                "price": np.float64,
                "amount": np.float64,
                "side": "category",
            }
        )
        
        book_df = pd.read_csv(
            self.book_ticker_path,
            usecols=book_cols,
            nrows=nrows_book,
            dtype={
                "timestamp": np.int64,
                "ask_price": np.float64,
                "ask_amount": np.float64,
                "bid_price": np.float64,
                "bid_amount": np.float64,
            }
        )
        
        # Sort explicitly by timestamp to guarantee ascending order for merge_asof
        trades_df = trades_df.sort_values("timestamp").reset_index(drop=True)
        book_df = book_df.sort_values("timestamp").reset_index(drop=True)
        
        return trades_df, book_df

    def load_and_merge(
        self, 
        nrows_trades: Optional[int] = None, 
        nrows_book: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Merge trades and book ticker data causally using merge_asof(direction='backward').
        
        This matches each trade with the most recent top-of-book state strictly before
        or at the trade execution timestamp, eliminating look-ahead bias.
        """
        trades_df, book_df = self.load_raw_data(
            nrows_trades=nrows_trades, 
            nrows_book=nrows_book
        )
        
        # Merge causality: trades is left, book is right, direction='backward'
        merged = pd.merge_asof(
            trades_df,
            book_df,
            on="timestamp",
            direction="backward",
            suffixes=("", "_book")
        )
        
        # Drop initial trades occurring prior to the first available book update
        merged = merged.dropna(subset=["ask_price", "bid_price"]).reset_index(drop=True)
        
        # Add human-readable datetime index
        merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="us")
        
        # Derived microstructure quantities
        merged["mid_price"] = (merged["ask_price"] + merged["bid_price"]) / 2.0
        merged["book_spread"] = merged["ask_price"] - merged["bid_price"]
        
        # Elapsed time dt between consecutive trade events in seconds
        dt_us = np.diff(merged["timestamp"].values, prepend=merged["timestamp"].values[0])
        merged["dt"] = np.clip(dt_us / 1e6, 0.0, 5.0) # max 5s pause clip for stability
        
        # Trade penetration distance from the prevailing mid-price
        is_buy = merged["side"] == "buy"
        merged["trade_distance"] = np.where(
            is_buy,
            merged["price"] - merged["mid_price"],
            merged["mid_price"] - merged["price"]
        )
        
        return merged
