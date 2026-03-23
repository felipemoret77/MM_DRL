#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 21:53:45 2025

@author: felipemoret
"""

import numpy as np
import pandas as pd
from typing import Tuple

def pre_process_lob_data(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Basic preprocessing for LOBSTER-style messages/orderbook data.

    Steps:
      1. Map numeric Direction / Type to human-readable strings.
      2. Rename best bid/ask index columns to PriceBestBid / PriceBestAsk.
      3. Attach best bid/ask sizes (level 1) from the order book.
      4. Compute Order Book Imbalance (OBI).
      5. Assert messages and orderbook have the same length.
      6. Chronologically split into TRAIN / TEST subsets.
    """
    # 1) Map Direction / Type to strings
    def _map_directions(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        type_str = df["Type"].astype(str)
        direction_str = np.full(len(df), "", dtype=object)

        # Market Orders
        mask_mo = type_str == "MO"
        direction_str[mask_mo & (df["Direction"] == -1)] = "sell"
        direction_str[mask_mo & (df["Direction"] == +1)] = "buy"

        # Limit Orders
        mask_lo = type_str == "LO"
        direction_str[mask_lo & (df["Direction"] == -1)] = "ask"
        direction_str[mask_lo & (df["Direction"] == +1)] = "bid"

        # Cancels (if exist)
        mask_c = type_str == "C"
        direction_str[mask_c & (df["Direction"] == -1)] = "cancel_ask"
        direction_str[mask_c & (df["Direction"] == +1)] = "cancel_bid"

        df["Direction"] = direction_str
        return df

    messages = _map_directions(messages)

    # 2) Rename best bid/ask index columns
    rename_map = {}
    if "IndBestBid" in messages.columns:
        rename_map["IndBestBid"] = "PriceBestBid"
    if "IndBestAsk" in messages.columns:
        rename_map["IndBestAsk"] = "PriceBestAsk"
    if rename_map:
        messages = messages.rename(columns=rename_map)

    # 3) Attach best bid/ask sizes (level 1) from orderbook
    if "BidSize_1" in orderbook.columns and "AskSize_1" in orderbook.columns:
        messages["SizeBestBid"] = orderbook["BidSize_1"].to_numpy()
        messages["SizeBestAsk"] = orderbook["AskSize_1"].to_numpy()
    else:
        raise KeyError(
            "orderbook must contain columns 'BidSize_1' and 'AskSize_1' "
            "to compute SizeBestBid / SizeBestAsk."
        )

    # 4) Compute Order Book Imbalance (OBI)
    den = messages["SizeBestBid"] + messages["SizeBestAsk"]
    messages["OBI"] = np.divide(
        messages["SizeBestBid"] - messages["SizeBestAsk"],
        den,
        out=np.zeros_like(den, dtype=float),
        where=den != 0,
    )

    # 5) Sanity check: same length
    if len(messages) != len(orderbook):
        raise ValueError(
            f"Messages and Orderbook must have the same number of rows! "
            f"Got len(messages)={len(messages)}, len(orderbook)={len(orderbook)}."
        )

    # 6) Chronological TRAIN / TEST split
    split_index = int(train_frac * len(messages))

    messages_train = messages.iloc[:split_index].reset_index(drop=True)
    messages_test = messages.iloc[split_index:].reset_index(drop=True)

    orderbook_train = orderbook.iloc[:split_index].reset_index(drop=True)
    orderbook_test = orderbook.iloc[split_index:].reset_index(drop=True)

    print(f"Train set: {len(messages_train)} rows")
    print(f"Test  set: {len(messages_test)} rows")

    return messages_train, messages_test, orderbook_train, orderbook_test
