#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight LOBSTER -> simulator preprocessing utilities.

The output schema is compatible with `MM_LOB_SIM.backtest_LOB_with_MM`.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _infer_levels(orderbook: pd.DataFrame) -> int:
    levels = 0
    for c in orderbook.columns:
        cl = c.lower()
        if cl.startswith("ask_price_"):
            try:
                levels = max(levels, int(cl.split("_")[-1]))
            except ValueError:
                pass
        elif cl.startswith("askprice_"):
            try:
                levels = max(levels, int(cl.split("_")[-1]))
            except ValueError:
                pass
    return levels


def _price_to_index(price: pd.Series, *, base_price: float, tick_size: float, grid_center: int) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce").astype(float)
    out = np.rint((p - float(base_price)) / float(tick_size)).astype(float) + int(grid_center)
    out = np.where(np.isfinite(out), out, -1.0)
    return pd.Series(out.astype(np.int64), index=price.index)


def convert_lobster_orderbook_to_sim(
    orderbook: pd.DataFrame,
    tick_size: float = 0.01,
    grid_center: int = 50,
    base_price: Optional[float] = None,
    n_levels_to_store: int = 5,
) -> pd.DataFrame:
    """
    Convert LOBSTER-like order book to simulator format.

    Output columns:
      AskPrice_k, AskSize_k (k=1..N), then BidPrice_k, BidSize_k (k=1..N)
    """
    if len(orderbook) == 0:
        return pd.DataFrame()

    n_levels = min(max(1, int(n_levels_to_store)), max(1, _infer_levels(orderbook)))

    ask1 = _first_existing(orderbook, ["ask_price_1", "AskPrice_1"])
    bid1 = _first_existing(orderbook, ["bid_price_1", "BidPrice_1"])
    if ask1 is None or bid1 is None:
        raise ValueError("orderbook must contain level-1 ask/bid price columns.")

    if base_price is None:
        mid0 = 0.5 * (
            float(pd.to_numeric(orderbook[ask1], errors="coerce").iloc[0])
            + float(pd.to_numeric(orderbook[bid1], errors="coerce").iloc[0])
        )
        base_price = mid0

    out: Dict[str, pd.Series] = {}

    # Ask side first (legacy notebook layout)
    for lvl in range(1, n_levels + 1):
        ask_p = _first_existing(orderbook, [f"ask_price_{lvl}", f"AskPrice_{lvl}"])
        ask_s = _first_existing(orderbook, [f"ask_size_{lvl}", f"AskSize_{lvl}"])
        if ask_p is None or ask_s is None:
            continue

        out[f"AskPrice_{lvl}"] = _price_to_index(
            orderbook[ask_p], base_price=base_price, tick_size=tick_size, grid_center=grid_center
        )
        out[f"AskSize_{lvl}"] = pd.to_numeric(orderbook[ask_s], errors="coerce").fillna(0.0).astype(float)

    # Bid side
    for lvl in range(1, n_levels + 1):
        bid_p = _first_existing(orderbook, [f"bid_price_{lvl}", f"BidPrice_{lvl}"])
        bid_s = _first_existing(orderbook, [f"bid_size_{lvl}", f"BidSize_{lvl}"])
        if bid_p is None or bid_s is None:
            continue

        out[f"BidPrice_{lvl}"] = _price_to_index(
            orderbook[bid_p], base_price=base_price, tick_size=tick_size, grid_center=grid_center
        )
        out[f"BidSize_{lvl}"] = pd.to_numeric(orderbook[bid_s], errors="coerce").fillna(0.0).astype(float)

    return pd.DataFrame(out, index=orderbook.index).reset_index(drop=True)


def convert_lobster_to_sim(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    tick_size: float = 0.01,
    grid_center: int = 50,
    base_price: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert LOBSTER-like messages to simulator event tape format.

    Output columns:
      Time, Type, Price, Direction, Size, Spread, MidPrice, Shift, Return,
      TotNumberBidOrders, TotNumberAskOrders, IndBestBid, IndBestAsk

    Notes
    -----
    - LOBSTER execution direction (Type 4/5) indicates the passive side.
      This converter flips execution directions to aggressor-side convention
      expected by `MM_LOB_SIM.backtest_LOB_with_MM`:
        +1 = buy aggressor, -1 = sell aggressor.
    - Numeric time columns are preserved as floats (seconds), avoiding
      accidental nanosecond timestamp conversion.
    """
    if len(messages) == 0 or len(orderbook) == 0:
        return pd.DataFrame()

    n = min(len(messages), len(orderbook))
    msg = messages.iloc[:n].copy().reset_index(drop=True)
    ob = orderbook.iloc[:n].copy().reset_index(drop=True)

    ob_sim = convert_lobster_orderbook_to_sim(
        ob,
        tick_size=tick_size,
        grid_center=grid_center,
        base_price=base_price,
        n_levels_to_store=_infer_levels(ob),
    )

    if base_price is None:
        ask1_col = _first_existing(ob, ["ask_price_1", "AskPrice_1"])
        bid1_col = _first_existing(ob, ["bid_price_1", "BidPrice_1"])
        if ask1_col is None or bid1_col is None:
            raise ValueError("orderbook must contain level-1 ask/bid price columns.")
        ask1 = pd.to_numeric(ob[ask1_col], errors="coerce")
        bid1 = pd.to_numeric(ob[bid1_col], errors="coerce")
        base_price = float((ask1.iloc[0] + bid1.iloc[0]) / 2.0)

    # Type mapping: LO=0, execution=1, cancel/delete=2, halt/other=3
    type_col = _first_existing(msg, ["type", "Type"])
    if type_col is None:
        raise ValueError("messages must contain a type column.")

    type_map = {
        "new_limit_order": 0,
        "execute_visible": 1,
        "execute_hidden": 1,
        "auction_trade": 3,
        "cancel_order": 2,
        "delete_order": 2,
        "trading_halt": 3,
        0: 0,
        1: 0,
        2: 2,
        3: 2,
        4: 1,
        5: 1,
        6: 3,
        7: 3,
    }
    type_raw = msg[type_col]
    type_sim = type_raw.map(type_map).fillna(3).astype(int)

    # Direction mapping
    dir_col = _first_existing(msg, ["direction", "Direction"])
    dir_map = {"buy": 1, "sell": -1, 1: 1, -1: -1}
    if dir_col is None:
        dir_raw = pd.Series(np.zeros(n, dtype=int))
    else:
        dir_raw = msg[dir_col].map(dir_map).fillna(0).astype(int)

    # LOBSTER execution direction (Type 4/5 -> sim Type 1) is passive-side.
    # Convert only executions to aggressor-side for the simulator:
    # +1 buy aggressor, -1 sell aggressor.
    dir_sim = pd.Series(
        np.where(type_sim.to_numpy() == 1, -dir_raw.to_numpy(), dir_raw.to_numpy()),
        index=msg.index,
    ).astype(int)

    # Size / time / price
    size_col = _first_existing(msg, ["size", "Size"])
    time_col = _first_existing(msg, ["time", "Time"])
    price_col = _first_existing(msg, ["price", "Price"])

    size_sim = pd.to_numeric(msg[size_col], errors="coerce").fillna(1.0).astype(float) if size_col else pd.Series(
        np.ones(n, dtype=float)
    )
    if time_col is None:
        time_sim = pd.Series(np.arange(n, dtype=float))
    else:
        time_raw = msg[time_col]
        if np.issubdtype(time_raw.dtype, np.datetime64):
            time_sim = pd.to_datetime(time_raw, errors="coerce")
        else:
            time_num = pd.to_numeric(time_raw, errors="coerce")
            # Keep numeric time (typical LOBSTER seconds) as float; otherwise parse datetime.
            if float(time_num.notna().mean()) >= 0.95:
                time_sim = time_num.astype(float)
            else:
                time_sim = pd.to_datetime(time_raw, errors="coerce")

    if price_col is None:
        raise ValueError("messages must contain a price column.")
    price_sim = _price_to_index(msg[price_col], base_price=base_price, tick_size=tick_size, grid_center=grid_center)

    bid_col = _first_existing(ob_sim, ["BidPrice_1"])
    ask_col = _first_existing(ob_sim, ["AskPrice_1"])
    bid_sz_col = _first_existing(ob_sim, ["BidSize_1"])
    ask_sz_col = _first_existing(ob_sim, ["AskSize_1"])
    if any(c is None for c in (bid_col, ask_col, bid_sz_col, ask_sz_col)):
        raise ValueError("Converted orderbook is missing level-1 bid/ask columns.")

    ind_best_bid = pd.to_numeric(ob_sim[bid_col], errors="coerce").fillna(-1).astype(int)
    ind_best_ask = pd.to_numeric(ob_sim[ask_col], errors="coerce").fillna(-1).astype(int)
    spread = (ind_best_ask - ind_best_bid).astype(int)
    mid = 0.5 * (ind_best_ask.astype(float) + ind_best_bid.astype(float))
    ret = mid.diff().fillna(0.0)

    out = pd.DataFrame(
        {
            "Time": time_sim,
            "Type": type_sim.astype(int),
            "Price": price_sim.astype(int),
            "Direction": dir_sim.astype(int),
            "Size": size_sim.astype(float),
            "Spread": spread.astype(int),
            "MidPrice": mid.astype(float),
            "Shift": np.zeros(n, dtype=int),
            "Return": ret.astype(float),
            "TotNumberBidOrders": pd.to_numeric(ob_sim[bid_sz_col], errors="coerce").fillna(0.0).astype(float),
            "TotNumberAskOrders": pd.to_numeric(ob_sim[ask_sz_col], errors="coerce").fillna(0.0).astype(float),
            "IndBestBid": ind_best_bid.astype(int),
            "IndBestAsk": ind_best_ask.astype(int),
        }
    )

    id_col = _first_existing(msg, ["order_id", "OrderID", "ID"])
    if id_col is not None:
        out.insert(2, "ID", pd.to_numeric(msg[id_col], errors="coerce").fillna(-1).astype(int))

    # Preserve raw LOBSTER type (1-7) for exact FIFO queue tracking.
    # The sim "Type" column collapses types 2/3 into 2, losing the
    # distinction needed for per-order queue reconstruction.
    out["LobsterType"] = pd.to_numeric(type_raw, errors="coerce").fillna(0).astype(int)

    return out
