#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGU2.py  --  Spread Gap Unit 2: LSTM-based Pseudo-Mid Trend Predictor
======================================================================

Architecture-first refactor of the SGU-2 LSTM script into an importable module.

Purpose and Theory
------------------
SGU2 is the second stage ("Spread Gap Unit 2") in a market-making Deep
Reinforcement Learning (DRL) pipeline.  While SGU1 engineers cross-sectional
features from each LOB (Limit Order Book) snapshot window (e.g., order
imbalance, VWAP ratios, spread statistics), SGU2 focuses on **predicting the
short-term trend** of a "pseudo-mid price" one window ahead.

The pseudo-mid price (defined in Appendix B of the reference paper) is a
trade-execution-weighted midpoint that reflects where aggressive orders
actually transacted during a window, rather than the simple (ask+bid)/2.
Specifically:

    m_i = 0.5 * (max_buy_price_i + min_sell_price_i)

with fallback logic when only one side (or neither) has executions during
the window (see ``compute_labels_realized_trend`` for full details).

The *label* (prediction target) is the return of the pseudo-mid between
consecutive windows:

    y_i = (m_i - m_{i-1}) / m_{i-1}        (simple return, default)
    y_i = log(m_i) - log(m_{i-1})           (log-return variant)

This label captures the "realized trend" of execution prices, which is more
informative for a market-maker than the raw mid-price change because it
incorporates the actual aggressiveness of market participants.

LSTM Architecture
-----------------
An LSTM (Long Short-Term Memory) recurrent neural network is used to predict
y_{t+1} from the past T_SEQ windows of features.  The input tensor for each
sample has shape (T_SEQ, D), where D = (number of SGU1 features) + 1 (the
SGU2 lagged return).  At each historical time step t in the sequence:

    - The first D_sgu1 channels are the SGU1 engineered features for that
      window (e.g., order imbalance, spread, volume metrics).
    - The last channel is the pseudo-mid return at that window (extracted
      from the lagged-return columns built by ``compute_features_SGU2``).

The LSTM processes the sequence and the hidden state at the final time step
is passed through a dropout layer and a linear head to produce a scalar
prediction (the normalized y_{t+1}).

Pipeline Overview
-----------------
1. **Label computation** (``compute_labels_realized_trend``):
   Iterates over LOB windows, computes the pseudo-mid per window, and
   derives inter-window returns.

2. **Feature construction** (``compute_features_SGU2``):
   Builds lagged return columns (label_lag1 ... label_lagL_max) that become
   the SGU2-specific features.

3. **Sequence building** (``build_sgu2_multi_feature_sequences``):
   Merges SGU1 cross-sectional features with SGU2 lagged returns into a 3D
   tensor (N, T_SEQ, D) with proper X_t -> y_{t+1} alignment.

4. **Normalization**:
   Features are flattened from 3D (N, T, F) to 2D (N*T, F) so that
   ``StandardScaler`` can compute per-feature mean/std.  The scaler is fit
   on the training split only to prevent data leakage.  Targets are also
   z-scored.

5. **Bucketization & sample weighting**:
   Absolute label values are split into small / medium / large buckets
   using quantiles of the non-zero training labels.  Each bucket receives
   a configurable weight (W_SMALL, W_MEDIUM, W_LARGE) to focus the loss
   on the more "tradable" (medium & large) moves.

6. **LHS hyperparameter search** (``train_sgu2_lstm_pipeline``):
   Latin Hypercube Sampling explores combinations of learning rate,
   hidden units, dropout, L2 regularization, batch size, etc.  Each
   candidate is trained with early stopping on a validation split.

7. **Warm-start refit**:
   The best hyperparameters are used to refit the model on the combined
   train + validation data (80%).  The refit is warm-started from the
   best model's weights (rather than random initialization) to preserve
   learned representations and converge faster.

8. **Save / Load / Cache utilities**:
   Trained models, scalers, hyperparameters, and metrics can be persisted
   to disk with content-addressed cache keys derived from a metadata dict
   (symbol, dates, delta_t, etc.), enabling reproducible re-loading.

Key goals
---------
- Keep the original computations/logic the same (only reorganized into functions).
- Provide a single high-level entrypoint that returns the trained model(s),
  scalers, and metadata (similar spirit to SGU1.py).
- Add small, optional utilities for caching and saving/loading artifacts.
- Do not execute training on import.

Inputs expected
---------------
This module assumes you already computed the feature DataFrames and label DataFrames:

    features_SGU_1 : pd.DataFrame  (must include "window_id" + engineered features)
    features_SGU_2 : pd.DataFrame  (must include "window_id" + label_lag* columns)
    labels_SGU_2_df: pd.DataFrame  (must include "window_id" + "label")

Optionally (for SGU1-only LSTM mode):
    labels_SGU_1_df: pd.DataFrame  (must include "window_id" + "label")

LOBSTER Data Conventions
------------------------
- Column names use CamelCase: AskPrice_1, BidPrice_1, Type, Direction, Price, Size.
- Direction == -1 means a sell limit order was executed, i.e., the aggressor
  was a BUYER (buy market order lifted the sell LO).
- Direction == +1 means a buy limit order was executed, i.e., the aggressor
  was a SELLER (sell market order hit the buy LO).
- Type == 4 corresponds to "execute_visible", which is the default execution
  type filtered in this module.

Notes
-----
The original script implemented a *unified* pipeline for SGU-1 and SGU-2.
We preserve this behavior via the `use_sgu1_only` flag.

"""

from __future__ import annotations

# ============================================================================
# Standard library imports
# ============================================================================
import contextlib      # For nullcontext() used in eval-mode gradient suppression
import warnings        # For user-facing warnings (e.g., security, cross-day masking)
import gc              # Garbage collection to free GPU memory between HP trials
import time            # Lightweight wall-clock timing for HP trials
import os              # CPU thread count for runtime stability tuning
import json            # Serialize artifacts metadata to human-readable JSON
import pickle          # Serialize scikit-learn scalers (binary format)
import random          # Python-level RNG seeding for reproducibility
import re              # Regular expressions for parsing lag column names
import hashlib         # SHA-1 hashing for content-addressed cache keys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================================
# Third-party imports
# ============================================================================
import numpy as np
import pandas as pd
# sliding_window_view creates zero-copy rolling windows over NumPy arrays,
# which is essential for efficiently building the (N, T_SEQ, F) tensor
# without explicit Python loops.
from numpy.lib.stride_tricks import sliding_window_view

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# StandardScaler is used to z-score normalize both features and targets.
# It is fit ONLY on the training split to prevent information leakage
# from validation/test data into the model.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tqdm import tqdm

# top_moves_windows is a helper from the LOB processor that partitions the
# order book timeline into non-overlapping windows of delta_t events each.
from LOB_processor import top_moves_windows


# ============================================================================
# SECTION 1: SGU2 LABEL AND FEATURE COMPUTATION
# ============================================================================
# This section contains the two main data-preparation functions:
#   1. compute_labels_realized_trend  -- computes pseudo-mid prices and labels
#   2. compute_features_SGU2          -- builds lagged return features
# These are called by the runner script BEFORE the LSTM training pipeline.
# ============================================================================

def compute_labels_realized_trend(
    orderbook: pd.DataFrame,
    messages: pd.DataFrame,
    delta_t: int,
    tick_size: Optional[float] = None,
    exec_types: Tuple = (4,),
    use_log_ret: bool = False,
    round_pseudo_mid_to_tick: bool = False,
    windows: Optional[List[Tuple[int, int]]] = None,
    window_day_ids: Optional[np.ndarray] = None,  # NEW: optional per-window day ID array
) -> pd.DataFrame:
    """
    Compute per-window labels from a pseudo-mid price m_i (App. B).

    Pseudo-mid m_i:
        0.5*(max P^buy_i + min P^sell_i)                  if both exist
        0.5*(avg ask_i + min P^sell_i)                    if only sells
        0.5*(max P^buy_i + avg bid_i)                     if only buys
        0.5*(avg ask_i + avg bid_i)                       if no trades

    Label:
        if use_log_ret is False (default):  y_i = (m_i - m_{i-1}) / m_{i-1}
        if use_log_ret is True:             y_i = log(m_i) - log(m_{i-1})

    LOBSTER Direction convention (passive side, integer):
        Direction == -1 : sell LO was executed → aggressor was BUYER
        Direction == +1 : buy  LO was executed → aggressor was SELLER

    round_pseudo_mid_to_tick:
        If True, pseudo-mid m_i is rounded to the nearest tick before
        computing returns. Default is False to avoid label quantization.

    windows : list of (int, int), optional
        Pre-computed day-safe windows (from ``top_moves_windows_by_day`` or
        equivalent).  When provided, the function uses these directly instead
        of rebuilding from the orderbook.  This is **critical** for multi-day
        concatenated data to avoid cross-overnight contamination of labels.

    window_day_ids : np.ndarray, optional
        Integer array of length len(windows) where window_day_ids[i] is the
        calendar-day identifier (e.g. 0, 1, 2, ...) of the day that window i
        belongs to.  When provided, any inter-window return that spans a day
        boundary (window_day_ids[i] != window_day_ids[i-1]) is set to NaN,
        preventing economically meaningless overnight returns from entering
        the label series.
    """
    # ---- Window source: use pre-computed if provided, else build from scratch ----
    # WHY: For multi-day datasets, windows should be pre-computed per day to
    # avoid a window that spans the overnight boundary (e.g., last events of
    # day 1 mixed with first events of day 2).  When `windows` is None, we
    # fall back to building them from the raw orderbook, which is only safe
    # for single-day data.
    if windows is None:
        windows: List[Tuple[int, int]] = top_moves_windows(orderbook, delta_t)

    # ---------------------------
    # Pre-convert all needed columns to NumPy arrays once for performance.
    # ---------------------------
    # WHY: Repeated pandas indexing inside a per-window loop is very slow.
    # Converting to NumPy upfront and slicing with integer indices is orders
    # of magnitude faster for thousands of windows.
    #
    # LOBSTER CamelCase convention:
    #   - "Type" column: integer event type; Type == 4 means "execute_visible"
    #     (a visible limit order was matched/executed).
    #   - "Direction" column: indicates the PASSIVE side of the trade:
    #       Direction == -1 => a sell limit order was executed => aggressor is BUYER
    #       Direction == +1 => a buy  limit order was executed => aggressor is SELLER
    #   - "Price" column: the execution price of the message.
    #   - "AskPrice_1" / "BidPrice_1": best ask/bid from the orderbook snapshot.
    is_exec = messages["Type"].isin(exec_types).to_numpy()
    prices = pd.to_numeric(messages["Price"], errors="coerce").to_numpy()
    dirs = messages["Direction"].to_numpy(dtype=int)

    ask_np = orderbook["AskPrice_1"].to_numpy()
    bid_np = orderbook["BidPrice_1"].to_numpy()

    rows = []
    pseudo_mids = []

    # ---------------------------
    # Main loop: iterate over each window to compute the pseudo-mid price.
    # ---------------------------
    # WHY: Each window represents delta_t consecutive LOB events.  We need to
    # inspect the trades that occurred within each window to determine
    # the maximum buy-aggressor price and the minimum sell-aggressor price,
    # which together form the pseudo-mid.
    for w_id, (start, end) in enumerate(windows):
        idx = slice(start, end)

        # Average best ask and best bid within this window.
        # These serve as FALLBACKS when one or both sides have no trades.
        ask_slice = ask_np[idx]
        bid_slice = bid_np[idx]
        avg_ask = float(np.nanmean(ask_slice)) if ask_slice.size else np.nan
        avg_bid = float(np.nanmean(bid_slice)) if bid_slice.size else np.nan

        # Filter to execution messages only (Type == 4 by default)
        win_exec_mask = is_exec[idx]

        if win_exec_mask.any():
            price_slice = prices[idx]
            dir_slice = dirs[idx]

            # Extract only the prices and directions of executed trades
            price_exec = price_slice[win_exec_mask]
            dir_exec = dir_slice[win_exec_mask]

            n_trades = len(price_exec)

            # LOBSTER direction convention (passive side):
            #   Direction == -1: a sell limit order was hit => the aggressor was a BUYER
            #     => this is a "buy trade" (buyer-initiated)
            #   Direction == +1: a buy limit order was hit  => the aggressor was a SELLER
            #     => this is a "sell trade" (seller-initiated)
            n_buy = int((dir_exec == -1).sum())
            n_sell = int((dir_exec == 1).sum())

            # Separate execution prices by trade direction
            buy_prices  = price_exec[dir_exec == -1]
            sell_prices = price_exec[dir_exec == 1]

            # Remove any NaN prices that may result from parsing issues
            buy_prices  = buy_prices[~np.isnan(buy_prices)]
            sell_prices = sell_prices[~np.isnan(sell_prices)]
        else:
            # No executions in this window at all
            price_exec = np.array([])
            dir_exec = np.array([])
            n_trades = 0
            n_buy = 0
            n_sell = 0
            buy_prices = np.array([])
            sell_prices = np.array([])

        # ---------------------------------------------------------------
        # Pseudo-mid computation with fallback logic (Appendix B)
        # ---------------------------------------------------------------
        # The pseudo-mid m_i is designed to capture where aggressive
        # orders actually transacted.  It uses the MAXIMUM price paid by
        # buy aggressors and the MINIMUM price received by sell aggressors.
        #
        # WHY max for buys, min for sells?  Because these extremes
        # represent the most aggressive fills on each side -- the prices
        # closest to "crossing the spread."  Their midpoint is the best
        # single-number summary of the window's execution-price center.
        #
        # Fallback hierarchy (when one or both sides have no trades):
        #   1. Both sides have trades => use max_buy and min_sell
        #   2. Only sell trades       => substitute avg_ask for max_buy
        #   3. Only buy trades        => substitute avg_bid for min_sell
        #   4. No trades at all       => fall back to avg_ask and avg_bid
        #      (effectively the time-averaged quoted midpoint)
        if buy_prices.size > 0 and sell_prices.size > 0:
            max_buy = float(np.max(buy_prices))
            min_sell = float(np.min(sell_prices))
            m_i = 0.5 * (max_buy + min_sell)
        elif buy_prices.size == 0 and sell_prices.size > 0:
            # No buy aggressors => substitute the average best ask price
            # WHY avg_ask? Because in the absence of buy trades, the best
            # ask is the closest observable proxy for where a buyer WOULD
            # have transacted.
            max_buy = np.nan
            min_sell = float(np.min(sell_prices))
            m_i = 0.5 * (avg_ask + min_sell)
        elif sell_prices.size == 0 and buy_prices.size > 0:
            # No sell aggressors => substitute the average best bid price
            max_buy = float(np.max(buy_prices))
            min_sell = np.nan
            m_i = 0.5 * (max_buy + avg_bid)
        else:
            # No trades at all => pure quoted midpoint
            max_buy = np.nan
            min_sell = np.nan
            m_i = 0.5 * (avg_ask + avg_bid)

        # Optional tick rounding: snap pseudo-mid to the nearest tick.
        # WHY this is off by default: rounding introduces label quantization
        # (many windows map to the same pseudo-mid), which makes the
        # continuous regression target artificially discrete and can
        # degrade LSTM training.
        if round_pseudo_mid_to_tick and (tick_size is not None) and np.isfinite(m_i):
            m_i = round(m_i / tick_size) * tick_size

        pseudo_mids.append(m_i)

        rows.append(dict(
            window_id=w_id, start=start, end=end,
            n_msgs=int(end - start), n_trades=n_trades, n_buy=n_buy, n_sell=n_sell,
            avg_ask=avg_ask, avg_bid=avg_bid, max_buy=max_buy, min_sell=min_sell,
            pseudo_mid=m_i, label=np.nan
        ))

    # ---------------------------------------------------------------
    # Compute inter-window returns from the pseudo-mid series
    # ---------------------------------------------------------------
    # WHY: The label y_i represents the CHANGE in pseudo-mid between
    # consecutive windows.  Window 0 has no predecessor, so its label
    # is NaN.  For window i >= 1:
    #     Simple return:  y_i = (m_i - m_{i-1}) / m_{i-1}
    #     Log return:     y_i = log(m_i) - log(m_{i-1})
    # Both measure the same concept (relative price movement) but log
    # returns are symmetric and additive, which can be preferable for
    # certain downstream models.
    out = pd.DataFrame(rows)
    m = out["pseudo_mid"].to_numpy(dtype=float)

    # Initialize all labels as NaN; only windows with valid predecessors
    # will receive a computed label.
    y = np.full(m.shape, np.nan, dtype=float)
    if len(m) >= 2:
        if use_log_ret:
            # Log-return variant: requires both m_i and m_{i-1} to be
            # finite and strictly positive (log is undefined for <= 0).
            valid = (
                np.isfinite(m[1:]) & np.isfinite(m[:-1]) &
                (m[1:] > 0.0) & (m[:-1] > 0.0)
            )
            y_part = np.zeros_like(m[1:])
            y_part[valid] = np.log(m[1:][valid]) - np.log(m[:-1][valid])
        else:
            # Simple return variant: requires m_{i-1} != 0 to avoid
            # division by zero.
            valid = (
                np.isfinite(m[1:]) & np.isfinite(m[:-1]) &
                (m[:-1] != 0.0)
            )
            y_part = np.zeros_like(m[1:])
            y_part[valid] = (m[1:][valid] - m[:-1][valid]) / m[:-1][valid]

        # Place computed returns at indices 1..N-1; index 0 stays NaN
        # because there is no "previous" window for the first one.
        y[1:] = np.where(valid, y_part, np.nan)

    # ---------------------------------------------------------------
    # MASK cross-day returns to NaN.
    #
    # WHY?  The day-safe windowing (top_moves_windows_by_day) ensures
    # no single window spans a day boundary.  But the RETURN between
    # the last window of day D and the first window of day D+1 still
    # gets computed.  These cross-day returns are economically
    # meaningless because:
    #   1. They span an ~18.5-hour overnight gap (not microstructure).
    #   2. Overnight price changes are driven by news/macro, not LOB.
    #   3. They are typically MUCH larger than intraday returns,
    #      dominating the MSE loss and corrupting the LSTM training.
    #
    # Setting them to NaN ensures they are excluded by the downstream
    # NaN-drop step in sequence building.
    # ---------------------------------------------------------------
    if window_day_ids is not None:
        day_arr = np.asarray(window_day_ids)
        if len(day_arr) == len(y):
            # Cross-day boundary: window i and i-1 on different days.
            cross_day = np.zeros(len(y), dtype=bool)
            cross_day[1:] = day_arr[1:] != day_arr[:-1]
            y[cross_day] = np.nan

    out["label"] = y
    return out

def compute_features_SGU2(
    orderbook: pd.DataFrame,
    messages: pd.DataFrame,
    windows: List[Tuple[int, int]],
    labels_df: pd.DataFrame,
    L_max: int = 10,
    window_day_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build SGU-2 features for an LSTM that predicts realized_trend labels.

    Predictors are raw lagged pseudo-returns y_{i-L}, L = 1,...,L_max,
    where y_i is the per-window realized trend label from labels_df.

    WHY lagged returns as features?
        The pseudo-mid returns encode recent price momentum and
        mean-reversion patterns.  By providing the LSTM with a history of
        L_max lagged returns, we allow it to learn autoregressive dynamics
        (e.g., "three consecutive positive returns are often followed by a
        reversal") without hardcoding any specific time-series model.

    Important:
        - No normalization is applied here.
        - Normalization (e.g. StandardScaler) is handled later in the
          LSTM training script, using only the TRAIN split to avoid leakage.
        - The first L_max rows will have NaN in some lag columns because
          there is no history to look back on.  These NaN rows are removed
          later during the sequence-building step.

    Parameters
    ----------
    orderbook : pd.DataFrame
        Unused here; kept for interface consistency with SGU1's feature
        builder, so both can be called with the same signature.
    messages : pd.DataFrame
        Unused here; kept for interface consistency.
    windows : list of (int, int)
        Windows used to build labels_df. Only length is checked for a
        sanity assertion.
    labels_df : pd.DataFrame
        Must contain columns ["window_id", "label"] with one row per window.
    L_max : int
        Maximum number of lags to compute (1..L_max).  This should be >= T_SEQ
        so that the sequence builder has enough lag columns to fill the full
        history.

    Returns
    -------
    features_SGU_2 : pd.DataFrame
        Columns:
            - "window_id"
            - "label_lag1", "label_lag2", ..., "label_lagL_max"
        Rows are sorted by window_id and aligned with labels_df.
    """
    # Ensure one row per window, sorted in chronological window_id order.
    # WHY drop_duplicates? Defensive: if labels_df somehow has duplicate
    # window_ids (e.g., from a multi-day concat without de-duplication),
    # we keep only the first occurrence to maintain a clean 1:1 mapping.
    lbl = (
        labels_df[["window_id", "label"]]
        .drop_duplicates("window_id")
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    # Consistency check: number of windows MUST match number of labels.
    # A mismatch indicates a data alignment problem where labels and features
    # were computed from different window sets, which would silently produce
    # incorrect lag features and corrupt the entire downstream pipeline.
    if len(windows) != len(lbl):
        raise ValueError(
            f"Window count ({len(windows)}) does not match label count "
            f"({len(lbl)}). This indicates a data alignment problem: "
            f"labels and features were computed from different window sets. "
            f"Ensure both use the same `windows` list."
        )

    # Base output: keep only window_id
    out = lbl[["window_id"]].copy()

    # Compute raw lag features: y_{i-L} for L = 1, 2, ..., L_max.
    # WHY: pd.Series.shift(L) moves the label column down by L positions,
    # so that row i gets the value from row i-L.  This produces the lagged
    # return columns that the LSTM will consume as one dimension of its
    # input tensor (the "SGU2 return lag" channel).
    # Note: the first L rows of label_lagL will be NaN because there are
    # not enough preceding windows to look back L steps.
    for L in range(1, L_max + 1):
        out[f"label_lag{L}"] = lbl["label"].shift(L)

    # --- Mask cross-day lags to NaN ---
    # If window_day_ids is provided, set lagged features to NaN wherever
    # the lagged window belongs to a different trading day than the current
    # window.  Without this, label_lag1 for the first window of day D+1
    # would contain the label from the last window of day D -- a value
    # contaminated by overnight dynamics (different regime, overnight gaps).
    if window_day_ids is not None:
        day_arr = np.asarray(window_day_ids)
        for L in range(1, L_max + 1):
            cross_day = np.zeros(len(out), dtype=bool)
            cross_day[:L] = True  # first L rows have no valid lag
            cross_day[L:] = day_arr[L:] != day_arr[:-L]
            out.loc[cross_day, f"label_lag{L}"] = np.nan

    # No normalization here; lags are returned as-is.
    # Normalization is deferred to the training pipeline where it is fit
    # exclusively on the training split to prevent data leakage.
    return out

# ============================================================================
# SECTION 2: SMALL IO HELPERS (save / load / caching)
# ============================================================================
# These utilities handle persistence of Python objects (scalers via pickle),
# NumPy arrays (via compressed .npz), and metadata (via sidecar JSON).
# They are used both by the main training pipeline (to cache prepared arrays
# and save trained artifacts) and by downstream code that loads a previously
# trained SGU2 model for inference.
#
# WHY separate save/load helpers?  Keeping IO logic in small, testable
# functions makes the main pipeline cleaner and allows callers to persist
# or load individual components (e.g., just the scaler) without running
# the full training pipeline.
# ============================================================================

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    """Create the directory (and parents) if it does not exist, then return it."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj: Any, path: PathLike) -> Path:
    """
    Pickle-save a Python object (e.g. scikit-learn StandardScaler, dicts).

    WHY pickle?  StandardScaler stores fitted mean_/scale_ arrays which
    are needed at inference time to normalize new data consistently with
    the training distribution.  Pickle is the simplest way to persist
    arbitrary Python objects while preserving their internal state.

    Uses HIGHEST_PROTOCOL for compact binary output.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: PathLike) -> Any:
    """Pickle-load a Python object (inverse of save_pickle)."""
    path = Path(path).expanduser().resolve()
    with open(path, "rb") as f:
        return pickle.load(f)


def save_npz_cache(
    path: PathLike,
    *,
    X_all: np.ndarray,
    y_all: np.ndarray,
    feature_cols: List[str],
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Save a lightweight cache of prepared arrays (X_all, y_all, feature names).

    WHY cache arrays?  Building the 3D tensor (N, T_SEQ, D) from raw
    DataFrames involves merging, sliding windows, and NaN filtering,
    which can take minutes for large datasets.  Caching the result lets
    the user skip this step on subsequent runs with the same data.

    This is optional and does not change any computation.

    We store arrays in `.npz` (compressed) and metadata (optional) in a
    sidecar `.json` file (same path, different extension) for human
    readability.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # savez_compressed uses zip compression to reduce disk footprint
    # for the potentially large (N, T, F) float32 tensor.
    np.savez_compressed(
        path,
        X_all=X_all,
        y_all=y_all,
        feature_cols=np.array(feature_cols, dtype=object),
    )

    # Optional sidecar JSON with metadata (e.g., T_SEQ, symbol, dates)
    # so a human can inspect the cache contents without loading it.
    meta_path = None
    if meta is not None:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)

    return path, meta_path


def load_npz_cache(path: PathLike) -> Dict[str, Any]:
    """
    Load a cache previously produced by ``save_npz_cache``.

    Returns a dict with keys: X_all, y_all, feature_cols, meta.
    If no sidecar JSON exists, meta will be None.
    """
    path = Path(path).expanduser().resolve()
    # SECURITY NOTE: allow_pickle=True is required because feature_cols is
    # stored as a NumPy object array (dtype=object).  This means the .npz
    # file could execute arbitrary code if tampered with.  Only load cache
    # files from trusted sources.  A safer alternative would be to store
    # feature_cols in the sidecar JSON instead of inside the .npz.
    with np.load(path, allow_pickle=True) as data:
        X_all = data["X_all"]
        y_all = data["y_all"]
        feature_cols = list(data["feature_cols"].tolist())

    # Attempt to load the sidecar JSON metadata (same base name, .json ext)
    meta_path = path.with_suffix(".json")
    meta = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return dict(X_all=X_all, y_all=y_all, feature_cols=feature_cols, meta=meta)


# ============================================================================
# SECTION 3: REPRODUCIBILITY AND DEVICE RESOLUTION
# ============================================================================
# WHY: Deep learning results can vary across runs due to non-deterministic
# GPU operations, random weight initialization, and data shuffling.
# Fixing all random seeds ensures that, given the same data and
# hyperparameters, the model produces identical results.
# ============================================================================

def set_seed(seed: int = 1337, fast_mode: bool = True) -> None:
    """
    Fix random seeds across all libraries (Python, NumPy, PyTorch) for
    reproducibility.

    Parameters
    ----------
    seed : int
        The seed value shared by all RNGs.
    fast_mode : bool
        If True, cuDNN is allowed to select the fastest convolution algorithm
        even if it is non-deterministic.  This is the default because LSTM
        operations benefit from cuDNN benchmarking and the non-determinism
        is negligible for this use case.  Set to False for bit-exact
        reproducibility (at the cost of ~10-30% slower training).
    """
    # Seed all four RNG sources used during training
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Seeds all GPUs if multi-GPU

    if fast_mode:
        # Allow cuDNN to auto-tune for the fastest algorithm per input size.
        # Slight non-determinism but noticeably faster LSTM forward/backward.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # Strict determinism: same inputs always produce same outputs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve the torch device for model placement.

    If None, automatically selects CUDA when a GPU is available, otherwise
    falls back to MPS (Apple Silicon) when available, and finally CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _configure_cpu_runtime(
    *,
    enable_safe_mode: bool = True,
    cpu_num_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Configure PyTorch CPU runtime knobs for SGU2 stability.

    Some environments can hang inside the first LSTM forward/backward call
    when oneDNN/MKLDNN + aggressive OpenMP threading interact with other
    libraries (e.g., XGBoost).  Safe mode disables MKLDNN and constrains
    thread pools to reduce deadlock risk.
    """
    cfg: Dict[str, Any] = {
        "safe_mode": bool(enable_safe_mode),
        "mkldnn_enabled_before": bool(torch.backends.mkldnn.enabled),
        "mkldnn_enabled_after": bool(torch.backends.mkldnn.enabled),
        "num_threads_before": int(torch.get_num_threads()),
        "num_threads_after": int(torch.get_num_threads()),
        "interop_threads_before": int(torch.get_num_interop_threads()),
        "interop_threads_after": int(torch.get_num_interop_threads()),
    }

    if not enable_safe_mode:
        return cfg

    # 1) Disable MKLDNN for CPU LSTM stability in problematic environments.
    if torch.backends.mkldnn.enabled:
        torch.backends.mkldnn.enabled = False

    # 2) Limit thread pools to avoid OpenMP oversubscription/deadlocks.
    if cpu_num_threads is None:
        cpu_num_threads = max(1, min(4, (os.cpu_count() or 1)))

    with contextlib.suppress(Exception):
        torch.set_num_threads(int(cpu_num_threads))
    with contextlib.suppress(Exception):
        # Can raise if called late in process lifecycle; ignore safely.
        torch.set_num_interop_threads(1)

    cfg["mkldnn_enabled_after"] = bool(torch.backends.mkldnn.enabled)
    cfg["num_threads_after"] = int(torch.get_num_threads())
    cfg["interop_threads_after"] = int(torch.get_num_interop_threads())
    return cfg


# ============================================================================
# SECTION 4: SEQUENCE BUILDERS
# ============================================================================
# These functions transform tabular DataFrames (one row per window) into 3D
# NumPy tensors suitable for LSTM training.
#
# KEY CONCEPT -- X_t -> y_{t+1} alignment:
#   The LSTM's job is to PREDICT the label of the NEXT window given a history
#   of T_SEQ PAST windows.  Therefore:
#     - X[i] = features from windows [i, i+1, ..., i+T_SEQ-1]
#     - y[i] = label at window i+T_SEQ  (the NEXT window after the history)
#   This "one-step-ahead" alignment is critical: the model never sees the
#   target's own features in its input, which prevents trivial look-ahead bias.
#
# Two variants are provided:
#   1. build_sgu1_sequences         -- SGU1 features only (ablation baseline)
#   2. build_sgu2_multi_feature_sequences -- SGU1 + SGU2 lagged returns (full)
# ============================================================================

def build_sgu1_sequences(
    features_SGU_1: pd.DataFrame,
    labels_SGU_1_df: pd.DataFrame,
    T_SEQ: int = 10,
    return_window_ids: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[str]],
    Tuple[np.ndarray, np.ndarray, List[str], np.ndarray],
]:
    """
    Build (X_all, y_all) for SGU-1 LSTM using ONLY SGU-1 engineered features.

    This is the simpler "ablation" variant where the LSTM input contains only
    the cross-sectional LOB features (order imbalance, spread, etc.) without
    the SGU2 lagged-return time series.

    Alignment is:
        X_t (history of T_SEQ windows)  --->  y_{t+1} (label of the NEXT window)

    For each target index i (i >= T_SEQ and i < N_total):
        * History windows: [i-T_SEQ, ..., i-1]
        * Features for history: feats[j] for j in [i-T_SEQ, ..., i-1]
        * Target: label at index i (the *next* window relative to last history).

    IMPORTANT:
      - We DO NOT drop any 'label_lag*' columns from SGU1 here.
      - All columns except 'window_id' and 'label' are used as features.

    Returns
    -------
    X_all : np.ndarray, shape (N_eff, T_SEQ, F)
        The 3D input tensor for the LSTM.
    y_all : np.ndarray, shape (N_eff,)
        The target labels (one per sequence).
    feature_cols : list of str
        Names of the F feature columns in the last axis of X_all.
    target_window_ids : np.ndarray, shape (N_eff,), optional
        Returned only when ``return_window_ids=True``. Each element is the
        raw ``window_id`` corresponding to the target ``y_all[i]``.
    """
    # --- 1) Merge and sort by window_id ---
    # WHY inner merge? We only keep windows that have BOTH features AND labels.
    # Windows without a label (e.g., window 0 which has no predecessor for the
    # return computation) are naturally excluded.
    sgu1 = features_SGU_1.sort_values("window_id").reset_index(drop=True).copy()
    lbls = labels_SGU_1_df.sort_values("window_id").reset_index(drop=True).copy()

    df = (
        sgu1.merge(lbls[["window_id", "label"]], on="window_id", how="inner")
            .sort_values("window_id")
            .reset_index(drop=True)
    )

    # --- 2) Feature columns: all except window_id + label ---
    # WHY exclude 'label'? Because 'label' is the target y -- including it
    # in X would leak future information into the model's input.
    feature_cols = [c for c in df.columns if c not in ["window_id", "label"]]
    if len(feature_cols) == 0:
        raise RuntimeError("No SGU1 feature columns found (besides window_id/label).")

    feats = df[feature_cols].to_numpy(dtype=np.float32)   # (N, F)
    labels = df["label"].to_numpy(dtype=np.float32)       # (N,)
    window_ids = df["window_id"].to_numpy(dtype=np.int64) # (N,)

    N_total, F = feats.shape
    # Need at least one target index i in [T_SEQ, N_total-1] => N_total > T_SEQ.
    if N_total <= T_SEQ:
        raise RuntimeError(
            f"Not enough windows for SGU1 sequences with next-step prediction: "
            f"N_total={N_total}, T_SEQ={T_SEQ}"
        )

    # --- 3) Build rolling histories via sliding_window_view ---
    # sliding_window_view creates a view (zero-copy) of overlapping windows.
    # For a 2D array of shape (N, F) with window_shape=T_SEQ on axis=0:
    #   output shape -> (N - T_SEQ + 1, F, T_SEQ)
    # We then transpose to (N_windows, T_SEQ, F) so that axis 1 = time.
    # WHY sliding_window_view?  It avoids an explicit Python loop over
    # thousands of windows and produces the result in a single vectorized op.
    feats_sw = sliding_window_view(feats, window_shape=T_SEQ, axis=0)
    # rearrange to (N_windows, T_SEQ, F) -- LSTM expects (batch, time, features)
    feats_sw = np.transpose(feats_sw, (0, 2, 1))

    # --- 4) Indices for X_t -> y_{t+1} alignment ---
    #
    # DETAILED EXPLANATION of the alignment:
    #   We want the model to predict label[i] given features from
    #   windows [i-T_SEQ, i-T_SEQ+1, ..., i-1].
    #
    #   In feats_sw, index 0 corresponds to windows [0, 1, ..., T_SEQ-1],
    #   index 1 corresponds to [1, 2, ..., T_SEQ], etc.
    #   So feats_sw[k] = features for windows [k, k+1, ..., k+T_SEQ-1].
    #
    #   For target index i, the history ends at window i-1, so the history
    #   starts at window i-T_SEQ.  The feats_sw index is therefore i-T_SEQ.
    #
    #   i ranges from T_SEQ (first valid target) to N_total-1 (last window).
    #   The feats_sw index = i - T_SEQ ranges from 0 to N_total-1-T_SEQ.
    #
    target_indices = np.arange(T_SEQ, N_total, dtype=int)   # i = T_SEQ .. N_total-1
    window_idx = target_indices - T_SEQ                     # indices into feats_sw

    max_hist_idx = feats_sw.shape[0] - 1                    # = N_total - T_SEQ
    valid_mask = window_idx <= max_hist_idx

    target_indices = target_indices[valid_mask]
    window_idx     = window_idx[valid_mask]

    X_all = feats_sw[window_idx]               # (N_eff, T_SEQ, F)
    y_all = labels[target_indices]             # (N_eff,)  -> label of NEXT window
    target_window_ids = window_ids[target_indices]  # (N_eff,)

    # --- 5) Mask out NaN/inf ---
    # WHY: Early windows may have NaN features (e.g., lagged columns with
    # insufficient history) or NaN labels (window 0 has no return).
    # Keeping these would produce NaN losses and corrupt gradient updates.
    mask_y = np.isfinite(y_all)
    mask_X = np.isfinite(X_all).all(axis=(1, 2))
    mask = mask_y & mask_X

    if not np.any(mask):
        raise RuntimeError("No valid SGU1 sequences (NaNs or inf everywhere).")

    X_all = X_all[mask]
    y_all = y_all[mask]
    target_window_ids = target_window_ids[mask]

    print("Built SGU-1 dataset for LSTM (SGU1 only, X_t → y_{t+1}):")
    print("  X_all shape:", X_all.shape, "(N, T, F)")
    print("  y_all shape:", y_all.shape, "(N,)")
    print("  feature dims:", feature_cols)

    if return_window_ids:
        return X_all, y_all, feature_cols, target_window_ids
    return X_all, y_all, feature_cols


def build_sgu2_multi_feature_sequences(
    features_SGU_1: pd.DataFrame,
    features_SGU_2: pd.DataFrame,
    labels_SGU_2_df: pd.DataFrame,
    T_SEQ: int = 10,
    return_window_ids: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[str]],
    Tuple[np.ndarray, np.ndarray, List[str], np.ndarray],
]:
    """
    Build (X_all, y_all) for SGU-2 LSTM using BOTH:
        - SGU-1 engineered features (features_SGU_1) as cross-sectional features
        - SGU-2 lagged returns as ONE time-series dimension.

    This is the FULL variant that gives the LSTM both microstructure features
    (from SGU1) and autoregressive return history (from SGU2), enabling it
    to learn both cross-sectional and temporal patterns jointly.

    Multi-feature tensor construction:
    For each target at window i (i >= T_SEQ), we construct a history of T_SEQ
    windows [i-T_SEQ, ..., i-1] with:
        * SGU1[t, :] = features_SGU_1 at historical window j = i-T_SEQ + t
          These are the D_sgu1 cross-sectional features (order imbalance,
          spread, VWAP, etc.) computed by SGU1 for each historical window.
        * SGU2[t]    = return of that historical window,
                       taken from lag_k in the row i:
                           j = i - lag_k  <->  lag_k = i - j
                       mapping:
                           t = 0        -> lag_k = T_SEQ  (oldest in history)
                           ...
                           t = T_SEQ-1  -> lag_k = 1      (most recent)
          This is a SINGLE scalar channel containing the pseudo-mid return
          at each historical time step, providing the LSTM with an explicit
          autoregressive signal.
        * y[i] = label(i) (labels_SGU_2_df["label"] at window i).

    Output:
        X_all: (N_eff, T_SEQ, D)  with D = (#SGU1_features) + 1 (sgu2_return_lag)
        y_all: (N_eff,)
        target_window_ids (optional): raw window_id for each target sample.
    """
    # --- 1) Sort all DataFrames by window_id to ensure chronological order ---
    # WHY: The sliding window and lag extraction assume that row index
    # corresponds to temporal order.  Sorting by window_id guarantees this.
    sgu1 = features_SGU_1.sort_values("window_id").reset_index(drop=True).copy()
    sgu2 = features_SGU_2.sort_values("window_id").reset_index(drop=True).copy()
    lbls = labels_SGU_2_df.sort_values("window_id").reset_index(drop=True).copy()

    # Drop any lag columns that might already exist in SGU1 to avoid
    # duplicate column names after the merge (pandas would suffix them
    # with _x/_y, breaking downstream column selection).
    lag_cols_in_sgu1 = [c for c in sgu1.columns if c.startswith("label_lag")]
    if lag_cols_in_sgu1:
        print(f"Dropping lag columns from SGU1 to avoid duplicates: {lag_cols_in_sgu1}")
        sgu1 = sgu1.drop(columns=lag_cols_in_sgu1)

    # --- 2) Merge everything on window_id ---
    # Inner join ensures we only keep windows present in ALL three sources.
    # Windows that exist in SGU1 but not SGU2 (or vice versa) are dropped.
    df = (
        sgu1.merge(sgu2, on="window_id", how="inner")
            .merge(lbls[["window_id", "label"]], on="window_id", how="inner")
            .sort_values("window_id")
            .reset_index(drop=True)
    )

    # --- 3) Identify SGU1 feature columns and SGU2 lag columns ---
    # SGU1 cols: everything in the SGU1 DataFrame except window_id and any
    # residual lag columns.  These are the cross-sectional microstructure
    # features (e.g., order_imbalance, spread_bps, vwap_ratio, ...).
    sgu1_cols = [
        c for c in sgu1.columns
        if c != "window_id" and not c.startswith("label_lag")
    ]

    # SGU2 lag columns: label_lag1, label_lag2, ..., label_lagK
    # Sort numerically so that lag_cols[0] = label_lag1 (most recent lag)
    # and lag_cols[-1] = label_lagK (oldest lag).
    lag_cols = [c for c in sgu2.columns if c.startswith("label_lag")]
    lag_cols = sorted(lag_cols, key=lambda s: int(re.findall(r"\d+", s)[0]))

    # We only need T_SEQ lag columns (one per time step in the sequence).
    # If L_max > T_SEQ, the extra lags are simply unused.
    K = len(lag_cols)
    if K < T_SEQ:
        raise ValueError(
            f"T_SEQ={T_SEQ} cannot be larger than number of lag columns in SGU2={K}"
        )

    lag_cols_used = lag_cols[:T_SEQ]

    # Sanity check: confirm these lag columns exist in the merged df
    missing = sorted(set(lag_cols_used) - set(df.columns))
    if missing:
        raise RuntimeError(
            f"The following lag columns are missing in merged df: {missing}\n"
            f"df has lag-like columns: {sorted([c for c in df.columns if c.startswith('label_lag')])}"
        )

    D_sgu1 = len(sgu1_cols)
    # Total feature dimension: D_sgu1 cross-sectional features + 1 scalar
    # return channel = D.
    D = D_sgu1 + 1  # +1 for SGU2 scalar time-series

    print(f"SGU1 features used (D_sgu1): {D_sgu1}")
    print(f"SGU2 lags used as time series: {lag_cols_used} (time-ordered reversed)")
    print(f"Total feature dimension D = {D} (SGU1 + 1 SGU2 dimension)")

    # --- 4) Convert to NumPy arrays ---
    N_total = len(df)
    if N_total <= T_SEQ:
        raise RuntimeError(
            f"Not enough windows: N_total={N_total}, but T_SEQ={T_SEQ}."
        )

    sgu1_arr = df[sgu1_cols].to_numpy(dtype=np.float32)         # (N, D_sgu1)
    lags_arr = df[lag_cols_used].to_numpy(dtype=np.float32)     # (N, T_SEQ) in lag1..lagT order
    labels = df["label"].to_numpy(dtype=np.float32)             # (N,)
    window_ids = df["window_id"].to_numpy(dtype=np.int64)       # (N,)

    # --- 5) Build SGU1 histories via sliding_window_view ---
    # Same approach as build_sgu1_sequences: create overlapping windows of
    # T_SEQ consecutive SGU1 feature vectors.
    sgu1_sw = sliding_window_view(sgu1_arr, window_shape=T_SEQ, axis=0)
    sgu1_sw = np.transpose(sgu1_sw, (0, 2, 1))                  # (N_windows, T_SEQ, D_sgu1)

    # Target indices: for target at window i, the SGU1 history comes from
    # feats_sw[i - T_SEQ], which covers windows [i-T_SEQ, ..., i-1].
    target_indices = np.arange(T_SEQ, N_total, dtype=int)
    window_idx = target_indices - T_SEQ

    sgu1_hist_all = sgu1_sw[window_idx]                         # (N_eff_raw, T_SEQ, D_sgu1)

    # --- 6) Build SGU2 scalar sequences (reverse lag order: T_SEQ..1) ---
    # DETAILED EXPLANATION of the lag-to-time-step mapping:
    #
    # For target window i, lags_arr[i, :] contains:
    #   column 0 = label_lag1 = return at window i-1  (most recent)
    #   column 1 = label_lag2 = return at window i-2
    #   ...
    #   column T_SEQ-1 = label_lagT = return at window i-T_SEQ (oldest)
    #
    # But the LSTM expects TIME-ORDERED input: time step 0 = oldest,
    # time step T_SEQ-1 = most recent.  So we REVERSE the lag order:
    #   lags_seq[t=0]        = label_lagT   (window i-T_SEQ, oldest)
    #   lags_seq[t=T_SEQ-1]  = label_lag1   (window i-1, most recent)
    #
    # This reversal ensures that the SGU2 return channel at time step t
    # corresponds to the SAME historical window as the SGU1 features at
    # time step t, maintaining temporal alignment across all D channels.
    lags_for_targets = lags_arr[target_indices, :]              # (N_eff_raw, T_SEQ) in lag1..lagT
    lags_seq_all = lags_for_targets[:, ::-1]                    # (N_eff_raw, T_SEQ) time-ordered

    # --- 7) Combine SGU1 + SGU2 into X_all ---
    # The final tensor has shape (N_eff_raw, T_SEQ, D) where:
    #   X_all[:, :, 0:D_sgu1]  = SGU1 cross-sectional features per time step
    #   X_all[:, :, D_sgu1]    = SGU2 pseudo-mid return at each time step
    # This allows the LSTM to jointly process microstructure features and
    # return momentum at every step of the sequence.
    N_eff_raw = len(target_indices)
    X_all = np.empty((N_eff_raw, T_SEQ, D), dtype=np.float32)
    X_all[:, :, :D_sgu1] = sgu1_hist_all
    X_all[:, :, D_sgu1] = lags_seq_all

    y_all = labels[target_indices]                              # (N_eff_raw,)
    target_window_ids = window_ids[target_indices]              # (N_eff_raw,)

    # --- 8) Mask out sequences with NaN/inf in X or y ---
    # WHY: NaN values arise from:
    #   - Early windows with insufficient lag history
    #   - Windows where the pseudo-mid could not be computed (no trades, no quotes)
    #   - Overnight boundaries in multi-day data
    # We must remove these to prevent NaN propagation during backpropagation.
    mask_y = np.isfinite(y_all)
    mask_X = np.isfinite(X_all).all(axis=(1, 2))
    mask = mask_y & mask_X

    if not np.any(mask):
        raise RuntimeError(
            "No valid sequences were built. "
            "Check NaNs in SGU1/SGU2 features or T_SEQ/window coverage."
        )

    X_all = X_all[mask]
    y_all = y_all[mask]
    target_window_ids = target_window_ids[mask]

    # Feature names for the D channels: SGU1 column names + one synthetic
    # name for the SGU2 return channel.
    feature_names = sgu1_cols + ["sgu2_return_lag"]

    print("Built SGU-2 multi-feature dataset (with SGU2 as 1D time series):")
    print("  X_all shape:", X_all.shape, "(N, T, D)")
    print("  y_all shape:", y_all.shape, "(N,)")
    print("  feature dims:", feature_names)

    if return_window_ids:
        return X_all, y_all, feature_names, target_window_ids
    return X_all, y_all, feature_names


# ============================================================================
# SECTION 5: DATASET AND MODEL DEFINITIONS
# ============================================================================
# This section defines the PyTorch Dataset wrapper and the LSTM model
# architecture.  Both are kept intentionally simple:
#   - The Dataset stores pre-normalized tensors and per-sample weights.
#   - The LSTM uses the hidden state at the LAST time step (many-to-one)
#     to produce a scalar regression output.
# ============================================================================

class SequenceDatasetWeighted(Dataset):
    """
    PyTorch Dataset holding (sequence, normalized label, sample weight) triples.

    WHY a custom Dataset instead of TensorDataset?  Because we need to carry
    a per-sample weight alongside each (X, y) pair.  The weight is used in
    the weighted MSE loss to give more importance to medium/large price moves
    (see bucketization section below).

    Parameters
    ----------
    X_seq : np.ndarray, shape (N, T, F)
        Normalized feature sequences.
    y : np.ndarray, shape (N,)
        Normalized target labels (z-scored pseudo-mid returns).
    w : np.ndarray, shape (N,)
        Positive per-sample weights (from bucketization: small/medium/large).
    T_expected : int
        Expected sequence length; used for a shape assertion.
    F_expected : int
        Expected number of features per time step; used for a shape assertion.
    """
    def __init__(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        T_expected: int,
        F_expected: int,
    ):
        # Defensive shape checks to catch misalignment bugs early.
        assert X_seq.ndim == 3 and X_seq.shape[1:] == (T_expected, F_expected), \
            f"Unexpected X shape: {X_seq.shape}"
        assert len(X_seq) == len(y) == len(w)
        # Convert NumPy arrays to PyTorch tensors once at construction time
        # rather than per-batch to avoid repeated conversion overhead.
        self.X = torch.from_numpy(X_seq.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))
        self.w = torch.from_numpy(w.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i], self.w[i]


class LSTMRegressor(nn.Module):
    """
    LSTM-based regressor for one-step-ahead pseudo-mid return prediction.

    Architecture:
        input:  (B, T, F)  -- batch of feature sequences
        LSTM:   processes the T time steps, producing hidden states (B, T, H)
        Take last hidden: seq[:, -1, :]  -> (B, H)
                          Uses only the FINAL time step's hidden state.
                          WHY? Because this is a many-to-one prediction task:
                          the entire history is summarized in the last hidden
                          state, which is then mapped to the scalar output.
        Dropout:  regularization to prevent overfitting on small datasets.
        Linear:   maps (B, H) -> (B, 1) -> squeeze -> (B,)
        output: (B,)   (one-step-ahead normalized label)

    Parameters
    ----------
    input_dim : int
        Number of features F per time step (= D_sgu1 + 1 for SGU2 mode).
    hidden_units : int
        LSTM hidden size H.  Larger values increase model capacity but
        also increase overfitting risk and training time.
    dropout_p : float
        Dropout probability applied AFTER the LSTM and BEFORE the linear head.
    num_layers : int
        Number of stacked LSTM layers.  Deeper LSTMs can capture more complex
        temporal dependencies but are harder to train.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 64,
        dropout_p: float = 0.4,
        num_layers: int = 1,
    ):
        super().__init__()
        # Inter-layer dropout: when num_layers > 1, PyTorch applies dropout
        # between each pair of stacked LSTM layers (but NOT after the final
        # layer's output).  This regularizes the hidden-to-hidden connections
        # between layers, reducing overfitting in deeper LSTM architectures.
        # When num_layers == 1, there are no inter-layer connections, so
        # dropout is set to 0.0 to avoid a spurious warning.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,  # Input/output tensors have batch as first dim
            dropout=dropout_p if num_layers > 1 else 0.0,  # inter-layer dropout
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -- batch of sequences
        seq, _ = self.lstm(x)          # seq: (B, T, H) -- all hidden states
        # Extract only the LAST time step's hidden state.
        # WHY the last? It encodes the cumulative memory of the entire
        # input sequence, making it the most information-rich single vector
        # for predicting the next window's return.
        last = seq[:, -1, :]           # last (most recent) time step -> (B, H)
        last = self.dropout(last)      # Regularize to reduce overfitting
        yhat = self.out(last)          # (B, 1) -- linear projection to scalar
        return yhat.squeeze(-1)        # (B,) -- remove trailing dimension


# ============================================================================
# SECTION 6: TRAINING UTILITIES
# ============================================================================
# Low-level helper functions used during the training loop:
#   - mse_per_sample:       element-wise squared error (before weighting)
#   - run_epoch_weighted:   one full pass through the DataLoader with
#                           sample-weighted MSE loss
#   - build_param_groups:   separate kernel/bias parameters for different
#                           L2 regularization strengths
#   - predict_numpy:        batch inference returning NumPy predictions
#   - compute_metrics_split: comprehensive evaluation metrics per split
# ============================================================================

def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Element-wise squared error: (pred - target)^2.

    WHY per-sample?  Because we need to multiply each sample's error by its
    bucket weight BEFORE averaging.  This weighted mean gives higher loss
    contribution to medium/large price moves, steering the LSTM to focus
    on "tradable" events rather than noise.
    """
    return (pred - target) ** 2


def run_epoch_weighted(
    loader,  # SequenceDatasetWeighted or DataLoader
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    batch_size: int = 2048,
) -> float:
    """
    Run one full epoch (forward + optional backward) with weighted MSE loss.

    The loss for each batch is:
        loss = mean( (yhat - y)^2 * w )
    where w is the per-sample weight from bucketization.

    WHY weighted MSE?  In financial data, the majority of windows have
    very small price changes ("noise").  Without weighting, the model
    would optimize mostly for predicting near-zero values.  By assigning
    higher weights to medium and large moves, we incentivize the model
    to get those predictions right at the expense of slightly worse
    accuracy on the abundant but less actionable small moves.

    In eval mode, uses torch.no_grad() to skip gradient computation,
    which is faster and uses less memory.

    Returns
    -------
    float
        Average weighted MSE over the entire epoch.
    """
    if train_mode:
        model.train()
        # nullcontext() is a no-op context manager -- gradients are computed
        # normally in training mode.
        grad_ctx = contextlib.nullcontext()
    else:
        model.eval()
        # torch.no_grad() disables gradient tracking for faster inference.
        grad_ctx = torch.no_grad()

    total, count = 0.0, 0

    # WORKAROUND: Accept either a DataLoader or a raw SequenceDatasetWeighted.
    # When a raw dataset is passed (to avoid DataLoader deadlocks caused by
    # multiprocessing resource_tracker from XGBoost), we iterate manually
    # using tensor slicing.
    if isinstance(loader, DataLoader):
        dataset = loader.dataset
        _bs = loader.batch_size or 512
    else:
        dataset = loader
        _bs = batch_size  # uses the function parameter default
    N = len(dataset)

    if hasattr(dataset, "X") and hasattr(dataset, "y") and hasattr(dataset, "w"):
        X_src = dataset.X
        y_src = dataset.y
        w_src = dataset.w

        if train_mode:
            indices = torch.randperm(N, dtype=torch.long)
        else:
            indices = torch.arange(N, dtype=torch.long)

        for start in range(0, N, _bs):
            idx = indices[start : start + _bs]
            xb = X_src.index_select(0, idx).to(device, non_blocking=True)
            yb = y_src.index_select(0, idx).to(device, non_blocking=True)
            wb = w_src.index_select(0, idx).to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad()

            with grad_ctx:
                yhat = model(xb)
                loss_samples = mse_per_sample(yhat, yb)
                loss = (loss_samples * wb).mean()

            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = xb.size(0)
            total += loss.item() * bs
            count += bs
    else:
        # Generic fallback for non-standard datasets.
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad()

            with grad_ctx:
                yhat = model(xb)
                loss_samples = mse_per_sample(yhat, yb)
                loss = (loss_samples * wb).mean()

            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = xb.size(0)
            total += loss.item() * bs
            count += bs

    return total / max(1, count)


def build_param_groups(model: nn.Module, l2_kernel: float, l2_bias: float) -> List[Dict[str, Any]]:
    """
    Separate model parameters into kernel (weight) and bias groups, each
    with its own L2 regularization strength (weight_decay).

    WHY different L2 for kernels vs biases?  Regularizing biases is
    generally less important (and can even hurt) because biases only shift
    activations, whereas kernels control the model's capacity to fit
    complex patterns.  Allowing separate L2 values gives the HP search
    an extra degree of freedom to find the right regularization balance.
    """
    kernel_params, bias_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or name.endswith(".bias"):
            bias_params.append(p)
        else:
            kernel_params.append(p)
    groups = []
    if kernel_params:
        groups.append(dict(params=kernel_params, weight_decay=l2_kernel))
    if bias_params:
        groups.append(dict(params=bias_params, weight_decay=l2_bias))
    return groups


def predict_numpy(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Run batch inference on a NumPy array and return predictions as NumPy.

    WHY a separate function?  During evaluation and metrics computation,
    we need to predict on full splits (train/val/test) which may not fit
    in GPU memory in a single forward pass.  This function handles the
    batching transparently.
    """
    model.eval()
    preds = []
    N_local = len(X)

    with torch.no_grad():
        for start in range(0, N_local, batch_size):
            end = start + batch_size
            x_batch = X[start:end]
            x_tensor = torch.from_numpy(
                x_batch.astype(np.float32, copy=False)
            ).to(device, non_blocking=True)
            out = model(x_tensor)
            preds.append(out.cpu().numpy())

    return np.concatenate(preds, axis=0)


def compute_metrics_split(
    name: str,
    y_true_real: np.ndarray,
    y_pred_real: np.ndarray,
    bucket_split: np.ndarray,
    q1_abs: float,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for a given data split.

    Metrics include:
    - RMSE, R-squared, relative RMSE (RMSE / std(y)), Pearson correlation
    - Directional Accuracy (DA): fraction of samples where sign(pred) == sign(true)
    - Per-bucket DA: separate DA for small, medium, and large moves
    - DA_tradable: DA restricted to medium + large moves (the ones a market
      maker would actually trade on)
    - DA_eps: directional accuracy with a neutral zone (moves within +/- q1
      are classified as zero)

    WHY per-bucket metrics?  A model could achieve high overall DA simply by
    predicting "approximately zero" for every window (since most windows have
    small moves).  Per-bucket metrics reveal whether the model actually
    captures the direction of the rarer but more impactful large moves.

    Parameters
    ----------
    name : str
        Label for the split (e.g., "TRAIN 64%", "TEST 20%"); used in print.
    y_true_real : np.ndarray
        Ground-truth labels in REAL (un-normalized) scale.
    y_pred_real : np.ndarray
        Model predictions in REAL scale (after inverse_transform).
    bucket_split : np.ndarray
        Per-sample bucket labels: 0=small, 1=medium, 2=large.
    q1_abs : float
        The q1 quantile threshold used for bucketization; also used as the
        epsilon for the neutral-zone DA metric.
    """
    # ---- Standard regression metrics ----
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    r2   = r2_score(y_true_real, y_pred_real)
    std_y = np.std(y_true_real)
    # Relative RMSE normalizes by the target's standard deviation, making
    # the metric comparable across assets with different volatility levels.
    rel_rmse = rmse / std_y if std_y > 0 else np.nan
    if (len(y_true_real) > 1) and (np.std(y_true_real) > 0) and (np.std(y_pred_real) > 0):
        corr = np.corrcoef(y_true_real, y_pred_real)[0, 1]
    else:
        corr = np.nan

    # ---- Raw directional accuracy (sign match) ----
    sign_true = np.sign(y_true_real)
    sign_pred = np.sign(y_pred_real)
    da_raw = np.mean(sign_true == sign_pred)

    # ---- Per-bucket masks ----
    is_small = bucket_split == 0
    is_med   = bucket_split == 1
    is_large = bucket_split == 2
    is_ml    = ~is_small  # medium + large = "tradable" moves

    def da_mask(mask):
        """Directional accuracy restricted to samples matching the mask."""
        if not np.any(mask):
            return np.nan
        return np.mean(sign_true[mask] == sign_pred[mask])

    # For SMALL bucket: instead of sign match, we check "neutral hit" --
    # does the model correctly predict a near-zero magnitude?
    # WHY: for small moves, getting the sign right is meaningless (it is
    # essentially noise).  What matters is that the model recognizes these
    # as "small / no trade" events and predicts a magnitude within the
    # neutral zone.
    eps_small = q1_abs

    def small_neutral_hit(y_true, y_pred, mask, eps):
        """Fraction of small-bucket predictions with |pred| <= eps."""
        if not np.any(mask):
            return np.nan
        yp = y_pred[mask]
        return np.mean(np.abs(yp) <= eps)

    da_small  = small_neutral_hit(y_true_real, y_pred_real, is_small, eps_small)
    da_medium = da_mask(is_med)
    da_large  = da_mask(is_large)
    da_tradable = da_mask(is_ml)

    # ---- Epsilon-thresholded directional accuracy ----
    # Classifies both true and predicted values into {-1, 0, +1} using
    # the q1 threshold as a neutral zone.  This avoids penalizing the model
    # for getting the "sign" wrong on tiny moves that are within noise.
    eps = q1_abs

    def sign_eps(x):
        """Ternary sign function with neutral zone [-eps, +eps]."""
        out = np.zeros_like(x)
        out[x > eps]  = 1
        out[x < -eps] = -1
        return out

    sign_true_eps = sign_eps(y_true_real)
    sign_pred_eps = sign_eps(y_pred_real)
    da_eps = np.mean(sign_true_eps == sign_pred_eps)

    print(f"[{name}] RMSE={rmse:.6f}  R²={r2:.4f}  Rel.RMSE={rel_rmse:.6f}  Corr={corr:.4f}")
    print(f"          DA_raw={da_raw:.4f}  DA_small(neutral)={da_small:.4f}  "
          f"DA_med={da_medium:.4f}  DA_large={da_large:.4f}")
    print(f"          DA_tradable(M+L)={da_tradable:.4f}  DA_eps(neutral |y|<=q1)={da_eps:.4f}")

    return dict(
        rmse=float(rmse),
        r2=float(r2),
        rel_rmse=float(rel_rmse) if np.isfinite(rel_rmse) else float("nan"),
        corr=float(corr) if np.isfinite(corr) else float("nan"),
        directional_accuracy=float(da_raw),
        da_small=float(da_small) if np.isfinite(da_small) else float("nan"),
        da_medium=float(da_medium) if np.isfinite(da_medium) else float("nan"),
        da_large=float(da_large) if np.isfinite(da_large) else float("nan"),
        da_tradable=float(da_tradable) if np.isfinite(da_tradable) else float("nan"),
        da_eps=float(da_eps),
    )


# ============================================================================
# SECTION 7: LATIN HYPERCUBE SAMPLING (LHS) SEARCH HELPERS
# ============================================================================
# WHY LHS instead of random search or grid search?
#   - Grid search is exhaustive but combinatorially explosive (3^8 > 6000
#     combinations for 8 hyperparameters with 3 choices each).
#   - Pure random search can "clump" in some regions and leave others
#     unexplored.
#   - LHS provides SPACE-FILLING coverage: it divides each hyperparameter's
#     range into n_samples equal-probability strata and ensures that each
#     stratum is sampled exactly once.  This gives better coverage of the
#     hyperparameter space with fewer trials than random search.
#
# The implementation below generates a Latin Hypercube in [0,1]^dim and
# then maps each [0,1] coordinate to a discrete grid option.
# ============================================================================

def _lhs(n_samples: int, dim: int, rng_: np.random.Generator) -> np.ndarray:
    """
    Generate a Latin Hypercube Sample matrix of shape (n_samples, dim).

    Each column is a random permutation of n_samples stratified samples
    in [0, 1].  The stratification ensures that when projected onto any
    single axis, the samples are uniformly spread across all n_samples
    strata (each stratum of width 1/n_samples is hit exactly once).

    Parameters
    ----------
    n_samples : int
        Number of hyperparameter combinations to generate.
    dim : int
        Number of hyperparameters (dimensions in the search space).
    rng_ : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    H : np.ndarray, shape (n_samples, dim)
        Matrix of uniform [0, 1] samples with Latin Hypercube structure.
    """
    # Create stratified samples: each sample falls in a unique stratum
    # of width 1/n_samples, with a random offset within its stratum.
    cut = (np.arange(n_samples) + rng_.random(n_samples)) / n_samples
    H = np.zeros((n_samples, dim), dtype=np.float64)
    for j in range(dim):
        # Independent random permutation per dimension ensures that the
        # joint distribution covers the space without alignment artifacts.
        P = rng_.permutation(n_samples)
        H[:, j] = cut[P]
    return H


def _choice_from_grid(options: List[Any], u_scalar: float) -> Any:
    """
    Map a uniform [0, 1) scalar to one of the discrete grid options.

    The [0, 1) interval is divided into len(options) equal bins:
      bin 0 = [0, 1/n), bin 1 = [1/n, 2/n), ..., bin n-1 = [(n-1)/n, 1)
    The scalar u is mapped to the bin it falls in.
    """
    n = len(options)
    idx = int(np.floor(u_scalar * n))
    if idx == n:
        idx = n - 1
    return options[idx]


def sample_params_lhs(
    n_samples: int,
    grid: Dict[str, List[Any]],
    rng_: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Generate n_samples hyperparameter combinations using Latin Hypercube
    Sampling over a discrete grid.

    Each hyperparameter key in ``grid`` maps to a list of candidate values.
    The LHS matrix provides a [0,1] coordinate per (sample, hyperparameter)
    pair, which is then mapped to a grid option via ``_choice_from_grid``.

    Returns
    -------
    samples : list of dict
        Each dict maps hyperparameter name -> selected value.
    """
    keys = list(grid.keys())
    U = _lhs(n_samples, len(keys), rng_)
    samples: List[Dict[str, Any]] = []
    for i in range(n_samples):
        u = U[i]
        params: Dict[str, Any] = {}
        for j, key in enumerate(keys):
            params[key] = _choice_from_grid(grid[key], float(u[j]))
        samples.append(params)
    return samples


# ============================================================================
# SECTION 8: HIGH-LEVEL PIPELINE (main function + save/load)
# ============================================================================
# This section contains the main training entrypoint and all supporting
# functions for the full SGU2 LSTM pipeline:
#   - DEFAULT_GRID:              default hyperparameter search space
#   - bucketize_array:           label bucketization (small/medium/large)
#   - _chronological_split:      train/val/test split
#   - _prepare_scalers_and_normalize: feature + target normalization
#   - _fit_one:                  train a single LSTM configuration
#   - train_sgu2_lstm_pipeline:  main entrypoint orchestrating everything
#   - _save_training_artifacts:  persist models and scalers to disk
#   - load_trained_sgu2_artifacts: reload a previously saved model package
# ============================================================================

# Default hyperparameter search grid.
# Each key maps to a list of candidate values that LHS will sample from.
# The grid is intentionally modest in size to keep the total number of
# trials manageable (n_samples_search=50 by default).
DEFAULT_GRID: Dict[str, List[Any]] = {
    "kernel_l2":    [1e-3, 5e-3, 1e-2],       # L2 weight decay for kernel (weight) params
    "bias_l2":      [1e-3, 5e-3, 1e-2],       # L2 weight decay for bias params
    "dropout":      [0.3, 0.5, 0.7],          # Dropout rate after LSTM last hidden state
    "epochs":       [10, 20],                   # Max training epochs per trial
    "batch_size":   [512, 1024, 2048],         # Mini-batch size for DataLoader
    "hidden_units": [16, 32],              # LSTM hidden dimension H
    "num_layers":   [1, 2],                 # Number of stacked LSTM layers
    "lr":           [1e-4, 3e-4, 5e-4, 1e-3],  # Adam learning rate
}



def bucketize_array(y: np.ndarray, q1_val: float, q2_val: float) -> np.ndarray:
    """
    Assign each label to a bucket based on its absolute magnitude.

    Bucketization schema:
        0 = small  (|y| <= q1)  -- noise / non-tradable moves
        1 = medium (q1 < |y| <= q2) -- moderate moves
        2 = large  (|y| > q2)  -- significant price movements

    WHY bucketize?  Financial return distributions are heavy-tailed: the
    vast majority of windows have tiny (near-zero) returns, while a small
    fraction exhibit large moves that are most relevant for trading.
    Bucketization enables:
      1. Per-bucket sample weighting in the loss (upweight large moves).
      2. Per-bucket evaluation metrics (did the model predict the *sign*
         of large moves correctly?).
      3. Optional exclusion of small moves from training to focus the
         model on the signal rather than the noise.

    Parameters
    ----------
    y : np.ndarray
        Array of labels (pseudo-mid returns).
    q1_val : float
        Absolute threshold separating small from medium (25th percentile
        of non-zero |y| in the training set, by default).
    q2_val : float
        Absolute threshold separating medium from large (75th percentile
        of non-zero |y| in the training set, by default).
    """
    a = np.abs(y)
    buckets = np.zeros_like(a, dtype=np.int8)
    buckets[a > q1_val] = 1   # medium
    buckets[a > q2_val] = 2   # large (overwrites the 1 set above for |y| > q2)
    return buckets


def _chronological_split(
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    frac_train: float = 0.64,
    frac_val: float = 0.16,
) -> Dict[str, Any]:
    """
    Split the dataset chronologically into train / val / test.

    Default fractions: 64% train, 16% validation, 20% test.

    WHY chronological (not random)?  Financial time series exhibit
    autocorrelation, regime changes, and non-stationarity.  A random
    split would leak future information into the training set (e.g.,
    the model could see a pattern from 3pm and use it to predict 1pm).
    A chronological split ensures that the model is always evaluated on
    data that comes AFTER its training data, mimicking real deployment.

    WHY 64/16/20?  This is a common split that provides enough training
    data (64%), a meaningful validation set for early stopping and HP
    selection (16%), and a held-out test set (20%) that is never used
    during model selection.
    """
    if not (0.0 < frac_train < 1.0):
        raise ValueError(f"frac_train must be in (0,1), got {frac_train}.")
    if not (0.0 <= frac_val < 1.0):
        raise ValueError(f"frac_val must be in [0,1), got {frac_val}.")
    if frac_train + frac_val >= 1.0:
        raise ValueError(
            f"frac_train + frac_val must be < 1.0, got {frac_train + frac_val}."
        )

    N = len(X_all)
    if N == 0:
        raise ValueError("Cannot split empty dataset (N=0).")

    i_train = int(np.floor(frac_train * N))
    i_val   = int(np.floor((frac_train + frac_val) * N))

    X_train_full, y_train_full = X_all[:i_train],      y_all[:i_train]
    X_val_full,   y_val_full   = X_all[i_train:i_val], y_all[i_train:i_val]
    X_test_full,  y_test_full  = X_all[i_val:],        y_all[i_val:]

    return dict(
        N=N,
        i_train=i_train,
        i_val=i_val,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_val_full=X_val_full,
        y_val_full=y_val_full,
        X_test_full=X_test_full,
        y_test_full=y_test_full,
    )


def _prepare_scalers_and_normalize(
    *,
    X_train_ml: np.ndarray,
    X_val_ml: np.ndarray,
    X_train_full: np.ndarray,
    X_val_full: np.ndarray,
    X_test_full: np.ndarray,
    y_train_ml: np.ndarray,
    y_val_ml: np.ndarray,
    y_train_full: np.ndarray,
    y_val_full: np.ndarray,
    y_test_full: np.ndarray,
) -> Dict[str, Any]:
    """
    Feature + target normalization (z-scoring) for all data splits.

    Normalization flow -- FEATURE NORMALIZATION (scaler_X):
    -------------------------------------------------------
    1. The 3D tensors (N, T, F) are FLATTENED to 2D (N*T, F) so that
       StandardScaler computes one mean and one std per feature column
       across ALL time steps and ALL samples.
       WHY flatten?  StandardScaler expects a 2D matrix (samples x features).
       By treating each (sample, time_step) pair as an independent row,
       we get a single mean/std per feature that is applied uniformly
       across time steps.  This is appropriate because the same feature
       (e.g., order imbalance) should have the same scale regardless of
       whether it appears at t=0 or t=T_SEQ-1 in the sequence.

    2. The scaler is FIT on X_train_full (the FULL training set, including
       ALL buckets) to capture the true feature distribution.  This avoids
       biasing the scaler statistics toward medium/large-move features when
       include_small_in_train=False.
       All other splits are TRANSFORMED using the same fitted scaler to
       prevent data leakage from val/test into the scaler statistics.

    3. After transformation, the 2D arrays are RESHAPED back to 3D (N, T, F).

    Normalization flow -- TARGET NORMALIZATION (scaler_y):
    -----------------------------------------------------
    1. Target labels y are z-scored: y_z = (y - mean) / std.
    2. The scaler is FIT on y_train_full (the FULL training set) to ensure
       the target mean/std reflect the complete return distribution.
       WHY normalize targets?  The LSTM's linear output layer produces
       unbounded values.  If the targets are in raw return scale (e.g.,
       1e-5 to 1e-3), the loss landscape becomes ill-conditioned.
       Z-scoring brings targets to mean ~= 0, std ~= 1, which makes the
       loss gradients well-behaved and the learning rate more interpretable.
    3. At inference time, predictions are de-normalized using
       scaler_y.inverse_transform() to recover the real return scale.
    """
    T = X_train_full.shape[1]
    F = X_train_full.shape[2]

    # Save original sample counts for reshaping back from 2D to 3D.
    n_train_ml = len(X_train_ml)
    n_val_ml = len(X_val_ml)
    n_train_full = len(X_train_full)
    n_val_full = len(X_val_full)
    n_test_full = len(X_test_full)

    # --- Step 1: Flatten 3D -> 2D for StandardScaler ---
    # Each row in the 2D matrix is one (sample, time_step) pair.
    X_train_ml_2d   = X_train_ml.reshape(-1, F)
    X_val_ml_2d     = X_val_ml.reshape(-1, F)
    X_train_full_2d = X_train_full.reshape(-1, F)
    X_val_full_2d   = X_val_full.reshape(-1, F)
    X_test_full_2d  = X_test_full.reshape(-1, F)

    # --- Step 2: Fit on FULL training data, transform all splits ---
    # --- CRITICAL: Fit the scaler on the FULL training set, NOT on the
    #     ML-filtered subset.  The scaler's statistics (mean, std) define
    #     the normalization "ruler" for ALL data — train, val, test.
    #     If we fit on the ML-filtered subset (which excludes small-bucket
    #     samples when include_small_in_train=False), the statistics would
    #     be biased toward medium/large-move feature distributions,
    #     creating a distributional mismatch at inference time.
    scaler_X = StandardScaler()
    scaler_X.fit(X_train_full_2d)  # Fit on FULL training set
    X_train_ml_2d   = scaler_X.transform(X_train_ml_2d)  # Transform ML subset
    X_val_ml_2d     = scaler_X.transform(X_val_ml_2d)
    X_train_full_2d = scaler_X.transform(X_train_full_2d)
    X_val_full_2d   = scaler_X.transform(X_val_full_2d)
    X_test_full_2d  = scaler_X.transform(X_test_full_2d)

    # --- Step 3: Reshape back to 3D (N, T, F) for LSTM consumption ---
    X_train_ml   = X_train_ml_2d.reshape(n_train_ml, T, F)
    X_val_ml     = X_val_ml_2d.reshape(n_val_ml,   T, F)
    X_train_full = X_train_full_2d.reshape(n_train_full, T, F)
    X_val_full   = X_val_full_2d.reshape(n_val_full,   T, F)
    X_test_full  = X_test_full_2d.reshape(n_test_full, T, F)

    # --- Step 4: Target normalization ---
    # Fit on the FULL training set targets, then transform all splits.
    # reshape(-1, 1) is needed because StandardScaler expects 2D input;
    # .ravel() flattens back to 1D after transformation.
    # CRITICAL: We fit on y_train_full (the COMPLETE training set) rather
    # than y_train_ml (the ML-filtered subset) for the same distributional
    # consistency reason as scaler_X above.  The target scaler's mean and
    # std should reflect the full return distribution, not just the
    # medium/large subset.
    scaler_y = StandardScaler()
    scaler_y.fit(y_train_full.reshape(-1, 1))  # Fit on FULL training set
    y_train_ml_z = scaler_y.transform(y_train_ml.reshape(-1, 1)).ravel()
    y_val_ml_z   = scaler_y.transform(y_val_ml.reshape(-1, 1)).ravel()
    y_train_full_z = scaler_y.transform(y_train_full.reshape(-1, 1)).ravel()
    y_val_full_z   = scaler_y.transform(y_val_full.reshape(-1, 1)).ravel()
    y_test_full_z  = scaler_y.transform(y_test_full.reshape(-1, 1)).ravel()

    return dict(
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        X_train_ml=X_train_ml,
        X_val_ml=X_val_ml,
        X_train_full=X_train_full,
        X_val_full=X_val_full,
        X_test_full=X_test_full,
        y_train_ml_z=y_train_ml_z,
        y_val_ml_z=y_val_ml_z,
        y_train_full_z=y_train_full_z,
        y_val_full_z=y_val_full_z,
        y_test_full_z=y_test_full_z,
    )


def _fit_one(
    *,
    train_dataset: SequenceDatasetWeighted,
    val_dataset: SequenceDatasetWeighted,
    device: torch.device,
    F: int,
    dropout_p: float,
    l2_kernel: float,
    l2_bias: float,
    lr: float,
    epochs: int,
    batch_size: int,
    hidden_units: int = 64,
    patience: int = 8,
    num_layers: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[nn.Module, float, float]:
    """
    Train ONE LSTM configuration (one hyperparameter combination) with
    early stopping, and return the trained model + validation metrics.

    This function is called once per LHS trial during the hyperparameter
    search.  It:
      1. Creates a fresh LSTMRegressor with the given hyperparameters.
      2. Trains for up to `epochs` epochs on the train subset.
      3. After each epoch, evaluates on the val subset.
      4. Tracks the best validation loss; if it does not improve for
         `patience` consecutive epochs, stops early.
      5. Restores the best-epoch weights before returning.

    WHY early stopping?  LSTMs are prone to overfitting, especially on
    small financial datasets.  Early stopping acts as implicit regularization
    by selecting the model checkpoint with the best generalization (val loss).

    Returns
    -------
    model : nn.Module
        The trained model with best-epoch weights loaded.
    best_val : float
        The best validation loss (weighted MSE, normalized scale).
    val_mae : float
        Validation MAE (normalized scale) at the best epoch.
    """
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after filtering/masking.")
    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty after filtering/masking. "
            "Adjust split fractions or set include_small_in_train=True."
        )

    # Instead of DataLoader (which can deadlock when a multiprocessing
    # resource_tracker is active from XGBoost), we pass the raw datasets
    # to run_epoch_weighted and iterate manually with tensor slicing.
    train_loader = train_dataset
    val_loader = val_dataset

    # Instantiate a fresh LSTM with the given hyperparameters and move to device.
    model = LSTMRegressor(
        input_dim=F,
        hidden_units=hidden_units,
        dropout_p=dropout_p,
        num_layers=num_layers,
    ).to(device)

    # Adam optimizer with separate L2 regularization for kernels and biases.
    optimizer = torch.optim.Adam(
        build_param_groups(model, l2_kernel, l2_bias),
        lr=lr,
    )

    # Early stopping state: track best validation loss and the corresponding
    # model weights (saved on CPU to avoid occupying GPU memory).
    best_val, best_state, no_improve = np.inf, None, 0

    # --- CPU stability: disable MKLDNN and limit threads ONCE ---
    if device.type == "cpu" and not getattr(_fit_one, "_cpu_configured", False):
        import os
        torch.backends.mkldnn.enabled = False
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _fit_one._cpu_configured = True
        print(f"    [_fit_one] CPU safe: mkldnn=False, threads={torch.get_num_threads()}", flush=True)

    for epoch_i in range(epochs):
        # Training pass: forward + backward + parameter update
        train_loss = run_epoch_weighted(train_loader, model, optimizer, device, True)
        if not np.isfinite(train_loss):
            warnings.warn(
                "Non-finite training loss encountered in SGU2 trial; "
                "stopping this trial early.",
                RuntimeWarning,
            )
            break
        # Validation pass: forward only (no gradients)
        val_loss = run_epoch_weighted(val_loader, model, optimizer, device, False)
        if not np.isfinite(val_loss):
            warnings.warn(
                "Non-finite validation loss encountered in SGU2 trial; "
                "stopping this trial early.",
                RuntimeWarning,
            )
            break
        if val_loss < best_val:
            # New best -- checkpoint the model weights (cloned to CPU).
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                # No improvement for `patience` epochs -- stop training
                # to prevent overfitting.
                break

    # Restore the best-epoch weights (the model may have degraded in
    # later epochs before early stopping kicked in).
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        # If no finite validation checkpoint was ever found, return sentinel
        # values so the outer LHS loop can safely skip this trial.
        return model, float("inf"), float("nan")

    # Compute validation MAE (in normalized/z-scored scale) as an additional
    # metric for the hyperparameter search progress bar.
    model.eval()
    y_true_val, y_pred_val = [], []
    # val_loader may be a raw SequenceDatasetWeighted (not a DataLoader),
    # so access .X/.y directly instead of .dataset/.batch_size.
    if isinstance(val_loader, DataLoader):
        val_dataset_local = val_loader.dataset
        val_bs = val_loader.batch_size or 512
    else:
        val_dataset_local = val_loader
        val_bs = batch_size
    with torch.no_grad():
        if hasattr(val_dataset_local, "X") and hasattr(val_dataset_local, "y"):
            Xv = val_dataset_local.X
            yv = val_dataset_local.y
            Nv = len(val_dataset_local)
            for start in range(0, Nv, val_bs):
                end = start + val_bs
                xb = Xv[start:end].to(device, non_blocking=True)
                yb = yv[start:end].to(device, non_blocking=True)
                yh = model(xb)
                y_true_val.append(yb.cpu().numpy())
                y_pred_val.append(yh.cpu().numpy())
        else:
            for xb, yb, wb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                yh = model(xb)
                y_true_val.append(yb.cpu().numpy())
                y_pred_val.append(yh.cpu().numpy())
    y_true_val = np.concatenate(y_true_val)
    y_pred_val = np.concatenate(y_pred_val)
    if np.isfinite(y_true_val).all() and np.isfinite(y_pred_val).all():
        val_mae = float(np.mean(np.abs(y_true_val - y_pred_val)))
    else:
        val_mae = float("nan")

    return model, float(best_val), val_mae


def train_sgu2_lstm_pipeline(
    *,
    features_SGU_1: pd.DataFrame,
    features_SGU_2: Optional[pd.DataFrame] = None,
    labels_SGU_2_df: Optional[pd.DataFrame] = None,
    labels_SGU_1_df: Optional[pd.DataFrame] = None,
    T_SEQ: int = 10,
    use_sgu1_only: bool = False,
    # train/val/test split
    frac_train: float = 0.64,
    frac_val: float = 0.16,
    # bucketization/weights
    include_small_in_train: bool = True,
    W_SMALL: float = 0.6,
    W_MEDIUM: float = 1.0,
    W_LARGE: float = 2.0,
    # random seeds
    seed: int = 1337,
    fast_seed_mode: bool = True,
    lhs_seed: int = 42,
    # search
    grid: Optional[Dict[str, List[Any]]] = None,
    n_samples_search: int = 50,
    patience: int = 8,
    min_refit_epochs: int = 5,
    # runtime
    device: Optional[Union[str, torch.device]] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    cpu_safe_mode: bool = True,
    cpu_num_threads: Optional[int] = None,
    # optional caching/saving
    cache_path: Optional[PathLike] = None,
    save_dir: Optional[PathLike] = None,
    overwrite: bool = False,
    # whether to return large arrays
    return_data: bool = False,
) -> Dict[str, Any]:
    """
    Main entrypoint for the SGU-2 LSTM training pipeline.

    This function orchestrates the entire training workflow:
      1. Seed + device setup
      2. Build or load the 3D feature tensor (X_all) and target vector (y_all)
      3. Chronological train/val/test split
      4. Bucketize labels into small/medium/large categories
      5. Apply sample weights per bucket
      6. Normalize features and targets using StandardScaler
      7. Run LHS hyperparameter search (each trial trains an LSTM with
         early stopping)
      8. Restore the best model from the search (best_model_64, trained on
         64% of data)
      9. Warm-start refit on train+val (80%) with the best hyperparameters
     10. Compute metrics on all splits for both models
     11. Optionally save all artifacts to disk
     12. Return everything in a structured dict

    The function supports two modes:
      - use_sgu1_only=True:  LSTM uses only SGU1 features (ablation)
      - use_sgu1_only=False: LSTM uses SGU1 + SGU2 lagged returns (full)

    IMPORTANT: The internal computations match the original script exactly.

    Returns
    -------
    dict with keys:
        best_model_64, final_model, scaler_X, scaler_y, artifacts,
        feature_names, bucket_thresholds, best, metrics, device,
        predictions, and optionally 'data' (if return_data=True).
    """
    # -------------------------------------------------------------------------
    # STEP 1: Seed + device setup
    # -------------------------------------------------------------------------
    # WHY seed everything first?  All subsequent random operations (weight
    # initialization, LHS sampling, data loading order) will be deterministic
    # given the same seed, enabling reproducible experiments.
    set_seed(seed, fast_mode=fast_seed_mode)
    device = resolve_device(device)
    print("Using device:", device)

    if device.type == "cpu":
        cpu_cfg = _configure_cpu_runtime(
            enable_safe_mode=cpu_safe_mode,
            cpu_num_threads=cpu_num_threads,
        )
        if cpu_cfg["safe_mode"]:
            print(
                "[SGU2] CPU safe mode:",
                f"mkldnn={cpu_cfg['mkldnn_enabled_before']}→{cpu_cfg['mkldnn_enabled_after']},",
                f"threads={cpu_cfg['num_threads_before']}→{cpu_cfg['num_threads_after']},",
                f"interop={cpu_cfg['interop_threads_before']}→{cpu_cfg['interop_threads_after']}",
            )

    # Quick MPS smoke test: if MPS is selected but fails a basic LSTM
    # forward+backward, fall back to CPU to avoid silent hangs.
    if device.type == "mps":
        try:
            _test_lstm = nn.LSTM(19, 16, 1, batch_first=True).to(device)
            _test_x = torch.randn(4, 10, 19, device=device)
            _test_out, _ = _test_lstm(_test_x)
            _test_loss = _test_out.mean()
            _test_loss.backward()
            _ = _test_loss.item()  # force sync
            del _test_lstm, _test_x, _test_out, _test_loss
            torch.mps.empty_cache()
            print("[SGU2] MPS smoke test passed.", flush=True)
        except Exception as e:
            print(f"[SGU2] MPS smoke test FAILED ({e}), falling back to CPU.", flush=True)
            device = torch.device("cpu")

    # DataLoader configuration: use 2 workers on GPU (to overlap data
    # loading with GPU computation) and 0 on CPU (to avoid overhead).
    if num_workers is None:
        num_workers = 2 if torch.cuda.is_available() else 0
    # pin_memory=True speeds up CPU->GPU transfers when using CUDA.
    if pin_memory is None:
        pin_memory = (device.type == "cuda")

    # -------------------------------------------------------------------------
    # STEP 2: Build X_all, y_all (optionally from cache)
    # -------------------------------------------------------------------------
    # WHY cache?  Building the 3D tensor involves DataFrame merges, sliding
    # windows, and NaN filtering which can take minutes on large datasets.
    # If a cache file exists at cache_path, we skip all that and load the
    # pre-computed arrays directly.
    if cache_path is not None and Path(cache_path).expanduser().resolve().exists():
        cache = load_npz_cache(cache_path)
        X_all = cache["X_all"]
        y_all = cache["y_all"]
        all_feature_cols = cache["feature_cols"]
        print(f"Loaded cached arrays from: {Path(cache_path).expanduser().resolve()}")
    else:
        if use_sgu1_only:
            if labels_SGU_1_df is None:
                raise ValueError("labels_SGU_1_df must be provided when use_sgu1_only=True")
            X_all, y_all, all_feature_cols = build_sgu1_sequences(
                features_SGU_1=features_SGU_1,
                labels_SGU_1_df=labels_SGU_1_df,
                T_SEQ=T_SEQ,
            )
        else:
            if features_SGU_2 is None or labels_SGU_2_df is None:
                raise ValueError("features_SGU_2 and labels_SGU_2_df must be provided when use_sgu1_only=False")
            X_all, y_all, all_feature_cols = build_sgu2_multi_feature_sequences(
                features_SGU_1=features_SGU_1,
                features_SGU_2=features_SGU_2,
                labels_SGU_2_df=labels_SGU_2_df,
                T_SEQ=T_SEQ,
            )

        if cache_path is not None:
            save_npz_cache(
                cache_path,
                X_all=X_all,
                y_all=y_all,
                feature_cols=all_feature_cols,
                meta=dict(T_SEQ=T_SEQ, use_sgu1_only=use_sgu1_only, seed=seed),
            )
            print(f"Saved cache arrays to: {Path(cache_path).expanduser().resolve()}")

    T = X_all.shape[1]  # sequence length
    F = X_all.shape[2]  # number of features per time step

    # -------------------------------------------------------------------------
    # 3) Chronological split (same default 64/16/20)
    # -------------------------------------------------------------------------
    split = _chronological_split(X_all, y_all, frac_train=frac_train, frac_val=frac_val)

    X_train_full = split["X_train_full"]
    y_train_full = split["y_train_full"]
    X_val_full   = split["X_val_full"]
    y_val_full   = split["y_val_full"]
    X_test_full  = split["X_test_full"]
    y_test_full  = split["y_test_full"]

    n_train_full, n_val_full, n_test_full = len(X_train_full), len(X_val_full), len(X_test_full)
    N = split["N"]
    if min(n_train_full, n_val_full, n_test_full) <= 0:
        raise ValueError(
            "Chronological split produced an empty partition. "
            f"train={n_train_full}, val={n_val_full}, test={n_test_full}. "
            "Adjust frac_train/frac_val or provide more data."
        )
    print(
        f"Total samples: {N:,} | "
        f"train={n_train_full:,} ({n_train_full / N:.1%}), "
        f"val={n_val_full:,} ({n_val_full / N:.1%}), "
        f"test={n_test_full:,} ({n_test_full / N:.1%})"
    )

    # Save ORIGINAL targets (real, un-normalized scale) before any
    # transformations.  These are needed later to compute metrics in the
    # original return scale (the model predicts in z-scored space, so we
    # must inverse-transform predictions and compare against these originals).
    y_train_orig = y_train_full.copy()
    y_val_orig   = y_val_full.copy()
    y_test_orig  = y_test_full.copy()

    # -------------------------------------------------------------------------
    # STEP 4: Bucketize labels by |y| on TRAIN split ONLY
    # -------------------------------------------------------------------------
    # WHY on train only?  The bucket thresholds (q1, q2) are statistics of
    # the data.  Computing them on the full dataset would leak information
    # from val/test into the training procedure.
    #
    # WHY non-zero returns?  Many windows have exactly zero return (no price
    # change).  Including them in the quantile computation would push q1
    # toward zero, making the "small" bucket trivially empty.  By computing
    # quantiles on non-zero |y| only, we get meaningful thresholds that
    # separate genuine small moves from medium and large ones.
    abs_y_train = np.abs(y_train_full)

    nonzero_mask = abs_y_train > 0
    if not np.any(nonzero_mask):
        raise RuntimeError("All train labels are zero; cannot define bucket thresholds.")

    abs_y_nz = abs_y_train[nonzero_mask]
    # 25th and 75th percentiles of non-zero absolute returns.
    # q1 separates "small" from "medium"; q2 separates "medium" from "large".
    q1, q2 = np.quantile(abs_y_nz, [0.25, 0.75])
    if q1 <= 0:
        # Safety fallback: if the 25th percentile is somehow zero (very
        # concentrated distribution), use the 5th percentile instead to
        # ensure a positive threshold for the neutral-zone metric.
        q1 = np.quantile(abs_y_nz, 0.05)

    print("Bucket thresholds (|y|, non-zero): q1 =", q1, ", q2 =", q2)

    bucket_train_full = bucketize_array(y_train_full, q1, q2)
    bucket_val_full   = bucketize_array(y_val_full,   q1, q2)
    bucket_test_full  = bucketize_array(y_test_full,  q1, q2)

    # Optional: sanity check of bucket distribution
    for name, b in [("train", bucket_train_full), ("val", bucket_val_full), ("test", bucket_test_full)]:
        uniq, cnt = np.unique(b, return_counts=True)
        print(name, dict(zip(uniq, cnt)))

    # -------------------------------------------------------------------------
    # STEP 5: Select training samples and assign per-bucket weights
    # -------------------------------------------------------------------------
    # Two modes:
    #   include_small_in_train=True  -> use ALL buckets (small+medium+large)
    #                                   but with lower weight for small
    #   include_small_in_train=False -> EXCLUDE small-bucket samples entirely;
    #                                   train only on medium+large
    # WHY the option to exclude small?  Small-bucket samples are mostly noise
    # and can dilute the gradient signal from the more informative medium/large
    # moves.  However, including them with a low weight (W_SMALL=0.6) can help
    # the model learn that "most of the time, the return is small" without
    # sacrificing too much signal quality.
    INCLUDE_SMALL_IN_TRAIN = include_small_in_train

    if INCLUDE_SMALL_IN_TRAIN:
        mask_train_ml = np.ones_like(bucket_train_full, dtype=bool)
        mask_val_ml   = np.ones_like(bucket_val_full, dtype=bool)
    else:
        mask_train_ml = bucket_train_full != 0
        mask_val_ml   = bucket_val_full   != 0

    X_train_ml = X_train_full[mask_train_ml]
    y_train_ml = y_train_full[mask_train_ml]
    b_train_ml = bucket_train_full[mask_train_ml]

    X_val_ml   = X_val_full[mask_val_ml]
    y_val_ml   = y_val_full[mask_val_ml]
    b_val_ml   = bucket_val_full[mask_val_ml]

    n_train_ml, n_val_ml = len(X_train_ml), len(X_val_ml)
    if n_train_ml == 0:
        raise ValueError(
            "No training samples remain after bucket filtering. "
            "Adjust include_small_in_train or bucket thresholds."
        )
    if n_val_ml == 0:
        raise ValueError(
            "No validation samples remain after bucket filtering. "
            "Adjust include_small_in_train or split fractions."
        )
    print(f"Train (used for loss): {n_train_ml:,} samples | Val: {n_val_ml:,} samples")

    # --------- Sample weights: SMALL / MEDIUM / LARGE ----------
    # WHY per-sample weights?  The weighted MSE loss is:
    #   loss = mean( w_i * (yhat_i - y_i)^2 )
    # By assigning W_LARGE > W_MEDIUM > W_SMALL, we tell the optimizer
    # that getting large-move predictions right is 2x more important than
    # medium moves and ~3.3x more important than small moves (with the
    # default weights 0.6 / 1.0 / 2.0).  This is analogous to importance
    # sampling: rare but valuable events receive higher weight.
    W_SMALL_  = W_SMALL
    W_MEDIUM_ = W_MEDIUM
    W_LARGE_  = W_LARGE

    w_train_ml = np.full_like(y_train_ml, W_MEDIUM_, dtype=np.float32)
    w_train_ml[b_train_ml == 0] = W_SMALL_
    w_train_ml[b_train_ml == 2] = W_LARGE_

    w_val_ml = np.full_like(y_val_ml, W_MEDIUM_, dtype=np.float32)
    w_val_ml[b_val_ml == 0] = W_SMALL_
    w_val_ml[b_val_ml == 2] = W_LARGE_

    # -------------------------------------------------------------------------
    # STEP 6: Feature + target normalization
    # -------------------------------------------------------------------------
    # See _prepare_scalers_and_normalize() docstring for full details on
    # the 3D->2D flatten, StandardScaler fit/transform, and reshape back.
    norm = _prepare_scalers_and_normalize(
        X_train_ml=X_train_ml,
        X_val_ml=X_val_ml,
        X_train_full=X_train_full,
        X_val_full=X_val_full,
        X_test_full=X_test_full,
        y_train_ml=y_train_ml,
        y_val_ml=y_val_ml,
        y_train_full=y_train_full,
        y_val_full=y_val_full,
        y_test_full=y_test_full,
    )

    scaler_X: StandardScaler = norm["scaler_X"]
    scaler_y: StandardScaler = norm["scaler_y"]

    X_train_ml = norm["X_train_ml"]
    X_val_ml = norm["X_val_ml"]
    X_train_full = norm["X_train_full"]
    X_val_full = norm["X_val_full"]
    X_test_full = norm["X_test_full"]

    y_train_ml_z = norm["y_train_ml_z"]
    y_val_ml_z = norm["y_val_ml_z"]

    print("Feature scaler (train subset) mean (first 3 feats):", scaler_X.mean_[:min(3, F)])
    print("Feature scaler (train subset) scale (first 3 feats):", scaler_X.scale_[:min(3, F)])
    print("Target scaler (train subset) mean/scale:",
          float(scaler_y.mean_[0]), float(scaler_y.scale_[0]))

    # -------------------------------------------------------------------------
    # STEP 7: Wrap normalized arrays into PyTorch Datasets
    # -------------------------------------------------------------------------
    # These datasets carry (X_normalized, y_z_scored, weight) triples and
    # are consumed by DataLoaders during training.
    train_dataset = SequenceDatasetWeighted(X_train_ml, y_train_ml_z, w_train_ml, T, F)
    val_dataset   = SequenceDatasetWeighted(X_val_ml,   y_val_ml_z,   w_val_ml,   T, F)

    # -------------------------------------------------------------------------
    # STEP 8: LHS hyperparameter search
    # -------------------------------------------------------------------------
    # The search evaluates n_samples_search (default=50) hyperparameter
    # combinations sampled via Latin Hypercube Sampling.  Each combination
    # trains a fresh LSTM on the training subset with early stopping,
    # and the candidate with the lowest validation loss is selected.
    #
    # WHY LHS over grid/random?  See the LHS section comments above.
    # In short, LHS provides better coverage of the search space than
    # random sampling with the same budget of trials.
    GRID = DEFAULT_GRID if grid is None else grid

    # Separate RNG for LHS so that changing the training seed does not
    # alter the hyperparameter sample set.
    rng = np.random.default_rng(lhs_seed)
    grid_list = sample_params_lhs(n_samples_search, GRID, rng)
    print(f"LHS hyperparameter search: sampling {n_samples_search} combinations.")
    if device.type == "cpu" and n_samples_search > 20:
        print(
            "[SGU2] CPU run detected. LHS progress is updated once per trial; "
            "the first update may take some time."
        )

    best: Optional[Dict[str, Any]] = None

    # Debug: confirm datasets and device before entering LHS loop
    print(f"[SGU2 DEBUG] train_dataset len={len(train_dataset)}, val_dataset len={len(val_dataset)}")
    print(f"[SGU2 DEBUG] device={device}, num_workers={num_workers}, pin_memory={pin_memory}")
    print(f"[SGU2 DEBUG] Starting LHS loop with {n_samples_search} trials...")
    import sys; sys.stdout.flush()

    with tqdm(
        grid_list,
        desc="LHS Grid",
        ncols=120,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as bar:
        bar.set_postfix(rmse="nan", mae="nan", best="inf")

        n_trials = len(grid_list)
        for trial_idx, combo in enumerate(bar, start=1):
            trial_start = time.perf_counter()
            print(f"[SGU2 DEBUG] Trial {trial_idx}/{n_trials}: {combo}", flush=True)
            # Unpack the hyperparameter combination for this trial
            kern_l2   = combo["kernel_l2"]
            bias_l2   = combo["bias_l2"]
            dropout_p = combo["dropout"]
            epochs    = combo["epochs"]
            batch     = combo["batch_size"]
            hid       = combo["hidden_units"]
            n_layers  = combo["num_layers"]
            lr        = combo["lr"]

            # Train one LSTM with this HP combination (with early stopping)
            model_tmp, val_loss, val_mae = _fit_one(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
                F=F,
                dropout_p=dropout_p,
                l2_kernel=kern_l2,
                l2_bias=bias_l2,
                lr=lr,
                epochs=epochs,
                batch_size=batch,
                hidden_units=hid,
                patience=patience,
                num_layers=n_layers,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            trial_seconds = time.perf_counter() - trial_start
            val_rmse = float(np.sqrt(val_loss)) if np.isfinite(val_loss) else float("inf")

            # Track the best trial by validation loss (weighted MSE).
            # We save the full state_dict (on CPU) so we can restore it later.
            if np.isfinite(val_loss) and ((best is None) or (val_loss < best["val_loss"])):
                best = dict(
                    val_loss=val_loss,
                    val_mae=val_mae,
                    combo=dict(
                        kernel_l2=kern_l2,
                        bias_l2=bias_l2,
                        dropout=dropout_p,
                        epochs=epochs,
                        batch_size=batch,
                        hidden_units=hid,
                        num_layers=n_layers,
                        lr=lr,
                    ),
                    state={k: v.cpu().clone() for k, v in model_tmp.state_dict().items()},
                )
            elif not np.isfinite(val_loss):
                warnings.warn(
                    f"Skipping SGU2 LHS trial {trial_idx}/{n_trials}: non-finite validation loss.",
                    RuntimeWarning,
                )

            best_rmse = np.sqrt(best["val_loss"]) if best is not None else np.inf

            bar.set_postfix(
                rmse=f"{val_rmse:.3e}",
                mae=f"{val_mae:.3e}",
                best=f"{best_rmse:.3e}",
                sec=f"{trial_seconds:.1f}",
            )
            bar.set_description(f"LHS Grid ({trial_idx}/{n_trials})")

            # Explicitly delete the temporary model and free GPU memory
            # to avoid OOM errors during a long search.
            del model_tmp
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError(
            "LHS search finished with no finite validation losses. "
            "Check input tensors for NaN/inf and consider narrowing the LR/L2 grid."
        )

    print(
        "\nBest (by weighted validation MSE, normalized):", best["combo"],
        "| val_RMSE(z)=", np.sqrt(best["val_loss"]),
        "| val_MAE(z)=", best["val_mae"],
    )

    # -------------------------------------------------------------------------
    # STEP 9: Restore best model from the HP search (trained on 64%)
    # -------------------------------------------------------------------------
    # Reconstruct the model architecture and load the best weights.
    # This "best_model_64" is kept as a reference point: it shows the
    # performance of the best hyperparameters trained on the 64% split only.
    best_model_64 = LSTMRegressor(
        input_dim=F,
        hidden_units=best["combo"]["hidden_units"],
        dropout_p=best["combo"]["dropout"],
        num_layers=best["combo"]["num_layers"],
    ).to(device)
    best_model_64.load_state_dict({k: v.to(device) for k, v in best["state"].items()})

    # -------------------------------------------------------------------------
    # STEP 10: Evaluate best_model_64 on all splits
    # -------------------------------------------------------------------------
    # Predictions are in z-scored space; inverse_transform converts them
    # back to the original return scale for interpretable metrics.
    y_pred_train_64_z = predict_numpy(best_model_64, X_train_full, device=device)
    y_pred_val_64_z   = predict_numpy(best_model_64, X_val_full,   device=device)
    y_pred_test_64_z  = predict_numpy(best_model_64, X_test_full,  device=device)

    y_pred_train_64_real = scaler_y.inverse_transform(y_pred_train_64_z.reshape(-1, 1)).ravel()
    y_pred_val_64_real   = scaler_y.inverse_transform(y_pred_val_64_z.reshape(-1, 1)).ravel()
    y_pred_test_64_real  = scaler_y.inverse_transform(y_pred_test_64_z.reshape(-1, 1)).ravel()

    print("\n================ METRICS FOR BEST MODEL (TRAINED ON 64% SUBSET) ================")
    metrics_train = compute_metrics_split(
        "TRAIN 64% (best model, real)",
        y_train_orig,
        y_pred_train_64_real,
        bucket_train_full,
        q1,
    )
    metrics_val = compute_metrics_split(
        "VAL 16% (best model, real)",
        y_val_orig,
        y_pred_val_64_real,
        bucket_val_full,
        q1,
    )
    metrics_test = compute_metrics_split(
        "TEST 20% (best model, real)",
        y_test_orig,
        y_pred_test_64_real,
        bucket_test_full,
        q1,
    )

    # -------------------------------------------------------------------------
    # STEP 11: Final refit on TRAIN+VAL (80%) -- WARM-START REFIT
    # -------------------------------------------------------------------------
    # WHY refit on 80% of data?
    #   The best_model_64 was trained on only the 64% training split.
    #   Now that we have selected the best hyperparameters, we can safely
    #   include the 16% validation data in the training pool to give the
    #   final production model more data to learn from.  The 20% test set
    #   remains completely untouched for final evaluation.
    #
    # WHY warm-start (not train from scratch)?
    #   Starting from the best_model_64's weights rather than random
    #   initialization has two advantages:
    #     1. The model retains the patterns already learned during the
    #        HP search, avoiding "catastrophic forgetting."
    #     2. Convergence is much faster because the weights are already
    #        in a good region of the loss landscape.
    #
    # EARLY STOPPING IN THE REFIT:
    #   Even though we are refitting on a larger dataset, we still need
    #   early stopping to prevent overtraining.  We carve out 10% of the
    #   train+val pool as a refit-validation set (the last 10%
    #   chronologically) and monitor loss on this small held-out slice.
    print("\n===================== REFIT MODEL ON TRAIN+VAL (80%) MED+LARGE =====================")

    # Concatenate the already-normalized train and val splits.
    # Note: these are ALREADY normalized by scaler_X (from step 6).
    X_tv_full   = np.concatenate([X_train_full, X_val_full], axis=0)
    y_tv_full   = np.concatenate([y_train_full, y_val_full], axis=0)
    bucket_tv   = np.concatenate([bucket_train_full, bucket_val_full], axis=0)

    # Apply the same bucket filtering as in the original training
    if INCLUDE_SMALL_IN_TRAIN:
        mask_tv_ml = np.ones_like(bucket_tv, dtype=bool)
    else:
        mask_tv_ml = bucket_tv != 0

    X_tv_ml = X_tv_full[mask_tv_ml]
    y_tv_ml = y_tv_full[mask_tv_ml]
    b_tv_ml = bucket_tv[mask_tv_ml]

    n_tv_ml = X_tv_ml.shape[0]
    print("Train+Val MED+LARGE size:", n_tv_ml)

    # Assign per-bucket weights (same scheme as in step 5)
    w_tv_ml = np.full_like(y_tv_ml, W_MEDIUM_, dtype=np.float32)
    w_tv_ml[b_tv_ml == 0] = W_SMALL_
    w_tv_ml[b_tv_ml == 2] = W_LARGE_

    assert X_tv_ml.shape[1:] == (T, F), f"Unexpected X_tv_ml shape: {X_tv_ml.shape}"

    # Z-score the targets using the SAME scaler_y fitted in step 6.
    # WHY the same scaler?  Using a different scaler would change the
    # meaning of the z-scores and make the warm-started weights
    # misaligned with the new target distribution.
    y_tv_ml_z  = scaler_y.transform(y_tv_ml.reshape(-1, 1)).ravel()

    tv_dataset = SequenceDatasetWeighted(X_tv_ml, y_tv_ml_z, w_tv_ml, T, F)

    # Create a new LSTMRegressor with the same architecture as the best trial.
    final_model = LSTMRegressor(
        input_dim=F,
        hidden_units=best["combo"]["hidden_units"],
        dropout_p=best["combo"]["dropout"],
        num_layers=best["combo"]["num_layers"],
    ).to(device)

    # WARM-START: initialize from best_model_64 weights instead of random.
    # This is the key step that transfers knowledge from the HP search
    # phase to the refit phase.  The model starts in a good region of
    # parameter space and only needs fine-tuning on the additional 16%
    # of data.
    final_model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})

    # Use the same optimizer configuration as the winning HP combination.
    optimizer_final = torch.optim.Adam(
        build_param_groups(final_model, best["combo"]["kernel_l2"], best["combo"]["bias_l2"]),
        lr=best["combo"]["lr"],
    )

    # Use the same epoch budget that worked best during the search.
    N_EPOCHS_FINAL = best["combo"]["epochs"]

    # Split the 80% train+val pool into refit_train (90%) + refit_val (10%)
    # for early stopping during the refit.  This prevents the warm-started
    # model from overfitting on the expanded dataset.
    n_tv_total = len(X_tv_ml)
    n_refit_val = max(1, int(0.1 * n_tv_total))
    n_refit_train = n_tv_total - n_refit_val

    refit_train_ds = SequenceDatasetWeighted(
        X_tv_ml[:n_refit_train], y_tv_ml_z[:n_refit_train],
        w_tv_ml[:n_refit_train], T, F,
    )
    refit_val_ds = SequenceDatasetWeighted(
        X_tv_ml[n_refit_train:], y_tv_ml_z[n_refit_train:],
        w_tv_ml[n_refit_train:], T, F,
    )
    # Pass raw datasets instead of DataLoader to avoid resource_tracker deadlock.
    refit_train_loader = refit_train_ds
    refit_val_loader = refit_val_ds

    best_refit_loss = np.inf
    best_refit_state = None
    no_improve_refit = 0
    refit_epoch_count = 0

    with tqdm(range(N_EPOCHS_FINAL), desc="Refit(train+val)") as bar:
        for _ in bar:
            refit_epoch_count += 1
            loss_tv = run_epoch_weighted(
                refit_train_loader, final_model, optimizer_final, device, True,
            )
            val_loss_refit = run_epoch_weighted(
                refit_val_loader, final_model, optimizer_final, device, False,
            )
            bar.set_postfix(train=f"{loss_tv:.3e}", val=f"{val_loss_refit:.3e}")

            if val_loss_refit < best_refit_loss:
                best_refit_loss = val_loss_refit
                best_refit_state = {
                    k: v.cpu().clone() for k, v in final_model.state_dict().items()
                }
                no_improve_refit = 0
            else:
                no_improve_refit += 1
                if no_improve_refit >= patience and refit_epoch_count >= min_refit_epochs:
                    break

    if best_refit_state is not None:
        final_model.load_state_dict(
            {k: v.to(device) for k, v in best_refit_state.items()}
        )

    # -------------------------------------------------------------------------
    # STEP 12: Final predictions + metrics
    # -------------------------------------------------------------------------
    # Predict on all three splits using the refitted final_model.
    # Predictions are in z-scored space, so we must inverse-transform them
    # back to real return scale for meaningful metrics.
    y_pred_tv_train_z = predict_numpy(final_model, X_train_full, device=device)
    y_pred_tv_val_z   = predict_numpy(final_model, X_val_full,   device=device)
    y_pred_tv_test_z  = predict_numpy(final_model, X_test_full,  device=device)

    # Inverse-transform: y_real = y_z * scale + mean
    y_pred_tv_train_real = scaler_y.inverse_transform(y_pred_tv_train_z.reshape(-1, 1)).ravel()
    y_pred_tv_val_real   = scaler_y.inverse_transform(y_pred_tv_val_z.reshape(-1, 1)).ravel()
    y_pred_tv_test_real  = scaler_y.inverse_transform(y_pred_tv_test_z.reshape(-1, 1)).ravel()

    print("\n===================== FINAL MODEL (TRAIN+VAL=80%) METRICS =====================")
    metrics_tv = compute_metrics_split(
        "TRAIN+VAL 80% (final, real)",
        np.concatenate([y_train_orig, y_val_orig]),
        np.concatenate([y_pred_tv_train_real, y_pred_tv_val_real]),
        np.concatenate([bucket_train_full, bucket_val_full]),
        q1,
    )
    metrics_final_test = compute_metrics_split(
        "TEST 20% (final, real)",
        y_test_orig,
        y_pred_tv_test_real,
        bucket_test_full,
        q1,
    )

    # -------------------------------------------------------------------------
    # STEP 13: Optional artifact saving (model/scalers/config)
    # -------------------------------------------------------------------------
    # If save_dir is provided, persist all artifacts needed to reload and
    # use the model later without retraining.
    artifacts = dict(
        input_dim=int(F),
        T_seq=int(T),
        best_combo=best["combo"],
        bucket_thresholds=dict(q1=float(q1), q2=float(q2)),
        feature_names=list(all_feature_cols),
        metrics_best_model_64=dict(train=metrics_train, val=metrics_val, test=metrics_test),
        metrics_final_model=dict(trainval=metrics_tv, test=metrics_final_test),
    )

    if save_dir is not None:
        save_dir = _ensure_dir(save_dir)
        _save_training_artifacts(
            save_dir=save_dir,
            artifacts=artifacts,
            best_model_64=best_model_64,
            final_model=final_model,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            overwrite=overwrite,
        )
        print(f"Saved artifacts to: {save_dir}")

    # -------------------------------------------------------------------------
    # STEP 14: Return everything in a structured dictionary
    # -------------------------------------------------------------------------
    out: Dict[str, Any] = dict(
        best_model_64=best_model_64,
        final_model=final_model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        artifacts=artifacts,
        feature_names=list(all_feature_cols),
        bucket_thresholds=dict(q1=float(q1), q2=float(q2)),
        best=best,
        metrics=dict(
            best_model_64=dict(train=metrics_train, val=metrics_val, test=metrics_test),
            final_model=dict(trainval=metrics_tv, test=metrics_final_test),
        ),
        device=device,
        # Real-scale targets and predictions — needed by downstream visualization.
        predictions=dict(
            y_train_orig=y_train_orig,
            y_val_orig=y_val_orig,
            y_test_orig=y_test_orig,
            y_pred_train_64_real=y_pred_train_64_real,
            y_pred_val_64_real=y_pred_val_64_real,
            y_pred_test_64_real=y_pred_test_64_real,
            y_pred_tv_train_real=y_pred_tv_train_real,
            y_pred_tv_val_real=y_pred_tv_val_real,
            y_pred_tv_test_real=y_pred_tv_test_real,
        ),
    )

    if return_data:
        # THE return_data NORMALIZATION FIX:
        # When the caller requests the full dataset (return_data=True), we
        # must normalize X_all using the SAME scaler_X that was fit on the
        # training split.  The individual splits (X_train_full, X_val_full,
        # X_test_full) are already normalized, but X_all is still in raw
        # scale because it was not passed through the scaler above.
        #
        # We flatten X_all from 3D (N, T, F) to 2D (N*T, F), apply the
        # scaler's transform (NOT fit_transform -- the scaler was already
        # fitted), and reshape back to 3D.
        #
        # We also keep the RAW (un-normalized) X_all as X_all_raw, because
        # some downstream code (e.g., visualization or feature importance
        # analysis) may need the original feature values.
        X_all_2d = X_all.reshape(-1, F)
        X_all_norm = scaler_X.transform(X_all_2d).reshape(X_all.shape[0], T, F)

        out["data"] = dict(
            X_all=X_all_norm,
            X_all_raw=X_all,
            y_all=y_all,
            splits=dict(
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_val_full=X_val_full,
                y_val_full=y_val_full,
                X_test_full=X_test_full,
                y_test_full=y_test_full,
            ),
            buckets=dict(
                bucket_train_full=bucket_train_full,
                bucket_val_full=bucket_val_full,
                bucket_test_full=bucket_test_full,
            ),
        )

    return out


def _save_training_artifacts(
    *,
    save_dir: Path,
    artifacts: Dict[str, Any],
    best_model_64: nn.Module,
    final_model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    overwrite: bool = False,
) -> None:
    """
    Save a minimal, reproducible artifact package to disk.

    The package consists of 5 files:
      - artifacts.json: Human-readable JSON containing the best hyperparameter
        combination, bucket thresholds (q1, q2), feature names, input_dim,
        T_seq, and evaluation metrics for both models.  This file serves as
        a "recipe" to reconstruct the model architecture and understand the
        training context without loading any binary files.
      - scaler_X.pkl: Fitted StandardScaler for features (needed at inference
        to normalize new input data the same way as training data).
      - scaler_y.pkl: Fitted StandardScaler for targets (needed at inference
        to inverse-transform predictions back to real return scale).
      - best_model_64.pt: PyTorch state_dict of the best model from the HP
        search (trained on 64% of data).  Useful for comparison.
      - final_model.pt: PyTorch state_dict of the production model
        (warm-start refitted on 80% of data).  This is the model to use
        in deployment.

    WHY save state_dicts instead of full models?  State dicts are more
    portable (no dependency on the exact class definition path), smaller,
    and recommended by PyTorch best practices.

    If overwrite=False (default), refuses to overwrite existing files to
    prevent accidental loss of previously trained artifacts.
    """
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    artifacts_path = save_dir / "artifacts.json"
    scalerX_path = save_dir / "scaler_X.pkl"
    scalery_path = save_dir / "scaler_y.pkl"
    best_path = save_dir / "best_model_64.pt"
    final_path = save_dir / "final_model.pt"

    # Safety check: refuse to overwrite unless explicitly requested.
    if not overwrite:
        for p in [artifacts_path, scalerX_path, scalery_path, best_path, final_path]:
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {p}")

    # Save the JSON artifact manifest (human-readable)
    with open(artifacts_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                input_dim=artifacts['input_dim'],
                T_seq=artifacts['T_seq'],
                best_combo=artifacts['best_combo'],
                bucket_thresholds=artifacts['bucket_thresholds'],
                feature_names=artifacts['feature_names'],
                metrics_best_model_64=artifacts['metrics_best_model_64'],
                metrics_final_model=artifacts['metrics_final_model'],
            ),
            f,
            indent=2,
            sort_keys=True,
        )

    # Save scikit-learn scalers via pickle
    save_pickle(scaler_X, scalerX_path)
    save_pickle(scaler_y, scalery_path)

    # Save PyTorch model weights (state_dicts only)
    torch.save(best_model_64.state_dict(), best_path)
    torch.save(final_model.state_dict(), final_path)


def load_trained_sgu2_artifacts(
    save_dir: PathLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load a previously saved artifact package (see ``_save_training_artifacts``).

    This function is the inverse of ``_save_training_artifacts``: it reads
    the JSON manifest, reconstructs the LSTMRegressor architecture using
    the saved hyperparameters, loads the state_dicts, and deserializes the
    scikit-learn scalers.

    WHY reconstruct from artifacts.json rather than pickling the whole model?
    Because PyTorch state_dicts are more robust across code changes.  If you
    rename a module or move a class to a different file, a pickled model
    would break, but a state_dict can still be loaded into the correctly
    defined architecture.

    Returns
    -------
    dict with keys:
        artifacts, scaler_X, scaler_y, best_model_64, final_model, device.
    """
    save_dir = Path(save_dir).expanduser().resolve()
    device = resolve_device(device)

    artifacts_path = save_dir / "artifacts.json"
    scalerX_path = save_dir / "scaler_X.pkl"
    scalery_path = save_dir / "scaler_y.pkl"
    best_path = save_dir / "best_model_64.pt"
    final_path = save_dir / "final_model.pt"

    with open(artifacts_path, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    scaler_X = load_pickle(scalerX_path)
    scaler_y = load_pickle(scalery_path)

    input_dim = int(artifacts["input_dim"])
    best_combo = artifacts["best_combo"]

    best_model_64 = LSTMRegressor(
        input_dim=input_dim,
        hidden_units=int(best_combo["hidden_units"]),
        dropout_p=float(best_combo["dropout"]),
        num_layers=int(best_combo["num_layers"]),
    ).to(device)

    final_model = LSTMRegressor(
        input_dim=input_dim,
        hidden_units=int(best_combo["hidden_units"]),
        dropout_p=float(best_combo["dropout"]),
        num_layers=int(best_combo["num_layers"]),
    ).to(device)

    best_model_64.load_state_dict(_safe_torch_load_state(best_path, device))
    final_model.load_state_dict(_safe_torch_load_state(final_path, device))

    return dict(
        artifacts=artifacts,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        best_model_64=best_model_64,
        final_model=final_model,
        device=device,
    )


def _safe_torch_load_state(path, device):
    """
    Load a PyTorch state dict with weights_only=True for security.

    WHY weights_only=True?  torch.load() by default uses Python's pickle
    module, which can execute arbitrary code during deserialization.
    Setting weights_only=True restricts loading to only tensor data,
    preventing potential code injection from tampered .pt files.

    Falls back gracefully for older PyTorch versions (< 1.13) that do
    not support the weights_only parameter.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# ============================================================================
# SECTION 9: KEYED CACHING UTILITIES (content-addressed artifact management)
# ============================================================================
# These utilities implement a content-addressed caching system for SGU2
# artifacts.  The "cache key" is a short SHA-1 hash derived from a metadata
# dictionary (symbol, dates, delta_t, etc.).  This ensures that:
#   - Artifacts trained with identical settings share the same key and can
#     be re-loaded without retraining.
#   - Changing ANY training parameter (even tick_size) produces a different
#     key, preventing stale cache hits.
#
# The pattern is:
#   1. Build a metadata dict via _sgu2_meta_config(...)
#   2. Compute artifact paths via sgu2_artifact_paths(meta=..., artifacts_dir=...)
#   3. Try loading with try_load_sgu2(...) -- returns None if cache miss
#   4. If cache miss, train and save with save_sgu2(...)
#
# This is the same caching strategy used by SGU1, enabling a consistent
# "train-once, reload-many-times" workflow across both modules.
# ============================================================================

def _sgu2_meta_config(
    *,
    symbol: str,
    base_dates: List[str],
    delta_t: int,
    exec_types: list,
    tick_size: float,
    T_SEQ: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a metadata dict that uniquely defines the SGU2 training context.

    WHY a metadata dict?  This dict captures ALL the parameters that affect
    the training output.  By hashing it, we get a unique cache key that
    changes automatically whenever any parameter changes.  This prevents
    the dangerous situation where you load cached artifacts that were
    trained with different settings than your current experiment.

    IMPORTANT:
    - SGU2.py should not depend on runner globals (SYMBOL, BASE_DATES, ...).
      The runner should pass these values explicitly.
    - If you change any field here (or pass different `extra`), the cache key
      changes automatically.
    """
    meta = {
        "symbol": str(symbol),
        "base_dates": list(base_dates),
        "delta_t": int(delta_t),
        "exec_types": list(exec_types),
        "tick_size": float(tick_size),
        "T_SEQ": int(T_SEQ),
        # add more knobs here if they should invalidate the cache:
        # e.g. day_start/day_end, lob_depth, feature-engineering versions, etc.
    }
    if extra:
        # Example: extra={"day_start": "10:00:00", "day_end": "15:30:00"}
        meta.update(extra)
    return meta


def _cache_key_from_meta(meta: Dict[str, Any]) -> str:
    """
    Compute a stable, short cache key from a metadata dictionary.

    Uses SHA-1 over the JSON-serialized dict (with sorted keys for
    stability).  Only the first 12 hex characters are used, providing
    enough uniqueness (16^12 ~ 2.8 * 10^14 possible keys) while keeping
    filenames readable.

    WHY JSON with sorted keys?  Dictionaries in Python do not guarantee
    insertion order in older versions, and even in modern Python, two
    dicts with the same key-value pairs but different insertion orders
    would produce different JSON without sort_keys=True.  Sorting
    ensures that logically identical metadata always produces the
    same hash.
    """
    s = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def sgu2_artifact_paths(
    *,
    meta: Dict[str, Any],
    artifacts_dir: PathLike,
) -> Dict[str, Any]:
    """
    Resolve artifact file paths that are unique for a given training context.

    Each artifact file is named with the pattern:
        SGU2_<artifact_type>_<cache_key>.<extension>

    This naming convention allows multiple training runs (with different
    parameters) to coexist in the same artifacts_dir without collisions.
    """
    artifacts_dir = _ensure_dir(artifacts_dir)
    key = _cache_key_from_meta(meta)
    return {
        "key": key,
        "dir": artifacts_dir,
        "artifacts": artifacts_dir / f"SGU2_artifacts_{key}.json",
        "scaler_X": artifacts_dir / f"SGU2_scaler_X_{key}.pkl",
        "scaler_y": artifacts_dir / f"SGU2_scaler_y_{key}.pkl",
        "best_model_64": artifacts_dir / f"SGU2_best_model_64_{key}.pt",
        "final_model": artifacts_dir / f"SGU2_final_model_{key}.pt",
        "meta": artifacts_dir / f"SGU2_meta_{key}.json",
    }


def try_load_sgu2(
    *,
    meta: Dict[str, Any],
    artifacts_dir: PathLike,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Attempt to load SGU2 artifacts for a given metadata context.

    This is a "cache-aware" loader: it computes the expected file paths
    from the metadata hash, checks if ALL required files exist, and if
    so, loads them and returns a fully reconstructed dict with models,
    scalers, and metadata.

    Returns None if any required artifact file is missing (cache miss).
    The caller should then fall back to training from scratch and saving
    the results with save_sgu2().
    """
    p = sgu2_artifact_paths(meta=meta, artifacts_dir=artifacts_dir)

    required = [p["artifacts"], p["scaler_X"], p["scaler_y"], p["best_model_64"], p["final_model"], p["meta"]]
    if not all(Path(x).exists() for x in required):
        return None

    device = resolve_device(device)

    # load JSON artifacts (contains model config to reconstruct the network)
    with open(p["artifacts"], "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    scaler_X = load_pickle(p["scaler_X"])
    scaler_y = load_pickle(p["scaler_y"])

    input_dim = int(artifacts["input_dim"])
    best_combo = artifacts["best_combo"]

    best_model_64 = LSTMRegressor(
        input_dim=input_dim,
        hidden_units=int(best_combo["hidden_units"]),
        dropout_p=float(best_combo["dropout"]),
        num_layers=int(best_combo["num_layers"]),
    ).to(device)

    final_model = LSTMRegressor(
        input_dim=input_dim,
        hidden_units=int(best_combo["hidden_units"]),
        dropout_p=float(best_combo["dropout"]),
        num_layers=int(best_combo["num_layers"]),
    ).to(device)

    def _safe_torch_load_state(path, device):
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            # Older PyTorch versions don't support weights_only
            return torch.load(path, map_location=device)
    
    best_model_64.load_state_dict(_safe_torch_load_state(p["best_model_64"], device))
    final_model.load_state_dict(_safe_torch_load_state(p["final_model"], device))

    with open(p["meta"], "r", encoding="utf-8") as f:
        meta_on_disk = json.load(f)

    return {
        "best_model_64": best_model_64,
        "final_model": final_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "artifacts": artifacts,
        "meta": meta_on_disk,
        "cache_key": p["key"],
        "device": device,
    }


def save_sgu2(
    *,
    out: Dict[str, Any],
    meta: Dict[str, Any],
    artifacts_dir: PathLike,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Save SGU2 outputs (models + scalers + metadata) using content-addressed
    file names derived from the metadata hash.

    This is the "cache-aware" saver that complements try_load_sgu2().
    The caller typically passes the raw output dict from
    ``train_sgu2_lstm_pipeline`` as ``out`` and the metadata dict from
    ``_sgu2_meta_config`` as ``meta``.

    The function saves 6 files:
      - SGU2_artifacts_<key>.json   -- model config + metrics
      - SGU2_scaler_X_<key>.pkl    -- feature scaler
      - SGU2_scaler_y_<key>.pkl    -- target scaler
      - SGU2_best_model_64_<key>.pt -- best model state_dict (64% train)
      - SGU2_final_model_<key>.pt  -- final model state_dict (80% train+val)
      - SGU2_meta_<key>.json       -- the metadata dict itself (for provenance)

    Returns
    -------
    dict with 'cache_key' and 'paths' (resolved file paths for each artifact).
    """
    # Compute content-addressed file paths from the metadata hash.
    p = sgu2_artifact_paths(meta=meta, artifacts_dir=artifacts_dir)

    # Safety: refuse to overwrite existing files unless explicitly requested.
    if not overwrite:
        for k in ["artifacts", "scaler_X", "scaler_y", "best_model_64", "final_model", "meta"]:
            if Path(p[k]).exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {p[k]}")

    # Extract or build the artifacts JSON dict.
    # If the caller passes the raw output of train_sgu2_lstm_pipeline(),
    # out["artifacts"] is already populated.  Otherwise, we reconstruct
    # a minimal artifacts dict from the available fields.
    artifacts = out.get("artifacts", None)
    if artifacts is None:
        # If the caller passes the raw output of `train_sgu2_lstm_pipeline`, build a minimal package
        t_seq = None
        if "T_SEQ" in out:
            t_seq = int(out["T_SEQ"])
        elif "T_SEQ" in meta:
            t_seq = int(meta["T_SEQ"])
        elif "T_seq" in meta:
            t_seq = int(meta["T_seq"])

        if t_seq is None:
            raise ValueError(
                "Cannot infer T_seq for SGU2 artifacts. "
                "Provide out['artifacts']['T_seq'], out['T_SEQ'], or meta['T_SEQ']."
            )

        artifacts = {
            "input_dim": int(out["best_model_64"].lstm.input_size),
            "T_seq": t_seq,
            "best_combo": out["best"]["combo"] if isinstance(out.get("best", None), dict) else out.get("best_combo", {}),
            "bucket_thresholds": out.get("bucket_thresholds", {}),
            "feature_names": out.get("feature_names", []),
            "metrics_best_model_64": out.get("metrics", {}).get("best_model_64", {}),
            "metrics_final_model": out.get("metrics", {}).get("final_model", {}),
        }
    else:
        # Ensure persisted artifacts always carry a valid T_seq.
        if ("T_seq" not in artifacts) or (artifacts["T_seq"] in (None, 0)):
            if "T_SEQ" in meta:
                artifacts["T_seq"] = int(meta["T_SEQ"])
            elif "T_seq" in meta:
                artifacts["T_seq"] = int(meta["T_seq"])
            else:
                raise ValueError(
                    "artifacts is missing a valid T_seq and meta does not provide T_SEQ/T_seq."
                )

    # Write the artifacts JSON (model config + metrics)
    with open(p["artifacts"], "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, sort_keys=True)

    # Persist scikit-learn scalers via pickle
    save_pickle(out["scaler_X"], p["scaler_X"])
    save_pickle(out["scaler_y"], p["scaler_y"])

    # Persist PyTorch model state_dicts
    torch.save(out["best_model_64"].state_dict(), p["best_model_64"])
    torch.save(out["final_model"].state_dict(), p["final_model"])

    # Save the metadata dict itself for provenance tracking.
    # WHY save meta separately?  So that a human (or script) can inspect
    # which settings produced this cache key without loading any binary files.
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return {"cache_key": p["key"], "paths": p}
