#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL_MM_strat_runner.py  --  End-to-end pipeline for DRL Market Making

Created on Tue Dec 16 03:57:51 2025
@author: felipemoret

Overview
--------
This script orchestrates the full pipeline from raw LOBSTER data to a
Deep Reinforcement Learning (DRL) market-making backtest.  It is designed
to run in Spyder (or any Python environment), NOT inside a notebook.

CONCEPTUAL SUMMARY
~~~~~~~~~~~~~~~~~~
The pipeline has a layered architecture that mirrors how information flows
from raw exchange data to actionable trading decisions:

    Raw LOBSTER data  (individual LOB events at microsecond resolution)
         |
         v
    Day-safe TOB-move windows  (aggregated "time bars" based on best-bid/ask changes)
         |
         v
    SGU-1 (XGBoost)  -->  Realized-range prediction  (spread/volatility signal)
    SGU-2 (LSTM)     -->  Pseudo-mid trend prediction (directional signal)
         |
         v
    DRL Agent  (uses SGU signals as state features to decide market-making actions)
         |
         v
    3-Phase Backtest  (train / validation / test -- always normalized from train)

WHY THIS ARCHITECTURE?
  - Market-making profitability depends on two key factors:
    (1) Spread dynamics -- how wide or narrow is the effective spread?  (SGU-1)
    (2) Price direction -- is the mid-price trending up or down?          (SGU-2)
  - Rather than feeding raw LOB data directly into a DRL agent (which would
    require enormous state spaces and training data), we use supervised-learning
    models (XGBoost, LSTM) as "feature extractors" that compress complex LOB
    dynamics into two interpretable signals.
  - The DRL agent then learns *when and how* to place/cancel limit orders
    given these compressed signals, which drastically reduces the RL problem's
    dimensionality and makes learning tractable.

DATA FLOW AND SPLITTING STRATEGY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The dataset is split TWICE, always chronologically (no shuffling):

  First split (SGU level):
    [-------- 64% Train --------|--- 16% Val ---|--- 20% Test ---]
    ^                                                            ^
    |-- SGU models trained here                                  |
    |-- SGU test set = "DRL Universe"  --------------------------+

  Second split (DRL level, within SGU test set only):
    SGU Test 20%:  [--- 64% DRL Train ---|-- 16% DRL Val --|-- 20% DRL Test --]

  This nested splitting ensures that:
    (a) SGU models never see the data the DRL agent trains on.
    (b) The DRL agent's test set is completely out-of-sample for both SGU and DRL.
    (c) No future information leaks into any training stage.

Pipeline stages
~~~~~~~~~~~~~~~
 0) Configuration         -- All user-editable knobs live in one place.
 1) Data loading          -- Load multi-day LOBSTER CSVs, normalize, filter.
 2) Window construction   -- Build day-safe TOB-move windows (no cross-day).
 3) Duration analysis     -- Compute and visualize window duration distribution.
 4) SGU-1 labels+features -- Realized-range labels, hand-crafted features.
 5) SGU-1 training        -- XGBoost with chronological 64/16/20 split.
 6) SGU-1 evaluation      -- Reconstruct splits, plot true vs predicted.
 7) SGU-2 labels+features -- Pseudo-mid trend labels, multi-feature set.
 8) SGU-2 training        -- LSTM with chronological 64/16/20 split.
 9) SGU-2 evaluation      -- Reconstruct splits, plot true vs predicted.
10) DRL policy setup      -- Map SGU predictions to DRL signal universe.
11) DRL backtests         -- Run train/val/test backtests via MM_LOB_SIM.

Data Convention (unified with LOB_data / notebook)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Column names   : CamelCase (AskPrice_1, BidPrice_1, Time, Type, Direction, ...)
- Type           : LOBSTER integer (1=LO, 2=Cancel, 3=Delete, 4=ExecVisible, 5=ExecHidden, 7=Halt)
- Direction      : LOBSTER integer (+1 = passive buy LO hit, -1 = passive sell LO hit)
                   IMPORTANT: in LOBSTER, Direction refers to the *passive* side.
                   Direction == -1 means the aggressor was a BUYER (they "hit" a sell LO).
                   Direction == +1 means the aggressor was a SELLER (they "hit" a buy LO).
- Time           : float seconds-after-midnight (NOT datetime)
                   e.g., 34200.123 = 09:30:00.123 AM.
                   This resets to 0 at midnight, which is why we need TimeAbs for multi-day.
- TimeAbs        : monotonically increasing = Time + 86400 * DayID
                   86400 = number of seconds in a day. This creates a globally unique timestamp
                   that never decreases, even across overnight gaps.
- exec_types     : (4,) -- integer tuple (Type 5 typically dropped by LOB_data)
                   Type 4 = "execute visible" = a visible limit order was filled.
                   We only count these as "trades" for label/feature computation.

Data cleaning: load_one_day() uses the full LOB_data cleaning pipeline
(trading halts, auctions, crossed prices, split executions, hidden orders)
followed by intraday time cutting -- matching the notebook's workflow.

Dependencies
~~~~~~~~~~~~
- LOB_processor.py         : TOB window builders, duration calculator
- SGU1.py                  : Realized-range labels, features, XGBoost training
- SGU2.py                  : Pseudo-mid trend labels, LSTM training
- lobster_preprocessing.py : LOBSTER -> simulator format converters
- Deep_RL_With_Signal_policy_factory.py : DRL policy wrapper
- MM_LOB_SIM.py            : Market-making backtester
"""

from __future__ import annotations

import os
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project module imports
# ---------------------------------------------------------------------------
# WHY: Each module below encapsulates one layer of the pipeline. This
# separation-of-concerns design means that changes to (say) the LSTM
# architecture in SGU2.py do not require touching the data loader or the
# backtester. Each module exposes a clean functional API.

# LOB_processor: window construction and duration computation over TOB moves.
#
# WHY we need this: Raw LOBSTER data has one row per LOB event (order
# submission, cancellation, execution, etc.). Most events do NOT change
# the best bid/ask. TOB-move windows aggregate events into meaningful
# "time bars" defined by actual changes at the top of the book, providing
# a natural sampling frequency for feature/label computation.
#
# - top_moves_windows:        builds windows on the full (concatenated) orderbook
#                              (DANGER: may cross day boundaries!)
# - top_moves_windows_by_day: builds windows PER DAY using DayID column,
#                              preventing overnight contamination
# - compute_window_durations:  measures wall-clock duration of each window in seconds
from LOB_processor import top_moves_windows, top_moves_windows_by_day, compute_window_durations

# lobster_preprocessing: converters from LOBSTER format to MM_LOB_SIM format.
#
# WHY: The backtester (MM_LOB_SIM) uses a grid-based price representation
# (prices indexed relative to a base_price), while LOBSTER uses absolute
# dollar prices. These converters handle the coordinate transformation:
#   LOBSTER absolute price -> grid index = (price - base_price) / tick_size + grid_center
from lobster_preprocessing import (
    convert_lobster_to_sim,
    convert_lobster_orderbook_to_sim,
)

# SGU-1 (Spread Gap Unit 1): realized-range labels + hand-crafted features.
#
# WHAT IT PREDICTS: The "realized price range" within each window -- essentially
# a measure of intra-window volatility / spread dynamics. This tells the
# market-maker "how much room is there to capture spread?"
#
# WHY XGBoost: Realized-range is a cross-sectional regression target with
# hand-crafted features (spread, volume imbalance, uptick ratio, VWAP slope).
# Tree-based models excel at this type of tabular regression with non-linear
# interactions and heterogeneous feature scales.
#
# - compute_labels_realized_range : per-window label y_i = f(buy_prices, sell_prices, ask, bid)
# - compute_features_SGU1         : spread, volume imbalance, upticks, VWAP, slope, etc.
# - train_SGU1                    : chronological XGBoost training with 64/16/20 split
# - _sgu1_meta_config / try_load / save : deterministic caching by config hash
from SGU1 import (
    compute_labels_realized_range, compute_features_SGU1, train_SGU1,
    _sgu1_meta_config, try_load_sgu1, save_sgu1
)

# SGU-2 (Spread Gap Unit 2): pseudo-mid trend labels + LSTM features.
#
# WHAT IT PREDICTS: The return of a "pseudo-mid" price (average of the
# highest buy aggressor price and lowest sell aggressor price within each
# window). This is a directional signal: "where is the price going?"
#
# WHY LSTM: Trend prediction requires capturing temporal dependencies across
# windows (momentum, mean-reversion patterns). An LSTM naturally handles
# sequential data with variable-length memory, making it well-suited for
# multi-step-ahead directional forecasting from windowed LOB features.
#
# - compute_labels_realized_trend : per-window label y_i = return of pseudo-mid price
# - compute_features_SGU2         : extended feature set (L_max lagged features)
# - train_sgu2_lstm_pipeline      : LSTM training with chronological 64/16/20 split
# - build_sgu2_multi_feature_sequences / build_sgu1_sequences : tensor builders
#     These convert 2D tabular features into 3D tensors of shape (N, T, F)
#     where N=samples, T=sequence_length, F=features -- the format LSTMs expect.
# - predict_numpy                 : inference helper for trained LSTM
import torch
from SGU2 import (
    compute_labels_realized_trend,
    compute_features_SGU2,
    train_sgu2_lstm_pipeline,
    build_sgu2_multi_feature_sequences,
    build_sgu1_sequences,
    predict_numpy,
    _sgu2_meta_config,
    try_load_sgu2,
    save_sgu2,
)

# IMPORTANT (Spyder/IPython): force-reload SGU2 so local edits are actually
# applied even when running in the same long-lived console.
import SGU2 as _SGU2_module
_SGU2_module = importlib.reload(_SGU2_module)

# Rebind SGU2 symbols to the freshly reloaded module.
compute_labels_realized_trend = _SGU2_module.compute_labels_realized_trend
compute_features_SGU2 = _SGU2_module.compute_features_SGU2
train_sgu2_lstm_pipeline = _SGU2_module.train_sgu2_lstm_pipeline
build_sgu2_multi_feature_sequences = _SGU2_module.build_sgu2_multi_feature_sequences
build_sgu1_sequences = _SGU2_module.build_sgu1_sequences
predict_numpy = _SGU2_module.predict_numpy
_sgu2_meta_config = _SGU2_module._sgu2_meta_config
try_load_sgu2 = _SGU2_module.try_load_sgu2
save_sgu2 = _SGU2_module.save_sgu2

# Guardrail: if this fails, you're still running an old SGU2 in memory.
if "cpu_safe_mode" not in inspect.signature(train_sgu2_lstm_pipeline).parameters:
    raise RuntimeError(
        "Outdated SGU2 module loaded in memory. Restart the console/kernel and run again."
    )

import xgboost as xgb

# LOB_data: full LOBSTER data loader with cleaning pipeline.
#
# WHY we need a dedicated loader: Raw LOBSTER files contain artifacts that
# would corrupt analysis if not removed:
#   - Trading halts (Type=7): prices freeze, volume becomes meaningless
#   - Opening/closing auctions: non-continuous-trading price discovery
#   - Crossed prices: ask < bid (data error or latency artifact)
#   - Split executions: one order filled across multiple price levels
#   - Hidden orders (Type=5): executions against hidden liquidity
# The LOB_data class handles ALL of these via load_and_clean_LOB_data(),
# then cut_before_and_after_LOB_data() removes the first/last N minutes
# of each session to avoid auction spillover effects.
from LOB_data import LOB_data as LOBDataLoader

# Deep_RL_With_Signal_policy_factory: wraps a DRL agent with SGU signals.
#
# WHY: The RL agent does not interact directly with SGU models. Instead,
# this factory class:
#   1. Pre-loads the full array of SGU predictions for a phase.
#   2. Normalizes them using statistics from a FIXED "ruler" dataset
#      (always the DRL-train set, to prevent information leakage).
#   3. Advances a cursor every delta_t TOB moves, feeding the next
#      (sgu1_signal, sgu2_signal) pair to the agent as state features.
# This decouples the SGU inference schedule from the message-level backtest loop.
from Deep_RL_With_Signal_policy_factory import Deep_RL_With_Signal_policy_factory


# ===========================================================================
# 0) CONFIGURATION -- All user-editable parameters live here
# ===========================================================================
# WHY ALL PARAMETERS ARE CENTRALIZED HERE:
# Having a single "control panel" at the top of the script ensures that:
#   (a) No magic numbers are buried deep in the code.
#   (b) Experiments are reproducible -- just save this config block.
#   (c) Cache invalidation is deterministic: a hash of these parameters
#       decides whether a cached model is still valid.

# ---- Artifact caching (to avoid retraining on every run) ----
# WHY CACHING MATTERS: Training XGBoost (SGU-1) takes minutes and the LSTM
# (SGU-2) can take 30+ minutes on CPU. Caching saves the trained model,
# scaler, and feature metadata to disk. A deterministic hash of the config
# parameters is used as the cache key, so changing ANY parameter (e.g.,
# delta_t, dates, tick_size) automatically invalidates the cache and
# triggers retraining. This prevents stale-model bugs.
ARTIFACTS_DIR = Path("./artifacts")

USE_SGU1_CACHE = True          # Try to load cached SGU1 model before training
FORCE_RETRAIN_SGU1 = False     # Force retrain even if cache exists

USE_SGU2_CACHE = True          # Try to load cached SGU2 model before training
FORCE_RETRAIN_SGU2 = False     # Force retrain even if cache exists

# ---- SGU-2 hyperparameters ----
# WHY THESE HYPERPARAMETERS EXIST:
# T_SEQ_SGU_2: The LSTM receives a sequence of T_SEQ consecutive windows as
#   input. A larger T_SEQ gives the LSTM more historical context (e.g., 10
#   windows ~ 50 TOB moves), but increases memory and training time. It also
#   means the first T_SEQ-1 windows of each split are "consumed" for context
#   and cannot produce predictions, slightly reducing the effective dataset size.
# L_MAX_SGU_2: Some SGU-2 features are lagged versions of base features
#   (e.g., spread_lag_1, spread_lag_2, ..., spread_lag_L_max). This controls
#   the maximum lag depth, adding autoregressive information to each timestep.
T_SEQ_SGU_2 = 10               # LSTM sequence length (number of past windows per sample)
L_MAX_SGU_2 = 20               # Maximum feature lag depth for SGU2
USE_LOG_RET_SGU_2 = True       # If True, labels are log-returns; if False, simple returns
                                # WHY log-returns: log(p_t/p_{t-1}) is additive over time,
                                # more symmetric, and better behaved for gradient-based
                                # optimization than simple returns (p_t - p_{t-1})/p_{t-1}.
ROUND_PSEUDO_MID_TO_TICK_SGU_2 = False  # Round pseudo-mid to nearest tick before computing returns
                                # WHY you might enable this: if the pseudo-mid is computed
                                # from execution prices that are always on-tick, rounding
                                # removes floating-point noise. Disabled by default because
                                # the pseudo-mid is an average and rounding introduces
                                # discretization artifacts.
USE_SGU1_ONLY_SGU_2 = False    # If True, train SGU2 LSTM using only SGU1 features (ablation)
                                # WHY this exists: ablation testing. If SGU2-specific features
                                # (lagged returns, trend indicators) don't improve over SGU1
                                # features alone, the extra complexity is not justified.
SGU2_LHS_SAMPLES = 50           # Number of LHS trials for SGU2 HP search; reduce (e.g., 10-20) for faster debugging
SGU2_EARLY_STOP_PATIENCE = 8    # Early-stopping patience (epochs without val improvement) per LHS trial
SGU2_MIN_REFIT_EPOCHS = 5       # Minimum epochs for final train+val refit
SGU2_CPU_SAFE_MODE = True       # Stabilize CPU LSTM training (disable MKLDNN + limit threads)
SGU2_CPU_THREADS = 1            # Threads used when CPU safe mode is enabled (1 = safest for deadlock issues)

# ---- Trading dates to process ----
# WHY MULTIPLE DAYS: A single day of LOBSTER data has ~500K-2M events.
# Using 8 days gives us enough data for meaningful train/val/test splits
# across both SGU and DRL stages. The dates must be business days (markets
# are closed on weekends/holidays). Note the gap between 08-01 (Fri) and
# 08-04 (Mon) -- this is a weekend, and between 08-08 (Fri) and 08-11 (Mon).
# Day-safe windowing (Section 3) ensures no window ever spans these gaps.
BASE_DATES = [
    "2025-08-01",
    "2025-08-04",
    "2025-08-05",
    #"2025-08-06",
    #"2025-08-07",
    #"2025-08-08",
    #"2025-08-11",
    #"2025-08-12",
]

# ---- Symbol and data path ----
# Each LOBSTER subscription produces files in the format:
#   {SYMBOL}_{DATE}_{START}_{END}_message_{DEPTH}.csv
#   {SYMBOL}_{DATE}_{START}_{END}_orderbook_{DEPTH}.csv
# where START/END are nanoseconds-since-midnight (34200000000 = 09:30:00).
SYMBOL = "AMZN"
DATA_PATH = Path("./Data/AMZN/")

# ---- LOBSTER price scaling ----
# LOBSTER stores prices as integers in units of 1/10000 of a dollar.
# Dividing by 10000 converts to dollars (e.g., 1855000 -> $185.50).
# WHY integers: LOBSTER avoids floating-point by storing prices as
# integer multiples of $0.0001, ensuring exact arithmetic in the data files.
# The LOB_data cleaning pipeline handles this conversion internally.
PRICE_DIVISOR = 10000.0

# ---- LOB_data cleaning parameters ----
# These control the LOB_data.load_and_clean_LOB_data() + cut_before_and_after_LOB_data()
# pipeline, which removes trading halts, auctions, crossed prices, split executions,
# and hidden orders -- matching the notebook's cleaning workflow.
#
# WHY CUT THE FIRST AND LAST 30 MINUTES:
# The US equity market opens at 09:30 and closes at 16:00 (Eastern).
# The first ~30 minutes exhibit abnormal patterns: opening auction effects,
# overnight information being digested, and wider-than-normal spreads.
# The last ~30 minutes exhibit closing-auction anticipation effects.
# Cutting 30 minutes from each end keeps only the "steady-state" continuous
# trading session (10:00 - 15:30), which is more representative for
# market-making strategy development.
MINUTES_CUT_BEGIN = 30             # Cut first 30 min of each day (09:30+30 = 10:00)
MINUTES_CUT_END   = 30             # Cut last 30 min of each day (16:00-30 = 15:30)
FLAG_DROP_HIDDEN_ORDERS = True     # Drop hidden order executions (Type 5)
                                   # WHY: Hidden orders are not visible in the LOB state.
                                   # Including their executions in labels/features would
                                   # create features based on information that a real-time
                                   # market participant could not observe.
LOB_DEPTH = 5                      # Number of orderbook levels in LOBSTER files
                                   # (5 levels = 5 ask + 5 bid price/size pairs = 20 columns)

# ---- Label parameters ----
TICK_SIZE = 0.01                   # Minimum price increment ($0.01 for US equities)
                                   # WHY this matters: tick_size determines the minimum
                                   # spread a market-maker can earn (1 tick = $0.01).
                                   # It also controls grid resolution in the backtester.

# ---- TOB window definition ----
# Each window spans exactly DELTA_T consecutive top-of-book (TOB) moves.
# A TOB move = any change in best bid/ask price OR volume at level 1.
#
# WHY DELTA_T = 5:
# This is a design choice balancing two concerns:
#   - Too small (1-2): windows are too noisy; most contain no trades.
#   - Too large (50+): windows span too much time; we lose granularity.
# DELTA_T = 5 means each window captures 5 consecutive changes at the
# best bid/ask, which typically spans ~0.1-2 seconds for liquid stocks
# like AMZN. This gives enough trades per window for meaningful labels
# while keeping windows short enough for responsive signals.
DELTA_T = 25

# ---- Execution type filter ----
# Which LOBSTER message Type integers count as trades for SGU label/feature computation.
# Type 4 = execute_visible.  Type 5 = execute_hidden (usually dropped by LOB_data).
#
# WHY ONLY TYPE 4: Type 5 (hidden executions) are already dropped by the
# cleaning pipeline (FLAG_DROP_HIDDEN_ORDERS=True). Even if they were
# present, we would not want to include them in label computation because
# they represent fills against invisible liquidity -- a market-maker
# cannot observe or react to hidden orders in real time.
EXEC_TYPES = (4,)

# ---- Time-of-day boundaries (in seconds after midnight) ----
# These define the trading window after applying MINUTES_CUT_BEGIN/END.
# Market opens at 09:30:00 (34200 sec) and closes at 16:00:00 (57600 sec).
# After cutting 30 min from each end:
#   Start: 34200 + 30*60 = 36000 (10:00:00)
#   End:   57600 - 30*60 = 55800 (15:30:00)
# These are used in the SGU-2 cache key to ensure that cached results
# are invalidated if the time window changes.
DAY_START_SEC = 34200 + MINUTES_CUT_BEGIN * 60
DAY_END_SEC   = 57600 - MINUTES_CUT_END * 60


# ===========================================================================
# 1) HELPER FUNCTIONS
# ===========================================================================
# These utility functions handle the interface between LOBSTER's raw file
# format and the internal representations used by the pipeline. They are
# intentionally stateless (no globals modified) to keep the pipeline
# deterministic and testable.

def infer_lob_depth_from_filename(orderbook_file: str) -> int:
    """
    Extract the LOB depth from a LOBSTER orderbook filename.

    LOBSTER filenames encode the depth as the last token before ".csv":
        "AMZN_2025-08-01_34200000_57600000_orderbook_5.csv"  ->  depth = 5

    Parameters
    ----------
    orderbook_file : str
        Filename (not full path) of the LOBSTER orderbook CSV.

    Returns
    -------
    int
        Number of price levels on each side (e.g. 5 = 5 ask + 5 bid levels).
    """
    depth_str = orderbook_file.rsplit("_", 1)[-1].split(".")[0]
    return int(depth_str)


def build_orderbook_columns(lob_depth: int) -> List[str]:
    """
    Build CamelCase column names for a LOBSTER orderbook with `lob_depth` levels.

    LOBSTER orderbook CSVs have no header.  The columns are interleaved:
        [AskPrice_1, AskSize_1, BidPrice_1, BidSize_1,
         AskPrice_2, AskSize_2, BidPrice_2, BidSize_2, ...]

    Parameters
    ----------
    lob_depth : int
        Number of price levels (e.g. 5).

    Returns
    -------
    List[str]
        Column names, length = 4 * lob_depth.
    """
    cols: List[str] = []
    for level in range(1, lob_depth + 1):
        cols.extend([
            f"AskPrice_{level}", f"AskSize_{level}",
            f"BidPrice_{level}", f"BidSize_{level}",
        ])
    return cols


def load_one_day(base_date_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean a single LOBSTER trading day using the LOB_data pipeline.

    This performs the FULL cleaning workflow (matching the notebook):
      1. Load raw CSVs via LOB_data class.
      2. Clean: remove trading halts, auctions, crossed prices, merge split
         executions, handle hidden orders.
      3. Cut: remove first/last MINUTES_CUT_BEGIN/END minutes of each day.
      4. Rename "ID" -> "OrderID" for compatibility with lobster_preprocessing.
      5. Reset indices to 0..N-1 for clean multi-day concatenation.

    WHY THIS FUNCTION EXISTS:
    Each LOBSTER trading day arrives as two CSV files (messages + orderbook).
    Before we can use this data, we must apply a standardized cleaning pipeline
    to remove artifacts that would corrupt our labels and features. By
    encapsulating this in a function, we guarantee that every day is processed
    identically -- no day gets special treatment, and the cleaning is
    consistent with the exploratory notebook's workflow.

    IMPORTANT DESIGN DECISION -- WHY WE COPY:
    We call .copy() on the output DataFrames to detach them from the LOB_data
    object's internal state. Without .copy(), subsequent calls to load_one_day()
    would overwrite the LOB_data object's internal buffers, causing subtle
    aliasing bugs during multi-day concatenation.

    Parameters
    ----------
    base_date_str : str
        Trading date in "YYYY-MM-DD" format (e.g. "2025-08-01").

    Returns
    -------
    (messages, orderbook) : Tuple[pd.DataFrame, pd.DataFrame]
        Both DataFrames are reset to index 0..N-1 and row-aligned.
        Messages columns: Time, Type, OrderID, Size, Price, Direction
        (DayID and TimeAbs are added later during concatenation.)
    """
    print(f"Processing {base_date_str}...")

    # --- Construct filenames following LOBSTER naming convention ---
    # LOBSTER filenames encode: symbol, date, start_time (ns), end_time (ns), type, depth.
    # 34200000 ms = 34200 seconds = 09:30:00 (market open)
    # 57600000 ms = 57600 seconds = 16:00:00 (market close)
    msg_label = f"{SYMBOL}_{base_date_str}_34200000_57600000_message_{LOB_DEPTH}.csv"
    ob_label  = f"{SYMBOL}_{base_date_str}_34200000_57600000_orderbook_{LOB_DEPTH}.csv"

    # --- Load + clean using LOB_data pipeline ---
    # The LOB_data class constructor takes the directory, message filename, orderbook
    # filename, and tick_size. It does NOT load data yet -- that happens in
    # load_and_clean_LOB_data().
    lob = LOBDataLoader(str(DATA_PATH), msg_label, ob_label, tick_size=TICK_SIZE)
    # load_and_clean_LOB_data() performs the full cleaning pipeline in sequence:
    #   1. Read CSVs into DataFrames
    #   2. Identify and remove trading halts (Type=7 and surrounding messages)
    #   3. Remove opening/closing auction messages
    #   4. Detect and fix crossed-price situations (ask < bid)
    #   5. Merge split executions (one order filled at multiple levels)
    #   6. Optionally drop hidden order executions (Type=5)
    # message_file_with_extra_column=True: LOBSTER files sometimes include an
    # extra trailing column (e.g., MPID). This flag tells the loader to expect it.
    lob.load_and_clean_LOB_data(
        flag_drop_hidden_orders=FLAG_DROP_HIDDEN_ORDERS,
        verbose=False,
        message_file_with_extra_column=True,
    )

    # --- Cut beginning/end of trading session ---
    # WHY: The opening (09:30-10:00) and closing (15:30-16:00) periods are
    # structurally different from mid-day continuous trading. Spreads are wider,
    # volume patterns reflect institutional order flow (e.g., VWAP algorithms
    # that execute near open/close), and price discovery is still settling.
    # Keeping this data would add noise to our features and labels without
    # adding representative training signal for market-making.
    lob.cut_before_and_after_LOB_data(MINUTES_CUT_BEGIN, MINUTES_CUT_END, verbose=False)

    # .copy() detaches from the LOB_data object's internal buffers to prevent
    # aliasing when we call load_one_day() again for the next trading day.
    messages  = lob.message_file.copy()
    orderbook = lob.ob_file.copy()

    # --- Rename "ID" -> "OrderID" for compatibility with lobster_preprocessing ---
    # WHY: LOB_data internally uses "ID" as the column name for order identifiers.
    # However, lobster_preprocessing.py (which converts to backtester format)
    # expects "OrderID". This rename bridges the two conventions. We only rename
    # if "ID" exists and "OrderID" does not, to be idempotent (safe to call twice).
    if "ID" in messages.columns and "OrderID" not in messages.columns:
        messages = messages.rename(columns={"ID": "OrderID"})

    # --- Reset indices to 0..N-1 for clean multi-day concatenation ---
    # WHY: After cleaning + cutting, the DataFrame indices have gaps (e.g.,
    # rows 0, 1, 5, 8, 12... where removed rows left holes). Resetting to
    # 0..N-1 ensures that when we pd.concat() multiple days with
    # ignore_index=True, the resulting DataFrame has a clean contiguous index
    # and that messages_all.iloc[i] corresponds to orderbook_all.iloc[i]
    # (row-alignment between the two DataFrames is preserved).
    messages  = messages.reset_index(drop=True)
    orderbook = orderbook.reset_index(drop=True)

    return messages, orderbook


def durations_histogram(durations_sec: pd.Series) -> None:
    """
    Plot a histogram of window durations, clipped at the 99th percentile.

    This helps visualize the distribution of how long (in seconds) each
    TOB-move window lasts, excluding extreme outliers.

    WHY THIS DIAGNOSTIC IS IMPORTANT:
    Window duration is a proxy for "how active was the market during this
    window?" Short durations (< 0.1s) mean rapid TOB changes (high activity).
    Long durations (> 10s) mean the book was relatively stable. Extremely
    long durations might indicate:
      - A window spanning a trading halt (should have been removed).
      - A cross-day window (should not exist if day-safe windowing worked).
      - A period of very low liquidity.
    Clipping at the 99th percentile focuses the histogram on the bulk of
    the distribution and prevents extreme outliers from compressing the x-axis.

    Parameters
    ----------
    durations_sec : pd.Series
        The "duration_sec" column from compute_window_durations().
    """
    s = durations_sec.dropna()

    # Handle edge cases: if durations are stored as datetime/timedelta, convert.
    # WHY: Depending on how compute_window_durations() is configured, it might
    # return durations as timedelta objects instead of plain floats. This
    # defensive handling ensures the histogram works regardless.
    if np.issubdtype(s.dtype, np.datetime64):
        dur = (s - s.min()).dt.total_seconds()
    elif np.issubdtype(s.dtype, np.timedelta64):
        dur = s.dt.total_seconds()
    else:
        dur = pd.to_numeric(s, errors="coerce")

    dur = dur.dropna().to_numpy()

    if dur.size == 0:
        return

    # Clip at 99th percentile to remove extreme outliers from the plot.
    # WHY 99th and not, say, 95th: We want to see the "long tail" behavior
    # but not let a handful of extreme values (e.g., 100+ seconds from a
    # nearly-halted period) squash the histogram into a single bar.
    p99 = np.percentile(dur, 99)
    dur_clipped = dur[dur <= p99]

    plt.figure(figsize=(10, 5))
    plt.hist(dur_clipped, bins=60, edgecolor="black", alpha=0.7)
    plt.xlabel(f"Duration (≤ 99th perc = {p99:,.2f})")
    plt.ylabel("Frequency")
    plt.title("Histogram of TOB-window durations -- clipped at 99th percentile")
    plt.grid(True, alpha=0.3)
    plt.show()


def build_backtest_inputs(
    messages_df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    tick_size: float,
    grid_center: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert LOBSTER-format data to MM_LOB_SIM backtest format.

    The backtester (MM_LOB_SIM) expects a specific grid-based representation
    produced by lobster_preprocessing.  This function computes the base_price
    from the first row's mid-price and delegates to the converters.

    WHY GRID-BASED REPRESENTATION:
    The MM_LOB_SIM backtester uses a fixed-size price grid (array of size ~100)
    where each cell represents one tick level. Prices are stored as INTEGER
    INDICES relative to a base_price, not as absolute dollar values. This makes
    the simulator code faster (array indexing instead of dictionary lookups)
    and normalizes the price representation across different stocks/dates.

    The mapping is:
        grid_index = round((absolute_price - base_price) / tick_size) + grid_center

    So if base_price = $185.50, tick_size = $0.01, grid_center = 50:
        $185.50 -> index 50
        $185.51 -> index 51
        $185.49 -> index 49

    WHY base_price FROM THE FIRST ROW'S MID:
    We anchor the grid to the initial mid-price so that the grid center (50)
    starts at the current fair value. This ensures both sides of the book
    (bid and ask) fit within the grid. If price drifts significantly during
    the backtest period, the grid might overflow -- but for typical intraday
    periods (a few hours of one stock), this is not a concern.

    Parameters
    ----------
    messages_df : pd.DataFrame
        Messages in CamelCase LOBSTER format (Time, Type, Price, Direction, Size, ...).
    orderbook_df : pd.DataFrame
        Orderbook in CamelCase format (AskPrice_1, AskSize_1, BidPrice_1, BidSize_1, ...).
    tick_size : float
        Minimum price increment (e.g. 0.01 for US equities).
    grid_center : int
        Center index of the simulator price grid (default 50).
        WHY 50: This gives ~50 ticks of room on each side ($0.50 for a $0.01 tick),
        which is usually sufficient for intraday price movements of liquid stocks.

    Returns
    -------
    (messages_bt, orderbook_bt) : Tuple[pd.DataFrame, pd.DataFrame]
        Data in MM_LOB_SIM format, ready for backtest_LOB_with_MM().
    """
    ask1_col = "AskPrice_1"
    bid1_col = "BidPrice_1"
    if ask1_col not in orderbook_df.columns or bid1_col not in orderbook_df.columns:
        raise ValueError("orderbook_df must contain AskPrice_1 / BidPrice_1 for backtest conversion.")

    # Compute base_price as the mid-price of the first row.
    # WHY mid-price: The mid-price = (best_ask + best_bid) / 2 is the natural
    # center of the order book and the most neutral anchor point. Using just
    # the ask or bid would bias the grid toward one side.
    ask1_0 = pd.to_numeric(orderbook_df[ask1_col], errors="coerce").iloc[0]
    bid1_0 = pd.to_numeric(orderbook_df[bid1_col], errors="coerce").iloc[0]
    base_price = float((ask1_0 + bid1_0) / 2.0)
    if not np.isfinite(base_price):
        raise ValueError(
            f"base_price is not finite ({base_price}). First-row ask={ask1_0}, "
            f"bid={bid1_0}. Check that orderbook_df has valid prices in row 0."
        )

    # Detect number of ask price levels to determine n_levels_to_store.
    # WHY: The converter needs to know how many price levels to process
    # from the LOBSTER orderbook. We infer this from the number of
    # "AskPrice_*" columns present, rather than hardcoding it.
    ask_level_cols = [c for c in orderbook_df.columns if str(c).startswith("AskPrice_")]
    n_levels_to_store = max(1, len(ask_level_cols))

    # Convert the orderbook snapshot (one row per event) to grid format.
    # Each row becomes a fixed-size array indexed by price grid position.
    orderbook_bt = convert_lobster_orderbook_to_sim(
        orderbook_df,
        tick_size=tick_size,
        grid_center=grid_center,
        base_price=base_price,
        n_levels_to_store=n_levels_to_store,
    )
    # Convert messages (order submissions, cancellations, executions) to
    # grid-relative prices. The simulator processes these sequentially to
    # reconstruct LOB dynamics and simulate order matching.
    messages_bt = convert_lobster_to_sim(
        messages_df,
        orderbook_df,
        tick_size=tick_size,
        grid_center=grid_center,
        base_price=base_price,
    )
    return messages_bt, orderbook_bt


def plot_true_vs_pred_series(y_true, y_pred, label, N=500):
    """
    Plot the first N observations of true vs predicted values as a time series.

    Useful for visually assessing whether the model captures regime changes
    and relative magnitude of the target variable.

    WHY A TIME-SERIES PLOT (not scatter):
    For chronologically ordered data, a time-series overlay reveals patterns
    that a scatter plot hides: Does the model lag behind regime changes? Does
    it over-smooth? Does it capture the right direction even if magnitude is
    off? These are critical for market-making signals where "direction" matters
    more than exact value prediction.

    WHY ONLY THE FIRST N=500 OBSERVATIONS:
    Plotting tens of thousands of points creates visual clutter. The first 500
    observations (from the beginning of each split) give a representative view
    while keeping the plot readable. For train splits, this shows early-period
    behavior; for test splits, it shows the first unseen data.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions (same length as y_true).
    label : str
        Plot title / description.
    N : int
        Number of leading observations to display (default 500).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    k = min(N, n)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:k], label="True")
    plt.plot(y_pred[:k], label="Predicted")
    plt.xlabel("Index (time order)")
    plt.ylabel("Target")
    plt.title(f"True vs Predicted (first {k} obs) - {label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # ===========================================================================
    # 2) DATA LOADING & CONCATENATION
    # ===========================================================================
    # Load each trading day, assign a sequential DayID, and concatenate.
    # DayID is essential for:
    #   - top_moves_windows_by_day(): prevents windows from crossing overnight gaps
    #   - TimeAbs computation: ensures monotonically increasing time across days
    #
    # WHY MULTI-DAY CONCATENATION IS NON-TRIVIAL:
    # Each LOBSTER day has independent row indexing (0..N_day) and timestamps
    # that reset at midnight (Time = seconds-since-midnight). Simply concatenating
    # would create duplicate indices, non-monotonic timestamps, and -- most
    # dangerously -- windows that span overnight gaps (e.g., 15:29:59 on Day 1
    # to 10:00:01 on Day 2). The DayID column solves all three problems:
    #   1. pd.concat(ignore_index=True) creates a clean global index.
    #   2. TimeAbs = Time + 86400 * DayID creates monotonically increasing time.
    #   3. top_moves_windows_by_day() uses DayID to build windows per-day only.
    #
    # DATA STRUCTURES AFTER THIS SECTION:
    #   messages_all  : one row per LOB event across all days
    #     Columns: Time, Type, OrderID, Size, Price, Direction, DayID, TimeAbs
    #   orderbook_all : one row per LOB event (same index as messages_all)
    #     Columns: AskPrice_1, AskSize_1, BidPrice_1, BidSize_1, ..., AskPrice_5, ...
    #   CRITICAL INVARIANT: messages_all.iloc[i] and orderbook_all.iloc[i]
    #   correspond to the SAME LOB event. They share the same row index.

    all_messages: List[pd.DataFrame] = []
    all_orderbooks: List[pd.DataFrame] = []

    for day_idx, d in enumerate(BASE_DATES):
        messages, orderbook = load_one_day(d)
        # DayID = 0, 1, 2, ... following the order in BASE_DATES.
        # This MUST be set before concatenation so that day-safe windows work correctly.
        # WHY integer (not date string): Integer DayIDs make arithmetic easy
        # (TimeAbs = Time + 86400 * DayID) and are naturally sortable.
        messages["DayID"] = day_idx
        all_messages.append(messages)
        all_orderbooks.append(orderbook)

    # Concatenate all days into a single DataFrame with a clean 0..N-1 index.
    # ignore_index=True ensures no duplicate indices from individual days.
    messages_all = pd.concat(all_messages, ignore_index=True)
    orderbook_all = pd.concat(all_orderbooks, ignore_index=True)

    # TimeAbs: absolute monotone time = Time (seconds since midnight) + 86400 * DayID.
    # This ensures that timestamps always increase, even across day boundaries.
    # Used by: compute_window_durations(), compute_features_SGU1() (prev_slope), etc.
    #
    # EXAMPLE:
    #   Day 0 (2025-08-01): Time=36000.5 (10:00 AM) -> TimeAbs = 36000.5 + 0 = 36000.5
    #   Day 1 (2025-08-04): Time=36000.5 (10:00 AM) -> TimeAbs = 36000.5 + 86400 = 122400.5
    #   The 86400 offset (seconds per day) guarantees strict monotonicity even
    #   though the raw Time resets each day. Note: weekends/holidays create jumps
    #   in DayID (e.g., Friday DayID=0, Monday DayID=1), but this is fine because
    #   TimeAbs only needs to be monotonically increasing, not uniformly spaced.
    messages_all["TimeAbs"] = messages_all["Time"] + 86400.0 * messages_all["DayID"].astype(float)

    print("\nFiles concatenated successfully!")
    print(f"Total Number of LOB events: {len(messages_all):,}")

    # --- Sanity checks ---
    # WHY: Quick visual inspection to verify that:
    #   (a) Prices are in the expected range (not raw LOBSTER integers)
    #   (b) DayID and TimeAbs look correct for the last day
    #   (c) The Type column contains only expected values (1, 2, 3, 4 -- no 5 or 7
    #       if hidden orders and halts were properly removed)
    print("\n--- orderbook_all.tail(2) ---")
    print(orderbook_all.tail(2))

    print("\n--- messages_all.tail(2) ---")
    print(messages_all.tail(2))

    print("\n--- messages_all['Type'].unique() ---")
    print(messages_all["Type"].unique())


    # ===========================================================================
    # 3) TOB WINDOW CONSTRUCTION (DAY-SAFE)
    # ===========================================================================
    # A "TOB move" = any change in best ask/bid price OR volume at level 1.
    # Each window groups exactly `delta_t` consecutive TOB moves.
    #
    # Day-safe: windows are built independently per trading day using DayID,
    # so no window ever spans an overnight boundary.  This is critical because
    # LOBSTER Time resets at midnight -- a cross-day window would have nonsensical
    # durations and mix two independent trading sessions.
    #
    # DETAILED EXPLANATION OF "DAY-SAFE" WINDOWING:
    # Consider two consecutive trading days:
    #   Day 0 ends at message index 500,000 (15:30:00 after cutting)
    #   Day 1 starts at message index 500,001 (10:00:00 after cutting)
    #
    # WITHOUT day-safe windowing (top_moves_windows on full concatenated data):
    #   A window might start at index 499,998 (Day 0, 15:29:58) and end at
    #   index 500,004 (Day 1, 10:00:02). This window would:
    #     - Span an overnight gap of ~18.5 hours
    #     - Have a nonsensical "duration" (TimeAbs difference) of ~18.5 hours
    #     - Mix end-of-day and beginning-of-day dynamics (completely different regimes)
    #     - Produce corrupt labels (e.g., realized range spanning two sessions)
    #
    # WITH day-safe windowing (top_moves_windows_by_day):
    #   Windows are constructed independently for each DayID. If the last few
    #   TOB moves of Day 0 are not enough to fill a complete window (< delta_t),
    #   they are simply discarded. The next window starts fresh on Day 1.
    #   This "wastes" a few data points at day boundaries (typically < delta_t
    #   per day, which is negligible) but guarantees data integrity.
    #
    # WHAT THE `windows` LIST CONTAINS:
    #   windows = [(start_msg_idx_0, end_msg_idx_0),
    #              (start_msg_idx_1, end_msg_idx_1),
    #              ...]
    #   Each tuple gives absolute message indices (into messages_all / orderbook_all)
    #   defining the span of one window. The windows are non-overlapping and
    #   chronologically ordered. They cover the entire multi-day dataset except
    #   for partial windows at day boundaries.

    delta_t = int(DELTA_T)
    windows = top_moves_windows_by_day(
        orderbook_all, delta_t=delta_t, day_ids=messages_all["DayID"]
    )

    print(f"\n#windows (delta_t={delta_t} TOB moves/window, day-safe): {len(windows):,}")

    # ---- Compute per-window DayID for cross-day masking ----
    # Each window is (start, end). The DayID of a window is determined by
    # the DayID of its first message (messages_all["DayID"].iloc[start]).
    # This is used by SGU-1 (to mask cross-day label lags) and SGU-2
    # (to mask cross-day inter-window returns).
    window_day_ids = np.array([
        int(messages_all["DayID"].iloc[s]) for s, e in windows
    ])
    print(f"Window DayIDs: {np.unique(window_day_ids)} ({len(window_day_ids)} windows)")


    # ===========================================================================
    # 4) WINDOW DURATION ANALYSIS
    # ===========================================================================
    # Compute how many seconds each window spans using TimeAbs (monotone time).
    # This helps verify that:
    #   (a) no window has negative duration (would indicate cross-day contamination)
    #   (b) the duration distribution looks reasonable for the chosen delta_t
    #
    # WHY DURATION ANALYSIS IS A CRITICAL DIAGNOSTIC:
    # The duration of a window (in wall-clock seconds) reveals the "speed" of
    # market activity during that window. For a liquid stock like AMZN with
    # delta_t=5 TOB moves, typical durations are:
    #   - 0.01 - 0.5 seconds: normal high-frequency activity
    #   - 0.5 - 5 seconds: moderate activity
    #   - 5+ seconds: very quiet period (could indicate low-liquidity regime)
    #   - 86400+ seconds: CROSS-DAY BUG (window spans an overnight gap)
    #   - Negative duration: DATA CORRUPTION (timestamps out of order)
    #
    # We use TimeAbs (not raw Time) because raw Time resets at midnight.
    # Using raw Time for a cross-day window would give nonsensical results
    # (e.g., Time_end=36000 on Day 1 minus Time_start=55800 on Day 0 = -19800).

    durations = compute_window_durations(messages_all, windows, time_col="TimeAbs")
    dur = durations["duration_sec"]

    # Report the 95th percentile and trimmed mean (excluding top 5% outliers).
    # WHY P95 instead of max: The maximum duration is often an extreme outlier
    # (perhaps the last window before market close on a quiet day). The P95
    # gives a more robust estimate of "typical long duration".
    # The trimmed mean (mean conditional on <= P95) provides a single summary
    # statistic for expected window duration.
    p95 = dur.quantile(0.95)
    mean_p95 = dur[dur <= p95].mean()

    print("\n--- Window duration stats ---")
    print(f"p95 duration_sec: {p95:.6f}")
    print(f"mean(duration_sec | <= p95): {mean_p95:.6f}")

    durations_histogram(durations["duration_sec"])


    # ===========================================================================
    # 5) SGU-1: LABELS (realized range) + FEATURES
    # ===========================================================================
    # SGU-1 labels measure the "realized price range" within each window:
    #   y_i = max(buy_aggressor_prices) - min(sell_aggressor_prices)
    # with fallbacks to avg_ask / avg_bid when one side has no trades.
    #
    # WHY "REALIZED RANGE" AS A LABEL:
    # The realized range captures the effective bid-ask spread that was actually
    # TRADED during a window. For a market-maker, this is directly related to
    # profitability: a wider realized range means more potential profit per
    # round-trip (buy at the low, sell at the high). The label is:
    #   y_i = max(buy aggressor prices) - min(sell aggressor prices)
    #         within window i
    # where "buy aggressor" = Direction == -1 (the buyer consumed a passive sell LO)
    #       "sell aggressor" = Direction == +1 (the seller consumed a passive buy LO)
    # When one side has no trades in a window (e.g., no sell aggressor), the
    # function falls back to the average ask (or bid) price, which is the best
    # available proxy for "what the other side of the trade would have cost."
    #
    # WHY WE PASS `windows` EXPLICITLY:
    # We pass the pre-computed day-safe `windows` so that labels use exactly
    # the same window definitions as features and durations.  Without this,
    # the function would rebuild windows internally via top_moves_windows()
    # on the full concatenated orderbook, re-introducing cross-day contamination.
    # This is a form of the DRY principle: compute windows ONCE, use everywhere.

    labels_SGU_1_df = compute_labels_realized_range(
        orderbook=orderbook_all,
        messages=messages_all,
        delta_t=delta_t,
        tick_size=TICK_SIZE,
        exec_types=EXEC_TYPES,
        windows=windows,
    )

    print("\n--- labels_SGU_1_df.tail() ---")
    print(labels_SGU_1_df.tail())

    # SGU-1 features: spread, volume imbalance, uptick %, trade counts,
    # rolling VWAP, mid-price slope, time-of-day, lagged labels, etc.
    #
    # WHY THESE SPECIFIC FEATURES:
    # Each feature captures a different aspect of LOB microstructure:
    #   - Spread: direct measure of illiquidity (wider = less liquid)
    #   - Volume imbalance: ratio of bid vs ask volume at level 1 (predicts direction)
    #   - Uptick %: fraction of trades at higher prices (momentum indicator)
    #   - Trade count: activity level (more trades = more information)
    #   - Rolling VWAP: volume-weighted average price (fair value estimator)
    #   - Mid-price slope: recent price trend (linear regression of mid-price)
    #   - Time-of-day: U-shaped intraday patterns in spread/volume (market microstructure)
    #   - Lagged labels: autoregressive signal (past realized range predicts future)
    #
    # The features are computed WITHIN each window, using messages and orderbook
    # data from the window's start to end indices. No future information leaks.
    #
    # To prevent data leakage, we first compute the large-trade threshold
    # (90th percentile of trade sizes) using ONLY trades from the training
    # split (first 64% of windows), then pass it to compute_features_SGU1
    # so that val/test features use the same threshold without seeing
    # future data.
    _frac_train = 0.64
    _n_train_windows = int(len(windows) * _frac_train)
    _exec_mask = messages_all["Type"].isin(EXEC_TYPES).to_numpy()
    _sizes = messages_all["Size"].to_numpy(dtype=float)
    _train_trade_sizes = []
    for _s, _e in windows[:_n_train_windows]:
        _win_exec = _exec_mask[_s:_e]
        _win_sizes = _sizes[_s:_e][_win_exec]
        _train_trade_sizes.append(_win_sizes[np.isfinite(_win_sizes)])
    _train_trade_sizes = np.concatenate(_train_trade_sizes) if _train_trade_sizes else np.array([])
    large_threshold_train = float(np.nanquantile(_train_trade_sizes, 0.90)) if _train_trade_sizes.size else 0.0
    print(f"Large-trade threshold (90th pct, train-only): {large_threshold_train:.2f}")

    features_SGU_1 = compute_features_SGU1(
        orderbook_all,
        messages_all,
        windows,
        labels_SGU_1_df,
        exec_types=EXEC_TYPES,
        window_day_ids=window_day_ids,
        large_threshold_override=large_threshold_train,
    )

    print("\n--- features_SGU_1.tail() ---")
    print(features_SGU_1.tail())


    # ===========================================================================
    # 6) SGU-1: TRAIN (or load cached) XGBoost model
    # ===========================================================================
    # The training function uses chronological splitting (64% train / 16% val / 20% test)
    # to avoid look-ahead bias.  The final model is refit on 80% (train + val).
    #
    # WHY CHRONOLOGICAL SPLITTING (NOT RANDOM):
    # Financial time series exhibit regime changes (trending periods, mean-reverting
    # periods, high-volatility events). Random splitting would leak future regimes
    # into the training set, making the model appear more accurate than it truly is.
    # Chronological splitting ensures the model is ALWAYS evaluated on strictly
    # future data, mimicking real deployment conditions.
    #
    # WHY 64/16/20 SPLIT RATIOS:
    #   - 64% train: enough data for XGBoost to capture patterns
    #   - 16% validation: used for early stopping (prevents overfitting)
    #   - 20% test: held-out for unbiased out-of-sample evaluation
    #   - 64+16 = 80%: the final model is refit on this combined set to maximize
    #     training data while using the test set ONLY for evaluation.
    #
    # WHY REFIT ON 80%:
    # After hyperparameter selection / early stopping on the 64/16 split, we know
    # the optimal number of boosting rounds. We then retrain from scratch on 80%
    # (train+val) to use as much data as possible for the final model. The test 20%
    # is NEVER used for training -- it is exclusively for evaluation and becomes
    # the "DRL universe" downstream.
    #
    # Caching: a deterministic hash of the configuration (symbol, dates, delta_t,
    # exec_types, tick_size) is used to detect whether a compatible model already
    # exists on disk, avoiding redundant retraining.

    # Build a metadata dictionary that uniquely identifies this SGU-1 configuration.
    # The hash of this metadata becomes the cache key.
    meta_sgu1 = _sgu1_meta_config(
        symbol=SYMBOL,
        base_dates=BASE_DATES,
        delta_t=DELTA_T,
        exec_types=EXEC_TYPES,
        tick_size=TICK_SIZE,
    )

    # Attempt to load from cache first (unless forced to retrain).
    out = None
    if USE_SGU1_CACHE and (not FORCE_RETRAIN_SGU1):
        out = try_load_sgu1(meta_sgu1, ARTIFACTS_DIR)
        if out is not None:
            print(f"Loaded cached SGU1 (key={out['cache_key']})")

    # If no cache hit (or forced retrain), train from scratch and save.
    if out is None:
        out = train_SGU1(features_SGU_1, labels_SGU_1_df, n_stage1 = 100,
            n_stage2 = 100,)
        save_sgu1(out, meta_sgu1, ARTIFACTS_DIR,             
        overwrite=True,)  # safe: we just trained, always save the result

        print("Trained and saved SGU1 artifacts")

    # Extract trained artifacts for downstream use.
    # WHY WE NEED ALL THREE ARTIFACTS:
    #   - final_model_80: the actual XGBoost model (Booster object) used for prediction
    #   - scaler: StandardScaler fit ONLY on the training 64%. This is critical:
    #     applying it to val/test data ensures we normalize using only past statistics
    #     (no information leakage from the future). The scaler stores mean and std
    #     of each feature column, computed from training data only.
    #   - feature_cols: the ordered list of feature column names. XGBoost's DMatrix
    #     requires features in the same order as training. If columns are reordered
    #     or renamed, predictions would silently produce garbage.
    SGU1_final_model_80 = out["final_model_80"]   # XGBoost model refit on 80% (train+val)
    SGU1_scaler = out["scaler"]                    # StandardScaler fit on training set only
    SGU1_feature_cols = out["feature_cols"]         # Ordered list of feature column names

    # SGU-1 prediction coverage diagnostic (relative to raw day-safe windows).
    # This includes expected losses from X_t->y_{t+1} shift and NaN filtering.
    _sgu1_splits = out.get("splits", {})
    _n_pred_sgu1 = (
        len(_sgu1_splits.get("X_train", []))
        + len(_sgu1_splits.get("X_val", []))
        + len(_sgu1_splits.get("X_test", []))
    )
    _n_win_total = len(windows)
    _n_miss_sgu1 = max(0, _n_win_total - _n_pred_sgu1)
    _pct_miss_sgu1 = (100.0 * _n_miss_sgu1 / _n_win_total) if _n_win_total > 0 else float("nan")
    print(
        f"[SGU1 Coverage] no-pred windows: {_n_miss_sgu1:,}/{_n_win_total:,} "
        f"({_pct_miss_sgu1:.2f}%)"
    )


    # ===========================================================================
    # 7) SGU-1: EVALUATION -- Reconstruct splits & plot true vs predicted
    # ===========================================================================
    # To plot per-split results, we must reconstruct the exact dataset that
    # train_SGU1() used internally: merge features with labels, align X_t -> y_{t+1},
    # drop NaN rows, and apply the same chronological split ratios.
    #
    # WHY RECONSTRUCTION IS NECESSARY:
    # The train_SGU1() function returns only the trained model, scaler, and feature
    # column names -- it does NOT return the split datasets themselves (to save
    # memory). To generate predictions per split for plotting, we must rebuild
    # the exact same (X, y) arrays that training used. Any mismatch (wrong column
    # order, different NaN handling, different alignment) would produce incorrect
    # evaluations. This reconstruction follows the EXACT same steps as train_SGU1.
    #
    # THE KEY ALIGNMENT PRINCIPLE: X_t -> y_{t+1}
    # We use features at time t to predict the label at time t+1 (the NEXT window).
    # This is a one-step-ahead prediction setup:
    #   - X_t = features computed from window i (e.g., current spread, volume imbalance)
    #   - y_{t+1} = realized range of window i+1 (what we are trying to predict)
    # WHY: In live trading, we observe the current window's features and must
    # predict what will happen in the NEXT window to set our quotes accordingly.
    # Using X_t to predict y_t (same window) would be "explaining" not "forecasting."

    # Small epsilon for log-transform inversion: log(y + eps) -> exp(.) - eps.
    EPS = 1e-6

    # If labels were log-transformed before training, set this True to invert.
    # Current train_SGU1 uses raw labels, so keep False.
    # WHY log-transform might be useful: realized-range labels are strictly positive
    # and right-skewed. log(y) makes the distribution more Gaussian, which can
    # improve MSE-based training. If used, predictions must be exp()-inverted.
    LABEL_IS_LOG_SGU_1 = False

    # --- Reconstruct the merged dataset ---
    # Step 1: Inner-join features and labels on window_id.
    # WHY inner join: Some windows may have NaN labels (e.g., no trades occurred)
    # or NaN features (e.g., lag features at the very beginning). Inner join
    # keeps only windows that have BOTH valid features AND valid labels.
    _df_SGU_1 = (
        features_SGU_1
        .merge(labels_SGU_1_df[["window_id", "label"]], on="window_id", how="inner")
        .sort_values("window_id")   # Maintain chronological order
        .reset_index(drop=True)
    )

    # Separate features (everything except window_id and label) from label.
    all_feat_cols_SGU_1 = [c for c in _df_SGU_1.columns if c not in ["window_id", "label"]]
    X_full_SGU_1 = _df_SGU_1[all_feat_cols_SGU_1].copy()
    y_full_SGU_1 = _df_SGU_1["label"].copy()
    wid_full_SGU_1 = _df_SGU_1["window_id"].copy()

    # Step 2: Align X_t -> y_{t+1} (one-step-ahead prediction).
    # We remove the last row of X (no future label to predict) and shift y
    # backward by 1 (so y[i] becomes the label of window i+1).
    # CONCRETE EXAMPLE:
    #   Before alignment:  X = [X_0, X_1, X_2, X_3, X_4]
    #                      y = [y_0, y_1, y_2, y_3, y_4]
    #   After alignment:   X = [X_0, X_1, X_2, X_3]      (dropped last)
    #                      y = [y_1, y_2, y_3, y_4]      (shifted left by 1)
    #   Now X[i] predicts y[i+1], which is the NEXT window's realized range.
    X_SGU_1 = X_full_SGU_1.iloc[:-1].copy()
    y_SGU_1 = y_full_SGU_1.shift(-1).iloc[:-1].copy()
    wid_target_SGU_1 = wid_full_SGU_1.shift(-1).iloc[:-1].copy()

    # Step 3: Drop any rows with NaN features or labels (maintains strict time order).
    # WHY: Some features (e.g., lagged labels at lag=5) are NaN for the first 5
    # windows. Dropping these rows rather than imputing preserves data integrity.
    # reset_index ensures clean integer indexing for downstream slicing.
    mask_SGU_1 = (~X_SGU_1.isna().any(axis=1)) & (y_SGU_1.notna())
    X_SGU_1 = X_SGU_1.loc[mask_SGU_1].reset_index(drop=True)
    y_SGU_1 = y_SGU_1.loc[mask_SGU_1].reset_index(drop=True)
    wid_target_SGU_1 = wid_target_SGU_1.loc[mask_SGU_1].astype(int).reset_index(drop=True)

    # Enforce the exact column order that the scaler and model expect.
    # WHY THIS IS CRITICAL: XGBoost's DMatrix matches features by column NAME,
    # but the StandardScaler matches by POSITION (column index). If columns are
    # in a different order than during training, the scaler would subtract the
    # wrong mean and divide by the wrong std, silently corrupting all predictions.
    feature_cols_SGU_1 = list(SGU1_feature_cols)
    X_SGU_1 = X_SGU_1[feature_cols_SGU_1]

    # --- Chronological split: 64% train / 16% val / 20% test ---
    # These ratios MUST match what train_SGU1() used internally, otherwise the
    # splits would be misaligned and we would evaluate on the wrong data.
    frac_train_SGU_1 = 0.64
    frac_val_SGU_1 = 0.16

    n_total_SGU_1 = len(X_SGU_1)
    n_train_SGU_1 = int(n_total_SGU_1 * frac_train_SGU_1)
    n_val_SGU_1 = int(n_total_SGU_1 * frac_val_SGU_1)

    # Split into train / val / test using simple integer slicing (chronological).
    # No shuffling! The temporal order is sacred in financial time series.
    X_train_raw_SGU_1 = X_SGU_1.iloc[:n_train_SGU_1].astype(np.float32).reset_index(drop=True)
    y_train_SGU_1 = y_SGU_1.iloc[:n_train_SGU_1].astype(np.float32).reset_index(drop=True)
    wid_train_SGU_1 = wid_target_SGU_1.iloc[:n_train_SGU_1].to_numpy(dtype=int)

    X_val_raw_SGU_1 = X_SGU_1.iloc[n_train_SGU_1 : n_train_SGU_1 + n_val_SGU_1].astype(np.float32).reset_index(drop=True)
    y_val_SGU_1 = y_SGU_1.iloc[n_train_SGU_1 : n_train_SGU_1 + n_val_SGU_1].astype(np.float32).reset_index(drop=True)
    wid_val_SGU_1 = wid_target_SGU_1.iloc[n_train_SGU_1 : n_train_SGU_1 + n_val_SGU_1].to_numpy(dtype=int)

    X_test_raw_SGU_1 = X_SGU_1.iloc[n_train_SGU_1 + n_val_SGU_1 :].astype(np.float32).reset_index(drop=True)
    y_test_SGU_1 = y_SGU_1.iloc[n_train_SGU_1 + n_val_SGU_1 :].astype(np.float32).reset_index(drop=True)
    wid_test_SGU_1 = wid_target_SGU_1.iloc[n_train_SGU_1 + n_val_SGU_1 :].to_numpy(dtype=int)

    # --- Apply the trained scaler (fit on train split only, inside train_SGU1) ---
    # WHY THE SCALER IS FIT ON TRAIN ONLY:
    # StandardScaler computes mean and std per feature column. If we fit it on
    # the full dataset (including val/test), the mean/std would contain future
    # information -- a subtle but real form of data leakage. By fitting ONLY
    # on the training set, the scaler's statistics represent what a real-time
    # system would know at the training cutoff point. We then apply (transform)
    # those same statistics to val and test data.
    #
    # The scaler is EXACTLY the same object that train_SGU1() used, loaded from
    # the cache or returned directly from training. This guarantees consistency.
    scaler_SGU_1 = SGU1_scaler
    X_train_SGU_1 = pd.DataFrame(scaler_SGU_1.transform(X_train_raw_SGU_1), columns=feature_cols_SGU_1)
    X_val_SGU_1   = pd.DataFrame(scaler_SGU_1.transform(X_val_raw_SGU_1),   columns=feature_cols_SGU_1)
    X_test_SGU_1  = pd.DataFrame(scaler_SGU_1.transform(X_test_raw_SGU_1),  columns=feature_cols_SGU_1)

    # --- Generate predictions from the final_model_80 (XGBoost) ---
    # NOTE: We predict with the 80%-refit model on ALL three splits.
    # For the train split, this gives "in-sample" predictions (the model saw this data).
    # For the val split, the model also saw this data (it was refit on train+val).
    # For the test split, this is truly out-of-sample (the model never saw it).
    # The test predictions are what matters for evaluation and downstream DRL use.
    final_model_80_SGU_1 = SGU1_final_model_80

    # XGBoost requires data wrapped in DMatrix objects with matching feature_names.
    dtrain_split_SGU_1 = xgb.DMatrix(X_train_SGU_1, feature_names=feature_cols_SGU_1)
    y_train_pred_80_SGU_1 = final_model_80_SGU_1.predict(dtrain_split_SGU_1)

    dval_split_SGU_1 = xgb.DMatrix(X_val_SGU_1, feature_names=feature_cols_SGU_1)
    y_val_pred_80_SGU_1 = final_model_80_SGU_1.predict(dval_split_SGU_1)

    dtest_split_SGU_1 = xgb.DMatrix(X_test_SGU_1, feature_names=feature_cols_SGU_1)
    y_test_pred_80_SGU_1 = final_model_80_SGU_1.predict(dtest_split_SGU_1)

    # --- Convert from log-space to original scale if needed ---
    # WHY: If labels were log-transformed as log(y + eps) before training,
    # predictions are in log-space and must be inverted via exp(.) - eps.
    # EPS prevents log(0) during training and is subtracted here for consistency.
    if LABEL_IS_LOG_SGU_1:
        y_train_orig_SGU_1 = np.exp(y_train_SGU_1.to_numpy()) - EPS
        y_val_orig_SGU_1   = np.exp(y_val_SGU_1.to_numpy())   - EPS
        y_test_orig_SGU_1  = np.exp(y_test_SGU_1.to_numpy())  - EPS
        y_train_pred_SGU_1 = np.exp(y_train_pred_80_SGU_1) - EPS
        y_val_pred_SGU_1   = np.exp(y_val_pred_80_SGU_1)   - EPS
        y_test_pred_SGU_1  = np.exp(y_test_pred_80_SGU_1)  - EPS
    else:
        y_train_orig_SGU_1 = y_train_SGU_1.to_numpy()
        y_val_orig_SGU_1   = y_val_SGU_1.to_numpy()
        y_test_orig_SGU_1  = y_test_SGU_1.to_numpy()
        y_train_pred_SGU_1 = y_train_pred_80_SGU_1
        y_val_pred_SGU_1   = y_val_pred_80_SGU_1
        y_test_pred_SGU_1  = y_test_pred_80_SGU_1

    # --- Plots: True vs Predicted for each split ---
    # WHY PLOT ALL THREE SPLITS:
    # - Train plot: checks for underfitting. If the model cannot fit training data,
    #   the architecture/features are insufficient.
    # - Val plot: checks for overfitting. If val performance is much worse than
    #   train, the model memorized training patterns that do not generalize.
    # - Test plot: final out-of-sample evaluation. This is the "real" performance.
    # NOTE: the label says "final model 80%" because predictions come from the
    # model refit on train+val (80%), not the early-stopped 64% model.
    plot_true_vs_pred_series(
        y_train_orig_SGU_1, y_train_pred_SGU_1,
        label="SGU-1 XGBoost (Realized Range) | Final 80% model on TRAIN split", N=500
    )
    plot_true_vs_pred_series(
        y_val_orig_SGU_1, y_val_pred_SGU_1,
        label="SGU-1 XGBoost (Realized Range) | Final 80% model on VAL split", N=500
    )
    plot_true_vs_pred_series(
        y_test_orig_SGU_1, y_test_pred_SGU_1,
        label="SGU-1 XGBoost (Realized Range) | Final 80% model on TEST split (out-of-sample)", N=200
    )


    # --- Gentle cleanup before SGU-2 ---
    import gc; gc.collect()

    # ===========================================================================
    # 8) SGU-2: LABELS (pseudo-mid trend) + FEATURES
    # ===========================================================================
    # SGU-2 labels measure the return of a "pseudo-mid" price between consecutive windows:
    #   pseudo_mid_i = 0.5 * (max_buy_price + min_sell_price)   [with fallbacks]
    #   y_i = (pseudo_mid_i - pseudo_mid_{i-1}) / pseudo_mid_{i-1}
    #
    # WHY "PSEUDO-MID" INSTEAD OF ACTUAL MID-PRICE:
    # The actual mid-price (best_ask + best_bid) / 2 reflects the LOB state at a
    # single instant. The pseudo-mid uses execution prices (max buy aggressor,
    # min sell aggressor) which reflect WHERE trades actually occurred during the
    # window. This is more informative for a market-maker because:
    #   - It reveals the effective transaction price, not just the quoted price.
    #   - It is less susceptible to "flickering" quotes that appear/disappear
    #     at the best level without generating trades.
    #   - The return of the pseudo-mid captures directional pressure from actual
    #     order flow, which is the most actionable signal for adjusting quotes.
    #
    # WHY LOG-RETURNS (USE_LOG_RET_SGU_2=True):
    # Log-returns = log(p_t / p_{t-1}) have several advantages over simple returns:
    #   1. Additivity: log-returns over multiple periods sum up (simpler aggregation)
    #   2. Symmetry: a +1% and -1% move have equal magnitude (unlike simple returns)
    #   3. Distribution: closer to Gaussian, which benefits gradient-based LSTM training
    #   4. Numerical stability: avoid division by small numbers near zero
    #
    # Same day-safe windows are passed to prevent cross-day label contamination.
    # Specifically, the return y_i is only computed when both window i and window
    # i-1 belong to the same day. The first window of each day has a NaN label.

    labels_SGU_2_df = compute_labels_realized_trend(
        orderbook=orderbook_all,
        messages=messages_all,
        delta_t=delta_t,
        tick_size=TICK_SIZE,
        exec_types=EXEC_TYPES,
        use_log_ret=USE_LOG_RET_SGU_2,
        round_pseudo_mid_to_tick=ROUND_PSEUDO_MID_TO_TICK_SGU_2,
        windows=windows,
        window_day_ids=window_day_ids,
    )

    print("\n--- labels_SGU_2_df.tail() ---")
    print(labels_SGU_2_df.tail())
    print("labels_SGU_2_df.shape:", labels_SGU_2_df.shape)

    # SGU-2 features: extended set with L_max lagged values, used as LSTM input.
    #
    # WHY L_MAX LAGGED FEATURES:
    # Unlike XGBoost (which sees a single row of features), the LSTM processes
    # SEQUENCES of feature vectors. Adding lagged versions of base features
    # (e.g., spread_lag_1, spread_lag_2, ..., spread_lag_20) enriches each
    # timestep with autoregressive information. The LSTM can then learn patterns
    # like "spread has been widening for the last 5 windows" without needing to
    # maintain that context in its hidden state alone.
    #
    # WHY A SEPARATE FEATURE SET FROM SGU-1:
    # SGU-2 features may include additional columns not in SGU-1 (e.g., the
    # pseudo-mid return itself at various lags, cumulative volume, etc.).
    # The feature set is designed to be consumed by an LSTM, where redundancy
    # across lags is not a problem (unlike tree models that might split on
    # correlated features sub-optimally).
    features_SGU_2 = compute_features_SGU2(
        orderbook_all,
        messages_all,
        windows,
        labels_SGU_2_df,
        L_max=L_MAX_SGU_2,
    )

    print("\n--- features_SGU_2.tail() ---")
    print(features_SGU_2.tail(5))
    print("features_SGU_2.shape:", features_SGU_2.shape)


    # ===========================================================================
    # 9) SGU-2: TRAIN (or load cached) LSTM model
    # ===========================================================================
    # The LSTM is trained on sequences of length T_SEQ_SGU_2 (default 10 windows).
    # Chronological split: 64% train / 16% val / 20% test (same as SGU-1).
    # Final model is refit on 80% (train + val).
    #
    # KEY DIFFERENCE FROM SGU-1 TRAINING:
    # SGU-1 (XGBoost) operates on FLAT 2D data: shape (N_samples, N_features).
    # SGU-2 (LSTM) operates on 3D SEQUENCE data: shape (N_samples, T_SEQ, N_features).
    # Each "sample" is a sliding window of T_SEQ consecutive feature vectors.
    # The LSTM processes this sequence step by step, building a hidden state
    # that captures temporal dependencies across windows.
    #
    # ADDITIONALLY, SGU-2 SCALES BOTH X AND Y:
    # - scaler_X: StandardScaler for input features (same concept as SGU-1)
    # - scaler_y: StandardScaler for labels (SGU-1 does NOT scale labels)
    # WHY SCALE LABELS: LSTM training uses MSE loss on the scaled labels.
    # Scaling labels to zero-mean, unit-variance ensures the loss function
    # treats all magnitudes equally and prevents gradient explosion/vanishing
    # when label magnitudes are very small (e.g., log-returns ~ 1e-4).
    # Predictions are in scaled space and must be inverse-transformed.
    #
    # THE LSTM TRAINING PIPELINE ALSO RECEIVES SGU-1 FEATURES:
    # The LSTM can optionally use SGU-1 features (spread, volume imbalance, etc.)
    # as part of its input, concatenated with SGU-2 specific features. This
    # creates a "multi-source" feature set where the LSTM benefits from both
    # hand-crafted microstructure features (SGU-1) and lagged trend features (SGU-2).

    # Use GPU if available for faster LSTM training; otherwise fall back to CPU.
    # NOTE: MPS (Apple Silicon GPU) can deadlock inside Spyder's IPython kernel
    # when the DataLoader iterates over large-ish datasets. Until PyTorch fixes
    # the MPS + IPython event-loop interaction, we default to CPU for safety.
    # To try MPS from a plain terminal: set DEVICE_SGU_2 = "mps" manually.
    if torch.cuda.is_available():
        DEVICE_SGU_2 = "cuda"
    else:
        DEVICE_SGU_2 = "cpu"

    # Build SGU-2 metadata for cache key computation.
    # The 'extra' dict captures ALL parameters that affect SGU-2 training
    # beyond the core (symbol, dates, delta_t, exec_types, tick_size).
    # If ANY of these change, the cache key changes and a retrain is triggered.
    meta_sgu2 = _sgu2_meta_config(
        symbol=SYMBOL,
        base_dates=BASE_DATES,
        delta_t=DELTA_T,
        exec_types=EXEC_TYPES,
        tick_size=TICK_SIZE,
        T_SEQ=T_SEQ_SGU_2,
        extra={
            "day_start": DAY_START_SEC,
            "day_end": DAY_END_SEC,
            "use_log_ret": USE_LOG_RET_SGU_2,
            "round_pseudo_mid_to_tick": ROUND_PSEUDO_MID_TO_TICK_SGU_2,
            "L_max": L_MAX_SGU_2,
            "use_sgu1_only": USE_SGU1_ONLY_SGU_2,
        },
    )

    out_SGU_2 = None

    # Attempt to load from cache.
    if USE_SGU2_CACHE and (not FORCE_RETRAIN_SGU2):
        out_SGU_2 = try_load_sgu2(meta=meta_sgu2, artifacts_dir=ARTIFACTS_DIR, device=DEVICE_SGU_2)
        if out_SGU_2 is not None:
            print(f"Loaded cached SGU2 (key={out_SGU_2['cache_key']}, device={out_SGU_2['device']})")

    # If no cache hit, train from scratch.
    if out_SGU_2 is None:
        # WHY SUBPROCESS: After SGU-1 (XGBoost with ThreadPoolExecutor), the
        # process's internal thread-pool / OpenMP state can deadlock PyTorch LSTM
        # forward passes. Running SGU-2 in a *fresh* subprocess with 'spawn' start
        # method guarantees a clean runtime without inherited thread-pool corruption.
        import tempfile, subprocess, pickle, sys

        print("[SGU2] Training in isolated subprocess to avoid thread-pool deadlock...", flush=True)

        # Serialize inputs to a temp file (avoids pipe size limits).
        _sgu2_inputs = dict(
            features_SGU_1=features_SGU_1,
            labels_SGU_1_df=labels_SGU_1_df,
            features_SGU_2=features_SGU_2,
            labels_SGU_2_df=labels_SGU_2_df,
            T_SEQ=T_SEQ_SGU_2,
            use_sgu1_only=USE_SGU1_ONLY_SGU_2,
            n_samples_search=SGU2_LHS_SAMPLES,
            patience=SGU2_EARLY_STOP_PATIENCE,
            min_refit_epochs=SGU2_MIN_REFIT_EPOCHS,
            cpu_safe_mode=SGU2_CPU_SAFE_MODE,
            cpu_num_threads=SGU2_CPU_THREADS,
            device=DEVICE_SGU_2,
            return_data=False,
        )
        _sgu2_in_path = os.path.join(tempfile.gettempdir(), "_sgu2_inputs.pkl")
        _sgu2_out_path = os.path.join(tempfile.gettempdir(), "_sgu2_outputs.pkl")
        with open(_sgu2_in_path, "wb") as _f:
            pickle.dump(_sgu2_inputs, _f, protocol=pickle.HIGHEST_PROTOCOL)

        # Build a small script that loads inputs, runs training, saves outputs.
        _sgu2_script = f"""
import sys, os, pickle
sys.path.insert(0, {repr(os.path.dirname(os.path.abspath(__file__)))})
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
with open({repr(_sgu2_in_path)}, "rb") as f:
    args = pickle.load(f)
from SGU2 import train_sgu2_lstm_pipeline
result = train_sgu2_lstm_pipeline(**args)
with open({repr(_sgu2_out_path)}, "wb") as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
print("[SGU2 subprocess] Done.", flush=True)
"""
        # Run in a fresh Python process (inherits nothing from XGBoost).
        _proc = subprocess.run(
            [sys.executable, "-c", _sgu2_script],
            capture_output=False,   # let stdout/stderr flow to terminal
            timeout=7200,           # 2-hour safety timeout
        )
        if _proc.returncode != 0:
            raise RuntimeError(
                f"SGU2 subprocess failed with return code {_proc.returncode}. "
                "Check the terminal output above for details."
            )
        with open(_sgu2_out_path, "rb") as _f:
            out_SGU_2 = pickle.load(_f)
        # Cleanup temp files.
        for _p in (_sgu2_in_path, _sgu2_out_path):
            try:
                os.remove(_p)
            except OSError:
                pass
        print("[SGU2] Subprocess training complete.", flush=True)

        save_sgu2(
            out=out_SGU_2,
            meta=meta_sgu2,
            artifacts_dir=ARTIFACTS_DIR,
            overwrite=True,  # safe: we just trained, always save the result
        )
        print("Trained and saved SGU2 artifacts")

    # SGU-2 prediction coverage diagnostic (relative to raw day-safe windows).
    # SGU-2 naturally drops more windows due to sequence warm-up (T_SEQ) and
    # finite-sequence filtering.
    _sgu2_preds = out_SGU_2.get("predictions", {})
    _n_pred_sgu2 = (
        len(_sgu2_preds.get("y_train_orig", []))
        + len(_sgu2_preds.get("y_val_orig", []))
        + len(_sgu2_preds.get("y_test_orig", []))
    )
    _n_miss_sgu2 = max(0, _n_win_total - _n_pred_sgu2)
    _pct_miss_sgu2 = (100.0 * _n_miss_sgu2 / _n_win_total) if _n_win_total > 0 else float("nan")
    print(
        f"[SGU2 Coverage] no-pred windows: {_n_miss_sgu2:,}/{_n_win_total:,} "
        f"({_pct_miss_sgu2:.2f}%)"
    )

    # Extract trained artifacts.
    # WHY TWO MODELS (best_model_64 vs final_model):
    #   - best_model_64: The LSTM checkpoint with the best validation loss during
    #     training on 64% of the data. This is the model BEFORE refitting on 80%.
    #     Useful for diagnosing whether the refit improved or degraded performance.
    #   - final_model: The LSTM retrained on 80% (train+val) using the same number
    #     of epochs as the best_model_64. This is the model used for production
    #     predictions and downstream DRL signal generation.
    SGU2_best_model_64 = out_SGU_2["best_model_64"]   # Best LSTM from train (64%) with val early-stop
    SGU2_final_model = out_SGU_2["final_model"]         # LSTM refit on 80% (train + val)
    SGU2_scaler_X = out_SGU_2["scaler_X"]               # StandardScaler for input features
    SGU2_scaler_y = out_SGU_2["scaler_y"]               # StandardScaler for labels
    SGU2_artifacts = out_SGU_2.get("artifacts", {})
    SGU2_feature_cols = SGU2_artifacts.get("feature_names", [])


    # ===========================================================================
    # 10) SGU-2: EVALUATION -- Reconstruct splits & plot true vs predicted
    # ===========================================================================
    # Rebuild the exact (X, y) tensors that training used, then predict per split.
    #
    # WHY RECONSTRUCTION IS NEEDED (SAME REASONING AS SGU-1 SECTION 7):
    # The training pipeline (train_sgu2_lstm_pipeline) returns only models and
    # scalers, not the data splits themselves. We must rebuild the sequences
    # to generate per-split predictions for diagnostic plotting.
    #
    # KEY DIFFERENCE FROM SGU-1 RECONSTRUCTION:
    # SGU-2 data is 3D: shape (N_samples, T_SEQ, N_features). The sequence
    # builder (build_sgu2_multi_feature_sequences) takes 2D feature DataFrames
    # and creates overlapping windows of length T_SEQ:
    #   Sample 0: [features of window 0, window 1, ..., window T_SEQ-1]
    #   Sample 1: [features of window 1, window 2, ..., window T_SEQ]
    #   Sample 2: [features of window 2, window 3, ..., window T_SEQ+1]
    #   ... and so on.
    # The label for each sample is the target at the LAST window in the sequence.
    # This means the first T_SEQ-1 windows are "consumed" as context and do not
    # produce standalone predictions. This is why the SGU-2 test set is slightly
    # shorter than SGU-1's test set (by T_SEQ-1 samples).

    # Build 3D tensors: shape (N_samples, T_SEQ, N_features)
    # The choice between build_sgu1_sequences and build_sgu2_multi_feature_sequences
    # depends on whether we are running the ablation experiment (SGU1 features only)
    # or the full multi-source feature set.
    if USE_SGU1_ONLY_SGU_2:
        X_all_SGU_2, y_all_SGU_2, feature_cols_used_SGU_2, target_window_ids_SGU_2 = build_sgu1_sequences(
            features_SGU_1=features_SGU_1,
            labels_SGU_1_df=labels_SGU_1_df,
            T_SEQ=T_SEQ_SGU_2,
            return_window_ids=True,
        )
    else:
        X_all_SGU_2, y_all_SGU_2, feature_cols_used_SGU_2, target_window_ids_SGU_2 = build_sgu2_multi_feature_sequences(
            features_SGU_1=features_SGU_1,
            features_SGU_2=features_SGU_2,
            labels_SGU_2_df=labels_SGU_2_df,
            T_SEQ=T_SEQ_SGU_2,
            return_window_ids=True,
        )

    # Validate that cached SGU2 feature cols match the rebuilt ones.
    if SGU2_feature_cols and feature_cols_used_SGU_2:
        if list(SGU2_feature_cols) != list(feature_cols_used_SGU_2):
            import warnings
            warnings.warn(
                f"SGU2 feature column mismatch! Cached artifact has "
                f"{len(SGU2_feature_cols)} cols, rebuilt has {len(feature_cols_used_SGU_2)} cols. "
                f"Cached: {SGU2_feature_cols[:5]}... Rebuilt: {feature_cols_used_SGU_2[:5]}... "
                f"Predictions may be misaligned.",
                stacklevel=1,
            )

    # Chronological split indices (same ratios as training).
    # MUST use np.floor to match the exact indices that training used.
    N_SGU_2 = len(X_all_SGU_2)
    i_train_SGU_2 = int(np.floor(0.64 * N_SGU_2))
    i_val_SGU_2 = int(np.floor((0.64 + 0.16) * N_SGU_2))

    X_train_full_SGU_2 = X_all_SGU_2[:i_train_SGU_2]
    y_train_full_SGU_2 = y_all_SGU_2[:i_train_SGU_2]
    wid_train_SGU_2 = target_window_ids_SGU_2[:i_train_SGU_2]

    X_val_full_SGU_2 = X_all_SGU_2[i_train_SGU_2:i_val_SGU_2]
    y_val_full_SGU_2 = y_all_SGU_2[i_train_SGU_2:i_val_SGU_2]
    wid_val_SGU_2 = target_window_ids_SGU_2[i_train_SGU_2:i_val_SGU_2]

    X_test_full_SGU_2 = X_all_SGU_2[i_val_SGU_2:]
    y_test_full_SGU_2 = y_all_SGU_2[i_val_SGU_2:]
    wid_test_SGU_2 = target_window_ids_SGU_2[i_val_SGU_2:]

    # Keep original (unscaled) labels for plotting true values.
    y_train_orig_SGU_2 = y_train_full_SGU_2.copy()
    y_val_orig_SGU_2   = y_val_full_SGU_2.copy()
    y_test_orig_SGU_2  = y_test_full_SGU_2.copy()

    # Dimension constants for reshaping during scaling.
    T_SGU_2 = X_all_SGU_2.shape[1]   # sequence length (e.g., 10)
    F_SGU_2 = X_all_SGU_2.shape[2]   # number of features per timestep


    def _scale_X_seq_SGU_2(X_seq: np.ndarray) -> np.ndarray:
        """
        Apply SGU2_scaler_X to a 3D tensor, matching the training procedure.

        Reshapes (N, T, F) -> (N*T, F) for the scaler, then back to (N, T, F).

        WHY THIS RESHAPE IS NECESSARY:
        sklearn's StandardScaler expects 2D input (samples x features). Our LSTM
        input is 3D (N_samples x T_SEQ x N_features). The trick is:
          1. Flatten the first two dimensions: (N, T, F) -> (N*T, F)
             This treats each timestep of each sample as an independent row.
          2. Apply the scaler (subtract mean, divide by std per feature column).
          3. Reshape back: (N*T, F) -> (N, T, F)
        This works correctly because the scaler operates independently per FEATURE
        COLUMN (axis 1), and the features at each timestep have the same meaning/scale.
        The scaler was fit on training data that was flattened the same way.
        """
        X2 = X_seq.reshape(-1, F_SGU_2)
        X2z = SGU2_scaler_X.transform(X2)
        return X2z.reshape(len(X_seq), T_SGU_2, F_SGU_2).astype(np.float32, copy=False)


    def _inv_scale_y_SGU_2(y_pred_z: np.ndarray) -> np.ndarray:
        """
        Inverse-transform LSTM predictions from standardized back to real scale.

        WHY WE NEED THIS:
        The LSTM was trained on standardized labels (zero-mean, unit-variance).
        Its raw output is in standardized space. To compare with true labels or
        feed into the DRL agent, we must undo the standardization:
          y_real = y_standardized * std_y + mean_y
        This is exactly what scaler_y.inverse_transform() computes.
        """
        return SGU2_scaler_y.inverse_transform(y_pred_z.reshape(-1, 1)).ravel()


    # Scale features per split using the training-set scaler (no information leakage).
    X_train_z_SGU_2 = _scale_X_seq_SGU_2(X_train_full_SGU_2)
    X_val_z_SGU_2   = _scale_X_seq_SGU_2(X_val_full_SGU_2)
    X_test_z_SGU_2  = _scale_X_seq_SGU_2(X_test_full_SGU_2)

    # WORKAROUND: Run ALL LSTM inference in a subprocess to avoid deadlocks.
    # The XGBoost ThreadPoolExecutor corrupts the process's OpenMP/thread state,
    # causing LSTM forward passes to deadlock. Same fix as training: subprocess.
    import tempfile, subprocess, pickle, sys

    def _predict_in_subprocess(model, X_arrays_dict, device_str):
        """Run predict_numpy for multiple arrays in a fresh subprocess."""
        _pred_in = os.path.join(tempfile.gettempdir(), "_sgu2_pred_inputs.pkl")
        _pred_out = os.path.join(tempfile.gettempdir(), "_sgu2_pred_outputs.pkl")
        with open(_pred_in, "wb") as f:
            pickle.dump({"model": model, "arrays": X_arrays_dict, "device": device_str},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

        _script = f"""
import sys, os, pickle, torch
sys.path.insert(0, {repr(os.path.dirname(os.path.abspath(__file__)))})
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.backends.mkldnn.enabled = False
torch.set_num_threads(4)
from SGU2 import predict_numpy
with open({repr(_pred_in)}, "rb") as f:
    data = pickle.load(f)
model = data["model"]
device = torch.device(data["device"] or "cpu")
model = model.to(device)
results = {{}}
for name, X in data["arrays"].items():
    results[name] = predict_numpy(model, X, device=device)
with open({repr(_pred_out)}, "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
print("[SGU2 predict subprocess] Done.", flush=True)
"""
        proc = subprocess.run([sys.executable, "-c", _script],
                              capture_output=False, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"SGU2 predict subprocess failed (rc={proc.returncode})")
        with open(_pred_out, "rb") as f:
            results = pickle.load(f)
        for p in (_pred_in, _pred_out):
            try: os.remove(p)
            except OSError: pass
        return results

    _dev_str = str(out_SGU_2.get("device", "cpu"))

    # --- 10A) Best model (pre-refit, trained on 64%) ---
    print("[SGU2] Running inference (best model 64%) in subprocess...", flush=True)
    _preds_64 = _predict_in_subprocess(
        SGU2_best_model_64,
        {"train": X_train_z_SGU_2, "val": X_val_z_SGU_2, "test": X_test_z_SGU_2},
        _dev_str,
    )
    y_pred_train_64_z_SGU_2 = _preds_64["train"]
    y_pred_val_64_z_SGU_2   = _preds_64["val"]
    y_pred_test_64_z_SGU_2  = _preds_64["test"]

    # Convert predictions from standardized space back to real (original) scale.
    y_pred_train_64_real_SGU_2 = _inv_scale_y_SGU_2(y_pred_train_64_z_SGU_2)
    y_pred_val_64_real_SGU_2   = _inv_scale_y_SGU_2(y_pred_val_64_z_SGU_2)
    y_pred_test_64_real_SGU_2  = _inv_scale_y_SGU_2(y_pred_test_64_z_SGU_2)

    print("\n===== Plotting SGU2: best model (pre-refit, trained on 64%) =====")
    plot_true_vs_pred_series(y_train_orig_SGU_2, y_pred_train_64_real_SGU_2,
                             "SGU-2 LSTM (Pseudo-Mid Trend) | Best 64% model on TRAIN split")
    plot_true_vs_pred_series(y_val_orig_SGU_2, y_pred_val_64_real_SGU_2,
                             "SGU-2 LSTM (Pseudo-Mid Trend) | Best 64% model on VAL split")
    plot_true_vs_pred_series(y_test_orig_SGU_2, y_pred_test_64_real_SGU_2,
                             "SGU-2 LSTM (Pseudo-Mid Trend) | Best 64% model on TEST split (out-of-sample)")

    # --- 10B) Final refit model (trained on 80% = train + val) ---
    X_tv_z_SGU_2 = np.concatenate([X_train_z_SGU_2, X_val_z_SGU_2], axis=0)
    y_tv_true_real_SGU_2 = np.concatenate([y_train_orig_SGU_2, y_val_orig_SGU_2], axis=0)

    print("[SGU2] Running inference (final model 80%) in subprocess...", flush=True)
    _preds_final = _predict_in_subprocess(
        SGU2_final_model,
        {"tv": X_tv_z_SGU_2, "test": X_test_z_SGU_2},
        _dev_str,
    )
    y_pred_tv_z_SGU_2 = _preds_final["tv"]
    y_pred_tv_real_SGU_2 = _inv_scale_y_SGU_2(y_pred_tv_z_SGU_2)

    y_pred_tv_test_z_SGU_2 = _preds_final["test"]
    y_pred_tv_test_real_SGU_2 = _inv_scale_y_SGU_2(y_pred_tv_test_z_SGU_2)

    print("\n===== Plotting SGU2: final refit model (trained on 80%) =====")
    plot_true_vs_pred_series(y_tv_true_real_SGU_2, y_pred_tv_real_SGU_2,
                             "SGU-2 LSTM (Pseudo-Mid Trend) | Final 80% model on TRAIN+VAL split")
    plot_true_vs_pred_series(y_test_orig_SGU_2, y_pred_tv_test_real_SGU_2,
                             "SGU-2 LSTM (Pseudo-Mid Trend) | Final 80% model on TEST split (out-of-sample)")


    # ===========================================================================
    # 11) DRL POLICY FACTORY & DATA SETUP (PRE-CALCULATED SIGNALS)
    # ===========================================================================
    # The DRL agent operates on "signals" produced by SGU-1 and SGU-2.
    # These signals are the model predictions (y_hat) from each SGU.
    #
    # WHAT ARE "SIGNALS" IN THIS CONTEXT?
    # A "signal" is a scalar value that summarizes a complex market state into
    # one number that is directly useful for decision-making:
    #   - SGU-1 signal (realized range prediction): "How wide will the effective
    #     spread be in the next window?" Higher = more profit potential, but also
    #     more risk. The market-maker can widen/narrow quotes accordingly.
    #   - SGU-2 signal (pseudo-mid trend prediction): "Which direction will the
    #     price move in the next window?" Positive = expect price to rise, so
    #     the market-maker should quote asymmetrically (raise both bid and ask).
    #
    # WHY THE SGU TEST SET BECOMES THE "DRL UNIVERSE":
    # The DRL agent must NEVER train on data that the SGU models also trained on.
    # If it did, the SGU predictions would be artificially good (in-sample), and
    # the DRL agent would learn to trust these inflated signals. On real data,
    # the SGU predictions would be worse (out-of-sample), and the DRL agent's
    # learned trust would be misplaced.
    #
    # By using only the SGU TEST set (last 20% of the chronological timeline),
    # we guarantee that:
    #   1. SGU models produce genuine out-of-sample predictions.
    #   2. The DRL agent learns from realistic (noisy, imperfect) signals.
    #   3. The DRL's own test set is out-of-sample for BOTH SGU and DRL.
    #
    # THE SECOND CHRONOLOGICAL SPLIT:
    # Within the DRL universe (SGU test 20%), we perform ANOTHER 64/16/20 split:
    #   - 64% DRL-train: The RL agent explores and learns a policy here.
    #   - 16% DRL-val:   Hyperparameter tuning / early stopping for the RL agent.
    #   - 20% DRL-test:  Final out-of-sample evaluation (never seen during training).
    #
    # NORMALIZATION PRINCIPLE -- "RULER" vs "STREAM":
    # The policy factory normalizes SGU signals to zero-mean, unit-variance before
    # feeding them to the RL agent. This normalization requires computing mean/std
    # statistics from some reference dataset.
    #
    #   "Ruler" (norm_data): The dataset from which normalization statistics
    #   (mean, std) are computed. This ALWAYS comes from DRL-train, regardless
    #   of which phase is being executed. WHY: Using val/test statistics for
    #   normalization would leak future information. The "ruler" is like a
    #   measuring stick calibrated on known data -- it must stay fixed.
    #
    #   "Stream" (execution_data): The actual signal values fed to the agent
    #   during the backtest. This changes per phase:
    #     Phase 1 (train):  stream = DRL-train signals
    #     Phase 2 (val):    stream = DRL-val signals
    #     Phase 3 (test):   stream = DRL-test signals
    #
    #   The policy factory applies: normalized_signal = (raw_signal - ruler_mean) / ruler_std
    #   where ruler_mean and ruler_std come from norm_data (always DRL-train).

    print("\n===================== DRL AGENT SETUP =====================")
    print("Defining DRL Dataset using the 'Test Set' (20%) from SGU training...")

    # --- 11A) Extract SGU predictions for the DRL universe ---
    # SGU-1 predictions: from the test split of the 80% refit XGBoost model.
    # These are the out-of-sample realized-range predictions.
    sgu1_test_df = pd.DataFrame({
        "window_id": wid_test_SGU_1.astype(int),
        "sgu1": np.asarray(y_test_pred_80_SGU_1, dtype=float),
    }).drop_duplicates("window_id").sort_values("window_id")
    # SGU-2 predictions: from the final refit LSTM on the test split.
    # These are the out-of-sample pseudo-mid trend predictions (in real scale,
    # already inverse-transformed from standardized space).
    sgu2_test_df = pd.DataFrame({
        "window_id": wid_test_SGU_2.astype(int),
        "sgu2": np.asarray(y_pred_tv_test_real_SGU_2, dtype=float),
    }).drop_duplicates("window_id").sort_values("window_id")

    # --- 11B) Align by window_id and enforce a continuous backtest universe ---
    # The backtest replays messages continuously and the policy cursor advances once
    # per decision window. To avoid drift, we build a continuous window-id range and
    # fill missing SGU predictions with neutral values.
    if sgu1_test_df.empty or sgu2_test_df.empty:
        raise RuntimeError("SGU test predictions are empty; cannot build DRL universe.")

    win_start = max(int(sgu1_test_df["window_id"].min()), int(sgu2_test_df["window_id"].min()))
    win_end = min(int(sgu1_test_df["window_id"].max()), int(sgu2_test_df["window_id"].max()))
    if win_end < win_start:
        raise RuntimeError(
            f"No overlapping SGU window_id range (SGU1=[{sgu1_test_df['window_id'].min()}, {sgu1_test_df['window_id'].max()}], "
            f"SGU2=[{sgu2_test_df['window_id'].min()}, {sgu2_test_df['window_id'].max()}])."
        )

    drl_window_ids = np.arange(win_start, win_end + 1, dtype=int)
    sgu1_series = sgu1_test_df.set_index("window_id")["sgu1"].reindex(drl_window_ids)
    sgu2_series = sgu2_test_df.set_index("window_id")["sgu2"].reindex(drl_window_ids)

    missing_sgu1 = int(sgu1_series.isna().sum())
    missing_sgu2 = int(sgu2_series.isna().sum())

    N_drl = len(drl_window_ids)
    print(f"DRL Universe Size (continuous window_id range): {N_drl} windows [{win_start}..{win_end}]")
    print(f"Missing predictions before fill -> SGU1: {missing_sgu1} | SGU2: {missing_sgu2}")

    # Fill missing windows: forward-fill first (propagate last known prediction),
    # then fill any remaining leading NaNs with the training-set mean.
    n_drl_train_tmp = max(1, int(0.64 * N_drl))
    sgu1_fill = float(np.nanmean(sgu1_series.iloc[:n_drl_train_tmp].to_numpy(dtype=float)))
    sgu2_fill = float(np.nanmean(sgu2_series.iloc[:n_drl_train_tmp].to_numpy(dtype=float)))
    if not np.isfinite(sgu1_fill):
        sgu1_fill = float(np.nanmean(sgu1_series.to_numpy(dtype=float)))
    if not np.isfinite(sgu2_fill):
        sgu2_fill = float(np.nanmean(sgu2_series.to_numpy(dtype=float)))
    if not np.isfinite(sgu1_fill):
        sgu1_fill = 0.0
    if not np.isfinite(sgu2_fill):
        sgu2_fill = 0.0

    sgu1_preds_drl_universe = sgu1_series.ffill().fillna(sgu1_fill).to_numpy(dtype=float)
    sgu2_preds_drl_universe = sgu2_series.ffill().fillna(sgu2_fill).to_numpy(dtype=float)

    # --- 11C) DRL chronological split: 64% train / 16% val / 20% test ---
    # This is the SECOND level of chronological splitting (the first was at the SGU level).
    # The DRL universe is already out-of-sample for SGU models.
    # Now we split it further for the RL agent's own train/val/test cycle.
    n_drl_train = int(0.64 * N_drl)
    n_drl_val   = int(0.16 * N_drl)
    n_drl_test  = N_drl - n_drl_train - n_drl_val   # Remainder goes to test (no rounding errors)

    # DRL TRAIN set: used for RL agent training AND normalization statistics ("ruler").
    # The mean and std of these signal arrays define the normalization constants
    # that are applied consistently across all three phases.
    sgu1_drl_train = sgu1_preds_drl_universe[:n_drl_train]
    sgu2_drl_train = sgu2_preds_drl_universe[:n_drl_train]

    # DRL VALIDATION set: used for RL hyperparameter tuning / early stopping.
    # The agent sees this data but does NOT use it for gradient updates (in typical
    # RL setups, this would be a separate set of episodes for policy evaluation).
    sgu1_drl_val = sgu1_preds_drl_universe[n_drl_train : n_drl_train + n_drl_val]
    sgu2_drl_val = sgu2_preds_drl_universe[n_drl_train : n_drl_train + n_drl_val]

    # DRL TEST set: held-out for final out-of-sample evaluation.
    # This data has NEVER been seen by:
    #   - SGU-1 model (it was in SGU's test set)
    #   - SGU-2 model (it was in SGU's test set)
    #   - DRL agent (it is in DRL's test set)
    # This is the truest measure of the full pipeline's out-of-sample performance.
    sgu1_drl_test = sgu1_preds_drl_universe[n_drl_train + n_drl_val :]
    sgu2_drl_test = sgu2_preds_drl_universe[n_drl_train + n_drl_val :]

    print(f"DRL Split -> Train: {len(sgu1_drl_train)} | Val: {len(sgu1_drl_val)} | Test: {len(sgu1_drl_test)}")


    # ===========================================================================
    # 12) EXECUTION: RUN BACKTESTS FOR EACH PHASE
    # ===========================================================================
    # The SGU signals were generated at the "window" timescale (aggregated TOB moves).
    # The backtester runs at the "message" timescale (individual LOBSTER events).
    # We must map window indices back to message indices to slice the raw data.
    #
    # WHY TWO TIMESCALES EXIST:
    # The SGU models operate at the "window" timescale: one prediction per
    # delta_t TOB moves (e.g., every 5 best-bid/ask changes). This is a natural
    # frequency for feature computation and prediction.
    #
    # The backtester (MM_LOB_SIM) operates at the "message" timescale: it
    # processes every individual LOB event (order submission, cancellation,
    # execution) to faithfully simulate order matching and queue dynamics.
    #
    # To run a backtest, we need:
    #   1. The raw messages/orderbook data for the relevant time period.
    #   2. The SGU signal arrays for that same period.
    # The mapping from windows to messages is handled by the `windows` list,
    # which stores (start_msg_idx, end_msg_idx) for each window.
    #
    # THE 3-PHASE BACKTEST STRUCTURE:
    # Phase 1 (Train):  Run the RL agent on DRL-train data.
    #   - Purpose: The agent learns from market interactions in this period.
    #   - Ruler = DRL-train, Stream = DRL-train.
    #   - PnL here reflects learning quality (expected to improve over episodes).
    #
    # Phase 2 (Validation):  Run the agent on DRL-val data.
    #   - Purpose: Tune RL hyperparameters (learning rate, epsilon, etc.).
    #   - Ruler = DRL-train (SAME as Phase 1!), Stream = DRL-val.
    #   - The ruler stays constant to prevent normalization leakage.
    #   - PnL here detects overfitting (if train PnL >> val PnL, the agent
    #     has memorized training patterns).
    #
    # Phase 3 (Test):  Run the agent on DRL-test data.
    #   - Purpose: Final out-of-sample evaluation. This number goes in the paper.
    #   - Ruler = DRL-train (STILL the same!), Stream = DRL-test.
    #   - The ruler NEVER changes. This is a fundamental principle:
    #     normalization statistics must come from training data only.
    #   - PnL here is the truest measure of the strategy's real-world viability.
    #
    # WHY THE RULER STAYS CONSTANT ACROSS ALL PHASES:
    # If we recomputed normalization stats for each phase (e.g., using val data
    # to normalize val signals), the agent would see signals in a different scale
    # than during training. This distribution shift would cause:
    #   - The policy to behave unpredictably (actions calibrated for one scale
    #     would be applied to a different scale).
    #   - Non-comparable PnL numbers across phases (different normalization =
    #     different effective signal magnitudes).
    # By always using DRL-train as the ruler, the agent always sees signals on
    # the same scale it was trained on, even when the underlying data changes.

    # Placeholder RL agent (replace with actual trained agent in production).
    # Returns action_idx=5 (hold) in the default 6-action generic space.
    # IMPORTANT: must return an int, not a tuple -- the controller's _decode_action
    # maps integer indices to simulator command tuples.
    # WHY A DUMMY AGENT: This placeholder allows testing the full pipeline
    # end-to-end without a trained RL agent. The "hold" action means the
    # market-maker does nothing, so PnL should be ~0 (modulo any initial
    # positions). Replace with a real trained agent for production use.
    def dummy_agent(s): return 5

    # ---------------------------------------------------------------
    # DRL UNIVERSE: Map signal indices to raw window indices
    # ---------------------------------------------------------------
    # We now carry explicit `drl_window_ids` (raw window_id for each DRL signal),
    # so mapping to message indices is exact and robust even when SGU training
    # dropped windows (day boundaries, lag warm-up, etc.).
    assert N_drl == len(sgu1_preds_drl_universe), (
        f"SGU alignment mismatch: N_drl={N_drl} vs len(sgu1)={len(sgu1_preds_drl_universe)}"
    )

    print(f"\n[Mapping Logic] Aligning Time Scales...")
    print(f"  Total raw windows: {len(windows)}")
    print(f"  DRL window-id range: {int(drl_window_ids[0])} .. {int(drl_window_ids[-1])}")
    print(f"  N_drl (continuous windows): {N_drl}")


    def get_message_slice_for_phase(start_win_rel: int, n_wins: int) -> Tuple[int, int]:
        """
        Translate relative window indices (within the DRL universe) to absolute
        message indices for slicing messages_all / orderbook_all.

        WHY THIS MAPPING IS NECESSARY:
        The DRL operates at the "window" level (one signal per window), but the
        backtester needs raw "message" level data (every individual LOB event).
        This function bridges the two timescales:

            DRL phase (e.g., "train")
                -> relative DRL indices [0, 1, ..., n_drl_train-1]
                -> raw window ids [drl_window_ids[0], ..., drl_window_ids[n_drl_train-1]]
                -> message indices [windows[wid0][0], ..., windows[widN][1]]

        The backtester then receives: messages_all.iloc[msg_start:msg_end]
        which contains ALL individual LOB events spanning those windows.

        VISUAL DIAGRAM:
        windows = [..., (50000, 50120), (50120, 50245), (50245, 50380), ...]
                         ^window 8000   ^window 8001    ^window 8002

        For start_win_rel=0, n_wins=3 (and drl_window_ids[0]=8000):
            abs_start_win = drl_window_ids[0] -> msg_start = 50000
            abs_end_win   = drl_window_ids[2] -> msg_end   = 50380
            => messages_all.iloc[50000:50380] covers all 3 windows

        Parameters
        ----------
        start_win_rel : int
            Starting window index relative to the DRL universe (0 = first DRL window).
        n_wins : int
            Number of windows in this phase.

        Returns
        -------
        (msg_start, msg_end) : Tuple[int, int]
            Absolute message indices for slicing: messages_all.iloc[msg_start:msg_end].
        """
        # Convert relative DRL index to index within drl_window_ids.
        abs_start_pos = int(start_win_rel)
        abs_end_pos = int(start_win_rel + n_wins - 1)
        if abs_start_pos < 0:
            abs_start_pos = 0
        if abs_end_pos >= len(drl_window_ids):
            import warnings
            warnings.warn(
                f"Requested DRL idx {abs_end_pos} exceeds max {len(drl_window_ids)-1}. "
                f"Clipping to {len(drl_window_ids)-1}. Actual n_wins used: "
                f"{len(drl_window_ids) - abs_start_pos} (requested {n_wins}).",
                stacklevel=2,
            )
            abs_end_pos = len(drl_window_ids) - 1

        # Extract message indices from the window structure.
        # Each window is a (start_msg_idx, end_msg_idx) tuple.
        # We take the START of the first window and the END of the last window
        # to get the full contiguous range of messages spanning all windows.
        abs_start_win = int(drl_window_ids[abs_start_pos])
        abs_end_win = int(drl_window_ids[abs_end_pos])
        msg_start = windows[abs_start_win][0]
        msg_end = windows[abs_end_win][1]

        return int(msg_start), int(msg_end)


    # --- General backtest configuration ---
    from MM_LOB_SIM import backtest_LOB_with_MM

    # BACKTEST_CFG: Parameters shared by all three backtest phases.
    # These are passed as **kwargs to backtest_LOB_with_MM().
    #
    # WHY THESE SPECIFIC SETTINGS:
    #   - tick_size: Must match the data's tick size ($0.01 for US equities).
    #     This controls the price grid resolution in the simulator.
    #   - exclude_self_from_state: If True, the agent's own orders are removed
    #     from the LOB state it observes (preventing the agent from "seeing itself").
    #     Set to False here because the RL agent is designed to see the full book
    #     including its own resting orders -- this gives it information about its
    #     own queue position, which is critical for market-making decisions.
    #   - use_queue_ahead: If True, the simulator tracks how many shares are
    #     queued ahead of the agent's limit orders (more realistic but slower).
    #     Set to False for faster execution during development/debugging.
    #   - pure_MM: If True, limits the action space to pure market-making actions
    #     (place/cancel limit orders only). If False, uses a "generic" action
    #     space that may include market orders and other actions. Set to False
    #     to give the RL agent maximum flexibility.
    BACKTEST_CFG = {
        "tick_size": TICK_SIZE,
        "exclude_self_from_state": False,  # Standard for RL: agent sees its own orders in book
        "use_queue_ahead": False,          # False = faster; True = more realistic queue modeling
        "pure_MM": False                   # "generic" action space (not pure market-making only)
    }


    # ---------------------------------------------------------------------------
    # PHASE 1: TRAINING BACKTEST
    # ---------------------------------------------------------------------------
    # Purpose: run the simulator with the RL agent on the DRL training data.
    # Normalization ("ruler"): DRL-train.  Execution ("stream"): DRL-train.
    #
    # WHAT HAPPENS IN THIS PHASE:
    # 1. Map DRL-train windows to raw message indices.
    # 2. Slice the raw messages/orderbook for this time period.
    # 3. Convert to backtester format (grid-based prices).
    # 4. Create a policy factory that wraps the RL agent with SGU signals.
    # 5. Run the backtester, which replays every LOB event and lets the agent
    #    respond with market-making actions.
    # 6. The backtester returns a tracking DataFrame with PnL, inventory,
    #    and other metrics at each timestep.
    print("\n>>> STARTING PHASE 1: TRAINING BACKTEST (DRL Train Split)")

    # Step 1-2: Map DRL-train window range [0, n_drl_train) to message indices
    # and slice the raw data.
    msg_start_train, msg_end_train = get_message_slice_for_phase(0, n_drl_train)

    msgs_train = messages_all.iloc[msg_start_train : msg_end_train].reset_index(drop=True)
    ob_train   = orderbook_all.iloc[msg_start_train : msg_end_train].reset_index(drop=True)

    # Step 3: Convert LOBSTER absolute prices to grid-based backtester format.
    # base_price is anchored to the first row's mid-price of this specific slice.
    msgs_train_bt, ob_train_bt = build_backtest_inputs(msgs_train, ob_train, tick_size=TICK_SIZE)

    # Step 4: Create the policy factory.
    # WHY A "FACTORY" AND NOT JUST THE AGENT:
    # The policy factory wraps the raw RL agent with signal management logic:
    #   a) It holds the full signal arrays (sgu1_execution_data, sgu2_execution_data).
    #   b) It tracks TOB moves in the backtester and advances a cursor every
    #      delta_t moves (matching the SGU window granularity).
    #   c) At each cursor advance, it reads the next signal value, normalizes it
    #      using norm_data statistics (ruler), and appends it to the agent's state.
    #   d) It then calls the agent's act function and returns the action to the
    #      backtester.
    # This decoupling means the RL agent itself does NOT need to know about
    # SGU models, normalization, or window synchronization -- it just sees
    # a state vector and returns an action integer.
    policy_for_training = Deep_RL_With_Signal_policy_factory(
        rl_act_fn=dummy_agent,
        # Normalization stats ("ruler"): always from DRL-train to prevent data leakage.
        # The factory computes mean(sgu1_norm_data), std(sgu1_norm_data), etc.
        # and uses these to standardize the execution data signals.
        sgu1_norm_data=sgu1_drl_train, sgu2_norm_data=sgu2_drl_train,
        # Signal stream: DRL-train (agent learns on this data).
        # For Phase 1, ruler and stream are the same dataset. This is expected:
        # during training, the agent is both normalized by AND operating on
        # the training data.
        sgu1_execution_data=sgu1_drl_train, sgu2_execution_data=sgu2_drl_train,
        # Synchronization: the policy advances its signal cursor every delta_t TOB moves,
        # matching the window granularity used to generate SGU signals.
        # WHY use_tob_update=True: The policy monitors the backtester's LOB state
        # and detects when the top-of-book changes. Every delta_t such changes,
        # it advances to the next signal. This keeps the signal cursor synchronized
        # with the window structure even though the backtester operates per-message.
        use_tob_update=True,
        n_tob_moves=delta_t,
        # throttle_every_n_steps=1: The agent is queried on every message step.
        # In production, you might throttle to reduce computational cost.
        throttle_every_n_steps=1,
        action_mode="generic",
        verbose=False
    )

    # Step 5-6: Run the backtest.
    # backtest_LOB_with_MM returns three values:
    #   - messages output (not used here, hence _)
    #   - orderbook output (not used here, hence _)
    #   - mm_track: DataFrame tracking the market-maker's state at each step
    #     (PnL, inventory, mid-price, spread, etc.)
    # controller=None: no external order flow controller (the backtester just
    # replays the historical messages).
    _, _, mm_track_train = backtest_LOB_with_MM(
        messages=msgs_train_bt, orderbook=ob_train_bt,
        controller=None, mm_policy=policy_for_training, **BACKTEST_CFG
    )
    print(f"Phase 1 Complete. Final PnL: {mm_track_train['MM_TotalPnL'].iloc[-1]:.2f}")


    # ---------------------------------------------------------------------------
    # PHASE 2: VALIDATION BACKTEST
    # ---------------------------------------------------------------------------
    # Purpose: evaluate the RL agent on unseen validation data (for tuning).
    # Normalization ("ruler"): DRL-train (same as Phase 1 -- no leakage!).
    # Execution ("stream"): DRL-val (the agent sees new data).
    #
    # WHY VALIDATION IS IMPORTANT:
    # The validation backtest answers the question: "Does the agent's learned
    # policy generalize to new (but similar) market conditions?"
    # If Phase 1 PnL is strongly positive but Phase 2 PnL is negative or flat,
    # the agent has OVERFIT to idiosyncratic patterns in the training data
    # (e.g., a specific price trend or volatility regime that does not recur).
    # In that case, RL hyperparameters (learning rate, exploration epsilon,
    # reward shaping) should be adjusted before running Phase 3.
    #
    # KEY DIFFERENCE FROM PHASE 1:
    # - Ruler (norm_data): UNCHANGED. Still DRL-train. This is critical:
    #   if we used DRL-val statistics for normalization, the agent would see
    #   signals on a different scale than during training, and any scale-sensitive
    #   policy decisions would break.
    # - Stream (execution_data): NOW DRL-val. The agent processes signal values
    #   from the validation period, which it has NEVER seen before.
    # - The message data also changes: we slice a different time period from
    #   messages_all / orderbook_all, corresponding to the validation windows.
    print("\n>>> STARTING PHASE 2: VALIDATION BACKTEST (DRL Val Split)")

    # Map DRL-val windows to message indices.
    # start_win_rel = n_drl_train: validation starts immediately after training.
    msg_start_val, msg_end_val = get_message_slice_for_phase(n_drl_train, n_drl_val)

    msgs_val = messages_all.iloc[msg_start_val : msg_end_val].reset_index(drop=True)
    ob_val   = orderbook_all.iloc[msg_start_val : msg_end_val].reset_index(drop=True)
    msgs_val_bt, ob_val_bt = build_backtest_inputs(msgs_val, ob_val, tick_size=TICK_SIZE)

    policy_for_validation = Deep_RL_With_Signal_policy_factory(
        rl_act_fn=dummy_agent,
        # Ruler: still DRL-train (prevents data leakage into normalization).
        # SAME arrays as Phase 1 -- the mean/std used for normalization do not change.
        sgu1_norm_data=sgu1_drl_train, sgu2_norm_data=sgu2_drl_train,
        # Stream: DRL-val (agent evaluated on this new, unseen data).
        # The factory will iterate through these arrays as the backtester
        # replays val-period LOB events.
        sgu1_execution_data=sgu1_drl_val, sgu2_execution_data=sgu2_drl_val,
        use_tob_update=True,
        n_tob_moves=delta_t,
        throttle_every_n_steps=1,
        action_mode="generic",
        verbose=False
    )

    _, _, mm_track_val = backtest_LOB_with_MM(
        messages=msgs_val_bt, orderbook=ob_val_bt,
        controller=None, mm_policy=policy_for_validation, **BACKTEST_CFG
    )
    print(f"Phase 2 Complete. Final PnL: {mm_track_val['MM_TotalPnL'].iloc[-1]:.2f}")


    # ---------------------------------------------------------------------------
    # PHASE 3: TEST BACKTEST (FINAL OUT-OF-SAMPLE EVALUATION)
    # ---------------------------------------------------------------------------
    # Purpose: final evaluation on held-out test data.
    # Normalization ("ruler"): DRL-train (same ruler as always).
    # Execution ("stream"): DRL-test (completely unseen data).
    #
    # THIS IS THE MOST IMPORTANT PHASE:
    # The test backtest PnL is the single most important number produced by this
    # pipeline. It represents the strategy's performance on data that:
    #   - SGU-1 model has NEVER seen (it was in SGU's test 20%)
    #   - SGU-2 model has NEVER seen (it was in SGU's test 20%)
    #   - DRL agent has NEVER seen (it is in DRL's test 20%)
    #   - Normalization statistics did NOT include this data
    #
    # This multi-layer out-of-sample guarantee makes the test PnL a credible
    # estimate of real-world performance. If Phase 3 PnL is positive and
    # meaningfully different from zero (after transaction costs), the strategy
    # has a genuine edge.
    #
    # WHAT IF TEST PNL IS NEGATIVE?
    # A negative test PnL could indicate:
    #   1. Overfitting at the SGU level (predictions are noise out-of-sample)
    #   2. Overfitting at the DRL level (policy exploits training-specific patterns)
    #   3. Non-stationarity (market regime changed between train and test periods)
    #   4. Insufficient signal strength (SGU predictions are too noisy to be useful)
    # Each failure mode requires a different response (more data, regularization,
    # feature engineering, or accepting that the signal is too weak).
    print("\n>>> STARTING PHASE 3: TEST BACKTEST (DRL Test Split)")

    # Map DRL-test windows to message indices.
    # start_win_rel = n_drl_train + n_drl_val: test starts after train and val.
    msg_start_test, msg_end_test = get_message_slice_for_phase(n_drl_train + n_drl_val, n_drl_test)

    msgs_test = messages_all.iloc[msg_start_test : msg_end_test].reset_index(drop=True)
    ob_test   = orderbook_all.iloc[msg_start_test : msg_end_test].reset_index(drop=True)
    msgs_test_bt, ob_test_bt = build_backtest_inputs(msgs_test, ob_test, tick_size=TICK_SIZE)

    policy_for_test = Deep_RL_With_Signal_policy_factory(
        rl_act_fn=dummy_agent,
        # Ruler: STILL DRL-train -- for the third and final time.
        # This is the normalization principle in action: one ruler, all phases.
        # The agent interprets signals using the same "yardstick" it learned with.
        sgu1_norm_data=sgu1_drl_train, sgu2_norm_data=sgu2_drl_train,
        # Stream: DRL-test (final out-of-sample evaluation).
        # These signal values are from the very end of the chronological timeline.
        # They represent the most recent (and most unknown) market conditions.
        sgu1_execution_data=sgu1_drl_test, sgu2_execution_data=sgu2_drl_test,
        use_tob_update=True,
        n_tob_moves=delta_t,
        throttle_every_n_steps=1,
        action_mode="generic",
        verbose=False
    )

    _, _, mm_track_test = backtest_LOB_with_MM(
        messages=msgs_test_bt, orderbook=ob_test_bt,
        controller=None, mm_policy=policy_for_test, **BACKTEST_CFG
    )
    print(f"Phase 3 Complete. Final PnL: {mm_track_test['MM_TotalPnL'].iloc[-1]:.2f}")

    # ===========================================================================
    # END OF PIPELINE
    # ===========================================================================
    # At this point, mm_track_train, mm_track_val, and mm_track_test contain
    # the full backtest results for each phase. These DataFrames can be used
    # for further analysis: Sharpe ratio computation, drawdown analysis,
    # inventory risk assessment, and comparison across different RL agents
    # or SGU model configurations.
