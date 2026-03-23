"""
==============================================================================
SGU1.py  --  Spread Gap Unit 1  (Realized-Range Prediction Module)
==============================================================================

PURPOSE
-------
This module is the first "Spread Gap Unit" (SGU-1) in a market-making Deep
Reinforcement Learning (DRL) system.  Its job is to predict the *realized
price range* that the market maker will face in the next top-of-book (TOB)
movement window.  That prediction is later consumed by the RL agent to set
optimal bid/ask quotes.

THEORETICAL BACKGROUND  (Appendix A of the reference paper)
------------------------------------------------------------
In high-frequency market making, the *spread* a market maker posts must
cover the adverse-selection cost incurred when informed traders pick off
stale quotes.  One empirical proxy for this cost is the **realized price
range** within a short window of TOB movements:

    y_i  =  max(P^buy_i)  -  min(P^sell_i)

where P^buy_i are the prices at which *buy aggressors* executed and P^sell_i
are the prices at which *sell aggressors* executed during window i.

When one side has no trades, we fall back to the average quoted best ask or
best bid over the window (see the four-case definition in
``compute_labels_realized_range``).  This fallback avoids NaN labels while
still encoding the prevailing spread level.

DATA CONVENTIONS  (LOBSTER format)
----------------------------------
- **Columns use CamelCase**: AskPrice_1, BidPrice_1, AskSize_1, BidSize_1,
  Type, Direction, Price, Size, Time, TimeAbs.
- **Direction** is the *passive* side (the limit order that was executed):
      Direction == -1  -->  a *sell* limit order was executed
                        -->  the *aggressor* (taker) was a BUYER
      Direction == +1  -->  a *buy*  limit order was executed
                        -->  the *aggressor* (taker) was a SELLER
  This is LOBSTER's convention: the sign refers to the resting order, not
  the initiator.  The code below inverts this to identify buy- vs.
  sell-aggressor trades.
- **Type 4** = ``execute_visible`` -- the default execution type we filter
  on.  Other types (e.g. hidden executions, cancellations) are excluded
  because they do not represent observable aggressive trading.
- **Time** is float seconds-after-midnight (resets each day).
  **TimeAbs** is a monotonically-increasing float across concatenated days,
  which is needed for the slope feature so that cross-day regressions do
  not see backward jumps.

PIPELINE OVERVIEW
-----------------
1. ``compute_labels_realized_range``   -- Label construction (y_i).
2. ``compute_features_SGU1``           -- Feature engineering (X_i).
3. ``train_SGU1``                      -- Two-stage LHS hyperparameter
   search with forward-chaining CV, followed by a final 80% refit.
4. Caching helpers (``save_sgu1``, ``try_load_sgu1``) for artifact
   persistence between runs.

The model predicts y_{t+1} from features at time t  (one-step-ahead).

==============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
# Standard-library type hints and utilities
from typing import Dict, Any, Optional, Tuple, List, Iterable
import warnings
import logging

from pathlib import Path

# Core numerical / data-science stack
import pandas as pd
import numpy as np
_XGB_IMPORT_ERROR = None
try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - environment-dependent import
    xgb = None
    _XGB_IMPORT_ERROR = exc

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  # for feature normalization
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Project-specific helper: builds non-overlapping windows of `delta_t`
# consecutive TOB (top-of-book) moves from orderbook snapshots.
from LOB_processor import top_moves_windows

# Serialization utilities for caching trained artifacts to disk
import json
import hashlib
from joblib import dump, load


# ============================================================================
# OPTIONAL DEPENDENCY GUARD
# ============================================================================
def _require_xgboost() -> None:
    """
    Ensure xgboost is importable before entering SGU1 routines that depend on it.
    """
    if xgb is None:
        raise RuntimeError(
            "xgboost is not importable in this environment. "
            "Fix the xgboost installation and retry SGU1 training/loading."
        ) from _XGB_IMPORT_ERROR


# ============================================================================
# SECTION 1: LABEL COMPUTATION  --  Realized Price Range (y_i)
# ============================================================================
# This section implements the label computation from Appendix A of the
# reference paper.  For each window of `delta_t` TOB moves, we compute a
# "modified realized price range" that captures the effective spread the
# market experienced.  This label is what the XGBoost model will learn to
# predict one step ahead.
# ============================================================================

def compute_labels_realized_range(
    orderbook: pd.DataFrame,
    messages: pd.DataFrame,
    delta_t: int,
    tick_size: Optional[float] = None,
    exec_types: Tuple = (4,),
    windows: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Compute the SGU1 label y_i for each TOB-window of length `delta_t` moves.

    Label (modified realized price range), for window i:
      - P^buy_i: prices associated with buy-aggressor executions.
      - P^sell_i: prices associated with sell-aggressor executions.
      - \\bar{P}^{ask}_i: mean best ask over the window.
      - \\bar{P}^{bid}_i: mean best bid over the window.

      y_i =
        max(P^buy_i) - min(P^sell_i)                              if both non-empty
        \\bar{P}^{ask}_i - min(P^sell_i)                           if P^buy_i empty
        max(P^buy_i) - \\bar{P}^{bid}_i                            if P^sell_i empty
        \\bar{P}^{ask}_i - \\bar{P}^{bid}_i                         if both empty

    Per Appendix A (Eq. 21) in the reference paper, labels are kept in
    absolute price units and then rounded to the tick grid (e.g., $0.01).

    WHY this specific formula?
    --------------------------
    The realized range max(P^buy) - min(P^sell) captures the *worst-case*
    spread that a market maker would have experienced if she had been
    continuously quoting at the best bid/ask.  A buy aggressor lifts the
    ask; a sell aggressor hits the bid.  The difference between the highest
    price a buyer paid and the lowest a seller received approximates the
    adverse-selection cost within the window.

    The fallback cases handle windows where only one side traded (or
    neither).  In those cases, we substitute the *average* quoted price on
    the missing side, which degrades gracefully to the quoted spread when
    no trades occur at all.

    Parameters
    ----------
    orderbook : pd.DataFrame
        Must have columns ['AskPrice_1','BidPrice_1'] and be aligned by
        index with `messages`.  (LOB_data / CamelCase convention.)
    messages : pd.DataFrame
        Must have at least columns ['Type','Direction','Price'] and be
        aligned by index with `orderbook`.

        LOBSTER Direction convention (passive side, integer):
          Direction == -1 : sell LO was executed --> aggressor was BUYER
          Direction == +1 : buy  LO was executed --> aggressor was SELLER
    delta_t : int
        Number of TOB moves per window.
    tick_size : float, optional
        If provided, each y_i is rounded to the nearest multiple of tick_size.
    exec_types : tuple of int
        Which LOBSTER message Type values count as trades/executions.
        Default (4,) = execute_visible only.
    windows : list of (int, int), optional
        Pre-computed day-safe windows (from ``top_moves_windows_by_day`` or
        equivalent).  When provided, the function uses these windows directly
        instead of rebuilding them from the orderbook.  This is **critical**
        for multi-day concatenated data: the caller should build windows
        per-day to avoid cross-overnight contamination, and then pass them
        here so that labels, features, and durations all share the exact
        same window definitions.

    Returns
    -------
    pd.DataFrame
        One row per window with:
         ['window_id','start','end','n_msgs','n_trades','n_buy','n_sell',
          'avg_ask','avg_bid','max_buy','min_sell','label']
    """
    # ---- Window source: use pre-computed if provided, else build from scratch ----
    # When operating on single-day data, building windows here is fine.
    # For multi-day concatenated DataFrames, the caller MUST pass day-safe
    # windows to prevent cross-overnight contamination.  Without this,
    # the last window of day D could bleed into the first rows of day D+1,
    # creating a label that mixes two trading sessions -- an economic
    # nonsense that would corrupt the model.
    if windows is None:
        windows: List[Tuple[int, int]] = top_moves_windows(orderbook, delta_t)

    # ---------------------------
    # Pre-convert all needed columns to NumPy once (LOB_data CamelCase convention).
    # Type is integer (4 = execute_visible), Direction is integer (+1/-1 passive side).
    #
    # WHY convert to NumPy up front?
    # Because we iterate over potentially thousands of windows below.
    # Repeated pandas .iloc / .loc indexing inside a Python loop is orders
    # of magnitude slower than slicing pre-materialized NumPy arrays.
    # This was a critical performance optimization that removed the original
    # bottleneck (see the Portuguese comment below, kept for historical context).
    # ---------------------------
    is_exec = messages["Type"].isin(exec_types).to_numpy()
    prices = pd.to_numeric(messages["Price"], errors="coerce").to_numpy()
    dirs = messages["Direction"].to_numpy(dtype=int)

    ask_np = orderbook["AskPrice_1"].to_numpy()
    bid_np = orderbook["BidPrice_1"].to_numpy()

    rows = []

    for w_id, (start, end) in enumerate(windows):
        # ------------------------------------------------------------------
        # For each window [start, end), compute the average quoted prices
        # (used as fallback when one side has no trades) and then classify
        # trades by aggressor side.
        # ------------------------------------------------------------------

        # Slice orderbook for averages over the window [start, end)
        ask_slice = ask_np[start:end]
        bid_slice = bid_np[start:end]
        avg_ask = float(np.nanmean(ask_slice)) if ask_slice.size else np.nan
        avg_bid = float(np.nanmean(bid_slice)) if bid_slice.size else np.nan

        # ---------------------------
        # AQUI ESTA O GARGALO REMOVIDO:
        # Antes: mascara gigante pd.Series(False,...)
        # Agora: slice direto via NumPy
        #
        # [Historical note, kept in original Portuguese]:
        # The bottleneck was removed here.  Previously, a full-length
        # pd.Series(False, ...) mask was created for every window, which
        # was extremely slow.  Now we slice the pre-materialized NumPy
        # arrays directly using Python's built-in slice object.
        # ---------------------------
        idx = slice(start, end)
        win_exec_mask = is_exec[idx]

        if not win_exec_mask.any():
            # ----------------------------------------------------------
            # CASE: No trades in the window.
            # Fallback: use the average quoted spread as the label.
            # WHY: If no aggressive trades occurred, the best available
            # measure of the "cost of trading" is simply the prevailing
            # bid-ask spread.  This keeps the label well-defined and
            # avoids NaN, which would cause sample loss during training.
            # ----------------------------------------------------------
            n_trades = 0
            n_buy = 0
            n_sell = 0
            max_buy = np.nan
            min_sell = np.nan
            label = float(avg_ask - avg_bid)

        else:
            price_slice = prices[idx]
            dir_slice = dirs[idx]

            price_exec = price_slice[win_exec_mask]
            dir_exec = dir_slice[win_exec_mask]

            n_trades = len(price_exec)

            # ----------------------------------------------------------
            # DIRECTION CONVENTION (LOBSTER):
            #   Direction == -1: a SELL limit order was executed
            #       --> the incoming (aggressive) order was a BUY
            #       --> this is a "buy trade" from the taker's perspective
            #   Direction == +1: a BUY limit order was executed
            #       --> the incoming (aggressive) order was a SELL
            #       --> this is a "sell trade" from the taker's perspective
            #
            # WHY this matters: The label formula uses max(P^buy) and
            # min(P^sell) where "buy" and "sell" refer to the AGGRESSOR
            # side.  If the convention were inverted, the label would be
            # economically meaningless.
            # ----------------------------------------------------------
            n_buy = int((dir_exec == -1).sum())
            n_sell = int((dir_exec == 1).sum())

            # Extract trade price sets by aggressor side
            buy_prices  = price_exec[dir_exec == -1]
            sell_prices = price_exec[dir_exec == 1]

            # Remove any NaN prices (defensive; should not happen with clean data)
            buy_prices = buy_prices[~np.isnan(buy_prices)]
            sell_prices = sell_prices[~np.isnan(sell_prices)]

            # ----------------------------------------------------------
            # Compute label according to the four-case fallback rules:
            #
            # Case 1 (both sides present):
            #   y = max(buy_prices) - min(sell_prices)
            #   This is the full realized range -- the most informative case.
            #
            # Case 2 (no buy trades):
            #   y = avg_ask - min(sell_prices)
            #   We substitute the average quoted ask for the missing buy
            #   side.  Rationale: the best ask is where a hypothetical
            #   buyer would have traded, so it proxies the missing
            #   max(P^buy).
            #
            # Case 3 (no sell trades):
            #   y = max(buy_prices) - avg_bid
            #   Symmetric reasoning: the average quoted bid proxies the
            #   missing min(P^sell).
            #
            # Case 4 (no trades at all -- should be caught above, but
            #   defensive):
            #   y = avg_ask - avg_bid  (the quoted spread)
            # ----------------------------------------------------------
            if buy_prices.size > 0 and sell_prices.size > 0:
                max_buy = float(np.max(buy_prices))
                min_sell = float(np.min(sell_prices))
                label = max_buy - min_sell
            elif buy_prices.size == 0 and sell_prices.size > 0:
                max_buy = np.nan
                min_sell = float(np.min(sell_prices))
                label = avg_ask - min_sell
            elif sell_prices.size == 0 and buy_prices.size > 0:
                max_buy = float(np.max(buy_prices))
                min_sell = np.nan
                label = max_buy - avg_bid
            else:
                max_buy = np.nan
                min_sell = np.nan
                label = avg_ask - avg_bid

        # Optional rounding to the exchange tick grid in absolute-price units
        # (Appendix A, Eq. 21 in the reference paper).
        if tick_size is not None and np.isfinite(label):
            tick = float(tick_size)
            if tick > 0 and np.isfinite(tick):
                label = round(label / tick) * tick

        rows.append(
            dict(
                window_id=w_id,
                start=start,
                end=end,
                n_msgs=int(end - start),
                n_trades=int(n_trades),
                n_buy=int(n_buy),
                n_sell=int(n_sell),
                avg_ask=avg_ask,
                avg_bid=avg_bid,
                max_buy=max_buy,
                min_sell=min_sell,
                label=float(label),
            )
        )

    return pd.DataFrame(rows)


# ============================================================================
# SECTION 2: UTILITY HELPERS
# ============================================================================
# Small pure-function helpers used by the feature engineering code below.
# ============================================================================

def _ensure_datetime(series: pd.Series) -> pd.Series:
    """
    Ensure that a pandas Series is of datetime64 dtype.  If it is not,
    attempt to coerce it.  This is a defensive helper used when the
    caller is unsure whether time columns have already been parsed.
    """
    if not np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series, errors="coerce")
    return series

def _concat_window_slices(idxs: List[Tuple[int, int]], arr: np.ndarray) -> np.ndarray:
    """
    Concatenate multiple (start, end) slices from a single NumPy array
    into one contiguous array.  Used by VWAP and slope computations to
    efficiently gather data from several *previous* windows without
    repeated DataFrame indexing.

    Parameters
    ----------
    idxs : list of (int, int)
        Each tuple is a (start, end) pair defining a half-open slice [start, end).
    arr : np.ndarray
        The source array to slice from.

    Returns
    -------
    np.ndarray
        The concatenation of all slices, or an empty array if no valid slices.
    """
    if not idxs:
        return np.empty(0, dtype=arr.dtype)
    parts = [arr[s:e] for s, e in idxs if e > s]
    return np.concatenate(parts) if parts else np.empty(0, dtype=arr.dtype)


# ============================================================================
# SECTION 3: FEATURE ENGINEERING  (X_i)
# ============================================================================
# This section constructs the feature vector X_i for each TOB-move window i.
# The features are designed to capture the *state of the market* at the
# beginning of window i, using ONLY information available up to (but not
# including) the window itself.  This strict temporal discipline ensures
# that when we later align X_t -> y_{t+1}, there is no look-ahead bias.
#
# Feature categories:
#   - Spread at window start:  The quoted spread is the most direct input
#     to the market maker's pricing decision.
#   - Volume imbalance:  Measures the directional pressure (buyers vs.
#     sellers) within the window.
#   - VWAP over previous windows:  Quote-weighted average price captures
#     the "fair price" consensus from recent TOB states.
#   - Mid-price slope:  A linear trend indicator over previous windows,
#     capturing short-term momentum or mean-reversion signals.
#   - Uptick percentage:  Fraction of consecutive trade-price increases,
#     another momentum proxy.
#   - Trade counts (rolling):  Activity levels, which correlate with
#     volatility and thus with the expected realized range.
#   - Large trades:  The number of trades exceeding a size threshold,
#     signaling informed or institutional activity.
#   - Time of day:  Captures intraday seasonality (e.g. wider spreads at
#     open/close).
#   - Lagged labels:  Autoregressive terms y_{t-1}, ..., y_{t-5} that
#     exploit persistence in the realized range series.
# ============================================================================

def compute_features_SGU1(
    orderbook: pd.DataFrame,
    messages: pd.DataFrame,
    windows: List[Tuple[int, int]],
    labels_df: pd.DataFrame,
    exec_types: Iterable = (4,),
    large_quantile: float = 0.90,
    large_threshold_override: Optional[float] = None,
    window_day_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build the feature matrix for SGU-1 (one row per TOB-move window).

    The feature set is intentionally over-complete; a subsequent feature-
    pruning step in ``train_SGU1`` drops the least-important features by
    XGBoost gain.

    Parameters
    ----------
    orderbook : pd.DataFrame
        LOBSTER orderbook snapshots (CamelCase columns).
    messages : pd.DataFrame
        LOBSTER message log (CamelCase columns).
    windows : list of (start, end)
        Pre-computed TOB-move windows (same ones used for label computation).
    labels_df : pd.DataFrame
        Output of ``compute_labels_realized_range``, containing at least
        ['window_id', 'label'].  Used to create lagged-label features.
    exec_types : iterable of int
        Which Type values count as executions (default: (4,) = execute_visible).
    large_quantile : float
        Quantile of trade sizes used to define "large" trades.
    large_threshold_override : float, optional
        If provided, overrides the quantile computation with a fixed threshold.
        WHY: When computing features for validation/test data, we must use
        the threshold computed on training data only -- otherwise the 90th
        percentile of *future* trade sizes leaks into the features, violating
        the temporal no-leakage principle.
    window_day_ids : np.ndarray, optional
        An integer array of length len(windows) indicating the trading day
        each window belongs to (e.g. 0, 0, 0, ..., 1, 1, 1, ...).  When
        provided, lagged-label features (label_lag1 ... label_lag5) are
        set to NaN wherever the lag crosses a day boundary.

        WHY?  Without this masking, label_lag1 for the first window of
        day D+1 would contain the label from the last window of day D.
        That value is contaminated by overnight dynamics (different
        regime, overnight gaps, opening auction effects) and would
        mislead the model into thinking the previous window's realized
        range is a valid autoregressive signal for the current window.

    Returns
    -------
    pd.DataFrame
        Columns: ['window_id'] + all feature columns.
        The DataFrame's ``attrs["large_threshold"]`` contains the
        large-trade threshold used (needed for val/test feature computation).
    """
    # ------------------------------------------------------------------
    # Pre-coerce all needed columns once (LOB_data CamelCase convention).
    # Time is float seconds-after-midnight; TimeAbs is monotone across days.
    #
    # WHY coerce to numeric?  LOBSTER files can occasionally have string
    # representations or mixed types after concatenation.  pd.to_numeric
    # with errors="coerce" handles this defensively.
    # ------------------------------------------------------------------
    t_intraday = pd.to_numeric(messages["Time"], errors="coerce").to_numpy(dtype=float)
    t_abs = pd.to_numeric(messages["TimeAbs"], errors="coerce").to_numpy(dtype=float)
    price = pd.to_numeric(messages["Price"], errors="coerce").to_numpy()
    size = pd.to_numeric(messages["Size"], errors="coerce").to_numpy()
    direction = messages["Direction"].to_numpy(dtype=int)

    ask = pd.to_numeric(orderbook["AskPrice_1"], errors="coerce").to_numpy()
    bid = pd.to_numeric(orderbook["BidPrice_1"], errors="coerce").to_numpy()
    ask_sz = pd.to_numeric(orderbook["AskSize_1"], errors="coerce").to_numpy()
    bid_sz = pd.to_numeric(orderbook["BidSize_1"], errors="coerce").to_numpy()
    mid = 0.5 * (ask + bid)

    # Execution filter once, as NumPy boolean array
    is_exec = messages["Type"].isin(exec_types).to_numpy()

    # ------------------------------------------------------------------
    # Threshold for "large" trades.
    # WHY a separate override path?  During training, the threshold is
    # computed on ALL execution sizes in the dataset (which is the training
    # set).  During inference or when computing features for val/test, we
    # must reuse the *training* threshold to avoid data leakage.  The
    # caller passes it as large_threshold_override.
    # ------------------------------------------------------------------
    if large_threshold_override is not None:
        large_threshold = large_threshold_override
    else:
        warnings.warn(
            "large_threshold_override not provided: computing large_threshold "
            "on the FULL dataset (train+val+test). This may cause data leakage "
            "if val/test data is included. Pass large_threshold_override (computed "
            "on training data only) to avoid this.",
            stacklevel=2,
        )
        global_trade_sizes = size[is_exec & np.isfinite(size)]
        large_threshold = np.nanquantile(global_trade_sizes, large_quantile) if global_trade_sizes.size else 0.0

    W = len(windows)
    data_rows: List[Dict] = []

    # ------------------------------------------------------------------
    # Main window loop: compute per-window (local) features.
    # These features use only data within each window [start, end).
    # ------------------------------------------------------------------
    for wid, (start, end) in enumerate(windows):
        rng = slice(start, end)

        # Restrict everything to the window via slicing
        win_exec_mask = is_exec[rng]

        # If there are executions, slice once and mask
        if win_exec_mask.any():
            price_win = price[rng][win_exec_mask]
            size_win = size[rng][win_exec_mask]
            dir_win = direction[rng][win_exec_mask]
        else:
            price_win = np.array([], dtype=float)
            size_win = np.array([], dtype=float)
            dir_win = np.array([], dtype=int)

        # ---- Feature: Spread at the beginning of the window ----
        # WHY: The quoted spread at window start is the single most
        # predictive feature for the realized range.  A wider spread at the
        # start of a window tends to persist and produce a wider realized
        # range in the following window.
        spread_start = (ask[start] - bid[start]) if start < len(ask) else np.nan

        # ---- Feature: Volume imbalance ----
        # Direction == -1 (passive sell was executed) = buy aggressor volume
        # Direction == +1 (passive buy was executed)  = sell aggressor volume
        #
        # WHY: Order flow imbalance is a classic microstructure signal.
        # When buyers dominate, the price tends to move up, which widens
        # the realized range on the buy side.  The imbalance captures this
        # directional pressure in [-1, +1].
        if size_win.size:
            buy_vol = np.nansum(size_win[dir_win == -1])
            sell_vol = np.nansum(size_win[dir_win == 1])
        else:
            buy_vol = 0.0
            sell_vol = 0.0

        denom = buy_vol + sell_vol
        vol_imbalance = (buy_vol - sell_vol) / denom if denom > 0 else 0.0
        total_vol = float(denom) if denom > 0 else 0.0

        # ---- Feature: Percentage of upticks within the window ----
        # WHY: Uptick ratio is a simple momentum proxy.  A high uptick
        # percentage suggests sustained buying pressure, which correlates
        # with a wider realized range (the market is "moving").
        if price_win.size >= 2:
            upticks = np.sum(price_win[1:] > price_win[:-1])
            pct_upticks = upticks / (price_win.size - 1)
        else:
            pct_upticks = 0.0

        # ---- Feature: Large trades ----
        # WHY: Large trades are often associated with informed or
        # institutional traders.  Their presence signals that the realized
        # range may be wider because informed traders move prices more.
        # We count large buys and large sells separately so the model can
        # distinguish directional large-trade pressure.
        if np.isfinite(large_threshold) and size_win.size:
            n_large_buys = int(np.sum((dir_win == -1) & (size_win >= large_threshold)))
            n_large_sells = int(np.sum((dir_win == 1) & (size_win >= large_threshold)))
        else:
            n_large_buys = 0
            n_large_sells = 0

        # ---- Feature: Time of day (fractional hours since midnight) ----
        # WHY: Intraday seasonality is a well-documented phenomenon.
        # Spreads and volatility are typically higher near the market
        # open and close, and lower during midday.  By giving the model
        # the hour-of-day, it can learn these U-shaped patterns.
        # Time is float seconds-after-midnight, so dividing by 3600 gives hours.
        if start < len(t_intraday) and np.isfinite(t_intraday[start]):
            tod_hours = float(t_intraday[start]) / 3600.0
        else:
            tod_hours = np.nan

        data_rows.append(dict(
            window_id=wid,
            start=start,
            end=end,
            spread_start=spread_start,
            vol_imbalance=vol_imbalance,
            total_vol=total_vol,
            pct_upticks=pct_upticks,
            n_trades=int(win_exec_mask.sum()),
            n_buys=int(np.sum(dir_win == -1)),
            n_sells=int(np.sum(dir_win == 1)),
            n_large_buys=n_large_buys,
            n_large_sells=n_large_sells,
            tod_hours=tod_hours,
        ))

    base = pd.DataFrame(data_rows)

    # ------------------------------------------------------------------
    # ROLLING / LOOKBACK FEATURES
    # ------------------------------------------------------------------
    # These features aggregate information from PREVIOUS windows, ensuring
    # strict temporal causality (no future information leaks into X_t).
    # ------------------------------------------------------------------

    def prev_sum(series: pd.Series, periods: int) -> pd.Series:
        """
        Rolling sum over the *previous* `periods` windows.
        Returns NaN until there are at least `periods` past windows available.

        WHY shift(1)?  We shift by 1 to exclude the CURRENT window from the
        sum.  This is critical: including the current window would leak
        information about the present into a feature that is supposed to
        represent only past activity.

        WHY min_periods=periods?  We require a full lookback of `periods`
        windows before returning a non-NaN value.  This avoids unreliable
        partial sums at the start of the series.
        """
        # shift(1) removes the current window; min_periods=periods enforces enough history
        return series.shift(1).rolling(periods, min_periods=periods).sum()

    def prev_vwap(
        ask_px: np.ndarray,
        bid_px: np.ndarray,
        ask_qty: np.ndarray,
        bid_qty: np.ndarray,
        idxs: List[Tuple[int, int]],
        back_windows: int
    ) -> np.ndarray:
        """
        Quote-based VWAP over the last `back_windows` *previous* windows.

        For each message/snapshot j, the quote VWAP contribution is:
            qvwap_j = (ask_px_j * ask_qty_j + bid_px_j * bid_qty_j) / (ask_qty_j + bid_qty_j)

        We aggregate all snapshots from windows [i-back_windows, ..., i-1]
        using the same total depth (ask_qty + bid_qty) as weight.

        Returns NaN until there are at least `back_windows` previous windows.

        WHY quoted VWAP?
        ----------------
        Trade-only VWAP can be undefined in sparse/no-trade windows and causes
        heavy missingness.  Quote VWAP is always tied to the visible top-of-book
        state, which is exactly what the MM observes when deciding quotes.
        """
        v = np.full(len(idxs), np.nan, dtype=float)

        for i in range(len(idxs)):
            # Not enough past windows yet -- leave as NaN
            if i < back_windows:
                continue

            # Gather data from the `back_windows` windows strictly before window i
            prev_idxs = idxs[i - back_windows : i]

            a_prev = _concat_window_slices(prev_idxs, ask_px)
            b_prev = _concat_window_slices(prev_idxs, bid_px)
            aq_prev = _concat_window_slices(prev_idxs, ask_qty)
            bq_prev = _concat_window_slices(prev_idxs, bid_qty)
            if a_prev.size == 0 or b_prev.size == 0:
                continue

            # Keep finite quote/depth values; allow zero depth but not negative depth.
            m = (
                np.isfinite(a_prev)
                & np.isfinite(b_prev)
                & np.isfinite(aq_prev)
                & np.isfinite(bq_prev)
                & (aq_prev >= 0)
                & (bq_prev >= 0)
            )
            if not m.any():
                continue

            a = a_prev[m]
            b = b_prev[m]
            aq = aq_prev[m]
            bq = bq_prev[m]

            w = aq + bq
            pos = w > 0
            if pos.any():
                num = np.nansum(a[pos] * aq[pos] + b[pos] * bq[pos])
                denom = np.nansum(w[pos])
                if denom > 0:
                    v[i] = float(num / denom)
                    continue

            # Fallback for zero-depth snapshots: average valid mid-quote.
            mids = 0.5 * (a + b)
            mids = mids[np.isfinite(mids)]
            if mids.size:
                v[i] = float(np.nanmean(mids))

        return v

    def prev_slope(
        mid: np.ndarray,
        idxs: List[Tuple[int, int]],
        s_windows: int,
        t_sec: np.ndarray,
    ) -> np.ndarray:
        """
        Linear slope of mid-price vs. time over the last `s_windows`
        previous windows.

        WHY a linear slope?
        -------------------
        The slope captures short-term momentum (positive slope = prices
        rising) or mean-reversion tendency.  Different horizons (s=1, 3, 5)
        allow the model to distinguish between very-short-term and
        medium-term trends.

        Forward-fill strategy:
            - Returns last valid slope when not enough data.
            - last_val starts at 0.0 (neutral slope).

        WHY forward-fill instead of NaN?
        ---------------------------------
        At the start of the dataset, there are not enough past windows to
        compute a slope.  Rather than dropping those rows (losing data) or
        using NaN (which XGBoost handles but imperfectly), we forward-fill
        with the last computed slope.  The initial value of 0.0 encodes a
        "no trend" prior, which is the least informative default.

        t_sec is TimeAbs in float seconds (monotonically increasing across
        days).  Using TimeAbs instead of intraday Time avoids the overnight
        discontinuity where seconds-after-midnight resets to a small value.
        """
        v = np.full(len(idxs), np.nan, dtype=float)
        last_val = np.nan  # value to forward-fill when slope cannot be computed

        for i in range(len(idxs)):
            if i < s_windows:
                v[i] = last_val
                continue

            prev_idxs = idxs[i - s_windows : i]

            mids_all = _concat_window_slices(prev_idxs, mid)
            tt_all   = _concat_window_slices(prev_idxs, t_sec)

            m = np.isfinite(mids_all) & np.isfinite(tt_all)

            # Not enough valid points --> ffill
            if m.sum() < 2:
                v[i] = last_val
                continue

            mids = mids_all[m]
            tt   = tt_all[m]

            # Offset time to start at zero (already in seconds, no /1e9 needed).
            # WHY offset?  np.cov / np.var are numerically more stable when
            # the values are centered near zero rather than being large
            # epoch-like numbers.
            tt = tt - tt[0]

            var_t = np.var(tt)
            if var_t <= 0:
                v[i] = last_val
                continue

            # Simple OLS slope: beta = cov(t, mid) / var(t)
            cov = np.cov(tt, mids, ddof=0)[0, 1]
            if not np.isfinite(cov):
                v[i] = last_val
                continue

            slope = float(cov / var_t)
            v[i] = slope
            last_val = slope  # update ffill buffer

        return v

    # ---- Rolling trade-count sums over previous p windows ----
    # WHY multiple horizons (1, 2, 3, 5, 10)?
    # Trade intensity is highly variable.  A single lookback period
    # cannot capture both sudden bursts (p=1) and sustained high activity
    # (p=10).  By providing multiple horizons, we let XGBoost learn which
    # time scale is most predictive for the current regime.
    for p in [1, 2, 3, 5, 10]:
        base[f"n_trades_prev_p{p}"] = prev_sum(base["n_trades"], p)

    # ---- Quote-VWAP over previous r windows ----
    # Multiple horizons (r=1, 3, 5) capture different scales of "fair value".
    for r in [1, 3, 5]:
        base[f"vwap_prev_r{r}"] = prev_vwap(ask, bid, ask_sz, bid_sz, windows, r)

    # ---- Slope of mid over previous s windows (using TimeAbs in float seconds) ----
    # Multiple horizons (s=1, 3, 5) capture short- to medium-term trends.
    for s in [1, 3, 5]:
        base[f"slope_prev_s{s}"] = prev_slope(mid, windows, s, t_abs)

    # ---- Merge labels and create lagged-label features ----
    # WHY lagged labels?
    # The realized range exhibits significant autocorrelation (persistence).
    # Today's spread is strongly predicted by yesterday's spread.  By
    # including y_{t-1} through y_{t-5}, we give the model access to the
    # recent history of the target variable, which acts as a powerful
    # autoregressive signal.  The model can learn how quickly shocks to
    # the realized range decay over successive windows.
    lbl = labels_df.sort_values("window_id")
    base = base.merge(lbl[["window_id", "label"]], on="window_id", how="left")
    for L in range(1, 6):
        base[f"label_lag{L}"] = base["label"].shift(L)

    # --- Mask cross-day lags to NaN ---
    # If window_day_ids is provided, set lagged features to NaN wherever
    # the lagged window belongs to a different trading day than the current
    # window.  Without this, label_lag1 for the first window of day D+1
    # would contain the label from the last window of day D -- a value
    # contaminated by overnight dynamics (different regime, overnight gaps).
    if window_day_ids is not None:
        day_arr = np.asarray(window_day_ids)
        for L in range(1, 6):
            # For row i, lag L looks back to row i-L.
            # If day_arr[i] != day_arr[i-L], it's a cross-day lag.
            cross_day = np.zeros(len(base), dtype=bool)
            cross_day[:L] = True  # first L rows have no valid lag
            cross_day[L:] = day_arr[L:] != day_arr[:-L]
            base.loc[cross_day, f"label_lag{L}"] = np.nan

    # ------------------------------------------------------------------
    # Assemble final feature column list.
    # This explicit list controls the ordering and inclusion of features
    # in the model.  Any column added to `data_rows` above but NOT listed
    # here (e.g. 'start', 'end') is excluded from training.
    # ------------------------------------------------------------------
    feature_cols = [
        "n_trades_prev_p1", "n_trades_prev_p2", "n_trades_prev_p3", "n_trades_prev_p5", "n_trades_prev_p10",
        "spread_start",
        "vol_imbalance",
        "vwap_prev_r1", "vwap_prev_r3", "vwap_prev_r5",
        "slope_prev_s1", "slope_prev_s3", "slope_prev_s5",
        "tod_hours",
        "total_vol",
        "pct_upticks",
        "n_large_buys", "n_large_sells",
        "label_lag1", "label_lag2", "label_lag3", "label_lag4", "label_lag5",
    ]

    features = base[["window_id"] + feature_cols].copy()

    # Store the large-trade threshold as a DataFrame attribute so the
    # caller can retrieve it later.  This is needed when computing
    # features for validation/test data: the caller must pass this value
    # as large_threshold_override to prevent data leakage (i.e., the
    # 90th-percentile trade size must come from training data only).
    #
    # Using DataFrame.attrs is lightweight and does not add a column to
    # the feature matrix.  Note that attrs are NOT preserved through all
    # pandas operations (e.g., merge, concat), so the caller should
    # extract the value immediately after calling this function.
    features.attrs["large_threshold"] = large_threshold
    return features


# ============================================================================
# SECTION 4: MODEL TRAINING  --  Two-Stage LHS Search + Forward-Chaining CV
# ============================================================================
# This is the core training pipeline for SGU-1.  It:
#   1. Aligns features X_t with labels y_{t+1} (one-step-ahead prediction).
#   2. Splits the data chronologically: 64% train / 16% val / 20% test.
#   3. Normalizes features using a StandardScaler fitted on train only.
#   4. Runs a two-stage Latin Hypercube Search (LHS) over XGBoost
#      hyperparameters:
#      - Stage 1: Broad exploration with 3-fold forward-chaining CV.
#      - Stage 2: Local refinement around the best Stage 1 anchors with
#        5-fold forward-chaining CV.
#   5. Trains a final model on train+val (80%) with the best hyperparameters.
#
# WHY two stages?
# ---------------
# A single-stage random search over a high-dimensional discrete grid is
# inefficient.  Stage 1 cheaply identifies promising regions of HP space
# (using only 3 CV folds and 1000 boosting rounds).  Stage 2 then zooms
# into the discrete neighborhood of the top-K Stage-1 configurations with
# a more thorough evaluation (5 folds, 2000 rounds).  This explore-then-
# exploit strategy finds better hyperparameters with fewer total
# evaluations than a flat random search.
#
# WHY forward-chaining (expanding-window) CV instead of k-fold?
# --------------------------------------------------------------
# Financial time series are non-stationary and autocorrelated.  Standard
# k-fold CV would train on future data and test on past data, which:
#   (a) inflates performance estimates due to temporal leakage, and
#   (b) selects HP configurations that exploit this leakage.
# Forward-chaining CV respects the arrow of time: each fold's training
# set is strictly before its validation set.
# ============================================================================

def train_SGU1(
    features_SGU_1: pd.DataFrame,
    labels_SGU_1_df: pd.DataFrame,
    *,
    frac_train: float = 0.64,
    frac_val: float = 0.16,
    tick_size: float = 0.01,
    seed: int = 42,
    n_stage1: int = 500,
    n_stage2: int = 150,
    top_quantile: float = 0.10,
    max_workers_cap: int = 8,
) -> Dict[str, Any]:
    """
    Encapsulates the original SGU-1 training script into a single callable.

    This function performs the full pipeline: data alignment, chronological
    splitting, feature normalization, two-stage hyperparameter search with
    forward-chaining CV, diagnostic metrics on all splits, and a final
    80% refit model for deployment.

    Inputs (minimum required)
    -------------------------
    features_SGU_1 : pd.DataFrame
        Must contain 'window_id' plus feature columns.  Output of
        ``compute_features_SGU1``.
    labels_SGU_1_df : pd.DataFrame
        Must contain columns ['window_id', 'label'].  Output of
        ``compute_labels_realized_range``.

    Keyword-only parameters
    -----------------------
    frac_train : float
        Fraction of data for the training split (default 0.64 = 64%).
    frac_val : float
        Fraction of data for the validation split (default 0.16 = 16%).
        The remaining (1 - frac_train - frac_val) goes to the test split.
    tick_size : float
        Exchange tick size, used for rounding predictions in the tick-
        rounding helper (not for label computation here).
    seed : int
        Random seed for reproducibility of LHS sampling.
    n_stage1 : int
        Number of HP configurations to evaluate in Stage 1 (broad search).
    n_stage2 : int
        Number of HP configurations to evaluate in Stage 2 (local search).
    top_quantile : float
        Fraction of Stage 1 results used as anchors for Stage 2.
    max_workers_cap : int
        Maximum number of parallel threads for HP evaluation.

    Returns
    -------
    dict with keys:
      - "final_model_80" : xgboost.Booster (trained on train+val = 80%)
      - "scaler"         : StandardScaler fitted on train(64%) only
      - "feature_cols"   : list[str] in training order
      - "best_params"    : dict used to train final models
      - "best_rounds"    : int boosting rounds selected by CV
      - "metrics_df"     : pd.DataFrame summary (same as script)
      - "drop_feats"     : list[str] dropped in the "reduced" diagnostics block
    """
    _require_xgboost()

    # Suppress verbose XGBoost warnings during hyperparameter search.
    # Scoped to xgboost module only to avoid masking legitimate warnings
    # from other libraries (e.g., scikit-learn, pandas).
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    logging.getLogger("xgboost").setLevel(logging.ERROR)

    # ============================================================================
    # STEP 0: Build the prediction dataset  (X_t --> y_{t+1})
    # ============================================================================
    # The key alignment: we use features from window t to predict the label
    # of window t+1.  This is the "one-step-ahead" prediction setup.
    # WHY?  In a live market-making scenario, the agent must decide its
    # quotes for the NEXT window using only information available NOW.
    # Therefore the model must be trained to predict y_{t+1} from X_t.
    # ============================================================================
    _df = (
        features_SGU_1
        .merge(labels_SGU_1_df[["window_id", "label"]], on="window_id", how="inner")
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    feature_cols = [c for c in _df.columns if c not in ["window_id", "label"]]
    X_full = _df[feature_cols].copy()
    y_full = _df["label"].copy()

    # Align X_t --> y_{t+1}:
    #   X = X_full[:-1]   (features at times 0, 1, ..., T-1)
    #   y = y_full shifted by -1, then [:-1]  (labels at times 1, 2, ..., T)
    # This ensures that row i of X corresponds to window i, and row i of y
    # corresponds to window i+1 -- the one-step-ahead target.
    X = X_full.iloc[:-1].copy()
    y = y_full.shift(-1).iloc[:-1].copy()

    # Drop rows with NaNs (strict time ordering).
    # WHY: NaNs arise at the start of the series where rolling features
    # (e.g. n_trades_prev_p10) do not yet have enough history, and from
    # the lagged-label features (label_lag5 needs 5 prior windows).
    # We drop rather than impute to avoid introducing noise.
    mask = (~X.isna().any(axis=1)) & (y.notna())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Keep a full clean copy for the chronological splits
    X_all, y_all = X.copy(), y.copy()

    # ============================================================================
    # STEP 1: Chronological split -- 64% train / 16% val / 20% test
    # ============================================================================
    # WHY chronological (no shuffle)?
    # Financial data is non-stationary.  Shuffling would let the model
    # "see" future data during training, producing overly optimistic
    # validation metrics and poor real-world performance.  By splitting
    # chronologically, train is always BEFORE val, and val is always
    # BEFORE test, mimicking the true deployment scenario.
    # ============================================================================
    n_total = len(X_all)
    if n_total == 0:
        raise ValueError(
            "No valid samples after X_t->y_{t+1} alignment and NaN filtering. "
            "Check feature/label construction and lag history."
        )
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            "Chronological split produced an empty partition. "
            f"train={n_train}, val={n_val}, test={n_test}. "
            "Adjust frac_train/frac_val or provide more data."
        )

    print(
        f"Total samples: {n_total:,} | "
        f"train={n_train:,} ({n_train / n_total:.1%}), "
        f"val={n_val:,} ({n_val / n_total:.1%}), "
        f"test={n_test:,} ({n_test / n_total:.1%})"
    )

    # Chronological splits (no shuffle)
    X_train = X_all.iloc[:n_train].astype(np.float32).reset_index(drop=True)
    y_train = y_all.iloc[:n_train].astype(np.float32).reset_index(drop=True)

    X_val = X_all.iloc[n_train : n_train + n_val].astype(np.float32).reset_index(drop=True)
    y_val = y_all.iloc[n_train : n_train + n_val].astype(np.float32).reset_index(drop=True)

    X_test = X_all.iloc[n_train + n_val :].astype(np.float32).reset_index(drop=True)
    y_test = y_all.iloc[n_train + n_val :].astype(np.float32).reset_index(drop=True)

    # ============================================================================
    # STEP 1.1: Feature normalization (no leakage: fit on train, apply to val/test)
    # ============================================================================
    # WHY normalize?
    # While XGBoost (tree-based) is invariant to monotonic feature
    # transformations, normalization helps in two ways:
    #   1. The composite scoring function uses Pearson correlation and
    #      relative RMSE, which can behave better on standardized data.
    #   2. Downstream consumers of the model (e.g. the RL agent) may
    #      expect normalized features for consistency.
    #
    # WHY fit only on train?
    # Fitting the scaler on the full dataset would leak information about
    # the validation and test distributions (their means and variances)
    # into the training features.  This is a subtle but important form of
    # data leakage that can inflate validation metrics.
    # ============================================================================
    scaler = StandardScaler()

    # Fit scaler only on training features to avoid information leakage
    scaler.fit(X_train)

    # Transform all splits using the same scaler
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val),   columns=X_val.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

    # Replace original feature matrices with their normalized versions
    X_train = X_train_scaled
    X_val   = X_val_scaled
    X_test  = X_test_scaled

    # For compatibility with the original code, we keep X,y as the (normalized) training subset
    X = X_train
    y = y_train

    print(f"Using {len(X):,} samples (64%) for randomized search & training.")

    # DMatrix once (reused by CV on the training subset).
    # WHY DMatrix?  XGBoost's DMatrix is an optimized internal data
    # structure that avoids repeated DataFrame-to-array conversions during
    # multi-fold CV and multi-round boosting.
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))

    # ============================================================================
    # STEP 2: Time-series folds (forward-chaining, no leakage) -- CORRECTED VERSION
    # ============================================================================
    # Forward-chaining (also called "expanding window") CV partitions the
    # training data into folds where each fold's training set is an initial
    # segment of the time series, and the validation set is the next
    # contiguous block.  This mimics the real deployment scenario where the
    # model is trained on all available history and then predicts the
    # immediate future.
    #
    # Schematically for 3 folds (K=3):
    #   Fold 0:  train = [0, ..., min_train)           test = [min_train, ..., min_train+b)
    #   Fold 1:  train = [0, ..., min_train+b)         test = [min_train+b, ..., min_train+2b)
    #   Fold 2:  train = [0, ..., min_train+2b)        test = [min_train+2b, ..., n)
    #
    # where b = (n - min_train) / K.
    #
    # WHY min_train_frac = 0.2?
    # We need a minimum amount of historical data to train a meaningful
    # model for the first fold.  20% of the training set is a reasonable
    # lower bound.
    # ============================================================================
    def make_time_series_folds(n_samples: int, nfold: int, min_train_frac: float = 0.2):
        """
        Build forward-chaining folds for time series CV with no overlap
        between train and test indices.

        For each fold k:
          - test (validation) is a contiguous block in time,
          - train uses only data strictly before that block,
          - the first test block starts at min_train_frac * n_samples.
        """
        indices = np.arange(n_samples)
        min_train = int(n_samples * min_train_frac)

        # total number of points reserved for testing across all folds
        n_test_total = n_samples - min_train
        if n_test_total <= 0:
            raise ValueError("Not enough samples to allocate test folds with the requested min_train_frac.")
        if n_test_total < nfold:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {nfold} time-series folds "
                f"with min_train_frac={min_train_frac}. Reduce folds or increase data."
            )

        # split the test region [min_train : n_samples) into nfold contiguous blocks
        fold_sizes = np.full(nfold, n_test_total // nfold, dtype=int)
        fold_sizes[: n_test_total % nfold] += 1

        folds = []
        start = min_train
        for fold_size in fold_sizes:
            stop = start + fold_size
            test_idx = indices[start:stop]
            train_idx = indices[:start]  # always strictly in the past
            folds.append((train_idx, test_idx))
            start = stop

        return folds

    # ---- Separate fold sets for Stage 1 (fast) and Stage 2 (refined) ----
    # WHY different fold counts?
    # Stage 1 evaluates 500 configurations -- speed matters.  3 folds give
    # a noisy but cheap estimate of generalization performance, sufficient
    # to identify promising regions.
    # Stage 2 evaluates only ~150 configurations in the best regions.  5
    # folds give a more stable estimate, reducing the chance of selecting
    # a lucky-but-bad configuration.
    N_FOLDS_STAGE1 = 3
    N_FOLDS_STAGE2 = 5

    FOLDS_TS_STAGE1 = make_time_series_folds(len(X), N_FOLDS_STAGE1, min_train_frac=0.2)
    FOLDS_TS_STAGE2 = make_time_series_folds(len(X), N_FOLDS_STAGE2, min_train_frac=0.2)

    # ============================================================================
    # STEP 3: Training helpers (GPU-safe training and CPU fallback)
    # ============================================================================
    # These helper functions wrap xgb.train with automatic GPU -> CPU
    # fallback.  WHY?  The code is designed to run on machines with or
    # without a CUDA-capable GPU.  When gpu_hist is requested but fails
    # (e.g. no GPU available, or out-of-memory), we silently retry with
    # the CPU 'hist' method.  This makes the pipeline portable without
    # requiring the user to manually configure device settings.
    # ============================================================================
    def _train_one_fold(params, d_tr, d_te, num_boost_round, early_stopping_rounds):
        """
        Train a single XGBoost model on one CV fold.
        Handles GPU --> CPU fallback automatically.
        """
        try:
            return xgb.train(
                params, d_tr,
                num_boost_round=num_boost_round,
                evals=[(d_te, "eval")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
        except xgb.core.XGBoostError:
            params_cpu = params.copy()
            params_cpu.pop("predictor", None)
            params_cpu["tree_method"] = "hist"
            params_cpu.setdefault("nthread", -1)
            return xgb.train(
                params_cpu, d_tr,
                num_boost_round=num_boost_round,
                evals=[(d_te, "eval")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )

    def train_final_model(params, dtrain_obj, num_boost_round):
        """
        Train the final model, preferring GPU if available.
        If GPU is not available, silently fall back to CPU 'hist'.
        """
        try:
            return xgb.train(params=params, dtrain=dtrain_obj, num_boost_round=num_boost_round)
        except xgb.core.XGBoostError:
            params_cpu = params.copy()
            params_cpu.pop("predictor", None)
            params_cpu["tree_method"] = "hist"
            params_cpu.setdefault("nthread", -1)
            return xgb.train(params=params_cpu, dtrain=dtrain_obj, num_boost_round=num_boost_round)

    # ============================================================================
    # STEP 4: Base XGBoost parameters (GPU preferred; fallback handled above)
    # ============================================================================
    # These are the "fixed" parameters that do NOT vary across the HP search.
    # The search varies: max_depth, min_child_weight, subsample,
    # colsample_bytree, eta (learning rate), and reg_lambda.
    # ============================================================================
    base = dict(
        objective="reg:squarederror",   # Standard squared-error loss for regression
        booster="gbtree",               # Gradient-boosted trees (not gblinear)
        eval_metric="rmse",             # Root mean squared error as the evaluation metric
        tree_method="gpu_hist",         # prefer GPU; cv_rmse() / train_final_model() fall back if not available
        predictor="gpu_predictor",
        max_bin=128,                    # Histogram bins; 128 is a good speed/accuracy trade-off
        random_state=seed,
        seed=seed,
        nthread=-1,                     # Use all available CPU threads (overridden to 1 inside parallel workers)
        verbosity=0,                    # Suppress XGBoost's own logging
    )

    # ============================================================================
    # STEP 5: Discrete hyperparameter grid + Latin Hypercube Sampling (LHS)
    # ============================================================================
    # WHY LHS instead of grid search or pure random search?
    # - Grid search is combinatorially explosive: 10*10*7*7*8*7 = 274,400
    #   configurations.  Evaluating each with 5-fold CV is infeasible.
    # - Pure random search samples the HP space uniformly but can leave
    #   large gaps and cluster in some regions by chance.
    # - LHS stratifies each dimension into n_samples equally-probable
    #   intervals, guaranteeing better coverage of the space with fewer
    #   samples.  This is the key insight from McKay et al. (1979).
    #
    # We map the continuous LHS samples [0,1]^6 to the discrete grid by
    # treating each [0,1] coordinate as an index into the sorted option
    # list for that hyperparameter.
    # ============================================================================
    rng = np.random.default_rng(seed)

    # Discrete hyperparameter grid, close to the paper's Table 2.
    # Each list is sorted in a meaningful order (increasing value).
    PARAM_GRID = {
        "max_depth":        [2, 3, 4, 5, 6, 7, 8, 9 ,10, 15],
        "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "subsample":        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eta":              [0.0005, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],   # learning_rate (sorted ascending)
        "reg_lambda":       [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],          # regularization term
    }

    def _lhs(n_samples, dim, rng_):
        """
        Latin Hypercube Sampling on [0,1]^dim.
        Produces a stratified design over the unit hypercube.

        HOW it works:
        1. Divide [0,1] into n_samples equal strata of width 1/n_samples.
        2. Within each stratum, sample a uniform point.
        3. For each dimension, randomly permute which stratum each sample
           falls into.  This ensures that the marginal distribution along
           each dimension is stratified (exactly one sample per stratum),
           while the joint distribution covers the space well.
        """
        cut = (np.arange(n_samples) + rng_.random(n_samples)) / n_samples
        H = np.zeros((n_samples, dim), dtype=np.float64)
        for j in range(dim):
            P = rng_.permutation(n_samples)
            H[:, j] = cut[P]
        return H

    def _choice_from_grid(options, u_scalar):
        """
        Pick a discrete hyperparameter value from 'options' using u in [0,1].

        Maps the continuous uniform [0,1] to a discrete index by flooring
        u * len(options).  This gives each option an equal probability of
        being selected (1/n), preserving the stratification property of LHS.
        """
        n = len(options)
        idx = int(np.floor(u_scalar * n))
        if idx == n:
            idx = n - 1
        return options[idx]

    def _map_sample(u):
        """
        Map a LHS point u in [0,1]^6 to a discrete param set.

        Each coordinate u_j is mapped to a discrete grid value for the
        corresponding hyperparameter.  The ordering of coordinates is:
          u0 -> max_depth
          u1 -> min_child_weight
          u2 -> subsample
          u3 -> colsample_bytree
          u4 -> eta (learning rate)
          u5 -> reg_lambda (L2 regularization)
        """
        u0, u1, u2, u3, u4, u5 = u

        # "Sensitive" hyperparameters (shape + learning rate)
        max_depth        = _choice_from_grid(PARAM_GRID["max_depth"],        u0)
        eta              = _choice_from_grid(PARAM_GRID["eta"],              u4)

        # Other structural hyperparameters
        min_child_weight = _choice_from_grid(PARAM_GRID["min_child_weight"], u1)
        subsample        = _choice_from_grid(PARAM_GRID["subsample"],        u2)
        colsample_bytree = _choice_from_grid(PARAM_GRID["colsample_bytree"], u3)
        reg_lambda       = _choice_from_grid(PARAM_GRID["reg_lambda"],       u5)

        return dict(
            max_depth        = int(max_depth),
            min_child_weight = float(min_child_weight),
            subsample        = float(subsample),
            colsample_bytree = float(colsample_bytree),
            eta              = float(eta),
            reg_lambda       = float(reg_lambda),
        )

    def sample_params_smart(n_samples, rng_):
        """
        Sample a list of parameter dictionaries using LHS + discrete grid mapping.
        This is the entry point that combines LHS over [0,1]^6 with the
        discrete-grid mapping to produce n_samples HP configurations with
        good coverage of the search space.
        """
        U = _lhs(n_samples, 6, rng_)
        return [_map_sample(U[i]) for i in range(n_samples)]

    # ============================================================================
    # STEP 6: Composite score -- RelRMSE + Correlation + Model complexity
    # ============================================================================
    # WHY a composite score instead of just RMSE?
    # Pure RMSE can be dominated by outlier windows with very large labels.
    # The composite score balances three objectives:
    #
    #   1. Relative RMSE (50% weight): Scale-invariant error measure.
    #      Dividing RMSE by the range of y_true ensures that the score is
    #      comparable across datasets with different label scales.
    #
    #   2. Pearson correlation (40% weight, as 1 - corr): Measures how well
    #      the model captures the ORDERING and LINEAR RELATIONSHIP of the
    #      labels.  A model with high correlation but slightly higher RMSE
    #      may actually be more useful for the RL agent, which cares about
    #      relative spread sizing, not absolute accuracy.
    #
    #   3. Model complexity penalty (10% weight): A regularization term that
    #      penalizes configurations with deep trees, low regularization,
    #      and high subsample/colsample.  This acts as an Occam's razor,
    #      favoring simpler models when predictive performance is similar.
    #      The penalty is intentionally weak (10%) so it only breaks ties.
    #
    # Lower composite score is better.
    # ============================================================================
    def compute_composite_score(y_true, y_pred, params_sel):
        """
        Compute a composite score to rank hyperparameter configurations.

        The score combines:
          - Relative RMSE (scale-invariant)
          - (1 - Pearson correlation)
          - A small model complexity penalty

        Lower score is better.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # --- Relative RMSE ---
        # WHY relative?  Absolute RMSE depends on the scale of y.  Dividing
        # by the range of y_true makes the metric comparable across different
        # label distributions.
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        y_min, y_max = float(np.min(y_true)), float(np.max(y_true))
        y_range = y_max - y_min
        if y_range > 0:
            rel_rmse = rmse / y_range
        else:
            rel_rmse = np.nan

        # --- Pearson correlation ---
        # WHY Pearson?  It measures linear agreement between predictions and
        # actuals.  For the RL agent, what matters most is whether the model
        # correctly orders windows by expected spread (rank correlation), and
        # Pearson is a good proxy for this when the relationship is roughly
        # linear.
        if (np.std(y_true) > 0) and (np.std(y_pred) > 0):
            corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            corr = 0.0  # neutral if degenerate

        # --- Simple model-complexity penalty ---
        # WHY penalize complexity?
        # Deep trees (high max_depth) and low regularization (low reg_lambda)
        # tend to overfit to the training data.  High subsample and
        # colsample_bytree reduce bagging/feature-subsampling regularization.
        # The penalty nudges the search toward simpler models when two
        # configurations have similar predictive performance.
        md   = params_sel["max_depth"]
        lam  = params_sel["reg_lambda"]
        subs = params_sel["subsample"]
        cols = params_sel["colsample_bytree"]

        # Normalize max_depth to [0, 1] over the grid range [2, 15]
        depth_pen = max(0.0, min(1.0, (md - 2) / (15 - 2)))  # [0,1] over full grid [2..15]

        # Penalize low lambda (less regularization = more complex)
        if lam < 1.0:
            lambda_pen = max(0.0, (1.0 - lam) / 1.0)
        else:
            lambda_pen = 0.0

        # Penalize very high subsample/colsample (less stochastic regularization)
        subs_pen = max(0.0, (subs - 0.7) / 0.3) if subs > 0.7 else 0.0
        cols_pen = max(0.0, (cols - 0.7) / 0.3) if cols > 0.7 else 0.0

        model_complexity = depth_pen + lambda_pen + subs_pen + cols_pen

        # --- Composite score ---
        # Weights: 50% prediction accuracy (relRMSE), 40% ranking ability
        # (1 - corr), 10% simplicity (complexity penalty).
        rmse_component = rel_rmse if np.isfinite(rel_rmse) else 1.0
        corr_component = 1.0 - corr
        complexity_component = model_complexity

        score = (
            0.5 * rmse_component
            + 0.4 * corr_component
            + 0.1 * complexity_component
        )

        return float(score), float(rel_rmse), float(corr), float(model_complexity)

    # ============================================================================
    # STEP 7: eval_combo -- Evaluate one HP configuration with forward-chaining CV
    # ============================================================================
    # This function is the workhorse called by the parallel HP search.
    # For a given set of hyperparameters, it:
    #   1. Trains XGBoost on each forward-chaining fold's training split.
    #   2. Collects out-of-fold (OOF) predictions on each fold's test split.
    #   3. Computes the composite score on the concatenated OOF predictions.
    #
    # WHY out-of-fold (OOF) predictions?
    # ----------------------------------
    # The previous version of this code trained on the full 64% training
    # subset and evaluated on the SAME data.  This is in-sample evaluation,
    # which biases HP selection toward configurations that overfit.  By
    # collecting OOF predictions (where each sample is predicted by a model
    # that never saw it during training), we get an unbiased estimate of
    # generalization performance, leading to better HP selection.
    # ============================================================================
    def eval_combo(params_sel, folds, num_boost_round, early_stopping_rounds):
        """
        Evaluate one parameter combo with manual forward-chaining CV.

        Collects out-of-fold (OOF) predictions and uses them for composite
        scoring (RelRMSE + Corr + complexity).  The previous version trained
        on the full 64% train subset and evaluated on the same data, which
        biased HP selection toward overfitting configurations.

        IMPORTANT:
          - nthread=1 inside each worker to avoid CPU oversubscription.
            WHY?  We run max_workers_cap parallel threads, each calling
            xgb.train.  If each XGBoost instance also uses all CPU threads,
            the total thread count explodes (max_workers * nthread), causing
            severe context-switching overhead and degraded performance.
          - We do not touch the 16%+20% held-out sets here.
        """
        params = base.copy()
        params.update(params_sel)
        params["nthread"] = 1

        feature_names = list(X.columns)
        # Pre-allocate OOF prediction array (NaN for non-test indices)
        oof_preds = np.full(len(y), np.nan)
        best_rounds_per_fold = []

        for train_idx, test_idx in folds:
            # Build DMatrix for this fold's train and test splits
            d_tr = xgb.DMatrix(
                X.iloc[train_idx], label=y.iloc[train_idx],
                feature_names=feature_names,
            )
            d_te = xgb.DMatrix(
                X.iloc[test_idx], label=y.iloc[test_idx],
                feature_names=feature_names,
            )

            # Train with early stopping on this fold's validation set
            model_fold = _train_one_fold(
                params, d_tr, d_te, num_boost_round, early_stopping_rounds,
            )

            # Store OOF predictions: each test sample is predicted exactly once
            oof_preds[test_idx] = model_fold.predict(d_te)
            # Record the number of boosting rounds before early stopping triggered
            best_rounds_per_fold.append(model_fold.best_iteration + 1)

        # Median best_round across folds (robust to outlier folds).
        # WHY median instead of mean?  If one fold has an outlier best_round
        # (e.g. due to a non-stationary regime shift), the median is more
        # robust than the mean.
        best_round = int(np.median(best_rounds_per_fold))

        # Compute metrics on OOF predictions only (non-NaN indices).
        # The first fold's training indices have NaN in oof_preds because
        # they were never in a test set.  We exclude them.
        valid = ~np.isnan(oof_preds)
        y_true_oof = np.asarray(y)[valid]
        y_pred_oof = oof_preds[valid]
        rmse_cv = float(np.sqrt(mean_squared_error(y_true_oof, y_pred_oof)))

        # Composite scoring uses OOF predictions (not in-sample)
        score, rel_rmse, corr, complexity = compute_composite_score(
            y_true_oof, y_pred_oof, params_sel,
        )

        return {
            "score":        score,
            "rmse_cv":      float(rmse_cv),
            "best_round":   int(best_round),
            "params_sel":   params_sel,
            "rel_rmse":     rel_rmse,
            "corr":         corr,
            "complexity":   complexity,
        }

    # ============================================================================
    # STEP 8: PARALLEL LHS Randomized Search + Adaptive Refinement (Two Stages)
    # ============================================================================
    print(
        f"\nLHS Randomized Search (parallel): "
        f"stage1={n_stage1} (3 folds, fast) | stage2={n_stage2} (5 folds, refined)"
    )

    best = None

    # ---- Stage 1: Broad exploration (3-fold CV, shorter training) ----
    # WHY 3-fold CV?  With 500 configurations to evaluate, speed is the
    # priority.  3 folds give a rough but fast estimate of each configuration's
    # quality, sufficient to identify the top ~10% of the search space.
    # WHY 1000 rounds with early stopping at 20?  This caps training time
    # while allowing good configurations to converge.  The early-stopping
    # window of 20 rounds prevents wasting time on configurations that
    # plateau early.
    stage1_params = sample_params_smart(n_stage1, rng)
    results_stage1 = []
    max_workers = min(max_workers_cap, n_stage1)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                eval_combo,
                p,
                FOLDS_TS_STAGE1,   # 3-fold CV for speed
                1000,              # num_boost_round upper bound
                20,                # early_stopping_rounds
            )
            for p in stage1_params
        ]

        with tqdm(total=len(futures), ncols=110, desc="Stage 1 (LHS, k=3)") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                results_stage1.append(res)

                # Track the global best across all Stage 1 evaluations
                if (best is None) or (res["score"] < best["score"]):
                    best = {
                        "score":      res["score"],
                        "rmse_cv":    res["rmse_cv"],
                        "best_round": res["best_round"],
                        "params":     res["params_sel"].copy(),
                        "rel_rmse":   res["rel_rmse"],
                        "corr":       res["corr"],
                        "complexity": res["complexity"],
                    }

                pbar.set_postfix(
                    {
                        "curr_score": f"{res['score']:.4f}",
                        "rmse":       f"{res['rmse_cv']:.4f}",
                        "corr":       f"{res['corr']:.3f}",
                        "best":       f"{best['score']:.4f}",
                    }
                )
                pbar.update(1)

    # ---- Stage 2: Local refinement around the best Stage 1 anchors ----
    # WHY local refinement?
    # Stage 1 identified the best ~10% of configurations (the "anchors").
    # Stage 2 exploits this information by sampling ONLY in the discrete
    # neighborhood of these anchors (+/- 1 grid step per hyperparameter).
    # This "zoom-in" strategy is much more efficient than continuing to
    # explore the full grid randomly.

    # Sort Stage 1 results by composite score (ascending = best first)
    results_stage1_sorted = sorted(results_stage1, key=lambda r: r["score"])
    # Select the top-K anchors (at least 5, or top_quantile fraction)
    K = max(5, int(len(results_stage1_sorted) * top_quantile))
    anchors = [res["params_sel"] for res in results_stage1_sorted[:K]]

    # ---- Compute a discrete centroid anchor over the top-K candidates ----
    # WHY a centroid?
    # The top-K anchors define a "good region" in HP space.  The centroid
    # (average position) of these anchors is likely to be a strong
    # configuration itself.  By including it as an extra anchor, we ensure
    # that Stage 2 explores around this consensus point.
    def _centroid_param(anchors_list, key, options):
        """
        Compute a discrete centroid for one hyperparameter (key) over the top anchors,
        by averaging their indices in the options list and rounding.

        WHY index averaging?  The grid values may not be uniformly spaced
        (e.g. eta = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0005]).
        Averaging the raw values would be dominated by large values.
        Averaging the *indices* treats each grid level as equidistant,
        which is more appropriate for discrete search spaces.
        """
        def _safe_index(val, opts):
            try:
                return opts.index(val)
            except ValueError:
                # Fall back to nearest value in options list
                return int(np.argmin([abs(v - val) for v in opts]))
        indices = [_safe_index(a[key], options) for a in anchors_list]
        mean_idx = int(round(float(np.mean(indices))))
        mean_idx = max(0, min(mean_idx, len(options) - 1))
        return options[mean_idx]

    centroid_anchor = {
        "max_depth":        _centroid_param(anchors, "max_depth",        PARAM_GRID["max_depth"]),
        "min_child_weight": _centroid_param(anchors, "min_child_weight", PARAM_GRID["min_child_weight"]),
        "subsample":        _centroid_param(anchors, "subsample",        PARAM_GRID["subsample"]),
        "colsample_bytree": _centroid_param(anchors, "colsample_bytree", PARAM_GRID["colsample_bytree"]),
        "eta":              _centroid_param(anchors, "eta",              PARAM_GRID["eta"]),
        "reg_lambda":       _centroid_param(anchors, "reg_lambda",       PARAM_GRID["reg_lambda"]),
    }
    # Include centroid as an extra anchor to zoom around
    anchors.append(centroid_anchor)

    def _local_discrete(anchor_value, options, u_scalar):
        """
        Given an anchor discrete value and its options list, pick a neighbor
        (same or +/-1 step) using u in [0,1].

        HOW it works:
        1. Find the anchor's index in the options list.
        2. Build a list of valid neighbor indices: {anchor_idx - 1, anchor_idx, anchor_idx + 1},
           clipped to [0, len(options) - 1].
        3. Use the LHS coordinate u in [0,1] to pick one of these neighbors.

        This ensures that Stage 2 configurations are always "close" to their
        anchor in the discrete grid, implementing the local refinement strategy.
        """
        anchor_idx = options.index(anchor_value)
        neighbor_idxs = [
            idx for idx in [anchor_idx - 1, anchor_idx, anchor_idx + 1]
            if 0 <= idx < len(options)
        ]
        k = int(np.floor(u_scalar * len(neighbor_idxs)))
        if k == len(neighbor_idxs):
            k = len(neighbor_idxs) - 1
        return options[neighbor_idxs[k]]

    def _local_perturb(anchor, u):
        """
        Local neighborhood around an anchor using discrete grids.
        Each hyperparameter can move to the same value or one of its neighbors.
        """
        u0, u1, u2, u3, u4, u5 = u

        max_depth        = _local_discrete(anchor["max_depth"],        PARAM_GRID["max_depth"],        u0)
        min_child_weight = _local_discrete(anchor["min_child_weight"], PARAM_GRID["min_child_weight"], u1)
        subsample        = _local_discrete(anchor["subsample"],        PARAM_GRID["subsample"],        u2)
        colsample_bytree = _local_discrete(anchor["colsample_bytree"], PARAM_GRID["colsample_bytree"], u3)
        eta              = _local_discrete(anchor["eta"],              PARAM_GRID["eta"],              u4)
        reg_lambda       = _local_discrete(anchor["reg_lambda"],       PARAM_GRID["reg_lambda"],       u5)

        return dict(
            max_depth        = int(max_depth),
            min_child_weight = float(min_child_weight),
            subsample        = float(subsample),
            colsample_bytree = float(colsample_bytree),
            eta              = float(eta),
            reg_lambda       = float(reg_lambda),
        )

    # Distribute the Stage 2 budget evenly across all anchors.
    # Each anchor gets `per_anchor` local perturbations via LHS.
    per_anchor = max(1, n_stage2 // len(anchors))
    stage2_params = []
    for a in anchors:
        U = _lhs(per_anchor, 6, rng)
        for i in range(per_anchor):
            stage2_params.append(_local_perturb(a, U[i]))

    # Stage 2 uses 5-fold CV (more stable) and allows more boosting rounds
    # (2000 with early stopping at 50) because we are evaluating fewer
    # configurations and want higher-quality estimates.
    results_stage2 = []
    max_workers = min(max_workers_cap, len(stage2_params))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                eval_combo,
                p,
                FOLDS_TS_STAGE2,   # 5-fold CV (more stable)
                2000,              # num_boost_round upper bound
                50,                # early_stopping_rounds
            )
            for p in stage2_params
        ]

        with tqdm(total=len(futures), ncols=110, desc="Stage 2 (local, k=5)") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                results_stage2.append(res)

                # Update global best if Stage 2 found something better
                if (best is None) or (res["score"] < best["score"]):
                    best = {
                        "score":      res["score"],
                        "rmse_cv":    res["rmse_cv"],
                        "best_round": res["best_round"],
                        "params":     res["params_sel"].copy(),
                        "rel_rmse":   res["rel_rmse"],
                        "corr":       res["corr"],
                        "complexity": res["complexity"],
                    }

                pbar.set_postfix(
                    {
                        "curr_score": f"{res['score']:.4f}",
                        "rmse":       f"{res['rmse_cv']:.4f}",
                        "corr":       f"{res['corr']:.3f}",
                        "best":       f"{best['score']:.4f}",
                    }
                )
                pbar.update(1)

    # Merge the best HP from the search with the fixed base params
    best_params = base.copy()
    best_params.update(best["params"])
    best_rounds = best["best_round"]

    print(
        f"\n[Smarter Randomized Search] best params:\n{best_params}\n"
        f"Best composite score: {best['score']:.6f} "
        f"| CV RMSE={best['rmse_cv']:.6f} "
        f"| relRMSE_train={best['rel_rmse']:.6f} "
        f"| corr_train={best['corr']:.4f} "
        f"| complexity={best['complexity']:.4f} "
        f"| rounds={best_rounds}"
    )

    # ============================================================================
    # STEP 9: Train final model on 64% (train subset only)
    # ============================================================================
    # WHY train on 64% only?  This model is used for diagnostic purposes:
    # to evaluate how well the chosen HP configuration fits the training
    # data and generalizes to the held-out val (16%) and test (20%) sets.
    # The "production" model trained on 80% comes later (Step 15).
    # ============================================================================
    final_model = train_final_model(best_params, dtrain, best_rounds)
    print(f"\nFinal model trained with {best_rounds} rounds (train=64%)")

    # ============================================================================
    # STEP 10: Feature pruning (drop 3 weakest by 'gain')
    # ============================================================================
    # WHY prune features?
    # Removing the least important features can reduce overfitting and
    # speed up inference.  We use XGBoost's built-in feature importance
    # (by "gain" -- the average improvement in the loss function when a
    # feature is used in a split) to identify the 3 weakest features.
    #
    # The "reduced" model is trained with the same HP and rounds but
    # without these 3 features, providing a diagnostic comparison.
    # If the reduced model performs similarly or better on val/test, it
    # suggests that those features were adding noise rather than signal.
    # ============================================================================
    K_drop = 3
    importances = final_model.get_score(importance_type="gain")
    drop_feats = [] if len(importances) == 0 else [
        f for f, _ in sorted(importances.items(), key=lambda x: x[1])[:K_drop]
    ]
    print(f"\nDropping {K_drop} least important features: {drop_feats}")

    # ---- Tick-size rounding helper ----
    def round_to_tick(x, tick=tick_size):
        """
        Round a continuous prediction to the nearest tick.
        Use this for execution / quoting logic, not for statistical metrics.

        WHY not round during training?
        The model learns on continuous labels to preserve gradient
        information.  Rounding is applied only at inference time when the
        RL agent needs to translate predictions into discrete quote prices.
        """
        return np.round(x / tick) * tick

    # ============================================================================
    # STEP 11: Diagnostic metrics helper
    # ============================================================================
    # This helper computes a comprehensive set of regression metrics for
    # each (split, model) combination.  It prints results to stdout and
    # accumulates them in `metrics_summary` for the final summary table.
    # ============================================================================
    metrics_summary = []  # will store one row per split/model

    def compute_regression_metrics(y_true, y_pred, label: str):
        """
        Compute:
          - RMSE:       Root Mean Squared Error (absolute)
          - MAE:        Mean Absolute Error (absolute)
          - Rel. RMSE:  RMSE / range(y_true) -- scale-invariant
          - Rel. MAE:   MAE / range(y_true) -- scale-invariant
          - Pearson:    Linear correlation between predictions and actuals
          - Spearman:   Rank correlation -- measures monotonic agreement,
                        robust to non-linear relationships

        WHY both Pearson and Spearman?
        Pearson captures linear agreement; Spearman captures rank agreement.
        If Pearson is high but Spearman is low, the model captures the scale
        but not the ordering.  If Spearman is high but Pearson is low, the
        model ranks correctly but may have a non-linear calibration issue.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # RMSE
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # Relative errors (scale-invariant, based on the range of y_true)
        y_min, y_max = float(np.min(y_true)), float(np.max(y_true))
        y_range = y_max - y_min
        if y_range > 0:
            rel_rmse = float(rmse / y_range)
            rel_mae = float(mae / y_range)
        else:
            rel_rmse = np.nan
            rel_mae = np.nan

        # Pearson correlation
        if (np.std(y_true) > 0) and (np.std(y_pred) > 0):
            pearson_corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            pearson_corr = np.nan

        # Spearman rank correlation (using pandas rank + Pearson on ranks)
        # WHY this manual approach instead of scipy.stats.spearmanr?
        # Avoids an extra dependency (scipy) while producing identical results.
        s_true = pd.Series(y_true).rank(method="average").to_numpy()
        s_pred = pd.Series(y_pred).rank(method="average").to_numpy()
        if (np.std(s_true) > 0) and (np.std(s_pred) > 0):
            spearman_corr = float(np.corrcoef(s_true, s_pred)[0, 1])
        else:
            spearman_corr = np.nan

        print(
            f"[{label}] "
            f"RMSE={rmse:.6f}  MAE={mae:.6f}  "
            f"Rel.RMSE={rel_rmse:.6f}  Rel.MAE={rel_mae:.6f}  "
            f"Pearson={pearson_corr:.4f}  Spearman={spearman_corr:.4f}"
        )

        metrics_summary.append(
            dict(
                split=label,
                rmse=rmse,
                mae=mae,
                rel_rmse=rel_rmse,
                rel_mae=rel_mae,
                pearson=pearson_corr,
                spearman=spearman_corr,
            )
        )

        return rmse, mae, rel_rmse, rel_mae, pearson_corr, spearman_corr

    # ============================================================================
    # STEP 12: Retrain reduced model (same params & rounds, fewer features)
    # ============================================================================
    # This is a diagnostic model: it uses the same HP and boosting rounds
    # as the full model but drops the 3 least important features.  Comparing
    # it against the full model on val/test reveals whether those features
    # contributed meaningfully or just added noise.
    # ============================================================================
    X_reduced = X.drop(columns=drop_feats, errors="ignore")
    dtrain_reduced = xgb.DMatrix(X_reduced, label=y, feature_names=list(X_reduced.columns))
    final_model_reduced = train_final_model(best_params, dtrain_reduced, best_rounds)

    # ============================================================================
    # STEP 13: TRAIN metrics (64%) -- In-sample performance
    # ============================================================================
    # These metrics are expected to be optimistic (low error, high correlation)
    # because the model was trained on this data.  They serve as a ceiling
    # for what the model can achieve and a check for gross training failures.
    # ============================================================================
    y_pred_train_full    = final_model.predict(dtrain)
    y_pred_train_reduced = final_model_reduced.predict(dtrain_reduced)

    compute_regression_metrics(y, y_pred_train_full,    label="TRAIN 64% Full")
    compute_regression_metrics(y, y_pred_train_reduced, label="TRAIN 64% Reduced")

    # ============================================================================
    # STEP 14: VALIDATION metrics (16%) -- First out-of-sample check
    # ============================================================================
    # The validation set was not used during training or HP search (the
    # search used forward-chaining CV *within* the 64% train set).  These
    # metrics give a first estimate of real generalization performance.
    # ============================================================================
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))
    y_val_pred_full = final_model.predict(dval)

    X_val_red = X_val.drop(columns=drop_feats, errors="ignore")
    dval_red = xgb.DMatrix(X_val_red, label=y_val, feature_names=list(X_val_red.columns))
    y_val_pred_red = final_model_reduced.predict(dval_red)

    compute_regression_metrics(y_val, y_val_pred_full,  label="VAL 16% Full")
    compute_regression_metrics(y_val, y_val_pred_red,   label="VAL 16% Reduced")

    # ============================================================================
    # STEP 15: TEST metrics (20%) -- Final holdout evaluation
    # ============================================================================
    # The test set is the ultimate check.  It was never used for training,
    # HP selection, or feature pruning.  These metrics estimate how the
    # model will perform in live deployment.
    # ============================================================================
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))
    y_test_pred_full = final_model.predict(dtest)

    X_test_red = X_test.drop(columns=drop_feats, errors="ignore")
    dtest_red = xgb.DMatrix(X_test_red, label=y_test, feature_names=list(X_test_red.columns))
    y_test_pred_red = final_model_reduced.predict(dtest_red)

    compute_regression_metrics(y_test, y_test_pred_full,  label="TEST 20% Full")
    compute_regression_metrics(y_test, y_test_pred_red,   label="TEST 20% Reduced")

    # ---- Metrics summary table ----
    metrics_df = pd.DataFrame(metrics_summary).set_index("split")
    print("\n===== Metrics summary (RMSE, MAE, Rel.RMSE, Rel.MAE, Pearson, Spearman) =====")
    print(metrics_df)

    # ============================================================================
    # STEP 16: FINAL MODEL TRAINED ON TRAIN + VAL (80%) FOR DEPLOYMENT
    # ============================================================================
    # WHY retrain on 80%?
    # The 64%-trained model was used for diagnostics and feature pruning.
    # For deployment, we want to use as much data as possible to maximize
    # the model's exposure to different market regimes.  By combining
    # train (64%) + val (16%) = 80%, we get a stronger model while still
    # keeping the test set (20%) completely untouched for final evaluation.
    #
    # WHY use the same best_rounds?
    # The optimal number of boosting rounds was determined by the CV search
    # on the 64% train set.  Reusing it for the 80% model is a practical
    # approximation.  An alternative would be to re-run CV on 80%, but
    # this is computationally expensive and the difference is typically small.
    #
    # The test-set metrics for this 80%-trained model are the definitive
    # performance numbers that should be reported.
    # ============================================================================
    print("\n===================== FINAL MODEL (TRAIN + VAL = 80%) =====================")

    # Combine train + validation (already normalized by the same scaler)
    X_train80 = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train80 = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    # DMatrix
    dtrain80 = xgb.DMatrix(X_train80, label=y_train80, feature_names=list(X_train80.columns))

    # Train final model (with GPU fallback) using best_rounds
    final_model_80 = train_final_model(best_params, dtrain80, best_rounds)

    # --- Predictions for TRAIN+VAL (80%) ---
    # This is in-sample for the final model; used as a sanity check.
    y_pred_train80 = final_model_80.predict(dtrain80)
    compute_regression_metrics(
        y_train80,
        y_pred_train80,
        label="TRAIN+VAL 80% FullModel"
    )

    # --- Predictions for TEST (20%) ---
    # This is the DEFINITIVE out-of-sample evaluation.
    # The 80%-trained model has never seen any test data.
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))
    y_test_pred_80 = final_model_80.predict(dtest)

    compute_regression_metrics(
        y_test,
        y_test_pred_80,
        label="TEST 20% FullModel (Trained on 80%)"
    )

    print("\n====================== END FINAL 80% MODEL BLOCK ======================\n")

    # ============================================================================
    # Return all artifacts needed by downstream consumers (RL agent, visualizations)
    # ============================================================================
    return {
        "final_model_80": final_model_80,       # The production model (trained on 80%)
        "scaler": scaler,                        # StandardScaler fitted on train (64%) only
        "feature_cols": list(X_train.columns),   # Feature column names in training order
        "best_params": best_params,              # Full XGBoost parameter dict
        "best_rounds": int(best_rounds),         # Optimal boosting rounds from CV
        "metrics_df": metrics_df,                # Summary table of all metrics
        "drop_feats": drop_feats,                # Features dropped in the reduced model
        "best": best,                            # Full best-result dict from HP search
        # Normalized splits -- needed by downstream visualization cells.
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        # Cached predictions in original scale -- avoids re-prediction downstream.
        "predictions": {
            "y_test_pred_80": y_test_pred_80,
            "y_test_pred_red": y_test_pred_red,
            "y_pred_train_reduced": y_pred_train_reduced,
        },
    }


# ============================================================================
# SECTION 5: ARTIFACT CACHING  (Save / Load trained SGU-1 models)
# ============================================================================
# These utility functions handle serialization and deserialization of
# trained SGU-1 artifacts (model, scaler, feature columns, metadata).
# They use a content-addressable caching scheme: the cache key is a SHA-1
# hash of the training configuration metadata, so different training
# contexts (symbol, dates, delta_t, etc.) produce different cache keys.
#
# WHY cache?
# Training SGU-1 can take tens of minutes (500 + 150 HP evaluations, each
# with multi-fold CV).  Caching allows the system to skip retraining when
# the same configuration is requested again, e.g. during iterative RL
# agent development or when resuming interrupted runs.
# ============================================================================

def _sgu1_meta_config(
    *,
    symbol: str,
    base_dates,
    delta_t: int,
    exec_types,
    tick_size: float,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build a metadata dict that uniquely identifies a SGU1 training context.
    No globals: everything is passed explicitly from the runner.

    This metadata is used for:
      1. Cache key generation (via SHA-1 hash of the JSON representation).
      2. Integrity checking (comparing saved metadata with requested metadata
         to detect stale or mismatched caches).

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    base_dates : iterable of str
        Trading dates included in the training data.
    delta_t : int
        Number of TOB moves per window.
    exec_types : iterable of int
        Which message Types count as executions.
    tick_size : float
        Exchange tick size.
    extra : dict, optional
        Any additional metadata (e.g. model version, feature set version).
    """
    meta = {
        "symbol": symbol,
        "base_dates": list(base_dates),
        "delta_t": int(delta_t),
        "exec_types": list(exec_types),
        "tick_size": float(tick_size),
        # Bump feature schema so cache keys change when VWAP logic changes.
        "sgu1_feature_schema": "quoted_vwap_top_of_book_v1",
    }
    if extra:
        meta.update(extra)
    return meta


def _cache_key_from_meta(meta: dict) -> str:
    """
    Compute a short, deterministic cache key from the metadata dict.

    HOW: Serialize the dict to sorted JSON (for determinism), hash with
    SHA-1, and take the first 12 hex characters.  This gives 48 bits of
    entropy, more than sufficient to avoid accidental collisions in
    practice (collision probability < 1e-14 for up to 1000 different
    training contexts).
    """
    s = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def sgu1_artifact_paths(meta: dict, artifacts_dir: Path) -> dict:
    """
    Generate the file paths for all SGU-1 artifacts given a metadata dict.

    Returns a dict with keys:
      - "key":    The cache key string (12 hex chars).
      - "model":  Path to the XGBoost model file (.json format).
      - "scaler": Path to the joblib-serialized StandardScaler.
      - "cols":   Path to the JSON file listing feature column names.
      - "meta":   Path to the JSON file containing the full metadata dict.

    All paths are deterministic functions of the metadata content, so the
    same training context always maps to the same file names.
    """
    key = _cache_key_from_meta(meta)
    return {
        "key": key,
        "model": artifacts_dir / f"SGU1_final_model_80_{key}.json",
        "scaler": artifacts_dir / f"SGU1_scaler_{key}.joblib",
        "cols": artifacts_dir / f"SGU1_feature_cols_{key}.json",
        "meta": artifacts_dir / f"SGU1_meta_{key}.json",
    }


def try_load_sgu1(meta: dict, artifacts_dir: Path, *, strict_meta: bool = True) -> Optional[dict]:
    """
    Attempt to load a previously-saved SGU-1 model from the cache.

    Parameters
    ----------
    meta : dict
        The metadata dict describing the requested training context.
    artifacts_dir : Path
        Directory where cached artifacts are stored.
    strict_meta : bool
        If True (default), the loaded metadata must exactly match the
        requested metadata.  If False, any cache with the same key is
        accepted (useful for debugging or when minor metadata fields
        have changed).

    Returns
    -------
    dict or None
        If the cache exists and matches, returns a dict with:
          - "final_model_80": xgb.Booster
          - "scaler": StandardScaler
          - "feature_cols": list[str]
          - "meta": dict (metadata from disk)
          - "cache_key": str
        If the cache does not exist or does not match, returns None.
    """
    _require_xgboost()

    p = sgu1_artifact_paths(meta, artifacts_dir)
    # Check that ALL required artifact files exist
    if not (p["model"].exists() and p["scaler"].exists() and p["cols"].exists() and p["meta"].exists()):
        return None

    with open(p["meta"], "r", encoding="utf-8") as f:
        meta_on_disk = json.load(f)

    if strict_meta and (meta_on_disk != meta):
        # Cache exists but does not match the requested training context.
        # WHY strict?  A hash collision or a metadata change (e.g. adding
        # a new date) could lead to loading a stale model.  Strict matching
        # prevents this.
        return None

    # Load the XGBoost Booster from JSON format
    booster = xgb.Booster()
    booster.load_model(str(p["model"]))

    # Load the scikit-learn StandardScaler from joblib
    scaler = load(p["scaler"])

    # Load the feature column names (preserving training order)
    with open(p["cols"], "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return {
        "final_model_80": booster,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "meta": meta_on_disk,
        "cache_key": p["key"],
    }


def save_sgu1(out: dict, meta: dict, artifacts_dir: Path) -> None:
    """
    Save a trained SGU-1 model and its associated artifacts to disk.

    Parameters
    ----------
    out : dict
        The output dictionary from ``train_SGU1``.  Must contain:
          - "final_model_80": xgb.Booster
          - "scaler": StandardScaler
          - "feature_cols": list[str]
    meta : dict
        The metadata dict describing the training context (from
        ``_sgu1_meta_config``).
    artifacts_dir : Path
        Directory where artifacts will be saved.  Created if it does not
        exist.

    The function saves four files:
      1. The XGBoost model in JSON format (portable, human-readable).
      2. The StandardScaler via joblib (binary, fast to load).
      3. The feature column names as a JSON list.
      4. The full metadata dict as a JSON file (for cache integrity checking).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    p = sgu1_artifact_paths(meta, artifacts_dir)

    out["final_model_80"].save_model(str(p["model"]))
    dump(out["scaler"], p["scaler"])

    with open(p["cols"], "w", encoding="utf-8") as f:
        json.dump(list(out["feature_cols"]), f, indent=2)

    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
