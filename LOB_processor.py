#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 03:54:04 2025

@author: felipemoret
"""

# Utility to compute labels y_i over windows defined by top-of-book (TOB) moves.
# Assumes LOBSTER-like data where `orderbook` and `messages` share the same row index,
# i.e., each row in `orderbook` corresponds to the book state *after* the message at the same index.


import pandas as pd
import numpy as np


from typing import Tuple, List, Optional, Union


def lob_top_move(
    orderbook: pd.DataFrame,
    price_cols=("AskPrice_1", "BidPrice_1"),
    size_cols=("AskSize_1", "BidSize_1"),
    treat_nan_as_change=True,
) -> np.ndarray:
    """
    Return indices where the top-of-book (TOB) changes,
    considering both price and volume (size) changes.

    A TOB move occurs whenever the best ask/bid price OR
    the corresponding volume changes compared to the previous row.

    The first row (index 0) is always considered a boundary.
    The last element is always len(orderbook), so slicing works naturally.

    Parameters
    ----------
    orderbook : pd.DataFrame
        Must contain columns for prices and sizes (ask/bid at level 1).
    price_cols : tuple of str
        Column names for top-level ask and bid prices.
    size_cols : tuple of str
        Column names for top-level ask and bid sizes.
    treat_nan_as_change : bool
        If True, transitions INTO or OUT OF NaN are treated as TOB changes.
        Two consecutive NaNs are NOT treated as a change (the book remains
        depleted on that side, so nothing actually moved).

    Returns
    -------
    np.ndarray
        Sorted array of integer indices marking the *starts* of TOB segments,
        with the last element being len(orderbook).
    """
    n = len(orderbook)
    changed = np.zeros(n, dtype=bool)
    if n == 0:
        return np.array([0], dtype=int)  # edge case: empty DataFrame

    # The first row always starts a new TOB segment
    changed[0] = True

    if n > 1:
        diffs = []

        def col_changed(col):
            """Return a boolean array where the given column changes."""
            # FIX 3a: Cast to float to avoid TypeError on integer columns.
            # LOBSTER size columns (e.g., AskSize_1) are often stored as int64.
            # np.isnan() is undefined for integer arrays and raises TypeError.
            # Casting to float ensures NaN checks work uniformly for ALL columns,
            # whether they hold prices (float) or sizes (int).
            arr = orderbook[col].to_numpy(dtype=float)
            curr, prev = arr[1:], arr[:-1]

            # FIX 3b: Correct NaN-to-NaN double counting.
            # Under IEEE 754, NaN != NaN evaluates to True. This means the old
            # code counted two consecutive NaN rows as a "change", which is wrong:
            # if the book is depleted (NaN) on two consecutive ticks, nothing
            # actually moved — we should NOT mark that as a TOB change.
            #
            # The fix: mask out positions where BOTH curr and prev are NaN so
            # that NaN-to-NaN transitions are explicitly excluded. Only real
            # transitions INTO NaN (value -> NaN) or OUT OF NaN (NaN -> value)
            # are counted when treat_nan_as_change is True.
            nan_curr = np.isnan(curr)
            nan_prev = np.isnan(prev)
            both_nan = nan_curr & nan_prev
            # Standard != comparison: NaN != NaN is True in IEEE 754.
            # We override this: two consecutive NaNs should NOT count as a change
            # (nothing actually moved — the book is still depleted on that side).
            # Only transitions INTO or OUT OF NaN count as real TOB changes.
            out = (curr != prev) & ~both_nan
            if treat_nan_as_change:
                # XOR: exactly one side is NaN (transition to/from NaN).
                out = out | (nan_curr ^ nan_prev)
            return out

        # Detect TOB price changes
        for c in price_cols:
            if c in orderbook.columns:
                diffs.append(col_changed(c))

        # Detect TOB size (volume) changes
        for c in size_cols:
            if c in orderbook.columns:
                diffs.append(col_changed(c))

        # FIX 4: Warn when no TOB columns are found.
        # If NONE of the expected price/size columns exist in the orderbook,
        # the function would silently return only the sentinel boundaries
        # (index 0 and len(orderbook)), producing a single giant window that
        # spans the entire dataset. This is almost certainly a column-naming
        # mismatch, not the user's intent. Raising a KeyError immediately
        # surfaces the problem with an actionable error message instead of
        # letting the pipeline produce meaningless results downstream.
        if not diffs:
            missing = [c for c in list(price_cols) + list(size_cols)
                       if c not in orderbook.columns]
            raise KeyError(
                f"None of the expected TOB columns were found in the orderbook. "
                f"Missing: {missing}. Available: {list(orderbook.columns[:10])}... "
                f"Check that column names match the expected convention "
                f"(e.g., 'AskPrice_1', 'BidPrice_1', 'AskSize_1', 'BidSize_1')."
            )

        # Combine all change signals
        if diffs:
            changed[1:] = np.logical_or.reduce(diffs)
        else:
            changed[1:] = False

    # Find indices where TOB changes
    boundaries = np.flatnonzero(changed)

    # Append the final sentinel index for easier slicing
    return np.append(boundaries, n)

def top_moves_windows(orderbook: pd.DataFrame, delta_t: int) -> List[Tuple[int, int]]:
    """
    Build consecutive windows, each containing `delta_t` top-of-book (TOB) moves.

    A TOB move is defined as a change in either best bid/ask price or volume.
    The function uses `lob_top_move()` to find all such boundaries.

    Parameters
    ----------
    orderbook : pd.DataFrame
        Full limit order book snapshot per message (must contain
        columns like AskPrice_1, BidPrice_1, AskSize_1, BidSize_1).
    delta_t : int
        Number of TOB moves per window (e.g., 20 or 30).

    Returns
    -------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) pairs, representing inclusive-exclusive
        row ranges. For window k, slice `orderbook[start:end]` to get its data.
    """
    # FIX 5: Validate delta_t before proceeding.
    # delta_t < 1 is nonsensical (you cannot have a window with zero or
    # negative TOB moves) and would cause the loop below to silently produce
    # no windows or, worse, windows with inverted start/end indices.
    # Failing early with a clear message prevents hard-to-diagnose bugs
    # in downstream labelling or duration computations.
    if delta_t < 1:
        raise ValueError(f"delta_t must be >= 1, got {delta_t}")

    # Get indices where TOB changes (price or size)
    boundaries = lob_top_move(orderbook)

    windows = []
    # There are (len(boundaries) - 1) TOB segments between boundaries.
    # Each window collects `delta_t` consecutive TOB segments (i.e., `delta_t` moves).
    for i in range(0, len(boundaries) - 1 - (delta_t - 1), delta_t):
        start = boundaries[i]
        end = boundaries[i + delta_t]  # exclusive end
        windows.append((start, end))

    return windows

def compute_window_durations(messages: pd.DataFrame, windows: List[Tuple[int, int]], time_col: str = "TimeAbs") -> pd.DataFrame:
    """
    Compute the time span (in seconds) of each TOB-based window.

    Supports two time representations:
      1. **datetime** (e.g., pd.Timestamp) — duration via .total_seconds().
      2. **numeric** (e.g., float seconds-after-midnight, or monotone TimeAbs)
         — duration via simple subtraction.

    The previous version unconditionally called ``pd.to_datetime()`` on numeric
    columns, which pandas interprets as *nanoseconds since epoch*.  This caused:
      - within-day durations on the order of 1e-9 seconds (wrong scale),
      - cross-day durations that were **negative** (because raw LOBSTER Time
        resets to ~36 000 s at each midnight).

    The fix is to detect the column dtype and operate accordingly, without any
    lossy pd.to_datetime conversion on numeric data.

    Parameters
    ----------
    messages : pd.DataFrame
        Must have a timestamp column named ``time_col``.
    windows : list[tuple[int, int]]
        Output from ``top_moves_windows`` or ``top_moves_windows_by_day``.
    time_col : str
        Column name of the time information in ``messages``.

    Returns
    -------
    pd.DataFrame
        Columns: window_id, start, end, t_start, t_end, duration_sec.
    """
    # FIX 6: Handle empty windows gracefully.
    # If the caller passes an empty window list (e.g., because the orderbook
    # had fewer TOB moves than delta_t, or a particular day had no data),
    # the loop below would produce an empty `rows` list and pd.DataFrame([])
    # would return a DataFrame with zero columns — breaking any downstream
    # code that expects the standard six-column schema. Returning an
    # explicitly-typed empty DataFrame preserves the column contract.
    if not windows:
        return pd.DataFrame(
            columns=["window_id", "start", "end", "t_start", "t_end", "duration_sec"]
        )

    times_series = messages[time_col]

    # ---- Branch on dtype: datetime vs numeric ----
    if np.issubdtype(times_series.dtype, np.datetime64):
        # Datetime path: use native Timestamp arithmetic.
        times = times_series
        is_datetime = True
    else:
        # Numeric path (float seconds, e.g. TimeAbs = Time + 86400*DayID).
        # Do NOT call pd.to_datetime — just keep the numbers as-is.
        times = pd.to_numeric(times_series, errors="coerce")
        is_datetime = False

    rows = []
    for i, (start, end) in enumerate(windows):
        # FIX 2: Replace silent clamping with explicit bounds validation.
        # The old code did `last_idx = min(end - 1, len(times) - 1)`, which
        # silently masked a serious data-alignment bug: if `windows` were
        # built from one DataFrame but `messages` is a different (shorter)
        # DataFrame, the clamping would quietly pair window boundaries with
        # the WRONG timestamps, producing silently incorrect durations.
        #
        # Fail-fast: if window indices are out of bounds, that means the
        # windows were built from a different DataFrame than `messages`.
        # This is a data alignment bug that must be caught, not silently
        # papered over.
        if start < 0 or start >= len(times):
            raise IndexError(
                f"Window {i}: start={start} is out of bounds for messages "
                f"of length {len(times)}."
            )
        last_idx = end - 1
        if last_idx < 0 or last_idx >= len(times):
            raise IndexError(
                f"Window {i}: end={end} (last_idx={last_idx}) is out of bounds "
                f"for messages of length {len(times)}."
            )

        t_start = times.iloc[start]
        t_end = times.iloc[last_idx]

        if is_datetime:
            duration = (t_end - t_start).total_seconds()
        else:
            duration = float(t_end - t_start)

        rows.append({
            "window_id": i,
            "start": start,
            "end": end,
            "t_start": t_start,
            "t_end": t_end,
            "duration_sec": float(duration),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Day-safe window builder
# =============================================================================

def top_moves_windows_by_day(
    orderbook: pd.DataFrame,
    delta_t: int,
    day_ids: Union[np.ndarray, pd.Series],
) -> List[Tuple[int, int]]:
    """
    Day-safe variant of ``top_moves_windows``.

    Builds TOB-move windows **independently for each trading day**, then
    concatenates them with indices translated back to the global (concatenated)
    coordinate system.  This guarantees that no window ever crosses an
    overnight boundary.

    Why this matters
    ----------------
    LOBSTER data is organized as one file pair per trading day.  When multiple
    days are concatenated into a single DataFrame, ``top_moves_windows`` has no
    awareness of day boundaries.  A window that starts near the close of Day N
    and ends at the open of Day N+1 will:

      * mix two independent trading sessions in one window,
      * produce misleading labels (realized range / trend over the gap),
      * yield negative durations when using raw Time (which resets at midnight),
        or ~18-hour durations when using TimeAbs.

    Parameters
    ----------
    orderbook : pd.DataFrame
        Full concatenated orderbook (all days stacked).
    delta_t : int
        Number of TOB moves per window (same semantics as ``top_moves_windows``).
    day_ids : array-like of int
        Per-row day identifier, aligned with ``orderbook``.  Rows that share
        the same ``day_ids`` value are treated as belonging to the same
        continuous trading session.  Typical sources:

        * ``messages["DayID"]`` from ``LOB_data`` class,
        * a manually assigned integer per date in a runner loop,
        * ``messages["time"].dt.date`` factorized to ints.

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) pairs with **global** indices into the
        concatenated orderbook / messages DataFrames — the same interface as
        ``top_moves_windows``, so all downstream code works unchanged.
    """
    day_ids_arr = np.asarray(day_ids)

    windows: List[Tuple[int, int]] = []

    for day_val in sorted(np.unique(day_ids_arr)):
        # --- 1. Identify rows belonging to this trading day ---
        day_positions = np.flatnonzero(day_ids_arr == day_val)
        if len(day_positions) == 0:
            continue

        # Global offset: the first row of this day in the concatenated DF.
        offset = int(day_positions[0])

        # FIX 1: Validate that this day's rows are contiguous in the concatenated DF.
        # The offset-based translation below (local_idx + offset) relies on the
        # assumption that all rows for a given day form a CONTIGUOUS block in the
        # concatenated DataFrame. If rows were filtered, shuffled, or sorted in a
        # way that scatters a day's rows across non-adjacent positions, the simple
        # "local index + offset" arithmetic maps to WRONG global indices — and
        # the error is completely silent (no exception, just wrong results).
        #
        # This check catches that scenario early with an actionable error message,
        # telling the user to reset the index after any filtering or reordering.
        expected_positions = np.arange(offset, offset + len(day_positions))
        if not np.array_equal(day_positions, expected_positions):
            raise ValueError(
                f"Day {day_val}: rows are not contiguous in the DataFrame. "
                f"Expected positions [{offset}..{offset + len(day_positions) - 1}], "
                f"but found gaps or re-ordering. Ensure the DataFrame has a clean "
                f"RangeIndex (call .reset_index(drop=True) after any filtering)."
            )

        # --- 2. Extract the orderbook for this day only ---
        # reset_index(drop=True) so that lob_top_move works with 0-based idx.
        ob_day = orderbook.iloc[day_positions].reset_index(drop=True)

        # --- 3. Build windows within this single day ---
        # Only complete windows (exactly delta_t TOB moves) are created;
        # the trailing partial window at end-of-day is discarded.
        day_windows = top_moves_windows(ob_day, delta_t=delta_t)

        # --- 4. Translate local indices back to global coordinates ---
        for local_start, local_end in day_windows:
            windows.append((local_start + offset, local_end + offset))

    return windows
