#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MM_policy_8.py — Always Fixed Offset from L1 Market-Making Policy (Optimized)

This module implements a Market-Making (MM) strategy that places bid and ask
quotes at a fixed tick offset from the Level-1 (L1) best bid and ask prices.

Architecture:
    The policy is built using the "factory + closure" pattern:
    - `always_fixed_offset_from_l1_mm_policy_factory(...)` is the factory.
    - It returns a `policy(state, mm)` function that the simulator calls on
      every event (order book update, fill, timer tick, etc.).
    - Internal mutable state lives in the closure dict `st`, persisting across
      calls without global variables or class instances.

Quoting Reference:
    bid = best_bid - offset_ticks
    ask = best_ask + offset_ticks

    offset_ticks > 0  ->  more passive (quotes sit deeper than L1)
    offset_ticks = 0  ->  quote exactly at L1
    offset_ticks < 0  ->  improve inside the spread (symmetric, auto-clamped)

Inventory Control:
    The policy enforces hard inventory limits. When the position reaches
    +inv_limit, it switches to ask_only mode (only sell to reduce exposure).
    When it reaches -inv_limit, it switches to bid_only mode (only buy).

Throttling:
    Three independent throttle mechanisms can be enabled to reduce message
    traffic and quote flickering:
    1. TOB (Top-of-Book) moves -- react after N market structure changes
    2. Event steps           -- react after N simulation events
    3. Time interval         -- react after T seconds of simulation time

    When multiple throttles are enabled, the policy uses OR-blocking
    semantics (any gate blocking = throttled). The policy only acts when
    ALL enabled gates are satisfied simultaneously. This matches the
    DQN controller's throttle behavior.

Decision Hierarchy (4 Gates):
    Gate 1 (CRITICAL): Mode changes and forbidden legs -- always immediate.
    Gate 2 (URGENT):   Fill detected -- repost quickly, preserve unfilled side.
    Gate 3 (THROTTLE): Hard throttle -- block refresh unless legs are missing.
    Gate 4 (NORMAL):   Routine refresh -- update only the side(s) that changed.

Created on Sun Jan 25 14:42:34 2026
@author: felipemoret
"""

from typing import Dict, Any, Tuple, Optional
import math


def always_fixed_offset_from_l1_mm_policy_factory(
    offset_ticks: int = 1,
    inv_limit: int = 10,

    # --- Throttling: TOB (Market Structure Moves) ---
    use_tob_update: bool = False,
    n_tob_moves: int = 10,

    # --- Throttling: Simulation Steps (Events) ---
    use_event_update: bool = False,
    n_events: int = 100,

    # --- Throttling: Simulation Time (Seconds) ---
    use_time_update: bool = False,
    min_time_interval: float = 1.0,

    # --- Refresh behavior ---
    # WARNING: Keeping this True kills queue priority on quiet markets,
    # because it forces cancel+replace on every refresh window even when
    # prices haven't moved. Only enable for debugging or when you need
    # guaranteed order freshness (e.g., exchange session timeouts).
    # Recommended: False
    refresh_if_unchanged: bool = False,
):
    """
    Factory that creates an Always-Fixed-Offset-from-L1 Market-Making Policy.

    This factory validates parameters, initializes internal state, and returns
    a ``policy`` function that the simulator will call on every event. The
    returned policy carries its own mutable state via closure variables,
    so multiple independent instances can coexist without interference.

    Parameters
    ----------
    offset_ticks : int
        Number of ticks to offset from L1 best bid/ask.
        Positive -> more passive (deeper), Negative -> inside spread
        improvement (clamped to avoid crossing), Zero -> quote at L1 exactly.

    inv_limit : int
        Maximum absolute inventory before switching to one-sided quoting.
        Must be >= 1.

    use_tob_update : bool
        Enable throttle based on Top-of-Book price changes.

    n_tob_moves : int
        Number of TOB price changes required before allowing a refresh.
        Only used if ``use_tob_update`` is True.

    use_event_update : bool
        Enable throttle based on simulation event count.

    n_events : int
        Number of simulation events required before allowing a refresh.
        Only used if ``use_event_update`` is True.

    use_time_update : bool
        Enable throttle based on elapsed simulation time.

    min_time_interval : float
        Minimum seconds between refreshes. Only used if ``use_time_update``
        is True. Must be > 0.

    refresh_if_unchanged : bool
        If True, cancel+replace orders even when computed prices haven't
        changed. This is almost never desirable because it destroys queue
        priority. Default: False.

    Returns
    -------
    policy : callable
        Function with signature ``policy(state: Dict, mm=None) -> Tuple``
        that returns an action tuple for the simulator.
    """

    # =========================================================================
    # 1) Parameter Validation
    # =========================================================================
    if int(inv_limit) < 1:
        raise ValueError("inv_limit must be >= 1")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")

    # Pre-compute validated thresholds. We enforce a minimum of 1 to avoid
    # degenerate behavior where threshold=0 means "always satisfied" (which
    # would effectively disable the throttle rather than making it stricter).
    threshold_tob = max(1, int(n_tob_moves))
    threshold_events = max(1, int(n_events))
    base_offset = int(offset_ticks)

    print("-" * 70)
    print("[Fixed Offset from L1 - Optimized] Initialized:")
    print(f"  offset_ticks={base_offset} | inv_limit={inv_limit} "
          f"| refresh_unchanged={refresh_if_unchanged}")
    print("  ref: bid=best_bid-offset, ask=best_ask+offset "
          "(offset<0 => inside, clamped)")
    print("-" * 70)

    # =========================================================================
    # 2) Internal Mutable State (Closure Variables)
    #
    # This dictionary persists across calls to ``policy()``. It tracks:
    #   - What prices we last quoted (for diff detection in Gate 4)
    #   - What mode we were in (for mode-change detection in Gate 1)
    #   - Throttle counters (TOB moves, event steps, last update time)
    #   - Fill accounting (for statistics and Gate 2 logic)
    #   - Order presence tracking (for simultaneous fill detection)
    # =========================================================================
    st = {
        # --- Last quoted prices (in grid-index space) ---
        # None means "no order on this side" (used in one-sided modes).
        "last_bid_px": None,
        "last_ask_px": None,

        # --- Last quoting mode ---
        # One of: "two_sided", "bid_only", "ask_only", or None (initial).
        "last_mode": None,

        # --- Throttle state ---
        # TOB throttle: tracks the last observed (bid_price, ask_price) key
        # and counts how many times it changed since the last refresh.
        "last_env_tob_key": None,
        "moves_tob": 0,
        # Event throttle: counts simulation steps since the last refresh.
        "event_steps": 0,
        # Time throttle: records the simulation timestamp of the last refresh.
        "last_update_time": -1.0,

        # --- Fill tracking ---
        # last_inv: inventory at the previous call (for delta computation).
        "last_inv": None,
        # last_has_bid / last_has_ask: order presence at the previous call.
        # Used as a secondary signal to detect simultaneous opposite-side
        # fills where the inventory delta cancels to zero.
        "last_has_bid": False,
        "last_has_ask": False,
        # Cumulative fill counters (for statistics / debugging only).
        "n_fills": 0,
        "n_fills_bid": 0,
        "n_fills_ask": 0,
    }

    # Track grid shifts: when the simulator re-centers the price grid around
    # a new base index, all grid-relative prices shift. We must adjust our
    # memorized prices (last_bid_px, last_ask_px) by the same amount to keep
    # them consistent with the new grid coordinates.
    # Stored as a one-element list so the inner function can mutate it
    # (Python closures can rebind lists but not plain scalars).
    prev_base_idx = [None]

    # =========================================================================
    # 3) Helper Functions
    # =========================================================================

    def _safe_int(x, d=-1):
        """
        Safely convert ``x`` to int. Returns ``d`` if conversion fails.

        This guards against edge cases where the simulator might pass
        None, NaN, or string values in the state dictionary.
        """
        try:
            return int(x)
        except Exception:
            return d

    def _safe_float(x, d=0.0):
        """
        Safely convert ``x`` to float. Returns ``d`` if conversion fails
        or the result is not finite (inf, NaN).
        """
        try:
            v = float(x)
            return v if math.isfinite(v) else d
        except Exception:
            return d

    def _get_env_tob(s):
        """
        Build a TOB (Top-of-Book) fingerprint key from the state dictionary.

        This key is used to detect market structure changes. A "TOB change"
        is defined as any change in the best bid/ask prices OR in the
        queue sizes at those levels. Both price moves and size changes
        (e.g., new orders joining or leaving the queue) count as TOB events.

        We prefer the ``_env`` variants of best_bid/ask, which represent the
        external market state (excluding the MM's own orders). If ``_env``
        keys are not available in the state dict, we fall back to the
        standard keys as a best-effort approximation.

        Returns
        -------
        tuple of (int, int, int, int)
            ``(best_bid_price, best_ask_price, bid_size, ask_size)``
            as seen by the external market. Used as a hashable key for
            change detection.
        """
        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        return (env_bb, env_ba, int(round(env_bs)), int(round(env_as)))

    def _consume_fill(inv: int, has_bid: bool, has_ask: bool):
        """
        Detect fills by combining inventory delta with order presence tracking.

        The primary signal is the inventory delta (change since last call):
          delta > 0  ->  net buying occurred (bid side was filled)
          delta < 0  ->  net selling occurred (ask side was filled)

        The secondary signal uses order presence (``has_bid``, ``has_ask``) to
        detect simultaneous opposite-side fills that cancel out in the delta:
          delta == 0 BUT we had a bid that disappeared -> bid likely filled
          delta == 0 BUT we had an ask that disappeared -> ask likely filled

        This two-signal approach fixes the edge case where both bid and ask
        fill between consecutive policy calls, producing delta=0 and hiding
        the fills entirely from a delta-only detector.

        Parameters
        ----------
        inv : int
            Current inventory.
        has_bid : bool
            Whether we currently have an active bid order on the book.
        has_ask : bool
            Whether we currently have an active ask order on the book.

        Returns
        -------
        tuple of (bool, int, int)
            ``(new_fill, bid_fills, ask_fills)``

            - ``new_fill``:  True if any fill was detected on this call.
            - ``bid_fills``: Estimated number of bid-side fills (units bought).
            - ``ask_fills``: Estimated number of ask-side fills (units sold).
        """
        # First call ever: initialize tracking state, report no fill.
        if st["last_inv"] is None:
            st["last_inv"] = int(inv)
            st["last_has_bid"] = has_bid
            st["last_has_ask"] = has_ask
            return False, 0, 0

        delta = int(inv) - int(st["last_inv"])
        had_bid = st["last_has_bid"]
        had_ask = st["last_has_ask"]

        # Update tracking state for the next call.
        st["last_inv"] = int(inv)
        st["last_has_bid"] = has_bid
        st["last_has_ask"] = has_ask

        # --- Primary detection: inventory delta ---
        # A positive delta means we bought (bid filled).
        # A negative delta means we sold (ask filled).
        bid_fills = max(0, delta)
        ask_fills = max(0, -delta)

        # --- Secondary detection: order disappearance with zero net delta ---
        # This catches the edge case where bid and ask fill simultaneously,
        # canceling each other out in the delta. We infer fills from the
        # fact that we had an active order on a side and it's now gone.
        if delta == 0:
            if had_bid and not has_bid:
                bid_fills = max(bid_fills, 1)
            if had_ask and not has_ask:
                ask_fills = max(ask_fills, 1)

        total = bid_fills + ask_fills
        if total == 0:
            return False, 0, 0

        return True, bid_fills, ask_fills

    def _update_tob_clock(key):
        """
        Increment the TOB move counter if the market structure fingerprint
        has changed since the last observation.

        This drives the TOB-based throttle: each time the external best
        bid or best ask price changes, we count it as one "move". The
        policy becomes eligible for a refresh once ``moves_tob`` reaches
        ``threshold_tob``.
        """
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s, current_time: float):
        """
        Reset all throttle clocks after taking an action.

        This is called whenever the policy decides to place, cancel, or
        refresh orders. Resetting the clocks ensures that the next action
        waits for the full throttle interval before firing again.

        Parameters
        ----------
        s : dict
            The current state dictionary (used to snapshot the TOB key).
        current_time : float
            The current simulation time in seconds.
        """
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        if use_event_update:
            st["event_steps"] = 0
        if use_time_update:
            st["last_update_time"] = current_time

    def _effective_offset_for_no_cross(bb: int, ba: int, off: int) -> int:
        """
        Clamp negative offsets to guarantee that bid < ask (no crossing).

        Positive offsets are inherently safe because they push quotes further
        apart from each other (more passive):
          bid = bb - offset (goes DOWN)    ask = ba + offset (goes UP)

        For negative offsets (inside-spread improvement), the quotes move
        TOWARD each other:
          bid = bb + k    (k = -off > 0, improving the bid upward)
          ask = ba - k    (improving the ask downward)

        For the quotes not to cross, we need:
          bb + k < ba - k   =>   2k < spread   =>   k <= floor((spread-1)/2)

        Examples illustrating the clamping behavior:
          spread=1 -> k_max=0 -> cannot improve safely, effective offset = 0
          spread=2 -> k_max=0 -> cannot improve safely, effective offset = 0
          spread=3 -> k_max=1 -> can improve by 1 tick on each side
          spread=4 -> k_max=1 -> can improve by 1 tick (conservative)
          spread=5 -> k_max=2 -> can improve by 2 ticks on each side

        Parameters
        ----------
        bb : int
            Best bid price (grid index).
        ba : int
            Best ask price (grid index).
        off : int
            Desired offset (can be negative for inside-spread improvement).

        Returns
        -------
        int
            Effective offset, clamped so that ``bid < ask`` is guaranteed
            when both sides are placed.
        """
        spread = int(ba - bb)
        if spread <= 0:
            # Pathological case: locked or crossed market. Return 0 and let
            # the caller handle it (typically by returning hold).
            return 0
        if off >= 0:
            return off

        # Negative offset: compute the maximum safe improvement depth.
        k_req = int(-off)
        k_max = max(0, (spread - 1) // 2)
        k = min(k_req, k_max)
        return -k

    def _target_quotes(bb: int, ba: int, desired_mode: str
                       ) -> Tuple[Optional[int], Optional[int]]:
        """
        Compute the target bid and ask prices based on L1 reference prices
        and the current quoting mode.

        The target prices are computed as:
          bid = best_bid - effective_offset
          ask = best_ask + effective_offset

        where ``effective_offset`` is the ``base_offset`` after no-cross
        clamping via ``_effective_offset_for_no_cross``.

        Additional safety checks are applied for one-sided modes to prevent
        the remaining quote from crossing into the opposite side of the book:
          bid_only: bid must be <= best_ask - 1 (don't lift the offer)
          ask_only: ask must be >= best_bid + 1 (don't hit the bid)

        Parameters
        ----------
        bb : int
            Current best bid price (grid index).
        ba : int
            Current best ask price (grid index).
        desired_mode : str
            One of ``"two_sided"``, ``"bid_only"``, ``"ask_only"``.

        Returns
        -------
        tuple of (Optional[int], Optional[int])
            ``(target_bid_price, target_ask_price)``.
            Returns ``(None, None)`` if the book is locked/crossed and
            quoting would be unsafe.
        """
        # If the book is locked (bb == ba) or crossed (bb > ba), it is
        # unsafe to place quotes. Signal the caller to hold.
        if bb >= ba:
            return None, None

        off_eff = _effective_offset_for_no_cross(bb, ba, base_offset)

        # Compute L1-referenced target prices.
        bid_px = bb - off_eff
        ask_px = ba + off_eff

        # For one-sided modes, clamp the remaining quote so it cannot
        # cross into the opposite side of the book:
        #   bid_only: our bid must stay below the best ask (don't lift)
        #   ask_only: our ask must stay above the best bid (don't hit)
        if desired_mode == "bid_only":
            bid_px = min(bid_px, ba - 1)
        elif desired_mode == "ask_only":
            ask_px = max(ask_px, bb + 1)

        return int(bid_px), int(ask_px)

    # =========================================================================
    # 4) MAIN POLICY FUNCTION
    #
    # This is the function returned by the factory and called by the simulator
    # on every event. It follows a strict priority-based decision hierarchy
    # (4 Gates) to determine the next action.
    #
    # The function signature supports two calling conventions:
    #   policy(state)        -- basic (no access to MM internals)
    #   policy(state, mm)    -- advanced (can read mm.base_price_idx for
    #                           grid shift detection)
    #
    # Return value: a tuple whose first element is the action name, followed
    # by action-specific parameters:
    #   ("hold",)                              -- do nothing this step
    #   ("place_bid_ask", bid_px, ask_px)      -- place/replace both sides
    #   ("cancel_bid_then_place", bid_px)      -- cancel bid, place new bid
    #   ("cancel_ask_then_place", ask_px)      -- cancel ask, place new ask
    #   ("cancel_all_then_place", side, price) -- cancel all orders, then
    #                                             place one side only;
    #                                             side: +1 (bid) or -1 (ask)
    # =========================================================================
    def policy(state: Dict[str, Any], mm=None) -> Tuple:

        # =================================================================
        # A) Time & Grid-Shift Detection
        #
        # The simulator may shift the price grid to keep the midpoint
        # centered (common in long-running simulations where the price
        # drifts far from the initial center). When this happens, all
        # grid indices change by a constant offset. We detect this by
        # monitoring mm.base_price_idx and adjust our memorized quote
        # prices accordingly, so that diff detection in Gate 4 remains
        # accurate.
        #
        # Without this adjustment, a grid shift would make the policy
        # think its quotes moved (because the old indices now point to
        # different prices), triggering unnecessary cancel+replace.
        # =================================================================
        current_time = _safe_float(state.get("time", 0.0))

        shift_happened = False
        if mm is not None and hasattr(mm, "base_price_idx"):
            try:
                cur_base = int(mm.base_price_idx)
            except Exception:
                cur_base = None

            if cur_base is not None:
                if prev_base_idx[0] is None:
                    # First call: just record the initial base index.
                    prev_base_idx[0] = cur_base
                else:
                    base_shift = cur_base - prev_base_idx[0]
                    if base_shift != 0:
                        shift_happened = True
                        # Adjust memorized prices to the new grid coordinates.
                        # If base shifted UP by N, the same absolute price is
                        # now at index (old_idx - N) in the new grid.
                        if st["last_bid_px"] is not None:
                            st["last_bid_px"] -= base_shift
                        if st["last_ask_px"] is not None:
                            st["last_ask_px"] -= base_shift
                        prev_base_idx[0] = cur_base

        # =================================================================
        # B) Update Throttle Counters
        #
        # These counters accumulate "activity" between refreshes. Each
        # throttle dimension has its own counter that resets when we act.
        # The counters are always updated, even if the corresponding
        # throttle is disabled, to allow runtime toggling.
        # =================================================================
        if use_event_update:
            st["event_steps"] += 1

        if use_tob_update:
            key = _get_env_tob(state)
            if shift_happened or st["last_env_tob_key"] is None:
                # After a grid shift, re-anchor the TOB key without counting
                # it as a market move (the grid moved, not the market).
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # =================================================================
        # C) Read Current State from the Simulator
        # =================================================================
        I_t = _safe_int(state.get("inventory", 0))
        has_bid = bool(state.get("has_bid", False))
        has_ask = bool(state.get("has_ask", False))
        mkt_bb = _safe_int(state.get("best_bid", -1))
        mkt_ba = _safe_int(state.get("best_ask", -1))

        # If the market is missing one or both sides (indicated by the
        # sentinel value -1), we cannot compute meaningful target quotes.
        # Hold and wait for the book to recover.
        if mkt_bb < 0 or mkt_ba < 0:
            return ("hold",)

        # =================================================================
        # D) Determine Desired Quoting Mode Based on Inventory
        #
        # The mode controls which sides of the book we quote on:
        #   two_sided : normal operation, quote both bid and ask
        #   ask_only  : inventory at +limit, stop buying, only sell
        #   bid_only  : inventory at -limit, stop selling, only buy
        #
        # This is a simple hard-limit inventory control. More sophisticated
        # policies (GLFT, Avellaneda-Stoikov) use continuous skewing instead.
        # =================================================================
        if I_t >= inv_limit:
            desired_mode = "ask_only"
        elif I_t <= -inv_limit:
            desired_mode = "bid_only"
        else:
            desired_mode = "two_sided"

        mode_changed = (st["last_mode"] != desired_mode)

        # =================================================================
        # E) Fill Detection
        #
        # Detect fills using two complementary signals:
        #   1. Inventory delta (primary): delta>0 -> bought, delta<0 -> sold
        #   2. Order disappearance (secondary): handles simultaneous fills
        #      where the delta cancels to zero but orders are gone
        #
        # The results drive Gate 2 (URGENT) and update fill counters.
        # =================================================================
        new_fill, bid_fills, ask_fills = _consume_fill(I_t, has_bid, has_ask)
        if new_fill:
            st["n_fills"] += (bid_fills + ask_fills)
            st["n_fills_bid"] += bid_fills
            st["n_fills_ask"] += ask_fills

        # =================================================================
        # F) Compute Missing and Forbidden Legs
        #
        # "Missing" = we SHOULD have an order on this side but don't.
        #   This can happen at startup, after a fill consumed the order,
        #   or if the exchange cancelled it (e.g., self-trade prevention).
        #
        # "Forbidden" = we HAVE an order on a side that our current mode
        #   says we should NOT be quoting (e.g., a bid order while in
        #   ask_only mode). Must be cancelled immediately in Gate 1.
        # =================================================================
        missing_bid = (not has_bid) and (desired_mode in ("two_sided", "bid_only"))
        missing_ask = (not has_ask) and (desired_mode in ("two_sided", "ask_only"))
        forbidden_bid = has_bid and (desired_mode == "ask_only")
        forbidden_ask = has_ask and (desired_mode == "bid_only")

        # =================================================================
        # G) Compute Target Quote Prices
        #
        # These are the prices we WANT to be quoting at, given the current
        # L1 prices and our offset parameter. The actual decision to act
        # (or hold) is made by the gates below.
        # =================================================================
        tgt_bid, tgt_ask = _target_quotes(mkt_bb, mkt_ba, desired_mode)

        # If we can't compute valid targets (locked/crossed book, or spread
        # too tight for our offset), hold and wait.
        if desired_mode == "two_sided" and (tgt_bid is None or tgt_ask is None):
            return ("hold",)
        if desired_mode == "bid_only" and tgt_bid is None:
            return ("hold",)
        if desired_mode == "ask_only" and tgt_ask is None:
            return ("hold",)

        # =================================================================
        #                    EXECUTION HIERARCHY (4 Gates)
        #
        # Gates are checked in strict priority order. Once a gate triggers,
        # we execute its action and return immediately, skipping all
        # lower-priority gates.
        # =================================================================

        # -------------------------------------------------------------
        # GATE 1: CRITICAL — Mode Change or Forbidden Legs
        #
        # Situations requiring IMMEDIATE action, regardless of throttles:
        #   - The quoting mode changed (e.g., two_sided -> ask_only
        #     because inventory hit the limit). We must cancel the
        #     forbidden side and repost the correct side(s).
        #   - We have a "forbidden leg" (an order on a side we should
        #     no longer be quoting). Must cancel immediately to stop
        #     accumulating inventory in the wrong direction.
        #
        # Action: cancel everything and repost the correct side(s).
        # -------------------------------------------------------------
        if mode_changed or forbidden_bid or forbidden_ask:
            st["last_mode"] = desired_mode
            _reset_all_clocks(state, current_time)

            if desired_mode == "two_sided":
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                return ("place_bid_ask", tgt_bid, tgt_ask)
            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, None
                return ("cancel_all_then_place", +1, tgt_bid)
            else:  # ask_only
                st["last_bid_px"], st["last_ask_px"] = None, tgt_ask
                return ("cancel_all_then_place", -1, tgt_ask)

        # -------------------------------------------------------------
        # GATE 2: URGENT — Fill Detected
        #
        # When a fill occurs, we need to repost the filled side promptly
        # to maintain continuous liquidity provision.
        #
        # Optimization: we try to preserve queue priority on the
        # non-filled side by only canceling/replacing the side that was
        # actually filled. Queue position is valuable in a price-time
        # priority book — canceling an order that didn't need to move
        # puts us at the back of the queue for free.
        #
        # Fill side is determined by the _consume_fill() helper, which
        # uses both inventory delta and order presence tracking.
        # -------------------------------------------------------------
        if new_fill:
            _reset_all_clocks(state, current_time)

            bid_ok = (st["last_bid_px"] == tgt_bid)
            ask_ok = (st["last_ask_px"] == tgt_ask)

            if desired_mode == "two_sided":
                # Case 1: Both sides are at wrong prices -> refresh both.
                if (not bid_ok) and (not ask_ok):
                    st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                    return ("place_bid_ask", tgt_bid, tgt_ask)

                # Case 2: Only bid filled, ask is still at correct price
                #         -> replace bid only, preserve ask's queue position.
                if bid_fills > 0 and ask_fills == 0 and ask_ok:
                    st["last_bid_px"] = tgt_bid
                    return ("cancel_bid_then_place", tgt_bid)

                # Case 3: Only ask filled, bid is still at correct price
                #         -> replace ask only, preserve bid's queue position.
                if ask_fills > 0 and bid_fills == 0 and bid_ok:
                    st["last_ask_px"] = tgt_ask
                    return ("cancel_ask_then_place", tgt_ask)

                # Case 4: Fallback — both sides filled, or some other
                #         combination we can't handle surgically.
                #         Refresh both sides.
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                return ("place_bid_ask", tgt_bid, tgt_ask)

            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, None
                return ("cancel_all_then_place", +1, tgt_bid)
            else:  # ask_only
                st["last_bid_px"], st["last_ask_px"] = None, tgt_ask
                return ("cancel_all_then_place", -1, tgt_ask)

        # -------------------------------------------------------------
        # GATE 3: HARD THROTTLE
        #
        # When throttles are enabled, we suppress routine refreshes
        # until enough "activity" has accumulated. This reduces message
        # traffic and prevents quote flickering on noisy markets.
        #
        # Throttle semantics (OR-blocking / AND-to-unblock):
        #   Any enabled gate whose threshold is NOT yet met will
        #   block the policy. The policy is only unthrottled when
        #   ALL enabled gates are satisfied simultaneously.
        #   This is consistent with the DQN controller's throttle
        #   semantics: "any gate blocking = agent is throttled."
        #
        # Bypass: if we are missing a required leg (e.g., our bid
        #   disappeared due to a fill or external cancel), we bypass
        #   the throttle to repost immediately. Being unquoted on a
        #   required side is worse than sending an extra message.
        # -------------------------------------------------------------
        is_throttled = False

        if use_event_update and (st["event_steps"] < threshold_events):
            is_throttled = True
        if use_time_update:
            if st["last_update_time"] < 0:
                pass  # First call after init: don't throttle.
            elif (current_time - st["last_update_time"]) < min_time_interval:
                is_throttled = True
        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_throttled = True

        if is_throttled:
            # Bypass throttle only if we are missing a required leg.
            if not (missing_bid or missing_ask):
                return ("hold",)

        # -------------------------------------------------------------
        # GATE 4: NORMAL REFRESH WINDOW
        #
        # At this point, all throttle conditions are satisfied (or no
        # throttles are enabled). We check whether our current quotes
        # need updating and act accordingly.
        #
        # Key design principle: only update the side(s) whose target
        # price actually changed. This preserves queue priority on the
        # unchanged side, which is critical for profitability in a
        # price-time priority order book.
        #
        # Mode-scoped diff checks: in one-sided modes, we ONLY check
        # the diff for the ACTIVE side. The inactive side's last_*_px
        # is intentionally set to None (meaning "no order"), which
        # would always differ from the computed target (an int),
        # causing spurious cancel+replace on every call. By scoping
        # the diff to the active side, we avoid this and correctly
        # hold when the active side's price is unchanged.
        # -------------------------------------------------------------

        if desired_mode == "two_sided":
            diff_bid = (st["last_bid_px"] != tgt_bid)
            diff_ask = (st["last_ask_px"] != tgt_ask)

            # If nothing changed and nothing is missing, hold. This
            # preserves queue priority — the most important optimization
            # for a passive market maker.
            if ((not refresh_if_unchanged) and (not diff_bid)
                    and (not diff_ask) and (not missing_bid)
                    and (not missing_ask)):
                _reset_all_clocks(state, current_time)
                return ("hold",)

            _reset_all_clocks(state, current_time)

            # Both sides changed or both missing -> refresh both.
            if (diff_bid and diff_ask) or (missing_bid and missing_ask):
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                return ("place_bid_ask", tgt_bid, tgt_ask)

            # Only bid changed or missing -> update bid only.
            if diff_bid or missing_bid:
                st["last_bid_px"] = tgt_bid
                return ("cancel_bid_then_place", tgt_bid)

            # Only ask changed or missing -> update ask only.
            if diff_ask or missing_ask:
                st["last_ask_px"] = tgt_ask
                return ("cancel_ask_then_place", tgt_ask)

            # Fallback: refresh_if_unchanged=True and nothing changed.
            # Refresh both sides as requested (at the cost of queue priority).
            st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
            return ("place_bid_ask", tgt_bid, tgt_ask)

        elif desired_mode == "bid_only":
            # In bid_only mode, we only care about the bid side.
            # The ask side is not quoted (last_ask_px = None), so we
            # ignore tgt_ask entirely to avoid false diffs.
            diff_bid = (st["last_bid_px"] != tgt_bid)

            if (not refresh_if_unchanged) and (not diff_bid) and (not missing_bid):
                _reset_all_clocks(state, current_time)
                return ("hold",)

            _reset_all_clocks(state, current_time)
            st["last_bid_px"], st["last_ask_px"] = tgt_bid, None
            return ("cancel_all_then_place", +1, tgt_bid)

        else:  # ask_only
            # In ask_only mode, we only care about the ask side.
            # The bid side is not quoted (last_bid_px = None), so we
            # ignore tgt_bid entirely to avoid false diffs.
            diff_ask = (st["last_ask_px"] != tgt_ask)

            if (not refresh_if_unchanged) and (not diff_ask) and (not missing_ask):
                _reset_all_clocks(state, current_time)
                return ("hold",)

            _reset_all_clocks(state, current_time)
            st["last_bid_px"], st["last_ask_px"] = None, tgt_ask
            return ("cancel_all_then_place", -1, tgt_ask)

    # =========================================================================
    # 5) Debug / Statistics Accessors
    #
    # These utility functions are attached as attributes on the ``policy``
    # function object, so the caller can inspect internal state without
    # modifying the function signature or the simulator's calling convention.
    #
    # Usage:
    #   policy.get_n_fills()              -> total fill count (int)
    #   policy.print_stats("MyPolicy")    -> print one-line status summary
    #   policy._debug_state               -> raw internal state dict
    # =========================================================================
    def get_n_fills() -> int:
        """Return the total number of fills detected since initialization."""
        return int(st["n_fills"])

    def print_stats(prefix: str = "AlwaysFixedOffsetFromL1"):
        """Print a one-line summary of internal state for debugging."""
        print(
            f"{prefix}: fills={st['n_fills']} "
            f"(bid={st['n_fills_bid']}, ask={st['n_fills_ask']}) | "
            f"mode={st['last_mode']} | "
            f"TOB={st['moves_tob']}/{threshold_tob} | "
            f"event_steps={st['event_steps']}/{threshold_events} | "
            f"last_time={st['last_update_time']:.2f}"
        )

    policy.get_n_fills = get_n_fills
    policy.print_stats = print_stats
    policy._debug_state = st

    return policy
