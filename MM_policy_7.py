#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 17:33:12 2026
Updated logic to fix Queue Priority bugs.
"""

from typing import Dict, Any, Tuple, Optional
import math

def always_fixed_offset_mm_policy_factory(
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

    # --- Mid handling ---
    mid_rounding: str = "round",   # "round" | "floor" | "ceil"

    # --- Refresh behavior ---
    # WARNING: Keeping this True kills queue priority on quiet markets.
    # Recommended: False
    refresh_if_unchanged: bool = False, 
):
    """
    Always Fixed Offset MM Policy (CORRECTED).
    
    Fixes implemented:
    1. Preserves queue priority on non-filled side during fills.
    2. Prevents unnecessary refreshes if target price == current price.
    """

    # -------------------------------------------------------------------------
    # 1) Validation
    # -------------------------------------------------------------------------
    if int(inv_limit) < 1:
        raise ValueError("inv_limit must be >= 1")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")
    if mid_rounding not in ("round", "floor", "ceil"):
        raise ValueError("mid_rounding must be one of: 'round', 'floor', 'ceil'")

    threshold_tob = max(1, int(n_tob_moves))
    threshold_events = max(1, int(n_events))
    base_offset = int(offset_ticks)
    if base_offset < 0:
        raise ValueError("offset_ticks must be >= 0")

    print("-" * 65)
    print(f"[Fixed Offset MM - Optimized] Initialized:")
    print(f"  offset={base_offset} | inv_limit={inv_limit} | refresh_unchanged={refresh_if_unchanged}")
    print("-" * 65)

    # =========================================================================
    # 2) Internal Mutable State
    # =========================================================================
    st = {
        "last_bid_px": None,
        "last_ask_px": None,
        "last_mode": None,

        # Throttles
        "last_env_tob_key": None,
        "moves_tob": 0,
        "event_steps": 0,
        "last_update_time": -1.0,

        # Fills
        "last_inv": None,
        "n_fills": 0,
        "n_fills_bid": 0,
        "n_fills_ask": 0,
    }

    prev_base_idx = [None]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _safe_int(x, d=-1):
        try: return int(x)
        except: return d

    def _safe_float(x, d=0.0):
        try: v = float(x); return v if math.isfinite(v) else d
        except: return d

    def _get_env_tob(s):
        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        # Check distinct sizes to detect queue changes roughly
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        return (env_bb, env_ba, env_bs, env_as)

    def _consume_fill(inv: int):
        if st["last_inv"] is None:
            st["last_inv"] = int(inv)
            return False, 0, 0
        delta = int(inv) - int(st["last_inv"])
        st["last_inv"] = int(inv)
        if delta == 0:
            return False, 0, 0
        return True, (+1 if delta > 0 else -1), abs(delta)

    def _update_tob_clock(key):
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s, current_time: float):
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        if use_event_update:
            st["event_steps"] = 0
        if use_time_update:
            st["last_update_time"] = current_time

    def _mid_from_bb_ba(bb: int, ba: int) -> int:
        m = 0.5 * (bb + ba)
        if mid_rounding == "floor": return int(math.floor(m))
        if mid_rounding == "ceil": return int(math.ceil(m))
        return int(round(m))

    def _target_quotes(bb: int, ba: int, desired_mode: str) -> Tuple[int, int]:
        mid = _mid_from_bb_ba(bb, ba)
        off = base_offset
        bid_px = mid - off
        ask_px = mid + off

        if desired_mode == "two_sided":
            # Safety: Prevent crossed/locked logic internally
            if bid_px >= ask_px:
                off = max(1, off)
                bid_px = mid - off
                ask_px = mid + off
                if bid_px >= ask_px: # Last resort
                    bid_px = mid - 1
                    ask_px = mid + 1
        return int(bid_px), int(ask_px)

    # =========================================================================
    # MAIN POLICY
    # =========================================================================
    def policy(state: Dict[str, Any], mm=None) -> Tuple:

        # --- A) Time & Shift Logic ---
        current_time = _safe_float(state.get("time", 0.0))

        shift_happened = False
        if mm is not None and hasattr(mm, "base_price_idx"):
            try: cur_base = int(mm.base_price_idx)
            except: cur_base = None
            if cur_base is not None:
                if prev_base_idx[0] is None:
                    prev_base_idx[0] = cur_base
                else:
                    base_shift = int(cur_base - int(prev_base_idx[0]))
                    if base_shift != 0:
                        shift_happened = True
                        if st["last_bid_px"] is not None: st["last_bid_px"] -= base_shift
                        if st["last_ask_px"] is not None: st["last_ask_px"] -= base_shift
                        prev_base_idx[0] = cur_base

        # --- B) Update Counters ---
        if use_event_update: st["event_steps"] += 1
        
        if use_tob_update:
            key = _get_env_tob(state)
            if shift_happened or st["last_env_tob_key"] is None:
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # --- C) Read State ---
        I_t = _safe_int(state.get("inventory", 0))
        has_bid = bool(state.get("has_bid", False))
        has_ask = bool(state.get("has_ask", False))
        mkt_bb = _safe_int(state.get("best_bid", -1))
        mkt_ba = _safe_int(state.get("best_ask", -1))

        if mkt_bb < 0 or mkt_ba < 0:
            return ("hold",)

        # --- Mode Logic ---
        if I_t >= inv_limit: desired_mode = "ask_only"
        elif I_t <= -inv_limit: desired_mode = "bid_only"
        else: desired_mode = "two_sided"

        mode_changed = (st["last_mode"] != desired_mode)

        # --- Fill Logic ---
        new_fill, filled_side, n_units = _consume_fill(I_t)
        if new_fill:
            st["n_fills"] += max(1, n_units)
            if filled_side == +1: st["n_fills_bid"] += max(1, n_units)
            else: st["n_fills_ask"] += max(1, n_units)

        # Missing / Forbidden legs
        missing_bid = (not has_bid) and (desired_mode in ("two_sided", "bid_only"))
        missing_ask = (not has_ask) and (desired_mode in ("two_sided", "ask_only"))
        forbidden_bid = has_bid and (desired_mode == "ask_only")
        forbidden_ask = has_ask and (desired_mode == "bid_only")

        # Target Quotes
        tgt_bid, tgt_ask = _target_quotes(mkt_bb, mkt_ba, desired_mode)

        # =====================================================================
        # EXECUTION HIERARCHY (4 Gates)
        # =====================================================================

        # GATE 1: CRITICAL (Mode Change / Forbidden legs)
        # -----------------------------------------------
        if mode_changed or forbidden_bid or forbidden_ask:
            st["last_mode"] = desired_mode
            _reset_all_clocks(state, current_time)

            if desired_mode == "two_sided":
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                return ("place_bid_ask", tgt_bid, tgt_ask)
            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, None
                return ("cancel_all_then_place", +1, tgt_bid)
            else: # ask_only
                st["last_bid_px"], st["last_ask_px"] = None, tgt_ask
                return ("cancel_all_then_place", -1, tgt_ask)

        # GATE 2: URGENT (Fills) - OPTIMIZED
        # -----------------------------------------------
        if new_fill:
            _reset_all_clocks(state, current_time)
            
            # Logic: If one side fills, we MUST replace it.
            # But we only replace the OTHER side if its price is wrong.
            
            # Helper to check if current quote on book matches target
            bid_ok = (st["last_bid_px"] == tgt_bid)
            ask_ok = (st["last_ask_px"] == tgt_ask)

            if desired_mode == "two_sided":
                # If we need to move both or both are messed up, place both
                if (not bid_ok) and (not ask_ok):
                    st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                    return ("place_bid_ask", tgt_bid, tgt_ask)
                
                # If Bid filled (or is missing) but Ask is fine:
                if (filled_side == +1) and ask_ok:
                    st["last_bid_px"] = tgt_bid
                    return ("cancel_bid_then_place", tgt_bid)
                
                # If Ask filled (or is missing) but Bid is fine:
                if (filled_side == -1) and bid_ok:
                    st["last_ask_px"] = tgt_ask
                    return ("cancel_ask_then_place", tgt_ask)
                
                # Fallback: Update both just in case
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                return ("place_bid_ask", tgt_bid, tgt_ask)

            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, None
                return ("cancel_all_then_place", +1, tgt_bid)
            
            else: # ask_only
                st["last_bid_px"], st["last_ask_px"] = None, tgt_ask
                return ("cancel_all_then_place", -1, tgt_ask)

        # GATE 3: HARD THROTTLE
        # -----------------------------------------------
        is_throttled = False
        if use_event_update and (st["event_steps"] < threshold_events): is_throttled = True
        if use_time_update:
            if st["last_update_time"] >= 0 and (current_time - st["last_update_time"]) < min_time_interval:
                is_throttled = True
        if use_tob_update and (st["moves_tob"] < threshold_tob): is_throttled = True

        if is_throttled:
            # Bypass throttle only if we are missing a required leg (e.g., startup/accidental cancel)
            if not (missing_bid or missing_ask):
                return ("hold",)

        # GATE 4: NORMAL REFRESH WINDOW - OPTIMIZED
        # -----------------------------------------------
        
        # Determine if prices actually changed
        diff_bid = (st["last_bid_px"] != tgt_bid)
        diff_ask = (st["last_ask_px"] != tgt_ask)

        # If we are strictly holding queue, and nothing changed, HOLD.
        if (not refresh_if_unchanged) and (not diff_bid) and (not diff_ask) and (not missing_bid) and (not missing_ask):
             _reset_all_clocks(state, current_time)
             return ("hold",)
        
        # If we got here, we update. 
        # OPTIMIZATION: Update only the side that changed to save messages/queue.
        
        _reset_all_clocks(state, current_time)

        if desired_mode == "two_sided":
            # If both changed or both missing:
            if (diff_bid and diff_ask) or (missing_bid and missing_ask):
                st["last_bid_px"], st["last_ask_px"] = tgt_bid, tgt_ask
                return ("place_bid_ask", tgt_bid, tgt_ask)
            
            # If only bid changed/missing:
            if diff_bid or missing_bid:
                st["last_bid_px"] = tgt_bid
                return ("cancel_bid_then_place", tgt_bid)
            
            # If only ask changed/missing:
            if diff_ask or missing_ask:
                st["last_ask_px"] = tgt_ask
                return ("cancel_ask_then_place", tgt_ask)
            
            # If forced refresh (refresh_if_unchanged=True)
            return ("place_bid_ask", tgt_bid, tgt_ask)

        elif desired_mode == "bid_only":
            st["last_bid_px"], st["last_ask_px"] = tgt_bid, None
            return ("cancel_all_then_place", +1, tgt_bid)
        
        else: # ask_only
            st["last_bid_px"], st["last_ask_px"] = None, tgt_ask
            return ("cancel_all_then_place", -1, tgt_ask)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def get_n_fills() -> int: return int(st["n_fills"])
    def print_stats(prefix: str = "AlwaysFixedOffset"):
        print(f"{prefix}: fills={st['n_fills']} | Mode={st['last_mode']} | "
              f"TOB={st['moves_tob']}/{threshold_tob} | "
              f"Time={st['last_update_time']:.2f}")

    policy.get_n_fills = get_n_fills
    policy.print_stats = print_stats
    policy._debug_state = st

    return policy