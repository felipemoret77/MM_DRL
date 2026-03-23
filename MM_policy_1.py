from typing import Dict, Any, Tuple, Optional, Callable
import math
import numpy as np

def alt_L1_wait_fill_policy_factory(
    max_inv: int = 50,
    level_offset: int = 0,

    # --- Throttling: TOB (Market Structure Moves) ---
    use_tob_update: bool = False,
    n_tob_moves: int = 10,

    # --- Throttling: Simulation Steps (Events) ---
    use_event_update: bool = False,
    n_events: int = 100,

    # --- Throttling: Simulation Time (Seconds) ---
    use_time_update: bool = False,
    min_time_interval: float = 1.0,
) -> Callable[[Dict[str, Any], Optional[Any]], Tuple]:
    """
    Alternate L1 "Ping-Pong" Policy with HARD THROTTLING.

    INTENDED STRATEGY:
    ------------------
    - Maintain exactly ONE live order at a time (Bid OR Ask).
    - Phase = BID: quote at BestBid (-offset). On fill -> switch to ASK.
    - Phase = ASK: quote at BestAsk (+offset). On fill -> switch to BID.

    IMPORTANT BUGFIX (WHY INVENTORY WAS SAW-TOOTHING TO ±max_inv):
    --------------------------------------------------------------
    The old "Gate 1" treated (not has_bid) / (not has_ask) as an inconsistency.
    But immediately after a fill, the filled order disappears (has_bid/has_ask can be False),
    so Gate 1 would *repost the same side* before the Fill-Reaction gate ran.
    That prevents ping-pong switching and makes inventory drift until forced by max_inv.

    NEW EXECUTION HIERARCHY:
    ------------------------
    1) CRITICAL: Inventory limit enforcement (force phase if at limits).
    2) CRITICAL: Ghost-order cleanup (wrong side active).
    3) URGENT:   Fill reaction (switch side on inventory change).
    4) URGENT:   Replenish if empty (ensure exactly one order is live).
    5) BLOCKED:  Throttles (apply ONLY to repricing, not to replenishment/fill).
    6) NORMAL:   Repricing (track target drift).
    """

    if int(max_inv) < 1:
        raise ValueError("max_inv must be >= 1")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")

    threshold_tob = max(1, int(n_tob_moves))
    threshold_events = max(1, int(n_events))

    print("-" * 65)
    print(f"[Alt L1 Ping-Pong - Fixed Relative Prices] Initialized:")
    print(f"  Max Inv={max_inv}, Level Offset={level_offset}")
    print(f"  Throttles: Events={use_event_update}({n_events}), "
          f"Time={use_time_update}({min_time_interval}s), "
          f"TOB={use_tob_update}({n_tob_moves})")
    print("-" * 65)

    # ---------------------------------------------------------------------
    # Internal Mutable State (closure)
    # ---------------------------------------------------------------------
    st = {
        "phase": "BID",          # "BID" or "ASK"
        "last_quote_side": 0,    # +1 bid, -1 ask
        "last_quote_px": None,  # last posted target px (relative)

        # throttles
        "last_env_tob_key": None,
        "moves_tob": 0,
        "event_steps": 0,
        "last_update_time": -1.0,

        # fill detection
        "last_inv": None,
        "n_fills": 0,
        "n_fills_bid": 0,
        "n_fills_ask": 0,
    }

    prev_base_idx = [None]

    # -----------------------------
    # Helpers
    # -----------------------------
    def _safe_int(x, d=None):
        try:
            return int(x)
        except Exception:
            return d

    def _safe_float(x, d=0.0):
        try:
            v = float(x)
            return v if math.isfinite(v) else d
        except Exception:
            return d

    def _get_env_tob(s):
        """Global TOB key: (BestBid, BestAsk, BidSize, AskSize)."""
        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        try:
            ibs, ias = int(round(env_bs)), int(round(env_as))
        except Exception:
            ibs, ias = 0, 0
        return (int(env_bb), int(env_ba), ibs, ias)

    def _consume_fill(inv: int):
        """
        Detect fills via inventory delta:
          delta > 0  => we bought (bid filled)
          delta < 0  => we sold  (ask filled)
        """
        if st["last_inv"] is None:
            st["last_inv"] = int(inv)
            return False, 0, 0
        delta = int(inv) - int(st["last_inv"])
        st["last_inv"] = int(inv)
        if delta == 0:
            return False, 0, 0
        return True, (+1 if delta > 0 else -1), abs(delta)

    # -----------------------------
    # Clock management
    # -----------------------------
    def _update_tob_clock(key):
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s, current_time):
        # Reset counters after we actually *do something* (place/cancel/refresh).
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        if use_event_update:
            st["event_steps"] = 0
        if use_time_update:
            st["last_update_time"] = current_time

    # ---------------------------------------------------------------------
    # MAIN POLICY
    # ---------------------------------------------------------------------
    def policy(state: Dict[str, Any], mm=None) -> Tuple:
        # --- A) Time & shift logic (relative-grid backtests) ---
        current_time = _safe_float(state.get("time", 0.0))

        shift_happened = False
        if mm is not None and hasattr(mm, "base_price_idx"):
            try:
                cur_base = int(mm.base_price_idx)
            except Exception:
                cur_base = None

            if cur_base is not None:
                if prev_base_idx[0] is None:
                    prev_base_idx[0] = cur_base
                else:
                    base_shift = int(cur_base - int(prev_base_idx[0]))
                    if base_shift != 0:
                        shift_happened = True
                        # Keep last_quote_px consistent in the *new* relative grid
                        if st["last_quote_px"] is not None:
                            st["last_quote_px"] -= base_shift
                        prev_base_idx[0] = cur_base

        # --- B) Update throttle counters ---
        if use_event_update:
            st["event_steps"] += 1

        if use_tob_update:
            key = _get_env_tob(state)
            if shift_happened or st["last_env_tob_key"] is None:
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # --- C) Read state snapshot ---
        I_t = _safe_int(state.get("inventory", 0), 0)
        has_bid = bool(state.get("has_bid", False))
        has_ask = bool(state.get("has_ask", False))

        eff_bb = _safe_int(state.get("best_bid"), None)
        eff_ba = _safe_int(state.get("best_ask"), None)
        if eff_bb is None or eff_ba is None:
            return ("hold",)

        # Targets (allow negative indices)
        gap = 1 if eff_ba <= eff_bb else 0  # crossed-book guard
        bid_target = eff_bb - int(level_offset) - gap
        ask_target = eff_ba + int(level_offset) + gap

        # --- D) Fill detection (inventory delta) ---
        new_fill, filled_side, n_units = _consume_fill(I_t)
        if new_fill:
            st["n_fills"] += max(1, int(n_units))
            if filled_side == +1:
                st["n_fills_bid"] += max(1, int(n_units))
            else:
                st["n_fills_ask"] += max(1, int(n_units))

        # =====================================================================
        # 1) CRITICAL: Inventory limit enforcement (force phase if at bounds)
        # =====================================================================
        # If we are at +max_inv, we must be in ASK phase (sell).
        # If we are at -max_inv, we must be in BID phase (buy).
        if I_t >= max_inv:
            st["phase"] = "ASK"
        elif I_t <= -max_inv:
            st["phase"] = "BID"

        # =====================================================================
        # 2) CRITICAL: Ghost-order cleanup (wrong-side active)
        # =====================================================================
        # NOTE: We DO NOT consider "missing order" a ghost.
        # Missing order happens naturally after a fill.
        # Ghost means: an order exists on the wrong side for the current phase.
        if st["phase"] == "ASK":
            if has_bid:
                # Wrong-side order exists -> wipe and place correct side
                st["last_quote_side"], st["last_quote_px"] = -1, ask_target
                _reset_all_clocks(state, current_time)
                return ("cancel_all_then_place", -1, ask_target)
        else:  # "BID"
            if has_ask:
                st["last_quote_side"], st["last_quote_px"] = +1, bid_target
                _reset_all_clocks(state, current_time)
                return ("cancel_all_then_place", +1, bid_target)

        # =====================================================================
        # 3) URGENT: Fill reaction (true ping-pong switch)
        # =====================================================================
        if new_fill:
            # Default ping-pong: buy fill -> sell next, sell fill -> buy next
            desired_phase = "ASK" if filled_side == +1 else "BID"

            # Safety: inventory limits override the desired phase
            if I_t >= max_inv:
                desired_phase = "ASK"
            elif I_t <= -max_inv:
                desired_phase = "BID"

            st["phase"] = desired_phase

            if st["phase"] == "ASK":
                new_side, new_px = -1, ask_target
            else:
                new_side, new_px = +1, bid_target

            st["last_quote_side"] = new_side
            st["last_quote_px"] = new_px
            _reset_all_clocks(state, current_time)
            return ("cancel_all_then_place", new_side, new_px)

        # =====================================================================
        # 4) URGENT: Replenish if empty (ensure exactly one live order)
        # =====================================================================
        # If we currently have *no* orders, we must place one according to phase.
        # This runs even under throttling, because "no order" is worse than "not repricing".
        if (not has_bid) and (not has_ask):
            if st["phase"] == "ASK":
                st["last_quote_side"], st["last_quote_px"] = -1, ask_target
                _reset_all_clocks(state, current_time)
                return ("cancel_all_then_place", -1, ask_target)
            else:
                st["last_quote_side"], st["last_quote_px"] = +1, bid_target
                _reset_all_clocks(state, current_time)
                return ("cancel_all_then_place", +1, bid_target)

        # =====================================================================
        # 5) BLOCKED: Hard throttles (apply only to repricing)
        # =====================================================================
        is_throttled = False

        if use_event_update and (st["event_steps"] < threshold_events):
            is_throttled = True

        if use_time_update:
            if st["last_update_time"] >= 0 and (current_time - st["last_update_time"]) < min_time_interval:
                is_throttled = True

        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_throttled = True

        # =====================================================================
        # 6) NORMAL: Repricing (track target drift)
        # =====================================================================
        if st["phase"] == "ASK":
            target_side, target_px = -1, ask_target
        else:
            target_side, target_px = +1, bid_target

        # Compare against last posted quote (what we *think* we have live)
        price_drifted = (st["last_quote_px"] != target_px) or (st["last_quote_side"] != target_side)

        if not price_drifted:
            # Optional: soft refresh clocks even on hold to avoid pathological throttles
            _reset_all_clocks(state, current_time)
            return ("hold",)

        if is_throttled:
            return ("hold",)

        # Execute repricing on the same side (since we keep exactly one live order)
        st["last_quote_side"] = target_side
        st["last_quote_px"] = target_px
        _reset_all_clocks(state, current_time)

        if target_side == +1:
            return ("cancel_bid_then_place", target_px)
        else:
            return ("cancel_ask_then_place", target_px)

    # ---------------------------------------------------------------------
    # Policy stats / debug hooks
    # ---------------------------------------------------------------------
    def get_n_fills() -> int:
        return int(st["n_fills"])

    def print_stats(prefix: str = "Alt_L1_PingPong"):
        print(f"{prefix}: fills={st['n_fills']} | phase={st['phase']} | "
              f"TOB={st['moves_tob']}/{threshold_tob} | "
              f"Evt={st['event_steps']}/{threshold_events} | "
              f"last_update_time={st['last_update_time']:.2f}")

    policy.get_n_fills = get_n_fills
    policy.print_stats = print_stats
    policy._debug_state = st

    return policy
