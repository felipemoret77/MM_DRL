from typing import Dict, Any, Tuple, Optional
import math

def alternating_place_bid_ask_and_cancel_policy_factory(
    n_steps: int = 10, # Maps to 'n_events' (Frequency of action)
    
    # --- New Optional Throttles (Hard Throttle Architecture) ---
    use_tob_update: bool = False,
    n_tob_moves: int = 10,
    use_time_update: bool = False,
    min_time_interval: float = 1.0,
):
    """
    Alternating "Place vs Cancel" Policy with HARD THROTTLING.
    
    BEHAVIOR:
    ---------
    - Simple State Machine: "place" <-> "cancel".
    - Waits strictly until the Throttle conditions are met.
      * 'n_steps' is treated as the Event Throttle.
    - Once Throttles open:
      1. If Mode == "place": Checks if book is valid. If yes, places Bid+Ask and switches mode to "cancel".
      2. If Mode == "cancel": Cancels all orders and switches mode to "place".
      3. Resets clocks only after a successful action.

    NOTE: This is useful for probing simulator latency or generating noise.
    """

    # -------------------------------------------------------------------------
    # 1. Parameter Validation
    # -------------------------------------------------------------------------
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")

    # Map n_steps to Event Threshold
    threshold_events = int(n_steps)
    threshold_tob = max(1, int(n_tob_moves))

    # Audit Log
    print("-" * 65)
    print(f"[Place/Cancel Hard Throttle] Initialized:")
    print(f"  Policy Name: alternating_place_bid_ask_and_cancel_policy_factory")
    print(f"  Frequency Control:")
    print(f"    - n_steps (Events): True (Every {threshold_events} ticks)")
    print(f"    - Time Throttle:    {use_time_update} (Every {min_time_interval}s)")
    print(f"    - TOB Moves:        {use_tob_update} (Every {n_tob_moves} changes)")
    print("-" * 65)

    # =========================================================================
    # 2. Internal Mutable State
    # =========================================================================
    st = {
        "mode": "place",  # "place" -> next trigger places; "cancel" -> next trigger cancels
        
        # Throttles
        "last_env_tob_key": None,
        "moves_tob": 0,
        "event_steps": 0,
        "last_update_time": -1.0,

        # Stats
        "n_cycles": 0,
    }

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
        """Global TOB key for throttling."""
        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        try: ibs, ias = int(round(env_bs)), int(round(env_as))
        except: ibs, ias = 0, 0
        return (int(env_bb), int(env_ba), ibs, ias)

    # -------------------------------------------------------------------------
    # Clock Management
    # -------------------------------------------------------------------------
    def _update_tob_clock(key):
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s, current_time):
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        
        # Always reset events (n_steps logic)
        st["event_steps"] = 0
        
        if use_time_update:
            st["last_update_time"] = current_time

    # =========================================================================
    # MAIN POLICY FUNCTION
    # =========================================================================
    def policy(state: Dict[str, Any], mm=None) -> Tuple:
        
        # --- A. Time Update ---
        current_time = _safe_float(state.get("time", 0.0))

        # --- B. Update Counters ---
        st["event_steps"] += 1

        if use_tob_update:
            key = _get_env_tob(state)
            if st["last_env_tob_key"] is None:
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # =====================================================================
        # GATE 1: HARD THROTTLE (The Timer)
        # =====================================================================
        is_ready = True 

        # Check Steps (n_steps)
        if st["event_steps"] < threshold_events:
            is_ready = False
        
        # Check Time
        if use_time_update:
            if st["last_update_time"] < 0: pass 
            elif (current_time - st["last_update_time"]) < min_time_interval:
                is_ready = False

        # Check TOB
        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_ready = False

        if not is_ready:
            return ("hold",)

        # =====================================================================
        # GATE 2: EXECUTION (Place or Cancel)
        # =====================================================================
        
        # --- PLACE PHASE ---
        if st["mode"] == "place":
            best_bid = int(state.get("best_bid", -1))
            best_ask = int(state.get("best_ask", -1))
            spread = int(state.get("spread", 0))

            # Sanity Check: Need a valid book
            if best_bid < 0 or best_ask < 0 or spread <= 0:
                # Book invalid. We do NOT reset clocks. We wait for next tick.
                # (Since throttle is technically "open", we just retry until book is good)
                return ("hold",)

            # Valid -> Execute
            st["mode"] = "cancel"
            _reset_all_clocks(state, current_time)
            return ("place_bid_ask", best_bid, best_ask)

        # --- CANCEL PHASE ---
        else: # mode == "cancel"
            st["mode"] = "place"
            st["n_cycles"] += 1
            _reset_all_clocks(state, current_time)
            return ("cancel_all",)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def print_stats(prefix: str = "PlaceCancel"):
        print(f"{prefix}: Cycles={st['n_cycles']} | Mode={st['mode']} | "
              f"Step={st['event_steps']}/{threshold_events} | "
              f"Time={st['last_update_time']:.2f}")

    policy.print_stats = print_stats
    policy._debug_state = st

    return policy