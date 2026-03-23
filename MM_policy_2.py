from typing import Dict, Any, Tuple, Optional
import math

def alternating_mo_policy_factory(
    max_inv: int = 50,
    start_side: int = +1,
    
    # --- Throttling: Simulation Steps (Frequency) ---
    # Defaulting to True/10 to mimic the original "n_steps=10" behavior
    use_event_update: bool = True, 
    n_events: int = 10,

    # --- Throttling: Simulation Time (Seconds) ---
    use_time_update: bool = False,
    min_time_interval: float = 1.0,

    # --- Throttling: TOB (Wait for Market Moves) ---
    use_tob_update: bool = False,
    n_tob_moves: int = 10,
):
    """
    Alternating Market Order (MO) Policy with HARD THROTTLING.
    
    BEHAVIOR:
    ---------
    - Waits strictly until the Throttle conditions are met (Time, Events, or TOB).
    - Once Throttles open:
      1. Checks Inventory Limits.
      2. If Safe: Fires MO (Buy or Sell) and Flips the side for next time.
      3. If Blocked: Does NOTHING, Resets Clocks, and waits for next cycle (Skip Turn).

    Standardized Architecture:
    --------------------------
    Uses the same internal clock logic as the GLFT and L1 policies for consistency.
    """

    if int(max_inv) < 1:
        raise ValueError("max_inv must be >= 1")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")
    if use_event_update and n_events < 1:
        raise ValueError("n_events must be >= 1.")

    # Cache thresholds
    threshold_tob = max(1, int(n_tob_moves))
    threshold_events = max(1, int(n_events))

    # Audit Log
    print("-" * 65)
    print(f"[Alternating MO Hard Throttle] Initialized:")
    print(f"  Max Inv={max_inv}, Start Side={'+1 (Buy)' if start_side >=0 else '-1 (Sell)'}")
    print(f"  Frequency Control:")
    print(f"    - Events (Steps): {use_event_update} (Every {n_events} ticks)")
    print(f"    - Time:           {use_time_update} (Every {min_time_interval}s)")
    print(f"    - TOB Moves:      {use_tob_update} (Every {n_tob_moves} changes)")
    print("-" * 65)

    # =========================================================================
    # Internal Mutable State
    # =========================================================================
    st = {
        # Strategy State
        "side": +1 if start_side >= 0 else -1,  # The side we WANT to execute next
        
        # Throttles
        "last_env_tob_key": None,
        "moves_tob": 0,
        "event_steps": 0,
        "last_update_time": -1.0,

        # Stats
        "n_mos": 0,
        "n_buy": 0,
        "n_sell": 0,
        "n_blocked": 0,
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
        """Resets all counters to 0. Effectively 'starts the timer' for the next trade."""
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        if use_event_update:
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
        if use_event_update:
            st["event_steps"] += 1

        if use_tob_update:
            key = _get_env_tob(state)
            if st["last_env_tob_key"] is None:
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # =====================================================================
        # GATE 1: HARD THROTTLE (The "Timer")
        # =====================================================================
        # Unlike MM policies, here "Held" means "Not Ready to Fire Yet".
        # We perform NO ACTION until the throttle releases.
        
        is_ready = True # Assume ready unless a throttle blocks us

        if use_event_update and (st["event_steps"] < threshold_events):
            is_ready = False
        
        if use_time_update:
            if st["last_update_time"] < 0: pass # First run allowed
            elif (current_time - st["last_update_time"]) < min_time_interval:
                is_ready = False

        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_ready = False

        # If not ready, we hold position (which for a Taker means doing nothing)
        if not is_ready:
            return ("hold",)

        # =====================================================================
        # GATE 2: INVENTORY SAFETY & EXECUTION
        # =====================================================================
        # The throttle is open. We attempt to fire.
        
        inv = _safe_int(state.get("inventory", 0))
        side_to_exec = st["side"]
        
        # Check Limits
        blocked = False
        if side_to_exec == +1 and inv >= int(max_inv):
            blocked = True
        if side_to_exec == -1 and inv <= -int(max_inv):
            blocked = True

        if blocked:
            st["n_blocked"] += 1
            # CRITICAL: We reset the clocks even if blocked.
            # This implements "Skip Turn". If we didn't reset, it would try again
            # immediately next tick (infinite loop of blocks), which violates "Every N Steps".
            _reset_all_clocks(state, current_time)
            return ("hold",)

        # FIRE!
        action = ("cross_buy",) if side_to_exec == +1 else ("cross_sell",)
        
        # Update Stats
        st["n_mos"] += 1
        if side_to_exec == +1: st["n_buy"] += 1
        else: st["n_sell"] += 1

        # Flip Side for Next Time
        st["side"] = -side_to_exec
        
        # Reset Timer
        _reset_all_clocks(state, current_time)
        
        return action

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def print_stats(prefix: str = "Alt_MO_Throttle"):
        print(f"{prefix}: n_mos={st['n_mos']} (B={st['n_buy']}, S={st['n_sell']}) | "
              f"Blocked={st['n_blocked']} | Next={st['side']} | "
              f"Evt={st['event_steps']}/{threshold_events} | "
              f"Time={st['last_update_time']:.2f}")

    policy.print_stats = print_stats
    policy._debug_state = st

    return policy