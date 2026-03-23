from typing import Dict, Any, Tuple
import math

def single_side_alternating_offset_policy_factory(
    side: int = +1,       # +1 for Bid, -1 for Ask
    offset_1: int = 0,    # Distance in ticks from Best
    offset_2: int = 5,    # Distance in ticks for the second state
    
    # --- Throttling (Frequency of the Switch) ---
    n_steps: int = 10,
    use_event_update: bool = True,
    
    use_tob_update: bool = False,
    n_tob_moves: int = 10,
    
    use_time_update: bool = False,
    min_time_interval: float = 1.0,
):
    """
    Single Side Alternating Offset Policy with HARD THROTTLING + FILL REPLENISHMENT.
    """

    # -------------------------------------------------------------------------
    # 1. Parameter Validation
    # -------------------------------------------------------------------------
    if side not in (+1, -1):
        raise ValueError("side must be +1 (Bid) or -1 (Ask)")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")

    threshold_events = int(n_steps)
    threshold_tob = max(1, int(n_tob_moves))

    # Audit Log (CORRIGIDO: Removido 'f' de strings sem variáveis)
    print("-" * 65)
    print("[Alternating Offset + Replenish (No Reset)] Initialized:")
    print(f"  Side: {side} | Offsets: {offset_1} <-> {offset_2}")
    print("  Logic: Wait Throttle -> Switch.")
    print("  IF FILL -> Repost Same Offset (KEEP TIMER RUNNING).")
    print("-" * 65)

    # =========================================================================
    # 2. Internal Mutable State
    # =========================================================================
    st = {
        "current_offset_idx": 1, # 1 for offset_1, 2 for offset_2
        "last_quote_px": None,
        
        # Throttles
        "last_env_tob_key": None,
        "moves_tob": 0,
        "event_steps": 0,
        "last_update_time": -1.0,

        # Fill Tracking
        "last_inv": None,
        "n_fills": 0,

        # Stats
        "n_switches": 0,
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
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        try: ibs, ias = int(round(env_bs)), int(round(env_as))
        except: ibs, ias = 0, 0
        return (int(env_bb), int(env_ba), ibs, ias)
        
    def _consume_fill(inv):
        if st["last_inv"] is None:
            st["last_inv"] = int(inv)
            return False, 0
        delta = int(inv) - int(st["last_inv"])
        st["last_inv"] = int(inv)
        if delta == 0: return False, 0
        return True, abs(delta)

    # -------------------------------------------------------------------------
    # Clock Management
    # -------------------------------------------------------------------------
    def _update_tob_clock(key):
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s, current_time):
        """Called ONLY when switching offsets (the hard update)."""
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        st["event_steps"] = 0
        if use_time_update:
            st["last_update_time"] = current_time

    # =========================================================================
    # MAIN POLICY FUNCTION
    # =========================================================================
    def policy(state: Dict[str, Any], mm=None) -> Tuple:
        
        # --- A. Time & Shift Logic ---
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
                        if st["last_quote_px"] is not None: 
                            st["last_quote_px"] -= base_shift
                        prev_base_idx[0] = cur_base

        # --- B. Update Counters ---
        if use_event_update:
            st["event_steps"] += 1

        if use_tob_update:
            key = _get_env_tob(state)
            if shift_happened or st["last_env_tob_key"] is None:
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)
                
        # --- C. Read State & Fills ---
        I_t = _safe_int(state.get("inventory", 0))
        best_bid = int(state.get("best_bid", -1))
        best_ask = int(state.get("best_ask", -1))
        
        new_fill, n_units = _consume_fill(I_t)
        if new_fill:
            st["n_fills"] += n_units

        if best_bid < 0 or best_ask < 0:
            return ("hold",)

        # Helper to calc price based on current offset state
        def get_target_px():
            active_offset = offset_1 if st["current_offset_idx"] == 1 else offset_2
            if side == +1: # BID
                return max(0, best_bid - active_offset)
            else: # ASK
                return best_ask + active_offset

        # =====================================================================
        # GATE 1: URGENT FILL REPLENISHMENT (Priority Over Throttle)
        # =====================================================================
        if new_fill:
            # DO NOT SWITCH OFFSET. Keep the current one.
            # DO NOT RESET CLOCKS. Keep the timer running towards the switch.
            target_px = get_target_px()
            st["last_quote_px"] = target_px
            
            # NOTE: We do NOT call _reset_all_clocks here.
            
            return ("cancel_all_then_place", side, target_px)

        # =====================================================================
        # GATE 2: HARD THROTTLE (Switch Timer)
        # =====================================================================
        is_ready = True 

        if use_event_update and (st["event_steps"] < threshold_events):
            is_ready = False
        if use_time_update:
            if st["last_update_time"] < 0: pass 
            elif (current_time - st["last_update_time"]) < min_time_interval:
                is_ready = False
        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_ready = False

        if not is_ready:
            return ("hold",)

        # =====================================================================
        # GATE 3: EXECUTION (Switch Offset & Place)
        # =====================================================================
        # If we reached here, the timer EXPIRED. We switch offsets.
        
        # 1. Toggle Offset State (1 -> 2 -> 1)
        if st["last_update_time"] >= 0:
            st["current_offset_idx"] = 2 if st["current_offset_idx"] == 1 else 1
            st["n_switches"] += 1

        # 2. Calculate New Target
        target_px = get_target_px()

        # 3. Execute & RESET TIMER
        st["last_quote_px"] = target_px
        _reset_all_clocks(state, current_time) # <--- Reset only happens here
        
        return ("cancel_all_then_place", side, target_px)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def print_stats(prefix: str = "AltOffset"):
        current_off = offset_1 if st["current_offset_idx"] == 1 else offset_2
        print(f"{prefix}: Fills={st['n_fills']} | Switches={st['n_switches']} | "
              f"Offset={current_off} | "
              f"Step={st['event_steps']}/{threshold_events} | "
              f"Time={st['last_update_time']:.2f}")

    policy.print_stats = print_stats
    policy._debug_state = st

    return policy