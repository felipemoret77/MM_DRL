#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep_RL_With_Signal_policy_factory.py
=====================================

Hybrid Controller for Deep Reinforcement Learning Market Making.

Architecture Overview
---------------------
This module sits between the **Market-Making Simulator** (MM_LOB_SIM) and the
**RL Agent** (neural network).  On every simulator step the controller:

    Simulator state  -->  [Throttle gate]  -->  [SGU signals]  -->  [RL agent]  -->  Action
                              |                     |                   |
                         skip if too     Z-score normalize      neural-net or
                         soon / no TOB   with TRAIN stats       placeholder fn
                         change

The controller does NOT train the agent; it only provides the decision pipeline
that wraps a trained (or placeholder) agent function.  Training would live in an
outer loop that calls ``act_func.controller.learn(...)`` after each step.

Two Operating Modes
-------------------
1. **"backtest"** --- Reads pre-computed SGU label/prediction arrays from disk.
   A cursor advances one position each time a new decision is due (not every
   simulator message --- see throttling).  This is fast and deterministic, ideal
   for evaluating different RL strategies on historical data.

2. **"simulation"** --- Calls live inference functions (e.g., XGBoost.predict,
   LSTM forward pass) on the current simulator state.  This is slower but
   required when the agent interacts with a stochastic simulator where future
   states depend on past actions (unlike backtest where the tape is fixed).

Normalization Guarantee
-----------------------
**Critical**: Z-score normalization always uses statistics (mu, std) from the
**training set**, regardless of mode.  This prevents data leakage:

    z = clip( (x - mu_train) / std_train,  -clip, +clip )

The "ruler" (norm stats) is fixed at construction time.  The "stream" (execution
data or live inference output) can be from any split (train, val, test).

Throttling System
-----------------
Not every simulator message warrants a new decision.  Three independent gates
control how often the agent actually re-evaluates:

1. **Step-based** (``throttle_every_n_steps``): act every N-th simulator step.
2. **TOB-based** (``use_tob_update`` + ``n_tob_moves``): act only after the
   top-of-book has changed at least N times (price OR size).
3. **Time-based** (``use_time_update`` + ``min_time_interval``): act only after
   at least T seconds have elapsed since the last decision.

When throttled, the controller returns ``("hold",)`` --- the simulator keeps the
agent's current orders unchanged.

**Important interaction with backtest mode**: the SGU cursor advances only on
non-throttled steps.  This keeps the offline signal stream synchronized with
actual decision points.  For example, if you have 500 SGU predictions and 500
TOB windows, the cursor advances once per window, not once per raw message.

Action Decoding
---------------
The RL agent outputs an integer ``action_idx``.  ``_decode_action`` maps this
to a simulator-understood tuple:

- **"generic"** mode (default, 6 actions):
    0 = place_bid at best_bid
    1 = place_ask at best_ask
    2 = place_bid_ask at best_bid, best_ask
    3 = cancel_bid
    4 = cancel_ask
    5 = hold (do nothing)

- **"pure_mm"** mode (N actions, each a spread level):
    action_idx = level -> place bid at (best_bid - level * tick_size),
                         place ask at (best_ask + level * tick_size).
    Level 0 = quote at the touch; higher levels = wider spreads.

Factory Function
----------------
``Deep_RL_With_Signal_policy_factory()`` is the main entry point.  It:
1. Creates a ``DeepRLSignalController`` instance.
2. Monkey-patches the RL agent function into the controller.
3. Returns the controller's ``.act()`` method --- a callable that the simulator
   invokes as its policy function.

Usage in the runner::

    policy = Deep_RL_With_Signal_policy_factory(
        rl_act_fn=my_trained_agent,      # state -> int (action index)
        sgu1_norm_data=sgu1_train_preds, # training set SGU1 predictions
        sgu2_norm_data=sgu2_train_preds, # training set SGU2 predictions
        mode="backtest",
        sgu1_execution_data=sgu1_test_preds,  # test set SGU1 predictions
        sgu2_execution_data=sgu2_test_preds,  # test set SGU2 predictions
    )
    # Now pass `policy` to the simulator as its policy function.
    # Access internals via: policy.controller.st, policy.controller.cursor, etc.

Dependencies
------------
- numpy
- No LOBSTER column access --- this module works entirely with simulator state
  dicts (keys: best_bid, best_ask, bid_size, ask_size, time, ...).

@author: felipemoret
"""

# ---------------------------------------------------------------------------
# Standard library and third-party imports.
# numpy is the only external dependency --- the module deliberately avoids
# heavy ML frameworks (torch, tensorflow) to keep the decision pipeline
# lightweight and framework-agnostic.  The RL agent itself (which *may* use
# PyTorch or TensorFlow) is injected via a callable, so this file never needs
# to import those libraries.
# ---------------------------------------------------------------------------
# FIX 7 (LOW): Removed `import math` — it was never used anywhere in this
# module.  Unused imports add cognitive overhead for readers who wonder
# "where is math used?" and can trigger linter warnings (F401).
import numpy as np
import warnings
from typing import Dict, Any, Tuple, Optional, Union, List, Callable


class DeepRLSignalController:
    """
    Central controller connecting the Market Maker Simulator, SGU signals
    (offline arrays or live inference), and the RL agent decision logic.

    This class is typically not instantiated directly --- use the factory
    function ``Deep_RL_With_Signal_policy_factory()`` instead.

    Why a separate class instead of a simple function?
    --------------------------------------------------
    Because the controller carries *stateful* information across steps:
      - The backtest cursor position (which SGU prediction are we on?).
      - Throttle counters (how many TOB changes since last decision?).
      - Cached last signals and last action (for diagnostics and fallbacks).
    A plain function would need global state or a closure; a class makes
    the lifecycle explicit and provides ``reset()`` for episode boundaries.

    Parameters
    ----------
    sgu1_norm_data : array-like
        SGU-1 predictions on the **training** set.  Used to compute Z-score
        normalization statistics (mean, std).  These stats are frozen at
        construction and never updated --- this is the "ruler".
    sgu2_norm_data : array-like
        Same, for SGU-2.
    mode : str
        "backtest" (read from arrays) or "simulation" (call live functions).
    sgu1_execution_data : array-like, optional
        SGU-1 predictions for the execution split (train/val/test backtest).
        Required when ``mode="backtest"``.
    sgu2_execution_data : array-like, optional
        Same, for SGU-2.
    sgu1_live_inference_fn : callable, optional
        Function ``(state_dict) -> float`` returning a raw SGU-1 prediction.
        Required when ``mode="simulation"``.
    sgu2_live_inference_fn : callable, optional
        Same, for SGU-2.
    action_mode : str
        "generic" (6 discrete actions) or "pure_mm" (N spread-level actions).
    n_actions : int
        Number of discrete actions the RL agent can choose from.
    tick_size : float
        Minimum price increment.  Used in "pure_mm" mode to compute spread
        offsets.  Must match the simulator's tick_size.
    signal_clip : float
        Symmetric clip bound for Z-scored signals: values are clamped to
        [-signal_clip, +signal_clip] after standardization.
    throttle_every_n_steps : int
        Agent re-evaluates every N-th simulator step.  1 = every step.
    use_tob_update : bool
        If True, require ``n_tob_moves`` top-of-book changes before acting.
    n_tob_moves : int
        Number of TOB changes required per decision (only if use_tob_update).
    use_time_update : bool
        If True, require ``min_time_interval`` seconds between decisions.
    min_time_interval : float
        Minimum seconds between decisions (only if use_time_update).
    verbose : bool
        If True, print initialization summary to stdout.
    """

    def __init__(
        self,
        # --- 1. Normalization Source ("The Ruler") ---
        # ALWAYS required.  Defines the Z-score statistics (mu, std) from the
        # TRAINING set.  These are frozen at construction to prevent data leakage.
        #
        # WHY both sgu1 and sgu2?  The SGU (Signal Generation Unit) framework
        # produces two independent signal streams --- typically one for short-term
        # direction (sgu1 = label-based) and one for mid-term momentum (sgu2 =
        # model prediction-based).  Each has its own distribution, so each needs
        # its own normalization statistics.
        sgu1_norm_data: Union[np.ndarray, list],
        sgu2_norm_data: Union[np.ndarray, list],

        # --- 2. Mode Configuration ---
        # "backtest" = replay from pre-computed arrays (fast, deterministic).
        # "simulation" = call live inference functions (slow, supports stochastic sim).
        #
        # WHY two modes?  In backtest, the order-book tape is replayed as-is ---
        # the agent's actions do NOT change future market states.  So SGU signals
        # can be pre-computed once and stored.  In simulation, the agent's fills
        # may alter the book, which changes future SGU inputs --- signals must be
        # computed on-the-fly from the *actual* (potentially modified) state.
        mode: str = "backtest",  # "backtest" or "simulation"

        # --- 3. Offline Data Source (for "backtest" mode) ---
        # Pre-computed SGU predictions to replay sequentially.
        # These are the actual signal values the agent will *see* at each decision
        # point.  They can come from any data split (train, val, test) --- what
        # matters is that the NORMALIZATION stats always come from training.
        sgu1_execution_data: Optional[Union[np.ndarray, list]] = None,
        sgu2_execution_data: Optional[Union[np.ndarray, list]] = None,

        # --- 4. Online Inference Source (for "simulation" mode) ---
        # Callable functions: state_dict -> raw_float_signal.
        # These are typically partial-applied model.predict calls, e.g.:
        #   lambda state: xgb_model.predict(extract_features(state))[0]
        # The controller does NOT care about the model architecture; it only
        # needs a float output that it can Z-score normalize.
        sgu1_live_inference_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        sgu2_live_inference_fn: Optional[Callable[[Dict[str, Any]], float]] = None,

        # --- RL Configuration ---
        action_mode: str = "generic",  # "generic" or "pure_mm"
        n_actions: int = 6,
        tick_size: float = 0.01,       # Used in pure_mm mode for spread offsets

        # --- Signal Config ---
        # WHY clip at all?  Without clipping, extreme outliers in the signal
        # (e.g., a flash crash producing z=15) would dominate the neural net's
        # input and potentially cause numerical instability.  Clipping at +/-3
        # retains 99.7% of the normal distribution's information while bounding
        # the feature range, which helps gradient-based learning stay stable.
        signal_clip: float = 3.0,      # Symmetric Z-score clip bound

        # --- Throttling Config ---
        # WHY throttle at all?  In a real LOB, messages arrive at microsecond
        # frequency.  Re-evaluating a neural network on every message is:
        #   (a) computationally wasteful --- most messages don't change the picture.
        #   (b) strategically harmful --- frequent order changes increase exchange
        #       fees and look like "flickering" to market surveillance.
        # Throttling lets the agent make deliberate, spaced-out decisions aligned
        # with meaningful market events (TOB changes) or time intervals.
        throttle_every_n_steps: int = 1,  # 1 = no step-based throttling
        use_tob_update: bool = False,
        n_tob_moves: int = 10,
        use_time_update: bool = False,
        min_time_interval: float = 1.0,

        verbose: bool = True,
    ):
        # Normalize mode string to lowercase to accept "Backtest", "SIMULATION", etc.
        self.mode = mode.lower()

        # -----------------------------------------------------------------
        # 1. Normalization Setup --- Compute and freeze Z-score statistics
        #    from the TRAINING set predictions.
        #
        #    These statistics are the "ruler": they define the scale for all
        #    future Z-score computations, whether on train, val, or test data.
        #
        #    ANALOGY: Imagine you built a thermometer (the ruler) by measuring
        #    temperature in New York (training set).  You now use that same
        #    thermometer to measure temperature in Tokyo (test set).  The
        #    thermometer's scale doesn't change --- but you can still get a
        #    meaningful reading.  If you recalibrated the thermometer for each
        #    city, you'd lose the ability to compare readings across cities.
        #    Similarly, using test-set statistics would leak future information
        #    and produce unreproducible results.
        # -----------------------------------------------------------------
        # Convert to float64 arrays.  float64 avoids precision loss when
        # computing mean/std over thousands of predictions (float32 can
        # accumulate rounding errors in running sums).
        norm1 = np.asarray(sgu1_norm_data, dtype=np.float64)
        norm2 = np.asarray(sgu2_norm_data, dtype=np.float64)

        # Fail fast if training data is empty --- this is a configuration error,
        # not a recoverable situation.  An empty training split means the SGU
        # pipeline upstream produced no predictions, and we cannot compute
        # meaningful statistics.
        if norm1.size == 0 or norm2.size == 0:
            raise ValueError(
                "Normalization data cannot be empty.  "
                "Ensure the training split produced SGU predictions."
            )

        # Compute the "ruler" statistics.  These are plain scalars (not arrays)
        # to keep the per-step Z-score computation as fast as possible.
        self.mu1, self.std1 = float(np.mean(norm1)), float(np.std(norm1, ddof=1))
        self.mu2, self.std2 = float(np.mean(norm2)), float(np.std(norm2, ddof=1))

        # Guard against degenerate training data (constant predictions).
        # If std ~ 0, Z-scores would explode.  Fallback to std=1 so that
        # z = (x - mu), preserving the centering but skipping the scaling.
        #
        # WHY std=1.0 and not, say, std=1e-9?  Because std=1.0 makes z = (x - mu)
        # which is equivalent to simple mean-centering.  This is the mildest
        # possible transformation: the signal retains its original scale minus
        # the constant offset.  Using a tiny std would amplify any noise to
        # extreme magnitudes, which is worse than no scaling at all.
        #
        # WHY threshold 1e-9 instead of exactly 0?  Floating-point arithmetic
        # can produce tiny but nonzero std for nearly-constant data (e.g.,
        # std=1e-15 from rounding noise).  Dividing by 1e-15 produces z-values
        # in the trillions, which is numerically catastrophic.  1e-9 is a
        # conservative threshold that catches this case.
        if self.std1 <= 1e-9:
            if verbose:
                print("Warning: SGU1 std is ~0 (constant predictions). "
                      "Defaulting to std=1.0.")
            self.std1 = 1.0
        if self.std2 <= 1e-9:
            if verbose:
                print("Warning: SGU2 std is ~0 (constant predictions). "
                      "Defaulting to std=1.0.")
            self.std2 = 1.0

        self.signal_clip = signal_clip
        if signal_clip is not None and signal_clip <= 0:
            raise ValueError(
                f"signal_clip must be positive (got {signal_clip}). "
                "Use None to disable clipping entirely."
            )

        # -----------------------------------------------------------------
        # 2. Mode Setup --- Configure the signal source (offline vs. online).
        #
        #    The two modes share the SAME normalization and throttling logic;
        #    the ONLY difference is WHERE the raw SGU values come from:
        #      - backtest:   self.exec1[cursor], self.exec2[cursor]
        #      - simulation: self.fn_s1(state),  self.fn_s2(state)
        # -----------------------------------------------------------------
        if self.mode == "backtest":
            # Validate that execution arrays were provided --- this is a hard
            # requirement in backtest mode.  Without them, there are no signals
            # to replay.
            if sgu1_execution_data is None or sgu2_execution_data is None:
                raise ValueError(
                    "In 'backtest' mode, both sgu1_execution_data and "
                    "sgu2_execution_data are required."
                )

            # Convert to float64 arrays and set up the sequential cursor.
            # The cursor advances by 1 each time a non-throttled decision is made.
            #
            # WHY float64 for execution data too?  Consistency with the norm
            # stats (also float64) avoids subtle type-promotion surprises when
            # computing z = (exec[i] - mu) / std.
            self.exec1 = np.asarray(sgu1_execution_data, dtype=np.float64)
            self.exec2 = np.asarray(sgu2_execution_data, dtype=np.float64)

            # Take the minimum length in case the two signal arrays differ in
            # size (e.g., if one SGU model produced fewer valid predictions).
            # This ensures we never index out of bounds on either array.
            self.max_steps = min(len(self.exec1), len(self.exec2))

            # The cursor is the READ POINTER into the execution arrays.
            # It starts at 0 and increments by 1 on each non-throttled step.
            # IMPORTANT: The cursor does NOT advance on throttled steps --- this
            # is what keeps the signal stream aligned with actual decision
            # points.  See _get_signals() for the advancement logic.
            self.cursor = 0

            if verbose:
                print(f"[Controller] Mode: BACKTEST "
                      f"(replay from arrays, N={self.max_steps})")

        elif self.mode == "simulation":
            # Validate that inference functions were provided --- this is a hard
            # requirement in simulation mode.  Without them, there is no way
            # to compute signals on-the-fly.
            if sgu1_live_inference_fn is None or sgu2_live_inference_fn is None:
                raise ValueError(
                    "In 'simulation' mode, both sgu1_live_inference_fn and "
                    "sgu2_live_inference_fn are required."
                )

            # Store function references for on-the-fly inference.
            # These are typically closures or partial-applied functions that
            # encapsulate the SGU model and feature extraction logic:
            #   e.g., fn_s1 = lambda state: xgb_model.predict(features(state))[0]
            #
            # The controller is deliberately agnostic to the model type (XGBoost,
            # LSTM, linear, etc.) --- it only needs a float output.
            self.fn_s1 = sgu1_live_inference_fn
            self.fn_s2 = sgu2_live_inference_fn

            if verbose:
                print("[Controller] Mode: SIMULATION "
                      "(live inference functions)")

        else:
            raise ValueError(
                f"Unknown mode: '{self.mode}'.  "
                f"Must be 'backtest' or 'simulation'."
            )

        # -----------------------------------------------------------------
        # 3. Action & Throttling Configuration
        #
        #    These parameters define the RL agent's action space and the
        #    frequency at which it makes decisions.  They are stored as
        #    instance attributes so they can be inspected or modified
        #    (carefully!) between episodes if needed.
        # -----------------------------------------------------------------
        self.action_mode = action_mode
        self.n_actions = n_actions
        self.tick_size = tick_size

        # FIX 5 (MEDIUM): Validate action_mode to catch typos at construction
        # time rather than at runtime. Without this check, a typo like
        # "generic_mm" would cause the agent to silently return ("hold",) on
        # every step --- extremely difficult to debug because no error is
        # raised; the agent simply appears to never trade.
        _VALID_ACTION_MODES = ("generic", "pure_mm")
        if self.action_mode not in _VALID_ACTION_MODES:
            raise ValueError(
                f"Unknown action_mode: '{self.action_mode}'. "
                f"Must be one of {_VALID_ACTION_MODES}."
            )

        # FIX 6 (MEDIUM): Validate n_actions vs action_mode consistency.
        # In "generic" mode, exactly 6 actions are hard-coded in _decode_action:
        #   0=place_bid, 1=place_ask, 2=place_bid_ask, 3=cancel_bid, 4=cancel_ask, 5=hold.
        # Using a different n_actions would cause the "hold" default (n_actions-1)
        # to map to an unintended action, which is dangerous.
        if self.action_mode == "generic" and self.n_actions != 6:
            warnings.warn(
                f"action_mode='generic' expects n_actions=6, got {self.n_actions}. "
                f"The default 'hold' action (index {self.n_actions - 1}) may not map "
                f"to the intended no-op. Proceed with caution.",
                UserWarning, stacklevel=2,
            )

        # Throttle gates --- each gate independently can suppress a decision.
        # max(1, ...) ensures we never get throttle_every_n_steps = 0, which
        # would cause a modulo-by-zero error in _check_throttling.
        self.throttle_every_n_steps = max(1, int(throttle_every_n_steps))
        self.use_tob = use_tob_update
        self.threshold_tob = n_tob_moves
        self.use_time = use_time_update
        self.min_time = min_time_interval

        # -----------------------------------------------------------------
        # 4. Internal State --- Mutable counters and caches.
        #    Call ``reset()`` to restore to initial state (e.g., between
        #    backtest episodes).
        #
        #    All mutable state lives in ``self.st`` (a dict) plus the cursor.
        #    Consolidating mutable state in one place makes it easy to
        #    serialize, inspect, and reset cleanly.
        # -----------------------------------------------------------------
        self._init_state()

    def _init_state(self):
        """
        Initialize / reset all mutable internal state to defaults.

        This method is called both at construction time (from __init__) and
        on explicit reset() calls between episodes.  It ensures a clean slate
        for each backtest run or simulation episode.

        WHY a separate method instead of inline in __init__?
        Because reset() also needs this logic, and duplicating it would be
        error-prone.  Any new mutable state added in the future only needs
        to be initialized in ONE place.
        """
        self.st = {
            # Counts every call to act(), including throttled ones.
            # Used by the step-based throttle gate (modulo check).
            #
            # FIX 1 (HIGH): Initialize to -1 so that the FIRST increment
            # (in act()) brings it to 0.  This ensures the very first call
            # passes the step gate (0 % N == 0 for all N), which is the
            # correct behavior: the agent should always make at least one
            # decision at the start of an episode.
            #
            # Previously this was initialized to 0, which meant the first
            # increment set it to 1, and 1 % N != 0 for all N > 1 ---
            # so the very first step was ALWAYS throttled, causing the
            # agent to miss its first decision opportunity.
            "step_counter": -1,
            # Number of TOB changes observed since last non-throttled decision.
            # Resets to 0 after each decision.  Used by the TOB-based throttle gate.
            "moves_tob": 0,
            # Last observed (best_bid, best_ask, bid_size, ask_size) tuple
            # for detecting TOB changes.  None on first call = forces detection
            # of the first TOB state as a "change" (which is correct --- the agent
            # has never seen any TOB state before).
            "last_env_tob_key": None,
            # Timestamp (float seconds) of the last non-throttled decision.
            # Initialized to -1.0 as a sentinel meaning "no decision yet".
            # The time gate uses this sentinel to avoid throttling the very first step.
            "last_update_time": -1.0,
            # Cached Z-scored SGU signals from the last non-throttled step.
            # These serve as FALLBACK values when the backtest cursor runs past
            # the end of the execution arrays (see _get_signals), and also as
            # diagnostic output for logging and analysis.
            "last_s1_norm": 0.0,
            "last_s2_norm": 0.0,
            # Last action index chosen by the RL agent.
            # Initialized to the last action index (n_actions - 1), which by
            # convention is "hold" in generic mode.  This means the agent's
            # implicit initial stance is "do nothing", which is the safest
            # default before the first real decision.
            "last_action_idx": self.n_actions - 1,  # default = last = hold
        }
        # Reset backtest cursor if applicable.
        # In simulation mode, there is no cursor --- signals are computed on-the-fly.
        if self.mode == "backtest":
            self.cursor = 0

    def reset(self):
        """
        Reset the controller to its initial state.

        Call this between episodes / backtest runs to:
        - Reset the backtest cursor to 0 (replay signals from the beginning).
        - Clear all throttle counters (step, TOB, time).
        - Clear cached signals and last action.

        Normalization statistics (mu, std) and mode configuration are preserved.

        WHEN to call this:
        - Before starting a new backtest on the same execution data.
        - Before starting a new episode in simulation mode.
        - After modifying execution arrays (e.g., loading a different test split).

        WHEN NOT to call this:
        - In the middle of an episode (you'd lose the cursor position and
          throttle synchronization).
        """
        self._init_state()

    # =====================================================================
    # HELPER METHODS
    # =====================================================================

    def _standardize(self, val: float, mu: float, std: float) -> float:
        """
        Apply Z-score normalization using frozen training statistics.

        Computes:  z = clip( (val - mu) / std,  -clip, +clip )

        This is the core normalization primitive.  It is deliberately simple
        and stateless --- all the statefulness (which mu/std to use) is managed
        by the caller (_get_signals).

        WHY Z-score and not min-max or rank normalization?
        - Z-score is shift-and-scale invariant: adding a constant to all
          training values doesn't change the normalized test values.
        - It naturally handles signals that may drift over time (common in
          financial data): even if the mean shifts between train and test,
          the Z-score still captures "how unusual is this value relative to
          what we saw in training?"
        - It plays well with neural networks: zero-centered inputs with
          unit variance are in the sweet spot for most activation functions.

        WHY clip after standardizing?
        - Outlier protection.  A Z-score of 10 means "10 standard deviations
          from the training mean" --- an event so rare it's likely noise or a
          data error.  Feeding z=10 into a neural network can saturate
          activations and destabilize gradients.
        - The default clip=3.0 retains 99.7% of a normal distribution's
          probability mass, so we lose almost no information for well-behaved
          signals.

        Parameters
        ----------
        val : float
            Raw SGU prediction value.
        mu : float
            Mean of the training set predictions (frozen at construction).
        std : float
            Std dev of the training set predictions (frozen at construction).

        Returns
        -------
        float
            Z-scored and clipped signal value.
        """
        z = (val - mu) / std
        # Apply symmetric clipping if a clip bound was configured.
        # signal_clip=None disables clipping entirely (not recommended).
        # We use max/min instead of np.clip for speed --- this function is called
        # on every non-throttled step, so avoiding numpy overhead matters.
        if self.signal_clip is not None:
            z = max(-self.signal_clip, min(self.signal_clip, z))
        return float(z)

    @staticmethod
    def _state_get_first(state: Dict[str, Any], keys: Tuple[str, ...], default: Any) -> Any:
        """
        Return the first non-None value found in ``state`` for a list of keys.

        This is used to support multiple naming conventions for the same field
        (e.g., ``bid_size`` vs ``bidsize``) while keeping one canonical access
        point in the controller.
        """
        for k in keys:
            if k in state and state[k] is not None:
                return state[k]
        return default

    def _extract_tob_key(self, state: Dict[str, Any]) -> Tuple[int, int, float, float]:
        """
        Build the TOB fingerprint used by the TOB throttle gate.

        Priority order:
        1) Environment keys (``*_env``) to avoid MM self-impact in backtests.
        2) Generic keys (``bid_size``/``ask_size``).
        3) Legacy keys (``bidsize``/``asksize``).
        """
        bb_raw = self._state_get_first(state, ("best_bid_env", "best_bid"), -1)
        ba_raw = self._state_get_first(state, ("best_ask_env", "best_ask"), -1)
        bs_raw = self._state_get_first(
            state,
            ("bid_size_env", "bidsize_env", "bid_size", "bidsize"),
            0.0,
        )
        as_raw = self._state_get_first(
            state,
            ("ask_size_env", "asksize_env", "ask_size", "asksize"),
            0.0,
        )

        try:
            bb = round(float(bb_raw), 10)
        except Exception:
            bb = -1
        try:
            ba = round(float(ba_raw), 10)
        except Exception:
            ba = -1
        try:
            bs = float(bs_raw)
        except Exception:
            bs = 0.0
        try:
            as_ = float(as_raw)
        except Exception:
            as_ = 0.0

        # One-time warning: if env sizes are unavailable and the state includes
        # MM's own orders, TOB window sync may drift from LOBSTER windows.
        if (
            self.use_tob
            and (self.mode == "backtest")
            and (not hasattr(self, "_warned_tob_env_size_fallback"))
        ):
            has_env_bid_size = ("bid_size_env" in state) or ("bidsize_env" in state)
            has_env_ask_size = ("ask_size_env" in state) or ("asksize_env" in state)
            exclude_self = bool(state.get("exclude_self_from_state", False))
            if (not has_env_bid_size or not has_env_ask_size) and (not exclude_self):
                warnings.warn(
                    "TOB gate is falling back to non-env size keys "
                    "('bid_size'/'bidsize', 'ask_size'/'asksize'). "
                    "With exclude_self_from_state=False this can drift from "
                    "LOBSTER TOB windows. For strict sync, provide env size keys "
                    "or run with exclude_self_from_state=True.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            self._warned_tob_env_size_fallback = True

        return (bb, ba, bs, as_)

    def _get_signals(self, state: Dict[str, Any]) -> Tuple[float, float]:
        """
        Retrieve the next pair of SGU signals and normalize them.

        In backtest mode, reads the next values from the pre-computed arrays
        and advances the cursor.  In simulation mode, calls the live inference
        functions on the current simulator state.

        Both paths normalize the raw signal using the frozen training statistics.

        THE CURSOR MECHANISM (backtest mode):
        ------------------------------------
        The cursor is an integer index into self.exec1 and self.exec2.  It
        starts at 0 and advances by EXACTLY 1 each time this method is called.

        Crucially, this method is ONLY called from act() on NON-THROTTLED steps.
        This means the cursor advances once per actual decision, not once per
        simulator message.  If the simulator sends 10,000 messages but only
        500 pass the throttle gates, the cursor advances 500 times --- perfectly
        aligned with the 500 SGU windows that were pre-computed.

        WHY this alignment matters:
        Each SGU prediction was computed over a specific TOB window (e.g.,
        "the next 10 TOB changes starting from event #3,200").  The throttle
        gates ensure the agent makes decisions at the same frequency as these
        windows.  If the cursor advanced on every message regardless of
        throttling, signal [i] would correspond to the wrong market state.

        END-OF-STREAM BEHAVIOR:
        When cursor >= max_steps (all pre-computed signals consumed), we
        return the LAST KNOWN cached signals instead of raising an error.
        This graceful degradation handles edge cases where the simulator
        runs slightly longer than the signal array (e.g., a partial last
        window, or a few trailing messages after the last SGU window ends).

        Parameters
        ----------
        state : dict
            Current simulator state dict (used in simulation mode only).

        Returns
        -------
        tuple of (float, float)
            (z_sgu1, z_sgu2): Z-scored and clipped SGU signals.
        """
        # --- BACKTEST: Read next value from pre-computed arrays ---
        if self.mode == "backtest":
            if self.cursor < self.max_steps:
                # Read the raw (un-normalized) SGU predictions at the current
                # cursor position.  These values are in the original model
                # output scale (e.g., predicted returns, probabilities, etc.).
                raw1 = self.exec1[self.cursor]
                raw2 = self.exec2[self.cursor]

                # FIX 4 (MEDIUM): Guard against NaN in pre-computed signal arrays.
                # NaN can appear when the SGU model failed to produce a prediction
                # for certain windows (e.g., insufficient data within the window).
                # Without this check, NaN propagates silently through Z-scoring
                # into the agent's state, potentially causing NaN actions or gradients.
                if np.isnan(raw1):
                    warnings.warn(
                        f"NaN in SGU1 execution array at cursor={self.cursor}. "
                        f"Using training mean as neutral fallback.",
                        RuntimeWarning, stacklevel=3,
                    )
                    raw1 = self.mu1  # z = (mu - mu) / std = 0.0
                if np.isnan(raw2):
                    warnings.warn(
                        f"NaN in SGU2 execution array at cursor={self.cursor}. "
                        f"Using training mean as neutral fallback.",
                        RuntimeWarning, stacklevel=3,
                    )
                    raw2 = self.mu2

                # Normalize using TRAINING statistics ("the ruler").
                # Even though raw1/raw2 may come from the test set, we ALWAYS
                # use mu1/std1 and mu2/std2 computed from the training set.
                z1 = self._standardize(raw1, self.mu1, self.std1)
                z2 = self._standardize(raw2, self.mu2, self.std2)

                # Advance the cursor AFTER reading.  This post-increment
                # ensures that the next call to _get_signals reads the NEXT
                # position, maintaining a strict one-to-one mapping between
                # non-throttled decision steps and signal array positions.
                self.cursor += 1
                return z1, z2
            else:
                # End of stream: repeat last known signals.
                # This can happen if the simulator runs longer than the number
                # of pre-computed windows (e.g., partial last window).
                #
                # WHY repeat instead of returning 0.0?  Because the last known
                # signal is a better estimate of the current market regime than
                # a neutral zero.  If SGU1 was strongly positive (bullish) at
                # the end of the array, it's more informative to keep that
                # signal than to suddenly go neutral, which could cause the
                # agent to make an abrupt position change at the end of the
                # episode.
                return self.st["last_s1_norm"], self.st["last_s2_norm"]

        # --- SIMULATION: Compute signals on-the-fly ---
        else:
            # Call the live inference functions.  Each function receives the
            # full simulator state dict and returns a single float.
            #
            # WHY try/except instead of letting exceptions propagate?
            # In a live trading or simulation environment, a transient failure
            # in the SGU model (e.g., a missing feature, a NaN in the input)
            # should NOT crash the entire simulation.  Instead, we fall back
            # to the training mean (mu) and issue a warning.  This "fail soft"
            # behavior keeps the simulation running while alerting the user.
            #
            # FIX 2 (HIGH): The fallback uses the training mean (mu), NOT 0.0.
            # After Z-scoring, z = (mu - mu) / std = 0.0 --- a truly neutral
            # signal.  The previous fallback of raw=0.0 was INCORRECT because
            # z = (0 - mu) / std = -mu/std, which is a directional signal
            # (not neutral) whenever mu != 0.  This could silently bias the
            # agent toward one direction during inference failures.
            try:
                raw1 = float(self.fn_s1(state))
            except Exception as exc:
                warnings.warn(
                    f"SGU1 live inference failed: {exc}. "
                    f"Using training mean (mu1={self.mu1}) as neutral fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # FALLBACK: Use the training mean (mu) so that after Z-scoring,
                # z = (mu - mu) / std = 0.0  --- a truly neutral signal.
                # The previous fallback of 0.0 was INCORRECT: z = (0 - mu) / std = -mu/std,
                # which is a directional signal, not neutral.
                raw1 = self.mu1
            if np.isnan(raw1):
                if not getattr(self, "_nan_warned_s1", False):
                    warnings.warn(
                        f"SGU1 returned NaN. Using training mean (mu1={self.mu1}) as fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._nan_warned_s1 = True
                raw1 = self.mu1

            try:
                raw2 = float(self.fn_s2(state))
            except Exception as exc:
                warnings.warn(
                    f"SGU2 live inference failed: {exc}. "
                    f"Using training mean (mu2={self.mu2}) as neutral fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # FALLBACK: Use the training mean (mu) so that after Z-scoring,
                # z = (mu - mu) / std = 0.0  --- a truly neutral signal.
                # The previous fallback of 0.0 was INCORRECT: z = (0 - mu) / std = -mu/std,
                # which is a directional signal, not neutral.
                raw2 = self.mu2
            if np.isnan(raw2):
                if not getattr(self, "_nan_warned_s2", False):
                    warnings.warn(
                        f"SGU2 returned NaN. Using training mean (mu2={self.mu2}) as fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._nan_warned_s2 = True
                raw2 = self.mu2

            # Same normalization as backtest mode --- the ruler is the same.
            z1 = self._standardize(raw1, self.mu1, self.std1)
            z2 = self._standardize(raw2, self.mu2, self.std2)

            return z1, z2

    def _check_throttling(self, state: Dict[str, Any]) -> bool:
        """
        Evaluate all throttle gates and return whether to suppress this step.

        Three independent gates can each suppress a decision:

        1. **Step gate**: fires only every N-th call to ``act()``.
        2. **TOB gate**: fires only after N top-of-book changes since the
           last decision.  A TOB change is detected by comparing the tuple
           (best_bid, best_ask, bid_size, ask_size) to the previous value.
        3. **Time gate**: fires only after T seconds have elapsed since the
           last decision.

        If ANY gate is active and its condition is not met, the step is
        throttled (returns True), and ``act()`` will return ``("hold",)``.

        GATE INTERACTION SEMANTICS:
        The gates use AND logic: ALL active gates must pass for a decision
        to be made.  For example, if both TOB and time gates are active,
        a decision requires BOTH enough TOB changes AND enough elapsed time.
        This is intentional --- it prevents the agent from acting on a single
        large TOB move that happened 0.001 seconds after the last decision.

        WHY these specific three gates?
        - Step-based: Simple baseline, useful for debugging and unit tests.
        - TOB-based: Aligns decisions with MEANINGFUL market events. In a
          fast market, TOB changes are frequent (many decisions per second);
          in a quiet market, they are rare (fewer decisions).  This adaptive
          pacing naturally adjusts to market activity.
        - Time-based: Provides a MINIMUM decision spacing regardless of
          market activity.  Prevents the agent from overtrading in volatile
          markets where TOB changes happen every millisecond.

        Parameters
        ----------
        state : dict
            Current simulator state.  Expected keys for TOB detection include
            ``best_bid``/``best_ask`` and either ``bid_size``/``ask_size`` or
            ``bidsize``/``asksize``. If present, ``best_bid_env``/
            ``best_ask_env`` and ``*_env`` sizes are preferred for synchronization.
            Expected key for time gate: ``time``.

        Returns
        -------
        bool
            True if the step should be throttled (skipped), False if a new
            decision should be made.
        """
        # 1. Update TOB change counter (if TOB throttling is enabled).
        #    We compare the current top-of-book 4-tuple to the last observed
        #    value.  Any change in price OR size counts as a TOB move,
        #    consistent with the SGU window definition in LOB_processor.
        #
        #    WHY include size changes, not just price changes?
        #    Because a large order being added or removed at the best price
        #    changes the queue depth and thus the fill probability --- even if
        #    the price doesn't move.  This is a material change for a market
        #    maker.  The SGU windowing in LOB_processor uses the same definition,
        #    so counting size changes here keeps the throttle aligned with the
        #    signal generation.
        if self.use_tob:
            # The 4-tuple acts as a hashable fingerprint of the current TOB.
            # Python tuple comparison is fast and handles floats correctly.
            key = self._extract_tob_key(state)

            # Compare to previous TOB fingerprint.  If different, increment
            # the move counter.  Note: on the very first call, last_env_tob_key
            # is None, so any real key will differ --- the first TOB observation
            # always counts as a "move", which is the desired behavior.
            if self.st["last_env_tob_key"] != key:
                self.st["moves_tob"] += 1
                self.st["last_env_tob_key"] = key

        # 2. Evaluate each gate.  If ANY gate blocks, the step is throttled.
        #    We start with False (not throttled) and flip to True if any gate
        #    says "wait".  This is equivalent to: allow = gate1_ok AND gate2_ok AND gate3_ok.
        is_throttled = False

        # Gate 1: Step-based --- only act every N-th step.
        # Uses modulo arithmetic: step 0 passes (0 % N == 0), then step N, 2N, etc.
        # When throttle_every_n_steps=1, every step passes (k % 1 == 0 always).
        if self.st["step_counter"] % self.throttle_every_n_steps != 0:
            is_throttled = True

        # Gate 2: TOB-based --- require at least N TOB changes since last decision.
        # The counter (moves_tob) resets to 0 after each non-throttled decision
        # (see _reset_throttles), so it accumulates changes within each
        # decision interval.
        if self.use_tob and (self.st["moves_tob"] < self.threshold_tob):
            is_throttled = True

        # Gate 3: Time-based --- require at least T seconds since last decision.
        #
        # FIX 3 (MEDIUM): When the sentinel (-1.0) is present, we now simply
        # skip the time gate entirely (let the step through) instead of
        # initializing last_update_time here.  The reference time will be set
        # later in _reset_throttles() when the first actual (non-throttled)
        # decision is made.  This prevents the time gate from anchoring its
        # reference to a step that was throttled by ANOTHER gate (e.g., the
        # step gate).  Previously, the else branch set last_update_time = now
        # even when the step would be throttled by another gate, causing the
        # time gate to measure from the wrong moment.
        if self.use_time:
            now = state.get("time", 0.0)
            try:
                if np.isnan(now):
                    is_throttled = True
                    return is_throttled
            except (TypeError, ValueError):
                pass
            if self.st["last_update_time"] < 0:
                # First step: no previous decision exists yet.
                # Do NOT throttle, and do NOT initialize the reference here.
                # The reference will be set in _reset_throttles() when the
                # first actual (non-throttled) decision is made.
                # This prevents the time gate from anchoring to a throttled step.
                pass  # explicitly do nothing --- let the step through
            elif (now - self.st["last_update_time"]) < self.min_time:
                is_throttled = True

        return is_throttled

    def _reset_throttles(self, state: Dict[str, Any]):
        """
        Reset throttle counters after a non-throttled decision is made.

        Called at the end of ``act()`` when the agent actually produces a new
        action (i.e., the step was not throttled).

        WHY reset moves_tob to 0 (not 1)?
        Because after a decision is made, we want to wait for threshold_tob
        NEW changes before the next decision.  Starting at 0 ensures we
        count exactly threshold_tob changes.  If we started at 1, we'd
        only require (threshold_tob - 1) changes, which would be off-by-one.

        WHY update last_update_time here (not in act)?
        To keep the timing reference as close as possible to the actual
        decision moment.  If we updated it at the START of act(), the time
        gate would measure from "when act() was entered" rather than "when
        the decision was finalized", which could be slightly different if
        signal retrieval or agent inference takes time.

        Parameters
        ----------
        state : dict
            Current simulator state.  Used to record the timestamp for the
            time-based throttle gate.
        """
        self.st["moves_tob"] = 0
        self.st["last_update_time"] = state.get("time", 0.0)

    # =====================================================================
    # AGENT INTERFACE --- Overridden by the factory's monkey-patch
    #
    # This section defines the contract between the controller and the RL
    # agent.  The controller calls _get_agent_action_idx(augmented_state)
    # and expects an integer action index in return.  The ACTUAL agent
    # implementation is injected at runtime via the factory function.
    #
    # WHY this design?  It follows the Strategy Pattern (GoF):
    # - The controller defines the ALGORITHM SKELETON (throttle -> signal ->
    #   augment -> decide -> decode).
    # - The specific DECISION STRATEGY (neural network, rule-based, random)
    #   is plugged in from outside.
    # - This allows the same controller to work with DQN, PPO, A2C, or even
    #   a simple hand-coded heuristic, without any code changes.
    # =====================================================================

    def _get_agent_action_idx(self, augmented_state: Dict[str, Any]) -> int:
        """
        Query the RL agent for an action index given the augmented state.

        This is a **placeholder** that always returns the last action index
        (hold in generic mode).  In production, the factory function replaces
        this method with a lambda that calls the actual RL agent function::

            controller._get_agent_action_idx = lambda s: rl_act_fn(s)

        WHY a placeholder instead of raising NotImplementedError?
        Because a "do nothing" default is SAFE.  If someone accidentally
        creates a controller without monkey-patching an agent, the policy
        will simply hold all positions --- it won't crash the simulator or
        place wild orders.  This is important for testing and debugging.

        Parameters
        ----------
        augmented_state : dict
            Simulator state dict augmented with ``"sgu1"`` and ``"sgu2"``
            keys containing the Z-scored SGU signals.

        Returns
        -------
        int
            Action index in [0, n_actions).
        """
        # Default: last action = hold (action 5 in 6-action generic mode).
        # Convention: the LAST action index is always "hold" / "do nothing".
        return self.n_actions - 1

    def learn(self, state, action, reward, next_state, done):
        """
        Hook for online RL training.

        Placeholder --- override in a subclass or monkey-patch to implement
        experience replay, policy gradient updates, etc.

        This method is exposed on the factory's return value as
        ``policy.learn(...)``, making it easy to integrate into a training
        loop without reaching into the controller internals:

            policy = Deep_RL_With_Signal_policy_factory(...)
            action = policy(state)
            reward = simulator.step(action)
            policy.learn(state, action, reward, next_state, done)

        In pure backtest/evaluation mode, this method is never called and
        does nothing.  In online training mode, the outer loop is expected
        to monkey-patch or subclass this with actual learning logic.

        Parameters
        ----------
        state : dict
            State before the action.
        action : int
            Action index chosen by the agent.
        reward : float
            Reward received after the action.
        next_state : dict
            State after the action.
        done : bool
            Whether the episode has ended.
        """
        pass

    # =====================================================================
    # MAIN DECISION PIPELINE: act()
    #
    # This is the HEART of the controller --- the method the simulator calls
    # on every step.  It orchestrates the full pipeline:
    #
    #   1. Increment step counter
    #   2. Check throttle gates    --> early return ("hold",) if throttled
    #   3. Retrieve SGU signals    --> advance cursor (backtest) or infer (sim)
    #   4. Augment state           --> add sgu1, sgu2 keys
    #   5. Query RL agent          --> get action index
    #   6. Reset throttle counters --> prepare for next decision interval
    #   7. Decode action           --> convert int to simulator tuple
    #
    # The method is intentionally kept LINEAR (no branching beyond the
    # throttle check) to make the flow easy to trace and debug.
    # =====================================================================

    def act(self, state: Dict[str, Any]) -> Tuple:
        """
        Main entry point: called by the simulator on every step.

        Pipeline:
        1. Increment step counter.
        2. Check throttle gates -> if throttled, return ("hold",).
        3. Retrieve and normalize SGU signals (advance cursor in backtest).
        4. Augment the simulator state with SGU signals.
        5. Query the RL agent for an action index.
        6. Reset throttle counters.
        7. Decode the action index into a simulator command tuple.

        Parameters
        ----------
        state : dict
            Current simulator state.  Expected keys depend on mode and
            throttling config, but typically include: ``best_bid``,
            ``best_ask``, ``bid_size``, ``ask_size``, ``time``, etc.

        Returns
        -------
        tuple
            Simulator command.  Examples:
            - ``("hold",)`` --- do nothing.
            - ``("place_bid", price)`` --- place a bid at the given price.
            - ``("place_bid_ask", bid_price, ask_price)`` --- two-sided quote.
            - ``("cancel_bid",)`` or ``("cancel_ask",)`` --- cancel an order.
        """
        # Every call to act() increments the counter, even if throttled.
        # This is essential for the step-based throttle gate (modulo check):
        # step_counter=0 passes, then we wait N-1 steps before the next pass.
        self.st["step_counter"] += 1

        # Step 1: Check throttle gates.
        # If throttled, return "hold" WITHOUT advancing the SGU cursor.
        # This keeps the offline signal stream synchronized with decision events.
        #
        # CRITICAL ORDERING: the throttle check MUST happen BEFORE _get_signals.
        # If we called _get_signals first, the cursor would advance even on
        # throttled steps, breaking the alignment between signal positions and
        # actual decision points.  In a backtest with 500 signals and 500 TOB
        # windows, this would cause signal[i] to be paired with the wrong window.
        if self._check_throttling(state):
            return ("hold",)

        # --- From here on, we are making a REAL decision (not throttled). ---

        # Step 2: Retrieve and normalize SGU signals.
        # In backtest mode, this advances the cursor by 1.
        # In simulation mode, this calls the live inference functions.
        s1, s2 = self._get_signals(state)
        # Cache the signals for:
        # (a) End-of-stream fallback (when cursor exceeds max_steps).
        # (b) External diagnostic access via self.st["last_s1_norm"].
        self.st["last_s1_norm"] = s1
        self.st["last_s2_norm"] = s2

        # Step 3: Augment the simulator state with the normalized SGU signals.
        # The RL agent receives these as additional features beyond the raw
        # market state (position, inventory, PnL, book snapshot, etc.).
        #
        # WHY state.copy() and not modify state in-place?
        # Because the simulator owns the `state` dict.  If we added keys to it
        # directly, those keys would persist in the simulator's internal state,
        # potentially causing confusion or subtle bugs (e.g., the simulator
        # might serialize its state and be surprised by extra keys).  Copying
        # ensures the simulator's state remains untouched.
        #
        # WHY a shallow copy (dict.copy()) and not a deep copy?
        # Performance.  The state dict contains only scalars (floats, ints, strings)
        # and occasionally small lists.  A shallow copy is sufficient because we
        # only ADD new keys ("sgu1", "sgu2"); we never modify existing values.
        # A deep copy would be wasteful and slow for no benefit.
        aug_state = state.copy()
        aug_state["sgu1"] = s1
        aug_state["sgu2"] = s2

        # Step 4: Query the RL agent.
        # In production, _get_agent_action_idx is monkey-patched by the factory
        # to call the actual trained agent.  The placeholder returns "hold".
        #
        # The agent receives the AUGMENTED state (with SGU signals) so it can
        # condition its decision on both market microstructure features AND
        # the directional/momentum signals.  This is the "hybrid" in "hybrid
        # controller" --- the agent sees both raw market data and processed signals.
        action_idx = self._get_agent_action_idx(aug_state)
        try:
            action_idx = int(action_idx)
        except (TypeError, ValueError):
            action_idx = self.n_actions - 1  # fallback to hold
        # Record the action for diagnostics and for the default "hold" fallback
        # in _get_agent_action_idx (though the latter doesn't currently use it).
        self.st["last_action_idx"] = action_idx

        # Step 5: Reset throttle counters and decode the action.
        # Reset BEFORE decode so that the throttle counters are clean for the
        # next interval.  Decode happens last because it's a pure function of
        # (action_idx, state) and doesn't affect internal state.
        self._reset_throttles(state)
        return self._decode_action(action_idx, aug_state)

    def _decode_action(
        self, action_idx: int, state: Dict[str, Any]
    ) -> Tuple:
        """
        Map an integer action index to a simulator command tuple.

        Two action spaces are supported:

        **"generic"** (default, 6 actions):
            0 -> place_bid at best_bid
            1 -> place_ask at best_ask
            2 -> place_bid_ask at (best_bid, best_ask) --- two-sided quote
            3 -> cancel_bid
            4 -> cancel_ask
            5 -> hold (no-op)

        **"pure_mm"** (N actions, each a spread level):
            action_idx = level -> place_bid_ask at:
                bid = best_bid  - level * tick_size
                ask = best_ask  + level * tick_size
            Level 0 quotes at the touch (tightest spread).
            Higher levels = wider spreads = more adverse-selection protection
            but lower fill probability.

        WHY two action spaces?
        - "generic" is flexible: the agent can choose to quote one side only,
          cancel orders, or hold.  This is suitable for strategies that may
          want to go directional (bid only or ask only) based on SGU signals.
        - "pure_mm" is specialized for pure market making: the agent ALWAYS
          quotes both sides, and the only choice is HOW WIDE to quote.  This
          simpler action space can be easier for the RL agent to learn in,
          because every action results in a two-sided quote (no "wasted"
          actions like cancel when no orders exist).

        WHY "place_bid_ask" and not two separate "place_bid" + "place_ask"?
        In real market making, you almost always want to quote both sides
        simultaneously.  The simulator's "place_bid_ask" is an ATOMIC operation
        that updates both sides in one step, avoiding the risk of being
        momentarily one-sided between two separate operations.

        Parameters
        ----------
        action_idx : int
            Action index from the RL agent.  Must be in [0, n_actions).
        state : dict
            Augmented simulator state (includes sgu1, sgu2 keys).

        Returns
        -------
        tuple
            Simulator command tuple.
        """
        # FIX 8 (LOW): Guard against out-of-range action indices.
        # This can happen if the RL agent has a bug (e.g., off-by-one in its
        # output layer, or a numerical issue causing an extreme Q-value to
        # select an invalid argmax).  Without this check, an out-of-range
        # index would silently fall through to "hold" in generic mode but
        # could produce a nonsensical spread level in pure_mm mode.
        # Issuing a warning makes the bug visible in logs.
        if action_idx < 0 or action_idx >= self.n_actions:
            warnings.warn(
                f"Action index {action_idx} out of valid range [0, {self.n_actions}). "
                f"Defaulting to hold. This likely indicates a bug in the RL agent.",
                RuntimeWarning, stacklevel=2,
            )
            return ("hold",)

        # Extract best bid and ask prices from the state.
        # These are the current top-of-book prices that the simulator provides.
        mkt_bb = state.get("best_bid")
        mkt_ba = state.get("best_ask")

        # Safety: if no valid market prices, do nothing.
        # This can happen at the very start of a simulation before the order
        # book has been initialized, or in edge cases where the book is empty
        # (all orders cancelled).  Attempting to place orders without valid
        # reference prices would produce nonsensical results.
        if mkt_bb is None or mkt_ba is None:
            return ("hold",)

        # --- Generic action space (default) ---
        # Six discrete actions covering the full range of market-making decisions:
        # quote one side, quote both sides, cancel one side, or do nothing.
        if self.action_mode == "generic":
            if action_idx == 0:
                # Place a bid (buy order) at the current best bid price.
                # This "joins the queue" at the touch --- highest priority for fills
                # but most exposed to adverse selection.
                return ("place_bid", mkt_bb)
            if action_idx == 1:
                # Place an ask (sell order) at the current best ask price.
                # Mirror of action 0 on the sell side.
                return ("place_ask", mkt_ba)
            if action_idx == 2:
                # Place both bid and ask --- the classic two-sided market-making quote.
                # This is the most common action for a market maker: earn the spread
                # by buying at the bid and selling at the ask.
                return ("place_bid_ask", mkt_bb, mkt_ba)
            if action_idx == 3:
                # Cancel the bid order.  Useful when the agent detects adverse
                # movement (e.g., SGU signals predict a price drop --- don't buy).
                return ("cancel_bid",)
            if action_idx == 4:
                # Cancel the ask order.  Mirror of action 3.
                return ("cancel_ask",)
            # action_idx == 5 or any unrecognized index -> hold
            # The "hold" action tells the simulator to leave all existing orders
            # unchanged.  This is the DEFAULT / SAFE action --- when in doubt,
            # do nothing.  Using it as the catch-all for unrecognized indices
            # provides robustness against bugs in the agent that produce
            # out-of-range action indices.
            return ("hold",)

        # --- Pure market-making action space ---
        # N actions, each representing a SPREAD LEVEL.  The agent always quotes
        # both sides; the only decision is how far from the touch to quote.
        elif self.action_mode == "pure_mm":
            level = action_idx
            # Compute the half-spread offset: how many ticks away from the
            # touch each side of the quote will be placed.
            #
            # level=0: bid at best_bid, ask at best_ask (quoting at the touch,
            #          tightest possible spread, highest fill probability but
            #          maximum adverse selection risk).
            # level=1: bid at best_bid - 1 tick, ask at best_ask + 1 tick
            #          (one tick wider, slightly less fill probability but
            #          better adverse selection protection).
            # level=N: progressively wider spreads.
            #
            # This parameterization is elegant because:
            # - The action space scales linearly with the number of spread levels.
            # - Each action has a clear economic interpretation.
            # - The agent learns a SINGLE number (how aggressive to be) rather
            #   than a complex combination of order types and prices.
            half_spread_offset = level * self.tick_size
            return (
                "place_bid_ask",
                mkt_bb - half_spread_offset,
                mkt_ba + half_spread_offset,
            )

        # Unknown action_mode -> safe fallback.
        # This should never happen if the constructor validated action_mode,
        # but defensive programming is cheap and prevents crashes.
        return ("hold",)


# =========================================================================
# FACTORY FUNCTION --- Main entry point for creating policy functions
#
# WHY A FACTORY FUNCTION INSTEAD OF DIRECT CONSTRUCTION?
# -------------------------------------------------------
# The simulator (MM_LOB_SIM) expects a simple callable:
#     policy(state: dict) -> tuple
#
# But the controller is a full object with state, configuration, and methods.
# The factory bridges this gap by:
#   1. Creating the controller object (with all its complexity).
#   2. Extracting the bound .act() method (a simple callable).
#   3. Attaching .controller and .learn references for external access.
#
# This gives the caller the SIMPLICITY of a function (just call policy(state))
# with the POWER of an object (inspect policy.controller.cursor, etc.).
#
# ALTERNATIVE CONSIDERED: Passing rl_act_fn to __init__.
# This would work, but would couple the controller class to the RL agent
# interface.  The monkey-patching approach keeps the class generic --- it can
# be tested with the default placeholder, subclassed for different agent
# interfaces, or composed in other ways.  The factory is a CONVENIENCE
# layer for the common case.
# =========================================================================

def Deep_RL_With_Signal_policy_factory(
    rl_act_fn: Callable[[Dict[str, Any]], int],
    sgu1_norm_data: Union[np.ndarray, list],
    sgu2_norm_data: Union[np.ndarray, list],
    # --- Mode ---
    mode: str = "backtest",
    # --- Offline args (backtest mode) ---
    sgu1_execution_data: Optional[Union[np.ndarray, list]] = None,
    sgu2_execution_data: Optional[Union[np.ndarray, list]] = None,
    # --- Online args (simulation mode) ---
    sgu1_live_inference_fn: Optional[Callable] = None,
    sgu2_live_inference_fn: Optional[Callable] = None,
    # --- All other controller config ---
    # **kwargs captures action_mode, n_actions, tick_size, signal_clip,
    # throttle_every_n_steps, use_tob_update, n_tob_moves, use_time_update,
    # min_time_interval, verbose --- anything DeepRLSignalController.__init__ accepts.
    # This keeps the factory signature stable even if the controller adds
    # new parameters in the future (Open/Closed Principle).
    **kwargs,
) -> Callable:
    """
    Factory function that creates a simulator-compatible policy function.

    This is the recommended way to instantiate the controller.  It:
    1. Creates a ``DeepRLSignalController`` with the given configuration.
    2. Monkey-patches ``_get_agent_action_idx`` to call ``rl_act_fn``.
    3. Returns the controller's ``act`` method as the policy function.

    The returned function has two extra attributes:
    - ``.controller``: reference to the ``DeepRLSignalController`` instance,
      for inspecting internal state (cursor position, cached signals, etc.).
    - ``.learn``: reference to the controller's ``learn`` method, for use
      in online RL training loops.

    Parameters
    ----------
    rl_act_fn : callable
        The RL agent's decision function.  Signature::

            rl_act_fn(state: dict) -> int

        where the returned int is an action index in [0, n_actions).
        The state dict will contain all simulator keys plus ``"sgu1"``
        and ``"sgu2"`` (Z-scored SGU signals).

        **Note**: For placeholder / dummy agents that always hold, return
        ``n_actions - 1`` (e.g., 5 for the default 6-action generic space),
        NOT a tuple like ``("hold",)``.  Returning a non-int will cause the
        action to fall through to the default "hold" case, which works but
        is fragile and breaks ``last_action_idx`` tracking.

    sgu1_norm_data : array-like
        SGU-1 predictions on the training set (for Z-score statistics).
    sgu2_norm_data : array-like
        SGU-2 predictions on the training set (for Z-score statistics).
    mode : str
        "backtest" or "simulation".
    sgu1_execution_data : array-like, optional
        SGU-1 predictions for the execution split (backtest mode only).
    sgu2_execution_data : array-like, optional
        SGU-2 predictions for the execution split (backtest mode only).
    sgu1_live_inference_fn : callable, optional
        Live SGU-1 inference function (simulation mode only).
    sgu2_live_inference_fn : callable, optional
        Live SGU-2 inference function (simulation mode only).
    **kwargs
        Additional keyword arguments forwarded to ``DeepRLSignalController``
        (e.g., ``action_mode``, ``tick_size``, ``signal_clip``,
        ``throttle_every_n_steps``, ``use_tob_update``, ``n_tob_moves``,
        ``use_time_update``, ``min_time_interval``, ``verbose``).

    Returns
    -------
    callable
        A policy function with signature ``(state: dict) -> tuple``.
        Compatible with ``MM_LOB_SIM`` as a policy function.
        Has ``.controller`` and ``.learn`` attributes.

    Examples
    --------
    Minimal backtest setup with a dummy agent that always holds::

        policy = Deep_RL_With_Signal_policy_factory(
            rl_act_fn=lambda s: 5,            # always hold
            sgu1_norm_data=train_sgu1_preds,
            sgu2_norm_data=train_sgu2_preds,
            mode="backtest",
            sgu1_execution_data=test_sgu1_preds,
            sgu2_execution_data=test_sgu2_preds,
        )
        action_tuple = policy(simulator_state)  # e.g., ("hold",)
        print(policy.controller.cursor)         # how far into the signal array

    Simulation mode with live XGBoost inference::

        policy = Deep_RL_With_Signal_policy_factory(
            rl_act_fn=dqn_agent.select_action,
            sgu1_norm_data=train_sgu1_preds,
            sgu2_norm_data=train_sgu2_preds,
            mode="simulation",
            sgu1_live_inference_fn=lambda s: xgb1.predict(featurize(s))[0],
            sgu2_live_inference_fn=lambda s: xgb2.predict(featurize(s))[0],
            use_tob_update=True,
            n_tob_moves=10,
        )
    """
    # 1. Create the controller instance.
    #    All configuration (normalization, mode, throttling, action space)
    #    is set up in the constructor.  After this call, the controller is
    #    fully configured but has a PLACEHOLDER agent (always returns "hold").
    controller = DeepRLSignalController(
        mode=mode,
        sgu1_norm_data=sgu1_norm_data,
        sgu2_norm_data=sgu2_norm_data,
        sgu1_execution_data=sgu1_execution_data,
        sgu2_execution_data=sgu2_execution_data,
        sgu1_live_inference_fn=sgu1_live_inference_fn,
        sgu2_live_inference_fn=sgu2_live_inference_fn,
        **kwargs,
    )

    # 2. Monkey-patch the RL agent function into the controller.
    #    This replaces the placeholder ``_get_agent_action_idx`` (which always
    #    returns "hold") with the actual trained agent.
    #
    #    WHY monkey-patch instead of passing to __init__?
    #    - The controller class is designed to be agent-agnostic: it handles
    #      signals, throttling, and action decoding.  The agent is pluggable.
    #    - Assigning to the instance attribute shadows the class method.
    #      Python calls the lambda with just (state,) --- no auto-self injection,
    #      because instance attributes are not descriptors.
    #
    #    TECHNICAL DETAIL: Why does this work?
    #    In Python, when you access controller._get_agent_action_idx, Python
    #    first checks the INSTANCE __dict__, then the CLASS __dict__.  By
    #    assigning a lambda to the instance, it shadows the class method.
    #    Unlike class methods, instance attributes are NOT descriptors, so
    #    Python does NOT inject `self` as the first argument.  The lambda
    #    receives only the explicit argument (state).  This is exactly what
    #    we want: rl_act_fn(state) is called directly, without the controller
    #    instance being passed.
    #
    #    WHY wrap in a lambda instead of assigning rl_act_fn directly?
    #    Defensive practice: the lambda adds a layer of indirection that
    #    protects against edge cases where rl_act_fn might be a bound method
    #    or a callable object with unexpected __get__ behavior.  The lambda
    #    guarantees a simple function call regardless of rl_act_fn's type.
    controller._get_agent_action_idx = lambda state: rl_act_fn(state)

    # 3. Return the bound `act` method as the policy function.
    #    Attach `.learn` and `.controller` for external access.
    #
    #    WHY act_func = controller.act (a bound method)?
    #    Because the simulator expects a CALLABLE, not an object.  A bound
    #    method is a callable that carries its `self` reference --- so when
    #    the simulator calls act_func(state), Python automatically passes
    #    the controller instance as `self`.  The caller doesn't need to know
    #    about the controller at all.
    #
    #    WHY attach .controller and .learn as attributes on the function?
    #    Python functions (and bound methods) are objects --- you can attach
    #    arbitrary attributes to them.  This lets the caller access the
    #    controller's internals (for logging, debugging, or online training)
    #    without breaking the simple callable interface:
    #       policy(state)                       -> make a decision
    #       policy.controller.cursor            -> inspect signal position
    #       policy.controller.st["last_s1_norm"]-> inspect last signal
    #       policy.learn(s, a, r, s', done)     -> online training step
    # WHY a wrapper class instead of setting attributes on the bound method?
    # In Python, bound methods (like controller.act) are ephemeral descriptor
    # objects --- they do NOT support arbitrary attribute assignment.  Doing
    #     act_func = controller.act
    #     act_func.learn = controller.learn   # <- raises AttributeError!
    # fails because bound methods lack a __dict__.  A thin wrapper class with
    # __call__ solves this: it IS a proper object (has __dict__), so we can
    # attach .learn and .controller, and it IS callable (has __call__), so the
    # simulator can still do policy(state) as before.
    class _PolicyCallable:
        """Thin callable wrapper around controller.act with extra attributes."""
        __slots__ = ("controller", "learn", "_act")

        def __init__(self, ctrl):
            self._act = ctrl.act
            self.controller = ctrl
            self.learn = ctrl.learn

        def __call__(self, state):
            return self._act(state)

    return _PolicyCallable(controller)
