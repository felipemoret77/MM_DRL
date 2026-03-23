#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLFT_policy_factory_enhanced.py — Enhanced GLFT Market-Making Policy
=====================================================================

**Author's Note**: This module implements an enhanced, production-grade
version of the Guéant–Lehalle–Fernandez-Tapia (GLFT) optimal market-making
model for use inside a LOBSTER event-driven limit order book simulator.


Academic Background & Theoretical Foundations
─────────────────────────────────────────────

The field of optimal market making was pioneered by Ho & Stoll (1981),
who first formulated the dealer's quoting problem as a stochastic control
problem with inventory risk.  Avellaneda & Stoikov (2008) modernised the
framework using continuous-time stochastic control over a diffusive
mid-price process with Poisson order arrivals:

    Avellaneda, M. & Stoikov, S. (2008).
    "High-Frequency Trading in a Limit Order Book."
    Quantitative Finance, 8(3), 217–224.

The GLFT extension generalises the Avellaneda–Stoikov solution to
handle finite-horizon inventory penalties and more realistic arrival
rate specifications:

    Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2012).
    "Optimal Portfolio Liquidation with Limit Orders."
    SIAM Journal on Financial Mathematics, 3(1), 740–764.

    Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013).
    "Dealing with the Inventory Risk: A Solution to the Market
    Making Problem."  Mathematics and Financial Economics, 7(4), 477–507.

The model assumes:
    1. Mid-price S_t follows arithmetic Brownian motion:
           dS_t = σ dW_t
       where σ is the instantaneous volatility (ticks per √second).

    2. Market orders arrive on each side as independent Poisson processes
       whose intensity decays exponentially with the distance δ between
       the limit order and the current mid-price:
           λ(δ) = A · exp(−κ · δ)
       where A is the base arrival rate at δ=0 (MOs per second touching
       the best quote) and κ governs how rapidly fill probability decays
       with distance.  This is sometimes called the "exponential fill
       probability" assumption.

    3. The market maker (MM) holds inventory q_t and maximises expected
       exponential utility of terminal wealth W_T:
           max E[ −exp(−γ · W_T) ]
       where γ > 0 is the risk aversion parameter.  The exponential
       (CARA) utility makes the problem analytically tractable because
       the certainty equivalent decomposes additively.

Under these assumptions, the Hamilton–Jacobi–Bellman (HJB) equation
for the value function V(t, x, q, S) admits a separation:
    V(t, x, q, S) = −exp(−γ(x + q·S)) · h(t, q)
where h(t, q) captures the inventory-dependent component.

The optimal controls (bid/ask offsets from fair value) are:

    δ*_bid = (1/κ)·ln(1 + γΔ/κ) + (γΔ/2κ)·√(σ²/AΔκ · (1+γΔ/κ)^(κ/γΔ+1))·q
    δ*_ask = (1/κ)·ln(1 + γΔ/κ) + (γΔ/2κ)·√(σ²/AΔκ · (1+γΔ/κ)^(κ/γΔ+1))·q

which can be decomposed into:

    q_bid = S − (half_spread + skew · q)
    q_ask = S + (half_spread − skew · q)

where:
    half_spread = c₁ + (Δ/2) · c₂ · σ     [base width, independent of inventory]
    skew        = c₂ · σ                    [inventory penalty per unit]

    c₁ = (1 / ξΔ) · ln(1 + ξΔ/κ)          [HJB-derived width coefficient]
    c₂ = √(γ / (2AΔκ)) · (1 + ξΔ/κ)^((κ/ξΔ + 1)/2)   [HJB-derived skew coeff]
    ξ  = ε (terminal penalty) or γ (infinite horizon)
    Δ  = lot size (default 1)

**Key insight**: the spread is symmetric in δ but asymmetric in q:
positive inventory widens the bid and tightens the ask, making it
more likely that the MM sells rather than buys — naturally reverting
inventory toward zero.  The larger γ, the more aggressive this reversion.

Practical interpretations of the parameters:
    - γ ↑  → wider spreads, stronger inventory reversion, more conservative
    - A ↑  → thinner spreads (more competition for fills at the front of queue)
    - κ ↑  → narrower spreads (MOs decay faster with depth → must quote tighter)
    - σ ↑  → wider spreads (more mid-price risk to compensate for)


Extensions Implemented in This Module
──────────────────────────────────────

1. **Microprice Fair Value** (``use_micro=True``):

   Replaces the simple midprice with the n-level volume-weighted microprice
   as proposed in:

       Stoikov, S. (2018).
       "The Micro-Price: A High-Frequency Estimator of Future Prices."
       Quantitative Finance, 18(12), 1959–1966.

   The microprice is a superior estimator of the "true" efficient price
   because it incorporates order book imbalance information.  The formula
   used here generalises the single-level microprice to n depth levels:

       μ = Σ_{k=0}^{n-1} w_k · (V^bid_k · P^ask_k + V^ask_k · P^bid_k)
           ─────────────────────────────────────────────────────────────────
                         Σ_{k=0}^{n-1} w_k · (V^bid_k + V^ask_k)

   where:
       P^bid_k, P^ask_k   = price at depth level k (0 = best)
       V^bid_k, V^ask_k   = volume (number of shares resting) at level k
       w_k = decay^k       = exponential decay weight (1.0 = uniform)

   **Economic intuition**: when the bid side has 500 lots and the ask side
   has only 100 lots, the bid is thicker and the ask is thinner.  The
   next trade is more likely to be a buy (aggressor hitting the thin ask),
   so the price is likely to move up.  The microprice captures this by
   shifting the reference price toward the heavier side, and the GLFT
   model then proactively adjusts quotes accordingly — tightening the ask
   (to capture the expected upward move) and widening the bid (to reduce
   adverse selection on the buy side).

2. **Online Recalibration** (``update_params=True``):

   Traditional GLFT implementations estimate parameters once offline and
   hold them static.  In real markets, microstructure regimes shift
   intraday (e.g., lunch lull vs. closing auction), so static parameters
   degrade within minutes.  This module recalibrates live:

   - **σ (mid-price volatility)**: Estimated as the Root Mean Square (RMS)
     of mid-price returns measured at each throttle gate, then rescaled
     from per-gate to per-second units:

         σ̂_per_sec = RMS(ΔP_gate) · √(1 / mean(ΔT_gate))

     The RMS formulation measures deviation from zero, NOT from the sample
     mean.  This is deliberate: it implements the "semivariance" concept.
     If the price is trending (e.g., all returns are +1 tick), the standard
     deviation is zero (no dispersion around the mean), but the RMS correctly
     reports the risk as σ > 0 (all returns contribute to inventory risk
     regardless of direction).

     For two-sided mode (``use_two_sided=True``), we compute directional
     semivariances:
         σ²_ask = (1/N) Σ max(r_i, 0)²    [upward moves → risk for ask LOs]
         σ²_bid = (1/N) Σ |min(r_i, 0)|²   [downward moves → risk for bid LOs]

     **FIX 4 (Semivariance)**: The original code used ``arr[arr > 0]``
     (boolean filtering), which shrank the array from N to ~N/2 elements,
     causing ``np.mean`` to divide by a smaller denominator and inflating σ
     by ~√2.  The corrected version uses ``np.maximum(arr, 0.0)`` — zeros
     the opposite side but preserves all N elements, yielding the correct
     "directional variance per unit time" interpretation.

   - **A, κ (market-order intensity parameters)**: Estimated by observing
     ALL market orders on the public tape — not just the MM's own fills.
     Using only own fills would create a **feedback death-spiral**: wide
     quotes → fewer fills → even higher estimated κ → even wider quotes →
     no fills at all.  Instead, we treat every MO in the market as a
     sample from the exponential intensity function λ(δ) = A·exp(−κδ).

     Two estimation methods are available (selected by ``use_censored_calib``):

     **(a) WLS Tail-Counting** (default, fast, incremental):
         For each MO, record how deep it penetrated the book.  Build
         cumulative tail counts: tail[k] = # MOs reaching depth ≥ k·½tick.
         Convert to intensity: λ̂[k] = tail[k] / elapsed_time.
         Fit log(λ̂) = log(A) − κδ via Weighted Least Squares.
         **FIX 1 (Ghost Starvation)**: After repeated decay cycles, deep
         buckets have tail[k] ≪ 1.0 but > 0.  These "ghost" observations
         pull the regression slope toward −∞ (κ → ∞), producing absurdly
         wide quotes.  Fixed by filtering out tail[k] < 1.0.

     **(b) Censored Waiting Times MLE** (``use_censored_calib=True``):
         Statistically rigorous method from survival analysis.  See the
         detailed description in Section 6 of this module and in
         ``censored_waiting_times_calib.py``.

   All new estimates are blended with existing parameters via EMA:
       param_new = blend · estimate + (1 − blend) · param_old
   This prevents parameter jumps from destabilising open positions.

3. **Two-Sided Asymmetric Parameters** (``use_two_sided=True``):

   Calibrates independent (A, κ, σ) triplets for bid and ask sides.

   **Economic rationale**: real markets exhibit structural asymmetry.  In
   a bear market, sell-side market orders are more frequent and more
   aggressive (deeper sweeps), so κ_ask > κ_bid and A_ask > A_bid.  The
   bid side faces lower execution pressure but higher adverse selection
   risk.  Symmetric parameters would average these regimes, over-quoting
   on the safe side and under-quoting on the dangerous side.  Separate
   parameters yield asymmetric half-spreads and skew, enabling the bot
   to be DEFENSIVE on the side with adverse selection risk and AGGRESSIVE
   on the side with high fill probability.

   All three features compose freely: any combination can be enabled.

4. **Multi-Day Support** (automatic day-boundary detection):

   When backtesting on concatenated multi-day LOBSTER data, the policy
   detects timestamp jumps backwards (e.g., 55799s → 36000s, indicating
   a new trading day) and automatically resets ALL calibration state.

   **Why this matters**: market microstructure changes dramatically
   between days.  Opening-auction dynamics (high volatility, wide spreads,
   aggressive MOs) differ fundamentally from closing dynamics (low
   volatility, narrow spreads, institutional VWAP flow).  Carrying over
   parameters from the previous day's close would cause the bot to quote
   with stale spreads for 20+ minutes until the EMA converges.  A clean
   reset with warmup re-observation avoids this regime mismatch.

5. **Terminal Inventory Liquidation** (``liquidate_terminal=True``):

   Optionally liquidates all open inventory at market prices when a day
   boundary is detected, simulating a firm that mandates flat positions
   overnight (common regulatory / risk management requirement).

   Liquidation is performed via aggressive market orders (``mm.cross_mo``)
   against the previous day's CLOSING order book, BEFORE the reset.  This
   ensures execution prices reflect realistic end-of-day slippage and
   market impact (the order book is still populated when the MOs are sent).
   The cash and inventory impact is recorded by the backtest engine as any
   other fill, correctly accounting for P&L from forced liquidation.

6. **Censored Waiting Times Calibration** (``use_censored_calib=True``):

   Replaces the online tail-counting A/κ estimation (method 2a above)
   with the statistically rigorous Censored Waiting Times MLE, computed
   entirely online inside the ``_MarketIntensityTracker``.  No external
   DataFrames are required.

   **Theoretical Background** (Survival Analysis / Censored MLE):

   The problem of estimating execution intensity from limit order
   fill/no-fill observations is a classic CENSORED DATA problem from
   survival analysis (Lawless, 2003; Klein & Moeschberger, 2003):

       Lawless, J. F. (2003).
       "Statistical Models and Methods for Lifetime Data."
       Wiley, 2nd edition.

   Within each throttle window [t₀, t₁), we place a hypothetical limit
   order at distance δ from mid and observe either:
     (a) A fill at time τ < ΔT         → uncensored observation (I=1)
     (b) No fill by window close (ΔT)  → RIGHT-CENSORED observation (I=0)
         The true fill time exists but is > ΔT; we only know τ > ΔT.

   Under the exponential waiting time model (memoryless Poisson arrivals):
       P(fill before t | δ) = 1 − exp(−λ(δ) · t)

   The MLE for λ from N windows with mixed censored/uncensored data is:

       λ̂(δ) = Σ_w I_w / Σ_w τ̃_w

   where:
       I_w    = 1 if fill observed, 0 if censored
       τ̃_w   = min(τ_w, ΔT_w)  = observed survival time (capped at window)

   This is the standard exponential censoring MLE (see Cox & Oakes, 1984,
   Chapter 2).  The denominator Σ τ̃ is the total "exposure time" — the
   cumulative time during which the LO was at risk of being filled.

   **Mid-Drift Correction**: Within a window, the mid-price drifts.  A
   hypothetical LO placed at price P_lo = mid₀ − δ at window start sees
   a changing effective distance from the current mid:
       Buy LO:  δ_eff(t) = δ + (mid(t) − mid₀)    [mid up → LO further away]
       Sell LO: δ_eff(t) = δ + (mid₀ − mid(t))     [mid down → LO further away]
   If δ_eff ≤ 0, the mid has crossed the LO price → trivial fill.

   After computing λ̂(δ_k) on a grid of depths, we fit:
       log(λ̂) = log(A) − κ · δ
   via Weighted Least Squares with weights = Σ τ̃ (proportional to the
   Fisher information of each MLE estimate — bins with more exposure
   time have more precise λ̂ estimates and should carry more weight).

   This implementation is **equivalent** to the batch method in
   ``censored_waiting_times_calib.py`` but operates fully online: raw MO
   events and window boundaries are accumulated in bounded deques inside
   the tracker, and the MLE + WLS fit is performed at each recalibration.

   Falls back to the WLS tail-counting method if censored fit fails
   (insufficient windows or degenerate data).

7. **Diagnostic Plotting** (``plot_recalib=True``):

   When enabled alongside censored calibration, maintains a **single
   dynamic figure** that is updated in-place at each recalibration
   (and at warmup completion).  The plot always shows the latest fit —
   no extra windows are created.  Useful for visually verifying that
   the exponential intensity model is a reasonable fit to the data.


Software Architecture
─────────────────────

**Factory + Closure Pattern**: The module is structured around the factory
function ``glft_enhanced_policy_factory()``, which:

    1. Validates all input parameters and raises informative errors
    2. Pre-computes the initial GLFT coefficients (c₁, c₂, half_spread, skew)
    3. Initialises all mutable state in closure variables
    4. Returns a lightweight ``policy(state, mm) -> Tuple`` callable

The returned ``policy`` function is the only public interface.  All mutable
state (parameters, coefficients, fill counts, throttle clocks, calibration
data) lives in the closure scope — invisible to the caller, preventing
accidental mutation.  This is equivalent to a stateful class but avoids
boilerplate and enforces encapsulation.

**Why not a class?**: The closure pattern produces a single callable that
exactly matches the backtest engine's ``mm_policy(state, mm) -> Tuple``
interface, with zero adapter code.  A class would require either a
``__call__`` method (adding indirection) or a wrapper lambda (hiding
the actual logic).  The factory also allows attaching debug accessors
(``policy.get_params()``, ``policy.print_stats()``) as function attributes.


Gate System (Throttling / Decision Hierarchy)
─────────────────────────────────────────────

To avoid excessive order modifications — which can trigger exchange message
rate limits (e.g., CME's 50 msgs/sec), increase transaction costs, and
amplify adverse selection — the policy routes decisions through a strict
4-gate priority cascade:

    ┌─────────────────────────────────────────────────────────────────┐
    │  GATE 1 — CRITICAL: Inventory limit breach / mode change.      │
    │    Fires IMMEDIATELY regardless of throttle.                   │
    │    Cancels the forbidden leg and places the correct leg.       │
    │    Suppressed entirely in MDP mode (strict gate discipline).   │
    │                                                                │
    │  GATE 2 — URGENT: Fill detected (inventory changed).           │
    │    Fires IMMEDIATELY to replenish the filled leg and maintain  │
    │    queue priority.  Every tick spent without a resting order    │
    │    is lost queue position.                                     │
    │    Suppressed entirely in MDP mode.                            │
    │                                                                │
    │  GATE 3 — THROTTLE: Hard rate-limiter.                         │
    │    Blocks action until at least one of:                        │
    │      (a) N events have occurred           (use_event_update)   │
    │      (b) T seconds have elapsed           (use_time_update)    │
    │      (c) N top-of-book changes observed   (use_tob_update)     │
    │    Exception: a completely "naked" MM (no orders at all) may   │
    │    bypass throttle to avoid missing the entire market.         │
    │                                                                │
    │  GATE 4 — NORMAL: Standard repricing at gate opening.          │
    │    Compares the newly computed optimal quotes with the cached   │
    │    prices.  If they differ, cancel and replace.  If identical, │
    │    hold (no message sent to exchange).                         │
    │    Triggers recalibration (sigma, A, kappa).                   │
    │    FIX 3: checks for forbidden legs even when prices match.    │
    └─────────────────────────────────────────────────────────────────┘

MDP Mode (``use_mdp=True``): Enforces strict Markov Decision Process
timing — actions ONLY occur when the throttle gate fires (Gate 4).
Gates 1 and 2 are suppressed.  This is useful when the policy is used
as the action space of a reinforcement learning agent, where the RL
framework expects a fixed decision cadence.

    FIX 3 — Forbidden Leg Leak:  In MDP mode, Gate 1 is suppressed, so
    a forbidden order (e.g., a bid order when the MM is at max long
    inventory) can persist indefinitely.  Gate 4 originally returned
    ("hold",) if the correct leg's price hadn't changed, ignoring the
    forbidden leg.  Fix: Gate 4 now checks for forbidden legs and acts
    on them even when the desired price hasn't changed.


References
──────────
[1] Ho, T. & Stoll, H. (1981). "Optimal dealer pricing under transactions
    and return uncertainty." J. Fin. Econ., 9(1), 47–73.
[2] Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a
    limit order book." Quant. Finance, 8(3), 217–224.
[3] Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2012). "Optimal
    portfolio liquidation with limit orders." SIAM J. Fin. Math., 3(1).
[4] Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). "Dealing
    with the inventory risk." Math. Fin. Econ., 7(4), 477–507.
[5] Stoikov, S. (2018). "The micro-price." Quant. Finance, 18(12).
[6] Cartea, Á., Jaimungal, S. & Penalva, J. (2015). "Algorithmic and
    High-Frequency Trading." Cambridge University Press.
[7] Guéant, O. (2017). "Optimal Market Making." Applied Mathematical
    Finance, 24(2), 112–154.
[8] Lawless, J. F. (2003). "Statistical Models and Methods for Lifetime
    Data." Wiley, 2nd ed.
[9] Cox, D. R. & Oakes, D. (1984). "Analysis of Survival Data."
    Chapman & Hall.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
from collections import deque
import math
import numpy as np


# =============================================================================
#
#  SECTION 1: GLFT COEFFICIENT COMPUTATION
#
#  Computes the closed-form optimal half-spread and inventory skew from
#  the GLFT model parameters (γ, κ, A, σ).
#
#  ─── Mathematical Derivation (sketch) ───
#
#  The MM's value function under exponential utility:
#      V(t, x, q, S) = −exp(−γ(x + q·S)) · h(t, q)
#
#  where x = cash, q = inventory, S = mid-price, γ = risk aversion.
#  Substituting into the HJB equation and solving for the optimal
#  bid/ask offsets δ*_b, δ*_a from fair value S:
#
#      0 = ∂h/∂t + A·exp(−κδ_b)·[h(q+Δ)/h(q)·exp(−γΔδ_b) − 1]
#                 + A·exp(−κδ_a)·[h(q−Δ)/h(q)·exp(−γΔδ_a) − 1]
#                 + ½σ²γ²q²Δ²·h
#
#  The first-order conditions w.r.t. δ_b and δ_a yield:
#
#      δ*_b = (1/γΔ)·ln[h(q)/h(q+Δ)]  +  (1/κ)·ln(1 + γΔ/κ)
#      δ*_a = (1/γΔ)·ln[h(q)/h(q−Δ)]  +  (1/κ)·ln(1 + γΔ/κ)
#
#  Under the Guéant–Lehalle–Fernandez-Tapia ansatz for h(t,q), the
#  inventory-dependent terms simplify to a linear function of q,
#  giving the decomposition:
#
#      δ*_b = half_spread + skew · q
#      δ*_a = half_spread − skew · q
#
#  so that:  q_bid = S − δ*_b ,  q_ask = S + δ*_a
#
#  The coefficients are (see Guéant et al. 2012, Proposition 4.1,
#  and Guéant 2017, Chapter 4):
#
#      c₁ = (1 / ξΔ) · ln(1 + ξΔ/κ)
#                                            [spread width component]
#           This term is independent of σ and q.  It captures the
#           "option value of the limit order" — the MM is essentially
#           writing a free option to the market and c₁ is the minimum
#           premium to break even against adverse selection.
#
#      c₂ = √(γ / (2AΔκ)) · (1 + ξΔ/κ)^((κ/(ξΔ) + 1)/2)
#                                            [inventory skew component]
#           This scales linearly with σ·q in the final quote.  Larger
#           γ (more risk averse) → larger c₂ → more aggressive skew.
#           Larger A (more MO activity) → smaller c₂ → less skew
#           (because fills are more frequent, so inventory mean-reverts
#           faster naturally).
#
#      half_spread = c₁ + (Δ/2) · c₂ · σ
#      skew_factor = c₂ · σ                   [ticks per unit inventory]
#
#  where ξ = ε (terminal inventory penalty) or γ (if ε not specified),
#  and Δ is the lot size parameter (default 1).
#
#  ─── Parameter Sensitivity ───
#
#      γ ↑  →  c₂ ↑  →  wider spreads, stronger inventory skew
#      A ↑  →  c₂ ↓  →  thinner spreads (more competition)
#      κ ↑  →  c₁ ↓  →  tighter quotes (MOs decay fast)
#      σ ↑  →  hs ↑  →  wider spreads (more mid-price risk)
#
# =============================================================================

def _compute_glft_coefficients(
    gamma: float, kappa: float, A: float, sigma: float,
    delta: float = 1.0, epsilon: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Compute the four GLFT optimal quoting coefficients.

    This function evaluates the closed-form solution to the GLFT stochastic
    control problem (Guéant et al. 2012, Proposition 4.1).  The solution
    determines how wide the MM should quote (half_spread) and how much to
    skew quotes in response to inventory (skew_factor).

    Parameters
    ----------
    gamma : float
        Risk aversion parameter γ > 0 (CARA coefficient).
        Controls the trade-off between expected P&L and inventory variance.
        Higher γ → wider spreads, more aggressive inventory mean-reversion.
        Typical range: 0.01–1.0 for tick-level LOB simulators.

    kappa : float
        Exponential decay rate of MO fill probability: λ(δ) = A·exp(−κδ).
        κ has units of (1/ticks).  Larger κ means fill probability drops
        faster with distance → MM must quote closer to mid.
        Typical range: 0.1–5.0 depending on tick size and liquidity.

    A : float
        Base arrival rate of market orders at the best quote (δ→0).
        Units: MOs per second.  Larger A means more fills per unit time
        → MM can afford thinner spreads (more competition at front of queue).
        Typical range: 0.001–1.0.

    sigma : float
        Mid-price volatility in ticks per √second.
        This is the diffusion coefficient of the arithmetic Brownian motion:
            dS_t = σ · dW_t
        Higher σ → wider spreads to compensate for mid-price risk.
        Estimated online from the RMS of gate-level returns rescaled by
        √(1/avg_gate_dt).

    delta : float
        Lot size parameter Δ (default 1.0).  In the GLFT model, each limit
        order has size Δ.  For single-lot strategies, Δ = 1.

    epsilon : float or None
        Terminal inventory penalty ξ.  In the finite-horizon GLFT problem,
        ε penalises residual inventory at time T:
            W_T = cash_T + q_T · S_T − ε · q_T²
        If None, defaults to γ (equivalent to infinite-horizon scaling where
        the terminal penalty matches the running inventory penalty).

    Returns
    -------
    (c1, c2, half_spread, skew_factor) : tuple of float
        c1           — spread width component (ticks).
                        Represents the "option premium" the MM charges
                        for providing liquidity, independent of inventory.
        c2           — raw skew coefficient (before sigma scaling).
                        Dimensionless multiplier that converts σ·q into
                        the inventory-dependent quote offset.
        half_spread  — optimal half-spread (ticks):  c₁ + (Δ/2)·c₂·σ
                        This is the distance from the fair-value reference
                        price to each quote when inventory q = 0.
        skew_factor  — optimal inventory skew per unit inventory (ticks):
                        c₂ · σ.  The bid is pushed further by +skew·q
                        and the ask is pulled closer by −skew·q.

    Notes
    -----
    The final optimal quotes are:
        q_bid = S − (half_spread + skew_factor · q)
        q_ask = S + (half_spread − skew_factor · q)

    For q > 0 (long inventory):
        - Bid is pushed DOWN (further from mid) → less eager to buy more.
        - Ask is pulled DOWN (closer to mid) → more eager to sell.
        This naturally encourages inventory mean-reversion.

    For q < 0 (short inventory): the skew reverses symmetrically.

    See Also
    --------
    Guéant, O. (2017). "Optimal Market Making." Eqn (4.9)–(4.12).
    Cartea, Jaimungal & Penalva (2015). Chapter 10.3.
    """
    # ξ (xi): terminal penalty parameter.
    # When ε is specified, ξ = ε → finite-horizon GLFT solution.
    # When ε is None, ξ = γ → infinite-horizon Avellaneda-Stoikov limit
    # (the terminal penalty matches the running risk aversion).
    xi = float(epsilon) if epsilon is not None else float(gamma)
    D = float(delta)

    # ─── c₁: Width component ───
    #
    #   c₁ = (1 / ξΔ) · ln(1 + ξΔ/κ)
    #
    # This is the "reservation spread" — the minimum half-spread the MM
    # would charge even with zero inventory.  It arises from the first-
    # order optimality condition of the HJB equation:
    #   ∂/∂δ [ A·exp(−κδ) · (δ − 1/(γΔ)·ln(h(q)/h(q±Δ))) ] = 0
    # The ln(1 + ξΔ/κ) term captures the trade-off between quote
    # aggressiveness (small δ → more fills) and per-fill profit
    # (large δ → more profit per fill but fewer fills).
    #
    # Economic interpretation: c₁ is the "option value of providing
    # liquidity".  The MM writes a free American option to incoming MOs;
    # c₁ is the minimum premium to compensate for the adverse selection
    # inherent in being picked off by informed traders.
    term = 1.0 + (xi * D) / float(kappa)
    c1 = (1.0 / (xi * D)) * math.log(term)

    # ─── c₂: Skew component ───
    #
    #   c₂ = √(γ / (2AΔκ)) · (1 + ξΔ/κ)^((κ/(ξΔ) + 1)/2)
    #
    # This coefficient determines how aggressively the MM tilts quotes in
    # response to inventory.  The final skew is: skew = c₂ · σ · q.
    #
    # Decomposition:
    #   - √(γ / (2AΔκ)):  Balance between risk aversion (γ, numerator) and
    #     fill rate (AΔκ, denominator).  High γ or low A → more aggressive
    #     skew (the MM is nervous and wants to shed inventory quickly, or
    #     fills are rare so must incentivise the unwinding side).
    #
    #   - (1 + ξΔ/κ)^(...):  Power term from the HJB solution.  Amplifies
    #     the skew when κ is small relative to ξΔ (shallow MO intensity
    #     decay → fills are possible further from mid → more room to skew).
    #
    # The exponent (κ/(ξΔ) + 1)/2 arises from the discrete inventory
    # difference h(q+Δ)/h(q) in the HJB value function.
    exponent = (float(kappa) / (xi * D)) + 1.0
    power_term = math.pow(term, exponent)
    pre_factor = float(gamma) / (2.0 * float(A) * D * float(kappa))
    c2 = math.sqrt(pre_factor * power_term)

    # ─── Final Optimal Quotes ───
    #
    #   half_spread = c₁ + (Δ/2) · c₂ · σ
    #   skew_factor = c₂ · σ
    #
    # The half_spread has two components:
    #   1. c₁ → inventory-independent reservation spread ("option premium")
    #   2. (Δ/2)·c₂·σ → volatility-dependent risk premium (grows with σ)
    #
    # The factor Δ/2 in front of the volatility term comes from the
    # quadratic inventory penalty term ½γσ²q²Δ² in the HJB.  When Δ=1
    # (single lot), this simplifies to σ·c₂/2.
    #
    # Final quotes (applied in the main policy function):
    #   q_bid = S − (half_spread + skew_factor · q)
    #   q_ask = S + (half_spread − skew_factor · q)
    half_spread = c1 + (D / 2.0) * c2 * float(sigma)
    skew = c2 * float(sigma)
    return c1, c2, half_spread, skew


# =============================================================================
#
#  SECTION 2: N-LEVEL MICROPRICE COMPUTATION
#
#  ─── Background ───
#
#  The midprice (best_bid + best_ask) / 2 is the simplest estimate of the
#  "efficient price" — the consensus fair value around which the market
#  fluctuates.  However, the midprice IGNORES order book imbalance, which
#  is a powerful predictor of short-term price direction.
#
#  The microprice (Stoikov, 2018) is a refined estimator that incorporates
#  volume imbalance across multiple depth levels:
#
#      Stoikov, S. (2018). "The Micro-Price: A High-Frequency Estimator
#      of Future Prices." Quantitative Finance, 18(12), 1959–1966.
#
#  ─── Intuition ───
#
#  Consider a snapshot where bid has 500 lots and ask has 100 lots at the
#  best level.  The ask is "thin" and likely to be consumed first by an
#  incoming buy MO.  When the ask is consumed, the best ask moves up by
#  1 tick, and the midprice rises by ½ tick.  The microprice anticipates
#  this by shifting the fair value toward the ask (upward).
#
#  More precisely, the microprice at level k is:
#
#      μ_k = (V^bid_k · P^ask_k + V^ask_k · P^bid_k) / (V^bid_k + V^ask_k)
#
#  This is a volume-weighted average where each side's volume "votes" for
#  the OPPOSITE side's price.  Heavy bid volume "votes" for the ask price
#  (predicting upward pressure), and vice versa.  The n-level microprice
#  averages across depths with exponential decay weights.
#
#  ─── Impact on GLFT Quoting ───
#
#  When the microprice replaces the midprice as the reference S in:
#      q_bid = S − (half_spread + skew · q)
#      q_ask = S + (half_spread − skew · q)
#
#  A higher S (bid imbalance → upward pressure) TIGHTENS the ask (more
#  aggressive, captures the expected upward move) and WIDENS the bid
#  (more defensive, avoids adverse selection on the buy side).  This
#  reduces the effective adverse selection cost by aligning quotes with
#  the anticipated short-term price direction.
#
#  ─── Multi-Level Extension ───
#
#  While the original Stoikov (2018) paper focuses primarily on the
#  single-level microprice (best bid/ask only), this implementation
#  generalises to n depth levels with exponential decay weighting.
#  Deeper levels carry less predictive power (a large resting order at
#  level 5 is less likely to be reached than at level 1), so the decay
#  parameter down-weights their contribution.
#
# =============================================================================

def _compute_microprice(
    depth_bids: List[Tuple[int, float]],
    depth_asks: List[Tuple[int, float]],
    n_levels: int,
    decay: float = 1.0,
) -> Optional[float]:
    """Compute the n-level volume-weighted microprice (Stoikov, 2018).

    The microprice is a volume-imbalance-adjusted fair-value estimator.
    It shifts the reference price toward the side with MORE resting
    liquidity, anticipating that the THINNER side will be consumed first
    by incoming market orders.

    Formula (generalised to n levels with exponential decay weights):

        μ = Σ_{k=0}^{n-1} w_k · (V^bid_k · P^ask_k + V^ask_k · P^bid_k)
            ─────────────────────────────────────────────────────────────────
                           Σ_{k=0}^{n-1} w_k · (V^bid_k + V^ask_k)

    where w_k = decay^k.  For decay=1.0 (uniform), all levels contribute
    equally.  For decay<1.0, deeper levels are geometrically down-weighted.

    **Cross-weighting intuition**: at each level k, the bid volume V^bid_k
    "votes" for the ask price P^ask_k (predicting upward pressure), and
    the ask volume V^ask_k "votes" for the bid price P^bid_k (predicting
    downward pressure).  The net result is a volume-weighted interpolation
    between the bid and ask prices that tracks order flow imbalance.

    Parameters
    ----------
    depth_bids : list of (price_idx, size)
        Bid side of the LOB.  Each entry is (price_index, volume) where
        price_index is in tick-index coordinates used by the simulator.
        Does not need to be sorted — sorting is handled internally.
    depth_asks : list of (price_idx, size)
        Ask side of the LOB, same format.
    n_levels : int
        Number of book depth levels to include.  n_levels=1 gives the
        standard single-level microprice.  n_levels=3–5 is typical for
        capturing the "depth of book" signal without excessive noise.
    decay : float
        Geometric decay weight per level: w_k = decay^k.
        - 1.0 (default) → uniform weighting across all n levels.
        - 0.5 → each subsequent level has half the weight of the previous.
        - 0.0 → only the best level contributes (reduces to single-level).

    Returns
    -------
    float or None
        Microprice in tick-index space (same coordinates as the LOB prices),
        or None if either side of the book is empty.

    Example
    -------
    If best bid = 100 with size 300, best ask = 101 with size 100:
        μ = (300·101 + 100·100) / (300 + 100) = 40300/400 = 100.75
    Compare midprice = (100 + 101)/2 = 100.50.
    The microprice is 0.25 ticks HIGHER, reflecting the bid-heavy imbalance
    (the ask is thin → likely to be consumed → price likely to rise).
    """
    if not depth_bids or not depth_asks:
        return None

    # Sort bid side descending by price (best bid = highest price first).
    # Sort ask side ascending by price (best ask = lowest price first).
    bids_sorted = sorted(depth_bids, key=lambda x: -x[0])
    asks_sorted = sorted(depth_asks, key=lambda x: x[0])

    # Only use the minimum of requested levels and available data.
    n = min(n_levels, len(bids_sorted), len(asks_sorted))
    if n == 0:
        return None

    num = 0.0   # Numerator:   Σ w_k · (V^bid_k · P^ask_k + V^ask_k · P^bid_k)
    den = 0.0   # Denominator: Σ w_k · (V^bid_k + V^ask_k)
    for k in range(n):
        # Exponential decay weight: w_0 = 1, w_1 = decay, w_2 = decay², ...
        w = math.pow(decay, k)
        p_bid, v_bid = float(bids_sorted[k][0]), float(bids_sorted[k][1])
        p_ask, v_ask = float(asks_sorted[k][0]), float(asks_sorted[k][1])

        total_vol = v_bid + v_ask
        if total_vol <= 0:
            continue   # Skip empty levels (can happen with padding)

        # Cross-weighting: each side's volume "votes" for the opposite price.
        #   V^bid weights P^ask → large bid volume pushes μ toward ask (up)
        #   V^ask weights P^bid → large ask volume pushes μ toward bid (down)
        num += w * (v_bid * p_ask + v_ask * p_bid)
        den += w * total_vol

    if den <= 0:
        return None

    return num / den


# =============================================================================
#
#  SECTION 3: MARKET INTENSITY TRACKER
#
#  ─── Purpose ───
#
#  Online estimator for the GLFT intensity parameters A and κ, which
#  govern the exponential fill probability model:
#
#      λ(δ) = A · exp(−κ · δ)
#
#  where:
#      δ = distance from the mid-price to the limit order (in tick units)
#      λ(δ) = expected MO arrival rate (fills per second) at distance δ
#      A = base arrival rate at δ→0 (intercept; MOs per second)
#      κ = exponential decay rate (slope; 1/tick units)
#
#  This functional form is the standard assumption in the GLFT model
#  (Guéant et al. 2012, 2013) and also appears in Cartea, Jaimungal &
#  Penalva (2015, Chapter 10).  It captures the empirical observation
#  that limit orders placed closer to the mid-price are filled more
#  frequently — approximately exponentially so.
#
#  ─── Why Observe the Public Tape (Not Own Fills) ───
#
#  A naive approach would estimate (A, κ) from the MM's OWN limit order
#  fills.  This creates a catastrophic FEEDBACK LOOP:
#
#      Wide quotes → fewer fills → smaller A estimate → even wider
#      quotes → zero fills → A → 0, κ → ∞ → quotes at infinity.
#
#  This is the "self-fulfilling prophecy" of market-making calibration:
#  the MM's quoting behaviour changes the data it calibrates from.
#
#  The correct approach is to observe ALL market orders on the PUBLIC
#  TAPE (LOBSTER message feed).  Every MO represents a potential fill
#  for ANY limit order at that depth, regardless of whether the MM
#  had an order there.  This gives an unbiased, exogenous estimate of
#  fill probability that is independent of the MM's current quotes.
#
#  ─── Method 1: WLS Tail-Counting (Fast, Incremental) ───
#
#  For each MO observed, we record the deepest depth index k it reached:
#      k = round(depth / half_tick) − 1
#
#  We maintain cumulative TAIL counts:
#      tail[k] = # MOs that reached depth ≥ (k+1)·half_tick
#
#  The tail count at depth k is the number of MOs that would have filled
#  a hypothetical LO sitting at distance (k+1)·half_tick.  Converting to
#  per-second intensity:
#      λ̂[k] = tail[k] / elapsed_time
#
#  Then fit the log-linear model via Weighted Least Squares:
#      log(λ̂[k]) = log(A) − κ · δ_k     where δ_k = (k+1)·half_tick
#
#  The intercept gives log(A), the slope gives −κ.
#  Weights = tail[k] (bins with more observations are more reliable).
#
#  ─── Method 2: Censored Waiting Times MLE (Rigorous) ───
#
#  See the detailed description in the module docstring (Section 6) and
#  in the method docstring of ``fit_A_kappa_censored()``.  This method
#  handles right-censored observations correctly via survival analysis
#  MLE, with mid-drift correction for intra-window price movement.
#
#  ─── Data Flow ───
#
#  The tracker receives data via two entry points:
#
#  1. ``observe(is_mo, depth, direction, timestamp, mid)``
#     Called at EVERY LOBSTER event (MO or not).  Records:
#       - WLS: cumulative tail counts (tail_buy, tail_sell)
#       - Censored: raw (timestamp, tick_k, mid_pre) per MO event
#       - Common: total event count and elapsed time
#
#  2. ``mark_window(t_now, mid_now)``
#     Called at each throttle-gate firing.  Records the window boundary
#     [t_start, t_now] with the mid-price at window start (mid_0) for
#     the censored MLE's mid-drift correction.
#
#  Data is queried via:
#  - ``fit_A_kappa(side)``         → WLS regression
#  - ``fit_A_kappa_censored(side)`` → Censored MLE + WLS on λ̂(δ)
#
#  ─── Memory Management ───
#
#  WLS accumulators (tail_buy, tail_sell) are compact fixed-size arrays.
#  Old observations are forgotten via explicit exponential decay
#  (``decay(factor)``).
#
#  Censored WT raw events are stored in bounded deques (FIFO eviction).
#  When the deque is full, the oldest events are automatically discarded.
#  This provides natural "forgetting" without explicit decay, and keeps
#  memory bounded regardless of backtest length.
#
# =============================================================================

class _MarketIntensityTracker:
    """Online estimator for the GLFT intensity parameters A and κ.

    This class is the statistical engine at the heart of the online
    recalibration system.  It observes the LOBSTER event stream in real
    time and estimates the parameters of the exponential fill probability
    model λ(δ) = A·exp(−κδ).

    Two estimation methods are maintained in parallel and can be queried
    independently:

    ═══════════════════════════════════════════════════════════════════
    METHOD 1: WLS Tail-Counting — ``fit_A_kappa()``
    ═══════════════════════════════════════════════════════════════════

    **Speed**: O(max_k) at fit time (fixed-size array operations).
    **Memory**: O(max_k) — two arrays of tail counts.
    **Approach**: Non-parametric → parametric two-step:

      Step 1 — Non-parametric tail counts:
        For each MO with deepest tick index k:
            tail[0], tail[1], ..., tail[k] all get +1
        This builds the cumulative tail: tail[k] = #{MOs reaching ≥ depth k}.

      Step 2 — Parametric WLS fit:
        intensity[k] = tail[k] / elapsed_time
        Fit log(intensity[k]) = log(A) − κ · δ_k  via WLS, weights = tail[k].

    **Advantages**: Very fast; O(1) per observe() call; no raw event storage.
    **Disadvantages**:
      - Ghost starvation (FIX 1): After repeated decay() calls, deep
        buckets have tail[k] ≈ 0.3 (fractional residuals).  These "ghost"
        points pull the regression toward κ → ∞.  Mitigated by filtering
        out tail[k] < 1.0.
      - No mid-drift correction: the tail count doesn't distinguish between
        windows where mid drifted toward vs. away from the hypothetical LO.
      - Right-censoring ignored: a window with NO MOs at depth k contributes
        nothing to the tail count, but should contribute to the denominator
        (the LO was "at risk" for the full window duration).

    ═══════════════════════════════════════════════════════════════════
    METHOD 2: Censored Waiting Times MLE — ``fit_A_kappa_censored()``
    ═══════════════════════════════════════════════════════════════════

    **Speed**: O(max_k × N_events × N_windows) at fit time.
    **Memory**: O(N_events + N_windows) — bounded deques.
    **Approach**: Direct MLE from survival analysis on per-window data.

    This method is **mathematically equivalent** to the batch calibrator
    in ``censored_waiting_times_calib.py`` (specifically, the backtest-mode
    ``_backtest_window_first_virtual_fill`` function), but operates fully
    online from accumulated events rather than external DataFrames.

    Detailed algorithm for each δ_k and each window [t₀, t₁, mid₀):

      (a) **Hypothetical LO placement**: A virtual limit order is "placed"
          at a FIXED price at the start of the window:
              Buy LO (bid):  P_lo = mid₀ − δ_k
              Sell LO (ask): P_lo = mid₀ + δ_k
          This price does NOT move during the window — it's a resting LO.

      (b) **Mid-drift correction**: During the window, the mid-price drifts.
          At each MO event i, the pre-event mid is mid_i ≠ mid₀.  The MO's
          depth is measured from mid_i, not mid₀.  So the "effective delta"
          from the MO's reference point to the LO's fixed price is:

              Buy LO:  δ_eff = δ_k + (mid_i − mid₀)
                       If mid drifted UP: δ_eff > δ_k (LO is further away
                       from the MO's reference → harder to fill).
                       If mid drifted DOWN past the LO: δ_eff ≤ 0 → trivial
                       fill (the LO is now at or through the mid).

              Sell LO: δ_eff = δ_k + (mid₀ − mid_i)
                       Symmetric logic for downward drift.

      (c) **Counterfactual fill check**: The MO at event i swept depth
          D_i ticks.  The actual depth in price units = (D_i + 1)·half_tick.
          A fill occurs if:
              (D_i + 1) · half_tick ≥ δ_eff
          i.e., the MO's sweep reached or exceeded the effective distance
          to the hypothetical LO.

      (d) **First fill wins**: Only the FIRST qualifying MO in the window
          matters (a real LO would be consumed by the first fill and removed
          from the book).  After the first fill, the remaining MOs in the
          window are ignored for this δ_k.

      (e) **Censored waiting time τ̃**:
            Fill at t_fill:  τ̃ = min(t_fill − t₀, ΔT),  I = 1 if τ < ΔT
            No fill:         τ̃ = ΔT (= t₁ − t₀),         I = 0
          where I is the uncensored indicator.

          Special case: τ ≤ 0 (fill at the placement step itself) → treated
          as censored (τ̃ = ΔT, I = 0) because we cannot meaningfully
          distinguish "instant fill" from a pre-existing queue fill.

      (f) **MLE**: λ̂(δ_k) = Σ_w I_w / Σ_w τ̃_w
          This is the standard exponential censoring MLE (Cox & Oakes, 1984).
          The denominator Σ τ̃ is the total "exposure time" at depth δ_k.

      (g) **WLS fit on log(λ̂) vs δ**:
          log(λ̂[k]) = log(A) − κ · δ_k, weighted by Σ τ̃[k].
          The weights are proportional to the Fisher information of each
          MLE estimate — bins with more total exposure time have more
          precise λ̂ values and should dominate the regression.

    **Advantages**: Handles right-censoring correctly; mid-drift correction;
    unbiased even in illiquid markets with sparse fills.
    **Disadvantages**: O(N_events × N_windows × max_k) — slower than WLS.
    Falls back to WLS if insufficient windows or degenerate data.

    ═══════════════════════════════════════════════════════════════════
    Design Choices & Equivalence with Batch Method
    ═══════════════════════════════════════════════════════════════════

    - Raw MO events store ``(timestamp, tick_index, mid_pre)`` — equivalent
      to the ``D_by_event`` + ``mid_pre_arr`` arrays in the batch method.
      The tick_index is computed identically:
          k = round(depth / half_tick) − 1

    - Window boundaries store ``(t_start, t_end, mid_0)`` — equivalent to
      the ``bt_intervals_used`` + ``mid_pre_arr[i0]`` in the batch method.

    - Bounded deques (FIFO eviction) provide natural forgetting for the
      censored method, keeping memory bounded.  WLS uses explicit
      exponential decay via ``decay(factor)``.

    - For LOBSTER data with ``split_sweeps=False`` (the default), the
      ``raw_price`` reported per MO is the deepest fill price, so
      ``abs(ev_px − ev_mid)`` captures the full sweep depth.  This makes
      the online depth measurement equivalent to the batch method's
      ``compute_nonqueue_deepest_tick_per_mo()`` for single-level MOs.
    """

    def __init__(self, max_depth_levels: int = 20, half_tick: float = 0.5,
                 max_events_stored: int = 50_000):
        """Initialise the tracker with empty accumulators.

        Parameters
        ----------
        max_depth_levels : int
            Number of discrete depth buckets in the δ grid.
            The grid spans δ_k = (k+1) · half_tick for k = 0, ..., max_k−1.
            With half_tick=0.5 and max_depth_levels=20, the grid covers
            depths from 0.5 to 10.0 ticks from mid.  This should be wide
            enough to capture the full decay of λ(δ) for typical markets.

        half_tick : float
            Half the tick size, used as the fundamental discretisation unit.
            For a tick size of 1.0 (e.g., 1 cent), half_tick = 0.5.
            The depth grid spacing is half_tick, matching the resolution
            of the LOBSTER order book data.

        max_events_stored : int
            Maximum number of raw MO events to store per side (buy/sell).
            When the deque is full, the oldest events are discarded (FIFO).
            50,000 events per side covers ~1–2 full trading days of a
            moderately liquid stock.  Larger values give more data for the
            censored MLE but use more memory.
        """
        self._max_k = max_depth_levels
        self._half_tick = half_tick

        # ═══════════════════════════════════════════════════════════════
        # WLS accumulators (Method 1)
        # ═══════════════════════════════════════════════════════════════
        #
        # _n_events: total number of events observed (MO + non-MO).
        #   Used as fallback denominator when elapsed_time is unavailable.
        #
        # _tail_buy[k]: cumulative count of BUY MOs that reached depth ≥ k.
        #   These are MOs that sweep the ASK side of the book.
        #   They would fill our SELL limit orders (ask-side LOs).
        #
        # _tail_sell[k]: cumulative count of SELL MOs that reached depth ≥ k.
        #   These are MOs that sweep the BID side of the book.
        #   They would fill our BUY limit orders (bid-side LOs).
        #
        # Convention:
        #   - To estimate fill probability for our BID (buy LO):
        #     use _tail_sell (sell MOs fill our bids).
        #   - To estimate fill probability for our ASK (sell LO):
        #     use _tail_buy (buy MOs fill our asks).
        #   This is inverted from the MO direction because the MO and
        #   the LO are on OPPOSITE sides of the trade.
        #
        # _elapsed_time: total seconds of observation (sum of inter-event Δt).
        #   Used to convert tail counts to per-second intensity:
        #     λ̂[k] = tail[k] / elapsed_time
        self._n_events = 0.0
        self._tail_buy = np.zeros(max_depth_levels, dtype=np.float64)
        self._tail_sell = np.zeros(max_depth_levels, dtype=np.float64)
        self._elapsed_time = 0.0
        self._last_time: Optional[float] = None

        # ═══════════════════════════════════════════════════════════════
        # Censored WT raw event storage (Method 2)
        # ═══════════════════════════════════════════════════════════════
        #
        # Raw MO events, stored separately by MO direction:
        #   Each entry: (timestamp, tick_index, mid_pre_event)
        #
        # _mo_events_buy: BUY market orders (aggressor buys, sweeps ask).
        #   In the censored MLE, these events can fill our ASK LOs.
        #   So when fitting side="ask", we draw from _mo_events_buy.
        #
        # _mo_events_sell: SELL market orders (aggressor sells, sweeps bid).
        #   These events can fill our BID LOs.
        #   When fitting side="bid", we draw from _mo_events_sell.
        #
        # Why bounded deques?
        #   - Memory bounded: O(max_events_stored) regardless of run length.
        #   - Natural forgetting: oldest events are automatically discarded.
        #   - No explicit decay needed (unlike WLS accumulators).
        self._mo_events_buy: deque = deque(maxlen=max_events_stored)
        self._mo_events_sell: deque = deque(maxlen=max_events_stored)

        # Window boundaries for censored MLE:
        #   Each entry: (t_start, t_end, mid_at_start)
        # A "window" corresponds to one throttle-gate interval.  The
        # censored MLE treats each window as an independent "experiment":
        #   "Did a hypothetical LO at distance δ get filled during [t0, t1)?"
        self._windows: deque = deque(maxlen=5_000)
        self._current_window_start: Optional[float] = None
        self._current_window_mid: Optional[float] = None

    def observe(self, is_mo: bool, depth: Optional[float],
                direction: int, timestamp: Optional[float] = None,
                mid: Optional[float] = None) -> None:
        """Record one market event from the LOBSTER tape.

        This method MUST be called for EVERY event — not just market orders.
        Non-MO events (limit order submissions, cancellations, etc.) are
        needed to maintain accurate elapsed_time and event counts, which are
        the DENOMINATORS for the intensity estimation.

        In the GLFT model:
            λ(δ) = A · exp(−κ · δ)  [MOs per second at distance δ]

        The "per second" normalisation requires knowing the total observation
        time, which is accumulated from the inter-event timestamps of ALL
        events.  If only MO events were passed, elapsed_time would be biased
        (summing only inter-MO intervals, missing the non-MO time).

        Parameters
        ----------
        is_mo : bool
            True if this event is a market-order execution (LOBSTER type 1).
            False for all other event types (limit submissions, cancellations,
            partial fills, etc.).

        depth : float or None
            Distance from the pre-event mid-price to the execution price,
            in the same tick units used by the LOB simulator.
            Computed as ``abs(execution_price − mid_price)`` in the caller.
            For the WLS method, this determines the depth bucket k.
            For the censored method, this is stored as the tick index D.
            Equivalent to ``(D + 1) * half_tick`` in the batch calibrator.
            None if the event is not a MO or the depth is unknown.

        direction : int
            Direction of the market order:
                +1 = BUY MO (the aggressor is buying, sweeping the ask side)
                -1 = SELL MO (the aggressor is selling, sweeping the bid side)
                 0 = unknown / not a MO

        timestamp : float or None
            Event timestamp in seconds since market open (LOBSTER convention:
            34200s = 9:30 AM for US equities).  Used for elapsed time
            computation and for timestamping raw events in the censored method.

        mid : float or None
            Pre-event mid-price in tick-index space.  This is the mid-price
            BEFORE the event occurred — critical for the censored MLE's
            mid-drift correction.  Equivalent to:
                mid_pre[i] = MidPrice[i] − Return[i]
            in the batch method's terminology.

            Why "pre-event"?  The MO changes the order book (consuming
            liquidity), which shifts the mid-price.  The depth of the MO
            is measured relative to the pre-event mid, not the post-event
            mid, because the hypothetical LO would have been placed BEFORE
            the MO arrived.
        """
        # Count ALL events (MO + non-MO) for the WLS intensity denominator.
        self._n_events += 1

        # ── Elapsed time accumulation ──
        # Track the total wall-clock observation time.  This is the
        # denominator for converting tail counts to per-second intensity:
        #   λ̂[k] = tail[k] / elapsed_time
        # We accumulate Δt between consecutive events.  Negative Δt
        # (e.g., from day boundaries) is ignored — the caller handles
        # day-boundary resets separately.
        if timestamp is not None:
            if self._last_time is not None:
                dt = timestamp - self._last_time
                if dt > 0:
                    self._elapsed_time += dt
            self._last_time = timestamp

        # Non-MO events contribute to elapsed_time but not to tail counts.
        if not is_mo or depth is None or depth < 0:
            return

        # ── Discrete depth bucket (tick index) ──
        # Convert continuous depth (in tick units) to an integer bucket:
        #   k = round(depth / half_tick) − 1
        #
        # This is equivalent to ``D_by_event[i]`` in the batch calibrator.
        # The "−1" accounts for the convention that k=0 corresponds to the
        # shallowest possible MO (touching only the best quote), which has
        # depth ≈ half_tick.
        #
        # Example with half_tick = 0.5:
        #   depth=0.5 → k=0 (MO touched best quote only)
        #   depth=1.0 → k=1 (MO swept 2 levels)
        #   depth=2.5 → k=4 (MO swept 5 levels deep)
        k = max(0, int(round(depth / self._half_tick)) - 1)
        k = min(k, self._max_k - 1)

        # ── WLS: Update cumulative tail distribution ──
        # A MO that reached depth k would have filled hypothetical LOs at
        # ALL depths ≤ k.  So tail[0], tail[1], ..., tail[k] all increment.
        # This is the cumulative distribution function:
        #   tail[k] = #{MOs with depth ≥ (k+1)·half_tick}
        if direction > 0:
            # BUY MO → sweeps ASK side → fills our SELL LOs → tail_buy
            self._tail_buy[:k + 1] += 1
        elif direction < 0:
            # SELL MO → sweeps BID side → fills our BUY LOs → tail_sell
            self._tail_sell[:k + 1] += 1

        # ── Censored WT: Store raw event for later MLE computation ──
        # Each event is stored as (timestamp, tick_index, mid_pre).
        # The mid_pre is needed for mid-drift correction in fit_A_kappa_censored().
        if timestamp is not None:
            mid_val = mid if (mid is not None and math.isfinite(mid)) else None
            if direction > 0:
                self._mo_events_buy.append((timestamp, k, mid_val))
            elif direction < 0:
                self._mo_events_sell.append((timestamp, k, mid_val))

    def mark_window(self, t_now: float, mid_now: Optional[float] = None) -> None:
        """Record a throttle-gate boundary (called at each gate firing).

        The censored waiting times method requires a sequence of non-overlapping
        time windows, each representing one "experiment":

            "If I had placed a hypothetical LO at distance δ from mid at the
             start of this window, would any MO during the window have filled it?"

        Each call to ``mark_window()`` simultaneously:
          1. CLOSES the previous window [t_start_prev, t_now)
          2. OPENS a new window starting at t_now

        The closed window is appended to ``self._windows`` as a tuple:
            (t_start, t_end, mid_at_start)

        The ``mid_at_start`` (= mid_0) is the mid-price when the window opened.
        This is the reference point for the hypothetical LO placement and for
        the mid-drift correction formula:
            Buy LO:  δ_eff = δ + (mid_event − mid_0)
            Sell LO: δ_eff = δ + (mid_0 − mid_event)

        Equivalent to ``mid_pre_arr[i0]`` in the batch calibrator's
        ``_backtest_window_first_virtual_fill()`` function.

        Parameters
        ----------
        t_now : float
            Current timestamp in seconds.  Becomes t_end of the closing
            window and t_start of the new window.
        mid_now : float or None
            Mid-price at window start (= the current mid when the gate fires).
            Stored as mid_0 for the NEW window (not the closing one — the
            closing window already has its mid_0 from when it was opened).
            If None (e.g., during a liquidity gap), the censored MLE will
            skip mid-drift correction for this window.
        """
        # Close the previous window if one exists and has positive duration.
        if self._current_window_start is not None and t_now > self._current_window_start:
            self._windows.append((
                self._current_window_start,     # t_start of this window
                t_now,                           # t_end   of this window
                self._current_window_mid,        # mid_0   at window start
            ))
        # Open a new window starting now.
        self._current_window_start = t_now
        self._current_window_mid = mid_now

    # -----------------------------------------------------------------
    # METHOD 1: WLS Tail-Counting Fit (Fast, Incremental)
    #
    # This is the simpler and faster of the two estimation methods.
    # It converts the accumulated tail counts into per-second intensities
    # and fits the log-linear model via Weighted Least Squares.
    #
    # Strengths: O(max_k) computation, no raw event storage needed.
    # Weaknesses: no mid-drift correction, no proper censoring handling,
    #             susceptible to ghost starvation (FIX 1).
    # -----------------------------------------------------------------
    def fit_A_kappa(
        self, side: Optional[str] = None, min_mos: int = 10,
    ) -> Optional[Tuple[float, float]]:
        """Fit A and κ via WLS regression on log-intensity vs depth.

        Algorithm:
          1. Select the appropriate tail array (buy, sell, or combined).
          2. Convert to per-second intensity: λ̂[k] = tail[k] / elapsed_time.
          3. Filter out ghost starvation points (tail[k] < 1.0 after decay).
          4. Fit: log(λ̂[k]) = log(A) − κ · δ_k via WLS, weights = tail[k].
          5. Extract: A = exp(intercept), κ = −slope.

        The WLS formulation minimises:
            Σ_k w_k · (log(λ̂_k) − (log(A) − κ·δ_k))²
        where w_k = tail[k].  Bins with more observations (more MOs
        reached that depth) carry more weight in the regression.

        Parameters
        ----------
        side : ``"bid"`` | ``"ask"`` | ``None``
            Which side's fill probability to estimate:
            - ``"bid"``: probability of our BID LO being filled → uses
              SELL MOs from ``_tail_sell`` (sell MOs hit the bid side).
            - ``"ask"``: probability of our ASK LO being filled → uses
              BUY MOs from ``_tail_buy`` (buy MOs hit the ask side).
            - ``None``: combined (both sides pooled for symmetric estimation).

        min_mos : int
            Minimum cumulative MO count at the shallowest depth (tail[0]).
            If fewer MOs have been observed, there is insufficient data
            for a reliable fit and None is returned.

        Returns
        -------
        (A, kappa) : tuple of float, or None
            A > 0: estimated base arrival rate (MOs per second at δ→0).
            κ > 0: estimated exponential decay rate (1/tick).
            Returns None if insufficient data or the regression yields
            non-positive parameters (which would violate the model's
            exponential form).
        """
        # Need a minimum number of observations for any meaningful fit.
        if self._n_events < 10:
            return None

        # ── Select tail array by side ──
        # Convention reminder:
        #   "bid" → our BID LO → filled by SELL MOs → _tail_sell
        #   "ask" → our ASK LO → filled by BUY MOs → _tail_buy
        if side == "bid":
            tail = self._tail_sell.copy()
        elif side == "ask":
            tail = self._tail_buy.copy()
        else:
            tail = self._tail_buy + self._tail_sell

        # Minimum data check: tail[0] is the count of ALL MOs at any depth.
        # If this is below the threshold, we lack sufficient statistics.
        if tail[0] < min_mos:
            return None

        # ── Convert tail counts to per-second intensity ──
        # If elapsed_time is available (accurate), use it as the denominator.
        # Otherwise, fall back to total event count (less accurate but avoids
        # division by zero when timestamps are missing).
        denom = self._elapsed_time if self._elapsed_time > 0 else float(self._n_events)
        intensity = tail / denom

        # ── Build data points for regression ──
        deltas = []
        log_lam = []
        weights = []
        for k in range(self._max_k):
            # Skip buckets with zero intensity (can't take log of 0).
            if intensity[k] <= 0:
                continue

            # ── FIX 1: Ghost Starvation Filter ──
            #
            # After repeated decay() calls, deep tail buckets accumulate
            # fractional residuals (e.g., tail[15] = 0.3).  These "ghost"
            # observations yield artificially low intensities:
            #   intensity[15] = 0.3 / 5000 = 6e-5
            # The log of this is very negative, pulling the regression
            # slope toward −∞ and inflating κ to absurd values (κ → 50+).
            #
            # This causes the GLFT model to produce infinite spreads:
            #   half_spread = c₁ + (Δ/2)·c₂·σ  where c₁ → ∞ as κ → ∞
            #
            # The fix is simple: ignore any bucket where tail[k] < 1.0.
            # A fractional count after decay is not a real observation —
            # it's a numerical artifact from the exponential forgetting.
            if tail[k] < 1.0:
                continue

            d = (k + 1) * self._half_tick
            deltas.append(d)
            log_lam.append(math.log(intensity[k]))
            weights.append(float(tail[k]))

        # ── Fallback: too few data points for regression ──
        # WLS needs at least 2 points to determine slope + intercept.
        # If only 1 valid point exists, we use a heuristic fallback:
        #   Assume κ = 1.0, then solve for A from the single observation:
        #     log(λ̂) = log(A) − κ·δ  →  A = λ̂ · exp(κ·δ)
        # FIX: was `exp(log_lam[0])` which is just λ̂[k0] — this ignores
        # the fact that deltas[0] ≥ 0.5 tick, severely underestimating A.
        if len(deltas) < 3:
            if len(deltas) >= 1:
                kappa_fallback = 1.0
                A_fallback = math.exp(log_lam[0] + kappa_fallback * deltas[0])
                if A_fallback > 0:
                    return A_fallback, kappa_fallback
            return None

        # ── Weighted Least Squares regression ──
        # Model: y = α + β·x  where y = log(λ̂), x = δ, α = log(A), β = −κ.
        #
        # The WLS normal equations with weights w:
        #   β = Σ w(x−x̄)(y−ȳ) / Σ w(x−x̄)²
        #   α = ȳ − β·x̄
        # where x̄ = Σ(wx)/Σw, ȳ = Σ(wy)/Σw.
        x = np.array(deltas)
        y = np.array(log_lam)
        w = np.array(weights)

        sw = w.sum()
        if sw <= 0:
            return None
        xbar = (w * x).sum() / sw
        ybar = (w * y).sum() / sw
        sxx = (w * (x - xbar) ** 2).sum()
        if sxx < 1e-30:
            return None    # All δ values identical → degenerate regression
        sxy = (w * (x - xbar) * (y - ybar)).sum()

        slope = sxy / sxx              # β = −κ (should be negative)
        intercept = ybar - slope * xbar  # α = log(A)

        kappa_hat = -slope             # κ = −β > 0 if slope < 0
        A_hat = math.exp(intercept)    # A = exp(α) > 0

        # Reject non-positive parameters (model violation).
        if kappa_hat <= 0 or A_hat <= 0:
            return None
        return A_hat, kappa_hat

    # -----------------------------------------------------------------
    # METHOD 2: Censored Waiting Times MLE (Statistically Rigorous)
    #
    # This is the more sophisticated estimation method.  It implements
    # the full censored-data MLE from survival analysis, with mid-drift
    # correction for intra-window price movement.
    #
    # Algorithmically EQUIVALENT to:
    #   - _backtest_window_first_virtual_fill()  (counterfactual fill check)
    #   - The delta-loop in fit_execution_intensity_censored_waiting_times()
    # from censored_waiting_times_calib.py, but computed from online
    # accumulated events rather than external DataFrames.
    #
    # Computational complexity:
    #   O(max_k × N_events_in_windows × N_windows)
    # For typical parameters (max_k=20, ~100 events/window, ~100 windows):
    #   ~200,000 iterations — fast enough for online recalibration.
    # -----------------------------------------------------------------
    def fit_A_kappa_censored(
        self, side: Optional[str] = None, min_windows: int = 10,
        max_windows: Optional[int] = None,
    ) -> Optional[Tuple[float, float]]:
        """Fit A and κ via censored waiting times MLE with mid-drift correction.

        This implements the exponential censoring model from survival analysis
        (Cox & Oakes, 1984; Lawless, 2003).  The key insight is that each
        throttle window provides either:
          - An **uncensored** observation: the LO was filled, and we know
            the exact waiting time τ = t_fill − t₀.
          - A **right-censored** observation: the LO was NOT filled by the
            end of the window, so we only know τ > ΔT.

        Ignoring censored observations (as the WLS method does) biases λ̂
        upward — we'd only see windows where fills happened, missing all
        the windows where the LO survived the full duration.  The censored
        MLE correctly accounts for this by including the censored exposure
        time in the denominator.

        ─── Full Algorithm ───

        For each depth bucket δ_k = (k+1) · half_tick, k = 0, ..., max_k−1:

          1. Initialise accumulators: fill_count[k] = 0, tau_denom[k] = 0.

          2. For each completed window [t₀, t₁, mid₀]:

             a) Binary-search for MO events within [t₀, t₁).
                (Events are pre-sorted by timestamp.)

             b) Scan events in chronological order.  For each MO at time
                t_i with tick depth D_i and pre-event mid mid_i:

                • **Mid-drift correction** (if mid₀ is available and
                  side ≠ None):

                  The hypothetical LO is at a FIXED price:
                      P_lo = mid₀ − δ_k  (buy LO)
                      P_lo = mid₀ + δ_k  (sell LO)

                  But the MO at event i sees a different mid (mid_i).
                  The "effective delta" from the MO's perspective:

                      Buy LO:  δ_eff = δ_k + (mid_i − mid₀)
                        If mid drifted UP by Δm:  δ_eff = δ_k + Δm
                        → LO is further from the MO's reference → harder
                        to fill.  Intuition: the market moved away from
                        our bid, so the MO would need to sweep deeper to
                        reach us.

                      Sell LO: δ_eff = δ_k + (mid₀ − mid_i)
                        Symmetric: if mid drifted DOWN, our ask is now
                        further from the MO's reference.

                      Special case: δ_eff ≤ 0 means the mid has drifted
                      PAST the LO's price.  For a buy LO, this means the
                      mid fell below P_lo — the LO is now above the mid
                      and any sell MO fills it trivially.

                • **Fill check**:
                  The MO swept (D_i + 1) · half_tick in depth.
                  Fill occurs if:  (D_i + 1) · half_tick ≥ δ_eff

                • **First fill wins**: break on the first qualifying MO.

             c) Compute censored waiting time τ̃ and indicator I:

                If fill found at t_fill:
                    τ = t_fill − t₀
                    If τ ≤ 0 or NaN:  τ̃ = ΔT, I = 0  (degenerate → censor)
                    Else:  τ̃ = min(τ, ΔT),  I = 1 if τ < ΔT else 0

                If no fill:
                    τ̃ = ΔT = t₁ − t₀,  I = 0  (right-censored)

             d) Update accumulators:
                    fill_count[k] += I
                    tau_denom[k]  += τ̃

          3. MLE per bucket:
                λ̂(δ_k) = fill_count[k] / tau_denom[k]

             This is the standard exponential censoring MLE.
             Proof: the log-likelihood for N windows is
                ℓ(λ) = Σ [I_w · ln(λ) − λ · τ̃_w]
             Setting dℓ/dλ = 0:
                Σ I_w / λ̂ − Σ τ̃_w = 0
                → λ̂ = Σ I_w / Σ τ̃_w     ∎

          4. WLS fit on log(λ̂) vs δ:
                log(λ̂[k]) = log(A) − κ · δ_k
             Weights = tau_denom[k] (∝ Fisher information of MLE estimate).
             Extract: A = exp(intercept), κ = −slope.

        Parameters
        ----------
        side : ``"bid"`` | ``"ask"`` | ``None``
            Which side to estimate:
            - ``"bid"``: our BID (buy) LO → filled by SELL MOs →
              events from ``_mo_events_sell``, target_side = +1.
            - ``"ask"``: our ASK (sell) LO → filled by BUY MOs →
              events from ``_mo_events_buy``, target_side = −1.
            - ``None``: combined (symmetric, no mid-drift correction).

        min_windows : int
            Minimum number of completed windows before attempting a fit.
            Too few windows → unstable MLE.  Default 10 is conservative;
            5 may suffice for moderately liquid markets.

        Returns
        -------
        (A, kappa) : tuple of float, or None
            A > 0: base arrival rate (MOs/sec at δ→0).
            κ > 0: exponential decay rate (1/tick).
            Returns None if insufficient data, degenerate regression,
            or non-positive parameter estimates.

            Also stores the last fit result in ``self._last_censored_fit``
            as a dict with keys:
                delta_grid, lambda_hat, fill_count, tau_denom,
                A, kappa, n_windows, side
            This is used by ``_plot_censored_fit()`` for diagnostic
            visualisation of the fitted intensity curve.
        """
        # ── Minimum data check ──
        if len(self._windows) < min_windows:
            return None

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Select MO events based on the target side.
        #
        # The convention is INVERTED from the MO direction:
        #   side="bid" → we want fills on our BID (buy) LO
        #              → SELL MOs fill our bids → use _mo_events_sell
        #              → target_side = +1 (our LO is a buy)
        #
        #   side="ask" → we want fills on our ASK (sell) LO
        #              → BUY MOs fill our asks → use _mo_events_buy
        #              → target_side = −1 (our LO is a sell)
        #
        #   side=None  → combined (symmetric) → both sides pooled
        #              → target_side = 0 → no mid-drift correction
        # ═══════════════════════════════════════════════════════════════
        if side == "bid":
            events = list(self._mo_events_sell)
            target_side = +1
        elif side == "ask":
            events = list(self._mo_events_buy)
            target_side = -1
        else:
            events = sorted(
                list(self._mo_events_buy) + list(self._mo_events_sell),
                key=lambda e: e[0],
            )
            target_side = 0

        if not events:
            return None

        # ═══════════════════════════════════════════════════════════════
        # Step 2: Pre-sort events and extract arrays for vectorised access.
        #
        # Events are tuples (timestamp, tick_index, mid_pre).
        # We convert to three aligned numpy arrays for efficient slicing
        # within binary-searched window ranges.
        # ═══════════════════════════════════════════════════════════════
        events.sort(key=lambda e: e[0])
        ev_times = np.array([e[0] for e in events])
        ev_ticks = np.array([e[1] for e in events])     # tick index D
        ev_mids = np.array([
            e[2] if e[2] is not None else float("nan") for e in events
        ])

        half_tick = self._half_tick
        windows = list(self._windows)

        # ── Limit to the most recent max_windows entries ──
        # This mirrors what recalib_window does for σ: only use a rolling
        # window of recent data so the fit adapts to regime changes instead
        # of averaging over the entire run since the last reset.
        if max_windows is not None and len(windows) > max_windows:
            windows = windows[-max_windows:]

        # ═══════════════════════════════════════════════════════════════
        # FIX: Detect the earliest event timestamp still in memory.
        #
        # The MO event deques are bounded (maxlen=50_000).  If the market
        # is very active, old events get evicted while the window deque
        # (maxlen=5_000) may still reference them.  A window whose entire
        # span [t₀, t₁) predates the oldest surviving event would yield
        # an empty searchsorted slice — falsely treated as "no fills"
        # (right-censored), inflating τ_denom and deflating λ̂.
        #
        # Solution: skip any window whose t₁ ≤ oldest surviving event
        # timestamp.  Such windows have no usable event data and must
        # not contribute to the MLE accumulators.
        # ═══════════════════════════════════════════════════════════════
        ev_t_min = float(ev_times[0]) if len(ev_times) > 0 else float("inf")

        # ═══════════════════════════════════════════════════════════════
        # Step 3: Per-δ accumulators.
        #
        # lambda_hat[k]: will hold the MLE λ̂(δ_k) after the loop.
        # tau_denom[k]:  total exposure time Σ τ̃ at depth δ_k.
        # fill_count[k]: total uncensored fills Σ I at depth δ_k.
        #
        # After the loop: λ̂[k] = fill_count[k] / tau_denom[k].
        # ═══════════════════════════════════════════════════════════════
        lambda_hat = np.zeros(self._max_k, dtype=np.float64)
        tau_denom = np.zeros(self._max_k, dtype=np.float64)
        fill_count = np.zeros(self._max_k, dtype=np.float64)

        # ═══════════════════════════════════════════════════════════════
        # Step 4: Main loop — iterate over windows × depth buckets.
        #
        # For each window [t₀, t₁, mid₀] and each depth δ_k:
        #   - Find MO events within [t₀, t₁) via binary search
        #   - Scan chronologically for the first qualifying fill
        #   - Compute τ̃ and I
        #   - Accumulate into fill_count[k] and tau_denom[k]
        # ═══════════════════════════════════════════════════════════════
        for (t0, t1, mid_0) in windows:
            dt_window = t1 - t0
            if dt_window <= 0:
                continue    # Degenerate window (zero duration)

            # FIX: Skip windows whose events have been evicted from memory.
            # If t₁ ≤ oldest surviving event, the searchsorted would return
            # an empty slice — creating false right-censoring (phantom
            # "no-fill" observations that bias λ̂ downward).
            if t1 <= ev_t_min:
                continue

            # Binary search for MO events within [t₀, t₁).
            # np.searchsorted with side="left" gives the first index ≥ t0/t1.
            i_start = int(np.searchsorted(ev_times, t0, side="left"))
            i_end = int(np.searchsorted(ev_times, t1, side="left"))

            # Slice the pre-sorted arrays to get only this window's events.
            win_times = ev_times[i_start:i_end]
            win_ticks = ev_ticks[i_start:i_end]
            win_mids = ev_mids[i_start:i_end]
            n_win = len(win_times)

            # Check if mid₀ is available for drift correction.
            has_mid0 = mid_0 is not None and math.isfinite(mid_0)

            # ── Inner loop: iterate over depth buckets ──
            for k in range(self._max_k):
                delta_k = (k + 1) * half_tick   # distance for this bucket
                found_fill = False
                fill_time = 0.0

                # Scan MO events in CHRONOLOGICAL order within this window.
                # The FIRST qualifying MO constitutes the fill (a real LO
                # would be consumed by the first fill and removed from the book).
                for j in range(n_win):
                    d_tick = win_ticks[j]
                    # Actual depth reached by this MO (in price units).
                    # The "+1" accounts for the tick-index convention:
                    # d_tick = 0 → the MO touched the best quote = 1·half_tick.
                    depth_reached = (d_tick + 1) * half_tick

                    # ── Mid-drift correction ──
                    # (Only applied when side is specified and mid₀ is known.)
                    if has_mid0 and target_side != 0:
                        mid_j = win_mids[j]
                        if not math.isfinite(mid_j):
                            continue   # Skip events with missing mid data.

                        if target_side == +1:
                            # ── Buy LO (bid side) ──
                            # Our LO sits at P_lo = mid₀ − δ_k (below mid₀).
                            # At event j, the mid has moved to mid_j.
                            # Effective delta from mid_j to P_lo:
                            #   δ_eff = P_lo − mid_j = ... = δ_k + (mid_j − mid₀)
                            #
                            # If mid went UP (mid_j > mid₀):
                            #   δ_eff > δ_k → LO is further from current mid
                            #   → harder to fill (MO must sweep deeper).
                            #
                            # If mid went DOWN past P_lo (mid_j < mid₀ − δ_k):
                            #   δ_eff < 0 → LO is above current mid
                            #   → any sell MO fills it trivially.
                            delta_eff = delta_k + (mid_j - mid_0)
                        else:
                            # ── Sell LO (ask side) ──
                            # Symmetric: P_lo = mid₀ + δ_k (above mid₀).
                            # δ_eff = δ_k + (mid₀ − mid_j)
                            # If mid went DOWN: δ_eff > δ_k → harder to fill.
                            # If mid went UP past P_lo: δ_eff < 0 → trivial fill.
                            delta_eff = delta_k + (mid_0 - mid_j)

                        # ── Special case: δ_eff ≤ 0 (trivial fill) ──
                        # The mid has drifted past the LO's price.
                        # Any MO on the correct side fills us immediately.
                        if delta_eff <= 0.0:
                            found_fill = True
                            fill_time = win_times[j]
                            break

                        # ── Standard fill check ──
                        # Did the MO sweep deep enough to reach our LO?
                        if depth_reached >= delta_eff:
                            found_fill = True
                            fill_time = win_times[j]
                            break
                    else:
                        # ── No drift correction ──
                        # (side=None → symmetric, or mid₀ unavailable.)
                        # Simple depth comparison without mid adjustment.
                        if depth_reached >= delta_k:
                            found_fill = True
                            fill_time = win_times[j]
                            break

                # ═══════════════════════════════════════════════════════
                # Compute τ̃ (censored waiting time) and I (indicator).
                #
                # These are the sufficient statistics for the exponential
                # censoring MLE:
                #   λ̂ = Σ I / Σ τ̃
                #
                # Three cases:
                #   (a) Fill found, τ > 0:  τ̃ = min(τ, ΔT), I = (τ < ΔT)
                #   (b) Fill found, τ ≤ 0:  τ̃ = ΔT, I = 0 (degenerate)
                #   (c) No fill:            τ̃ = ΔT, I = 0 (right-censored)
                # ═══════════════════════════════════════════════════════
                if found_fill:
                    tau = fill_time - t0
                    if tau <= 0.0 or not math.isfinite(tau):
                        # Case (b): fill at the placement step itself
                        # (tau ≈ 0 or negative due to timestamp rounding).
                        # We cannot distinguish this from a pre-existing
                        # queue fill, so treat conservatively as censored.
                        tau_denom[k] += dt_window
                    else:
                        # Case (a): genuine fill at a known time.
                        # τ̃ = min(τ, ΔT) — cap at window duration.
                        # I = 1 only if the fill happened BEFORE the
                        # window closed (τ < ΔT).  If τ ≥ ΔT, the fill
                        # technically happened "at" the boundary and we
                        # treat it as censored (I = 0) to be conservative.
                        tau_tilde = min(tau, dt_window)
                        I = 1 if tau < dt_window else 0
                        fill_count[k] += I
                        tau_denom[k] += tau_tilde
                else:
                    # Case (c): no qualifying MO found in this window.
                    # The LO survived the full window duration (right-censored).
                    # The exposure time is the full ΔT.
                    tau_denom[k] += dt_window

        # ═══════════════════════════════════════════════════════════════
        # Step 5: Compute per-bucket MLE.
        #
        # λ̂(δ_k) = fill_count[k] / tau_denom[k]
        #
        # This is the standard MLE for the exponential distribution with
        # right-censored data.  Mathematical proof:
        #
        #   Log-likelihood for N windows with exponential(λ) waiting times:
        #     ℓ(λ) = Σ_w [ I_w · ln(λ) − λ · τ̃_w ]
        #
        #   Setting dℓ/dλ = 0:
        #     Σ_w I_w / λ̂ = Σ_w τ̃_w
        #     → λ̂ = Σ_w I_w / Σ_w τ̃_w              ∎
        #
        # The denominator Σ τ̃ is the total "exposure time" — the cumulative
        # time during which the hypothetical LO was at risk of being filled.
        # Including censored windows (I=0, τ̃=ΔT) correctly increases the
        # denominator, preventing the upward bias that would result from
        # only counting windows with fills.
        # ═══════════════════════════════════════════════════════════════
        valid_k = []
        for k in range(self._max_k):
            if tau_denom[k] > 0:
                lambda_hat[k] = fill_count[k] / tau_denom[k]
            if lambda_hat[k] > 0:
                valid_k.append(k)

        # ── Fallback for sparse data ──
        # Need at least 2 valid λ̂ points for a meaningful regression
        # (1 point → degenerate fit with κ=1.0 as default).
        if len(valid_k) < 2:
            if len(valid_k) == 1:
                k0 = valid_k[0]
                d0 = (k0 + 1) * half_tick
                # Heuristic: assume κ=1.0 and solve for A from the single point.
                A_fb = lambda_hat[k0] * math.exp(1.0 * d0)
                if A_fb > 0:
                    return A_fb, 1.0
            return None

        # ═══════════════════════════════════════════════════════════════
        # Step 6: WLS fit on log(λ̂) vs δ.
        #
        # Model: log(λ̂[k]) = log(A) − κ · δ_k
        # This is a standard linear regression in the log-transformed space.
        #
        # Weights = tau_denom[k] (total exposure time at depth k).
        # WHY tau_denom as weights?  The variance of the MLE λ̂ is
        # approximately 1/(n·λ) = 1/Σ(τ̃) for the exponential model.
        # So Var(log λ̂) ≈ 1/(Σ I) ≈ 1/(λ · Σ τ̃).  Using weights ∝ Σ τ̃
        # gives an approximate GLS (Generalised Least Squares) weighting
        # that is proportional to the Fisher information of each MLE point.
        #
        # WLS normal equations (same as WLS in fit_A_kappa):
        #   β = Σ w(x−x̄)(y−ȳ) / Σ w(x−x̄)²
        #   α = ȳ − β·x̄
        # where y = log(λ̂), x = δ, w = tau_denom.
        # Then: κ = −β, A = exp(α).
        # ═══════════════════════════════════════════════════════════════
        deltas_fit = np.array([(k + 1) * half_tick for k in valid_k])
        log_lam = np.log(np.array([lambda_hat[k] for k in valid_k]))
        weights = np.array([tau_denom[k] for k in valid_k])

        sw = weights.sum()
        if sw <= 0:
            return None
        xbar = (weights * deltas_fit).sum() / sw
        ybar = (weights * log_lam).sum() / sw
        sxx = (weights * (deltas_fit - xbar) ** 2).sum()
        if sxx < 1e-30:
            return None    # Degenerate: all δ values identical
        sxy = (weights * (deltas_fit - xbar) * (log_lam - ybar)).sum()

        slope = sxy / sxx              # β = −κ
        intercept = ybar - slope * xbar  # α = log(A)

        kappa_hat = -slope
        A_hat = math.exp(intercept)

        # Reject non-positive parameters (violate the exponential model).
        if kappa_hat <= 0 or A_hat <= 0:
            return None

        # ═══════════════════════════════════════════════════════════════
        # Step 7: Store last fit result for diagnostic plotting.
        #
        # The ``_plot_censored_fit()`` helper reads this dict to produce
        # the scatter plot of λ̂(δ) points and the fitted A·exp(−κδ) curve.
        # Also useful for post-hoc analysis and debugging.
        #
        # Results are stored PER SIDE in a dict keyed by side label
        # ("bid", "ask", or "combined"), so that two-sided fits do not
        # overwrite each other.
        # ═══════════════════════════════════════════════════════════════
        fit_result = {
            "delta_grid": np.array([(k + 1) * half_tick for k in range(self._max_k)]),
            "lambda_hat": lambda_hat.copy(),
            "fill_count": fill_count.copy(),
            "tau_denom": tau_denom.copy(),
            "A": A_hat,
            "kappa": kappa_hat,
            "n_windows": len(windows),
            "side": side,
        }
        if self._last_censored_fit is None:
            self._last_censored_fit = {}
        side_key = side if side is not None else "combined"
        self._last_censored_fit[side_key] = fit_result

        return A_hat, kappa_hat

    # -----------------------------------------------------------------
    # State Management
    #
    # Two operations for controlling the tracker's memory:
    #
    #   reset() — Nuclear option.  Clears ALL data (both methods).
    #             Used at day boundaries when ALL historical data is
    #             invalidated (new day = new microstructure regime).
    #
    #   decay() — Soft forgetting.  Multiplies WLS accumulators by a
    #             factor ∈ (0, 1), geometrically down-weighting old data.
    #             Called after each recalibration so that recent observations
    #             dominate the next fit.  Does NOT affect censored WT data
    #             (which uses FIFO eviction via bounded deques instead).
    # -----------------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated data (hard reset).

        Called at day boundaries to prevent stale data from a previous
        trading session from contaminating the new day's calibration.
        Clears BOTH estimation methods:
          - WLS: tail counts, event count, elapsed time
          - Censored: raw MO events, window boundaries, last fit result
        """
        # WLS accumulators
        self._n_events = 0.0
        self._tail_buy[:] = 0.0
        self._tail_sell[:] = 0.0
        self._elapsed_time = 0.0
        self._last_time = None
        # Censored WT data
        self._mo_events_buy.clear()
        self._mo_events_sell.clear()
        self._windows.clear()
        self._current_window_start = None
        self._current_window_mid = None
        self._last_censored_fit = {}

    def decay(self, factor: float = 0.7) -> None:
        """Exponentially forget old WLS observations (soft decay).

        Multiplies all WLS accumulators by ``factor`` ∈ (0, 1):
            tail[k]      ← tail[k] · factor
            n_events     ← n_events · factor
            elapsed_time ← elapsed_time · factor

        This implements geometric forgetting: an observation from N
        recalibrations ago has effective weight factor^N.  With the
        default factor=0.95 and recalibration every 20 gates, an
        observation from 100 gates ago has weight 0.95^5 ≈ 0.77;
        from 500 gates ago: 0.95^25 ≈ 0.28.

        **Only affects WLS** (Method 1).  The censored WT method (Method 2)
        uses bounded deques with FIFO eviction instead — when the deque
        is full, the oldest events are automatically discarded.  This is
        a design choice: exponential decay on discrete tail counts can
        create "ghost" fractional observations (the FIX 1 problem), while
        FIFO eviction cleanly removes the oldest complete events.

        Parameters
        ----------
        factor : float
            Decay factor in (0, 1).  Typical values:
              0.95 → slow forgetting (long memory, stable but slow to adapt)
              0.80 → moderate forgetting
              0.50 → aggressive forgetting (short memory, fast adaptation)
        """
        self._n_events = self._n_events * factor
        self._tail_buy *= factor
        self._tail_sell *= factor
        self._elapsed_time *= factor

    @property
    def n_mos(self) -> int:
        """Total MO count at the shallowest depth level (both sides).

        This is tail_buy[0] + tail_sell[0], i.e., the number of MOs that
        reached at least depth 0 (= touched the best quote).  Since every
        MO reaches at least the best quote, this equals the total MO count
        (modulo decay).  Used as a minimum-data check before fitting.
        """
        return int(self._tail_buy[0] + self._tail_sell[0])

    @property
    def n_events(self) -> int:
        """Total event count (MOs + non-MOs), subject to decay."""
        return int(self._n_events)


# =============================================================================
#
#  SECTION 4: FACTORY FUNCTION
#
#  The factory pattern encapsulates all parameter validation, initial
#  coefficient computation, and closure state setup.  It returns a
#  lightweight ``policy(state, mm)`` callable that the backtest engine
#  invokes at every simulation tick.
#
# =============================================================================

def glft_enhanced_policy_factory(
    # ── Base GLFT parameters ──
    gamma: float,
    kappa: Optional[float] = None,
    A: Optional[float] = None,
    sigma: Optional[float] = None,
    inv_limit: int = 10,
    round_to_int: bool = True,
    delta: float = 1.0,
    epsilon: Optional[float] = None,

    # ── Post-GLFT adjustment factors ──
    adj_spread: float = 1.0,
    adj_skew: float = 1.0,

    # ── Microprice ──
    use_micro: bool = False,
    micro_n_levels: int = 3,
    micro_decay: float = 1.0,

    # ── Online Recalibration ──
    update_params: bool = False,
    recalib_every_n: int = 20,
    recalib_window: int = 500,
    recalib_min_fills: int = 15,
    recalib_blend: float = 0.3,
    recalib_decay: float = 0.95,

    # ── Warmup (observation-only period before trading) ──
    warmup_gates: int = 0,

    # ── Two-Sided Asymmetric Parameters ──
    use_two_sided: bool = False,
    kappa_bid: Optional[float] = None,
    A_bid: Optional[float] = None,
    sigma_bid: Optional[float] = None,
    kappa_ask: Optional[float] = None,
    A_ask: Optional[float] = None,
    sigma_ask: Optional[float] = None,

    # ── Throttling (rate-limiting) ──
    use_tob_update: bool = False,
    n_tob_moves: int = 10,
    use_event_update: bool = False,
    n_events: int = 100,
    use_time_update: bool = False,
    min_time_interval: float = 1.0,

    # ── MDP Mode (strict gate-based control) ──
    use_mdp: bool = False,

    # ── Terminal Inventory Liquidation ──
    liquidate_terminal: bool = False,

    # ── Calibration Sampling Frequency ──
    # Decouples calibration data collection from the strategy throttle.
    # When set (e.g., 1.0s), mid-price samples and censored MLE windows
    # are recorded at this interval — much higher frequency than the
    # strategy gate (min_time_interval).  This gives the volatility
    # signature and censored MLE far more data points per recalibration.
    # When None, falls back to the legacy behaviour (sample at gate rate).
    time_interval_calib: Optional[float] = None,

    # ── Volatility Signature Plateau ──
    # When time_interval_calib is set, sigma is estimated via the
    # volatility signature curve: σ(τ) = √(E[Δm²]/τ).  These params
    # define the SEARCH range (in calib buckets) over which the auto-
    # plateau detector looks for the flattest window.  The detector
    # slides a window across σ(τ) and picks the region with minimum
    # coefficient of variation (CV = std/mean) — the "true" diffusive σ.
    sigma_plateau_min: int = 5,     # start of search range (in calib buckets)
    sigma_plateau_max: int = 500,   # end of search range (in calib buckets)

    # ── Censored Waiting Times Calibration ──
    use_censored_calib: bool = False,
    plot_recalib: bool = False,
    use_log_scale_plot: bool = True,

    # ── Progress Bar ──
    n_total_steps: Optional[int] = None,
):
    """
    Enhanced GLFT market-making policy factory.

    Creates and returns a ``policy(state, mm)`` callable that implements
    the full GLFT optimal quoting logic with online recalibration,
    microprice, and two-sided parameter support.

    Parameters
    ----------
    gamma : float
        Risk aversion parameter (γ > 0).  Higher values → wider spreads
        and more aggressive inventory skew (risk-averse).

    kappa, A, sigma : float or None
        Base GLFT intensity parameters.  If any are None, the policy
        enters warmup mode and will not trade until it has observed
        enough market data to calibrate them.  Requires ``update_params=True``
        and ``warmup_gates > 0``.

    inv_limit : int
        Maximum absolute inventory.  When |inventory| ≥ inv_limit, the
        policy switches to one-sided quoting (ask_only or bid_only) to
        reduce exposure.

    round_to_int : bool
        If True, bid prices are floored and ask prices are ceiled to
        integer tick indices (standard for tick-based LOB simulators).

    delta : float
        Lot size parameter in the GLFT equations (default 1.0).

    epsilon : float or None
        Terminal inventory penalty.  If None, defaults to gamma.

    adj_spread : float
        Multiplicative scaling factor on the computed half-spread.
        1.0 = theoretical GLFT spread, >1.0 = wider, <1.0 = tighter.

    adj_skew : float
        Multiplicative scaling factor on the inventory skew.
        1.0 = full GLFT skew, 0.0 = no inventory skew at all.

    use_micro : bool
        Enable n-level microprice as the fair-value reference.

    micro_n_levels : int
        Number of order book levels to include in microprice.

    micro_decay : float
        Exponential decay per level for microprice weights (1.0 = uniform).

    update_params : bool
        Enable online recalibration of A, κ, σ.

    recalib_every_n : int
        Recalibrate every N gate activations.

    recalib_window : int
        Maximum size of the rolling observation window (number of gates).

    recalib_min_fills : int
        Minimum MO count in the tracker before recalibrating A/κ.

    recalib_blend : float
        EMA blending factor: param = blend · new_estimate + (1−blend) · old.

    recalib_decay : float
        Exponential decay factor applied to the intensity tracker after
        each recalibration.  Controls how fast old data is forgotten.

    warmup_gates : int
        Number of gate activations to observe before trading begins.
        During warmup, the policy returns ("hold",) but collects data.

    use_two_sided : bool
        Maintain separate (A, κ, σ) for bid and ask sides.

    kappa_bid, A_bid, sigma_bid, kappa_ask, A_ask, sigma_ask : float or None
        Initial per-side parameters.  Default to the symmetric values.

    use_tob_update, n_tob_moves : bool, int
        TOB (Top-of-Book) change throttle.

    use_event_update, n_events : bool, int
        Event count throttle.

    use_time_update, min_time_interval : bool, float
        Wall-clock time throttle (seconds between gate openings).

    use_mdp : bool
        If True, enforce strict Markov Decision Process timing —
        actions only occur when the throttle gate fires.  Gate 1 and
        Gate 2 (mode-change and fill-replenishment) are suppressed.

    liquidate_terminal : bool
        If True, the policy liquidates all open inventory at market
        prices when a day boundary is detected (i.e., before the new
        trading day begins).  This simulates a firm that mandates flat
        positions overnight.  Liquidation is performed via aggressive
        market orders (``mm.cross_mo``) at the prevailing best bid/ask,
        so the resulting execution prices reflect realistic end-of-day
        slippage.  The cash and inventory impact is recorded by the
        backtest engine as any other market order fill.

    use_censored_calib : bool
        If True, replace the online tail-counting A/κ estimation with
        the statistically rigorous **censored waiting times** MLE,
        computed entirely online inside the ``_MarketIntensityTracker``.

        At each throttle gate, the tracker records a window boundary.
        For each depth δ_k and each completed window [t0, t1):
          - If a MO reached depth ≥ δ_k at time t_fill:
            τ̃ = t_fill − t0  (uncensored observation, I=1)
          - If no MO reached that depth:
            τ̃ = t1 − t0      (right-censored observation, I=0)

        The MLE for the exponential model:
            λ̂(δ_k) = Σ_w I_w / Σ_w τ̃_w

        Then A and κ are extracted via WLS on log(λ̂) vs δ, weighted
        by Σ τ̃ (proportional to Fisher information).

        This method handles right-censored observations correctly,
        producing unbiased intensity estimates even in illiquid markets.
        No external DataFrames are required — all data is accumulated
        online from the events already observed via ``observe()``.

        Falls back to the WLS tracker method if the censored fit fails.

    Returns
    -------
    policy : callable
        ``policy(state, mm) -> Tuple`` with the standard action interface.
    """

    # =====================================================================
    # 4.1) Parameter Validation
    # =====================================================================
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    _needs_warmup = (A is None or kappa is None or sigma is None)
    if _needs_warmup and warmup_gates <= 0:
        raise ValueError(
            "kappa, A, sigma are None but warmup_gates=0.  Either provide "
            "initial parameters or set warmup_gates > 0 to calibrate from data."
        )
    if _needs_warmup and not update_params:
        raise ValueError(
            "kappa, A, sigma are None but update_params=False.  "
            "Online recalibration must be enabled to calibrate from data."
        )
    if A is not None and A <= 0:
        raise ValueError("A must be > 0.")
    if kappa is not None and kappa <= 0:
        raise ValueError("kappa must be > 0.")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be > 0.")

    if int(inv_limit) < 1:
        raise ValueError("inv_limit must be >= 1.")
    if delta <= 0:
        raise ValueError("delta must be > 0.")
    if epsilon is not None and epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if use_time_update and min_time_interval <= 0:
        raise ValueError("min_time_interval must be > 0.")
    if micro_n_levels < 1:
        raise ValueError("micro_n_levels must be >= 1.")
    if not (0.0 <= recalib_blend <= 1.0):
        raise ValueError("recalib_blend must be in [0, 1].")
    if warmup_gates < 0:
        raise ValueError("warmup_gates must be >= 0.")

    threshold_tob = max(1, int(n_tob_moves))
    threshold_events = max(1, int(n_events))

    # =====================================================================
    # 4.2) Initialize GLFT Coefficients
    # =====================================================================
    val_epsilon = float(epsilon) if epsilon is not None else float(gamma)
    val_delta = float(delta)

    # Placeholder defaults for warmup mode (overwritten by first calibration).
    _A_default = float(A) if A is not None else 1.0
    _k_default = float(kappa) if kappa is not None else 1.0
    _s_default = float(sigma) if sigma is not None else 1.0

    _A_bid = float(A_bid) if A_bid is not None else _A_default
    _k_bid = float(kappa_bid) if kappa_bid is not None else _k_default
    _s_bid = float(sigma_bid) if sigma_bid is not None else _s_default

    _A_ask = float(A_ask) if A_ask is not None else _A_default
    _k_ask = float(kappa_ask) if kappa_ask is not None else _k_default
    _s_ask = float(sigma_ask) if sigma_ask is not None else _s_default

    # Mutable parameter dict (updated by online recalibration).
    params = {
        "A_bid": _A_bid, "kappa_bid": _k_bid, "sigma_bid": _s_bid,
        "A_ask": _A_ask, "kappa_ask": _k_ask, "sigma_ask": _s_ask,
    }

    # Parameter history for plotting evolution over time.
    # Each entry is (time, dict) where dict contains the current params
    # plus optional keys: plateau_sym, plateau_bid, plateau_ask (raw
    # unblended σ estimates from the volatility signature).
    param_history: List[Tuple[float, Dict[str, float]]] = []
    # Latest raw plateau values (updated by recalibration, stored in history).
    _latest_plateau: Dict[str, Optional[float]] = {
        "sym": None, "bid": None, "ask": None,
    }
    # PnL history: (time, cash, inventory, mid, wealth)
    _pnl_history: List[Tuple[float, float, float, float, float]] = []
    # Level history: (time, bid_offset, ask_offset) where offset is relative
    # to best_bid (for bids) and best_ask (for asks).  Level 0 = at the BBO.
    _level_history: List[Tuple[float, Optional[int], Optional[int]]] = []
    if not _needs_warmup:
        param_history.append((0.0, dict(params)))

    def _recompute_coefficients():
        """Recompute GLFT half-spread and skew coefficients from current params."""
        if use_two_sided:
            c1_b, c2_b, hs_b, sk_b = _compute_glft_coefficients(
                gamma, params["kappa_bid"], params["A_bid"], params["sigma_bid"],
                val_delta, val_epsilon)
            c1_a, c2_a, hs_a, sk_a = _compute_glft_coefficients(
                gamma, params["kappa_ask"], params["A_ask"], params["sigma_ask"],
                val_delta, val_epsilon)
            coeffs["hs_bid"] = hs_b
            coeffs["sk_bid"] = sk_b
            coeffs["hs_ask"] = hs_a
            coeffs["sk_ask"] = sk_a
        else:
            c1, c2, hs, sk = _compute_glft_coefficients(
                gamma, params["kappa_bid"], params["A_bid"], params["sigma_bid"],
                val_delta, val_epsilon)
            coeffs["hs_bid"] = hs
            coeffs["sk_bid"] = sk
            coeffs["hs_ask"] = hs
            coeffs["sk_ask"] = sk

    coeffs: Dict[str, float] = {}
    _recompute_coefficients()

    # =====================================================================
    # 4.3) Initialization Log
    # =====================================================================
    print("-" * 65)
    print("[GLFT Enhanced] Initialized:")
    print(f"  gamma={gamma}, inv_limit={inv_limit}, delta={val_delta}, eps={val_epsilon}")
    if _needs_warmup:
        print(f"  ** WARMUP MODE: {warmup_gates} gates of observation before trading **")
        print(f"  Initial params: A={A}, kappa={kappa}, sigma={sigma} (will be calibrated)")
    elif use_two_sided:
        print(f"  BID: A={params['A_bid']:.4f}, kappa={params['kappa_bid']:.4f}, "
              f"sigma={params['sigma_bid']:.6f}")
        print(f"  ASK: A={params['A_ask']:.4f}, kappa={params['kappa_ask']:.4f}, "
              f"sigma={params['sigma_ask']:.6f}")
    else:
        print(f"  A={params['A_bid']:.4f}, kappa={params['kappa_bid']:.4f}, "
              f"sigma={params['sigma_bid']:.6f}")
    print(f"  Microprice: {use_micro} (n_levels={micro_n_levels}, decay={micro_decay})")
    print(f"  Recalibration: {update_params} (every={recalib_every_n}, "
          f"blend={recalib_blend}, window={recalib_window}, decay={recalib_decay})")
    print(f"  Warmup: {warmup_gates} gates")
    print(f"  Adjustments: adj_spread={adj_spread}, adj_skew={adj_skew}")
    print(f"  Two-Sided: {use_two_sided}")
    print(f"  Throttles: Evt={use_event_update}({n_events}), "
          f"Time={use_time_update}({min_time_interval}s), "
          f"TOB={use_tob_update}({n_tob_moves}), MDP={use_mdp}")
    if not _needs_warmup:
        print(f"  [Coeffs] HS_bid={coeffs['hs_bid']:.4f}, SK_bid={coeffs['sk_bid']:.4f}, "
              f"HS_ask={coeffs['hs_ask']:.4f}, SK_ask={coeffs['sk_ask']:.4f}")
    print("-" * 65)

    # =====================================================================
    # 4.4) Closure State
    #
    # All mutable state lives in this dict and in the mutable containers
    # below.  The closure pattern avoids class boilerplate while keeping
    # state encapsulated and invisible to the caller.
    # =====================================================================
    st = {
        # ── Quote tracking ──
        "last_bid_px": None,        # last bid price placed (tick index)
        "last_ask_px": None,        # last ask price placed (tick index)
        "last_mode": None,          # "two_sided", "bid_only", or "ask_only"

        # ── Throttle clocks ──
        "last_env_tob_key": None,   # (best_bid, best_ask, bid_size, ask_size) tuple
        "moves_tob": 0,            # TOB changes since last gate
        "event_steps": 0,          # events since last gate
        "last_update_time": -1.0,  # timestamp of last gate firing

        # ── Fill tracking ──
        "last_inv": None,           # previous inventory (for delta detection)
        "last_has_bid": False,      # previous has_bid state
        "last_has_ask": False,      # previous has_ask state
        "n_fills": 0,              # total fills
        "n_fills_bid": 0,          # bid-side fills
        "n_fills_ask": 0,          # ask-side fills

        # ── Recalibration counters ──
        "gate_count": 0,           # total gates fired (for recalib_every_n modulo)
        "recalib_count": 0,        # total recalibrations performed

        # ── Warmup state ──
        "warming_up": warmup_gates > 0,
        "warmup_gate_count": 0,    # gates observed during warmup
    }

    prev_base_idx = [None]          # for grid-shift detection

    # ── Recalibration Data Stores ──
    #
    # LEGACY (gate-level): used when time_interval_calib is None.
    #   Sigma estimation at the THROTTLE GATE timescale:
    #     σ = RMS(gate_returns) · √(1/avg_gate_ΔT)
    #
    # HIGH-FREQ (time_interval_calib): used when time_interval_calib is set.
    #   Sigma estimation via volatility signature plateau:
    #     σ(τ) = √(E[Δm²]/τ), extract plateau for diffusive σ.
    #   Windows for censored MLE are also created at this frequency.
    mid_returns: deque = deque(maxlen=recalib_window)
    gate_dts: deque = deque(maxlen=recalib_window)
    mid_at_last_gate = [None]
    time_at_last_gate = [None]

    # ── High-frequency mid sampler (for sigma signature + censored MLE) ──
    # Samples mid-price every time_interval_calib seconds.  The deque is
    # bounded to hold recalib_window gates worth of HF samples so that
    # recalibration uses the same temporal depth as the gate-level mode.
    #
    # Conversion: recalib_window gates × (gate_interval / calib_interval)
    # gives the number of HF samples that span the same wall-clock window.
    # Example: recalib_window=30, gate=60s, calib=1s → 30×60 = 1800 samples.
    if time_interval_calib is not None and time_interval_calib > 0:
        _hf_samples_per_gate = max(1, int(round(min_time_interval / time_interval_calib)))
        _hf_max_windows = recalib_window * _hf_samples_per_gate
        # Deque holds exactly _hf_max_windows samples so that σ, A, κ
        # all operate over the same temporal window.
        _hf_mids_maxlen = max(2000, _hf_max_windows)
    else:
        _hf_samples_per_gate = 1
        _hf_max_windows = recalib_window
        _hf_mids_maxlen = 0
    _hf_mids: deque = deque(maxlen=_hf_mids_maxlen)
    _hf_last_sample_time = [None]

    def _hf_sample(current_time: float, mid_now: float):
        """High-frequency sampler: record mid and mark censored-MLE window."""
        if time_interval_calib is None or time_interval_calib <= 0:
            return
        if not math.isfinite(mid_now):
            return

        last_t = _hf_last_sample_time[0]
        if last_t is not None:
            dt = current_time - last_t
            if dt < 0:
                # Time wrap (day boundary) — reset
                _hf_last_sample_time[0] = current_time
                return
            if dt < time_interval_calib:
                return  # not yet time for next sample

        _hf_mids.append((current_time, mid_now))
        # Mark a window boundary for censored MLE at high frequency.
        market_tracker.mark_window(current_time, mid_now=mid_now)
        _hf_last_sample_time[0] = current_time

    # Cache for the latest signature curve data (used by the signature plot).
    _last_signature: Dict[str, Any] = {
        "taus_sec": None,     # τ values in seconds
        "sig_sym": None,      # σ_sym(τ) array in ticks/√s
        "sig_up": None,       # σ_up(τ) array in ticks/√s (or None)
        "sig_dn": None,       # σ_dn(τ) array in ticks/√s (or None)
        "plateau_sym": None,  # scalar plateau value
        "plateau_up": None,
        "plateau_dn": None,
        "plat_lo_sec": None,  # detected plateau window left bound (seconds)
        "plat_hi_sec": None,  # detected plateau window right bound (seconds)
    }

    def _find_plateau(sig_arr, taus_arr, min_window: int = 5):
        """Auto-detect the flattest region of a signature curve.

        Slides a window across the valid (> 0) entries of ``sig_arr``
        and returns the window whose coefficient of variation
        (CV = std / mean) is minimal — i.e. the most constant region.

        Parameters
        ----------
        sig_arr : np.ndarray
            σ(τ) values (1-D, same length as ``taus_arr``).
        taus_arr : np.ndarray
            Corresponding τ values (used to report bounds).
        min_window : int
            Minimum number of points in the sliding window.

        Returns
        -------
        (plateau_value, lo_tau, hi_tau) or (None, None, None)
            plateau_value : median of the detected plateau window.
            lo_tau, hi_tau : τ boundaries of that window.
        """
        valid_mask = sig_arr > 0
        idx_valid = np.where(valid_mask)[0]
        if len(idx_valid) < min_window:
            return None, None, None

        vals = sig_arr[idx_valid]
        tau_vals = taus_arr[idx_valid]
        n = len(vals)

        # Try window sizes from min_window up to ~60 % of the curve,
        # pick the (window_size, position) with lowest CV.
        best_cv = float("inf")
        best_lo = 0
        best_hi = n - 1
        max_win = max(min_window, int(n * 0.6))

        for w in range(min_window, max_win + 1):
            # Cumulative sums for O(1) per-window mean/var
            cs = np.cumsum(vals)
            cs2 = np.cumsum(vals ** 2)
            for start in range(n - w + 1):
                end = start + w
                s = cs[end - 1] - (cs[start - 1] if start > 0 else 0.0)
                s2 = cs2[end - 1] - (cs2[start - 1] if start > 0 else 0.0)
                mean = s / w
                if mean <= 0:
                    continue
                var = s2 / w - mean ** 2
                std = math.sqrt(max(0.0, var))
                cv = std / mean
                if cv < best_cv:
                    best_cv = cv
                    best_lo = start
                    best_hi = end - 1

        plateau_val = float(np.median(vals[best_lo:best_hi + 1]))
        return plateau_val, float(tau_vals[best_lo]), float(tau_vals[best_hi])

    def _compute_sigma_signature(two_sided: bool = False):
        """Compute σ from the volatility signature plateau.

        Uses automatic plateau detection: slides a window across the
        σ(τ) curve and picks the region with minimum coefficient of
        variation (CV = std/mean).  The plateau σ is the **median**
        of that window (robust to outliers at the edges).

        Returns (sigma_sym, sigma_up, sigma_dn) where:
          sigma_sym : symmetric σ (full RMS)
          sigma_up  : upward semivariance σ (for ask side)
          sigma_dn  : downward semivariance σ (for bid side)
        All in ticks/√second.  Returns (None, None, None) if insufficient data.

        Side effect: stores the raw signature curve in ``_last_signature``
        for plotting (including detected plateau boundaries).
        """
        if len(_hf_mids) < 50:
            return None, None, None

        # Use only the last _hf_max_windows samples so that σ operates
        # over the same temporal window as the censored MLE (A/κ).
        _all_mids = [m for _, m in _hf_mids]
        if len(_all_mids) > _hf_max_windows:
            _all_mids = _all_mids[-_hf_max_windows:]
        mids = np.array(_all_mids)

        # Build tau grid over the full search range.
        n = len(mids)
        tau_min = max(1, sigma_plateau_min)
        tau_max = min(sigma_plateau_max, n // 3)
        if tau_max <= tau_min:
            tau_max = max(tau_min + 1, n // 4)
        if tau_max <= tau_min or tau_min >= n:
            return None, None, None

        taus = np.unique(np.geomspace(tau_min, tau_max, num=50, dtype=int))
        taus = taus[(taus >= 1) & (taus < n)]
        if len(taus) < 5:
            return None, None, None

        # σ(τ) = √(E[Δm²] / τ)   [ticks / √bucket]
        dt_calib = float(time_interval_calib)
        scale_to_sec = math.sqrt(1.0 / dt_calib) if dt_calib > 0 else 1.0

        sig_sym = np.zeros(len(taus))
        sig_up = np.zeros(len(taus))
        sig_dn = np.zeros(len(taus))

        for i, tau in enumerate(taus):
            diffs = mids[tau:] - mids[:-tau]
            if len(diffs) == 0:
                continue
            var = float(np.nanmean(diffs ** 2))
            sig_sym[i] = math.sqrt(max(0.0, var / tau))

            if two_sided:
                ups = np.maximum(diffs, 0.0)
                dns = np.abs(np.minimum(diffs, 0.0))
                sig_up[i] = math.sqrt(max(0.0, float(np.nanmean(ups ** 2)) / tau))
                sig_dn[i] = math.sqrt(max(0.0, float(np.nanmean(dns ** 2)) / tau))

        # ── Auto-detect plateau (minimum CV window) ──
        taus_f = taus.astype(float)
        plat_sym, lo_tau, hi_tau = _find_plateau(sig_sym, taus_f)
        if plat_sym is None or plat_sym <= 0:
            return None, None, None
        sigma_sym_val = plat_sym * scale_to_sec

        sigma_up_val = None
        sigma_dn_val = None
        plat_lo_sec = lo_tau * dt_calib if lo_tau is not None else None
        plat_hi_sec = hi_tau * dt_calib if hi_tau is not None else None
        if two_sided:
            p_up, _, _ = _find_plateau(sig_up, taus_f)
            p_dn, _, _ = _find_plateau(sig_dn, taus_f)
            if p_up is not None and p_up > 0:
                sigma_up_val = p_up * scale_to_sec
            if p_dn is not None and p_dn > 0:
                sigma_dn_val = p_dn * scale_to_sec

        # Cache curve data for plotting (convert taus to seconds)
        taus_sec = taus_f * dt_calib
        _last_signature["taus_sec"] = taus_sec
        _last_signature["sig_sym"] = sig_sym * scale_to_sec
        _last_signature["sig_up"] = (sig_up * scale_to_sec) if two_sided else None
        _last_signature["sig_dn"] = (sig_dn * scale_to_sec) if two_sided else None
        _last_signature["plateau_sym"] = sigma_sym_val
        _last_signature["plateau_up"] = sigma_up_val
        _last_signature["plateau_dn"] = sigma_dn_val
        _last_signature["plat_lo_sec"] = plat_lo_sec
        _last_signature["plat_hi_sec"] = plat_hi_sec

        return sigma_sym_val, sigma_up_val, sigma_dn_val

    # Market intensity tracker (A, κ estimation from public tape).
    market_tracker = _MarketIntensityTracker(max_depth_levels=20, half_tick=0.5)

    # ══════════════════════════════════════════════════════════════════════
    #  DAY-BOUNDARY DETECTION
    #
    #  When backtesting on concatenated multi-day LOBSTER data, the
    #  timestamp resets at the start of each new day (e.g., 55799 → 36000).
    #  Without detection, this causes:
    #    - Permanent throttle lock (current_time − last_update_time < 0)
    #    - Stale parameters from prior day's closing regime
    #    - Corrupted sigma estimates from phantom price jumps
    #
    #  We track the last seen timestamp and trigger a full reset when
    #  time jumps backwards by more than 5 minutes (300 seconds).
    # ══════════════════════════════════════════════════════════════════════
    last_seen_time = [None]

    # =====================================================================
    # 4.5) Helper Functions
    # =====================================================================

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

    _EPS = 1e-12

    def _bid_to_idx(x: float) -> int:
        """Convert continuous bid price to integer tick index (floor)."""
        if not round_to_int:
            return int(x)
        return int(math.floor(x + _EPS))

    def _ask_to_idx(x: float) -> int:
        """Convert continuous ask price to integer tick index (ceil)."""
        if not round_to_int:
            return int(x)
        return int(math.ceil(x - _EPS))

    def _get_env_tob(s):
        """Extract top-of-book signature for TOB change detection."""
        env_bb = _safe_int(s.get("best_bid_env", s.get("best_bid", -1)), -1)
        env_ba = _safe_int(s.get("best_ask_env", s.get("best_ask", -1)), -1)
        env_bs = _safe_float(s.get("bidsize_env", s.get("bidsize", 0.0)), 0.0)
        env_as = _safe_float(s.get("asksize_env", s.get("asksize", 0.0)), 0.0)
        try:
            ibs, ias = int(round(env_bs)), int(round(env_as))
        except Exception:
            ibs, ias = 0, 0
        return (env_bb, env_ba, ibs, ias)

    # -----------------------------------------------------------------
    # Fill Detection
    #
    # Detects fills by comparing current vs previous inventory and
    # order presence.  Handles both direct inventory changes and
    # "ghost fills" where inventory doesn't change but an order
    # disappears (e.g., partial fill of size 0 in the sim).
    # -----------------------------------------------------------------
    def _consume_fill(inv, has_bid, has_ask):
        if st["last_inv"] is None:
            st["last_inv"] = int(inv)
            st["last_has_bid"] = has_bid
            st["last_has_ask"] = has_ask
            return False, 0, 0

        delta_inv = int(inv) - int(st["last_inv"])
        had_bid = st["last_has_bid"]
        had_ask = st["last_has_ask"]

        st["last_inv"] = int(inv)
        st["last_has_bid"] = has_bid
        st["last_has_ask"] = has_ask

        bid_fills = max(0, delta_inv)       # inventory up → bid was filled
        ask_fills = max(0, -delta_inv)      # inventory down → ask was filled

        # Ghost fill detection: order disappeared but inventory unchanged.
        if delta_inv == 0:
            if had_bid and not has_bid:
                bid_fills = max(bid_fills, 1)
            if had_ask and not has_ask:
                ask_fills = max(ask_fills, 1)

        total = bid_fills + ask_fills
        if total == 0:
            return False, 0, 0
        return True, bid_fills, ask_fills

    # -----------------------------------------------------------------
    # Clock Management
    #
    # Resets all throttle counters after a gate fires (action taken).
    # -----------------------------------------------------------------
    def _update_tob_clock(key):
        if st["last_env_tob_key"] != key:
            st["moves_tob"] += 1
            st["last_env_tob_key"] = key

    def _reset_all_clocks(s, current_time):
        if use_tob_update:
            st["moves_tob"] = 0
            st["last_env_tob_key"] = _get_env_tob(s)
        if use_event_update:
            st["event_steps"] = 0
        if use_time_update:
            st["last_update_time"] = current_time

    # -----------------------------------------------------------------
    # Microprice Reference Price
    # -----------------------------------------------------------------
    def _get_ref_price(state, mm) -> float:
        """Return the fair-value reference price (mid or microprice)."""
        mid = _safe_float(state.get("mid", 0.0), float("nan"))

        if not use_micro:
            return mid

        depth_bids = None
        depth_asks = None

        if mm is not None:
            snap = getattr(mm, "_last_env_snapshot", None)
            if snap is not None:
                depth_bids = snap.get("full_depth_bids", None)
                depth_asks = snap.get("full_depth_asks", None)

        if depth_bids is None or depth_asks is None:
            return mid

        micro = _compute_microprice(depth_bids, depth_asks,
                                    micro_n_levels, micro_decay)
        if micro is None:
            return mid

        return micro

    # ══════════════════════════════════════════════════════════════════════
    #  DAY-BOUNDARY RESET
    #
    #  When a new trading day is detected (time jumps backwards by >300s),
    #  this function resets ALL calibration state so the bot starts fresh.
    #
    #  Why this is necessary:
    #  - Market microstructure changes dramatically between days (opening
    #    vs closing regime: volatility, MO intensity, book depth).
    #  - Carrying over parameters from the previous day's close would
    #    cause the bot to quote with stale spreads/skew for 20+ minutes
    #    until the EMA converges to the new regime.
    #  - The warmup period ensures the bot observes the new day's
    #    characteristics before committing capital.
    #
    #  What gets reset:
    #  - All sigma estimation data (mid_returns, gate_dts, anchors)
    #  - The MarketIntensityTracker (A, κ estimation data)
    #  - All closure state (quote tracking, throttle clocks, fill tracking)
    #  - Warmup is re-entered if warmup_gates > 0
    #  - Grid shift tracking (prev_base_idx)
    #  - All existing orders are cancelled via mm.cancel_all()
    # ══════════════════════════════════════════════════════════════════════
    def _reset_for_new_day(current_time: float):
        """Reset all calibration state for a fresh trading day."""
        # Clear sigma estimation data
        mid_returns.clear()
        gate_dts.clear()
        mid_at_last_gate[0] = None
        time_at_last_gate[0] = None

        # Clear HF calibration state
        _hf_mids.clear()
        _hf_last_sample_time[0] = None

        # Clear A/κ estimation data
        market_tracker.reset()

        # Clear param history so dynamic plots start fresh for the new day
        param_history.clear()

        # Reset dynamic plot figures so old curves don't accumulate
        import matplotlib.pyplot as _plt_reset
        if _live_plot.get("fig") is not None:
            try:
                _plt_reset.close(_live_plot["fig"])
            except Exception:
                pass
            _live_plot["fig"] = None
            _live_plot["axes"] = None
            _live_plot["n_sides"] = 0
            # Reset display_handle so a new output cell is created for the
            # new day — the old cell becomes a frozen snapshot of the
            # previous day's last state.
            _live_plot["display_handle"] = None

        # Same for the param-evolution plot, signature plot, PnL and levels
        _live_plot_params["display_handle"] = None
        _live_plot_sig["display_handle"] = None
        _live_plot_pnl["display_handle"] = None
        _live_plot_levels["display_handle"] = None
        _pnl_history.clear()
        _level_history.clear()

        # Clear cached signature curve data
        for _k in _last_signature:
            _last_signature[_k] = None

        # Clear plateau tracker
        _latest_plateau["sym"] = None
        _latest_plateau["bid"] = None
        _latest_plateau["ask"] = None

        # Reset quote and throttle state
        st["last_bid_px"] = None
        st["last_ask_px"] = None
        st["last_mode"] = None
        st["last_env_tob_key"] = None
        st["moves_tob"] = 0
        st["event_steps"] = 0
        st["last_update_time"] = current_time
        st["last_inv"] = None
        st["last_has_bid"] = False
        st["last_has_ask"] = False
        st["gate_count"] = 0

        # Re-enter warmup to recalibrate from the new day's data
        if warmup_gates > 0:
            st["warming_up"] = True
            st["warmup_gate_count"] = 0

        # Reset grid shift tracking (new day = new price grid)
        prev_base_idx[0] = None

        print(f"[GLFT Enhanced] New day detected at t={current_time:.2f} — "
              f"resetting calibration, entering warmup.")

    # -----------------------------------------------------------------
    # Sigma Scale Factor
    #
    # Converts gate-level volatility to per-second units.
    #
    # The mid-price returns are measured at the throttle gate timescale
    # (e.g., every 60 seconds).  To obtain σ in "ticks per √second"
    # (the unit required by the GLFT equations), we rescale:
    #
    #     σ_per_second = RMS(gate_returns) · √(1 / avg_gate_dt)
    #
    # This is the standard √T rescaling from financial mathematics:
    # if σ_Δt is volatility over interval Δt, then σ_1s = σ_Δt / √Δt.
    # -----------------------------------------------------------------
    def _sigma_scale_factor() -> float:
        """Compute the rescaling factor from gate-level to per-second σ."""
        if len(gate_dts) < 2:
            return 1.0
        avg_dt = float(np.mean(np.array(gate_dts)))
        if avg_dt <= 0:
            return 1.0
        return math.sqrt(1.0 / avg_dt)

    # -----------------------------------------------------------------
    # Online Recalibration Engine
    #
    # Called at every gate activation.  Records the mid-price return
    # and gate elapsed time, then (every recalib_every_n gates)
    # re-estimates σ, A, and κ from the accumulated data.
    #
    # Sigma estimation:
    #   - Uses RMS (Root Mean Square) instead of std for semivariance.
    #     RMS measures deviation from zero, not from the sample mean.
    #     This correctly captures directional risk when the price is
    #     trending (std would underestimate because it removes the trend).
    #   - For two-sided mode, filters up/down returns on-the-fly from
    #     the single rolling deque (avoids deque desync during trends).
    #
    # A/κ estimation:
    #   - Censored WT MLE (use_censored_calib=True): uses per-window
    #     fill/censored observations accumulated inside the tracker.
    #   - WLS fallback: tail-counting with ghost starvation filter.
    #   - After fitting, the WLS accumulators are decayed.
    #
    # All new estimates are blended with existing params via EMA:
    #   param_new = blend · estimate + (1 − blend) · param_old
    # -----------------------------------------------------------------
    def _maybe_recalibrate(current_time: float, mid_now: float = float("nan")):
        """Record per-gate observation and maybe recalibrate parameters."""
        if not update_params:
            return

        _hf_active = (time_interval_calib is not None and time_interval_calib > 0)

        # Record a window boundary for censored waiting times.
        # When HF mode is active, _hf_sample already calls mark_window
        # at the higher cadence — skip the gate-level call to avoid
        # double-marking (which would create spurious short windows).
        if not _hf_active:
            market_tracker.mark_window(current_time, mid_now=mid_now)

        # ── Record mid-price return and gate elapsed time ──
        # When HF mode is active, sigma comes from the volatility
        # signature over _hf_mids — gate-level returns are not needed.
        if not _hf_active:
            if math.isfinite(mid_now):
                if mid_at_last_gate[0] is not None and time_at_last_gate[0] is not None:
                    dt = current_time - time_at_last_gate[0]
                    if dt > 0:
                        ret = mid_now - mid_at_last_gate[0]
                        mid_returns.append(ret)
                        gate_dts.append(dt)
                mid_at_last_gate[0] = mid_now
                time_at_last_gate[0] = current_time

        # Only recalibrate every recalib_every_n gates.
        st["gate_count"] += 1
        if st["gate_count"] % recalib_every_n != 0:
            return

        changed = False

        # ── σ recalibration ──
        if _hf_active:
            # HF mode: use volatility signature plateau for σ estimation.
            sig_result = _compute_sigma_signature(two_sided=use_two_sided)
            sigma_sym, sigma_up_val, sigma_dn_val = sig_result
            if sigma_sym is not None and sigma_sym > 0:
                _latest_plateau["sym"] = sigma_sym
                if use_two_sided:
                    s_ask = sigma_up_val if sigma_up_val and sigma_up_val > 0 else sigma_sym
                    s_bid = sigma_dn_val if sigma_dn_val and sigma_dn_val > 0 else sigma_sym
                    _latest_plateau["ask"] = s_ask
                    _latest_plateau["bid"] = s_bid
                    params["sigma_ask"] = (recalib_blend * s_ask
                                           + (1 - recalib_blend) * params["sigma_ask"])
                    params["sigma_bid"] = (recalib_blend * s_bid
                                           + (1 - recalib_blend) * params["sigma_bid"])
                else:
                    _latest_plateau["ask"] = sigma_sym
                    _latest_plateau["bid"] = sigma_sym
                    params["sigma_bid"] = (recalib_blend * sigma_sym
                                           + (1 - recalib_blend) * params["sigma_bid"])
                    params["sigma_ask"] = params["sigma_bid"]
                changed = True
        else:
            # Legacy gate-level RMS σ estimation.
            scale = _sigma_scale_factor()

            if len(mid_returns) >= 10:
                all_ret = np.array(mid_returns)
                sigma_est = float(np.sqrt(np.mean(all_ret ** 2))) * scale

                if use_two_sided:
                    arr_up = np.maximum(all_ret, 0.0)
                    arr_dn = np.abs(np.minimum(all_ret, 0.0))

                    if len(all_ret) >= 3:
                        sigma_up_est = float(np.sqrt(np.mean(arr_up ** 2))) * scale
                        if sigma_up_est == 0.0 and sigma_est > 0:
                            sigma_up_est = sigma_est
                        params["sigma_ask"] = (recalib_blend * sigma_up_est
                                               + (1 - recalib_blend) * params["sigma_ask"])
                        changed = True

                    if len(all_ret) >= 3:
                        sigma_dn_est = float(np.sqrt(np.mean(arr_dn ** 2))) * scale
                        if sigma_dn_est == 0.0 and sigma_est > 0:
                            sigma_dn_est = sigma_est
                        params["sigma_bid"] = (recalib_blend * sigma_dn_est
                                               + (1 - recalib_blend) * params["sigma_bid"])
                        changed = True
                else:
                    if sigma_est > 0:
                        params["sigma_bid"] = (recalib_blend * sigma_est
                                               + (1 - recalib_blend) * params["sigma_bid"])
                        params["sigma_ask"] = params["sigma_bid"]
                        changed = True

        # ── A, κ recalibration ──
        # Two methods available:
        #   (a) Censored Waiting Times MLE (use_censored_calib=True):
        #       Online, statistically rigorous.  Uses the raw MO events
        #       and window boundaries accumulated inside the tracker.
        #       Falls back to (b) if the censored fit returns None.
        #   (b) WLS Tail-Counting (default):
        #       Fast, incremental, susceptible to ghost starvation
        #       in deep buckets (mitigated by FIX 1 tail<1.0 filter).
        _ak_fitted = False

        if use_censored_calib:
            # Clear previous fit results so the plot shows only THIS recalib.
            market_tracker._last_censored_fit = {}
            if use_two_sided:
                fit_b = market_tracker.fit_A_kappa_censored(side="bid", max_windows=_hf_max_windows)
                fit_a = market_tracker.fit_A_kappa_censored(side="ask", max_windows=_hf_max_windows)
                if fit_b is not None:
                    params["A_bid"] = (recalib_blend * fit_b[0]
                                       + (1 - recalib_blend) * params["A_bid"])
                    params["kappa_bid"] = (recalib_blend * fit_b[1]
                                           + (1 - recalib_blend) * params["kappa_bid"])
                    _ak_fitted = True
                    changed = True
                if fit_a is not None:
                    params["A_ask"] = (recalib_blend * fit_a[0]
                                       + (1 - recalib_blend) * params["A_ask"])
                    params["kappa_ask"] = (recalib_blend * fit_a[1]
                                           + (1 - recalib_blend) * params["kappa_ask"])
                    _ak_fitted = True
                    changed = True
            else:
                fit_sym = market_tracker.fit_A_kappa_censored(side=None, max_windows=_hf_max_windows)
                if fit_sym is not None:
                    params["A_bid"] = (recalib_blend * fit_sym[0]
                                       + (1 - recalib_blend) * params["A_bid"])
                    params["kappa_bid"] = (recalib_blend * fit_sym[1]
                                           + (1 - recalib_blend) * params["kappa_bid"])
                    params["A_ask"] = params["A_bid"]
                    params["kappa_ask"] = params["kappa_bid"]
                    _ak_fitted = True
                    changed = True

        # Fallback to WLS tracker method if censored calib is off or failed.
        if not _ak_fitted and market_tracker.n_mos >= recalib_min_fills:
            if use_two_sided:
                fit_b = market_tracker.fit_A_kappa(side="bid", min_mos=recalib_min_fills)
                if fit_b is not None:
                    params["A_bid"] = (recalib_blend * fit_b[0]
                                       + (1 - recalib_blend) * params["A_bid"])
                    params["kappa_bid"] = (recalib_blend * fit_b[1]
                                           + (1 - recalib_blend) * params["kappa_bid"])
                    changed = True
                fit_a = market_tracker.fit_A_kappa(side="ask", min_mos=recalib_min_fills)
                if fit_a is not None:
                    params["A_ask"] = (recalib_blend * fit_a[0]
                                       + (1 - recalib_blend) * params["A_ask"])
                    params["kappa_ask"] = (recalib_blend * fit_a[1]
                                           + (1 - recalib_blend) * params["kappa_ask"])
                    changed = True
            else:
                fit_sym = market_tracker.fit_A_kappa(side=None, min_mos=recalib_min_fills)
                if fit_sym is not None:
                    params["A_bid"] = (recalib_blend * fit_sym[0]
                                       + (1 - recalib_blend) * params["A_bid"])
                    params["kappa_bid"] = (recalib_blend * fit_sym[1]
                                           + (1 - recalib_blend) * params["kappa_bid"])
                    params["A_ask"] = params["A_bid"]
                    params["kappa_ask"] = params["kappa_bid"]
                    changed = True

            # Decay old tracker observations so recent data dominates.
            market_tracker.decay(recalib_decay)

        if changed:
            _recompute_coefficients()
            st["recalib_count"] += 1
            _snap = dict(params)
            _snap["plateau_sym"] = _latest_plateau["sym"]
            _snap["plateau_bid"] = _latest_plateau["bid"]
            _snap["plateau_ask"] = _latest_plateau["ask"]
            param_history.append((current_time, _snap))

        # ── Plot censored intensity curve if requested ──
        if plot_recalib and _ak_fitted and use_censored_calib:
            _plot_censored_fit(market_tracker, current_time, st["recalib_count"])
            _plot_signature()
            _plot_param_evolution()
            _plot_pnl()
            _plot_levels()

    # -----------------------------------------------------------------
    #  Plotting helper: DYNAMIC censored intensity curve.
    #
    #  Uses a single persistent matplotlib Figure that is updated
    #  in-place at every recalibration — no new windows are spawned.
    #  The figure shows:
    #    - Coloured dots: empirical λ̂(δ_k) from the censored MLE
    #    - Fitted curve: A·exp(−κδ) from the WLS regression
    #
    #  The axes are cleared and redrawn each time, so the plot always
    #  reflects the LATEST fit.  If the user closes the window, a new
    #  one is automatically created on the next recalibration.
    #
    #  This visualisation is critical for verifying that:
    #    1. The exponential model is a reasonable fit to the data.
    #    2. The WLS regression isn't being pulled by outlier bins.
    #    3. The number of windows (n_win) is sufficient for stable λ̂.
    # -----------------------------------------------------------------
    # Persistent figure state for dynamic plot updates.
    # A single matplotlib Figure is kept alive and redrawn in-place at
    # each recalibration.  Works in both Jupyter (via IPython.display)
    # and standalone scripts (via plt.ion + canvas flush).
    _live_plot: Dict[str, Any] = {
        "fig": None, "axes": None, "n_sides": 0,
        "display_handle": None,   # IPython DisplayHandle for in-place update
        "in_jupyter": None,       # lazy-detected
    }

    def _is_jupyter() -> bool:
        """Detect if we are running inside a Jupyter notebook."""
        if _live_plot["in_jupyter"] is not None:
            return _live_plot["in_jupyter"]
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell is None:
                _live_plot["in_jupyter"] = False
            else:
                _live_plot["in_jupyter"] = (
                    shell.__class__.__name__ == "ZMQInteractiveShell"
                )
        except Exception:
            _live_plot["in_jupyter"] = False
        return _live_plot["in_jupyter"]

    def _plot_censored_fit(tracker, t_now: float, recalib_idx: int):
        """Dynamically update a single persistent figure with the latest
        censored-WT fit results.

        Works in two rendering paths:

        **Jupyter notebook** — uses ``IPython.display.display`` with a
        ``DisplayHandle`` so that calling ``display_handle.update(fig)``
        replaces the previous output in the *same cell output area*.

        **Standalone / terminal** — uses ``plt.ion()`` interactive mode
        with ``canvas.draw_idle()`` + ``flush_events()``.

        Parameters
        ----------
        tracker : _MarketIntensityTracker
            The tracker whose ``_last_censored_fit`` dict contains
            the most recent fit results (keyed by side label).
        t_now : float
            Current timestamp (for the plot title).
        recalib_idx : int
            Recalibration counter (for the plot title).
        """
        fits_dict = getattr(tracker, "_last_censored_fit", None)
        if not fits_dict:
            return
        import matplotlib.pyplot as plt

        # Collect sides that have valid data (λ̂ > 0 in at least 1 bin).
        valid_fits = []
        for side_key, fit in fits_dict.items():
            if fit is not None and np.any(fit["lambda_hat"] > 0):
                valid_fits.append((side_key, fit))

        if not valid_fits:
            return

        n_sides = len(valid_fits)
        jupyter = _is_jupyter()

        # ── Create or recreate figure if needed ──
        need_new_fig = (
            _live_plot["fig"] is None
            or (not jupyter and not plt.fignum_exists(_live_plot["fig"].number))
            or _live_plot["n_sides"] != n_sides
        )

        if need_new_fig:
            if _live_plot["fig"] is not None:
                try:
                    plt.close(_live_plot["fig"])
                except Exception:
                    pass
            fig, axes = plt.subplots(
                1, n_sides,
                figsize=(8 * n_sides, 4),
                squeeze=False,
            )
            _live_plot["fig"] = fig
            _live_plot["axes"] = axes
            _live_plot["n_sides"] = n_sides
            # Do NOT reset display_handle — in Jupyter we reuse the
            # same DisplayHandle across figure recreations so that
            # .update() replaces the SAME output area (no duplicates).

            if not jupyter:
                plt.ion()
        else:
            fig = _live_plot["fig"]
            axes = _live_plot["axes"]

        # Per-side colour scheme.
        _colors = {
            "bid": ("royalblue", "navy"),
            "ask": ("crimson", "darkred"),
            "combined": ("steelblue", "crimson"),
        }

        for idx, (side_key, fit) in enumerate(valid_fits):
            ax = axes[0, idx]
            ax.clear()

            delta_grid = fit["delta_grid"]
            lam_hat = fit["lambda_hat"]
            A_fit = fit["A"]
            kappa_fit = fit["kappa"]
            side_label = fit.get("side", "combined") or "combined"

            scatter_c, line_c = _colors.get(side_key, ("steelblue", "crimson"))

            mask = lam_hat > 0
            if mask.sum() < 1:
                continue

            ax.scatter(delta_grid[mask], lam_hat[mask],
                       marker="o", s=30, color=scatter_c, zorder=3,
                       label=r"$\hat{\lambda}(\delta)$ (MLE)")

            d_fine = np.linspace(delta_grid[0], delta_grid[-1], 200)
            ax.plot(d_fine, A_fit * np.exp(-kappa_fit * d_fine),
                    color=line_c, linewidth=2, zorder=2,
                    label=rf"$A e^{{-\kappa\delta}}$  A={A_fit:.3f}, $\kappa$={kappa_fit:.3f}")

            ax.set_xlabel(r"Distance from mid $\delta$ (tick units)")
            ax.set_ylabel(r"Execution intensity $\lambda(\delta)$")
            ax.set_title(f"Censored WT Fit — recalib #{recalib_idx}  "
                         f"t={t_now:.1f}  side={side_label}  "
                         f"(n_win={fit['n_windows']})")
            ax.legend(fontsize=9)
            if use_log_scale_plot:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

        # ── Render ──
        if jupyter:
            # Jupyter inline backend: render fig to a PNG Image and use a
            # persistent DisplayHandle so .update() replaces the SAME output
            # area in the cell — no duplicate plots, no clearing of other
            # output (prints, tqdm bars, etc.).
            import io
            from IPython.display import display, Image as IPyImage

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = IPyImage(data=buf.getvalue())
            buf.close()

            if _live_plot["display_handle"] is None:
                _live_plot["display_handle"] = display(img, display_id=True)
            else:
                _live_plot["display_handle"].update(img)

            plt.close(fig)
            _live_plot["fig"] = None  # recreate next time (closed above)
        else:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # -----------------------------------------------------------------
    #  Dynamic σ evolution plot.
    #
    #  Shows the time series of σ_bid and σ_ask (and A_bid, κ_bid,
    #  A_ask, κ_ask) across recalibrations.  Uses the same DisplayHandle
    #  pattern as _plot_censored_fit for Jupyter compatibility.
    # -----------------------------------------------------------------
    _live_plot_params: Dict[str, Any] = {
        "display_handle": None,
    }

    def _plot_param_evolution():
        """Dynamically update a plot showing σ (with plateau), A, κ evolution."""
        if len(param_history) < 2:
            return

        import matplotlib.pyplot as plt

        times = [t for t, _ in param_history]
        sigma_bid = [p["sigma_bid"] for _, p in param_history]
        sigma_ask = [p["sigma_ask"] for _, p in param_history]
        A_bid = [p.get("A_bid", 0) for _, p in param_history]
        A_ask = [p.get("A_ask", 0) for _, p in param_history]
        kappa_bid = [p.get("kappa_bid", 0) for _, p in param_history]
        kappa_ask = [p.get("kappa_ask", 0) for _, p in param_history]

        # Raw plateau values (may be None for legacy mode entries)
        plat_sym = [p.get("plateau_sym") for _, p in param_history]
        plat_bid = [p.get("plateau_bid") for _, p in param_history]
        plat_ask = [p.get("plateau_ask") for _, p in param_history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))

        # ── σ panel ──
        ax = axes[0]
        # EMA-blended σ (what the strategy actually uses)
        ax.plot(times, sigma_bid, color="royalblue", linewidth=1.5,
                marker=".", markersize=4, label=r"$\sigma_{bid}$ (EMA)")
        ax.plot(times, sigma_ask, color="crimson", linewidth=1.5,
                marker=".", markersize=4, label=r"$\sigma_{ask}$ (EMA)")
        # Raw plateau values (unblended — what the signature measured)
        _has_plateau = any(v is not None for v in plat_sym)
        if _has_plateau:
            # Filter to only entries with plateau data
            t_p = [t for t, v in zip(times, plat_bid) if v is not None]
            p_bid = [v for v in plat_bid if v is not None]
            p_ask = [v for v in plat_ask if v is not None]
            p_sym = [v for v in plat_sym if v is not None]
            if t_p:
                ax.plot(t_p, p_bid, color="royalblue", linewidth=1.0,
                        linestyle="--", alpha=0.5, marker="x", markersize=5,
                        label=r"$\sigma_{bid}^{plat}$ (raw)")
                ax.plot(t_p, p_ask, color="crimson", linewidth=1.0,
                        linestyle="--", alpha=0.5, marker="x", markersize=5,
                        label=r"$\sigma_{ask}^{plat}$ (raw)")
                ax.plot(t_p, p_sym, color="gray", linewidth=1.0,
                        linestyle=":", alpha=0.6, marker="+", markersize=5,
                        label=r"$\sigma_{sym}^{plat}$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\sigma$ (ticks/$\sqrt{s}$)")
        ax.set_title("Volatility Evolution")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # ── A panel ──
        ax = axes[1]
        ax.plot(times, A_bid, color="royalblue", linewidth=1.5,
                marker=".", markersize=4, label=r"$A_{bid}$")
        ax.plot(times, A_ask, color="crimson", linewidth=1.5,
                marker=".", markersize=4, label=r"$A_{ask}$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$A$ (intensity intercept)")
        ax.set_title("Intensity A Evolution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── κ panel ──
        ax = axes[2]
        ax.plot(times, kappa_bid, color="royalblue", linewidth=1.5,
                marker=".", markersize=4, label=r"$\kappa_{bid}$")
        ax.plot(times, kappa_ask, color="crimson", linewidth=1.5,
                marker=".", markersize=4, label=r"$\kappa_{ask}$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\kappa$ (intensity decay)")
        ax.set_title(r"Intensity $\kappa$ Evolution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        jupyter = _is_jupyter()
        if jupyter:
            import io
            from IPython.display import display, Image as IPyImage

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = IPyImage(data=buf.getvalue())
            buf.close()

            if _live_plot_params["display_handle"] is None:
                _live_plot_params["display_handle"] = display(img, display_id=True)
            else:
                _live_plot_params["display_handle"].update(img)
            plt.close(fig)
        else:
            if not hasattr(_plot_param_evolution, "_ion_set"):
                plt.ion()
                _plot_param_evolution._ion_set = True
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # -----------------------------------------------------------------
    #  Dynamic volatility signature plot.
    #
    #  Shows σ(τ) curves from the latest _compute_sigma_signature call.
    #  Updated in-place like the censored fit plot — always shows the
    #  LATEST signature used to compute the plateau.
    # -----------------------------------------------------------------
    _live_plot_sig: Dict[str, Any] = {
        "display_handle": None,
    }

    def _plot_signature():
        """Dynamically update the volatility signature plot."""
        taus = _last_signature.get("taus_sec")
        sig_sym = _last_signature.get("sig_sym")
        if taus is None or sig_sym is None:
            return

        import matplotlib.pyplot as plt

        has_sides = _last_signature.get("sig_up") is not None
        n_panels = 2 if has_sides else 1
        fig, axes_arr = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4))
        if n_panels == 1:
            axes_arr = [axes_arr]

        # Plateau window bounds (for shading)
        plat_lo = _last_signature.get("plat_lo_sec")
        plat_hi = _last_signature.get("plat_hi_sec")

        # ── Symmetric signature ──
        ax = axes_arr[0]
        ax.plot(taus, sig_sym, color="black", linewidth=1.5,
                marker="o", markersize=4, label=r"$\sigma_{sym}(\tau)$")
        plat = _last_signature.get("plateau_sym")
        if plat is not None:
            ax.axhline(plat, color="black", linestyle="--", alpha=0.6,
                       label=f"plateau = {plat:.4f}")
        # Shade the auto-detected plateau window
        if plat_lo is not None and plat_hi is not None:
            ax.axvspan(plat_lo, plat_hi, color="gold", alpha=0.15,
                       label=f"window [{plat_lo:.0f}s, {plat_hi:.0f}s]")
        if has_sides:
            sig_up = _last_signature["sig_up"]
            sig_dn = _last_signature["sig_dn"]
            ax.plot(taus, sig_up, color="crimson", linewidth=1.0,
                    marker="^", markersize=3, alpha=0.7,
                    label=r"$\sigma_{ask}(\tau)$")
            ax.plot(taus, sig_dn, color="royalblue", linewidth=1.0,
                    marker="v", markersize=3, alpha=0.7,
                    label=r"$\sigma_{bid}(\tau)$")
        ax.set_xlabel(r"$\tau$ (seconds)")
        ax.set_ylabel(r"$\sigma(\tau)$ (ticks/$\sqrt{s}$)")
        ax.set_title("Volatility Signature (latest)")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── Bid / Ask separate panel ──
        if has_sides:
            ax = axes_arr[1]
            sig_up = _last_signature["sig_up"]
            sig_dn = _last_signature["sig_dn"]
            ax.plot(taus, sig_up, color="crimson", linewidth=1.5,
                    marker="^", markersize=4, label=r"$\sigma_{ask}(\tau)$")
            ax.plot(taus, sig_dn, color="royalblue", linewidth=1.5,
                    marker="v", markersize=4, label=r"$\sigma_{bid}(\tau)$")
            plat_up = _last_signature.get("plateau_up")
            plat_dn = _last_signature.get("plateau_dn")
            if plat_up is not None:
                ax.axhline(plat_up, color="crimson", linestyle="--", alpha=0.6,
                           label=f"ask plateau = {plat_up:.4f}")
            if plat_dn is not None:
                ax.axhline(plat_dn, color="royalblue", linestyle="--", alpha=0.6,
                           label=f"bid plateau = {plat_dn:.4f}")
            if plat_lo is not None and plat_hi is not None:
                ax.axvspan(plat_lo, plat_hi, color="gold", alpha=0.15)
            ax.set_xlabel(r"$\tau$ (seconds)")
            ax.set_ylabel(r"$\sigma(\tau)$ (ticks/$\sqrt{s}$)")
            ax.set_title("Bid / Ask Signatures (latest)")
            ax.set_xscale("log")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

        jupyter = _is_jupyter()
        if jupyter:
            import io
            from IPython.display import display, Image as IPyImage

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = IPyImage(data=buf.getvalue())
            buf.close()

            if _live_plot_sig["display_handle"] is None:
                _live_plot_sig["display_handle"] = display(img, display_id=True)
            else:
                _live_plot_sig["display_handle"].update(img)
            plt.close(fig)
        else:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # -----------------------------------------------------------------
    #  PnL dynamic plot — cumulative wealth (cash + inventory × mid)
    # -----------------------------------------------------------------
    _live_plot_pnl: Dict[str, Any] = {"display_handle": None}

    def _plot_pnl():
        """Dynamically update the cumulative PnL plot."""
        if len(_pnl_history) < 2:
            return

        import matplotlib.pyplot as plt

        times = [h[0] for h in _pnl_history]
        wealth = [h[4] for h in _pnl_history]
        cash = [h[1] for h in _pnl_history]
        inv = [h[2] for h in _pnl_history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

        # Panel 1: Wealth (PnL)
        ax = axes[0]
        w0 = wealth[0]
        pnl = [w - w0 for w in wealth]
        ax.plot(times, pnl, color="forestgreen", linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.fill_between(times, pnl, 0,
                         where=[p >= 0 for p in pnl],
                         color="forestgreen", alpha=0.15, interpolate=True)
        ax.fill_between(times, pnl, 0,
                         where=[p < 0 for p in pnl],
                         color="crimson", alpha=0.15, interpolate=True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PnL (ticks)")
        ax.set_title("Cumulative PnL (mark-to-market)")
        ax.grid(True, alpha=0.3)

        # Panel 2: Inventory
        ax = axes[1]
        ax.plot(times, inv, color="darkorange", linewidth=1.0)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Inventory (units)")
        ax.set_title("Inventory")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        jupyter = _is_jupyter()
        if jupyter:
            import io
            from IPython.display import display, Image as IPyImage

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = IPyImage(data=buf.getvalue())
            buf.close()

            if _live_plot_pnl["display_handle"] is None:
                _live_plot_pnl["display_handle"] = display(img, display_id=True)
            else:
                _live_plot_pnl["display_handle"].update(img)
            plt.close(fig)
        else:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # -----------------------------------------------------------------
    #  Order levels dynamic plot — bid/ask offset from BBO
    # -----------------------------------------------------------------
    _live_plot_levels: Dict[str, Any] = {"display_handle": None}

    def _plot_levels():
        """Dynamically update the MM order-level plot relative to BBO."""
        if len(_level_history) < 2:
            return

        import matplotlib.pyplot as plt

        times = [h[0] for h in _level_history]
        bid_off = [h[1] for h in _level_history]
        ask_off = [h[2] for h in _level_history]

        fig, ax = plt.subplots(1, 1, figsize=(14, 3.5))

        # Bid offsets (negative = deeper in book)
        t_bid = [t for t, b, _ in zip(times, bid_off, ask_off) if b is not None]
        v_bid = [b for b in bid_off if b is not None]
        # Ask offsets (positive = deeper in book)
        t_ask = [t for t, _, a in zip(times, bid_off, ask_off) if a is not None]
        v_ask = [a for a in ask_off if a is not None]

        if t_bid:
            ax.scatter(t_bid, v_bid, color="royalblue", s=4, alpha=0.5, label="Bid offset")
        if t_ask:
            ax.scatter(t_ask, v_ask, color="crimson", s=4, alpha=0.5, label="Ask offset")

        ax.axhline(0, color="gray", linewidth=1.5, linestyle="-", alpha=0.6,
                   label="BBO (level 0)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Offset from BBO (ticks)")
        ax.set_title("MM Order Levels (relative to BBO)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        jupyter = _is_jupyter()
        if jupyter:
            import io
            from IPython.display import display, Image as IPyImage

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = IPyImage(data=buf.getvalue())
            buf.close()

            if _live_plot_levels["display_handle"] is None:
                _live_plot_levels["display_handle"] = display(img, display_id=True)
            else:
                _live_plot_levels["display_handle"].update(img)
            plt.close(fig)
        else:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # -----------------------------------------------------------------
    #  Progress bar: lightweight tqdm wrapper updated at each policy call.
    #  Only active when ``n_total_steps`` is provided.  The bar is created
    #  lazily on the first policy invocation so it starts at 0 %.
    # -----------------------------------------------------------------
    _pbar: Dict[str, Any] = {"bar": None, "step": 0}

    def _update_progress():
        """Advance the tqdm progress bar by one step."""
        if n_total_steps is None or n_total_steps <= 0:
            return
        if _pbar["bar"] is None:
            from tqdm.auto import tqdm
            _pbar["bar"] = tqdm(
                total=n_total_steps,
                desc="GLFT Backtest",
                unit="step",
                miniters=max(1, n_total_steps // 200),  # refresh ~200 times
                smoothing=0.05,
            )
        _pbar["step"] += 1
        _pbar["bar"].update(1)
        if _pbar["step"] >= n_total_steps:
            _pbar["bar"].close()

    # =====================================================================
    #
    #  SECTION 5: MAIN POLICY FUNCTION
    #
    #  This is the "brain" of the market maker.  Called at every simulation
    #  tick (every LOBSTER event), it decides whether to place, modify,
    #  cancel, or hold orders.
    #
    #  The decision flows through a strict priority hierarchy (gates):
    #
    #    ┌─────────────────────────────────────────────────────────┐
    #    │  A) Day-boundary detection + Grid-shift adjustment     │
    #    │  B) Throttle counter updates                           │
    #    │  C) Read state (inventory, reference price, mode)      │
    #    │  D) Fill detection + Market observation (tape reading)  │
    #    │  WARMUP) Observation-only period (if active)           │
    #    │  E) Compute optimal bid/ask quotes from GLFT model     │
    #    │  F) Determine missing / forbidden order legs           │
    #    │  GATE 1) Inventory limit breach → immediate action     │
    #    │  GATE 2) Fill detected → immediate replenishment       │
    #    │  GATE 3) Hard throttle → suppress if within interval   │
    #    │  GATE 4) Normal repricing → update if prices changed   │
    #    └─────────────────────────────────────────────────────────┘
    #
    # =====================================================================
    def policy(state: Dict[str, Any], mm=None) -> Tuple:

        # =================================================================
        # A) TIME MANAGEMENT, DAY-BOUNDARY DETECTION & GRID-SHIFT
        # =================================================================
        current_time = _safe_float(state.get("time", 0.0))

        # ── Progress bar update ──
        _update_progress()

        # ── Day-boundary detection ──
        # Concatenated multi-day LOBSTER data causes the timestamp to jump
        # backwards at day boundaries (e.g., 55799s → 36000s).  A jump of
        # more than 5 minutes backwards is treated as a new trading day.
        #
        # On detection:
        #   0. (Optional) Liquidate open inventory at market prices
        #   1. All calibration state is reset (_reset_for_new_day)
        #   2. All existing orders are cancelled (mm.cancel_all)
        #   3. The policy re-enters warmup to recalibrate from scratch
        #   4. Returns ("hold",) for this tick
        if last_seen_time[0] is not None:
            time_jump = current_time - last_seen_time[0]
            if time_jump < -300:
                # ── Terminal liquidation (optional) ──
                # If liquidate_terminal is enabled, flatten all open
                # inventory via aggressive market orders BEFORE resetting
                # state.  This must happen first, while the previous day's
                # order book is still populated and the MM object holds the
                # correct position.  The cross_mo method handles sweeping
                # multiple price levels when qty > 1.
                if liquidate_terminal and mm is not None:
                    inv_now = _safe_int(state.get("inventory", 0), 0)
                    if inv_now > 0:
                        px, qty = mm.cross_mo(-1, qty=abs(inv_now))
                        print(f"[GLFT] Day-boundary liquidation: SOLD {qty} "
                              f"units @ px={px} to flatten long inventory")
                    elif inv_now < 0:
                        px, qty = mm.cross_mo(+1, qty=abs(inv_now))
                        print(f"[GLFT] Day-boundary liquidation: BOUGHT {qty} "
                              f"units @ px={px} to flatten short inventory")

                _reset_for_new_day(current_time)
                if mm is not None:
                    mm.cancel_all()
                last_seen_time[0] = current_time
                return ("hold",)
        last_seen_time[0] = current_time

        # ── Grid-shift detection ──
        # The LOB simulator uses a relative price grid centered near the mid.
        # When the mid drifts far from the grid center, the engine shifts the
        # entire grid (base_price_idx changes).  We must adjust our cached
        # order prices and the mid anchor to match the new coordinates.
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
                    base_shift = cur_base - prev_base_idx[0]
                    if base_shift != 0:
                        shift_happened = True
                        if st["last_bid_px"] is not None:
                            st["last_bid_px"] -= base_shift
                        if st["last_ask_px"] is not None:
                            st["last_ask_px"] -= base_shift
                        # Adjust the mid anchor so the next gate return is
                        # not corrupted by the grid shift (which would inject
                        # a phantom price jump into the sigma estimate).
                        if mid_at_last_gate[0] is not None:
                            mid_at_last_gate[0] -= base_shift
                        # Also adjust the HF mid buffer so the volatility
                        # signature is not corrupted by the grid shift.
                        if len(_hf_mids) > 0:
                            _hf_mids_shifted = deque(
                                ((t, m - base_shift) for t, m in _hf_mids),
                                maxlen=_hf_mids.maxlen,
                            )
                            _hf_mids.clear()
                            _hf_mids.extend(_hf_mids_shifted)
                        prev_base_idx[0] = cur_base

        # =================================================================
        # B) THROTTLE COUNTER UPDATES
        #
        # These counters track how many events / TOB changes / seconds
        # have elapsed since the last gate firing.  Gate 3 uses them to
        # decide whether the bot is allowed to act.
        # =================================================================
        if use_event_update:
            st["event_steps"] += 1
        if use_tob_update:
            key = _get_env_tob(state)
            if shift_happened or st["last_env_tob_key"] is None:
                st["last_env_tob_key"] = key
            else:
                _update_tob_clock(key)

        # =================================================================
        # C) READ MARKET STATE
        # =================================================================
        I_t = _safe_int(state.get("inventory", 0), 0)
        has_bid = bool(state.get("has_bid", False))
        has_ask = bool(state.get("has_ask", False))

        # Fair-value reference price (midprice or microprice).
        ref_price = _get_ref_price(state, mm)
        if not math.isfinite(ref_price):
            return ("hold",)

        # Inventory gating: determine desired quoting mode.
        if I_t >= inv_limit:
            desired_mode = "ask_only"       # long inventory → only sell
        elif I_t <= -inv_limit:
            desired_mode = "bid_only"       # short inventory → only buy
        else:
            desired_mode = "two_sided"

        mode_changed = (st["last_mode"] != desired_mode)

        # =================================================================
        # D) FILL DETECTION
        # =================================================================
        new_fill, bid_fills, ask_fills = _consume_fill(I_t, has_bid, has_ask)
        if new_fill:
            st["n_fills"] += (bid_fills + ask_fills)
            st["n_fills_bid"] += bid_fills
            st["n_fills_ask"] += ask_fills

        # =================================================================
        # D2) MARKET OBSERVATION — Feed public tape MOs to intensity tracker
        #
        # This is the "tape reading" engine.  At every event, we check if
        # it was a market order execution.  If so, we measure how deep it
        # penetrated the book and feed that observation to the tracker.
        # This data is later used by the WLS regression to estimate A and κ.
        # =================================================================
        if mm is not None:
            snap = getattr(mm, "_last_env_snapshot", None)
            if snap is not None:
                ev_type = snap.get("raw_type", -1)
                if ev_type == 1:  # LOBSTER type 1 = market order execution
                    ev_dir = int(snap.get("raw_direction", 0))
                    ev_px = _safe_float(snap.get("raw_price", None), None)
                    ev_mid = _safe_float(state.get("mid", None), None)
                    if ev_px is not None and ev_mid is not None:
                        depth = abs(ev_px - ev_mid)
                        market_tracker.observe(True, depth, ev_dir, current_time, mid=ev_mid)
                    else:
                        market_tracker.observe(True, None, ev_dir, current_time, mid=ev_mid)
                else:
                    market_tracker.observe(False, None, 0, current_time)

        # =================================================================
        # D3) HIGH-FREQUENCY CALIBRATION SAMPLER
        #
        # When time_interval_calib is set, sample mid-price at high
        # frequency (e.g., every 1s) for the volatility signature and
        # censored MLE — independently of the strategy throttle gate.
        # =================================================================
        if time_interval_calib is not None:
            _mid_for_hf = _safe_float(state.get("mid", None), float("nan"))
            if math.isfinite(_mid_for_hf):
                _hf_sample(current_time, _mid_for_hf)

        # =================================================================
        # D4) PnL TRACKING — Record wealth snapshot (throttled to ~1s)
        # =================================================================
        if plot_recalib:
            _pnl_dt = current_time - (_pnl_history[-1][0] if _pnl_history else -999.0)
            if _pnl_dt >= 1.0 or _pnl_dt < -60.0:
                _pnl_cash = _safe_float(state.get("cash", 0.0))
                _pnl_inv = float(I_t)
                _pnl_mid = _safe_float(state.get("mid", 0.0))
                _pnl_wealth = _pnl_cash + _pnl_inv * _pnl_mid
                _pnl_history.append((current_time, _pnl_cash, _pnl_inv, _pnl_mid, _pnl_wealth))

        # =================================================================
        # WARMUP PHASE — Observation-only period before trading
        #
        # During warmup, the policy does NOT place any orders.  It only
        # collects market data (mid-price returns, MO arrival depths)
        # to calibrate the initial parameters.
        #
        # Warmup completes when:
        #   1. warmup_gate_count >= warmup_gates, AND
        #   2. Sufficient data for at least one successful calibration
        #
        # After warmup, the first calibration runs with blend=1.0
        # (full replacement, not EMA) and trading begins immediately.
        # =================================================================
        if st["warming_up"]:
            # Seed the time clock on the very first event.
            if use_time_update and st["last_update_time"] < 0:
                st["last_update_time"] = current_time

            # Check if a throttle gate fires (same logic as Gate 3).
            warmup_gate_fired = False
            if use_event_update and st["event_steps"] >= threshold_events:
                warmup_gate_fired = True
            if use_time_update:
                dt_warmup = current_time - st["last_update_time"]
                if dt_warmup < 0:
                    # Time wrap within warmup: reset clock.
                    st["last_update_time"] = current_time
                elif dt_warmup >= min_time_interval:
                    warmup_gate_fired = True
            if use_tob_update and st["moves_tob"] >= threshold_tob:
                warmup_gate_fired = True
            if not (use_event_update or use_time_update or use_tob_update):
                warmup_gate_fired = True

            if warmup_gate_fired:
                _reset_all_clocks(state, current_time)
                st["warmup_gate_count"] += 1

                # Record mid return and gate elapsed time (strict sync).
                mid_now = _safe_float(state.get("mid", None), float("nan"))
                _hf_active = (time_interval_calib is not None and time_interval_calib > 0)

                # When HF mode is active, _hf_sample handles mark_window
                # and mid recording at its own cadence — skip gate-level.
                if not _hf_active:
                    market_tracker.mark_window(current_time, mid_now=mid_now if math.isfinite(mid_now) else None)
                    if math.isfinite(mid_now):
                        if mid_at_last_gate[0] is not None and time_at_last_gate[0] is not None:
                            dt = current_time - time_at_last_gate[0]
                            if dt > 0:
                                ret = mid_now - mid_at_last_gate[0]
                                mid_returns.append(ret)
                                gate_dts.append(dt)
                        mid_at_last_gate[0] = mid_now
                        time_at_last_gate[0] = current_time

                # ── Check if warmup is complete ──
                if st["warmup_gate_count"] >= warmup_gates:
                    # FIX 5 — Strict Warmup Synchronization
                    # Both calibration engines (σ AND A/κ) must succeed
                    # before the bot exits warmup.  Partial calibration
                    # (e.g. σ ok but A/κ failed) would let the bot trade
                    # with stale/default parameters, producing nonsensical
                    # quotes.  We track each independently and only release
                    # the bot when both flags are True.
                    calib_sigma = False
                    calib_ak = False

                    # σ calibration: signature (HF) or gate-level RMS.
                    if _hf_active:
                        sig_result = _compute_sigma_signature(two_sided=use_two_sided)
                        sigma_sym, sigma_up_val, sigma_dn_val = sig_result
                        if sigma_sym is not None and sigma_sym > 0:
                            _latest_plateau["sym"] = sigma_sym
                            if use_two_sided:
                                s_ask = sigma_up_val if sigma_up_val and sigma_up_val > 0 else sigma_sym
                                s_bid = sigma_dn_val if sigma_dn_val and sigma_dn_val > 0 else sigma_sym
                                _latest_plateau["ask"] = s_ask
                                _latest_plateau["bid"] = s_bid
                                params["sigma_ask"] = s_ask
                                params["sigma_bid"] = s_bid
                            else:
                                _latest_plateau["ask"] = sigma_sym
                                _latest_plateau["bid"] = sigma_sym
                                params["sigma_bid"] = sigma_sym
                                params["sigma_ask"] = sigma_sym
                            calib_sigma = True
                    else:
                        scale = _sigma_scale_factor()
                        if len(mid_returns) >= 3:
                            all_ret = np.array(mid_returns)
                            sigma_est = float(np.sqrt(np.mean(all_ret ** 2))) * scale
                            if sigma_est > 0:
                                params["sigma_bid"] = sigma_est
                                params["sigma_ask"] = sigma_est
                                calib_sigma = True
                                if use_two_sided:
                                    arr_up = np.maximum(all_ret, 0.0)
                                    arr_dn = np.abs(np.minimum(all_ret, 0.0))
                                    params["sigma_ask"] = float(np.sqrt(np.mean(arr_up ** 2))) * scale
                                    params["sigma_bid"] = float(np.sqrt(np.mean(arr_dn ** 2))) * scale

                    # A, κ calibration — try censored method first, fallback to tracker.
                    if use_censored_calib:
                        market_tracker._last_censored_fit = {}
                        if use_two_sided:
                            fit_b = market_tracker.fit_A_kappa_censored(side="bid")
                            fit_a = market_tracker.fit_A_kappa_censored(side="ask")
                            if fit_b is not None and fit_a is not None:
                                params["A_bid"], params["kappa_bid"] = fit_b
                                params["A_ask"], params["kappa_ask"] = fit_a
                                calib_ak = True
                        else:
                            fit_sym = market_tracker.fit_A_kappa_censored(side=None, max_windows=_hf_max_windows)
                            if fit_sym is not None:
                                params["A_bid"] = params["A_ask"] = fit_sym[0]
                                params["kappa_bid"] = params["kappa_ask"] = fit_sym[1]
                                calib_ak = True

                    # Fallback: tracker-based WLS regression.
                    if not calib_ak:
                        if use_two_sided:
                            fit_b = market_tracker.fit_A_kappa(side="bid")
                            fit_a = market_tracker.fit_A_kappa(side="ask")
                            if fit_b is not None and fit_a is not None:
                                params["A_bid"], params["kappa_bid"] = fit_b
                                params["A_ask"], params["kappa_ask"] = fit_a
                                calib_ak = True
                        else:
                            fit_sym = market_tracker.fit_A_kappa(side=None)
                            if fit_sym is not None:
                                params["A_bid"] = params["A_ask"] = fit_sym[0]
                                params["kappa_bid"] = params["kappa_ask"] = fit_sym[1]
                                calib_ak = True

                    if calib_sigma and calib_ak:
                        _recompute_coefficients()
                        st["warming_up"] = False
                        st["recalib_count"] += 1
                        _snap = dict(params)
                        _snap["plateau_sym"] = _latest_plateau["sym"]
                        _snap["plateau_bid"] = _latest_plateau["bid"]
                        _snap["plateau_ask"] = _latest_plateau["ask"]
                        param_history.append((current_time, _snap))
                        market_tracker.decay(recalib_decay)
                        print(f"[GLFT Enhanced] Warmup complete at t={current_time:.2f} "
                              f"after {st['warmup_gate_count']} gates.")
                        print(f"  Calibrated: A_b={params['A_bid']:.4f} "
                              f"k_b={params['kappa_bid']:.4f} "
                              f"s_b={params['sigma_bid']:.6f} | "
                              f"A_a={params['A_ask']:.4f} "
                              f"k_a={params['kappa_ask']:.4f} "
                              f"s_a={params['sigma_ask']:.6f}")
                        if plot_recalib and use_censored_calib:
                            _plot_censored_fit(market_tracker, current_time, st["recalib_count"])
                            _plot_signature()
                            _plot_param_evolution()
                    # else: not enough data yet, keep warming up.

            return ("hold",)

        # =================================================================
        # E) COMPUTE OPTIMAL QUOTES (GLFT Model)
        #
        # The GLFT optimal quotes are:
        #   q_bid = S − (half_spread + skew · inventory)
        #   q_ask = S + (half_spread − skew · inventory)
        #
        # where S is the reference price (mid or microprice), and
        # half_spread / skew come from the GLFT coefficient computation
        # scaled by the user's adjustment factors.
        # =================================================================
        hs_bid = coeffs["hs_bid"] * adj_spread
        sk_bid = coeffs["sk_bid"] * adj_skew
        hs_ask = coeffs["hs_ask"] * adj_spread
        sk_ask = coeffs["sk_ask"] * adj_skew

        q_bid = ref_price - (hs_bid + sk_bid * I_t)
        q_ask = ref_price + (hs_ask - sk_ask * I_t)

        bid_px = _bid_to_idx(q_bid)
        ask_px = _ask_to_idx(q_ask)

        # Enforce minimum spread of 1 tick.
        if ask_px <= bid_px:
            ask_px = bid_px + 1

        # Record order levels relative to BBO (for dynamic plot, throttled ~1s)
        if plot_recalib:
            _lv_dt = current_time - (_level_history[-1][0] if _level_history else -999.0)
            if _lv_dt >= 1.0 or _lv_dt < -60.0:
                _bb_env = _safe_int(state.get("best_bid_env", state.get("best_bid", None)), None)
                _ba_env = _safe_int(state.get("best_ask_env", state.get("best_ask", None)), None)
                _bid_off = (bid_px - _bb_env) if _bb_env is not None else None
                _ask_off = (ask_px - _ba_env) if _ba_env is not None else None
                _level_history.append((current_time, _bid_off, _ask_off))

        # =================================================================
        # F) DETERMINE MISSING / FORBIDDEN ORDER LEGS
        #
        # missing_bid/ask: we SHOULD have an order on this side but don't.
        # forbidden_bid/ask: we HAVE an order on a side we shouldn't
        #   (e.g., bid order when we're at max long inventory).
        # =================================================================
        missing_bid = (not has_bid) and (desired_mode in ("two_sided", "bid_only"))
        missing_ask = (not has_ask) and (desired_mode in ("two_sided", "ask_only"))
        forbidden_bid = has_bid and (desired_mode == "ask_only")
        forbidden_ask = has_ask and (desired_mode == "bid_only")

        # =================================================================
        # GATE 1: CRITICAL — Inventory limit breach / mode change
        #
        # Highest priority.  If the inventory limit was just breached or
        # a forbidden leg exists, act IMMEDIATELY regardless of throttle.
        # Suppressed in MDP mode (strict gate discipline).
        # =================================================================
        if (mode_changed or forbidden_bid or forbidden_ask) and not use_mdp:
            st["last_mode"] = desired_mode
            _reset_all_clocks(state, current_time)
            _maybe_recalibrate(current_time, _safe_float(state.get("mid", 0.0), float("nan")))

            if desired_mode == "two_sided":
                st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                return ("place_bid_ask", bid_px, ask_px)
            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = bid_px, None
                return ("cancel_all_then_place", +1, bid_px)
            else:
                st["last_bid_px"], st["last_ask_px"] = None, ask_px
                return ("cancel_all_then_place", -1, ask_px)

        # =================================================================
        # GATE 2: URGENT — Fill detected
        #
        # When an order is filled, immediately replenish to maintain
        # queue priority and avoid being naked (unquoted) on one side.
        # Suppressed in MDP mode (fills are handled at the next gate).
        # =================================================================
        if new_fill and not use_mdp:
            _reset_all_clocks(state, current_time)
            _maybe_recalibrate(current_time, _safe_float(state.get("mid", 0.0), float("nan")))

            bid_match = (st["last_bid_px"] == bid_px)
            ask_match = (st["last_ask_px"] == ask_px)

            if desired_mode == "two_sided":
                if (not bid_match) and (not ask_match):
                    st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                    return ("place_bid_ask", bid_px, ask_px)

                if bid_fills > 0 and ask_fills == 0 and ask_match:
                    st["last_bid_px"] = bid_px
                    return ("cancel_bid_then_place", bid_px)

                if ask_fills > 0 and bid_fills == 0 and bid_match:
                    st["last_ask_px"] = ask_px
                    return ("cancel_ask_then_place", ask_px)

                # Fallback: both sides need updating.
                st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                return ("place_bid_ask", bid_px, ask_px)

            elif desired_mode == "bid_only":
                st["last_bid_px"], st["last_ask_px"] = bid_px, None
                return ("cancel_all_then_place", +1, bid_px)
            else:
                st["last_bid_px"], st["last_ask_px"] = None, ask_px
                return ("cancel_all_then_place", -1, ask_px)

        # =================================================================
        # GATE 3: HARD THROTTLE — Rate-limiting
        #
        # Prevents excessive order modifications that could trigger
        # exchange rate limits or amplify adverse selection costs.
        #
        # The throttle fires if ANY active throttle condition is not met:
        #   - Event count < threshold
        #   - Time elapsed < min_time_interval
        #   - TOB changes < threshold
        #
        # Time-wrap protection: if current_time < last_update_time
        # (indicating a data loop), reset the clock instead of getting
        # permanently stuck in throttled state.
        #
        # MDP mode: throttle is absolute — no exceptions.  The MM waits
        # for the gate to open even if completely naked (both sides missing).
        # =================================================================
        is_throttled = False
        if use_event_update and (st["event_steps"] < threshold_events):
            is_throttled = True
        if use_time_update:
            if st["last_update_time"] < 0:
                pass    # clock not yet seeded
            else:
                dt = current_time - st["last_update_time"]
                if dt < 0:
                    # Time went backwards (data loop / day wrap).
                    # Reset clock to avoid permanent throttle lock.
                    st["last_update_time"] = current_time
                elif dt < min_time_interval:
                    is_throttled = True
        if use_tob_update and (st["moves_tob"] < threshold_tob):
            is_throttled = True

        if is_throttled:
            # In MDP mode, the throttle is absolute — no exceptions.
            # Even if both sides are missing (completely naked), the MM
            # waits for the throttle to expire before placing new orders.
            # This guarantees strict Markov Decision Process discipline:
            # actions happen ONLY at gate boundaries, never in between.
            if use_mdp:
                return ("hold",)
            # Non-MDP: allow immediate placement only when a leg is missing
            # (fill-replenishment was already handled by Gate 2 above).
            if not (missing_bid or missing_ask):
                return ("hold",)

        # =================================================================
        # GATE 4: NORMAL REPRICING
        #
        # The standard path.  The throttle gate has opened, so we check
        # whether the optimal quote prices have changed since our last
        # placement.  If so, cancel and replace.  If not, hold.
        #
        # Also triggers recalibration (recording the gate observation and
        # potentially re-estimating parameters).
        # =================================================================
        st["last_mode"] = desired_mode
        _maybe_recalibrate(current_time, _safe_float(state.get("mid", 0.0), float("nan")))

        if desired_mode == "two_sided":
            need_bid = missing_bid or (st["last_bid_px"] != bid_px)
            need_ask = missing_ask or (st["last_ask_px"] != ask_px)

            if not need_bid and not need_ask:
                _reset_all_clocks(state, current_time)
                return ("hold",)

            _reset_all_clocks(state, current_time)

            if need_bid and need_ask:
                st["last_bid_px"], st["last_ask_px"] = bid_px, ask_px
                return ("place_bid_ask", bid_px, ask_px)
            if need_bid:
                st["last_bid_px"] = bid_px
                return ("cancel_bid_then_place", bid_px)
            # need_ask
            st["last_ask_px"] = ask_px
            return ("cancel_ask_then_place", ask_px)

        elif desired_mode == "bid_only":
            need_bid = missing_bid or (st["last_bid_px"] != bid_px)
            # FIX 3 — Forbidden Leg Leak in MDP Mode
            # With use_mdp=True, Gate 1 (which normally cancels forbidden
            # orders immediately) is suppressed.  If we are in bid_only
            # mode and the bid price hasn't changed, we must NOT return
            # hold while a forbidden ASK order is still alive on the
            # exchange — it would keep selling and push inventory to −∞.
            # Only hold if the bid is correct AND no forbidden ask exists.
            if not need_bid and not forbidden_ask:
                _reset_all_clocks(state, current_time)
                return ("hold",)
            _reset_all_clocks(state, current_time)
            st["last_bid_px"], st["last_ask_px"] = bid_px, None
            return ("cancel_all_then_place", +1, bid_px)

        else:  # ask_only
            need_ask = missing_ask or (st["last_ask_px"] != ask_px)
            # FIX 3 — Mirror of above: in ask_only mode, a forbidden BID
            # left alive would keep buying and push inventory to +∞.
            if not need_ask and not forbidden_bid:
                _reset_all_clocks(state, current_time)
                return ("hold",)
            _reset_all_clocks(state, current_time)
            st["last_bid_px"], st["last_ask_px"] = None, ask_px
            return ("cancel_all_then_place", -1, ask_px)

    # =====================================================================
    # SECTION 6: DEBUG / STATISTICS ACCESSORS
    # =====================================================================
    def get_n_fills() -> int:
        return int(st["n_fills"])

    def get_params() -> Dict[str, float]:
        """Return current parameter values (useful after recalibration)."""
        return dict(params)

    def get_coefficients() -> Dict[str, float]:
        """Return current GLFT coefficients."""
        return dict(coeffs)

    def print_stats(prefix: str = "GLFT-E") -> None:
        print(
            f"{prefix}: fills={st['n_fills']} "
            f"(bid={st['n_fills_bid']}, ask={st['n_fills_ask']}) | "
            f"mode={st['last_mode']} | recalibs={st['recalib_count']} | "
            f"TOB={st['moves_tob']}/{threshold_tob} | "
            f"Evt={st['event_steps']}/{threshold_events} | "
            f"Time={st['last_update_time']:.2f}"
        )
        if update_params:
            print(
                f"  Params: A_b={params['A_bid']:.3f} k_b={params['kappa_bid']:.3f} "
                f"s_b={params['sigma_bid']:.5f} | "
                f"A_a={params['A_ask']:.3f} k_a={params['kappa_ask']:.3f} "
                f"s_a={params['sigma_ask']:.5f}"
            )
            print(
                f"  Coeffs: HS_b={coeffs['hs_bid']:.4f} SK_b={coeffs['sk_bid']:.4f} | "
                f"HS_a={coeffs['hs_ask']:.4f} SK_a={coeffs['sk_ask']:.4f}"
            )

    # Attach accessors to the policy callable for external inspection.
    policy.get_n_fills = get_n_fills
    policy.get_params = get_params
    policy.get_coefficients = get_coefficients
    policy.print_stats = print_stats
    policy._debug_state = st
    policy._params = params
    policy._coeffs = coeffs
    policy._market_tracker = market_tracker
    policy._mid_returns = mid_returns
    policy._param_history = param_history

    return policy
