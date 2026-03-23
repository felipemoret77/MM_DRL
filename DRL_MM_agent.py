"""
DRL_MM_agent.py  --  Neural-network Market-Making agent

Implements the DRL agent described in:
    Gašperov & Kostanjčar (2021), "Market Making With Signals Through
    Deep Reinforcement Learning", IEEE Access, Vol. 9.

Architecture (Paper Section IV-A, Table 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The agent is a small, deterministic feed-forward neural network that maps
the current market state directly to bid/ask quote offsets:

    State  ────►  FC(3→32, ReLU)  ────►  FC(32→32, ReLU)  ────►  FC(32→2, Linear)
                                                                       │
                                                                       ▼
                                                              [raw_ask, raw_bid]
                                                                       │
                                                              × output_scale (5)
                                                                       │
                                                                  round()
                                                                       │
                                                              [ask_offset, bid_offset]
                                                                  (integer ticks)

State Space (Paper Eq. 5)
~~~~~~~~~~~~~~~~~~~~~~~~~
    S_t = [I_t, RR_t, TR_t]

    - I_t  : current inventory (signed integer, e.g. -2 to +2)
    - RR_t : SGU-1 signal (realized price range, z-score normalized)
    - TR_t : SGU-2 signal (pseudo-mid trend, z-score normalized)

All three inputs are z-score normalized using statistics from the DRL
training set ("ruler").  The inventory is normalized the same way
(mean/std from training episodes).

Action Space (Paper Eq. 6)
~~~~~~~~~~~~~~~~~~~~~~~~~~
    A_t = [Q_ask - Q^bask_t,  Q^bask_t - Q_bid]

    - A_{t,1} = ask_offset  : how many ticks ABOVE the current best ask
                               to post the ask limit order.
    - A_{t,2} = bid_offset  : how many ticks BELOW the current best bid
                               to post the bid limit order.

    Larger offsets = more conservative (wider spread, higher capture
    probability but lower fill probability).
    Smaller offsets = more aggressive (tighter spread, lower capture
    but higher fill probability).
    Negative offsets = quoting INSIDE the spread (crossing the book).

    The bid-ask spread is implicitly encoded: if ask_offset = 2 and
    bid_offset = 1 and the current spread is 1 tick, then:
        quoted_spread = current_spread + ask_offset + bid_offset
                      = 1 + 2 + 1 = 4 ticks

Output Scaling (Paper Section IV-A)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The NN output layer is LINEAR (no activation), so raw outputs can be
any real number.  To convert to meaningful tick offsets:

    1. Multiply by `output_scale` (default 5, from Paper Table 4).
       This determines the "reach" — how far from the touch the agent
       can quote without needing to learn very large weight magnitudes.

    2. Round to nearest integer (ticks are discrete).

Example:
    NN raw output:  [0.62, 0.38]
    × scale (5):    [3.10, 1.90]
    round():        [3,    2]
    → Post ask at best_ask + 3 ticks, bid at best_bid - 2 ticks.

Why Scaling Helps
~~~~~~~~~~~~~~~~~
Without scaling, the NN would need to produce outputs in the range
[0, ~10] directly.  With orthogonal initialization (gain=0.9) and
small biases (0.05), the initial outputs are close to 0.  The scaling
factor lets the NN work in a comfortable range near [-1, +1] while
still producing meaningful tick-sized offsets.  This is analogous to
"action scaling" commonly used in continuous-action RL (e.g., MuJoCo
environments where actions are in [-1, 1] and then scaled to torques).

Initialization (Paper Section IV-D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Weights use orthogonal initialization with gain=0.9 (not Xavier).
Biases are set to a small constant (0.05, not zero).

Why orthogonal?  The paper cites [28] (Saxe et al., 2014) showing
that orthogonal initialization preserves gradient norms across layers,
reducing vanishing/exploding gradient problems.  While neuroevolution
is gradient-free, orthogonal init also produces more diverse initial
policies in the zeroth generation (population), which improves the
genetic algorithm's starting coverage of the search space.

Why constant bias = 0.05 (not zero)?  A small positive bias means
that initial offsets are slightly positive (conservative quoting),
which is a safer starting point for a market-maker than zero offsets
(which would mean quoting at the touch, exposing to adverse selection).

Training Method
~~~~~~~~~~~~~~~
This agent is trained via NEUROEVOLUTION (genetic algorithm), NOT
gradient-based RL.  The weights are set externally by the GA:

    for generation in range(N_generations):
        for individual in population:
            individual.set_weights(chromosome)
            fitness = run_backtest(individual)
        population = select_and_mutate(population, fitnesses)

The agent itself has no optimizer, no loss function, no replay buffer.
It is a pure function approximator: state → action.

Inventory Constraints (Paper Algorithm 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The agent's offsets are subject to inventory limits [I_min, I_max]:
    - If I_t == I_max: only the ASK order is posted (no more buying)
    - If I_t == I_min: only the BID order is posted (no more selling)
    - Otherwise: both orders are posted

This constraint is NOT enforced inside the NN — it is applied
EXTERNALLY by the backtester/controller after the NN produces its
raw offsets.  The NN always outputs two offsets; the controller
decides whether to use both or only one.

Dependencies
~~~~~~~~~~~~
- PyTorch (torch, torch.nn)
- NumPy (for weight manipulation in neuroevolution)
"""

from __future__ import annotations

import math
import copy
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# MM AGENT — Feed-forward NN for market-making quote placement
# ═══════════════════════════════════════════════════════════════════════════

class MMAgent(nn.Module):
    """
    Deterministic market-making agent: state → (ask_offset, bid_offset).

    The architecture faithfully follows Paper Section IV-A:
        Input(3) → FC(32, ReLU) → FC(32, ReLU) → FC(2, Linear)
                                                       × output_scale
                                                       → round()

    Parameters
    ----------
    hidden_size : int
        Number of neurons per hidden layer (Paper: 32).
    output_scale : float
        Multiplicative factor applied to NN outputs before rounding
        (Paper Table 4: 5).  Controls the maximum quoting distance
        in ticks from the touch.
    init_gain : float
        Gain parameter for orthogonal weight initialization
        (Paper Section IV-D: 0.9).
    init_bias : float
        Constant value for bias initialization (Paper: 0.05).
    """

    def __init__(
        self,
        hidden_size: int = 32,
        output_scale: float = 5.0,
        init_gain: float = 0.9,
        init_bias: float = 0.05,
    ):
        super().__init__()

        self.output_scale = output_scale

        # ── Network layers ──────────────────────────────────────────
        # Input: 3 features  [inventory_z, sgu1_z, sgu2_z]
        # Output: 2 floats   [ask_offset_raw, bid_offset_raw]
        #
        # ReLU is used in hidden layers (Paper Section IV-A).
        # The output layer is LINEAR — no activation — so that the
        # agent can produce both positive and negative offsets
        # (negative = quoting inside the spread).
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),    # Input → Hidden 1
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # Hidden 1 → Hidden 2
            nn.ReLU(),
            nn.Linear(hidden_size, 2),    # Hidden 2 → Output (ask, bid)
        )

        # ── Initialization (Paper Section IV-D) ────────────────────
        # Orthogonal weights + constant biases.  Applied to all Linear
        # layers in the network.
        self._init_weights(init_gain, init_bias)

    def _init_weights(self, gain: float, bias_val: float) -> None:
        """
        Apply orthogonal initialization to weights and constant to biases.

        Orthogonal initialization (Saxe et al., 2014) constructs weight
        matrices from the orthogonal group O(n).  For a matrix W of shape
        (out, in):
            1. Sample a random matrix M ~ N(0, 1) of shape (out, in).
            2. Compute SVD: M = U Σ V^T.
            3. Set W = U (if out ≤ in) or W = V^T (if out > in).
            4. Multiply by `gain` to control the spectral radius.

        This ensures that W^T W ≈ gain² × I, which preserves the norm
        of activations as they flow through layers — critical for stable
        forward passes even without gradient-based training.
        """
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, bias_val)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Raw forward pass: state → 2 unscaled floats.

        Parameters
        ----------
        state_tensor : torch.Tensor, shape (B, 3) or (3,)
            Batch of states [inventory_z, sgu1_z, sgu2_z].

        Returns
        -------
        torch.Tensor, shape (B, 2) or (2,)
            Raw (unscaled) outputs [ask_raw, bid_raw].
        """
        return self.net(state_tensor)

    @torch.no_grad()
    def act(self, state_tensor: torch.Tensor) -> Tuple[int, int]:
        """
        Full action pipeline: state → scaled, rounded integer tick offsets.

        This is the method called during backtest evaluation:
            1. Forward pass → raw floats.
            2. Multiply by output_scale → tick-scale floats.
            3. Round to nearest integer → discrete tick offsets.

        Parameters
        ----------
        state_tensor : torch.Tensor, shape (3,)
            Single state [inventory_z, sgu1_z, sgu2_z].

        Returns
        -------
        (ask_offset, bid_offset) : Tuple[int, int]
            Integer tick offsets for ask and bid quotes.
            ask_offset > 0 means quoting ABOVE best ask (conservative).
            bid_offset > 0 means quoting BELOW best bid (conservative).
        """
        raw = self.forward(state_tensor)              # shape (2,)
        scaled = raw * self.output_scale              # e.g. [0.6, 0.4] → [3.0, 2.0]
        ask_off = int(torch.round(scaled[0]).item())  # 3.0 → 3
        bid_off = int(torch.round(scaled[1]).item())  # 2.0 → 2
        return ask_off, bid_off

    @torch.no_grad()
    def act_from_dict(self, state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Convenience method: extract features from a state dict and act.

        The state dict is produced by DeepRLSignalController, which
        augments the simulator's raw state with z-scored SGU signals.

        Expected keys:
            "inventory" : int or float (current MM inventory)
            "sgu1"      : float (z-scored SGU-1 signal)
            "sgu2"      : float (z-scored SGU-2 signal)

        The inventory is used RAW here (not z-scored).  The caller
        (neuroevolution trainer or policy controller) is responsible
        for normalizing it if desired.
        """
        inv = float(state.get("inventory", 0))
        s1 = float(state.get("sgu1", 0.0))
        s2 = float(state.get("sgu2", 0.0))
        t = torch.tensor([inv, s1, s2], dtype=torch.float32)
        return self.act(t)

    # ═══════════════════════════════════════════════════════════════
    # NEUROEVOLUTION HELPERS
    # ═══════════════════════════════════════════════════════════════
    # These methods support the genetic algorithm by providing
    # efficient weight get/set operations and mutation.

    def get_flat_weights(self) -> np.ndarray:
        """
        Extract all trainable parameters as a single 1D numpy array.

        This is the agent's "chromosome" in genetic algorithm terms.
        The GA operates on this flat vector: crossover, mutation, and
        selection all happen in chromosome space.

        Returns
        -------
        np.ndarray, shape (N_params,)
            Concatenation of all weight matrices and bias vectors,
            flattened in the order they appear in self.net.

        Example for default architecture (3→32→32→2):
            Layer 0: weight (32×3) + bias (32)  =  128 params
            Layer 1: weight (32×32) + bias (32) = 1056 params
            Layer 2: weight (2×32) + bias (2)   =   66 params
            Total: 1250 parameters
        """
        parts = []
        for p in self.parameters():
            parts.append(p.data.cpu().numpy().ravel())
        return np.concatenate(parts)

    def set_flat_weights(self, flat: np.ndarray) -> None:
        """
        Load a flat weight vector (chromosome) into the network.

        This is the inverse of get_flat_weights().  The GA calls this
        after creating a new individual (via mutation or crossover)
        to inject the new chromosome into the NN.

        Parameters
        ----------
        flat : np.ndarray, shape (N_params,)
            Must have exactly the same length as get_flat_weights().

        Raises
        ------
        ValueError
            If the flat vector has the wrong number of elements.
        """
        expected = sum(p.numel() for p in self.parameters())
        if len(flat) != expected:
            raise ValueError(
                f"Weight vector has {len(flat)} elements, "
                f"expected {expected}."
            )
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(flat[offset:offset + n])
                .reshape(p.shape)
                .to(p.device, p.dtype)
            )
            offset += n

    def num_params(self) -> int:
        """Total number of trainable parameters (chromosome length)."""
        return sum(p.numel() for p in self.parameters())

    def clone(self) -> "MMAgent":
        """
        Create an independent copy of this agent (deep copy).

        Used by the GA to create offspring: clone the parent, then
        mutate the clone's weights without affecting the parent.
        """
        return copy.deepcopy(self)

    def mutate(self, mutation_rate: float, mutation_std: float = 0.1) -> None:
        """
        Apply Gaussian perturbation to a random subset of weights.

        For each parameter element, with probability `mutation_rate`,
        add a sample from N(0, mutation_std²).  This is the standard
        mutation operator for neuroevolution (Paper Section IV-D).

        Parameters
        ----------
        mutation_rate : float
            Probability of mutating each individual weight.
            Paper: starts at 0.02, decays exponentially.
        mutation_std : float
            Standard deviation of the Gaussian perturbation.
            Controls the "step size" of mutations.  Larger values
            explore more aggressively but may destroy good solutions.
        """
        for p in self.parameters():
            # Create a binary mask: which weights to mutate
            mask = torch.rand_like(p.data) < mutation_rate
            # Sample perturbations for ALL weights (cheap), but
            # zero out the ones we don't want to mutate via the mask
            noise = torch.randn_like(p.data) * mutation_std
            p.data.add_(noise * mask.float())


# ═══════════════════════════════════════════════════════════════════════════
# ADVERSARY AGENT (Paper Section III-C, IV-B)
# ═══════════════════════════════════════════════════════════════════════════

class MMAdversary(nn.Module):
    """
    Adversarial agent that perturbs the MM agent's quotes.

    The adversary's goal is to MINIMIZE the MM agent's return by
    strategically displacing its bid/ask quotes.  This forces the
    MM agent to learn robust policies that generalize beyond the
    training data distribution.

    Architecture (Paper Section IV-B):
        Input(3) → FC(12, ReLU) → FC(2, Linear)

    The adversary is deliberately SMALLER than the MM agent (12 vs 32
    hidden neurons, 1 vs 2 hidden layers).  This prevents it from
    perfectly countering the MM agent, which would make learning
    impossible — instead, it provides a calibrated level of
    perturbation that improves robustness without blocking progress.

    State Space (Paper Eq. 8):
        S'_t = [I_t, 𝟙{Q^ask exec}, 𝟙{Q^bid exec}]

    The adversary observes the MM's inventory AND whether each quote
    was executed in the previous step.  This gives it "clairvoyance"
    about the immediate past — it knows which side of the book was
    hit and can anticipate the MM's likely response.

    Action Space (Paper Eq. 9):
        A'_t = [A^dis_{t,1} - A_{t,1},  A^dis_{t,2} - A_{t,2}]

    The adversary's action is a DISPLACEMENT of the MM's quotes:
        displaced_ask = original_ask + adversary_ask_displacement
        displaced_bid = original_bid + adversary_bid_displacement

    The total displacement is bounded:
        |A'_{t,1}| + |A'_{t,2}| ≤ adversary_power (in ticks)

    This prevents the adversary from completely destroying the MM's
    strategy — it can only make small perturbations.

    Parameters
    ----------
    hidden_size : int
        Hidden layer neurons (Paper: 12).
    output_scale : float
        Scaling factor for adversary outputs (Paper: 0.05).
        Much smaller than MM agent's scale (5) because adversary
        displacements should be small perturbations, not large moves.
    adversary_power : float
        Maximum total displacement budget in ticks (Paper Table 4: 500).
        NOTE: this is measured in "non-physical time" units per episode,
        not per-step.  The per-step budget depends on episode length.
    """

    def __init__(
        self,
        hidden_size: int = 12,
        output_scale: float = 0.05,
        adversary_power: float = 500.0,
        init_gain: float = 0.9,
        init_bias: float = 0.05,
    ):
        super().__init__()

        self.output_scale = output_scale
        self.adversary_power = adversary_power

        # Shallower network than MM agent: 1 hidden layer (Paper IV-B)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),   # Input → Hidden
            nn.ReLU(),
            nn.Linear(hidden_size, 2),   # Hidden → Output (ask_disp, bid_disp)
        )

        # Same initialization scheme as MM agent
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=init_gain)
                nn.init.constant_(module.bias, init_bias)

    @torch.no_grad()
    def displace(
        self,
        inventory: float,
        ask_executed: bool,
        bid_executed: bool,
    ) -> Tuple[int, int]:
        """
        Compute quote displacements given the adversary's state.

        Parameters
        ----------
        inventory : float
            MM agent's current inventory (normalized).
        ask_executed : bool
            Whether the MM's ask was executed in the previous step.
        bid_executed : bool
            Whether the MM's bid was executed in the previous step.

        Returns
        -------
        (ask_displacement, bid_displacement) : Tuple[int, int]
            Integer tick displacements to apply to MM's quotes.
            The sum of absolute values is clipped to adversary_power.
        """
        state = torch.tensor(
            [inventory, float(ask_executed), float(bid_executed)],
            dtype=torch.float32,
        )
        raw = self.net(state) * self.output_scale
        ask_d = raw[0].item()
        bid_d = raw[1].item()

        # Clip total displacement to adversary_power budget
        total = abs(ask_d) + abs(bid_d)
        if total > self.adversary_power and total > 0:
            scale = self.adversary_power / total
            ask_d *= scale
            bid_d *= scale

        return int(round(ask_d)), int(round(bid_d))

    # ── Neuroevolution helpers (same interface as MMAgent) ──────────

    def get_flat_weights(self) -> np.ndarray:
        parts = []
        for p in self.parameters():
            parts.append(p.data.cpu().numpy().ravel())
        return np.concatenate(parts)

    def set_flat_weights(self, flat: np.ndarray) -> None:
        expected = sum(p.numel() for p in self.parameters())
        if len(flat) != expected:
            raise ValueError(
                f"Weight vector has {len(flat)} elements, "
                f"expected {expected}."
            )
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(flat[offset:offset + n])
                .reshape(p.shape)
                .to(p.device, p.dtype)
            )
            offset += n

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def clone(self) -> "MMAdversary":
        return copy.deepcopy(self)

    def mutate(self, mutation_rate: float, mutation_std: float = 0.1) -> None:
        for p in self.parameters():
            mask = torch.rand_like(p.data) < mutation_rate
            noise = torch.randn_like(p.data) * mutation_std
            p.data.add_(noise * mask.float())
