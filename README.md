# Market Making With Signals Through Deep Reinforcement Learning

A reproduction and extension of the paper **"Market Making With Signals Through Deep Reinforcement Learning"** by Bruno Gašperov and Zvonko Kostanjčar (IEEE Access, 2021, DOI: [10.1109/ACCESS.2021.3074782](https://doi.org/10.1109/ACCESS.2021.3074782)). The original paper PDF is included in this repository.

## Paper Summary

The paper presents a novel model-free Deep Reinforcement Learning (DRL) framework for optimal market making (MM) that incorporates predictive signals. Market making is the process of simultaneously quoting bid and ask limit orders on a Limit Order Book (LOB), aiming to capture the bid-ask spread while minimizing inventory risk.

### Key Contributions

- **Novel State Space & Action Space**: A continuous, tick-based action space where the agent outputs bid/ask quote offsets relative to the current best bid/ask — encoding the spread implicitly and eliminating the need for discretized actions common in prior DRL approaches.
- **Signal Generating Units (SGUs)**: Two standalone supervised learning modules that feed predictive signals into the RL agent's state:
  - **SGU1**: An XGBoost-based model for realized price range prediction (volatility proxy)
  - **SGU2**: An LSTM-based model for short-term trend (mid-price direction) prediction
- **Neuroevolution Training**: Uses genetic algorithms instead of gradient-based methods to train the DRL agent's neural network policy, avoiding noisy gradient issues common in stochastic financial environments.
- **Adversarial Reinforcement Learning**: An adversary agent is introduced to strategically displace the MM agent's quotes, improving robustness to model uncertainty and enhancing generalization.
- **Interpretability**: Partial Dependence Plots (PDPs) are used to provide insight into the learned policy's behavior with respect to inventory, volatility signals, and trend signals.

### Architecture

- **MM Agent**: Feed-forward neural network (2 hidden layers, 32 neurons each, ReLU activations) mapping a 3D state (inventory, price range signal, trend signal) directly to continuous bid/ask offsets.
- **Adversary Agent**: Shallow neural network (1 hidden layer, 12 neurons) that learns to perturb the MM agent's quotes.
- **Reward Function**: Symmetrically dampened PnL with an absolute-value inventory penalty term (λ parameter for risk aversion), incentivizing spread capture while discouraging inventory accumulation.

### Results

Evaluated on historical tick-by-tick BTC/USD data from Bitstamp (Sep 1 – Sep 30, 2020), the DRL agent achieves:
- **20-30% higher terminal wealth** than benchmark strategies (FOIC, GLFT)
- **~60% lower inventory risk** (Mean Absolute Position) compared to equivalent-return benchmarks
- Superior PnL-to-MAP ratio, Maximum Drawdown, and overall return-to-risk performance

## Repository Structure

| File | Description |
|------|-------------|
| `DRL_MM_agent.py` | Deep RL market making agent implementation |
| `DRL_MM_strat_runner.py` | Strategy runner / backtesting engine |
| `Deep_RL_With_Signal_policy_factory.py` | DRL policy factory with signal integration |
| `GLFT_policy_factory.py` | Guéant-Lehalle-Fernandez-Tapia analytical benchmark |
| `GLFT_policy_factory_enhanced.py` | Enhanced GLFT policy with extensions |
| `LOB_SIM_SANTA_FE.py` | LOB simulator based on Santa Fe model |
| `MM_LOB_SIM.py` | Market making LOB simulation engine |
| `LOB_data.py` | LOB data structures and utilities |
| `LOB_processor.py` | LOB data processing pipeline |
| `SGU1.py` | Signal Generating Unit 1 — XGBoost price range predictor |
| `SGU2.py` | Signal Generating Unit 2 — LSTM trend predictor |
| `MM_policy_1.py` – `MM_policy_8.py` | Various MM policy implementations and benchmarks |
| `NMZI_parameters.py` | Non-Markovian Zero Intelligence model parameters |
| `RLController.py` | Reinforcement learning controller |
| `calibrate_trading_intensity.py` | Trading intensity calibration (Avellaneda-Stoikov) |
| `censored_waiting_times_calib.py` | Censored waiting times calibration |
| `animate_LOB_sim.py` | LOB simulation visualization / animation |
| `lobster_preprocessing.py` | LOBSTER data preprocessing utilities |
| `pre_process_lob_data.py` | LOB data preprocessing |
| `paper_mm_signals_drl.pdf` | Original paper (IEEE Access, 2021) |

## Benchmarks Implemented

- **FOIC(N, M, c)**: Fixed Offset with Inventory Constraints
- **GLFT(γ, c)**: Guéant-Lehalle-Fernandez-Tapia optimal quotes
- **LIIC(a, b, c)**: Linear in Inventory with Inventory Constraints

## References

- Gašperov, B., & Kostanjčar, Z. (2021). *Market Making With Signals Through Deep Reinforcement Learning*. IEEE Access, 9, 61611-61622.
- Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217-224.
- Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2012). *Dealing with the inventory risk: a solution to the market making problem*. Mathematics and Financial Economics, 7(4), 477-507.
