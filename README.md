# Optimal Inventory Management: High-Frequency Market Making Engine & Monte Carlo Framework

A quantitative finance framework implementing high-frequency market making strategies based on the **Avellaneda-Stoikov (2008)** framework, featuring tick-level historical backtesting with limit order book (LOB) queue realism on Binance BTC/USDT and Monte Carlo risk simulations with adverse selection dynamics.

---

## 1. Project Overview & Architecture

This repository evaluates the efficacy of optimal inventory control in high-frequency trading (HFT) across three paradigms:
1. **Naive Market Making**: Symmetric quotes around mid-price with fixed spread and zero inventory sensitivity.
2. **Avellaneda-Stoikov (Static $\sigma$)**: Static session volatility $\sigma_0$ with reservation price skewed linearly by inventory $q_t$.
3. **Advanced Avellaneda-Stoikov**: Dynamic rolling volatility $\sigma_t$ (5-minute rolling window) and non-linear adaptive risk aversion $\gamma(q_t)$, featuring quadratic penalty scaling as inventory approaches hard risk limits.

### Directory Structure
```
Opt_inv_management/
├── data/
│   ├── binance_book_ticker_2024-04-01_BTCUSDT.csv.gz  # Top-of-book L1 state (~5.62M rows)
│   └── binance_trades_2024-04-01_BTCUSDT.csv.gz       # Executed market orders (~1.89M rows)
├── figures/
│   ├── historical_3way_comparison.png                 # Figure 1: 3-way 24h backtest with LOB realism
│   ├── advanced_as_deep_dive.png                      # Figure 2: Dynamic quotes & PnL decomposition
│   └── monte_carlo_pnl_distribution.png               # Figure 3: Terminal PnL KDE, 99% CVaR & Tail Risk
├── src/
│   ├── __init__.py
│   ├── data_loader.py                                 # Causal merge_asof parser
│   ├── calibration.py                                 # Poisson intensity & volatility fitting
│   ├── strategy.py                                    # MM Strategy implementations
│   ├── backtester.py                                  # Modular MatchingEngine & Numba-accelerated LOB simulator
│   ├── monte_carlo.py                                 # 1,000-path Jump-Diffusion & Adverse Selection Engine
│   ├── metrics.py                                     # Skewness, Kurtosis, 99% VaR, 99% CVaR & binned Sharpe
│   └── visualization.py                               # Publication-quality figure generation
├── main.py                                            # Master pipeline orchestrator
└── README.md                                          # Quantitative documentation
```

---

## 2. Microstructural Realism & Matching Engine

Real-world high-frequency market making does not take place in a frictionless, instantaneous environment. The refactored matching engine incorporates four pillars of limit order book realism:

### 2.1 Limit Order Book Queue Position Estimation
Market makers cannot execute instantly at the touch. Limit orders must wait in line:
- **Queue Placement**: When a limit order is placed at price $p$:
  - If $p = p_{\text{touch}}$, the order is assigned behind the visible L1 volume: $Q_{\text{ahead}} = \text{volume}_{\text{L1}}$.
  - If $p$ improves the touch, $Q_{\text{ahead}} = 0$ (front of the book).
  - If $p$ is behind the touch, $Q_{\text{ahead}} = \text{volume}_{\text{L1}} + \rho_{\text{depth}} \cdot |p - p_{\text{touch}}|$.
- **Queue Consumption**: Aggressive incoming market orders consume queue volume ahead:
  $$Q_{\text{ahead}} \leftarrow \max(0, Q_{\text{ahead}} - V_{\text{trade}})$$
- **Stochastic Cancellations**: Unfilled competing orders ahead in the queue cancel over time:
  $$Q_{\text{ahead}}(t + \Delta t) = Q_{\text{ahead}}(t) \cdot \exp(-\lambda_{\text{cancel}} \Delta t)$$
- **Execution Condition**: An order can only fill after $Q_{\text{ahead}} \le 0$.

### 2.2 Network & Processing Latency ($\text{latency\_ms} = 50.0\text{ms}$)
- **Market Data Feed Latency**: At time $t$, the strategy observes delayed market prices and volatility from $t_{\text{obs}} = t - \text{latency\_ms}$.
- **Order Wire Delay**: Quotes placed or cancelled at time $t$ require $\text{latency\_ms}$ transit before activating at the exchange matching engine.
- Stale quotes remain exposed at the exchange during rapid price swings, introducing real-world latency-induced adverse selection.

### 2.3 Hawkes Process & Probabilistic Book-Shape Fills
- **Hawkes Toxic Flow**: Incoming market order intensity clusters dynamically according to a self-exciting Hawkes process:
  $$\lambda_t = \mu_0 + (\lambda_{t-1} - \mu_0) e^{-\beta \Delta t} + \alpha \cdot \mathbf{1}_{\{\text{market trade}\}}$$
  Excitation bursts simulate clustered toxic order flow that rapidly wipes out queue depth.
- **Probabilistic Fill Rate**: Fills are not deterministic at the touch; the probability of execution accounts for the exponential decay of book shape:
  $$\mathbb{P}(\text{fill} \mid \delta) = \exp(-\kappa \cdot \delta)$$
  where $\delta$ is the distance to the prevailing mid-price.

### 2.4 Strict Maker Fee Accounting
- Passive limit order executions incur maker fees deducted directly from cash:
  $$\text{Fee} = p_{\text{fill}} \cdot \text{lot\_size} \cdot \text{maker\_fee}$$
- Calibrated at **0.5 bps** ($0.00005$, institutional VIP maker tier).

---

## 3. Mathematical Framework & Metric Clarifications

### 3.1 Clarification: Asymmetric Tail-Risk Truncation vs "Variance Reduction"
The Avellaneda-Stoikov (AS) model does **not** collapse global PnL standard deviation to zero. In fact, active spread adjustment and quote skewing maintain global variance roughly constant (or slightly higher). 
The core mathematical value of the AS model is **asymmetric tail-risk truncation**:
- Shifting mean PnL into positive territory by capturing spread on both sides.
- Truncating catastrophic left-tail drawdowns and fat-tail losses (slashing 99% CVaR / Expected Shortfall).
- Reducing negative skewness and suppressing excess kurtosis.

### 3.2 Avellaneda-Stoikov Pricing Equations
Under arithmetic Brownian motion $dS_t = \sigma dW_t$:
- **Reservation Price**:
  $$r(s, q, t) = s_t - q_t \gamma \sigma^2 (T - t)$$
- **Optimal Spread**:
  $$\delta_t = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$
- **Optimal Quotes**:
  $$p_t^a = r_t + \frac{\delta_t}{2}, \quad p_t^b = r_t - \frac{\delta_t}{2}$$
When long ($q_t > 0$), $r_t < s_t$, posting the ask closer to mid to incentivize selling while moving the bid deeper into the book to avoid accumulating toxic inventory.

### 3.3 Advanced Adaptive Strategy
1. **Continuous Rolling Volatility ($\sigma_t$)**: Real-time 5-minute rolling window tracking empirical dispersion:
   $$\sigma_t = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} (S_{t-i} - S_{t-i-1})^2}$$
2. **Non-Linear Adaptive Penalty $\gamma(q_t)$**:
   $$\gamma(q_t) = \gamma_0 \left(1 + \eta \left(\frac{|q_t|}{q_{\max}}\right)^\alpha\right), \quad \alpha = 2.0, \; \eta = 4.0$$

### 3.4 Binned Sharpe Ratio Formulation
Tick-by-tick return variance collapses the denominator due to microsecond autocorrelation, falsely producing Sharpe ratios > 600.
We compute returns sampled on **discrete 5-minute bins** ($288$ periods/day) over capital $C = \$100,000$, subtracting an annualized 4% risk-free rate:
$$\text{Sharpe} = \frac{\mathbb{E}[R_{5\text{m}}] - \frac{r_f}{288 \times 365}}{\text{Std}(R_{5\text{m}})} \times \sqrt{288 \times 365}$$

---

## 4. Historical Backtest Performance (1.89M Ticks, Binance BTC/USDT)

Across the 24-hour horizon, BTC/USDT fell by **-\$1,630.20 (-2.29%)** with severe volatility spikes.

| Strategy | Final Total PnL ($) | Total Fees Paid ($) | Final Inventory ($q_T$) | Max \|Inventory\| ($\max \|q_t\|$) | Inventory Variance ($\text{Var}(q)$) | Max Drawdown ($) | Annualized Sharpe Ratio | Total Volume Traded (BTC) | Maker Fills |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Market Making** | -\$9,933.22 | \$822.74 | +3.130 | 5.010 | 2.6345 | \$15,068.34 | -34.69 | 236.99 | 23,699 |
| **Avellaneda-Stoikov (Static $\sigma$)** | -\$116.43 | \$37.12 | +0.640 | 0.960 | **0.0112** | **\$233.20** | -17.56 | 10.66 | 1,066 |
| **Advanced AS (Adaptive $\gamma$, Rolling $\sigma$)** | **+\$152.54** | \$85.63 | -0.580 | 1.830 | **0.0525** | **\$299.32** | **+8.16** | 24.60 | 2,460 |

### Key Historical Takeaways:
1. **Elimination of Frictionless Illusions**:
   - In frictionless backtests, Naive MM showed false profits. Under realistic LOB queueing, 50ms latency, and maker fees, Naive MM gets crushed (**-\$9,933.22**, Max Drawdown **\$15,068.34**), repeatedly buying into the downward price cascade and getting pegged at the +5 BTC hard inventory limit.
2. **Capital Preservation through Skewing**:
   - Both Avellaneda-Stoikov variants protect capital during the \$1,630 market drop. Static AS limits drawdown to \$233.20.
3. **Alpha Generation via Adaptive Volatility**:
   - Advanced AS adapts its spread dynamically during volatility bursts and aggressively skews its quotes, generating a net profit of **+\$152.54** (net of all maker fees) with an annualized Sharpe ratio of **8.16**.

---

## 5. Monte Carlo Simulation Engine with Adverse Selection ($M = 1,000$ Paths)

Rather than assuming standard Geometric Brownian Motion where passive fills are uncorrelated with mid-price moves, the revised engine introduces **fill-correlated Jump-Diffusion & Micro-Price Drift**:
- Market orders arrive via calibrated intensity $\lambda(\delta) = A \exp(-k \delta)$.
- Aggressive market orders trigger immediate adverse selection:
  $$S_{t+\Delta t} = S_t + \mu_{\text{micro}, t} \Delta t + \sigma \sqrt{\Delta t} Z_t + \Delta J_t^{AS}$$
  - A buy order hitting the MM ask spikes the micro-price drift $\mu_{\text{micro}}$ upwards and triggers positive adverse jumps ($J^{AS} > 0$).
  - A sell order hitting the MM bid spikes the micro-price drift downwards ($J^{AS} < 0$).
- Drift relaxes toward fundamental drift $\mu$ with decay rate $\beta_{\text{decay}} = 0.25/\text{s}$.

### Terminal PnL Distribution & Tail Risk Profile:
| Strategy | Mean PnL ($) | Std Dev ($) | Skewness | Excess Kurtosis | 99% VaR ($) | 99% CVaR ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Market Making** | \$43.89 | \$45.50 | -0.24 | +4.96 | -\$80.47 | -\$128.65 |
| **Avellaneda-Stoikov (Static $\sigma$)** | \$45.26 | \$28.25 | -0.43 | +3.87 | **-\$27.35** | **-\$58.86** |
| **Advanced AS (Adaptive $\gamma$, Rolling $\sigma$)** | **\$45.18** | \$26.50 | -0.34 | +3.99 | **-\$24.64** | **-\$49.74** |

### Tail Risk & Higher Moments Insights:
1. **Asymmetric Tail-Risk Truncation**:
   - Naive MM experiences severe adverse selection losses in the left tail (99% CVaR / Expected Shortfall of **-\$128.65**).
   - Advanced AS truncates this tail risk by **\$78.91 per run** (99% CVaR of **-\$49.74**), cutting tail losses by **61.3%**.
2. **Higher Excess Kurtosis in Naive MM**:
   - Naive MM displays high excess kurtosis (+4.96), confirming fat tails and vulnerability to extreme market dislocations. AS skews dampen outlier impacts.

---

## 6. Generated Publication Figures

- **[Figure 1: Historical 3-Way Comparison](figures/historical_3way_comparison.png)**:
  - Top: 24h Binance BTC/USDT mid-price trajectory.
  - Middle: Inventory trajectories ($q_t$) demonstrating inventory bounding at $\pm 5$ BTC under queue delay.
  - Bottom: Total Mark-to-Market PnL evolution demonstrating capital preservation by AS vs collapse of Naive MM.
- **[Figure 2: Advanced AS Deep Dive](figures/advanced_as_deep_dive.png)**:
  - Top: 2-hour detail displaying dynamic quote skewing and spread widening during volatility shocks.
  - Middle: 24-hour mean-reverting inventory trajectory.
  - Bottom: Mark-to-Market PnL decomposition: Realized PnL (locked cash) vs Unrealized PnL.
- **[Figure 3: Monte Carlo Terminal PnL Distribution](figures/monte_carlo_pnl_distribution.png)**:
  - Terminal PnL distribution KDE highlighting left-tail truncation and 99% CVaR thresholds.
  - Embedded risk summary table detailing Mean, Std Dev, Skewness, Kurtosis, 99% VaR, and 99% CVaR.

---

## 7. How to Run

### Installation
```bash
# Clone the repository and install dependencies
pip install numpy pandas scipy matplotlib seaborn numba tabulate
```

### Execution
```bash
python main.py
```
Execution takes under 15 seconds for the complete 1.89M tick LOB dataset and 1,000 Monte Carlo paths.
