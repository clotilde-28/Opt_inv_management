"""
Monte Carlo Simulation Engine for High-Frequency Market Making Strategies.
Simulates M=1000 price paths using a Jump-Diffusion / Micro-Price Drift model
incorporating fill-correlated adverse selection dynamics and toxic market flow.

Key Risk Principles:
- The AS model does NOT achieve global "variance reduction"; instead, it performs
  asymmetric tail-risk truncation, shifting mean PnL and eliminating catastrophic
  left-tail inventory losses while global standard deviation remains steady.
- Evaluates 99% VaR, 99% CVaR (Expected Shortfall), Skewness, and Excess Kurtosis.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from numba import njit
from src.metrics import compute_distribution_metrics, compute_var_cvar


@njit
def _run_mc_kernel_adverse_selection(
    M: int,
    N: int,
    dt: float,
    S0: float,
    mu_base: float,
    sigma: float,
    A: float,
    k: float,
    adverse_jump_prob: float,
    adverse_jump_size: float,
    adverse_drift_impact: float,
    drift_decay_rate: float,
    fixed_spread: float,
    gamma_static: float,
    gamma_0: float,
    eta: float,
    alpha: float,
    q_max: float,
    lot_size: float,
    ema_window_sec: float = 60.0
):
    """
    Numba-accelerated Monte Carlo simulation with micro-price adverse selection
    and correlated jump-diffusion dynamics across M paths and N time steps.
    """
    T = N * dt
    pnl_naive = np.zeros(M)
    pnl_static = np.zeros(M)
    pnl_advanced = np.zeros(M)
    
    alpha_ema = 1.0 - np.exp(-dt / ema_window_sec)
    decay_factor = np.exp(-drift_decay_rate * dt)
    
    for m in range(M):
        S = S0
        micro_drift = mu_base
        
        # State tracking for Naive
        q_n = 0.0
        cash_n = 0.0
        
        # State tracking for Static AS
        q_s = 0.0
        cash_s = 0.0
        
        # State tracking for Advanced AS
        q_a = 0.0
        cash_a = 0.0
        rolling_var = sigma ** 2
        
        for i in range(N):
            t = i * dt
            tau = max(T - t, 1.0)
            
            # --- 1. Strategy Quotes Calculation ---
            # Strategy 1: Naive (fixed spread, zero inventory skew)
            d_a_n = fixed_spread / 2.0
            d_b_n = fixed_spread / 2.0
            
            # Strategy 2: Static AS (linear reservation price skew)
            var_tau_s = (sigma ** 2) * tau
            spr_s = gamma_static * var_tau_s + (2.0 / gamma_static) * np.log(1.0 + gamma_static / k)
            d_a_s = max(0.5 * spr_s - q_s * gamma_static * var_tau_s, 0.01)
            d_b_s = max(0.5 * spr_s + q_s * gamma_static * var_tau_s, 0.01)
            
            # Strategy 3: Advanced AS (rolling vol, non-linear adaptive gamma)
            norm_q = min(abs(q_a) / q_max, 1.0)
            gamma_q = gamma_0 * (1.0 + eta * (norm_q ** alpha))
            rolling_sig = np.sqrt(max(rolling_var, 1e-4))
            var_tau_a = (rolling_sig ** 2) * tau
            spr_a = gamma_q * var_tau_a + (2.0 / gamma_q) * np.log(1.0 + gamma_q / k)
            d_a_a = max(0.5 * spr_a - q_a * gamma_q * var_tau_a, 0.01)
            d_b_a = max(0.5 * spr_a + q_a * gamma_q * var_tau_a, 0.01)
            
            # --- 2. Exogenous Market Order Arrivals ---
            # Order penetration depths sampled from calibrated Poisson distribution lambda(delta) = A * exp(-k * delta)
            u_ask_arrival = np.random.random()
            u_bid_arrival = np.random.random()
            
            # Ask side: incoming market BUY orders
            market_buy_occurred = False
            penetration_ask = 0.0
            prob_arrival_ask = 1.0 - np.exp(-A * dt)
            if u_ask_arrival < prob_arrival_ask:
                market_buy_occurred = True
                u_depth = np.random.random()
                penetration_ask = -np.log(max(u_depth, 1e-7)) / k
                
            # Bid side: incoming market SELL orders
            market_sell_occurred = False
            penetration_bid = 0.0
            prob_arrival_bid = 1.0 - np.exp(-A * dt)
            if u_bid_arrival < prob_arrival_bid:
                market_sell_occurred = True
                u_depth = np.random.random()
                penetration_bid = -np.log(max(u_depth, 1e-7)) / k
                
            # --- 3. Order Execution Matching ---
            # Ask execution: Market buys execute against MM asks if distance <= penetration
            ask_filled_any = False
            if market_buy_occurred:
                if d_a_n <= penetration_ask and q_n > -q_max:
                    cash_n += (S + d_a_n) * lot_size
                    q_n -= lot_size
                    ask_filled_any = True
                if d_a_s <= penetration_ask and q_s > -q_max:
                    cash_s += (S + d_a_s) * lot_size
                    q_s -= lot_size
                    ask_filled_any = True
                if d_a_a <= penetration_ask and q_a > -q_max:
                    cash_a += (S + d_a_a) * lot_size
                    q_a -= lot_size
                    ask_filled_any = True
                    
            # Bid execution: Market sells execute against MM bids if distance <= penetration
            bid_filled_any = False
            if market_sell_occurred:
                if d_b_n <= penetration_bid and q_n < q_max:
                    cash_n -= (S - d_b_n) * lot_size
                    q_n += lot_size
                    bid_filled_any = True
                if d_b_s <= penetration_bid and q_s < q_max:
                    cash_s -= (S - d_b_s) * lot_size
                    q_s += lot_size
                    bid_filled_any = True
                if d_b_a <= penetration_bid and q_a < q_max:
                    cash_a -= (S - d_b_a) * lot_size
                    q_a += lot_size
                    bid_filled_any = True

            # --- 4. Adverse Selection & Micro-Price Dynamics ---
            # Micro-price drift decay towards base drift
            micro_drift = mu_base + (micro_drift - mu_base) * decay_factor
            
            # Adverse selection jumps & drift shock correlated with order flow
            jump_term = 0.0
            if market_buy_occurred:
                # Market buy pushes micro-price up (adverse to short MM inventory)
                micro_drift += adverse_drift_impact
                if np.random.random() < adverse_jump_prob:
                    jump_term += adverse_jump_size * (1.0 + np.random.exponential(1.0)) * 0.5
                    
            if market_sell_occurred:
                # Market sell pushes micro-price down (adverse to long MM inventory)
                micro_drift -= adverse_drift_impact
                if np.random.random() < adverse_jump_prob:
                    jump_term -= adverse_jump_size * (1.0 + np.random.exponential(1.0)) * 0.5
                    
            # Mid-price evolution: Drift + Diffusion + Correlated Adverse Jumps
            z = np.random.normal(0.0, 1.0)
            dS = micro_drift * dt + sigma * np.sqrt(dt) * z + jump_term
            S += dS
            
            # Update rolling volatility proxy
            inst_var = (dS ** 2) / dt
            rolling_var = (1.0 - alpha_ema) * rolling_var + alpha_ema * inst_var
            
        # Terminal Mark-to-Market PnL
        pnl_naive[m] = cash_n + q_n * S
        pnl_static[m] = cash_s + q_s * S
        pnl_advanced[m] = cash_a + q_a * S
        
    return pnl_naive, pnl_static, pnl_advanced


class MonteCarloEngine:
    """
    Monte Carlo simulation runner for high-frequency market making risk evaluation.
    Incorporates micro-price drift and jump-diffusion adverse selection dynamics.
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        A: float,
        k: float,
        M: int = 1000,
        N: int = 1800,
        dt: float = 1.0,
        adverse_jump_prob: float = 0.70,
        adverse_jump_size: float = 2.50,
        adverse_drift_impact: float = 0.04,
        drift_decay_rate: float = 0.25,
        fixed_spread: float = 6.0,
        gamma_static: float = 5e-6,
        gamma_0: float = 5e-6,
        eta: float = 3.0,
        alpha: float = 2.0,
        q_max: float = 5.0,
        lot_size: float = 0.01,
        random_seed: int = 42
    ):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.A = A
        self.k = k
        self.M = M
        self.N = N
        self.dt = dt
        self.adverse_jump_prob = adverse_jump_prob
        self.adverse_jump_size = adverse_jump_size
        self.adverse_drift_impact = adverse_drift_impact
        self.drift_decay_rate = drift_decay_rate
        self.fixed_spread = fixed_spread
        self.gamma_static = gamma_static
        self.gamma_0 = gamma_0
        self.eta = eta
        self.alpha = alpha
        self.q_max = q_max
        self.lot_size = lot_size
        self.random_seed = random_seed

    def simulate(self) -> Dict[str, any]:
        """
        Execute Monte Carlo simulation across all M paths with adverse selection dynamics.
        """
        np.random.seed(self.random_seed)
        
        # Warm-up / compile Numba kernel
        _run_mc_kernel_adverse_selection(
            M=2, N=5, dt=self.dt, S0=self.S0, mu_base=self.mu, sigma=self.sigma,
            A=self.A, k=self.k,
            adverse_jump_prob=self.adverse_jump_prob,
            adverse_jump_size=self.adverse_jump_size,
            adverse_drift_impact=self.adverse_drift_impact,
            drift_decay_rate=self.drift_decay_rate,
            fixed_spread=self.fixed_spread,
            gamma_static=self.gamma_static, gamma_0=self.gamma_0,
            eta=self.eta, alpha=self.alpha, q_max=self.q_max, lot_size=self.lot_size
        )
        
        # Full simulation
        pnl_n, pnl_s, pnl_a = _run_mc_kernel_adverse_selection(
            M=self.M, N=self.N, dt=self.dt, S0=self.S0, mu_base=self.mu, sigma=self.sigma,
            A=self.A, k=self.k,
            adverse_jump_prob=self.adverse_jump_prob,
            adverse_jump_size=self.adverse_jump_size,
            adverse_drift_impact=self.adverse_drift_impact,
            drift_decay_rate=self.drift_decay_rate,
            fixed_spread=self.fixed_spread,
            gamma_static=self.gamma_static, gamma_0=self.gamma_0,
            eta=self.eta, alpha=self.alpha, q_max=self.q_max, lot_size=self.lot_size
        )
        
        # Calculate full statistical and tail risk metrics
        def get_full_stats(pnl_arr: np.ndarray) -> Dict[str, any]:
            metrics = compute_distribution_metrics(pnl_arr, alpha=0.99)
            metrics["pnl"] = pnl_arr
            return metrics
            
        return {
            "Naive": get_full_stats(pnl_n),
            "Static AS": get_full_stats(pnl_s),
            "Advanced AS": get_full_stats(pnl_a),
            "params": {
                "M": self.M,
                "N": self.N,
                "dt": self.dt,
                "S0": self.S0,
                "mu": self.mu,
                "sigma": self.sigma,
                "A": self.A,
                "k": self.k,
                "adverse_jump_prob": self.adverse_jump_prob,
                "adverse_jump_size": self.adverse_jump_size,
                "adverse_drift_impact": self.adverse_drift_impact
            }
        }
