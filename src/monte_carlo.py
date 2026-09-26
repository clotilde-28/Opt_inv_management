"""
Monte Carlo Simulation Engine for Market Making Strategies.
Simulates M=1000 price paths using Geometric/Arithmetic Brownian Motion matching
empirical drift and volatility, comparing terminal PnL distributions and CVaR across strategies.
"""

from typing import Dict, Tuple
import numpy as np
from numba import njit
from src.metrics import compute_var_cvar


@njit
def _run_mc_kernel(
    M: int,
    N: int,
    dt: float,
    S0: float,
    mu: float,
    sigma: float,
    A: float,
    k: float,
    fixed_spread: float,
    gamma_static: float,
    gamma_0: float,
    eta: float,
    alpha: float,
    q_max: float,
    lot_size: float,
    max_spread: float = 25.0,
    kappa: float = 0.35,
    ema_window_sec: float = 300.0
):
    """
    Numba-accelerated Monte Carlo simulation across M paths and N time steps.
    """
    T = N * dt
    pnl_naive = np.zeros(M)
    pnl_static = np.zeros(M)
    pnl_advanced = np.zeros(M)
    
    alpha_ema = 1.0 - np.exp(-dt / ema_window_sec)
    alpha_ofi = 1.0 - np.exp(-dt / 60.0)
    
    for m in range(M):
        S = S0
        
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
        ofi = 0.0
        
        for i in range(N):
            t = i * dt
            tau = max(T - t, 1.0)
            
            # 1. Brownian increment
            z = np.random.normal(0.0, 1.0)
            dS = mu * dt + sigma * np.sqrt(dt) * z
            S += dS
            
            # Update EMA volatility proxy
            inst_var = (dS ** 2) / dt
            rolling_var = (1.0 - alpha_ema) * rolling_var + alpha_ema * inst_var
            rolling_sig = np.sqrt(max(rolling_var, 1e-4))
            
            # Update localized OFI proxy from price shock direction
            ofi = (1.0 - alpha_ofi) * ofi + alpha_ofi * (z / 3.0)
            ofi = min(max(ofi, -0.8), 0.8)
            
            # Common random variables for paired order arrival across strategies
            u_ask = np.random.random()
            u_bid = np.random.random()
            
            # --- Strategy 1: Naive ---
            d_a_n = fixed_spread / 2.0
            d_b_n = fixed_spread / 2.0
            prob_a_n = 1.0 - np.exp(-A * np.exp(-k * d_a_n) * dt)
            prob_b_n = 1.0 - np.exp(-A * np.exp(-k * d_b_n) * dt)
            
            if u_ask < prob_a_n and q_n > -q_max:
                cash_n += (S + d_a_n) * lot_size
                q_n -= lot_size
            if u_bid < prob_b_n and q_n < q_max:
                cash_n -= (S - d_b_n) * lot_size
                q_n += lot_size
                
            # --- Strategy 2: Static AS ---
            var_tau_s = (sigma ** 2) * tau
            spr_s = gamma_static * var_tau_s + (2.0 / gamma_static) * np.log(1.0 + gamma_static / k)
            d_a_s = max(0.5 * spr_s - q_s * gamma_static * var_tau_s, 0.01)
            d_b_s = max(0.5 * spr_s + q_s * gamma_static * var_tau_s, 0.01)
            
            prob_a_s = 1.0 - np.exp(-A * np.exp(-k * d_a_s) * dt)
            prob_b_s = 1.0 - np.exp(-A * np.exp(-k * d_b_s) * dt)
            
            if u_ask < prob_a_s and q_s > -q_max:
                cash_s += (S + d_a_s) * lot_size
                q_s -= lot_size
            if u_bid < prob_b_s and q_s < q_max:
                cash_s -= (S - d_b_s) * lot_size
                q_s += lot_size
                
            # --- Strategy 3: Advanced AS (EMA Vol, Spread Cap, Asymmetric k) ---
            norm_q = min(abs(q_a) / q_max, 1.0)
            gamma_q = gamma_0 * (1.0 + eta * (norm_q ** alpha))
            var_tau_a = (rolling_sig ** 2) * tau
            
            ka = k * (1.0 - kappa * ofi)
            kb = k * (1.0 + kappa * ofi)
            d_a_base = 0.5 * gamma_q * var_tau_a + (1.0 / gamma_q) * np.log(1.0 + gamma_q / ka)
            d_b_base = 0.5 * gamma_q * var_tau_a + (1.0 / gamma_q) * np.log(1.0 + gamma_q / kb)
            raw_spr = d_a_base + d_b_base
            
            if raw_spr > max_spread:
                scale = max_spread / raw_spr
                d_a_base *= scale
                d_b_base *= scale
                
            d_a_a = min(max(d_a_base - q_a * gamma_q * var_tau_a, 0.01), max_spread)
            d_b_a = min(max(d_b_base + q_a * gamma_q * var_tau_a, 0.01), max_spread)
            
            prob_a_a = 1.0 - np.exp(-A * np.exp(-ka * d_a_a) * dt)
            prob_b_a = 1.0 - np.exp(-A * np.exp(-kb * d_b_a) * dt)
            
            if u_ask < prob_a_a and q_a > -q_max:
                cash_a += (S + d_a_a) * lot_size
                q_a -= lot_size
            if u_bid < prob_b_a and q_a < q_max:
                cash_a -= (S - d_b_a) * lot_size
                q_a += lot_size
                
        # Terminal Mark-to-Market PnL
        pnl_naive[m] = cash_n + q_n * S
        pnl_static[m] = cash_s + q_s * S
        pnl_advanced[m] = cash_a + q_a * S
        
    return pnl_naive, pnl_static, pnl_advanced


class MonteCarloEngine:
    """
    Monte Carlo simulation runner for market making strategy risk evaluation.
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
        fixed_spread: float = 6.0,
        gamma_static: float = 5e-6,
        gamma_0: float = 5e-6,
        eta: float = 3.0,
        alpha: float = 2.0,
        q_max: float = 5.0,
        lot_size: float = 0.01,
        max_spread: float = 25.0,
        kappa: float = 0.35,
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
        self.fixed_spread = fixed_spread
        self.gamma_static = gamma_static
        self.gamma_0 = gamma_0
        self.eta = eta
        self.alpha = alpha
        self.q_max = q_max
        self.lot_size = lot_size
        self.max_spread = max_spread
        self.kappa = kappa
        self.random_seed = random_seed

    def simulate(self) -> Dict[str, any]:
        """
        Execute Monte Carlo simulation across all M paths.
        """
        np.random.seed(self.random_seed)
        
        # Warm-up / compile
        _run_mc_kernel(
            M=2, N=5, dt=self.dt, S0=self.S0, mu=self.mu, sigma=self.sigma,
            A=self.A, k=self.k, fixed_spread=self.fixed_spread,
            gamma_static=self.gamma_static, gamma_0=self.gamma_0,
            eta=self.eta, alpha=self.alpha, q_max=self.q_max, lot_size=self.lot_size,
            max_spread=self.max_spread, kappa=self.kappa
        )
        
        # Full simulation
        pnl_n, pnl_s, pnl_a = _run_mc_kernel(
            M=self.M, N=self.N, dt=self.dt, S0=self.S0, mu=self.mu, sigma=self.sigma,
            A=self.A, k=self.k, fixed_spread=self.fixed_spread,
            gamma_static=self.gamma_static, gamma_0=self.gamma_0,
            eta=self.eta, alpha=self.alpha, q_max=self.q_max, lot_size=self.lot_size,
            max_spread=self.max_spread, kappa=self.kappa
        )
        
        # Calculate statistics
        def get_stats(pnl_arr: np.ndarray) -> Dict[str, float]:
            var_99, cvar_99 = compute_var_cvar(pnl_arr, alpha=0.99)
            return {
                "mean": float(np.mean(pnl_arr)),
                "std": float(np.std(pnl_arr)),
                "var_99": var_99,
                "cvar_99": cvar_99,
                "pnl": pnl_arr
            }
            
        return {
            "Naive": get_stats(pnl_n),
            "Static AS": get_stats(pnl_s),
            "Advanced AS": get_stats(pnl_a),
            "params": {
                "M": self.M,
                "N": self.N,
                "dt": self.dt,
                "S0": self.S0,
                "mu": self.mu,
                "sigma": self.sigma,
                "A": self.A,
                "k": self.k
            }
        }
