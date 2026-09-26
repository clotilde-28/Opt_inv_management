"""
High-frequency event-driven backtesting engine for market making strategies.
Simulates Poisson order matching calibrated to Tardis tick data with exact PnL accounting.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit


@njit
def _run_backtest_kernel(
    mid_prices: np.ndarray,
    dts: np.ndarray,
    rolling_sigmas: np.ndarray,
    uniform_random_a: np.ndarray,
    uniform_random_b: np.ndarray,
    k_bids: np.ndarray,
    k_asks: np.ndarray,
    A: float,
    k: float,
    strategy_type: int, # 0: Naive, 1: Static AS, 2: Advanced AS
    fixed_spread: float,
    gamma_static: float,
    static_sigma: float,
    gamma_0: float,
    eta: float,
    alpha: float,
    max_spread: float,
    q_max: float,
    lot_size: float,
    T: float
):
    """
    Numba-optimized core simulation loop for maximum HF execution speed.
    strategy_type:
      0 = Naive (fixed spread, zero skew)
      1 = Static AS (static sigma, fixed gamma)
      2 = Advanced AS (EMA sigma, adaptive gamma, spread cap, asymmetric k)
    """
    N = len(mid_prices)
    
    inventory = np.zeros(N)
    cash = np.zeros(N)
    total_pnl = np.zeros(N)
    realized_pnl = np.zeros(N)
    unrealized_pnl = np.zeros(N)
    ask_quotes = np.zeros(N)
    bid_quotes = np.zeros(N)
    spreads = np.zeros(N)
    res_prices = np.zeros(N)
    
    q = 0.0
    c = 0.0
    avg_cost = 0.0
    realized = 0.0
    total_vol = 0.0
    
    cum_time = 0.0
    
    for i in range(N):
        s = mid_prices[i]
        dt = dts[i]
        cum_time += dt
        tau = max(T - cum_time, 1.0)
        
        ka = k_asks[i]
        kb = k_bids[i]
        
        # --- Strategy Quote Calculations ---
        if strategy_type == 0:
            # Naive MM
            half_spread = fixed_spread / 2.0
            p_ask = s + half_spread
            p_bid = s - half_spread
            delta_a = half_spread
            delta_b = half_spread
            spr = fixed_spread
            r = s
            
        elif strategy_type == 1:
            # Static AS
            sig_sq = static_sigma ** 2
            var_tau = sig_sq * tau
            spr = gamma_static * var_tau + (2.0 / gamma_static) * np.log(1.0 + gamma_static / k)
            r = s - q * gamma_static * var_tau
            delta_a = max(0.5 * spr - q * gamma_static * var_tau, 0.01)
            delta_b = max(0.5 * spr + q * gamma_static * var_tau, 0.01)
            p_ask = s + delta_a
            p_bid = s - delta_b
            
        else:
            # Advanced AS (EMA Vol + Adaptive Gamma + Spread Cap + Asymmetric k)
            sig = rolling_sigmas[i]
            norm_q = min(abs(q) / q_max, 1.0)
            gamma_q = gamma_0 * (1.0 + eta * (norm_q ** alpha))
            var_tau = (sig ** 2) * tau
            
            # Asymmetric base half-spreads
            d_a_base = 0.5 * gamma_q * var_tau + (1.0 / gamma_q) * np.log(1.0 + gamma_q / ka)
            d_b_base = 0.5 * gamma_q * var_tau + (1.0 / gamma_q) * np.log(1.0 + gamma_q / kb)
            raw_spr = d_a_base + d_b_base
            
            # Circuit Breaker / Spread Cap
            if raw_spr > max_spread:
                scale = max_spread / raw_spr
                d_a_base *= scale
                d_b_base *= scale
                raw_spr = max_spread
                
            spr = raw_spr
            r = s - q * gamma_q * var_tau
            delta_a = min(max(d_a_base - q * gamma_q * var_tau, 0.01), max_spread)
            delta_b = min(max(d_b_base + q * gamma_q * var_tau, 0.01), max_spread)
            p_ask = s + delta_a
            p_bid = s - delta_b

        # --- Fill Probabilities via Calibrated Poisson Intensity with Asymmetric k ---
        prob_a = 1.0 - np.exp(-A * np.exp(-ka * delta_a) * dt)
        prob_b = 1.0 - np.exp(-A * np.exp(-kb * delta_b) * dt)
        
        # Order Execution Checks
        # Ask fill: MM sells lot_size at p_ask
        if uniform_random_a[i] < prob_a and q > -q_max:
            fill_price = p_ask
            c += fill_price * lot_size
            total_vol += lot_size
            
            # Accounting: selling inventory
            if q > 1e-8:
                # Reducing or closing long
                closed_qty = min(lot_size, q)
                realized += closed_qty * (fill_price - avg_cost)
                rem = lot_size - closed_qty
                if rem > 1e-8:
                    # Flipped to short
                    q = -rem
                    avg_cost = fill_price
                else:
                    q -= closed_qty
                    if abs(q) < 1e-8:
                        q = 0.0
                        avg_cost = 0.0
            else:
                # Adding to short position
                new_q = q - lot_size
                avg_cost = (abs(q) * avg_cost + lot_size * fill_price) / abs(new_q)
                q = new_q

        # Bid fill: MM buys lot_size at p_bid
        if uniform_random_b[i] < prob_b and q < q_max:
            fill_price = p_bid
            c -= fill_price * lot_size
            total_vol += lot_size
            
            # Accounting: buying inventory
            if q < -1e-8:
                # Reducing or closing short
                closed_qty = min(lot_size, abs(q))
                realized += closed_qty * (avg_cost - fill_price)
                rem = lot_size - closed_qty
                if rem > 1e-8:
                    # Flipped to long
                    q = rem
                    avg_cost = fill_price
                else:
                    q += closed_qty
                    if abs(q) < 1e-8:
                        q = 0.0
                        avg_cost = 0.0
            else:
                # Adding to long position
                new_q = q + lot_size
                avg_cost = (q * avg_cost + lot_size * fill_price) / new_q
                q = new_q

        # Mark-to-Market Accounting
        unrealized = q * (s - avg_cost) if abs(q) > 1e-8 else 0.0
        tot = c + q * s
        
        inventory[i] = q
        cash[i] = c
        total_pnl[i] = tot
        realized_pnl[i] = realized
        unrealized_pnl[i] = tot - realized # Guaranteed exact decomposition
        ask_quotes[i] = p_ask
        bid_quotes[i] = p_bid
        spreads[i] = p_ask - p_bid
        res_prices[i] = r
        
    return (
        inventory,
        cash,
        total_pnl,
        realized_pnl,
        unrealized_pnl,
        ask_quotes,
        bid_quotes,
        spreads,
        res_prices,
        total_vol
    )


class Backtester:
    """
    High-frequency market making backtesting engine.
    """

    def __init__(
        self,
        merged_data: pd.DataFrame,
        A: float,
        k: float,
        static_sigma: float,
        rolling_sigma: np.ndarray,
        k_bids: Optional[np.ndarray] = None,
        k_asks: Optional[np.ndarray] = None,
        T: float = 86400.0,
        lot_size: float = 0.01,
        q_max: float = 5.0,
        random_seed: int = 42
    ):
        self.data = merged_data
        self.A = A
        self.k = k
        self.static_sigma = static_sigma
        self.rolling_sigma = rolling_sigma
        self.T = T
        self.lot_size = lot_size
        self.q_max = q_max
        self.random_seed = random_seed
        
        # Precompute common arrays
        self.mid_prices = self.data["mid_price"].values.astype(np.float64)
        self.dts = self.data["dt"].values.astype(np.float64)
        self.N = len(self.mid_prices)
        
        self.k_bids = k_bids.astype(np.float64) if k_bids is not None else np.full(self.N, k, dtype=np.float64)
        self.k_asks = k_asks.astype(np.float64) if k_asks is not None else np.full(self.N, k, dtype=np.float64)
        
        # Paired random uniforms for identical order flow arrival across strategies
        np.random.seed(self.random_seed)
        self.uniform_random_a = np.random.uniform(0.0, 1.0, self.N)
        self.uniform_random_b = np.random.uniform(0.0, 1.0, self.N)

    def run_strategy(
        self,
        strategy_type: str,
        fixed_spread: float = 6.0,
        gamma_static: float = 5e-6,
        gamma_0: float = 5e-6,
        eta: float = 3.0,
        alpha: float = 2.0,
        max_spread: float = 25.0
    ) -> Dict[str, any]:
        """
        Run backtest for one of the three strategies: 'naive', 'static_as', 'advanced_as'.
        """
        strat_map = {"naive": 0, "static_as": 1, "advanced_as": 2}
        if strategy_type not in strat_map:
            raise ValueError(f"Unknown strategy: {strategy_type}")
            
        strat_id = strat_map[strategy_type]
        
        (
            inv,
            cash,
            tot_pnl,
            real_pnl,
            unreal_pnl,
            asks,
            bids,
            spreads,
            res_p,
            total_vol
        ) = _run_backtest_kernel(
            mid_prices=self.mid_prices,
            dts=self.dts,
            rolling_sigmas=self.rolling_sigma,
            uniform_random_a=self.uniform_random_a,
            uniform_random_b=self.uniform_random_b,
            k_bids=self.k_bids,
            k_asks=self.k_asks,
            A=self.A,
            k=self.k,
            strategy_type=strat_id,
            fixed_spread=fixed_spread,
            gamma_static=gamma_static,
            static_sigma=self.static_sigma,
            gamma_0=gamma_0,
            eta=eta,
            alpha=alpha,
            max_spread=max_spread,
            q_max=self.q_max,
            lot_size=self.lot_size,
            T=self.T
        )
        
        return {
            "inventory": inv,
            "cash": cash,
            "total_pnl": tot_pnl,
            "realized_pnl": real_pnl,
            "unrealized_pnl": unreal_pnl,
            "ask_quotes": asks,
            "bid_quotes": bids,
            "spreads": spreads,
            "reservation_prices": res_p,
            "total_volume": total_vol,
            "timestamps": self.data["timestamp"].values,
            "datetime": self.data["datetime"],
            "mid_prices": self.mid_prices
        }
