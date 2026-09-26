"""
Market making strategy implementations based on the Avellaneda-Stoikov (AS) framework.
Includes:
1. Naive Market Making (symmetric fixed spread, zero inventory control)
2. Avellaneda-Stoikov (Static sigma, linear inventory skew)
3. Advanced Avellaneda-Stoikov (Rolling sigma, non-linear adaptive risk aversion gamma(q))
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for high-frequency market making strategies.
    """

    def __init__(self, name: str, q_max: float = 5.0, lot_size: float = 0.01):
        self.name = name
        self.q_max = q_max
        self.lot_size = lot_size

    @abstractmethod
    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        tau: float,
        sigma: float,
        k: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute (ask_price, bid_price, delta_ask, delta_bid).
        """
        pass


class NaiveStrategy(BaseStrategy):
    """
    Naive Market Making Strategy:
    - Quotes placed symmetrically around mid-price: p_ask = s + delta/2, p_bid = s - delta/2
    - Constant spread delta_fixed
    - Zero inventory sensitivity (pure passive quoting)
    """

    def __init__(
        self, 
        fixed_spread: float = 6.0, 
        q_max: float = 5.0, 
        lot_size: float = 0.01
    ):
        super().__init__(name="Naive MM", q_max=q_max, lot_size=lot_size)
        self.fixed_spread = fixed_spread

    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        tau: float,
        sigma: float,
        k: float
    ) -> Tuple[float, float, float, float]:
        half_spread = self.fixed_spread / 2.0
        p_ask = mid_price + half_spread
        p_bid = mid_price - half_spread
        delta_ask = half_spread
        delta_bid = half_spread
        return p_ask, p_bid, delta_ask, delta_bid


class StaticASStrategy(BaseStrategy):
    """
    Standard Avellaneda-Stoikov Strategy (Static Volatility):
    - Constant sigma estimated at t=0
    - Constant risk aversion parameter gamma
    - Reservation price r(s, q, t) = s - q * gamma * sigma^2 * tau
    - Optimal spread delta = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)
    - Quotes skewed by inventory:
      p_ask = r + delta / 2
      p_bid = r - delta / 2
    """

    def __init__(
        self,
        gamma: float = 5e-6,
        static_sigma: float = 4.5,
        q_max: float = 5.0,
        lot_size: float = 0.01,
        min_half_spread: float = 0.01
    ):
        super().__init__(name="AS (Static sigma)", q_max=q_max, lot_size=lot_size)
        self.gamma = gamma
        self.static_sigma = static_sigma
        self.min_half_spread = min_half_spread

    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        tau: float,
        sigma: float,
        k: float
    ) -> Tuple[float, float, float, float]:
        sig = self.static_sigma
        var_tau = (sig ** 2) * max(tau, 1.0)
        
        # AS optimal spread
        spread = self.gamma * var_tau + (2.0 / self.gamma) * np.log(1.0 + self.gamma / k)
        
        # Reservation price skew
        r = mid_price - inventory * self.gamma * var_tau
        
        p_ask = r + spread / 2.0
        p_bid = r - spread / 2.0
        
        delta_ask = max(p_ask - mid_price, self.min_half_spread)
        delta_bid = max(mid_price - p_bid, self.min_half_spread)
        
        # Adjust quotes to conform to positive distances
        p_ask = mid_price + delta_ask
        p_bid = mid_price - delta_bid
        
        return p_ask, p_bid, delta_ask, delta_bid


class AdvancedASStrategy(BaseStrategy):
    """
    Advanced Avellaneda-Stoikov Strategy:
    - Continuous EMA volatility smoothing sigma_t (dampens microstructure noise, zero boundary jumps)
    - Adaptive non-linear risk aversion gamma(q) = gamma_0 * (1 + eta * (|q| / q_max)^alpha)
    - Circuit Breaker / Spread Cap: prevents spread explosion during extreme volatility surges
    - Asymmetric Liquidity (k_bid, k_ask): adjusts half-spreads to localized Order Flow Imbalance (OFI)
    - Dynamically widens spread in turbulent regimes to prevent adverse selection
    - Aggressively skews reservation price as inventory nears risk limits
    """

    def __init__(
        self,
        gamma_0: float = 5e-6,
        eta: float = 3.0,
        alpha: float = 2.0,
        max_spread: float = 25.0,
        q_max: float = 5.0,
        lot_size: float = 0.01,
        min_half_spread: float = 0.01
    ):
        super().__init__(name="Advanced AS (Adaptive)", q_max=q_max, lot_size=lot_size)
        self.gamma_0 = gamma_0
        self.eta = eta
        self.alpha = alpha
        self.max_spread = max_spread
        self.min_half_spread = min_half_spread

    def get_adaptive_gamma(self, inventory: float) -> float:
        """
        Scale risk aversion non-linearly with inventory magnitude.
        """
        norm_inv = min(abs(inventory) / self.q_max, 1.0)
        return self.gamma_0 * (1.0 + self.eta * (norm_inv ** self.alpha))

    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        tau: float,
        sigma: float,
        k: float,
        k_bid: Optional[float] = None,
        k_ask: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        gamma_q = self.get_adaptive_gamma(inventory)
        var_tau = (sigma ** 2) * max(tau, 1.0)
        
        kb = k_bid if k_bid is not None else k
        ka = k_ask if k_ask is not None else k
        
        # Asymmetric base half-spreads from HJB first-order conditions
        d_a_base = 0.5 * gamma_q * var_tau + (1.0 / gamma_q) * np.log(1.0 + gamma_q / ka)
        d_b_base = 0.5 * gamma_q * var_tau + (1.0 / gamma_q) * np.log(1.0 + gamma_q / kb)
        
        # Circuit Breaker / Spread Cap: cap the baseline spread during volatility spikes
        raw_spread = d_a_base + d_b_base
        if raw_spread > self.max_spread:
            scale = self.max_spread / raw_spread
            d_a_base *= scale
            d_b_base *= scale
            
        # Inventory skewing from reservation price:
        # Long inventory (q > 0) lowers ask (to sell) and lowers bid (avoid buying)
        inv_skew = inventory * gamma_q * var_tau
        delta_ask = max(d_a_base - inv_skew, self.min_half_spread)
        delta_bid = max(d_b_base + inv_skew, self.min_half_spread)
        
        # Cap individual quotes to avoid runaway distances
        delta_ask = min(delta_ask, self.max_spread)
        delta_bid = min(delta_bid, self.max_spread)
        
        p_ask = mid_price + delta_ask
        p_bid = mid_price - delta_bid
        
        return p_ask, p_bid, delta_ask, delta_bid
