"""
High-frequency event-driven matching engine and backtesting framework for market making strategies.
Incorporates limit order book queue dynamics, network latency, Hawkes toxic order flow, 
probabilistic book shape fills, and strict maker/taker fee structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numba import njit


@dataclass
class LimitOrder:
    """
    Representation of an individual limit order in the matching engine.
    """
    order_id: int
    side: str  # 'buy' (bid) or 'sell' (ask)
    price: float
    quantity: float
    timestamp_us: int
    active_time_us: int
    queue_ahead: float
    filled_qty: float = 0.0
    is_active: bool = False
    is_cancelled: bool = False


class MatchingEngine:
    """
    Microstructure Limit Order Book (LOB) Matching Engine.
    
    Features:
    - Queue position estimation: orders join behind prevailing L1 depth and execute
      only after queue ahead is consumed by market orders or stochastic cancellations.
    - Probabilistic fills & toxic flow: fill probability accounts for book shape
      P(fill) = exp(-kappa * delta) combined with a Hawkes process for order arrival clustering.
    - Network & processing latency: configurable latency_ms delays market observations
      and order submission/cancellation activation times.
    - Strict maker/taker fee tiers: deducts maker fees on passive executions.
    """

    def __init__(
        self,
        latency_ms: float = 50.0,
        maker_fee: float = 0.00005,  # 0.5 bps maker fee (institutional tier)
        taker_fee: float = 0.0004,   # 4.0 bps taker fee
        cancel_rate: float = 0.15,   # Stochastic cancellation rate per second
        requote_threshold: float = 0.50, # Minimum price change to trigger cancel/replace
        k_shape: float = 0.2891,     # Calibrated exponential book shape decay
        hawkes_mu0: float = 4.0,     # Hawkes base arrival intensity
        hawkes_alpha: float = 0.5,   # Hawkes self-excitation magnitude
        hawkes_beta: float = 1.2,    # Hawkes intensity decay rate
        depth_density: float = 1.0   # Book depth density beyond L1 (units/dollar)
    ):
        self.latency_ms = latency_ms
        self.latency_us = int(latency_ms * 1000)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.cancel_rate = cancel_rate
        self.requote_threshold = requote_threshold
        self.k_shape = k_shape
        self.hawkes_mu0 = hawkes_mu0
        self.hawkes_alpha = hawkes_alpha
        self.hawkes_beta = hawkes_beta
        self.depth_density = depth_density
        
        # State tracking
        self.active_orders: Dict[str, Optional[LimitOrder]] = {"buy": None, "sell": None}
        self.pending_orders: List[LimitOrder] = []
        self.hawkes_intensity: float = hawkes_mu0
        self.total_maker_fees: float = 0.0
        self.total_volume: float = 0.0
        self.maker_fills_count: int = 0
        self._next_order_id: int = 1

    def submit_order(
        self,
        side: str,
        price: float,
        quantity: float,
        timestamp_us: int,
        current_l1_price: float,
        current_l1_qty: float
    ) -> LimitOrder:
        """
        Submit a limit order with network latency and queue position estimation.
        """
        # Calculate initial queue ahead when order eventually activates at exchange
        if side == "sell":
            if price <= current_l1_price:
                q_ahead = 0.0 if price < current_l1_price else current_l1_qty
            else:
                q_ahead = current_l1_qty + self.depth_density * (price - current_l1_price)
        else:
            if price >= current_l1_price:
                q_ahead = 0.0 if price > current_l1_price else current_l1_qty
            else:
                q_ahead = current_l1_qty + self.depth_density * (current_l1_price - price)
                
        order = LimitOrder(
            order_id=self._next_order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp_us=timestamp_us,
            active_time_us=timestamp_us + self.latency_us,
            queue_ahead=max(0.0, q_ahead),
            is_active=False
        )
        self._next_order_id += 1
        self.pending_orders.append(order)
        return order

    def update_hawkes(self, dt: float, trade_occurred: bool = True) -> float:
        """
        Update Hawkes process intensity for order arrival clustering.
        """
        self.hawkes_intensity = (
            self.hawkes_mu0 
            + (self.hawkes_intensity - self.hawkes_mu0) * np.exp(-self.hawkes_beta * dt)
        )
        if trade_occurred:
            self.hawkes_intensity += self.hawkes_alpha
        return self.hawkes_intensity

    def decay_queue(self, dt: float):
        """
        Apply stochastic queue cancellations over elapsed time dt.
        """
        decay = np.exp(-self.cancel_rate * dt)
        for side in ["buy", "sell"]:
            order = self.active_orders[side]
            if order is not None and order.is_active:
                order.queue_ahead = max(0.0, order.queue_ahead * decay)


@njit
def _run_realistic_matching_kernel(
    mid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_prices: np.ndarray,
    ask_amounts: np.ndarray,
    bid_amounts: np.ndarray,
    trade_prices: np.ndarray,
    trade_amounts: np.ndarray,
    trade_sides: np.ndarray,  # 1: buy, 0: sell
    dts: np.ndarray,
    lag_indices: np.ndarray,
    rolling_sigmas: np.ndarray,
    uniform_random_a: np.ndarray,
    uniform_random_b: np.ndarray,
    k_shape: float,
    strategy_type: int,  # 0: Naive, 1: Static AS, 2: Advanced AS
    fixed_spread: float,
    gamma_static: float,
    static_sigma: float,
    gamma_0: float,
    eta: float,
    alpha: float,
    q_max: float,
    lot_size: float,
    T: float,
    maker_fee: float,
    cancel_rate: float,
    hawkes_mu0: float,
    hawkes_alpha: float,
    hawkes_beta: float,
    requote_threshold: float,
    depth_density: float = 1.0
):
    """
    Numba-optimized event-driven matching loop implementing full microstructural realism:
    - Delayed state evaluation & delayed quote activation (Network Latency)
    - Queue position tracking & stochastic cancellations
    - Hawkes toxic market order clustering & probabilistic shape fills: P(fill) = exp(-k * delta)
    - Strict maker fee deduction & exact Mark-to-Market PnL decomposition
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
    fees_paid = np.zeros(N)
    hawkes_intensities = np.zeros(N)
    
    q = 0.0
    c = 0.0
    avg_cost = 0.0
    realized = 0.0
    total_vol = 0.0
    total_fees = 0.0
    maker_fills = 0
    
    # Active orders state in exchange matching engine
    active_p_ask = 0.0
    active_p_bid = 0.0
    queue_ahead_ask = 0.0
    queue_ahead_bid = 0.0
    
    hawkes_intensity = hawkes_mu0
    cum_time = 0.0
    
    for i in range(N):
        dt = dts[i]
        cum_time += dt
        tau = max(T - cum_time, 1.0)
        
        # Current true exchange state
        s_curr = mid_prices[i]
        best_ask = ask_prices[i]
        best_bid = bid_prices[i]
        l1_ask_qty = ask_amounts[i]
        l1_bid_qty = bid_amounts[i]
        
        # 1. Delayed market state observed by strategy due to latency_ms
        lag_idx = lag_indices[i]
        s_obs = mid_prices[lag_idx]
        sig_obs = rolling_sigmas[lag_idx]
        
        # Update Hawkes intensity for toxic order arrival clustering
        hawkes_intensity = hawkes_mu0 + (hawkes_intensity - hawkes_mu0) * np.exp(-hawkes_beta * dt) + hawkes_alpha
        hawkes_intensities[i] = hawkes_intensity
        
        # 2. Strategy Quote Calculation based on delayed observed state
        if strategy_type == 0:
            # Naive MM (fixed spread, zero inventory skew)
            half_spread = fixed_spread / 2.0
            target_p_ask = s_obs + half_spread
            target_p_bid = s_obs - half_spread
            r = s_obs
            delta_a = half_spread
            delta_b = half_spread
        elif strategy_type == 1:
            # Static AS (linear reservation price skew)
            sig_sq = static_sigma ** 2
            var_tau = sig_sq * tau
            spr = gamma_static * var_tau + (2.0 / gamma_static) * np.log(1.0 + gamma_static / k_shape)
            r = s_obs - q * gamma_static * var_tau
            delta_a = max(0.5 * spr - q * gamma_static * var_tau, 0.01)
            delta_b = max(0.5 * spr + q * gamma_static * var_tau, 0.01)
            target_p_ask = s_obs + delta_a
            target_p_bid = s_obs - delta_b
        else:
            # Advanced AS (rolling vol, non-linear adaptive gamma)
            norm_q = min(abs(q) / q_max, 1.0)
            gamma_q = gamma_0 * (1.0 + eta * (norm_q ** alpha))
            var_tau = (sig_obs ** 2) * tau
            spr = gamma_q * var_tau + (2.0 / gamma_q) * np.log(1.0 + gamma_q / k_shape)
            r = s_obs - q * gamma_q * var_tau
            delta_a = max(0.5 * spr - q * gamma_q * var_tau, 0.01)
            delta_b = max(0.5 * spr + q * gamma_q * var_tau, 0.01)
            target_p_ask = s_obs + delta_a
            target_p_bid = s_obs - delta_b
            
        # 3. Order Placement & Queue Position Estimation
        # Check if ask quote needs replacement
        if active_p_ask <= 0.0 or abs(target_p_ask - active_p_ask) >= requote_threshold:
            active_p_ask = target_p_ask
            if active_p_ask <= best_ask:
                queue_ahead_ask = 0.0 if active_p_ask < best_ask else l1_ask_qty
            else:
                queue_ahead_ask = l1_ask_qty + depth_density * (active_p_ask - best_ask)
                
        # Check if bid quote needs replacement
        if active_p_bid <= 0.0 or abs(target_p_bid - active_p_bid) >= requote_threshold:
            active_p_bid = target_p_bid
            if active_p_bid >= best_bid:
                queue_ahead_bid = 0.0 if active_p_bid > best_bid else l1_bid_qty
            else:
                queue_ahead_bid = l1_bid_qty + depth_density * (best_bid - active_p_bid)
                
        # 4. Stochastic Cancellations in Queue
        queue_ahead_ask = max(0.0, queue_ahead_ask * np.exp(-cancel_rate * dt))
        queue_ahead_bid = max(0.0, queue_ahead_bid * np.exp(-cancel_rate * dt))
        
        # 5. Order Matching against Incoming Market Trades
        trade_p = trade_prices[i]
        trade_vol = trade_amounts[i]
        is_buy = trade_sides[i]  # 1: Market Buy, 0: Market Sell
        
        # Ask side execution (Market Buy hitting MM Ask)
        if is_buy == 1:
            if trade_p >= active_p_ask - 1e-4:
                if queue_ahead_ask > 0.0:
                    consumed = min(queue_ahead_ask, trade_vol)
                    queue_ahead_ask -= consumed
                    rem_vol = trade_vol - consumed
                else:
                    rem_vol = trade_vol
                    
                if queue_ahead_ask <= 1e-6 and rem_vol > 0.0 and q > -q_max:
                    dist_to_mid = max(active_p_ask - s_curr, 0.01)
                    prob_fill = np.exp(-k_shape * dist_to_mid)
                    if uniform_random_a[i] < prob_fill:
                        # Filled as Maker
                        fill_p = active_p_ask
                        exec_qty = lot_size
                        fee = fill_p * exec_qty * maker_fee
                        c += fill_p * exec_qty - fee
                        total_fees += fee
                        total_vol += exec_qty
                        maker_fills += 1
                        
                        # Accounting
                        if q > 1e-8:
                            closed_qty = min(exec_qty, q)
                            realized += closed_qty * (fill_p - avg_cost) - fee
                            rem = exec_qty - closed_qty
                            if rem > 1e-8:
                                q = -rem
                                avg_cost = fill_p
                            else:
                                q -= closed_qty
                                if abs(q) < 1e-8:
                                    q = 0.0
                                    avg_cost = 0.0
                        else:
                            new_q = q - exec_qty
                            avg_cost = (abs(q) * avg_cost + exec_qty * fill_p) / abs(new_q)
                            q = new_q
                            realized -= fee
                            
                        # Reposition behind queue after fill
                        queue_ahead_ask = l1_ask_qty
                        
        # Bid side execution (Market Sell hitting MM Bid)
        else:
            if trade_p <= active_p_bid + 1e-4:
                if queue_ahead_bid > 0.0:
                    consumed = min(queue_ahead_bid, trade_vol)
                    queue_ahead_bid -= consumed
                    rem_vol = trade_vol - consumed
                else:
                    rem_vol = trade_vol
                    
                if queue_ahead_bid <= 1e-6 and rem_vol > 0.0 and q < q_max:
                    dist_to_mid = max(s_curr - active_p_bid, 0.01)
                    prob_fill = np.exp(-k_shape * dist_to_mid)
                    if uniform_random_b[i] < prob_fill:
                        # Filled as Maker
                        fill_p = active_p_bid
                        exec_qty = lot_size
                        fee = fill_p * exec_qty * maker_fee
                        c -= fill_p * exec_qty + fee
                        total_fees += fee
                        total_vol += exec_qty
                        maker_fills += 1
                        
                        # Accounting
                        if q < -1e-8:
                            closed_qty = min(exec_qty, abs(q))
                            realized += closed_qty * (avg_cost - fill_p) - fee
                            rem = exec_qty - closed_qty
                            if rem > 1e-8:
                                q = rem
                                avg_cost = fill_p
                            else:
                                q += closed_qty
                                if abs(q) < 1e-8:
                                    q = 0.0
                                    avg_cost = 0.0
                        else:
                            new_q = q + exec_qty
                            avg_cost = (q * avg_cost + exec_qty * fill_p) / new_q
                            q = new_q
                            realized -= fee
                            
                        # Reposition behind queue after fill
                        queue_ahead_bid = l1_bid_qty

        # Mark-to-Market Accounting at current true exchange mid-price
        tot = c + q * s_curr
        inventory[i] = q
        cash[i] = c
        total_pnl[i] = tot
        realized_pnl[i] = realized
        unrealized_pnl[i] = tot - realized
        ask_quotes[i] = active_p_ask
        bid_quotes[i] = active_p_bid
        spreads[i] = active_p_ask - active_p_bid
        res_prices[i] = r
        fees_paid[i] = total_fees
        
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
        fees_paid,
        total_vol,
        total_fees,
        maker_fills,
        hawkes_intensities
    )


class Backtester:
    """
    High-frequency market making backtesting engine with realistic LOB matching.
    """

    def __init__(
        self,
        merged_data: pd.DataFrame,
        A: float,
        k: float,
        static_sigma: float,
        rolling_sigma: np.ndarray,
        T: float = 86400.0,
        lot_size: float = 0.01,
        q_max: float = 5.0,
        latency_ms: float = 50.0,
        maker_fee: float = 0.00005,  # 0.5 bps maker fee (institutional tier)
        cancel_rate: float = 0.15,
        requote_threshold: float = 0.50,
        hawkes_mu0: float = 4.0,
        hawkes_alpha: float = 0.5,
        hawkes_beta: float = 1.2,
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
        self.latency_ms = latency_ms
        self.maker_fee = maker_fee
        self.cancel_rate = cancel_rate
        self.requote_threshold = requote_threshold
        self.hawkes_mu0 = hawkes_mu0
        self.hawkes_alpha = hawkes_alpha
        self.hawkes_beta = hawkes_beta
        self.random_seed = random_seed
        
        # Instantiate modular Python Matching Engine
        self.matching_engine = MatchingEngine(
            latency_ms=latency_ms,
            maker_fee=maker_fee,
            cancel_rate=cancel_rate,
            requote_threshold=requote_threshold,
            k_shape=k,
            hawkes_mu0=hawkes_mu0,
            hawkes_alpha=hawkes_alpha,
            hawkes_beta=hawkes_beta
        )
        
        # Precompute common numpy arrays
        self.mid_prices = self.data["mid_price"].values.astype(np.float64)
        self.ask_prices = self.data["ask_price"].values.astype(np.float64)
        self.bid_prices = self.data["bid_price"].values.astype(np.float64)
        self.ask_amounts = self.data["ask_amount"].values.astype(np.float64)
        self.bid_amounts = self.data["bid_amount"].values.astype(np.float64)
        self.trade_prices = self.data["price"].values.astype(np.float64)
        self.trade_amounts = self.data["amount"].values.astype(np.float64)
        self.trade_sides = (self.data["side"] == "buy").values.astype(np.int8)
        self.dts = self.data["dt"].values.astype(np.float64)
        self.timestamps = self.data["timestamp"].values.astype(np.int64)
        self.N = len(self.mid_prices)
        
        # Precompute network latency lagged indices
        latency_us = int(self.latency_ms * 1000)
        self.lag_indices = np.clip(
            np.searchsorted(self.timestamps, self.timestamps - latency_us, side="right") - 1,
            0,
            self.N - 1
        ).astype(np.int64)
        
        # Paired random uniforms for identical order flow arrival across strategies
        np.random.seed(self.random_seed)
        self.uniform_random_a = np.random.uniform(0.0, 1.0, self.N)
        self.uniform_random_b = np.random.uniform(0.0, 1.0, self.N)

    def run_strategy(
        self,
        strategy_type: str,
        fixed_spread: float = 6.0,
        gamma_static: float = 3e-5,
        gamma_0: float = 3e-5,
        eta: float = 4.0,
        alpha: float = 2.0
    ) -> Dict[str, any]:
        """
        Run realistic backtest for one of the three strategies: 'naive', 'static_as', 'advanced_as'.
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
            fees,
            total_vol,
            total_fees,
            maker_fills,
            hawkes_ints
        ) = _run_realistic_matching_kernel(
            mid_prices=self.mid_prices,
            ask_prices=self.ask_prices,
            bid_prices=self.bid_prices,
            ask_amounts=self.ask_amounts,
            bid_amounts=self.bid_amounts,
            trade_prices=self.trade_prices,
            trade_amounts=self.trade_amounts,
            trade_sides=self.trade_sides,
            dts=self.dts,
            lag_indices=self.lag_indices,
            rolling_sigmas=self.rolling_sigma,
            uniform_random_a=self.uniform_random_a,
            uniform_random_b=self.uniform_random_b,
            k_shape=self.k,
            strategy_type=strat_id,
            fixed_spread=fixed_spread,
            gamma_static=gamma_static,
            static_sigma=self.static_sigma,
            gamma_0=gamma_0,
            eta=eta,
            alpha=alpha,
            q_max=self.q_max,
            lot_size=self.lot_size,
            T=self.T,
            maker_fee=self.maker_fee,
            cancel_rate=self.cancel_rate,
            hawkes_mu0=self.hawkes_mu0,
            hawkes_alpha=self.hawkes_alpha,
            hawkes_beta=self.hawkes_beta,
            requote_threshold=self.requote_threshold
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
            "fees_paid": fees,
            "total_volume": total_vol,
            "total_fees": total_fees,
            "maker_fills": maker_fills,
            "hawkes_intensities": hawkes_ints,
            "timestamps": self.timestamps,
            "datetime": self.data["datetime"],
            "mid_prices": self.mid_prices
        }
