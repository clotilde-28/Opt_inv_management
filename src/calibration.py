"""
Microstructure calibration module:
- Poisson order arrival intensity lambda(delta) = A * exp(-k * delta)
- Static and rolling volatility estimation (sigma_static, sigma_t)
- Empirical drift (mu) and volatility (sigma) for Monte Carlo path simulation
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class MarketCalibrator:
    """
    Calibrates market microstructure parameters from Tardis tick and book data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with merged tick dataset containing timestamp, side, price, mid_price, dt.
        """
        self.data = data
        self.total_duration_sec = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]) / 1e6

    def calibrate_order_intensity(
        self, 
        delta_min: float = 0.005, 
        delta_max: float = 5.0, 
        num_points: int = 40
    ) -> Dict[str, float]:
        """
        Calibrate lambda(delta) = A * exp(-k * delta) by fitting empirical arrival rates
        at varying distance depths delta from the mid-price.
        """
        deltas = np.linspace(delta_min, delta_max, num_points)
        
        # Calculate rates for buys (ask side) and sells (bid side)
        buys = self.data[self.data["side"] == "buy"]
        sells = self.data[self.data["side"] == "sell"]
        
        buy_pen = buys["price"].values - buys["mid_price"].values
        sell_pen = sells["mid_price"].values - sells["price"].values
        
        rates = []
        for d in deltas:
            cnt_buy = np.count_nonzero(buy_pen >= d)
            cnt_sell = np.count_nonzero(sell_pen >= d)
            # Average rate across both sides per second
            rate = 0.5 * (cnt_buy + cnt_sell) / self.total_duration_sec
            rates.append(rate)
            
        rates = np.array(rates)
        valid = rates > 0
        
        x = deltas[valid]
        y = np.log(rates[valid])
        
        res = stats.linregress(x, y)
        k = -res.slope
        A = float(np.exp(res.intercept))
        r_squared = float(res.rvalue**2)
        
        return {
            "A": A,
            "k": k,
            "r_squared": r_squared
        }

    def estimate_static_volatility(self, first_n_minutes: float = 30.0) -> float:
        """
        Estimate static volatility sigma_0 from the first N minutes of mid-price data.
        Returns volatility in dollars per sqrt(second).
        """
        t0 = self.data["timestamp"].iloc[0]
        cutoff_us = t0 + int(first_n_minutes * 60 * 1e6)
        
        subset = self.data[self.data["timestamp"] <= cutoff_us]
        if len(subset) < 2:
            subset = self.data.iloc[:1000]
            
        mids = subset["mid_price"].values
        dts = subset["dt"].values[1:]
        diffs = np.diff(mids)
        
        # Filter zero dt or extreme microsecond anomalies
        valid = dts > 1e-5
        if np.sum(valid) > 0:
            var_rate = np.mean((diffs[valid]**2) / dts[valid])
            sigma = float(np.sqrt(max(var_rate, 1e-4)))
        else:
            sigma = float(np.std(diffs))
            
        return sigma

    def compute_rolling_volatility(
        self, 
        window_sec: float = 300.0,
        resample_interval: str = "1s"
    ) -> pd.Series:
        """
        Compute rolling window standard deviation of 1-second price differences.
        Mapped back or returned as a continuous series.
        """
        # Resample mid price to regular 1s grid
        s_1s = self.data.set_index("datetime")["mid_price"].resample(resample_interval).last().ffill()
        diffs_1s = s_1s.diff()
        
        # Rolling standard deviation over window_sec (e.g., 300 seconds = 5 minutes)
        window_pts = max(int(window_sec), 10)
        rolling_std = diffs_1s.rolling(window_pts, min_periods=10).std().bfill()
        
        return rolling_std

    def estimate_empirical_gbm_params(self) -> Tuple[float, float, float]:
        """
        Estimate empirical S0, drift mu (per second), and volatility sigma (per sqrt(second))
        for Monte Carlo simulation matching the 24-hour dataset.
        """
        s_1s = self.data.set_index("datetime")["mid_price"].resample("1s").last().ffill().dropna().values
        S0 = float(s_1s[0])
        
        # Dollar increments
        diffs = np.diff(s_1s)
        mu = float(np.mean(diffs)) # average dollar drift per second
        sigma = float(np.std(diffs)) # dollar volatility per sqrt(second)
        
        return S0, mu, sigma
