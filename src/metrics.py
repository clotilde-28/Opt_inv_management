"""
Performance and risk analytics module for market making strategies.
Computes PnL, Inventory Variance, Max Drawdown, Sharpe Ratio, VaR, and CVaR.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_max_drawdown(pnl_series: np.ndarray) -> float:
    """
    Calculate maximum peak-to-trough drawdown in dollars.
    """
    running_max = np.maximum.accumulate(pnl_series)
    drawdowns = running_max - pnl_series
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def compute_sharpe_ratio(
    pnl_series: np.ndarray, 
    time_series: np.ndarray, 
    freq: str = "5min",
    risk_free_rate: float = 0.04,
    capital: float = 100_000.0,
    annualization_factor: float = None
) -> float:
    """
    Calculate annualized Sharpe ratio from discrete binned period returns.
    
    Parameters
    ----------
    pnl_series : np.ndarray
        Cumulative Mark-to-Market PnL array ($).
    time_series : np.ndarray or pd.Series or pd.DatetimeIndex
        Array of time steps dt (in seconds), timestamps, or datetime objects.
    freq : str, default "5min"
        Resampling interval (e.g. "5min" for 5-minute bins, "1h" for hourly bins).
    risk_free_rate : float, default 0.04
        Annualized risk-free rate (e.g. 0.04 for 4%).
    capital : float, default 100,000.0
        Nominal capital allocation ($) used to compute discrete period returns.
    annualization_factor : float, optional
        Custom annualization scalar. If None, automatically computed as
        sqrt(periods_per_year) (e.g., sqrt(288 * 365) for 5min, sqrt(24 * 365) for 1h).
    """
    if len(pnl_series) < 2:
        return 0.0
        
    # Convert time_series into DatetimeIndex
    if isinstance(time_series, (pd.DatetimeIndex, pd.Series)) and pd.api.types.is_datetime64_any_dtype(time_series):
        dt_idx = pd.DatetimeIndex(time_series)
    elif isinstance(time_series, np.ndarray) and np.issubdtype(time_series.dtype, np.datetime64):
        dt_idx = pd.DatetimeIndex(time_series)
    else:
        # Check if Unix timestamps or dt array
        ts_arr = np.asarray(time_series)
        if len(ts_arr) > 0 and ts_arr[0] > 1e9:
            unit = "ms" if ts_arr[0] < 1e12 else "us"
            dt_idx = pd.to_datetime(ts_arr, unit=unit)
        else:
            # Array of inter-arrival seconds dt
            cum_seconds = np.cumsum(ts_arr)
            base_time = pd.Timestamp("2024-04-01 00:00:00")
            dt_idx = base_time + pd.to_timedelta(cum_seconds, unit="s")
            
    pnl_s = pd.Series(pnl_series, index=dt_idx)
    # Deduplicate index if multiple trades share the same microsecond
    pnl_s = pnl_s[~pnl_s.index.duplicated(keep="last")]
    
    # Resample PnL to regular fixed time bins
    resampled_pnl = pnl_s.resample(freq).last().ffill().bfill()
    period_pnl = resampled_pnl.diff().dropna()
    if len(period_pnl) < 2:
        return 0.0
        
    # Discrete period returns
    period_returns = period_pnl / capital
    
    # Determine periods per year and annualization scalar
    bin_seconds = pd.Timedelta(freq).total_seconds()
    periods_per_day = 86400.0 / bin_seconds
    periods_per_year = periods_per_day * 365.0
    
    ann_scalar = np.sqrt(periods_per_year) if annualization_factor is None else annualization_factor
    
    # Adjust annual risk-free rate for bin frequency
    rf_period = risk_free_rate / periods_per_year
    excess_returns = period_returns - rf_period
    
    std_returns = float(np.std(period_returns, ddof=1))
    if std_returns <= 1e-12:
        return 0.0
        
    sharpe = (float(np.mean(excess_returns)) / std_returns) * ann_scalar
    return float(sharpe)


def compute_var_cvar(pnl_distribution: np.ndarray, alpha: float = 0.99) -> Tuple[float, float]:
    """
    Calculate Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall)
    at the specified confidence level alpha (default 99%).
    
    VaR_99 is defined as the loss at the (1-alpha) quantile: -q_{0.01}.
    CVaR_99 is the expected loss given that loss exceeds VaR: E[-PnL | -PnL >= VaR].
    We also return the cutoff threshold in PnL terms.
    """
    losses = -pnl_distribution
    cutoff_percentile = (1.0 - alpha) * 100.0 # 1st percentile of PnL
    
    var_threshold_loss = float(np.percentile(losses, alpha * 100.0))
    var_pnl_threshold = float(np.percentile(pnl_distribution, cutoff_percentile))
    
    # Conditional expectation on tail losses
    tail_losses = losses[losses >= var_threshold_loss]
    if len(tail_losses) > 0:
        cvar_loss = float(np.mean(tail_losses))
    else:
        cvar_loss = var_threshold_loss
        
    cvar_pnl_threshold = -cvar_loss
    
    return var_pnl_threshold, cvar_pnl_threshold


def generate_performance_dataframe(results: Dict[str, Dict[str, any]]) -> pd.DataFrame:
    """
    Construct a formatted DataFrame summarizing backtest performance across strategies.
    Strictly includes:
    - Final Total PnL
    - Final Inventory (q_T)
    - Maximum Absolute Inventory (max |q_t|)
    - Inventory Variance
    - Maximum Drawdown (in $)
    - Estimated Sharpe Ratio
    - Total Volume Traded
    """
    rows = []
    for strat_name, metrics in results.items():
        rows.append({
            "Strategy": strat_name,
            "Final Total PnL ($)": f"${metrics['final_pnl']:,.2f}",
            "Final Inventory (q_T)": f"{metrics['final_inventory']:+.3f}",
            "Max |Inventory| (max |q_t|)": f"{metrics['max_abs_inventory']:.3f}",
            "Inventory Variance (Var(q))": f"{metrics['inventory_variance']:.4f}",
            "Max Drawdown ($)": f"${metrics['max_drawdown']:,.2f}",
            "Estimated Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
            "Total Volume Traded (BTC)": f"{metrics['total_volume']:.3f}",
        })
        
    df = pd.DataFrame(rows).set_index("Strategy")
    return df
