"""
Performance and risk analytics module for market making strategies.
Computes PnL, Inventory Variance, Max Drawdown, Annualized Sharpe Ratio, 
99% VaR, 99% CVaR (Expected Shortfall), Skewness, and Kurtosis.

Crucial Microstructural & Risk Clarifications:
- The Avellaneda-Stoikov (AS) model achieves asymmetric tail-risk reduction (truncating
  worst-case inventory losses and improving Expected Shortfall), NOT global "variance reduction".
  Global standard deviation remains constant or slightly increases due to active spread adjustments.
- Sharpe ratios must be computed on discrete, meaningful time bins (e.g. 1-minute, 5-minute, or 1-hour)
  to avoid microsecond tick autocorrelation that artificially collapses standard deviation to zero.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_max_drawdown(pnl_series: np.ndarray) -> float:
    """
    Calculate maximum peak-to-trough drawdown in dollars.
    """
    if len(pnl_series) == 0:
        return 0.0
    running_max = np.maximum.accumulate(pnl_series)
    drawdowns = running_max - pnl_series
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def compute_skewness(distribution: np.ndarray) -> float:
    """
    Calculate Fisher-Pearson standardized sample skewness.
    Negative skewness indicates a fat or prolonged left tail of extreme losses (typical of naive MM).
    """
    if len(distribution) < 3:
        return 0.0
    arr = np.asarray(distribution, dtype=np.float64)
    std = np.std(arr, ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(stats.skew(arr, bias=False))


def compute_kurtosis(distribution: np.ndarray, excess: bool = True) -> float:
    """
    Calculate sample excess kurtosis (Fisher definition: normal distribution = 0.0).
    High excess kurtosis indicates heavy tail risk and propensity for severe outliers.
    """
    if len(distribution) < 4:
        return 0.0
    arr = np.asarray(distribution, dtype=np.float64)
    std = np.std(arr, ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(stats.kurtosis(arr, fisher=excess, bias=False))


def compute_var_cvar(
    pnl_distribution: np.ndarray, 
    alpha: float = 0.99
) -> Tuple[float, float]:
    """
    Calculate Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall)
    at the specified confidence level alpha (default 99%).
    
    Returns
    -------
    var_threshold : float
        PnL at the (1-alpha) quantile (e.g., 1st percentile of PnL for alpha=0.99).
        If positive, the strategy is in profit even at the 99% cutoff. If negative, it represents a loss.
    cvar_threshold : float
        Conditional expectation of PnL given that PnL is at or below the VaR cutoff:
        E[PnL | PnL <= VaR_threshold]. This directly measures tail expected shortfall.
    """
    pnl_arr = np.asarray(pnl_distribution, dtype=np.float64)
    if len(pnl_arr) == 0:
        return 0.0, 0.0
        
    cutoff_percentile = (1.0 - alpha) * 100.0  # e.g., 1.0 for alpha=0.99
    var_pnl_threshold = float(np.percentile(pnl_arr, cutoff_percentile))
    
    # Conditional expectation on tail losses below or at VaR cutoff
    tail_pnls = pnl_arr[pnl_arr <= var_pnl_threshold]
    if len(tail_pnls) > 0:
        cvar_pnl_threshold = float(np.mean(tail_pnls))
    else:
        cvar_pnl_threshold = var_pnl_threshold
        
    return var_pnl_threshold, cvar_pnl_threshold


def compute_distribution_metrics(
    pnl_distribution: np.ndarray, 
    alpha: float = 0.99
) -> Dict[str, float]:
    """
    Compute comprehensive statistical and tail risk profile of a PnL distribution:
    - Mean PnL
    - Standard Deviation (note: AS shifts mean and truncates left tail, not variance reduction)
    - Skewness (measures left-tail asymmetry)
    - Excess Kurtosis (measures tail fatness)
    - 99% VaR
    - 99% CVaR (Expected Shortfall)
    """
    pnl_arr = np.asarray(pnl_distribution, dtype=np.float64)
    var_99, cvar_99 = compute_var_cvar(pnl_arr, alpha=alpha)
    
    return {
        "mean": float(np.mean(pnl_arr)),
        "std": float(np.std(pnl_arr, ddof=1)) if len(pnl_arr) > 1 else 0.0,
        "skewness": compute_skewness(pnl_arr),
        "kurtosis": compute_kurtosis(pnl_arr, excess=True),
        "var_99": var_99,
        "cvar_99": cvar_99,
    }


def compute_sharpe_ratio(
    pnl_series: np.ndarray, 
    time_series: Union[np.ndarray, pd.Series, pd.DatetimeIndex], 
    freq: str = "1min",
    risk_free_rate: float = 0.04,
    capital: float = 100_000.0,
    annualization_factor: Optional[float] = None
) -> float:
    """
    Calculate annualized Sharpe ratio from discrete binned period returns.
    
    Parameters
    ----------
    pnl_series : np.ndarray
        Cumulative Mark-to-Market PnL array ($).
    time_series : np.ndarray or pd.Series or pd.DatetimeIndex
        Array of timestamps (in microseconds or datetime), or dt increments.
    freq : str, default "1min"
        Resampling interval ("1min" for 1-minute bins, "5min" for 5-minute bins, "1h" for hourly).
    risk_free_rate : float, default 0.04
        Annualized risk-free rate (e.g. 0.04 for 4%).
    capital : float, default 100,000.0
        Nominal capital allocation ($) used to compute discrete period returns.
    annualization_factor : float, optional
        Custom annualization scalar. If None, automatically computed as sqrt(periods_per_year).
    """
    if len(pnl_series) < 2:
        return 0.0
        
    # Convert time_series into DatetimeIndex
    if isinstance(time_series, (pd.DatetimeIndex, pd.Series)) and pd.api.types.is_datetime64_any_dtype(time_series):
        dt_idx = pd.DatetimeIndex(time_series)
    elif isinstance(time_series, np.ndarray) and np.issubdtype(time_series.dtype, np.datetime64):
        dt_idx = pd.DatetimeIndex(time_series)
    else:
        ts_arr = np.asarray(time_series)
        if len(ts_arr) > 0 and ts_arr[0] > 1e12:
            # Microseconds Unix epoch timestamp
            dt_idx = pd.to_datetime(ts_arr, unit="us")
        elif len(ts_arr) > 0 and ts_arr[0] > 1e9:
            # Milliseconds Unix epoch timestamp
            dt_idx = pd.to_datetime(ts_arr, unit="ms")
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
        
    # Discrete period returns relative to nominal capital
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
    if std_returns <= 1e-10:
        return 0.0
        
    sharpe = (float(np.mean(excess_returns)) / std_returns) * ann_scalar
    return float(sharpe)


def generate_performance_dataframe(results: Dict[str, Dict[str, any]]) -> pd.DataFrame:
    """
    Construct a formatted DataFrame summarizing backtest performance across strategies
    under realistic microstructure execution (with queueing, latency, and fee tiers).
    """
    rows = []
    for strat_name, metrics in results.items():
        row = {
            "Strategy": strat_name,
            "Final Total PnL ($)": f"${metrics['final_pnl']:,.2f}",
            "Total Fees Paid ($)": f"${metrics.get('total_fees', 0.0):,.2f}",
            "Final Inventory (q_T)": f"{metrics['final_inventory']:+.3f}",
            "Max |Inventory| (max |q_t|)": f"{metrics['max_abs_inventory']:.3f}",
            "Inventory Variance (Var(q))": f"{metrics['inventory_variance']:.4f}",
            "Max Drawdown ($)": f"${metrics['max_drawdown']:,.2f}",
            "Estimated Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
            "Total Volume Traded (BTC)": f"{metrics['total_volume']:.3f}",
        }
        if "maker_fills" in metrics:
            row["Maker Fills"] = f"{int(metrics['maker_fills']):,}"
        rows.append(row)
        
    df = pd.DataFrame(rows).set_index("Strategy")
    return df
