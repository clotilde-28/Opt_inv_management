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
    dt_series: np.ndarray, 
    annualization_factor: float = np.sqrt(365 * 24 * 3600)
) -> float:
    """
    Estimate annualized Sharpe ratio from discrete PnL increments.
    """
    if len(pnl_series) < 2:
        return 0.0
        
    diffs = np.diff(pnl_series)
    std_diff = np.std(diffs)
    if std_diff <= 1e-8:
        return 0.0
        
    # Average time step
    mean_dt = np.mean(dt_series[1:]) if len(dt_series) > 1 else 1.0
    if mean_dt <= 0:
        mean_dt = 1.0
        
    # Step Sharpe scaled to annual
    step_sharpe = np.mean(diffs) / std_diff
    annual_sharpe = step_sharpe * np.sqrt((365 * 24 * 3600) / mean_dt)
    return float(annual_sharpe)


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
