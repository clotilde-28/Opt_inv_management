"""
Publication-quality visualization module for High-Frequency Market Making backtest results.
Generates:
1. Historical 3-Way Strategy Comparison (Mid-price, Inventory, Total MtM PnL)
2. Advanced AS Deep Dive (Dynamic Quotes & Skewing, Inventory, Realized vs Unrealized PnL)
3. Monte Carlo PnL Distribution (KDE with 99% CVaR vertical lines and summary statistics table)
"""

from typing import Dict, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure publication aesthetics
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.1,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "normal",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

# Curated harmonious color palette
COLOR_NAIVE = "#E63946"      # Crimson / Coral
COLOR_STATIC = "#2B6CB0"     # Royal Blue
COLOR_ADVANCED = "#0D9488"   # Deep Emerald / Teal
COLOR_MID = "#1E293B"        # Slate Charcoal
COLOR_ASK = "#EF4444"        # Red for Asks
COLOR_BID = "#10B981"        # Green for Bids
COLOR_REALIZED = "#2563EB"   # Blue for Realized PnL
COLOR_UNREALIZED = "#F59E0B" # Amber for Unrealized PnL


def plot_historical_comparison(
    hist_results: Dict[str, Dict[str, any]],
    q_max: float = 5.0,
    save_path: str = "figures/historical_3way_comparison.png",
    downsample_factor: int = 20
) -> str:
    """
    Figure 1: Historical 3-Way Comparison (3 subplots in one figure):
    - Top: Mid-Price evolution over the 24h horizon.
    - Middle: Inventory Trajectories (q_t) for Naive, Static AS, and Advanced AS with hard limits.
    - Bottom: Total Mark-to-Market PnL Evolution for the 3 strategies.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract timestamps and downsample for ultra-smooth rendering
    sample_strat = next(iter(hist_results.values()))
    datetimes = pd.to_datetime(sample_strat["datetime"])[::downsample_factor]
    mid_prices = sample_strat["mid_prices"][::downsample_factor]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Historical 24-Hour Market Making Backtest: Binance BTC/USDT", y=0.98)
    
    # 1. Top Subplot: Mid-Price Evolution
    ax0 = axes[0]
    ax0.plot(datetimes, mid_prices, color=COLOR_MID, linewidth=1.5, label="BTC/USDT Mid-Price")
    ax0.set_title("Mid-Price Evolution (24-Hour Horizon)")
    ax0.set_ylabel("Price ($)")
    ax0.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax0.grid(True)
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    # Annotate price move
    p_start, p_end = mid_prices[0], mid_prices[-1]
    net_move = p_end - p_start
    ax0.annotate(
        f"24h Move: ${net_move:,.1f} ({net_move/p_start*100:+.2f}%)",
        xy=(datetimes.iloc[-1], p_end),
        xytext=(datetimes.iloc[int(len(datetimes)*0.75)], p_end + 300),
        arrowprops=dict(facecolor="#64748B", arrowstyle="->", lw=1.2),
        fontsize=9.5, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.3", fc="#F1F5F9", ec="#94A3B8")
    )
    
    # 2. Middle Subplot: Inventory Trajectories
    ax1 = axes[1]
    color_map = {
        "naive": COLOR_NAIVE,
        "static_as": COLOR_STATIC,
        "advanced_as": COLOR_ADVANCED
    }
    label_map = {
        "naive": "Naive (Fixed Spread, Zero Skew)",
        "static_as": "Avellaneda-Stoikov (Static \u03c3)",
        "advanced_as": "Advanced AS (Rolling \u03c3 & Adaptive \u03b3)"
    }
    
    for key, data in hist_results.items():
        q_series = data["inventory"][::downsample_factor]
        ax1.plot(
            datetimes, 
            q_series, 
            color=color_map[key], 
            linewidth=1.4, 
            label=label_map[key],
            alpha=0.85
        )
        
    # Hard inventory limit lines
    ax1.axhline(q_max, color="#991B1B", linestyle=":", linewidth=1.4, label=f"Hard Limit (+{q_max:.1f} BTC)")
    ax1.axhline(-q_max, color="#991B1B", linestyle=":", linewidth=1.4, label=f"Hard Limit (-{q_max:.1f} BTC)")
    ax1.axhline(0.0, color="#64748B", linestyle="-", linewidth=0.8, alpha=0.5)
    
    ax1.set_title("Inventory Trajectories ($q_t$) & Risk Bound Enforcement")
    ax1.set_ylabel("Inventory (BTC)")
    ax1.set_ylim(-q_max * 1.25, q_max * 1.25)
    ax1.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", ncol=2)
    ax1.grid(True)
    
    # 3. Bottom Subplot: Total Mark-to-Market PnL Evolution
    ax2 = axes[2]
    for key, data in hist_results.items():
        pnl_series = data["total_pnl"][::downsample_factor]
        final_val = pnl_series[-1]
        ax2.plot(
            datetimes, 
            pnl_series, 
            color=color_map[key], 
            linewidth=1.6, 
            label=f"{label_map[key]} — End: ${final_val:,.1f}"
        )
        
    ax2.axhline(0.0, color="#64748B", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.set_title("Total Mark-to-Market PnL Evolution ($X_t + q_t S_t$)")
    ax2.set_ylabel("Mark-to-Market PnL ($)")
    ax2.set_xlabel("Time (UTC)")
    ax2.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax2.grid(True)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_advanced_as_deep_dive(
    adv_data: Dict[str, any],
    raw_sigmas: Optional[np.ndarray] = None,
    q_max: float = 5.0,
    save_path: str = "figures/advanced_as_deep_dive.png",
    downsample_factor: int = 15,
    window_hours: float = 2.0
) -> str:
    """
    Figure 2: Advanced AS Deep Dive (4 subplots in one figure):
    - Subplot 1 (Top): Mid-Price along with the dynamic Bid/Ask quotes showing skewing and spread widening (2h zoom).
    - Subplot 2: Dynamic Volatility Trajectory (sigma_t): EMA Volatility vs Raw Rolling Volatility over the full 24h horizon.
    - Subplot 3: Full 24-hour Inventory trajectory (q_t) exhibiting rapid mean reversion.
    - Subplot 4 (Bottom): Exact PnL Decomposition into Realized PnL vs Unrealized PnL vs Total MtM.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    datetimes = pd.to_datetime(adv_data["datetime"])
    mids = adv_data["mid_prices"]
    asks = adv_data["ask_quotes"]
    bids = adv_data["bid_quotes"]
    inv = adv_data["inventory"]
    realized = adv_data["realized_pnl"]
    unrealized = adv_data["unrealized_pnl"]
    tot_pnl = adv_data["total_pnl"]
    
    # Volatility series
    ema_vol = adv_data.get("ema_sigmas", adv_data.get("rolling_sigmas", None))
    raw_vol = raw_sigmas if raw_sigmas is not None else adv_data.get("raw_sigmas", None)
    
    # For Top Plot: Select a highly active 2-hour window to vividly display quote skewing and spread widening
    start_time = datetimes.iloc[0] + pd.Timedelta(hours=4.0)
    end_time = start_time + pd.Timedelta(hours=window_hours)
    
    mask = (datetimes >= start_time) & (datetimes <= end_time)
    sub_dt = datetimes[mask][::downsample_factor]
    sub_mid = mids[mask][::downsample_factor]
    sub_ask = asks[mask][::downsample_factor]
    sub_bid = bids[mask][::downsample_factor]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.suptitle("Advanced Avellaneda-Stoikov Deep Dive: Dynamic Quotes, Volatility Dampening & PnL Decomposition", y=0.98)
    
    # 1. Top Subplot: Dynamic Quotes & Microstructure Skewing (2-hour Detailed Window)
    ax0 = axes[0]
    ax0.plot(sub_dt, sub_mid, color=COLOR_MID, linewidth=1.5, label="Mid-Price ($S_t$)")
    ax0.plot(sub_dt, sub_ask, color=COLOR_ASK, linewidth=1.1, linestyle="--", alpha=0.9, label=r"Dynamic Ask Quote ($r_t^a$)")
    ax0.plot(sub_dt, sub_bid, color=COLOR_BID, linewidth=1.1, linestyle="--", alpha=0.9, label=r"Dynamic Bid Quote ($r_t^b$)")
    ax0.fill_between(sub_dt, sub_bid, sub_ask, color="#0D9488", alpha=0.12, label=r"Dynamic AS Spread ($\delta_t$)")
    
    ax0.set_title(f"Dynamic Bid/Ask Quotes & Inventory Skewing (Detailed {window_hours}h Regime)")
    ax0.set_ylabel("Price ($)")
    ax0.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", ncol=2)
    ax0.grid(True)
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.1f}"))
    
    # 2. Subplot 2: Dynamic Volatility Trajectory (sigma_t): EMA Volatility vs Raw Rolling Volatility
    ax1 = axes[1]
    ds_dt = datetimes[::downsample_factor]
    
    if raw_vol is not None:
        ds_raw_vol = raw_vol[::downsample_factor]
        ax1.plot(
            ds_dt, ds_raw_vol, 
            color="#94A3B8", linewidth=1.0, alpha=0.75, 
            label="Raw Rolling Std (5-min window: noisy, sudden boundary drop-offs)"
        )
    if ema_vol is not None:
        ds_ema_vol = ema_vol[::downsample_factor]
        ax1.plot(
            ds_dt, ds_ema_vol, 
            color="#0284C7", linewidth=1.6, 
            label=r"EMA Smoothed Volatility ($\sigma_t$, span=300s: smooth transitions, dampens noise)"
        )
        
    ax1.axvspan(start_time, end_time, color="#FDE047", alpha=0.20, label=f"Top Plot Regime ({window_hours}h)")
    ax1.set_title(r"Dynamic Volatility Trajectory ($\sigma_t$): Continuous EMA Smoothing vs Raw Rolling Window")
    ax1.set_ylabel(r"Volatility $\sigma_t$ ($\$/\sqrt{\mathrm{s}}$)")
    ax1.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax1.grid(True)
    
    # 3. Subplot 3: Inventory Trajectory over the entire 24 hours
    ax2 = axes[2]
    ds_inv = inv[::downsample_factor]
    
    ax2.plot(ds_dt, ds_inv, color=COLOR_ADVANCED, linewidth=1.4, label="Advanced AS Inventory ($q_t$)")
    ax2.axhline(q_max, color="#991B1B", linestyle=":", linewidth=1.3, label=f"Max Limit (+{q_max} BTC)")
    ax2.axhline(-q_max, color="#991B1B", linestyle=":", linewidth=1.3, label=f"Min Limit (-{q_max} BTC)")
    ax2.axhline(0.0, color="#64748B", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.axvspan(start_time, end_time, color="#FDE047", alpha=0.20, label=f"Top Plot Window ({window_hours}h)")
    
    ax2.set_title("Full 24-Hour Inventory Trajectory ($q_t$) with Adaptive Mean-Reversion")
    ax2.set_ylabel("Inventory (BTC)")
    ax2.set_ylim(-q_max * 1.25, q_max * 1.25)
    ax2.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", ncol=2)
    ax2.grid(True)
    
    # 4. Subplot 4 (Bottom): PnL Decomposition (Realized vs Unrealized vs Total MtM)
    ax3 = axes[3]
    ds_real = realized[::downsample_factor]
    ds_unreal = unrealized[::downsample_factor]
    ds_tot = tot_pnl[::downsample_factor]
    
    ax3.plot(ds_dt, ds_tot, color="#0F172A", linewidth=1.8, label=f"Total MtM PnL ($X_t + q_t S_t$) — End: ${ds_tot[-1]:,.1f}")
    ax3.plot(ds_dt, ds_real, color=COLOR_REALIZED, linewidth=1.3, label=f"Realized PnL (Locked Cash) — End: ${ds_real[-1]:,.1f}")
    ax3.plot(ds_dt, ds_unreal, color=COLOR_UNREALIZED, linewidth=1.2, linestyle="-.", label="Unrealized PnL ($q_t(S_t - C_t)$)")
    ax3.axhline(0.0, color="#64748B", linestyle="-", linewidth=0.8, alpha=0.5)
    
    ax3.set_title("Advanced AS PnL Decomposition: Realized vs Unrealized vs Total Mark-to-Market")
    ax3.set_ylabel("PnL ($)")
    ax3.set_xlabel("Time (UTC)")
    ax3.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax3.grid(True)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_monte_carlo_distribution(
    mc_results: Dict[str, Dict[str, any]],
    save_path: str = "figures/monte_carlo_pnl_distribution.png"
) -> str:
    """
    Figure 3: Monte Carlo PnL Distribution (1 plot):
    - Kernel Density Estimation (KDE) plot showing distribution of Terminal PnL for Naive, AS Static, Advanced AS.
    - Distinct colors and alpha=0.4 fill area.
    - 99% Conditional VaR (CVaR) vertical dashed lines for all three strategies.
    - Rigid, perfectly aligned table generated via matplotlib.table.table (ax.table).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
    strats = ["Naive", "Static AS", "Advanced AS"]
    colors = {
        "Naive": COLOR_NAIVE,
        "Static AS": COLOR_STATIC,
        "Advanced AS": COLOR_ADVANCED
    }
    
    # Plot KDE with distinct colors and alpha=0.4 fill area
    for s_name in strats:
        pnl = mc_results[s_name]["pnl"]
        c = colors[s_name]
        
        # Seaborn KDE with alpha=0.4 fill
        sns.kdeplot(
            pnl,
            ax=ax,
            color=c,
            fill=True,
            alpha=0.4,
            linewidth=2.2,
            label=f"{s_name} (Std: ${mc_results[s_name]['std']:.1f})"
        )
        
        # 99% CVaR vertical dashed line up to table boundary
        cvar_val = mc_results[s_name]["cvar_99"]
        ax.axvline(
            cvar_val,
            ymin=0.0,
            ymax=0.62,
            color=c,
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label=f"{s_name} 99% CVaR (${cvar_val:.1f})"
        )
        
    ax.set_title("Monte Carlo Terminal PnL Distribution: Progressive Variance Reduction (M = 1,000 Paths)")
    ax.set_xlabel("Terminal Mark-to-Market PnL ($)")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max * 1.18)
    
    # Rigid, perfectly aligned table using ax.table
    col_labels = ["Strategy", "Mean PnL", "Std Dev", "99% VaR", "99% CVaR"]
    cell_text = []
    cell_colors = []
    
    row_bg_tints = {
        "Naive": ["#FEE2E2"] * 5,
        "Static AS": ["#E0F2FE"] * 5,
        "Advanced AS": ["#CCFBF1"] * 5
    }
    
    for s_name in strats:
        st = mc_results[s_name]
        sign_var = "+" if st['var_99'] > 0 else ""
        sign_cvar = "+" if st['cvar_99'] > 0 else ""
        cell_text.append([
            s_name,
            f"${st['mean']:,.2f}",
            f"${st['std']:,.2f}",
            f"{sign_var}${st['var_99']:,.2f}",
            f"{sign_cvar}${st['cvar_99']:,.2f}"
        ])
        cell_colors.append(row_bg_tints[s_name])
        
    the_table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#E2E8F0"] * 5,
        loc="upper left",
        bbox=[0.02, 0.67, 0.40, 0.27],
        zorder=10
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9.5)
    the_table.set_zorder(10)
    
    for (r, c), cell in the_table.get_celld().items():
        cell.set_edgecolor("#94A3B8")
        cell.set_linewidth(1.0)
        cell.set_alpha(1.0)
        cell.set_zorder(10)
        if r == 0:
            cell.set_text_props(weight="bold", color="#0F172A")
        else:
            cell.set_text_props(color="#1E293B")
    
    ax.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
