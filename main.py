"""
Optimal Inventory Management
High-Frequency Market Making Backtester & Monte Carlo Simulation Framework

Comparing:
1. Naive Market Making (Fixed spread, zero inventory control)
2. Avellaneda-Stoikov (Static sigma, linear reservation price skew)
3. Advanced Avellaneda-Stoikov (Rolling sigma_t, adaptive non-linear gamma(q))

Dataset: Tardis.dev Binance BTC/USDT (24-Hour Tick & Order Book Data)
"""

import sys
import time
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.data_loader import TardisDataLoader
from src.calibration import MarketCalibrator
from src.backtester import Backtester
from src.monte_carlo import MonteCarloEngine
from src.metrics import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    generate_performance_dataframe,
    compute_var_cvar
)
from src.visualization import (
    plot_historical_comparison,
    plot_advanced_as_deep_dive,
    plot_monte_carlo_distribution
)


def main():
    print("=" * 80)
    print("  OPTIMAL INVENTORY MANAGEMENT: HIGH-FREQUENCY MM & MONTE CARLO ENGINE")
    print("=" * 80)
    
    book_file = "data/binance_book_ticker_2024-04-01_BTCUSDT.csv.gz"
    trades_file = "data/binance_trades_2024-04-01_BTCUSDT.csv.gz"
    
    # ---------------------------------------------------------
    # STEP 1: DATA LOADING & CAUSAL ALIGNMENT
    # ---------------------------------------------------------
    print("\n[Step 1/5] Loading and causally merging Tardis datasets...")
    t0 = time.time()
    loader = TardisDataLoader(book_ticker_path=book_file, trades_path=trades_file)
    merged_data = loader.load_and_merge()
    print(f"  -> Successfully merged {len(merged_data):,} trade events with book state in {time.time()-t0:.2f}s")
    print(f"  -> Horizon: {merged_data['datetime'].iloc[0]} to {merged_data['datetime'].iloc[-1]}")
    print(f"  -> Mid-price start: ${merged_data['mid_price'].iloc[0]:,.2f} | end: ${merged_data['mid_price'].iloc[-1]:,.2f}")
    
    # ---------------------------------------------------------
    # STEP 2: MICROSTRUCTURE CALIBRATION
    # ---------------------------------------------------------
    print("\n[Step 2/5] Calibrating microstructure parameters...")
    t1 = time.time()
    calibrator = MarketCalibrator(merged_data)
    
    # Calibrate Poisson intensity lambda(delta) = A * exp(-k * delta)
    intensity_params = calibrator.calibrate_order_intensity()
    A = intensity_params["A"]
    k = intensity_params["k"]
    r2 = intensity_params["r_squared"]
    print(f"  -> Order Intensity Fit: lambda(delta) = {A:.4f} * exp(-{k:.4f} * delta) [R^2 = {r2:.4f}]")
    
    # Calibrate initial static volatility (first 30 minutes)
    static_sigma = calibrator.estimate_static_volatility(first_n_minutes=30.0)
    print(f"  -> Initial Static Volatility (sigma_0, first 30m): ${static_sigma:.4f} /sqrt(s)")
    
    # Compute rolling volatility series (5-minute rolling window)
    print("  -> Computing rolling volatility (sigma_t, 5-min window)...")
    rolling_vol_series = calibrator.compute_rolling_volatility(window_sec=300.0)
    
    # Map rolling vol onto the merged trade timestamps
    roll_df = pd.DataFrame({
        "datetime": rolling_vol_series.index,
        "rolling_sigma": rolling_vol_series.values
    }).dropna()
    
    merged_with_vol = pd.merge_asof(
        merged_data,
        roll_df,
        on="datetime",
        direction="backward"
    )
    merged_with_vol["rolling_sigma"] = merged_with_vol["rolling_sigma"].bfill().fillna(static_sigma)
    rolling_sigma_arr = merged_with_vol["rolling_sigma"].values.astype(np.float64)
    print(f"  -> Rolling Volatility stats: mean=${rolling_sigma_arr.mean():.2f}, min=${rolling_sigma_arr.min():.2f}, max=${rolling_sigma_arr.max():.2f}")
    
    # Estimate full-horizon drift and volatility for Monte Carlo
    S0, mu_emp, sigma_emp = calibrator.estimate_empirical_gbm_params()
    print(f"  -> Full Empirical GBM: S0=${S0:,.2f}, mu=${mu_emp:+.4f}/s, sigma=${sigma_emp:.4f}/sqrt(s)")
    print(f"  Microstructure calibration completed in {time.time()-t1:.2f}s")
    
    # ---------------------------------------------------------
    # STEP 3: HISTORICAL BACKTESTING (3 STRATEGIES)
    # ---------------------------------------------------------
    print("\n[Step 3/5] Executing High-Frequency Backtest on 1.89M trade ticks...")
    t2 = time.time()
    
    q_max = 5.0          # Max absolute inventory (BTC)
    lot_size = 0.01      # Lot size per execution (BTC)
    fixed_spread = 6.0   # Fixed spread for Naive ($6.00 ~ 0.85 bps)
    gamma_static = 5e-6  # Static AS risk aversion (tuned for active market participation)
    gamma_0 = 5e-6       # Advanced AS base risk aversion
    eta = 3.0            # Advanced AS non-linear penalty multiplier
    alpha = 2.0          # Quadratic penalty exponent
    T_horizon = 86400.0  # 24 hours in seconds
    
    # Baseline AS spread at q=0
    baseline_as_spread = (2.0 / gamma_static) * np.log(1.0 + gamma_static / k)
    print(f"  -> Baseline AS Spread (q=0): ${baseline_as_spread:.2f} vs Naive Fixed Spread: ${fixed_spread:.2f}")
    
    backtester = Backtester(
        merged_data=merged_with_vol,
        A=A,
        k=k,
        static_sigma=static_sigma,
        rolling_sigma=rolling_sigma_arr,
        T=T_horizon,
        lot_size=lot_size,
        q_max=q_max,
        random_seed=42
    )
    
    strategies = ["naive", "static_as", "advanced_as"]
    hist_results = {}
    metrics_summary = {}
    
    strat_display_names = {
        "naive": "Naive Market Making",
        "static_as": "Avellaneda-Stoikov (Static sigma)",
        "advanced_as": "Advanced AS (Adaptive gamma, Rolling sigma)"
    }
    
    for s_key in strategies:
        st_start = time.time()
        res = backtester.run_strategy(
            strategy_type=s_key,
            fixed_spread=fixed_spread,
            gamma_static=gamma_static,
            gamma_0=gamma_0,
            eta=eta,
            alpha=alpha
        )
        hist_results[s_key] = res
        
        # Calculate key performance metrics
        pnl = res["total_pnl"]
        inv = res["inventory"]
        dts = backtester.dts
        
        final_pnl = pnl[-1]
        final_inv = inv[-1]
        max_abs_inv = float(np.max(np.abs(inv)))
        inv_var = float(np.var(inv))
        max_dd = compute_max_drawdown(pnl)
        sharpe = compute_sharpe_ratio(pnl, dts)
        tot_vol = res["total_volume"]
        
        metrics_summary[strat_display_names[s_key]] = {
            "final_pnl": final_pnl,
            "final_inventory": final_inv,
            "max_abs_inventory": max_abs_inv,
            "inventory_variance": inv_var,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "total_volume": tot_vol
        }
        print(f"  -> [{strat_display_names[s_key]}] completed in {time.time()-st_start:.2f}s | Final PnL: ${final_pnl:,.2f} | Max|q|: {max_abs_inv:.2f} BTC | Var(q): {inv_var:.4f}")
        
    print(f"  All historical backtests completed in {time.time()-t2:.2f}s")
    
    # Generate and print performance summary table
    print("\n" + "=" * 80)
    print("  HISTORICAL BACKTEST PERFORMANCE & RISK SUMMARY")
    print("=" * 80)
    perf_df = generate_performance_dataframe(metrics_summary)
    
    # Print markdown table strictly formatted
    print(perf_df.to_markdown())
    print("=" * 80)
    
    # ---------------------------------------------------------
    # STEP 4: GENERATE HISTORICAL FIGURES
    # ---------------------------------------------------------
    print("\n[Step 4/5] Generating publication-quality figures...")
    fig1_path = "figures/historical_3way_comparison.png"
    fig2_path = "figures/advanced_as_deep_dive.png"
    
    p1 = plot_historical_comparison(hist_results, q_max=q_max, save_path=fig1_path)
    print(f"  -> Generated Figure 1: {p1}")
    
    p2 = plot_advanced_as_deep_dive(hist_results["advanced_as"], q_max=q_max, save_path=fig2_path)
    print(f"  -> Generated Figure 2: {p2}")
    
    # ---------------------------------------------------------
    # STEP 5: MONTE CARLO SIMULATION ENGINE
    # ---------------------------------------------------------
    print("\n[Step 5/5] Executing Monte Carlo Engine (M = 1,000 paths)...")
    t3 = time.time()
    mc_engine = MonteCarloEngine(
        S0=S0,
        mu=mu_emp,
        sigma=sigma_emp,
        A=A,
        k=k,
        M=1000,
        N=1800, # 1800 1-second steps (30 min trading horizon)
        dt=1.0,
        fixed_spread=fixed_spread,
        gamma_static=gamma_static,
        gamma_0=gamma_0,
        eta=eta,
        alpha=alpha,
        q_max=q_max,
        lot_size=lot_size,
        random_seed=42
    )
    
    mc_results = mc_engine.simulate()
    print(f"  -> Monte Carlo 1,000 paths completed in {time.time()-t3:.2f}s")
    
    # Display Monte Carlo Tail Risk Metrics
    print("\n  MONTE CARLO TERMINAL PnL RISK PROFILE (M = 1,000 paths):")
    mc_table_rows = []
    for s_name in ["Naive", "Static AS", "Advanced AS"]:
        st = mc_results[s_name]
        mc_table_rows.append({
            "Strategy": s_name,
            "Mean PnL ($)": f"${st['mean']:,.2f}",
            "Std Dev ($)": f"${st['std']:,.2f}",
            "99% VaR ($)": f"${st['var_99']:,.2f}",
            "99% CVaR ($)": f"${st['cvar_99']:,.2f}",
        })
    mc_df = pd.DataFrame(mc_table_rows).set_index("Strategy")
    print(mc_df.to_markdown())
    
    fig3_path = "figures/monte_carlo_pnl_distribution.png"
    p3 = plot_monte_carlo_distribution(mc_results, save_path=fig3_path)
    print(f"  -> Generated Figure 3: {p3}")
    
    print("\n" + "=" * 80)
    print("  SIMULATION PIPELINE SUCCESSFULLY COMPLETED!")
    print(f"  Artifacts saved in: {os.path.abspath('figures')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
