"""
analysis.py — Walk-Forward Analysis + Monte Carlo Simulation

Runs three analyses and compares them:
  1. Full Backtest (baseline)
  2. Walk-Forward Analysis — 12-month train / 6-month test windows, step 3 months
  3. Monte Carlo Simulation — 5,000 random trade-order shuffles on full backtest trades

Usage: python3 analysis.py
"""

import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from options_backtest import (
    SP500_TICKERS, INITIAL_CASH, MAX_TRADE_EXPOSURE_PCT,
    fetch_all, compute_indicators,
    run_portfolio_backtest, monthly_report,
)

os.makedirs("results", exist_ok=True)

TRAIN_MONTHS  = 12
TEST_MONTHS   = 6
STEP_MONTHS   = 3
MC_SIMS       = 5_000


# ── 1. Full Backtest ──────────────────────────────────────────────────────────

def run_full_backtest(ticker_data: dict, indicators: dict) -> dict:
    print("\n" + "="*60)
    print("  [1/3] FULL BACKTEST")
    print("="*60)
    r = run_portfolio_backtest(ticker_data, indicators, verbose=True)
    print(f"  Final Value  : ${r['final_value']:>12,.2f}")
    print(f"  Total Return : {r['total_return']:>+11.2f}%")
    print(f"  Max Drawdown : {r['max_drawdown']:>11.2f}%")
    print(f"  Trades       : {r['total_trades']:>12}")
    print(f"  Win Rate     : {r['win_rate']:>11.2f}%")
    return r


# ── 2. Walk-Forward Analysis ──────────────────────────────────────────────────

def run_walk_forward(ticker_data: dict, indicators: dict) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  [2/3] WALK-FORWARD ANALYSIS")
    print(f"        Train: {TRAIN_MONTHS}mo | Test: {TEST_MONTHS}mo | Step: {STEP_MONTHS}mo")
    print("="*60)

    # Find data date range
    all_dates = sorted(set().union(*[set(df.index) for df in ticker_data.values()]))
    data_start = pd.Timestamp(all_dates[0]).tz_localize(None)
    data_end   = pd.Timestamp(all_dates[-1]).tz_localize(None)

    windows = []
    train_start = data_start
    while True:
        train_end = train_start + relativedelta(months=TRAIN_MONTHS)
        test_end  = train_end  + relativedelta(months=TEST_MONTHS)
        if test_end > data_end:
            break
        windows.append((train_start, train_end, test_end))
        train_start += relativedelta(months=STEP_MONTHS)

    if not windows:
        print("  Not enough data for walk-forward windows.")
        return pd.DataFrame()

    rows = []
    for i, (tr_s, tr_e, te_e) in enumerate(windows, 1):
        print(f"\n  Window {i}/{len(windows)}: "
              f"Train [{tr_s.date()} → {tr_e.date()}] | "
              f"Test  [{tr_e.date()} → {te_e.date()}]")

        train_r = run_portfolio_backtest(ticker_data, indicators,
                                         start_dt=tr_s, end_dt=tr_e)
        test_r  = run_portfolio_backtest(ticker_data, indicators,
                                         start_dt=tr_e, end_dt=te_e)

        print(f"    In-sample  return: {train_r['total_return']:>+8.2f}%  "
              f"drawdown: {train_r['max_drawdown']:>7.2f}%  "
              f"trades: {train_r['total_trades']}")
        print(f"    Out-sample return: {test_r['total_return']:>+8.2f}%  "
              f"drawdown: {test_r['max_drawdown']:>7.2f}%  "
              f"trades: {test_r['total_trades']}")

        rows.append({
            "Window":            i,
            "Train Start":       tr_s.date(),
            "Train End":         tr_e.date(),
            "Test End":          te_e.date(),
            "In-Sample Ret %":   round(train_r["total_return"], 2),
            "Out-Sample Ret %":  round(test_r["total_return"],  2),
            "In-Sample DD %":    round(train_r["max_drawdown"],  2),
            "Out-Sample DD %":   round(test_r["max_drawdown"],   2),
            "In-Sample Trades":  train_r["total_trades"],
            "Out-Sample Trades": test_r["total_trades"],
            "In-Sample WinRate": round(train_r["win_rate"], 2),
            "Out-Sample WinRate":round(test_r["win_rate"],  2),
        })

    df = pd.DataFrame(rows)

    avg_in  = df["In-Sample Ret %"].mean()
    avg_out = df["Out-Sample Ret %"].mean()
    consistency = len(df[df["Out-Sample Ret %"] > 0]) / len(df) * 100

    print(f"\n  Summary:")
    print(f"    Avg in-sample return   : {avg_in:>+8.2f}%")
    print(f"    Avg out-of-sample ret  : {avg_out:>+8.2f}%")
    print(f"    Out-of-sample positive : {consistency:.0f}% of windows")
    print(f"    Degradation ratio      : {avg_out/avg_in*100:.1f}% of in-sample" if avg_in != 0 else "")

    return df


# ── 3. Monte Carlo Simulation ─────────────────────────────────────────────────

def run_monte_carlo(trades_log: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print(f"  [3/3] MONTE CARLO SIMULATION ({MC_SIMS:,} runs)")
    print("="*60)

    if trades_log.empty:
        print("  No trades to simulate.")
        return {}

    # Use trade return as % of premium paid (roi), and exposure fraction of portfolio
    # Each trade deploys ~2% of portfolio — we simulate compounding with that assumption
    log = trades_log.copy()
    log["roi"] = log["pnl"] / log["premium_paid"].replace(0, np.nan)
    log = log.dropna(subset=["roi"])
    rois = log["roi"].values

    print(f"  Using {len(rois)} trades | Avg ROI per trade: {rois.mean()*100:+.1f}%")

    final_values  = []
    max_drawdowns = []
    rng = np.random.default_rng(seed=42)

    for _ in range(MC_SIMS):
        # Bootstrap: sample WITH replacement — each sim gets a different mix of outcomes
        sampled   = rng.choice(rois, size=len(rois), replace=True)
        portfolio = INITIAL_CASH
        equity    = [portfolio]

        for roi in sampled:
            # Each trade deploys 2% of current portfolio as premium
            premium    = portfolio * MAX_TRADE_EXPOSURE_PCT
            trade_pnl  = premium * roi
            portfolio  = max(portfolio + trade_pnl, 0)
            equity.append(portfolio)

        equity = np.array(equity)
        peak   = np.maximum.accumulate(equity)
        dd     = (equity - peak) / np.where(peak > 0, peak, 1) * 100
        final_values.append(equity[-1])
        max_drawdowns.append(dd.min())

    fv  = np.array(final_values)
    mdd = np.array(max_drawdowns)

    results = {
        "mean_final":    np.mean(fv),
        "median_final":  np.median(fv),
        "p5_final":      np.percentile(fv, 5),
        "p25_final":     np.percentile(fv, 25),
        "p75_final":     np.percentile(fv, 75),
        "p95_final":     np.percentile(fv, 95),
        "pct_profitable": (fv > INITIAL_CASH).mean() * 100,
        "mean_mdd":      np.mean(mdd),
        "worst_mdd":     np.min(mdd),
        "raw_finals":    fv,
        "raw_mdds":      mdd,
    }

    print(f"\n  Starting capital        : ${INITIAL_CASH:>12,.2f}")
    print(f"  Mean final value        : ${results['mean_final']:>12,.2f}")
    print(f"  Median final value      : ${results['median_final']:>12,.2f}")
    print(f"  5th  percentile (worst) : ${results['p5_final']:>12,.2f}")
    print(f"  25th percentile         : ${results['p25_final']:>12,.2f}")
    print(f"  75th percentile         : ${results['p75_final']:>12,.2f}")
    print(f"  95th percentile (best)  : ${results['p95_final']:>12,.2f}")
    print(f"  % simulations profitable: {results['pct_profitable']:>11.1f}%")
    print(f"  Mean max drawdown       : {results['mean_mdd']:>11.2f}%")
    print(f"  Worst max drawdown      : {results['worst_mdd']:>11.2f}%")

    return results


# ── 4. Comparison Summary ─────────────────────────────────────────────────────

def print_comparison(full_r: dict, wf_df: pd.DataFrame, mc_r: dict):
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)

    print(f"\n  {'Metric':<35} {'Full BT':>12} {'WF Out-Smp':>12} {'MC Median':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")

    full_ret  = full_r["total_return"]
    wf_ret    = wf_df["Out-Sample Ret %"].mean() if not wf_df.empty else float("nan")
    mc_med    = (mc_r.get("median_final", INITIAL_CASH) - INITIAL_CASH) / INITIAL_CASH * 100

    full_dd   = full_r["max_drawdown"]
    wf_dd     = wf_df["Out-Sample DD %"].mean() if not wf_df.empty else float("nan")
    mc_dd     = mc_r.get("mean_mdd", float("nan"))

    full_wr   = full_r["win_rate"]
    wf_wr     = wf_df["Out-Sample WinRate"].mean() if not wf_df.empty else float("nan")

    print(f"  {'Return (%)':<35} {full_ret:>+11.2f}% {wf_ret:>+11.2f}% {mc_med:>+11.2f}%")
    print(f"  {'Max Drawdown (%)':<35} {full_dd:>11.2f}% {wf_dd:>11.2f}% {mc_dd:>11.2f}%")
    print(f"  {'Win Rate (%)':<35} {full_wr:>11.2f}% {wf_wr:>11.2f}%  {'N/A':>11}")

    print(f"\n  Monte Carlo confidence range:")
    print(f"    5th  pct (bad case)  → ${mc_r.get('p5_final', 0):>10,.2f}  "
          f"({(mc_r.get('p5_final', INITIAL_CASH)/INITIAL_CASH-1)*100:>+.1f}%)")
    print(f"    Median               → ${mc_r.get('median_final', 0):>10,.2f}  "
          f"({(mc_r.get('median_final', INITIAL_CASH)/INITIAL_CASH-1)*100:>+.1f}%)")
    print(f"    95th pct (good case) → ${mc_r.get('p95_final', 0):>10,.2f}  "
          f"({(mc_r.get('p95_final', INITIAL_CASH)/INITIAL_CASH-1)*100:>+.1f}%)")
    print(f"    % profitable runs    : {mc_r.get('pct_profitable', 0):.1f}%")

    if not wf_df.empty:
        print(f"\n  Walk-Forward window detail:")
        print(wf_df[["Window","Train Start","Test End",
                      "In-Sample Ret %","Out-Sample Ret %",
                      "In-Sample DD %","Out-Sample DD %"]].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data (shared across all analyses)...")
    ticker_data = fetch_all(SP500_TICKERS)
    indicators  = {t: compute_indicators(df) for t, df in ticker_data.items()}

    full_r = run_full_backtest(ticker_data, indicators)
    wf_df  = run_walk_forward(ticker_data, indicators)
    mc_r   = run_monte_carlo(full_r["trades_log"])

    print_comparison(full_r, wf_df, mc_r)

    # Save results
    full_r["equity_curve"].to_csv("results/equity_curve.csv")
    full_r["trades_log"].to_csv("results/trades_log.csv", index=False)
    monthly_report(full_r["equity_curve"], full_r["trades_log"]).to_csv("results/monthly_report.csv")
    if not wf_df.empty:
        wf_df.to_csv("results/walk_forward.csv", index=False)
    if mc_r:
        pd.DataFrame({
            "final_value": mc_r["raw_finals"],
            "max_drawdown": mc_r["raw_mdds"],
        }).to_csv("results/monte_carlo_runs.csv", index=False)

    print("\nAll results saved to results/")


if __name__ == "__main__":
    main()
