"""
Portfolio Options Backtester — 1h candles

Realistic constraints:
  - IV premium: hist vol × 1.25
  - Bid/ask spread: $0.15/share on entry & exit
  - Max total options exposure: 20% of portfolio
  - Max 2% of portfolio per individual trade
  - 50% stop-loss on premium paid
  - IB commission: $0.65/contract

Accepts optional start_dt / end_dt to slice the backtest window,
allowing walk-forward analysis without re-downloading data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from options_pricer import bs_call, bs_put, historical_vol

# ── Tickers ───────────────────────────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD",  "QCOM", "TXN",  "AMAT", "LRCX", "KLAC",
    "MU",   "ADI",  "INTC", "MRVL", "ORCL", "IBM",  "NOW",  "CRM",  "INTU", "PANW",
    "AMZN", "TSLA", "GOOGL","META", "NFLX", "UBER", "BKNG", "ABNB", "EBAY", "ETSY",
    "WMT",  "COST", "HD",   "LOW",  "TGT",  "AMGN", "MCD",  "SBUX", "NKE",  "TJX",
    "JPM",  "BAC",  "WFC",  "GS",   "MS",   "C",    "AXP",  "BX",   "SCHW", "CB",
    "V",    "MA",   "SPGI", "MCO",  "CME",  "ICE",  "AON",  "TRV",  "PGR",
    "LLY",  "UNH",  "JNJ",  "ABBV", "MRK",  "TMO",  "ABT",  "BMY",  "GILD", "VRTX",
    "ISRG", "SYK",  "MDT",  "BSX",  "ELV",  "CI",   "HCA",  "ZTS",  "REGN", "MRNA",
    "CAT",  "GE",   "HON",  "DE",   "ETN",  "RTX",  "PH",   "ITW",  "MMM",  "EMR",
    "XOM",  "CVX",  "COP",  "SLB",  "EOG",  "MPC",  "PSX",  "VLO",  "OXY",
    "PG",   "PEP",  "KO",   "MO",   "MDLZ", "CL",   "GIS",  "HSY",  "SJM",
    "NEE",  "DUK",  "SO",   "AEP",  "EXC",  "BRK-B","LIN",  "ECL",  "ACN",  "ADP",
]

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CASH            = 25_000.0
MAX_TRADE_EXPOSURE_PCT  = 0.02
MAX_TOTAL_EXPOSURE_PCT  = 0.20
RISK_FREE_RATE          = 0.05
OPTION_DTE              = 30
STOP_LOSS_PCT           = 0.50
IV_PREMIUM_MULT         = 1.25
BID_ASK_SPREAD          = 0.15
EMA_FAST                = 12
EMA_SLOW                = 26
RSI_PERIOD              = 14
RSI_OB                  = 70
RSI_OS                  = 30
COMMISSION_PER_CONTRACT = 0.65


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_all(tickers: list, batch_size: int = 25) -> dict:
    print(f"Fetching hourly data for {len(tickers)} tickers...")
    data    = {}
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    for b_num, batch in enumerate(batches, 1):
        print(f"  Batch {b_num}/{len(batches)}")
        raw = yf.download(
            batch, period="730d", interval="1h",
            auto_adjust=True, progress=False, group_by="ticker"
        )
        for ticker in batch:
            try:
                df = raw.copy() if len(batch) == 1 else raw[ticker].copy()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) > 100:
                    data[ticker] = df
            except Exception:
                pass

    print(f"  Loaded {len(data)}/{len(tickers)} tickers.\n")
    return data


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    bull = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["rsi"] < RSI_OB)
    bear = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["rsi"] > RSI_OS)
    df["signal"] = 0
    df.loc[bull, "signal"] =  1
    df.loc[bear, "signal"] = -1
    return df.dropna()


def get_hist_vol(df: pd.DataFrame, idx: int, window: int = 30) -> float:
    prices = df["Close"].iloc[max(0, idx - window): idx + 1].values
    return max(historical_vol(prices, window=min(window, len(prices) - 1)) * IV_PREMIUM_MULT, 0.10)


# ── Core backtester ───────────────────────────────────────────────────────────

def run_portfolio_backtest(
    ticker_data: dict,
    indicators:  dict,
    start_dt=None,
    end_dt=None,
    initial_cash: float = INITIAL_CASH,
    verbose: bool = False,
) -> dict:
    """
    Run portfolio backtest optionally sliced to [start_dt, end_dt].
    Indicators always use full history for proper warmup.
    """
    all_times = sorted(set().union(*[set(df.index) for df in ticker_data.values()]))

    if start_dt:
        start_dt  = pd.Timestamp(start_dt).tz_localize("America/New_York") if start_dt.tzinfo is None else start_dt
        all_times = [t for t in all_times if t >= start_dt]
    if end_dt:
        end_dt    = pd.Timestamp(end_dt).tz_localize("America/New_York") if end_dt.tzinfo is None else end_dt
        all_times = [t for t in all_times if t <= end_dt]

    ticker_idx = {t: {dt: i for i, dt in enumerate(df.index)} for t, df in ticker_data.items()}

    cash         = initial_cash
    positions    = []
    trades_log   = []
    equity_curve = []

    for dt in all_times:
        options_exposure = sum(p["current_value"] for p in positions)
        portfolio_val    = cash + options_exposure
        equity_curve.append({"datetime": dt, "portfolio_value": portfolio_val})

        # ── Mark-to-market ──
        for pos in positions[:]:
            ticker = pos["ticker"]
            idx    = ticker_idx[ticker].get(dt)
            if idx is None:
                continue
            df    = ticker_data[ticker]
            price = float(df["Close"].iloc[idx])
            T     = max((pos["expiry"] - dt).total_seconds() / (365 * 24 * 3600), 0)
            vol   = get_hist_vol(df, idx)

            raw = bs_call(price, pos["strike"], T, RISK_FREE_RATE, vol) if pos["type"] == "call" \
                  else bs_put(price, pos["strike"], T, RISK_FREE_RATE, vol)
            pos["current_value"] = max(raw - BID_ASK_SPREAD * 0.5, 0) * pos["contracts"] * 100

            if pos["current_value"] <= pos["premium_paid"] * (1 - STOP_LOSS_PCT):
                cash += pos["current_value"]
                trades_log.append({"ticker": ticker, "type": pos["type"],
                                   "entry_date": pos["entry_date"], "exit_date": dt,
                                   "pnl": pos["current_value"] - pos["premium_paid"],
                                   "premium_paid": pos["premium_paid"],
                                   "portfolio_at_entry": pos.get("portfolio_at_entry", INITIAL_CASH),
                                   "exit_reason": "stop_loss"})
                positions.remove(pos)
                continue

            if T <= 0:
                intrinsic = max(price - pos["strike"], 0) if pos["type"] == "call" else max(pos["strike"] - price, 0)
                payout    = max(intrinsic - BID_ASK_SPREAD, 0) * pos["contracts"] * 100
                cash     += payout
                trades_log.append({"ticker": ticker, "type": pos["type"],
                                   "entry_date": pos["entry_date"], "exit_date": dt,
                                   "pnl": payout - pos["premium_paid"],
                                   "premium_paid": pos["premium_paid"],
                                   "portfolio_at_entry": pos.get("portfolio_at_entry", INITIAL_CASH),
                                   "exit_reason": "expiry"})
                positions.remove(pos)

        # ── Entry signals ──
        options_exposure = sum(p["current_value"] for p in positions)
        portfolio_val    = cash + options_exposure
        max_total_exp    = portfolio_val * MAX_TOTAL_EXPOSURE_PCT

        for ticker, ind_df in indicators.items():
            if dt not in ind_df.index:
                continue
            signal = ind_df.loc[dt, "signal"]
            if signal == 0:
                continue
            opt_type = "call" if signal == 1 else "put"
            if any(p["ticker"] == ticker and p["type"] == opt_type for p in positions):
                continue
            if options_exposure >= max_total_exp:
                continue

            df  = ticker_data[ticker]
            idx = ticker_idx[ticker].get(dt)
            if idx is None:
                continue
            price = float(df["Close"].iloc[idx])
            if price <= 0:
                continue

            max_premium = min(portfolio_val * MAX_TRADE_EXPOSURE_PCT, max_total_exp - options_exposure)
            if cash < max_premium or max_premium <= 0:
                continue

            vol         = get_hist_vol(df, idx)
            model_price = bs_call(price, round(price), OPTION_DTE / 365, RISK_FREE_RATE, vol) \
                          if opt_type == "call" \
                          else bs_put(price, round(price), OPTION_DTE / 365, RISK_FREE_RATE, vol)
            entry_price = model_price + BID_ASK_SPREAD
            if entry_price <= 0.01:
                continue

            contracts  = max(1, int(max_premium / (entry_price * 100)))
            premium    = entry_price * contracts * 100
            commission = contracts * COMMISSION_PER_CONTRACT
            if cash < premium + commission:
                continue

            cash             -= premium + commission
            options_exposure += premium
            positions.append({
                "ticker": ticker, "type": opt_type, "strike": round(price),
                "contracts": contracts, "premium_paid": premium, "current_value": premium,
                "entry_date": dt, "expiry": dt + timedelta(days=OPTION_DTE),
                "portfolio_at_entry": portfolio_val,
            })

    # Close remaining positions
    for pos in positions:
        ticker  = pos["ticker"]
        df      = ticker_data[ticker]
        price   = float(df["Close"].iloc[-1])
        last_dt = df.index[-1]
        T       = max((pos["expiry"] - last_dt).total_seconds() / (365 * 24 * 3600), 0)
        vol     = get_hist_vol(df, len(df) - 1)
        raw     = bs_call(price, pos["strike"], T, RISK_FREE_RATE, vol) if pos["type"] == "call" \
                  else bs_put(price, pos["strike"], T, RISK_FREE_RATE, vol)
        payout  = max(raw - BID_ASK_SPREAD, 0) * pos["contracts"] * 100
        cash   += payout
        trades_log.append({"ticker": ticker, "type": pos["type"],
                           "entry_date": pos["entry_date"], "exit_date": last_dt,
                           "pnl": payout - pos["premium_paid"],
                           "premium_paid": pos["premium_paid"],
                           "portfolio_at_entry": pos.get("portfolio_at_entry", INITIAL_CASH),
                           "exit_reason": "end_of_backtest"})

    eq_df  = pd.DataFrame(equity_curve).set_index("datetime")
    peak   = eq_df["portfolio_value"].cummax()
    max_dd = round(((eq_df["portfolio_value"] - peak) / peak * 100).min(), 2)
    log    = pd.DataFrame(trades_log)
    wins   = len(log[log["pnl"] > 0]) if not log.empty else 0

    return {
        "final_value":  cash,
        "total_return": (cash - initial_cash) / initial_cash * 100,
        "total_trades": len(trades_log),
        "win_rate":     wins / len(log) * 100 if not log.empty else 0,
        "max_drawdown": max_dd,
        "trades_log":   log,
        "equity_curve": eq_df,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def monthly_report(equity_curve: pd.DataFrame, trades_log: pd.DataFrame, initial: float = INITIAL_CASH):
    eq  = equity_curve.copy()
    eq.index = pd.to_datetime(eq.index)
    mv  = eq["portfolio_value"].resample("ME").last().ffill()
    prev = mv.shift(1).fillna(initial)
    report = pd.DataFrame({
        "End Value ($)": mv.round(2),
        "Month P&L ($)": (mv - prev).round(2),
        "Return (%)":    ((mv / prev - 1) * 100).round(2),
    })
    report.index = report.index.strftime("%Y-%m")

    if not trades_log.empty:
        tlog = trades_log.copy()
        tlog["exit_date"] = pd.to_datetime(tlog["exit_date"]).dt.tz_localize(None)
        tlog["month"]     = tlog["exit_date"].values.astype("datetime64[M]")
        mt = tlog.groupby("month").agg(trades=("pnl","count"), wins=("pnl", lambda x: (x>0).sum()))
        mt["win_rate"] = (mt["wins"] / mt["trades"] * 100).round(1)
        mt.index = mt.index.strftime("%Y-%m")
        report   = report.join(mt[["trades","win_rate"]], how="left")
        report.rename(columns={"trades":"# Trades","win_rate":"Win%"}, inplace=True)
        report["# Trades"] = report["# Trades"].fillna(0).astype(int)
        report["Win%"]     = report["Win%"].fillna(0)

    return report


def print_results(results: dict, label: str = "FULL BACKTEST"):
    r = results
    ic = r["final_value"] - r["total_return"] / 100 * r["final_value"] / (1 + r["total_return"] / 100)
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Final Value    : ${r['final_value']:>12,.2f}")
    print(f"  Total Return   : {r['total_return']:>+11.2f}%")
    print(f"  Max Drawdown   : {r['max_drawdown']:>11.2f}%")
    print(f"  Total Trades   : {r['total_trades']:>12}")
    print(f"  Win Rate       : {r['win_rate']:>11.2f}%")
    print(f"{'='*55}")
    mr = monthly_report(r["equity_curve"], r["trades_log"])
    print(mr.to_string())


def main():
    import os
    os.makedirs("results", exist_ok=True)
    ticker_data = fetch_all(SP500_TICKERS)
    indicators  = {t: compute_indicators(df) for t, df in ticker_data.items()}
    results     = run_portfolio_backtest(ticker_data, indicators)
    print_results(results)
    results["equity_curve"].to_csv("results/equity_curve.csv")
    results["trades_log"].to_csv("results/trades_log.csv", index=False)
    monthly_report(results["equity_curve"], results["trades_log"]).to_csv("results/monthly_report.csv")
    print("\nSaved results/")


if __name__ == "__main__":
    main()
