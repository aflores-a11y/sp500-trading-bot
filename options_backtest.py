"""
Portfolio Options Backtester — 1h candles

Realistic constraints:
  - IV premium        : hist vol × 1.25
  - IV crush          : vol × 0.50 for 6 bars after earnings (collapses overnight)
  - Earnings blackout : no new entries within 2 days of known earnings
  - Earnings gap      : bid/ask spread × 3 on bars with >3% overnight price gap
  - Liquidity cost    : spread scaled by avg hourly volume tier
  - Bid/ask spread    : $0.15/share base (entry full, exit half)
  - Max total exposure: 20% of portfolio in options at once
  - Max per-trade     : 2% of portfolio
  - Stop-loss         : 50% of premium paid
  - IB commission     : $0.65/contract

Accepts optional start_dt / end_dt for walk-forward slicing.
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
COMMISSION_PER_CONTRACT = 0.65
EMA_FAST                = 12
EMA_SLOW                = 26
RSI_PERIOD              = 14
RSI_OB                  = 70
RSI_OS                  = 30

# ── Realism parameters ────────────────────────────────────────────────────────
IV_PREMIUM_MULT     = 1.25   # implied vol is ~25% above realized
IV_CRUSH_MULT       = 0.50   # vol drops 50% right after earnings
IV_CRUSH_BARS       = 6      # crush lasts ~6 hourly bars (~1 trading day)
EARNINGS_BLACKOUT   = 2      # skip new entries within N days of earnings
BASE_SPREAD         = 0.15   # base bid/ask half-spread per share
GAP_THRESHOLD       = 0.03   # overnight gap > 3% = earnings/news gap
GAP_SPREAD_MULT     = 3.0    # spread multiplier during gap bars
LIQ_HIGH_VOL        = 3_000_000   # avg hourly volume above = high liquidity
LIQ_MED_VOL         = 500_000     # avg hourly volume above = medium liquidity
LIQ_SPREAD_MED      = 1.5         # spread multiplier for medium liquidity
LIQ_SPREAD_LOW      = 2.5         # spread multiplier for low liquidity


# ── Data fetching ─────────────────────────────────────────────────────────────

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


def fetch_earnings_dates(tickers: list) -> dict:
    """
    Returns {ticker: set_of_normalized_dates} for all historical earnings.
    Gracefully skips tickers where data is unavailable.
    """
    print("Fetching earnings dates (IV crush & blackout windows)...")
    dates = {}
    for ticker in tickers:
        try:
            t  = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                normalized = set(
                    pd.to_datetime(ed.index)
                    .tz_localize(None if ed.index.tz is None else None, ambiguous="NaT")
                    .normalize()
                )
                dates[ticker] = {d for d in normalized if not pd.isna(d)}
            else:
                dates[ticker] = set()
        except Exception:
            dates[ticker] = set()
    found = sum(1 for v in dates.values() if v)
    print(f"  Earnings dates found for {found}/{len(tickers)} tickers.\n")
    return dates


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    bull = (df["ema_fast"] > df["ema_slow"]) & \
           (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & \
           (df["rsi"] < RSI_OB)
    bear = (df["ema_fast"] < df["ema_slow"]) & \
           (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & \
           (df["rsi"] > RSI_OS)

    df["signal"] = 0
    df.loc[bull, "signal"] =  1
    df.loc[bear, "signal"] = -1

    # Flag gap-open bars (>3% overnight gap) for spread widening
    df["gap"] = (df["Open"] - df["Close"].shift(1)).abs() / df["Close"].shift(1)
    return df.dropna()


# ── Vol & spread helpers ──────────────────────────────────────────────────────

def get_vol(df: pd.DataFrame, idx: int,
            ticker: str, dt,
            earnings_dates: dict,
            pos_entry_dt=None) -> float:
    """
    Historical vol with IV premium.
    Applies IV crush for IV_CRUSH_BARS bars after any earnings date
    that occurred after the position was entered (or in the last 24h).
    """
    prices   = df["Close"].iloc[max(0, idx - 30): idx + 1].values
    base_vol = max(historical_vol(prices, window=min(30, len(prices) - 1)) * IV_PREMIUM_MULT, 0.10)

    e_dates = earnings_dates.get(ticker, set())
    if not e_dates:
        return base_vol

    current_day = pd.Timestamp(dt).tz_localize(None).normalize()

    # Check if any earnings date fell within the last IV_CRUSH_BARS hours
    for ed in e_dates:
        try:
            ed_ts = pd.Timestamp(ed).normalize()
        except Exception:
            continue
        hours_since = (current_day - ed_ts).total_seconds() / 3600
        if 0 <= hours_since <= IV_CRUSH_BARS:
            # IV crush: also only apply if position was entered before earnings
            if pos_entry_dt is None or pd.Timestamp(pos_entry_dt).tz_localize(None).normalize() <= ed_ts:
                return base_vol * IV_CRUSH_MULT

    return base_vol


def get_spread(df: pd.DataFrame, idx: int) -> float:
    """
    Effective bid/ask spread per share, combining:
      - Liquidity tier (based on avg hourly volume)
      - Gap multiplier (if large overnight gap = earnings/news event)
    """
    avg_vol = float(df["Volume"].iloc[max(0, idx - 20): idx + 1].mean())

    if avg_vol >= LIQ_HIGH_VOL:
        spread = BASE_SPREAD
    elif avg_vol >= LIQ_MED_VOL:
        spread = BASE_SPREAD * LIQ_SPREAD_MED
    else:
        spread = BASE_SPREAD * LIQ_SPREAD_LOW

    # Widen spread if this is a gap-open bar
    gap = float(df["gap"].iloc[idx]) if "gap" in df.columns else 0.0
    if gap >= GAP_THRESHOLD:
        spread *= GAP_SPREAD_MULT

    return spread


def near_earnings(dt, ticker: str, earnings_dates: dict, days: int = EARNINGS_BLACKOUT) -> bool:
    """Returns True if dt is within `days` calendar days of any known earnings date."""
    e_dates = earnings_dates.get(ticker, set())
    current = pd.Timestamp(dt).tz_localize(None).normalize()
    for ed in e_dates:
        try:
            ed_ts = pd.Timestamp(ed).normalize()
        except Exception:
            continue
        if abs((current - ed_ts).days) <= days:
            return True
    return False


# ── Core backtester ───────────────────────────────────────────────────────────

def run_portfolio_backtest(
    ticker_data:    dict,
    indicators:     dict,
    earnings_dates: dict = None,
    start_dt=None,
    end_dt=None,
    initial_cash:   float = INITIAL_CASH,
    verbose:        bool  = False,
) -> dict:
    if earnings_dates is None:
        earnings_dates = {}

    all_times = sorted(set().union(*[set(df.index) for df in ticker_data.values()]))

    if start_dt:
        s = pd.Timestamp(start_dt)
        if s.tzinfo is None:
            s = s.tz_localize("America/New_York")
        all_times = [t for t in all_times if t >= s]
    if end_dt:
        e = pd.Timestamp(end_dt)
        if e.tzinfo is None:
            e = e.tz_localize("America/New_York")
        all_times = [t for t in all_times if t <= e]

    ticker_idx   = {t: {dt: i for i, dt in enumerate(df.index)} for t, df in ticker_data.items()}
    cash         = initial_cash
    positions    = []
    trades_log   = []
    equity_curve = []

    for dt in all_times:
        options_exposure = sum(p["current_value"] for p in positions)
        portfolio_val    = cash + options_exposure
        equity_curve.append({"datetime": dt, "portfolio_value": portfolio_val})

        # ── Mark-to-market open positions ─────────────────────────────────────
        for pos in positions[:]:
            ticker = pos["ticker"]
            idx    = ticker_idx[ticker].get(dt)
            if idx is None:
                continue
            df    = ticker_data[ticker]
            price = float(df["Close"].iloc[idx])
            T     = max((pos["expiry"] - dt).total_seconds() / (365 * 24 * 3600), 0)

            # Vol with IV crush applied if earnings fell after entry
            vol    = get_vol(df, idx, ticker, dt, earnings_dates, pos["entry_date"])
            spread = get_spread(df, idx)

            raw = bs_call(price, pos["strike"], T, RISK_FREE_RATE, vol) if pos["type"] == "call" \
                  else bs_put(price, pos["strike"], T, RISK_FREE_RATE, vol)

            # Mark at mid minus half spread (conservative exit assumption)
            pos["current_value"] = max(raw - spread * 0.5, 0) * pos["contracts"] * 100

            # Stop-loss
            if pos["current_value"] <= pos["premium_paid"] * (1 - STOP_LOSS_PCT):
                cash += pos["current_value"]
                trades_log.append({
                    "ticker": ticker, "type": pos["type"],
                    "entry_date": pos["entry_date"], "exit_date": dt,
                    "pnl": pos["current_value"] - pos["premium_paid"],
                    "premium_paid": pos["premium_paid"],
                    "portfolio_at_entry": pos.get("portfolio_at_entry", initial_cash),
                    "exit_reason": "stop_loss",
                })
                positions.remove(pos)
                continue

            # Expiry — pay full spread on close
            if T <= 0:
                intrinsic = max(price - pos["strike"], 0) if pos["type"] == "call" \
                            else max(pos["strike"] - price, 0)
                payout = max(intrinsic - spread, 0) * pos["contracts"] * 100
                cash  += payout
                trades_log.append({
                    "ticker": ticker, "type": pos["type"],
                    "entry_date": pos["entry_date"], "exit_date": dt,
                    "pnl": payout - pos["premium_paid"],
                    "premium_paid": pos["premium_paid"],
                    "portfolio_at_entry": pos.get("portfolio_at_entry", initial_cash),
                    "exit_reason": "expiry",
                })
                positions.remove(pos)

        # ── Entry signals ──────────────────────────────────────────────────────
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

            # ── Earnings blackout: skip entries near earnings ──────────────────
            if near_earnings(dt, ticker, earnings_dates):
                continue

            df  = ticker_data[ticker]
            idx = ticker_idx[ticker].get(dt)
            if idx is None:
                continue
            price = float(df["Close"].iloc[idx])
            if price <= 0:
                continue

            max_premium = min(portfolio_val * MAX_TRADE_EXPOSURE_PCT,
                              max_total_exp - options_exposure)
            if cash < max_premium or max_premium <= 0:
                continue

            # Vol with IV premium (no crush on entry)
            vol    = get_vol(df, idx, ticker, dt, earnings_dates)
            spread = get_spread(df, idx)

            model_price = bs_call(price, round(price), OPTION_DTE / 365, RISK_FREE_RATE, vol) \
                          if opt_type == "call" \
                          else bs_put(price, round(price), OPTION_DTE / 365, RISK_FREE_RATE, vol)

            # Pay full spread on entry
            entry_price = model_price + spread
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

    # ── Close remaining positions at last price ────────────────────────────────
    for pos in positions:
        ticker  = pos["ticker"]
        df      = ticker_data[ticker]
        last_idx = len(df) - 1
        price   = float(df["Close"].iloc[last_idx])
        last_dt = df.index[last_idx]
        T       = max((pos["expiry"] - last_dt).total_seconds() / (365 * 24 * 3600), 0)
        vol     = get_vol(df, last_idx, ticker, last_dt, earnings_dates, pos["entry_date"])
        spread  = get_spread(df, last_idx)
        raw     = bs_call(price, pos["strike"], T, RISK_FREE_RATE, vol) if pos["type"] == "call" \
                  else bs_put(price, pos["strike"], T, RISK_FREE_RATE, vol)
        payout  = max(raw - spread, 0) * pos["contracts"] * 100
        cash   += payout
        trades_log.append({
            "ticker": ticker, "type": pos["type"],
            "entry_date": pos["entry_date"], "exit_date": last_dt,
            "pnl": payout - pos["premium_paid"],
            "premium_paid": pos["premium_paid"],
            "portfolio_at_entry": pos.get("portfolio_at_entry", initial_cash),
            "exit_reason": "end_of_backtest",
        })

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

def monthly_report(equity_curve: pd.DataFrame, trades_log: pd.DataFrame,
                   initial: float = INITIAL_CASH):
    eq   = equity_curve.copy()
    eq.index = pd.to_datetime(eq.index)
    mv   = eq["portfolio_value"].resample("ME").last().ffill()
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
        mt = tlog.groupby("month").agg(
            trades=("pnl", "count"),
            wins=("pnl", lambda x: (x > 0).sum())
        )
        mt["win_rate"] = (mt["wins"] / mt["trades"] * 100).round(1)
        mt.index = mt.index.strftime("%Y-%m")
        report = report.join(mt[["trades", "win_rate"]], how="left")
        report.rename(columns={"trades": "# Trades", "win_rate": "Win%"}, inplace=True)
        report["# Trades"] = report["# Trades"].fillna(0).astype(int)
        report["Win%"]     = report["Win%"].fillna(0)

    return report


def print_results(results: dict, label: str = "BACKTEST RESULTS"):
    r = results
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Final Value    : ${r['final_value']:>12,.2f}")
    print(f"  Total Return   : {r['total_return']:>+11.2f}%")
    print(f"  Max Drawdown   : {r['max_drawdown']:>11.2f}%")
    print(f"  Total Trades   : {r['total_trades']:>12}")
    print(f"  Win Rate       : {r['win_rate']:>11.2f}%")
    print(f"{'='*55}")
    print(monthly_report(r["equity_curve"], r["trades_log"]).to_string())


def main():
    import os
    os.makedirs("results", exist_ok=True)
    ticker_data    = fetch_all(SP500_TICKERS)
    earnings_dates = fetch_earnings_dates(list(ticker_data.keys()))
    indicators     = {t: compute_indicators(df) for t, df in ticker_data.items()}
    results        = run_portfolio_backtest(ticker_data, indicators, earnings_dates)
    print_results(results)
    results["equity_curve"].to_csv("results/equity_curve.csv")
    results["trades_log"].to_csv("results/trades_log.csv", index=False)
    monthly_report(results["equity_curve"], results["trades_log"]).to_csv("results/monthly_report.csv")
    print("\nSaved to results/")


if __name__ == "__main__":
    main()
