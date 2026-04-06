"""
SP500 Live Trader — Paper Trading via IB Gateway
--------------------------------------------------
Runs TWO concurrent strategies each day after market close (4:05 PM ET):

  1. EQUITY strategy  (strategy.py logic)
     SMA20/50 crossover + RSI14 on daily bars.
     Trades stocks directly. 10 % of account per position. 5 % stop-loss.

  2. OPTIONS strategy  (options_backtest.py logic)
     EMA12/26 crossover + RSI14 on daily bars.
     Bull signal  → buy ATM call  (~30 DTE)
     Bear signal  → buy ATM put   (~30 DTE)
     2 % of account per trade, 20 % max total options exposure.
     50 % stop-loss on premium. Close when DTE <= 5.
     Skips entries within 2 days of earnings.
     Options positions are persisted in results/options_positions.json.

Usage:
    python live_trader.py          # waits for 4:05 PM ET each trading day
    python live_trader.py --now    # runs one cycle immediately (for testing)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
import yfinance as yf

from config import (
    # IB connection
    IB_HOST, IB_PORT, IB_CLIENT_ID,
    # Equity strategy
    TICKERS, SMA_FAST, SMA_SLOW, RSI_PERIOD,
    RSI_ENTRY_MAX, RSI_OVERBOUGHT, STOP_LOSS_PCT, MAX_POSITION_PCT,
    # Equity additional signals
    RSI_OVERSOLD_ENTRY, RSI_OVERSOLD_EXIT,
    BB_PERIOD, BB_STD, BB_RSI_MAX,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, MACD_RSI_MAX,
    # Equity take-profit & trailing stop
    TAKE_PROFIT_PCT, TRAILING_STOP_ACTIVATE, TRAILING_STOP_PCT,
    # Options strategy
    OPTIONS_TICKERS,
    OPT_EMA_FAST, OPT_EMA_SLOW, OPT_RSI_PERIOD, OPT_RSI_OB, OPT_RSI_OS,
    OPT_DTE, OPT_CLOSE_DTE,
    OPT_MAX_TRADE_PCT, OPT_MAX_TOTAL_PCT, OPT_STOP_LOSS_PCT, OPT_TAKE_PROFIT_PCT,
    OPT_COMMISSION, OPT_EARNINGS_DAYS,
    # Options additional signals
    OPT_RSI_EXTREME_OB, OPT_RSI_EXTREME_OS,
    OPT_BB_PERIOD, OPT_BB_STD,
    OPT_MACD_FAST, OPT_MACD_SLOW, OPT_MACD_SIGNAL,
)
from ib_broker import IBBroker
from options_pricer import bs_call, bs_put, historical_vol

RISK_FREE_RATE  = 0.05
IV_PREMIUM_MULT = 1.25   # implied vol ≈ 25 % above realized (matches backtest)

EQUITY_POSITIONS_FILE = "results/equity_positions.json"

# ── logging ───────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("results/live_trader.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
OPTIONS_POSITIONS_FILE = "results/options_positions.json"

# ── shared market data ────────────────────────────────────────────────────────

def fetch_daily_bars(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    df = yf.download(
        ticker, period=f"{lookback_days}d", interval="1d",
        auto_adjust=True, progress=False,
    )
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ════════════════════════════════════════════════════════════════════════════════
#  EQUITY STRATEGY
# ════════════════════════════════════════════════════════════════════════════════

def compute_equity_signal(df: pd.DataFrame) -> dict:
    """
    Multi-signal equity strategy. Generates BUY on ANY of:
      1. SMA golden cross + RSI < 65
      2. RSI oversold bounce (RSI < 30)
      3. Bollinger Band lower touch + RSI < 45
      4. MACD histogram flip positive + RSI < 60
    Generates SELL on ANY of:
      1. SMA death cross
      2. RSI > 75 (overbought)
    """
    close = df["Close"].squeeze()
    sma_f = ta.sma(close, length=SMA_FAST)
    sma_s = ta.sma(close, length=SMA_SLOW)
    rsi   = ta.rsi(close, length=RSI_PERIOD)

    # Bollinger Bands
    bb_mid = ta.sma(close, length=BB_PERIOD)
    bb_std = close.rolling(BB_PERIOD).std()
    bb_lower = bb_mid - BB_STD * bb_std if bb_mid is not None and bb_std is not None else None

    # MACD
    ema_fast_line = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow_line = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast_line - ema_slow_line
    macd_signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line

    if sma_f is None or sma_s is None or rsi is None or len(sma_f.dropna()) < 2:
        return {"signal": None}

    f_curr, f_prev = float(sma_f.iloc[-1]), float(sma_f.iloc[-2])
    s_curr, s_prev = float(sma_s.iloc[-1]), float(sma_s.iloc[-2])
    rsi_curr       = float(rsi.iloc[-1])
    price          = float(close.iloc[-1])

    # ── BUY signals (any one triggers entry) ──
    buy_signals = []

    # 1. Golden cross + RSI filter
    golden_cross = (f_prev <= s_prev) and (f_curr > s_curr)
    if golden_cross and rsi_curr < RSI_ENTRY_MAX:
        buy_signals.append("golden_cross")

    # 2. RSI oversold bounce
    if rsi_curr < RSI_OVERSOLD_ENTRY:
        buy_signals.append("rsi_oversold")

    # 3. Bollinger Band lower touch + RSI confirmation
    if bb_lower is not None and len(bb_lower.dropna()) > 0:
        bb_low_val = float(bb_lower.iloc[-1])
        if price <= bb_low_val and rsi_curr < BB_RSI_MAX:
            buy_signals.append("bb_lower")

    # 4. MACD histogram flip positive + RSI confirmation
    if len(macd_hist.dropna()) >= 2:
        hist_curr = float(macd_hist.iloc[-1])
        hist_prev = float(macd_hist.iloc[-2])
        if hist_prev <= 0 and hist_curr > 0 and rsi_curr < MACD_RSI_MAX:
            buy_signals.append("macd_flip")

    # ── SELL signals ──
    death_cross = (s_prev <= f_prev) and (s_curr > f_curr)
    sell = death_cross or rsi_curr > RSI_OVERBOUGHT

    signal = None
    reason = None
    if buy_signals:
        signal = "BUY"
        reason = "+".join(buy_signals)
    elif sell:
        signal = "SELL"
        reason = "death_cross" if death_cross else "rsi_overbought"

    return {
        "signal":     signal,
        "reason":     reason,
        "price":      price,
        "rsi":        rsi_curr,
        "sma_fast":   f_curr,
        "sma_slow":   s_curr,
        "stop_price": round(price * (1 - STOP_LOSS_PCT), 4) if signal == "BUY" else None,
    }


def calc_equity_shares(account_value: float, price: float) -> int:
    return max(1, int(account_value * MAX_POSITION_PCT / price))


def load_equity_positions() -> dict:
    """Load persisted equity position metadata (entry price, high-water mark)."""
    if os.path.exists(EQUITY_POSITIONS_FILE):
        with open(EQUITY_POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_equity_positions(positions: dict):
    """Persist equity position metadata to JSON file."""
    with open(EQUITY_POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def run_equity_cycle(broker: IBBroker, account_value: float):
    logger.info("--- EQUITY CYCLE ---")
    ib_positions = broker.get_positions()
    eq_meta = load_equity_positions()
    logger.info(f"Open equity positions: {list(ib_positions.keys()) or 'none'}")

    for ticker in TICKERS:
        try:
            df = fetch_daily_bars(ticker)
            if len(df) < SMA_SLOW + RSI_PERIOD + 5:
                continue

            sig       = compute_equity_signal(df)
            signal    = sig["signal"]
            price     = sig["price"]
            in_pos    = ticker in ib_positions and ib_positions[ticker]["qty"] > 0

            logger.info(
                f"  {ticker:>6} | ${price:>8.2f} | RSI={sig['rsi']:>5.1f} | "
                f"SMA{SMA_FAST}={sig['sma_fast']:>8.2f} | SMA{SMA_SLOW}={sig['sma_slow']:>8.2f} | "
                f"sig={sig['reason'] or '--'} | pos={'YES' if in_pos else 'no'}"
            )

            # ── Check take-profit & trailing stop for open positions ──
            if in_pos and ticker in eq_meta:
                meta = eq_meta[ticker]
                entry_price = meta["entry_price"]
                high_water  = meta.get("high_water", entry_price)

                # Update high-water mark
                if price > high_water:
                    high_water = price
                    eq_meta[ticker]["high_water"] = high_water

                gain_pct = (price - entry_price) / entry_price

                # Take-profit: close at 8% gain
                if gain_pct >= TAKE_PROFIT_PCT:
                    qty = int(ib_positions[ticker]["qty"])
                    logger.info(f"  >>> TAKE-PROFIT {ticker} | gain={gain_pct:.1%} >= {TAKE_PROFIT_PCT:.0%}")
                    broker.cancel_open_orders(ticker)
                    broker.place_market_sell(ticker, qty)
                    eq_meta.pop(ticker, None)
                    continue

                # Trailing stop: once up 3%, trail at 2% below high
                if gain_pct >= TRAILING_STOP_ACTIVATE:
                    trail_price = high_water * (1 - TRAILING_STOP_PCT)
                    if price <= trail_price:
                        qty = int(ib_positions[ticker]["qty"])
                        logger.info(
                            f"  >>> TRAILING-STOP {ticker} | price=${price:.2f} <= "
                            f"trail=${trail_price:.2f} (high=${high_water:.2f})"
                        )
                        broker.cancel_open_orders(ticker)
                        broker.place_market_sell(ticker, qty)
                        eq_meta.pop(ticker, None)
                        continue

            # ── Signal-based entry ──
            if signal == "BUY" and not in_pos:
                qty = calc_equity_shares(account_value, price)
                logger.info(f"  >>> BUY  {qty} {ticker} @ ~{price:.2f}  stop={sig['stop_price']:.2f}  reason={sig['reason']}")
                broker.cancel_open_orders(ticker)
                broker.place_market_buy(ticker, qty, stop_price=sig["stop_price"])
                eq_meta[ticker] = {
                    "entry_price": price,
                    "high_water": price,
                    "entry_date": datetime.now().isoformat(),
                    "reason": sig["reason"],
                }

            # ── Signal-based exit ──
            elif signal == "SELL" and in_pos:
                qty = int(ib_positions[ticker]["qty"])
                logger.info(f"  >>> SELL {qty} {ticker} @ ~{price:.2f}  reason={sig['reason']}")
                broker.cancel_open_orders(ticker)
                broker.place_market_sell(ticker, qty)
                eq_meta.pop(ticker, None)

        except Exception as exc:
            logger.error(f"  {ticker} equity error: {exc}", exc_info=True)

    # Clean up metadata for positions no longer held
    for ticker in list(eq_meta.keys()):
        if ticker not in ib_positions or ib_positions.get(ticker, {}).get("qty", 0) <= 0:
            eq_meta.pop(ticker, None)

    save_equity_positions(eq_meta)


# ════════════════════════════════════════════════════════════════════════════════
#  OPTIONS STRATEGY
# ════════════════════════════════════════════════════════════════════════════════

def compute_options_signal(df: pd.DataFrame) -> dict:
    """
    Multi-signal options strategy. Generates CALL on ANY of:
      1. EMA crossover bullish + RSI < 70
      2. RSI extreme oversold (< 20)
      3. Bollinger Band lower touch
      4. MACD histogram flip positive
    Generates PUT on ANY of:
      1. EMA crossover bearish + RSI > 30
      2. RSI extreme overbought (> 80)
      3. Bollinger Band upper touch
      4. MACD histogram flip negative
    """
    close = df["Close"].squeeze()

    ema_f = close.ewm(span=OPT_EMA_FAST, adjust=False).mean()
    ema_s = close.ewm(span=OPT_EMA_SLOW, adjust=False).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(OPT_RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(OPT_RSI_PERIOD).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger Bands
    bb_mid = close.rolling(OPT_BB_PERIOD).mean()
    bb_std = close.rolling(OPT_BB_PERIOD).std()
    bb_upper = bb_mid + OPT_BB_STD * bb_std
    bb_lower = bb_mid - OPT_BB_STD * bb_std

    # MACD
    macd_fast_line = close.ewm(span=OPT_MACD_FAST, adjust=False).mean()
    macd_slow_line = close.ewm(span=OPT_MACD_SLOW, adjust=False).mean()
    macd_line = macd_fast_line - macd_slow_line
    macd_signal_line = macd_line.ewm(span=OPT_MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line

    if len(ema_f.dropna()) < 2:
        return {"signal": None}

    f_curr, f_prev = float(ema_f.iloc[-1]), float(ema_f.iloc[-2])
    s_curr, s_prev = float(ema_s.iloc[-1]), float(ema_s.iloc[-2])
    rsi_curr       = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    price          = float(close.iloc[-1])

    # ── CALL signals (bullish — any one triggers) ──
    call_signals = []

    # 1. EMA crossover bullish
    if (f_prev <= s_prev) and (f_curr > s_curr) and (rsi_curr < OPT_RSI_OB):
        call_signals.append("ema_cross")

    # 2. RSI extreme oversold
    if rsi_curr < OPT_RSI_EXTREME_OS:
        call_signals.append("rsi_extreme")

    # 3. Bollinger Band lower touch
    if len(bb_lower.dropna()) > 0 and price <= float(bb_lower.iloc[-1]):
        call_signals.append("bb_lower")

    # 4. MACD histogram flip positive
    if len(macd_hist.dropna()) >= 2:
        h_curr, h_prev = float(macd_hist.iloc[-1]), float(macd_hist.iloc[-2])
        if h_prev <= 0 and h_curr > 0:
            call_signals.append("macd_flip")

    # ── PUT signals (bearish — any one triggers) ──
    put_signals = []

    # 1. EMA crossover bearish
    if (f_prev >= s_prev) and (f_curr < s_curr) and (rsi_curr > OPT_RSI_OS):
        put_signals.append("ema_cross")

    # 2. RSI extreme overbought
    if rsi_curr > OPT_RSI_EXTREME_OB:
        put_signals.append("rsi_extreme")

    # 3. Bollinger Band upper touch
    if len(bb_upper.dropna()) > 0 and price >= float(bb_upper.iloc[-1]):
        put_signals.append("bb_upper")

    # 4. MACD histogram flip negative
    if len(macd_hist.dropna()) >= 2:
        h_curr, h_prev = float(macd_hist.iloc[-1]), float(macd_hist.iloc[-2])
        if h_prev >= 0 and h_curr < 0:
            put_signals.append("macd_flip")

    signal = None
    reason = None
    if call_signals:
        signal = "CALL"
        reason = "+".join(call_signals)
    elif put_signals:
        signal = "PUT"
        reason = "+".join(put_signals)

    return {
        "signal":   signal,
        "reason":   reason,
        "price":    price,
        "rsi":      rsi_curr,
        "ema_fast": f_curr,
        "ema_slow": s_curr,
    }


def is_near_earnings(ticker: str, days: int = OPT_EARNINGS_DAYS) -> bool:
    """Return True if today is within `days` of a known earnings date."""
    try:
        t  = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return False
        today = pd.Timestamp.now(tz=None).normalize()
        for date in pd.to_datetime(ed.index).tz_localize(None).normalize():
            if abs((today - date).days) <= days:
                return True
    except Exception:
        pass
    return False


def load_options_positions() -> dict:
    """Load persisted options positions from JSON file."""
    if os.path.exists(OPTIONS_POSITIONS_FILE):
        with open(OPTIONS_POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_options_positions(positions: dict):
    """Persist options positions to JSON file."""
    with open(OPTIONS_POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def run_options_cycle(broker: IBBroker, account_value: float):
    logger.info("--- OPTIONS CYCLE ---")
    positions = load_options_positions()

    # ── Step 1: manage existing options positions ─────────────────────────────
    options_exposure = 0.0

    for key in list(positions.keys()):
        pos = positions[key]
        ticker   = pos["ticker"]
        opt_type = pos["type"]           # "call" or "put"
        right    = "C" if opt_type == "call" else "P"

        try:
            from ib_insync import Option
            contract = Option(
                ticker, pos["expiry_str"], pos["strike"], right,
                "SMART", currency="USD",
            )
            broker.ib.qualifyContracts(contract)

            mid = broker.get_option_mid_price(contract)
            if mid <= 0.01:
                # BS fallback for mark-to-market
                df_u  = fetch_daily_bars(ticker, lookback_days=60)
                prices = df_u["Close"].squeeze().values
                vol    = historical_vol(prices, window=20) * IV_PREMIUM_MULT
                days_left = (pd.Timestamp(pos["expiry_date"]) - pd.Timestamp.now()).days
                T      = max(days_left / 365.0, 1e-6)
                mid    = (bs_call(float(df_u["Close"].iloc[-1]), pos["strike"], T, RISK_FREE_RATE, vol)
                          if opt_type == "call"
                          else bs_put(float(df_u["Close"].iloc[-1]), pos["strike"], T, RISK_FREE_RATE, vol))
                mid    = round(max(mid, 0.01), 2)
            current_value = mid * pos["contracts"] * 100
            pos["current_value"] = current_value

            days_to_expiry = (
                pd.Timestamp(pos["expiry_date"]) - pd.Timestamp.now()
            ).days

            pnl_pct = (current_value - pos["premium_paid"]) / pos["premium_paid"] * 100

            logger.info(
                f"  {ticker:>6} {opt_type:<4} | value=${current_value:>7.2f} | "
                f"paid=${pos['premium_paid']:>7.2f} | P&L={pnl_pct:>+6.1f}% | DTE={days_to_expiry}"
            )

            should_close = False
            close_reason = ""

            if current_value <= pos["premium_paid"] * (1 - OPT_STOP_LOSS_PCT):
                should_close = True
                close_reason = f"stop-loss ({pnl_pct:.1f}%)"
            elif current_value >= pos["premium_paid"] * (1 + OPT_TAKE_PROFIT_PCT):
                should_close = True
                close_reason = f"take-profit ({pnl_pct:.1f}%)"
            elif days_to_expiry <= OPT_CLOSE_DTE:
                should_close = True
                close_reason = f"near expiry (DTE={days_to_expiry})"

            if should_close:
                logger.info(f"  >>> CLOSE {ticker} {opt_type} — {close_reason}")
                sell_trade = broker.place_option_order(contract, pos["contracts"], "SELL")
                if broker.wait_for_fill(sell_trade, timeout=30):
                    del positions[key]
                else:
                    logger.warning(f"  {ticker}: SELL order not filled — keeping position in JSON")
            else:
                options_exposure += current_value

        except Exception as exc:
            logger.error(f"  {ticker} options manage error: {exc}", exc_info=True)
            options_exposure += pos.get("current_value", 0)

    # ── Step 2: scan for new entry signals ────────────────────────────────────
    max_total_exp = account_value * OPT_MAX_TOTAL_PCT

    for ticker in OPTIONS_TICKERS:
        if options_exposure >= max_total_exp:
            logger.info(f"  Max options exposure reached (${options_exposure:.0f}), stopping scan")
            break

        # Skip if already holding a position in this ticker
        if f"{ticker}_call" in positions or f"{ticker}_put" in positions:
            continue

        try:
            df = fetch_daily_bars(ticker, lookback_days=80)
            if len(df) < OPT_EMA_SLOW + OPT_RSI_PERIOD + 5:
                continue

            sig    = compute_options_signal(df)
            signal = sig["signal"]
            price  = sig["price"]

            if signal is None:
                continue

            # Earnings blackout
            if is_near_earnings(ticker):
                logger.info(f"  {ticker}: near earnings — skipping")
                continue

            opt_type = signal.lower()   # "call" or "put"
            right    = "C" if signal == "CALL" else "P"
            pos_key  = f"{ticker}_{opt_type}"

            logger.info(
                f"  {ticker:>6} {opt_type:<4} SIGNAL ({sig['reason']}) | "
                f"${price:.2f} | RSI={sig['rsi']:.1f} | "
                f"EMA{OPT_EMA_FAST}={sig['ema_fast']:.2f} | EMA{OPT_EMA_SLOW}={sig['ema_slow']:.2f}"
            )

            # Find ATM contract
            contract = broker.get_atm_option_contract(ticker, right, price, target_dte=OPT_DTE)
            if contract is None:
                continue

            # Get mid price — fall back to Black-Scholes if no market data subscription
            mid = broker.get_option_mid_price(contract)
            if mid <= 0.01:
                prices = df["Close"].squeeze().values
                vol    = historical_vol(prices, window=20) * IV_PREMIUM_MULT
                T      = OPT_DTE / 365.0
                mid    = (bs_call(price, contract.strike, T, RISK_FREE_RATE, vol)
                          if signal == "CALL"
                          else bs_put(price, contract.strike, T, RISK_FREE_RATE, vol))
                mid   += 0.15   # add base bid/ask half-spread
                mid    = round(mid, 2)
                logger.info(f"  {ticker}: IB quote unavailable — BS fallback ${mid:.2f} (vol={vol:.1%})")
                if mid <= 0.01:
                    continue

            # Position sizing: 2 % of account, respect remaining headroom
            max_premium    = min(account_value * OPT_MAX_TRADE_PCT, max_total_exp - options_exposure)
            num_contracts  = max(1, int(max_premium / (mid * 100)))
            premium        = round(mid * num_contracts * 100, 2)
            commission     = round(num_contracts * OPT_COMMISSION, 2)

            if premium + commission > account_value * OPT_MAX_TRADE_PCT * 1.05:
                num_contracts = 1
                premium       = round(mid * 100, 2)
                commission    = OPT_COMMISSION

            logger.info(
                f"  >>> OPEN {opt_type.upper()} {ticker} | "
                f"strike=${contract.strike} exp={contract.lastTradeDateOrContractMonth} | "
                f"{num_contracts}x @ ${mid:.2f} | total=${premium:.2f} + ${commission:.2f} comm"
            )

            trade = broker.place_option_order(contract, num_contracts, "BUY", limit_price=mid)

            if not broker.wait_for_fill(trade, timeout=30):
                logger.warning(f"  {ticker}: BUY order not filled within 30s — not recording position")
                continue

            # Use actual fill price, not estimated mid
            avg_fill = trade.orderStatus.avgFillPrice or mid
            actual_premium = round(avg_fill * num_contracts * 100, 2)
            logger.info(f"  {ticker}: filled @ ${avg_fill:.2f} — premium=${actual_premium:.2f}")

            expiry_dt = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")
            positions[pos_key] = {
                "ticker":        ticker,
                "type":          opt_type,
                "contracts":     num_contracts,
                "premium_paid":  actual_premium,
                "current_value": actual_premium,
                "entry_date":    datetime.now().isoformat(),
                "expiry_date":   expiry_dt.strftime("%Y-%m-%d"),
                "expiry_str":    contract.lastTradeDateOrContractMonth,
                "strike":        contract.strike,
                "ib_conid":      contract.conId,
            }
            options_exposure += actual_premium

        except Exception as exc:
            logger.error(f"  {ticker} options entry error: {exc}", exc_info=True)

    save_options_positions(positions)
    open_count = len(positions)
    logger.info(f"  Options positions open: {open_count} | Exposure: ${options_exposure:,.2f}")


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN CYCLE
# ════════════════════════════════════════════════════════════════════════════════

def run_cycle(broker: IBBroker):
    logger.info("==========================================")
    logger.info("  Trading cycle started")
    logger.info("==========================================")

    account_value = broker.get_account_value()
    logger.info(f"Net liquidation: ${account_value:,.2f}")

    run_equity_cycle(broker, account_value)
    run_options_cycle(broker, account_value)

    logger.info("==========================================")
    logger.info("  Trading cycle complete")
    logger.info("==========================================\n")


# ── scheduling ────────────────────────────────────────────────────────────────

MARKET_OPEN_TIME  = dtime(9, 35)   # 5 min after open so data settles
MARKET_CLOSE_TIME = dtime(16, 5)   # 5 min after close for final bar
LOOP_SLEEP_SECS   = 60             # pause between cycles to avoid API hammering


def secs_until_next_run() -> float:
    """
    Returns seconds to wait before the next cycle should fire.
    Returns 0 if we are currently inside market hours and due to run.
    """
    now = datetime.now(ET)

    # ── weekend: wait until Monday 9:35 AM ──
    if now.weekday() >= 5:
        days = 7 - now.weekday()   # Sat → 2, Sun → 1
        nxt  = (now + pd.Timedelta(days=days)).replace(
            hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute,
            second=0, microsecond=0,
        )
        return (nxt - now).total_seconds()

    open_today  = now.replace(hour=MARKET_OPEN_TIME.hour,  minute=MARKET_OPEN_TIME.minute,  second=0, microsecond=0)
    close_today = now.replace(hour=MARKET_CLOSE_TIME.hour, minute=MARKET_CLOSE_TIME.minute, second=0, microsecond=0)

    # ── before market open: wait until 9:35 AM ──
    if now < open_today:
        return (open_today - now).total_seconds()

    # ── after market close: wait until next trading day's 9:35 AM ──
    if now > close_today:
        days = 3 if now.weekday() == 4 else 1   # Friday → Monday
        nxt  = (now + pd.Timedelta(days=days)).replace(
            hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute,
            second=0, microsecond=0,
        )
        return (nxt - now).total_seconds()

    # ── inside market hours: run now ──
    return 0.0


def wait_if_needed():
    """Block until market hours, logging progress every 5 minutes."""
    while True:
        secs = secs_until_next_run()
        if secs == 0:
            return
        logger.info(f"Market closed — sleeping {secs / 60:.1f} min until next session ...")
        time.sleep(min(secs, 300))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SP500 Live Trader — Equity + Options (IB Paper)")
    parser.add_argument("--now", action="store_true",
                        help="Run one cycle immediately (skip waiting for market close)")
    args = parser.parse_args()

    logger.info("SP500 Trading Bot — Paper Trading Mode (Equity + Options)")
    logger.info(f"Equity tickers  : {TICKERS}")
    logger.info(f"Options universe: {len(OPTIONS_TICKERS)} tickers")
    logger.info(f"IB host         : {IB_HOST}:{IB_PORT}  clientId={IB_CLIENT_ID}")

    broker = IBBroker(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID)
    broker.connect()

    try:
        if args.now:
            run_cycle(broker)
        else:
            while True:
                wait_if_needed()   # blocks outside market hours (9:35 AM–4:05 PM ET)
                try:
                    if not broker.is_connected():
                        logger.warning("Not connected before cycle — reconnecting ...")
                        broker.reconnect()
                    run_cycle(broker)
                except (ConnectionError, RuntimeError, OSError) as exc:
                    logger.error(f"Connection lost during cycle: {exc} — attempting reconnect ...")
                    try:
                        broker.reconnect()
                        logger.info("Reconnected — will retry on next cycle")
                    except ConnectionError as reconn_exc:
                        logger.critical(f"Reconnect failed: {reconn_exc} — sleeping 5 min before retry")
                        time.sleep(300)
                time.sleep(LOOP_SLEEP_SECS)   # 60 s between cycles during market hours
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
