"""
Strategy Engine — Adaptive Options Strategy Selector
─────────────────────────────────────────────────────
Analyzes market conditions per ticker and selects the optimal options strategy.

Decision inputs:
  1. Direction   — bullish / bearish / neutral  (from signal indicators)
  2. Strength    — strong / moderate / weak      (how many signals agree)
  3. IV Rank     — high / medium / low           (current IV vs 1-year range)
  4. IV vs HV    — overpriced / fair / cheap     (implied vs realized vol)

Strategy menu:
  ┌──────────────────────┬───────────┬────────┬───────────────────────────────┐
  │ Strategy             │ Direction │ IV     │ When to use                   │
  ├──────────────────────┼───────────┼────────┼───────────────────────────────┤
  │ Buy call             │ Bullish   │ Low    │ Cheap options, strong trend   │
  │ Buy put              │ Bearish   │ Low    │ Cheap options, strong trend   │
  │ Bull put spread      │ Bullish   │ High   │ Sell rich premium, defined ↓  │
  │ Bear call spread     │ Bearish   │ High   │ Sell rich premium, defined ↓  │
  │ Iron condor          │ Neutral   │ High   │ Range-bound, sell both sides  │
  │ Buy straddle         │ Volatile  │ Low    │ Expect big move, cheap vol    │
  │ Calendar spread      │ Neutral   │ Mixed  │ Near-term IV > long-term      │
  │ Put-call parity arb  │ Any       │ Any    │ Mispricing detected           │
  └──────────────────────┴───────────┴────────┴───────────────────────────────┘
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from options_pricer import bs_call, bs_put, historical_vol


# ── Strategy types ───────────────────────────────────────────────────────────

STRATEGY_BUY_CALL       = "buy_call"
STRATEGY_BUY_PUT        = "buy_put"
STRATEGY_BULL_PUT_SPREAD = "bull_put_spread"
STRATEGY_BEAR_CALL_SPREAD = "bear_call_spread"
STRATEGY_IRON_CONDOR    = "iron_condor"
STRATEGY_BUY_STRADDLE   = "buy_straddle"
STRATEGY_CALENDAR_SPREAD = "calendar_spread"
STRATEGY_PUT_CALL_ARB   = "put_call_arb"


@dataclass
class MarketRegime:
    """Snapshot of market conditions for a single ticker."""
    ticker: str
    price: float
    direction: str          # "bullish", "bearish", "neutral"
    strength: int           # number of agreeing signals (0-4)
    iv_rank: float          # 0-100 percentile
    iv_current: float       # current implied vol estimate
    hv_current: float       # current realized (historical) vol
    iv_hv_ratio: float      # iv / hv — >1 means IV is overpriced
    rsi: float
    signal_reasons: list = field(default_factory=list)


@dataclass
class StrategySelection:
    """The chosen strategy and its parameters."""
    strategy: str           # one of STRATEGY_* constants
    ticker: str
    direction: str
    price: float
    # Leg details
    strike_sell: Optional[float] = None     # strike to sell (spreads/condors)
    strike_buy: Optional[float] = None      # strike to buy (protection leg)
    strike_sell_2: Optional[float] = None   # second sell strike (condor)
    strike_buy_2: Optional[float] = None    # second buy strike (condor)
    right: Optional[str] = None             # "C" or "P" (single-leg)
    dte_near: int = 21                      # days to expiry (near leg)
    dte_far: int = 45                       # days to expiry (far leg, calendar)
    contracts: int = 1
    estimated_credit: float = 0.0           # premium collected (sells)
    estimated_debit: float = 0.0            # premium paid (buys)
    max_loss: float = 0.0
    max_profit: float = 0.0
    score: float = 0.0                      # strategy quality score
    reason: str = ""                        # human-readable explanation


# ── Regime detection ─────────────────────────────────────────────────────────

def compute_iv_rank(hv_series: pd.Series, current_iv: float) -> float:
    """
    IV rank: where current IV sits within the 1-year range.
    0 = at the low, 100 = at the high.
    Uses historical vol as IV proxy (since we don't have live IV surface).
    """
    if len(hv_series) < 20:
        return 50.0
    iv_min = hv_series.min()
    iv_max = hv_series.max()
    if iv_max == iv_min:
        return 50.0
    return float(np.clip((current_iv - iv_min) / (iv_max - iv_min) * 100, 0, 100))


def compute_rolling_hv(close: pd.Series, windows: list = None) -> pd.Series:
    """Compute rolling historical volatility (annualized) over multiple windows."""
    if windows is None:
        windows = [20]
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(windows[0]).std() * np.sqrt(252)


def detect_regime(
    df: pd.DataFrame,
    signal_result: dict,
    iv_premium_mult: float = 1.25,
) -> MarketRegime:
    """
    Analyze a ticker's data and signals to determine market regime.

    Parameters
    ----------
    df              : OHLCV DataFrame (daily bars, >= 252 rows ideal)
    signal_result   : output from compute_options_signal() or compute_equity_signal()
    iv_premium_mult : multiplier for HV -> IV estimate
    """
    close = df["Close"].squeeze()
    price = float(close.iloc[-1])

    # Historical vol (current)
    hv_current = historical_vol(close.values, window=20)

    # Rolling HV series for IV rank calculation
    hv_series = compute_rolling_hv(close, windows=[20]).dropna()

    # Estimate current IV (HV * premium multiplier)
    iv_current = hv_current * iv_premium_mult

    # IV rank over available history
    iv_rank = compute_iv_rank(hv_series, iv_current)

    # IV / HV ratio
    iv_hv_ratio = iv_current / hv_current if hv_current > 0 else 1.0

    # Direction and strength from signals
    signal = signal_result.get("signal")
    reason = signal_result.get("reason", "") or ""
    reasons = reason.split("+") if reason else []
    strength = len(reasons)

    if signal in ("BUY", "CALL"):
        direction = "bullish"
    elif signal in ("SELL", "PUT"):
        direction = "bearish"
    else:
        direction = "neutral"

    rsi = signal_result.get("rsi", 50.0)

    return MarketRegime(
        ticker=df.attrs.get("ticker", "???"),
        price=price,
        direction=direction,
        strength=strength,
        iv_rank=iv_rank,
        iv_current=iv_current,
        hv_current=hv_current,
        iv_hv_ratio=iv_hv_ratio,
        rsi=rsi,
        signal_reasons=reasons,
    )


# ── Put-call parity check ───────────────────────────────────────────────────

def check_put_call_parity(
    price: float,
    strike: float,
    call_mid: float,
    put_mid: float,
    T: float,
    r: float = 0.05,
    threshold: float = 0.30,
) -> Optional[dict]:
    """
    Check for put-call parity violations: C - P = S - K*e^(-rT)
    Returns arbitrage details if |deviation| > threshold, else None.
    """
    pv_strike = strike * np.exp(-r * T)
    theoretical_diff = price - pv_strike
    actual_diff = call_mid - put_mid
    deviation = actual_diff - theoretical_diff

    if abs(deviation) > threshold:
        if deviation > 0:
            # Call overpriced: sell call, buy put, buy stock
            return {
                "type": "sell_call_buy_put",
                "deviation": deviation,
                "action": f"Sell call + buy put + buy stock (dev=${deviation:.2f})",
            }
        else:
            # Put overpriced: buy call, sell put, short stock
            return {
                "type": "buy_call_sell_put",
                "deviation": abs(deviation),
                "action": f"Buy call + sell put + short stock (dev=${abs(deviation):.2f})",
            }
    return None


# ── Strategy scoring & selection ─────────────────────────────────────────────

def _score_buy_call(regime: MarketRegime) -> float:
    """Score buying a call: best when bullish + IV is cheap."""
    if regime.direction != "bullish":
        return 0.0
    score = 40.0
    # Stronger signal = better
    score += regime.strength * 10
    # Low IV = cheap options = better for buying
    score += max(0, (50 - regime.iv_rank)) * 0.6
    # Low IV/HV ratio = underpriced
    if regime.iv_hv_ratio < 1.1:
        score += 15
    return score


def _score_buy_put(regime: MarketRegime) -> float:
    """Score buying a put: best when bearish + IV is cheap."""
    if regime.direction != "bearish":
        return 0.0
    score = 40.0
    score += regime.strength * 10
    score += max(0, (50 - regime.iv_rank)) * 0.6
    if regime.iv_hv_ratio < 1.1:
        score += 15
    return score


def _score_bull_put_spread(regime: MarketRegime) -> float:
    """Score selling a bull put spread: best when mildly bullish + IV is high."""
    if regime.direction == "bearish":
        return 0.0
    score = 30.0
    # Bullish gets a boost, neutral is okay too
    if regime.direction == "bullish":
        score += 15
    # High IV = rich premium to sell
    score += max(0, (regime.iv_rank - 40)) * 0.6
    # High IV/HV = overpriced vol = good for selling
    if regime.iv_hv_ratio > 1.2:
        score += 15
    # Moderate strength is fine (don't need strong conviction to sell)
    score += min(regime.strength, 2) * 5
    return score


def _score_bear_call_spread(regime: MarketRegime) -> float:
    """Score selling a bear call spread: best when mildly bearish + IV is high."""
    if regime.direction == "bullish":
        return 0.0
    score = 30.0
    if regime.direction == "bearish":
        score += 15
    score += max(0, (regime.iv_rank - 40)) * 0.6
    if regime.iv_hv_ratio > 1.2:
        score += 15
    score += min(regime.strength, 2) * 5
    return score


def _score_iron_condor(regime: MarketRegime) -> float:
    """Score iron condor: best when neutral/range-bound + IV is high."""
    # Penalize strong directional signals
    if regime.strength >= 3:
        return 0.0
    score = 25.0
    # Neutral is best
    if regime.direction == "neutral":
        score += 20
    # High IV is essential for condors
    score += max(0, (regime.iv_rank - 50)) * 0.8
    if regime.iv_hv_ratio > 1.3:
        score += 15
    # Low RSI extremes (range-bound market)
    if 35 < regime.rsi < 65:
        score += 10
    return score


def _score_buy_straddle(regime: MarketRegime) -> float:
    """Score buying a straddle: best when expecting big move + IV is cheap."""
    # Good when signals conflict or multiple strong signals from different directions
    score = 20.0
    # Low IV = cheap to buy both sides
    score += max(0, (40 - regime.iv_rank)) * 0.8
    if regime.iv_hv_ratio < 0.9:
        score += 20
    # RSI near extremes suggests potential reversal / big move
    if regime.rsi < 25 or regime.rsi > 75:
        score += 15
    return score


def _score_calendar_spread(regime: MarketRegime) -> float:
    """Score calendar spread: best when neutral + near-term IV elevated."""
    if regime.strength >= 3:
        return 0.0
    score = 20.0
    if regime.direction == "neutral":
        score += 15
    # Medium-high IV benefits calendar spreads
    if 40 < regime.iv_rank < 75:
        score += 15
    if 40 < regime.rsi < 60:
        score += 10
    return score


STRATEGY_SCORERS = {
    STRATEGY_BUY_CALL:        _score_buy_call,
    STRATEGY_BUY_PUT:         _score_buy_put,
    STRATEGY_BULL_PUT_SPREAD: _score_bull_put_spread,
    STRATEGY_BEAR_CALL_SPREAD: _score_bear_call_spread,
    STRATEGY_IRON_CONDOR:     _score_iron_condor,
    STRATEGY_BUY_STRADDLE:    _score_buy_straddle,
    STRATEGY_CALENDAR_SPREAD: _score_calendar_spread,
}


def select_strategy(
    regime: MarketRegime,
    account_value: float,
    max_trade_pct: float = 0.02,
    spread_width: float = 5.0,
    condor_wing_width: float = 5.0,
    dte_short: int = 21,
    dte_long: int = 45,
    r: float = 0.05,
    min_score: float = 35.0,
) -> Optional[StrategySelection]:
    """
    Score all strategies and return the highest-scoring one.
    Returns None if no strategy scores above min_score.
    """
    scores = {}
    for name, scorer in STRATEGY_SCORERS.items():
        scores[name] = scorer(regime)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_name, best_score = ranked[0]

    if best_score < min_score:
        return None

    price = regime.price
    iv = regime.iv_current
    T_short = dte_short / 365.0
    T_long = dte_long / 365.0
    max_premium = account_value * max_trade_pct

    sel = StrategySelection(
        strategy=best_name,
        ticker=regime.ticker,
        direction=regime.direction,
        price=price,
        score=best_score,
    )

    # ── Fill in strategy-specific parameters ──

    if best_name == STRATEGY_BUY_CALL:
        strike = round(price)
        premium = bs_call(price, strike, T_short, r, iv)
        sel.strike_buy = strike
        sel.right = "C"
        sel.dte_near = dte_short
        sel.estimated_debit = round(premium, 2)
        sel.max_loss = round(premium, 2)
        sel.max_profit = round(premium * 3, 2)  # estimate
        sel.contracts = max(1, int(max_premium / (premium * 100))) if premium > 0.01 else 0
        sel.reason = f"Bullish + low IV (rank={regime.iv_rank:.0f}) -> buy call"

    elif best_name == STRATEGY_BUY_PUT:
        strike = round(price)
        premium = bs_put(price, strike, T_short, r, iv)
        sel.strike_buy = strike
        sel.right = "P"
        sel.dte_near = dte_short
        sel.estimated_debit = round(premium, 2)
        sel.max_loss = round(premium, 2)
        sel.max_profit = round(premium * 3, 2)
        sel.contracts = max(1, int(max_premium / (premium * 100))) if premium > 0.01 else 0
        sel.reason = f"Bearish + low IV (rank={regime.iv_rank:.0f}) -> buy put"

    elif best_name == STRATEGY_BULL_PUT_SPREAD:
        sell_strike = round(price - price * 0.02)   # slightly OTM put sell
        buy_strike = sell_strike - spread_width       # protection leg
        credit = bs_put(price, sell_strike, T_short, r, iv) - \
                 bs_put(price, buy_strike, T_short, r, iv)
        credit = max(credit, 0.05)
        sel.strike_sell = sell_strike
        sel.strike_buy = buy_strike
        sel.right = "P"
        sel.dte_near = dte_short
        sel.estimated_credit = round(credit, 2)
        sel.max_loss = round(spread_width - credit, 2)
        sel.max_profit = round(credit, 2)
        sel.contracts = max(1, int(max_premium / (sel.max_loss * 100))) if sel.max_loss > 0 else 1
        sel.reason = f"Bullish + high IV (rank={regime.iv_rank:.0f}) -> sell put spread"

    elif best_name == STRATEGY_BEAR_CALL_SPREAD:
        sell_strike = round(price + price * 0.02)   # slightly OTM call sell
        buy_strike = sell_strike + spread_width       # protection leg
        credit = bs_call(price, sell_strike, T_short, r, iv) - \
                 bs_call(price, buy_strike, T_short, r, iv)
        credit = max(credit, 0.05)
        sel.strike_sell = sell_strike
        sel.strike_buy = buy_strike
        sel.right = "C"
        sel.dte_near = dte_short
        sel.estimated_credit = round(credit, 2)
        sel.max_loss = round(spread_width - credit, 2)
        sel.max_profit = round(credit, 2)
        sel.contracts = max(1, int(max_premium / (sel.max_loss * 100))) if sel.max_loss > 0 else 1
        sel.reason = f"Bearish + high IV (rank={regime.iv_rank:.0f}) -> sell call spread"

    elif best_name == STRATEGY_IRON_CONDOR:
        # Sell puts below, sell calls above, buy wings for protection
        put_sell = round(price - price * 0.03)
        put_buy = put_sell - condor_wing_width
        call_sell = round(price + price * 0.03)
        call_buy = call_sell + condor_wing_width
        put_credit = bs_put(price, put_sell, T_short, r, iv) - \
                     bs_put(price, put_buy, T_short, r, iv)
        call_credit = bs_call(price, call_sell, T_short, r, iv) - \
                      bs_call(price, call_buy, T_short, r, iv)
        total_credit = max(put_credit + call_credit, 0.10)
        sel.strike_sell = put_sell
        sel.strike_buy = put_buy
        sel.strike_sell_2 = call_sell
        sel.strike_buy_2 = call_buy
        sel.dte_near = dte_short
        sel.estimated_credit = round(total_credit, 2)
        sel.max_loss = round(condor_wing_width - total_credit, 2)
        sel.max_profit = round(total_credit, 2)
        sel.contracts = max(1, int(max_premium / (sel.max_loss * 100))) if sel.max_loss > 0 else 1
        sel.reason = f"Neutral + high IV (rank={regime.iv_rank:.0f}) -> iron condor"

    elif best_name == STRATEGY_BUY_STRADDLE:
        strike = round(price)
        call_prem = bs_call(price, strike, T_short, r, iv)
        put_prem = bs_put(price, strike, T_short, r, iv)
        total_debit = call_prem + put_prem
        sel.strike_buy = strike
        sel.dte_near = dte_short
        sel.estimated_debit = round(total_debit, 2)
        sel.max_loss = round(total_debit, 2)
        sel.max_profit = round(total_debit * 3, 2)  # estimate
        sel.contracts = max(1, int(max_premium / (total_debit * 100))) if total_debit > 0.01 else 0
        sel.reason = f"Expecting big move + cheap IV (rank={regime.iv_rank:.0f}) -> buy straddle"

    elif best_name == STRATEGY_CALENDAR_SPREAD:
        strike = round(price)
        near_prem = bs_call(price, strike, T_short, r, iv)
        far_prem = bs_call(price, strike, T_long, r, iv)
        debit = far_prem - near_prem  # pay for far, collect for near
        debit = max(debit, 0.10)
        sel.strike_buy = strike
        sel.strike_sell = strike  # same strike, different expiry
        sel.dte_near = dte_short
        sel.dte_far = dte_long
        sel.estimated_debit = round(debit, 2)
        sel.max_loss = round(debit, 2)
        sel.max_profit = round(debit * 1.5, 2)  # estimate
        sel.contracts = max(1, int(max_premium / (debit * 100))) if debit > 0.01 else 0
        sel.reason = f"Neutral + medium IV (rank={regime.iv_rank:.0f}) -> calendar spread"

    # Skip if zero contracts
    if sel.contracts <= 0:
        return None

    return sel


def rank_all_strategies(regime: MarketRegime) -> list:
    """Return all strategies ranked by score (for logging/debugging)."""
    results = []
    for name, scorer in STRATEGY_SCORERS.items():
        score = scorer(regime)
        results.append({"strategy": name, "score": round(score, 1)})
    return sorted(results, key=lambda x: -x["score"])
