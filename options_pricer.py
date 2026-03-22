import numpy as np
from scipy.stats import norm


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price."""
    if T <= 1e-6:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price."""
    if T <= 1e-6:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def historical_vol(prices: np.ndarray, window: int = 20, trading_periods: int = 252) -> float:
    """Annualized historical volatility from price series."""
    if len(prices) < window + 1:
        return 0.25  # fallback 25%
    log_returns = np.diff(np.log(prices[-window - 1:]))
    return float(np.std(log_returns) * np.sqrt(trading_periods))
