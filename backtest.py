import yfinance as yf
import pandas as pd
from backtesting import Backtest
from strategy import RSIMAStrategy

# Top 10 S&P 500 stocks by market cap for testing
SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "LLY", "AVGO", "TSLA"
]

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
INITIAL_CASH = 10_000
COMMISSION = 0.0035  # IB per-share rate approximated as fraction


def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def run_backtest(ticker: str) -> dict:
    print(f"Backtesting {ticker}...")
    df = fetch_data(ticker)
    if df.empty or len(df) < 210:
        print(f"  Not enough data for {ticker}, skipping.")
        return None

    bt = Backtest(df, RSIMAStrategy, cash=INITIAL_CASH, commission=COMMISSION, exclusive_orders=True)
    stats = bt.run()

    return {
        "Ticker": ticker,
        "Return [%]": round(stats["Return [%]"], 2),
        "Buy & Hold Return [%]": round(stats["Buy & Hold Return [%]"], 2),
        "Max Drawdown [%]": round(stats["Max. Drawdown [%]"], 2),
        "# Trades": stats["# Trades"],
        "Win Rate [%]": round(stats["Win Rate [%]"], 2) if stats["# Trades"] > 0 else 0,
        "Sharpe Ratio": round(stats["Sharpe Ratio"], 3),
    }


def main():
    results = []
    for ticker in SP500_TICKERS:
        result = run_backtest(ticker)
        if result:
            results.append(result)

    df_results = pd.DataFrame(results)
    print("\n===== BACKTEST RESULTS =====")
    print(df_results.to_string(index=False))

    df_results.to_csv("results/backtest_results.csv", index=False)
    print("\nResults saved to results/backtest_results.csv")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
