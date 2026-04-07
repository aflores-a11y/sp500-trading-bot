# ─── Interactive Brokers connection ───────────────────────────────────────────
IB_HOST      = "127.0.0.1"
IB_PORT      = 4002        # IB Gateway paper trading (TWS paper = 7497, TWS live = 7496)
IB_CLIENT_ID = 1           # Unique client ID — change if running multiple bots

# ─── Universe ─────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "LLY", "AVGO", "TSLA", "JPM",
]

# ─── Strategy parameters (must match strategy.py) ─────────────────────────────
SMA_FAST         = 20
SMA_SLOW         = 50
RSI_PERIOD       = 14
RSI_ENTRY_MAX    = 65    # RSI must be below this to enter (crossover signal)
RSI_OVERBOUGHT   = 75    # RSI above this triggers exit
STOP_LOSS_PCT    = 0.05  # 5 % hard stop

# ─── Additional equity signal parameters ──────────────────────────────────────
# RSI mean-reversion
RSI_OVERSOLD_ENTRY   = 30    # buy when RSI drops below this (oversold bounce)
RSI_OVERSOLD_EXIT    = 50    # exit RSI-oversold trade when RSI recovers above this

# Bollinger Band
BB_PERIOD        = 20
BB_STD           = 2.0
BB_RSI_MAX       = 45    # RSI must be below this to confirm BB lower-band entry

# MACD
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
MACD_RSI_MAX     = 60    # RSI must be below this to confirm MACD entry

# ─── Equity take-profit & trailing stop ───────────────────────────────────────
TAKE_PROFIT_PCT      = 0.08   # close at 8 % gain
TRAILING_STOP_ACTIVATE = 0.03 # activate trailing stop once up 3 %
TRAILING_STOP_PCT    = 0.02   # trail 2 % below the high-water mark

# ─── Equity position sizing ───────────────────────────────────────────────────
MAX_POSITION_PCT = 0.10  # max 10 % of net liquidation per equity position

# ─── Options universe ─────────────────────────────────────────────────────────
OPTIONS_TICKERS = [
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
    "NEE",  "DUK",  "SO",   "AEP",  "EXC",  "LIN",  "ECL",  "ACN",  "ADP",
]

# ─── Options strategy parameters (matches options_backtest.py) ─────────────────
OPT_EMA_FAST        = 12
OPT_EMA_SLOW        = 26
OPT_RSI_PERIOD      = 14
OPT_RSI_OB          = 70    # overbought — required for bear signal (crossover)
OPT_RSI_OS          = 30    # oversold   — required for bull signal (crossover)
OPT_DTE             = 30    # target days-to-expiry for new contracts
OPT_CLOSE_DTE       = 5     # close position when DTE falls below this

# ─── Additional options signal parameters ─────────────────────────────────────
# RSI extreme entries (no crossover needed)
OPT_RSI_EXTREME_OB  = 80    # RSI above this -> buy put (strong overbought)
OPT_RSI_EXTREME_OS  = 20    # RSI below this -> buy call (strong oversold)

# Bollinger Band signals
OPT_BB_PERIOD       = 20
OPT_BB_STD          = 2.0

# MACD histogram reversal
OPT_MACD_FAST       = 12
OPT_MACD_SLOW       = 26
OPT_MACD_SIGNAL     = 9

# ─── Strategy engine parameters ───────────────────────────────────────────────
# IV rank thresholds (0-100 percentile over ~1 year HV)
IV_RANK_LOW          = 30     # below this = IV is cheap -> favor buying
IV_RANK_HIGH         = 60     # above this = IV is rich -> favor selling

# Spread construction
SPREAD_WIDTH         = 5.0    # $ width of vertical spreads
CONDOR_WING_WIDTH    = 5.0    # $ width of each condor wing
DTE_SHORT            = 21     # DTE for near-term legs / single-leg trades
DTE_LONG             = 45     # DTE for far-term leg (calendar spreads)

# Strategy selection
MIN_STRATEGY_SCORE   = 35.0   # minimum score to open any trade

# ─── Options position sizing ───────────────────────────────────────────────────
OPT_MAX_TRADE_PCT   = 0.02   # max 2 % of account per options trade
OPT_MAX_TOTAL_PCT   = 1.00   # max 100 % of account in options at once
OPT_STOP_LOSS_PCT   = 0.50   # exit when option loses 50 % of premium paid
OPT_TAKE_PROFIT_PCT = 0.20   # exit when option gains 20 % of premium paid
OPT_COMMISSION      = 0.65   # IB per-contract commission ($)
OPT_EARNINGS_DAYS   = 2      # earnings blackout window (days)
