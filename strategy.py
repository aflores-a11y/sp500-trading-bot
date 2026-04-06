from backtesting import Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import pandas as pd
import numpy as np


class RSIMAStrategy(Strategy):
    # SMA crossover parameters
    sma_fast = 20
    sma_slow = 50
    rsi_period = 14
    rsi_overbought = 75
    rsi_entry_max = 65
    stop_loss_pct = 0.05  # 5%

    # RSI mean-reversion
    rsi_oversold_entry = 30
    rsi_oversold_exit = 50

    # Bollinger Band
    bb_period = 20
    bb_std = 2.0
    bb_rsi_max = 45

    # MACD
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    macd_rsi_max = 60

    # Take-profit & trailing stop
    take_profit_pct = 0.08
    trailing_stop_activate = 0.03
    trailing_stop_pct = 0.02

    def init(self):
        close = pd.Series(self.data.Close)

        self.sma_fast_line = self.I(
            lambda x: ta.sma(pd.Series(x), length=self.sma_fast).values,
            self.data.Close
        )
        self.sma_slow_line = self.I(
            lambda x: ta.sma(pd.Series(x), length=self.sma_slow).values,
            self.data.Close
        )
        self.rsi = self.I(
            lambda x: ta.rsi(pd.Series(x), length=self.rsi_period).values,
            self.data.Close
        )

        # Bollinger lower band
        self.bb_lower = self.I(
            lambda x: (
                pd.Series(x).rolling(self.bb_period).mean()
                - self.bb_std * pd.Series(x).rolling(self.bb_period).std()
            ).values,
            self.data.Close,
        )

        # MACD histogram
        self.macd_hist = self.I(
            lambda x: (
                (s := pd.Series(x)),
                (ml := s.ewm(span=self.macd_fast, adjust=False).mean()
                      - s.ewm(span=self.macd_slow, adjust=False).mean()),
                ml - ml.ewm(span=self.macd_signal, adjust=False).mean(),
            )[-1].values,
            self.data.Close,
        )

        self._entry_price = None
        self._high_water = None

    def next(self):
        price = self.data.Close[-1]

        # ── Manage open position: take-profit & trailing stop ──
        if self.position:
            if self._entry_price is not None:
                if self._high_water is None or price > self._high_water:
                    self._high_water = price

                gain_pct = (price - self._entry_price) / self._entry_price

                # Take-profit
                if gain_pct >= self.take_profit_pct:
                    self.position.close()
                    self._entry_price = None
                    self._high_water = None
                    return

                # Trailing stop (once activated)
                if gain_pct >= self.trailing_stop_activate and self._high_water:
                    trail_price = self._high_water * (1 - self.trailing_stop_pct)
                    if price <= trail_price:
                        self.position.close()
                        self._entry_price = None
                        self._high_water = None
                        return

            # Signal-based exit: death cross or RSI overbought
            if (
                crossover(self.sma_slow_line, self.sma_fast_line)
                or self.rsi[-1] > self.rsi_overbought
            ):
                self.position.close()
                self._entry_price = None
                self._high_water = None
            return

        # ── Entry signals (any one triggers a buy) ──
        buy = False

        # 1. Golden Cross + RSI filter
        if (
            crossover(self.sma_fast_line, self.sma_slow_line)
            and self.rsi[-1] < self.rsi_entry_max
        ):
            buy = True

        # 2. RSI oversold bounce
        if not buy and self.rsi[-1] < self.rsi_oversold_entry:
            buy = True

        # 3. Bollinger Band lower touch + RSI confirmation
        if not buy and not np.isnan(self.bb_lower[-1]):
            if price <= self.bb_lower[-1] and self.rsi[-1] < self.bb_rsi_max:
                buy = True

        # 4. MACD histogram flip positive + RSI confirmation
        if not buy and len(self.macd_hist) >= 2:
            if (
                not np.isnan(self.macd_hist[-1])
                and not np.isnan(self.macd_hist[-2])
                and self.macd_hist[-2] <= 0
                and self.macd_hist[-1] > 0
                and self.rsi[-1] < self.macd_rsi_max
            ):
                buy = True

        if buy:
            sl = price * (1 - self.stop_loss_pct)
            self.buy(sl=sl)
            self._entry_price = price
            self._high_water = price
