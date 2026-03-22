from backtesting import Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import pandas as pd


class RSIMAStrategy(Strategy):
    # Parameters (tunable for optimization)
    sma_fast = 20
    sma_slow = 50
    rsi_period = 14
    rsi_overbought = 75
    rsi_entry_max = 65
    stop_loss_pct = 0.05  # 5%

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

    def next(self):
        price = self.data.Close[-1]

        # Entry: Golden Cross + RSI not overbought
        if (
            crossover(self.sma_fast_line, self.sma_slow_line)
            and self.rsi[-1] < self.rsi_entry_max
        ):
            sl = price * (1 - self.stop_loss_pct)
            self.buy(sl=sl)

        # Exit: Death Cross or RSI overbought
        elif self.position:
            if (
                crossover(self.sma_slow_line, self.sma_fast_line)
                or self.rsi[-1] > self.rsi_overbought
            ):
                self.position.close()
