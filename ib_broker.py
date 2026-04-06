"""
IB TWS Paper-Trading Broker
----------------------------
Wraps ib_insync to provide a simple interface for the live trader:
  - connect / disconnect
  - read account value and open positions
  - place equity market orders with an optional stop-loss
  - place options limit/market orders
  - find the nearest ATM option contract via IB option chain
  - cancel open orders for a symbol
"""

import logging
from datetime import datetime, timedelta

from ib_insync import IB, LimitOrder, MarketOrder, Option, Stock, StopOrder

logger = logging.getLogger(__name__)


class IBBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self):
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        logger.info(f"Connected to IB TWS at {self.host}:{self.port} (clientId={self.client_id})")

    def disconnect(self):
        self.ib.disconnect()
        logger.info("Disconnected from IB TWS")

    def reconnect(self, retries: int = 5, delay: float = 10.0):
        """Disconnect and reconnect to IB TWS, retrying up to `retries` times."""
        import time as _time
        try:
            self.ib.disconnect()
        except Exception:
            pass
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Reconnect attempt {attempt}/{retries} ...")
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                logger.info("Reconnected to IB TWS")
                return
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")
                if attempt < retries:
                    _time.sleep(delay)
        raise ConnectionError(f"Could not reconnect to IB TWS after {retries} attempts")

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    # ── account information ───────────────────────────────────────────────────

    def get_account_value(self) -> float:
        """Return net liquidation value in USD."""
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        raise RuntimeError("Could not retrieve NetLiquidation from IB account")

    def get_positions(self) -> dict:
        """Return {symbol: {"qty": float, "avg_cost": float}} for all open positions."""
        positions = {}
        for pos in self.ib.positions():
            if pos.position != 0:
                positions[pos.contract.symbol] = {
                    "qty": pos.position,
                    "avg_cost": pos.avgCost,
                }
        return positions

    # ── order management ──────────────────────────────────────────────────────

    def _qualify(self, ticker: str) -> Stock:
        contract = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        return contract

    def place_market_buy(self, ticker: str, qty: int, stop_price: float = None):
        """
        Place a market BUY order.
        If stop_price is given, a separate stop-loss SELL order is also submitted.
        Returns the primary Trade object.
        """
        contract = self._qualify(ticker)

        buy_order = MarketOrder("BUY", qty)
        trade = self.ib.placeOrder(contract, buy_order)
        logger.info(f"[ORDER] BUY  {qty:>5} {ticker} @ MKT")

        if stop_price:
            stop_order = StopOrder("SELL", qty, round(stop_price, 2))
            self.ib.placeOrder(contract, stop_order)
            logger.info(f"[ORDER] STOP {qty:>5} {ticker} @ {stop_price:.2f}")

        return trade

    def place_market_sell(self, ticker: str, qty: int):
        """Place a market SELL order to close a long position."""
        contract = self._qualify(ticker)
        order = MarketOrder("SELL", qty)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"[ORDER] SELL {qty:>5} {ticker} @ MKT")
        return trade

    def cancel_open_orders(self, ticker: str):
        """Cancel all open orders for a given ticker (e.g. stale stops)."""
        for trade in self.ib.openTrades():
            if trade.contract.symbol == ticker:
                self.ib.cancelOrder(trade.order)
                logger.info(f"[CANCEL] Cancelled open order for {ticker} (orderId={trade.order.orderId})")

    # ── options ───────────────────────────────────────────────────────────────

    def get_atm_option_contract(
        self,
        symbol: str,
        right: str,
        current_price: float,
        target_dte: int = 30,
    ):
        """
        Find and qualify the nearest ATM option contract ~target_dte days out.

        Parameters
        ----------
        symbol        : ticker symbol (e.g. "AAPL")
        right         : "C" for call, "P" for put
        current_price : current underlying price (used to find ATM strike)
        target_dte    : desired days-to-expiry

        Returns the qualified Option contract, or None on failure.
        """
        stock = Stock(symbol, "SMART", "USD")
        try:
            self.ib.qualifyContracts(stock)
        except Exception as e:
            logger.warning(f"{symbol}: Could not qualify stock — {e}")
            return None

        chains = self.ib.reqSecDefOptParams(symbol, "", "STK", stock.conId)
        if not chains:
            logger.warning(f"{symbol}: No option chain returned by IB")
            return None

        # Prefer SMART exchange; fall back to first available
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        # Pick expiry closest to target_dte
        target_date = datetime.now() + timedelta(days=target_dte)
        expirations = sorted(chain.expirations)
        if not expirations:
            logger.warning(f"{symbol}: Option chain has no expirations")
            return None
        best_expiry = min(
            expirations,
            key=lambda x: abs((datetime.strptime(x, "%Y%m%d") - target_date).days),
        )

        # Pick ATM strike
        strikes = sorted(chain.strikes)
        if not strikes:
            logger.warning(f"{symbol}: Option chain has no strikes")
            return None
        best_strike = min(strikes, key=lambda x: abs(x - current_price))
        strike_idx = strikes.index(best_strike)

        # Try up to 3 expiries closest to target_dte; within each try ±2 strikes
        sorted_expiries = sorted(
            expirations,
            key=lambda x: abs((datetime.strptime(x, "%Y%m%d") - target_date).days),
        )
        for expiry in sorted_expiries[:3]:
            for offset in [0, 1, -1, 2, -2]:
                idx = strike_idx + offset
                if not (0 <= idx < len(strikes)):
                    continue
                candidate_strike = strikes[idx]
                candidate = Option(
                    symbol, expiry, candidate_strike, right, chain.exchange, currency="USD"
                )
                try:
                    self.ib.qualifyContracts(candidate)
                except Exception:
                    continue
                if not candidate.conId:
                    continue
                logger.info(
                    f"[OPT CONTRACT] {symbol} {right} strike={candidate_strike} "
                    f"exp={expiry} exchange={chain.exchange}"
                )
                return candidate

        logger.warning(f"{symbol}: Could not qualify option contract near strike {best_strike}")
        return None

    def get_option_mid_price(self, contract, timeout: float = 3.0) -> float:
        """
        Request a market-data snapshot and return the bid/ask mid price.
        Tries live data first (type 1), then falls back to delayed data (type 3).
        Returns 0.0 if no quote is available from either source.
        """
        bid = ask = None
        for mdt in (1, 3):
            self.ib.reqMarketDataType(mdt)
            ticker = self.ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
            self.ib.sleep(timeout)
            self.ib.cancelMktData(contract)

            bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else None

            if bid or ask:
                if mdt == 3:
                    logger.info(f"[OPT PRICE] Using delayed quote for {contract.symbol} "
                                f"{contract.right} ${contract.strike}")
                break

        self.ib.reqMarketDataType(1)  # restore live mode

        delayed = (mdt == 3)
        if bid and ask:
            mid = round((bid + ask) / 2, 2)
        elif ask:
            mid = round(ask, 2)
        elif bid:
            mid = round(bid, 2)
        else:
            mid = None

        if mid is not None:
            if delayed:
                # Delayed quotes can be 15+ min stale; apply a 3 % discount so the
                # resulting limit order stays within IB's NBBO price-protection band.
                mid = round(mid * 0.97, 2)
                logger.info(f"[OPT PRICE] Delayed quote discounted 3%% -> ${mid} "
                            f"({contract.symbol} {contract.right} ${contract.strike})")
            return mid
        logger.warning(f"[OPT PRICE] No quote for {contract.symbol} {contract.right} "
                       f"${contract.strike} — skipping")
        return 0.0


    def place_option_order(
        self,
        contract,
        qty: int,
        action: str,
        limit_price: float = None,
    ):
        """
        Place an options order.
        Uses a limit order at limit_price if provided, otherwise market.
        Rounds limit_price to the nearest $0.05 (standard options tick).
        """
        if limit_price and limit_price > 0:
            # Snap to nearest $0.05 tick
            limit_price = round(round(limit_price / 0.05) * 0.05, 2)
            order = LimitOrder(action, qty, limit_price)
            price_str = f"LMT ${limit_price:.2f}"
        else:
            order = MarketOrder(action, qty)
            price_str = "MKT"

        trade = self.ib.placeOrder(contract, order)
        logger.info(
            f"[OPT ORDER] {action} {qty}x {contract.symbol} {contract.right} "
            f"${contract.strike} exp={contract.lastTradeDateOrContractMonth} @ {price_str}"
        )
        return trade

    def wait_for_fill(self, trade, timeout: float = 30.0) -> bool:
        """
        Poll up to `timeout` seconds for a trade to reach Filled status.
        Returns True if filled, False if cancelled/rejected/timed-out.

        IB error 10349 ("TIF set to DAY by preset") initially marks the order
        Cancelled but then re-submits it automatically.  We ignore that first
        transient Cancelled and keep waiting as long as the order comes back
        Submitted within a short grace period.
        """
        import time as _time
        deadline = _time.monotonic() + timeout
        tif_resubmit_grace = 5.0   # seconds to wait for IB to re-submit after 10349
        cancelled_at = None

        while _time.monotonic() < deadline:
            self.ib.sleep(1)
            status = trade.orderStatus.status

            if status == "Filled":
                return True

            if status in ("Cancelled", "Inactive") and cancelled_at is None:
                # Check whether this is the transient 10349 cancellation
                tif_cancel = any(
                    e.errorCode == 10349 for e in trade.log if hasattr(e, "errorCode")
                )
                if tif_cancel:
                    cancelled_at = _time.monotonic()
                    continue   # give IB time to re-submit
                logger.warning(f"[ORDER] Order ended with status '{status}' (not filled)")
                return False

            if cancelled_at is not None:
                if status == "Submitted":
                    cancelled_at = None   # successfully re-submitted, keep waiting
                elif _time.monotonic() - cancelled_at > tif_resubmit_grace:
                    logger.warning(f"[ORDER] Order not re-submitted after TIF adjustment — giving up")
                    return False

        logger.warning(f"[ORDER] Fill timeout after {timeout:.0f}s — cancelling order")
        self.ib.cancelOrder(trade.order)
        return False
