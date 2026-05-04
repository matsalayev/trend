"""
Trend Following Robot — Main Robot (v2.0 Smart Trend Following)

Tick loop:
1. Market data: narx, candles (primary + HTF), balans
2. Indikatorlar yangilash (EMA, ATR, ADX, Supertrend)
3. Position mavjud bo'lsa: trailing stop, partial TP, SL check
4. Circuit breaker (max drawdown)
5. Signal check (agar pozitsiya yo'q va cooldown tugagan)
6. Yangi pozitsiya ochish
7. Status log
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .api_client import BitgetAPIError, BitgetClient
from .config import RobotConfig
from .indicators import Candle
from .strategy import Position, SignalType, TrailingPhase, TrendStrategy

logger = logging.getLogger(__name__)


class RobotState(Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════════════════
#                               CANDLE CACHE
# ═══════════════════════════════════════════════════════════════════════════════


class CandleCache:
    """Shamcha keshi — API chaqiruvlarni kamaytirish."""

    def __init__(self, max_size: int = 500, full_reload_interval: float = 60.0):
        self._candles = {}
        self._last_fetch = 0.0
        self._last_full_reload = 0.0
        self._max_size = max_size
        self._full_reload_interval = full_reload_interval
        self._lock = asyncio.Lock()

    async def get_candles(self, client: BitgetClient, symbol: str,
                         timeframe: str = "15m", limit: int = 200) -> List[Candle]:
        async with self._lock:
            now = time.time()
            if now - self._last_fetch < 1.0 and self._candles:
                return self._get_sorted()
            try:
                if now - self._last_full_reload >= self._full_reload_interval or not self._candles:
                    raw = await client.get_candles(symbol, timeframe, limit)
                    self._merge(raw)
                    self._last_full_reload = now
                else:
                    raw = await client.get_candles(symbol, timeframe, 5)
                    self._merge(raw)
                self._last_fetch = now
            except Exception as e:
                logger.warning(f"Candle fetch xato: {e}")
            return self._get_sorted()

    def _merge(self, raw_candles):
        if not raw_candles:
            return
        for c in raw_candles:
            try:
                ts = int(c[0])
                self._candles[ts] = Candle(
                    timestamp=ts, open=float(c[1]), high=float(c[2]),
                    low=float(c[3]), close=float(c[4]),
                    volume=float(c[5]) if len(c) > 5 else 0.0,
                )
            except (IndexError, ValueError):
                continue
        if len(self._candles) > self._max_size:
            keys = sorted(self._candles.keys())
            for k in keys[: len(keys) - self._max_size]:
                del self._candles[k]

    def _get_sorted(self):
        return [self._candles[k] for k in sorted(self._candles.keys())]


# ═══════════════════════════════════════════════════════════════════════════════
#                               TREND ROBOT
# ═══════════════════════════════════════════════════════════════════════════════


class TrendRobot:
    """Smart Trend Following v2.0 — asosiy robot klassi."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.state = RobotState.IDLE
        self.client: Optional[BitgetClient] = None
        self.strategy: Optional[TrendStrategy] = None

        self.current_price: float = 0.0
        self.balance: float = 0.0
        self.initial_balance: float = 0.0
        self.candles: List[Candle] = []
        self.htf_candles: List[Candle] = []

        self._candle_cache = CandleCache()
        self._htf_cache = CandleCache(full_reload_interval=300.0)  # HTF cache slower refresh

        self.start_time: Optional[datetime] = None
        self.tick_count: int = 0
        self.last_bar_time: int = 0

        self._running: bool = False
        self._order_lock = asyncio.Lock()

        # Sync race guard
        self._empty_sync_count = 0

        self._current_leverage: int = config.trading.LEVERAGE
        self._session_trade_amount: float = 0.0
        self._symbol_info: Optional[dict] = None

        self.global_limit_hit: Optional[Dict[str, Any]] = None

    # ─── LIFECYCLE ───────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        logger.info("Initializing trend robot...")
        self.state = RobotState.STARTING
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Config xato: {e}")
            self.state = RobotState.ERROR
            return False

        self.client = BitgetClient(self.config.api)
        try:
            self.balance = await self.client.get_balance()
            self.initial_balance = self.balance
            logger.info(f"Connected. Balance: ${self.balance:.2f}")

            try:
                await self.client.set_margin_mode(
                    symbol=self.config.trading.SYMBOL,
                    margin_mode=self.config.trading.MARGIN_MODE,
                )
            except Exception as e:
                if not any(k in str(e).lower() for k in ("already", "same", "40867")):
                    logger.warning(f"Margin mode xato: {e}")

            try:
                await self.client.set_position_mode(hedge_mode=True)
            except Exception as e:
                if not any(k in str(e).lower() for k in ("already", "same", "40756", "400172")):
                    logger.warning(f"Position mode xato: {e}")

            # Symbol info + leverage cap
            try:
                self._symbol_info = await self.client.get_symbol_info(self.config.trading.SYMBOL)
                if self._symbol_info and "maxLever" in self._symbol_info:
                    max_lev = int(self._symbol_info["maxLever"])
                    if self.config.trading.LEVERAGE > max_lev:
                        logger.warning(
                            f"Capping leverage {self.config.trading.LEVERAGE}x -> {max_lev}x"
                        )
                        self.config.trading.LEVERAGE = max_lev
            except Exception as e:
                logger.warning(f"Symbol info xato: {e}")

            # Set leverage
            try:
                await self.client.set_leverage(
                    symbol=self.config.trading.SYMBOL,
                    leverage=self.config.trading.LEVERAGE,
                )
                self._current_leverage = self.config.trading.LEVERAGE
            except BitgetAPIError as e:
                if "40797" in str(e) or "exceeded" in str(e).lower():
                    try:
                        for side in ("long", "short"):
                            await self.client.set_leverage_with_side(
                                symbol=self.config.trading.SYMBOL,
                                leverage=self.config.trading.LEVERAGE,
                                hold_side=side,
                            )
                        self._current_leverage = self.config.trading.LEVERAGE
                    except BitgetAPIError:
                        logger.warning("Leverage fallback failed, using 10x")
                        self._current_leverage = 10
                        self.config.trading.LEVERAGE = 10

            self.strategy = TrendStrategy(self.config)
            if self._symbol_info:
                self.strategy.set_symbol_info(self._symbol_info)

            self.state = RobotState.IDLE
            logger.info(f"Robot initialized for {self.config.trading.SYMBOL}")
            return True

        except BitgetAPIError as e:
            logger.error(f"API xato: {e}")
            self.state = RobotState.ERROR
            return False

    async def start(self) -> None:
        if not self.client or not self.strategy:
            if not await self.initialize():
                return

        self.state = RobotState.RUNNING
        self._running = True
        self.start_time = datetime.now(timezone.utc)
        self.tick_count = 0

        logger.info(f"Trading started: {self.config.trading.SYMBOL}")

        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self.config.TICK_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Trading cancelled")
        except KeyboardInterrupt:
            logger.info("Trading interrupted")
        except Exception as e:
            logger.error(f"Trading xato: {e}", exc_info=True)
            self.state = RobotState.ERROR
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self.state == RobotState.STOPPED:
            return
        logger.info("Stopping robot...")
        self.state = RobotState.STOPPING
        self._running = False

        if self.strategy and self.strategy.position:
            try:
                await self._close_position(self.strategy.position, self.current_price, "MANUAL_STOP")
            except Exception as e:
                logger.error(f"Close position xato: {e}")

        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass

        self.state = RobotState.STOPPED
        logger.info("Robot stopped")

    async def pause(self) -> None:
        if self.state == RobotState.RUNNING:
            self.state = RobotState.PAUSED
            logger.info("Paused")

    async def resume(self) -> None:
        if self.state == RobotState.PAUSED:
            self.state = RobotState.RUNNING
            logger.info("Resumed")

    # ─── TICK LOOP ───────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        if self.state in (RobotState.STOPPING, RobotState.STOPPED, RobotState.PAUSED):
            return

        self.tick_count += 1

        try:
            # 1. Market data
            await self._update_market_data()

            if not self.candles:
                return

            # 2. Indikatorlar yangilash
            self.strategy.update_indicators(self.candles)
            if self.htf_candles and self.config.trend.use_htf_filter:
                self.strategy.update_htf_indicators(self.htf_candles)

            # 3. Position mavjud bo'lsa — manage
            if self.strategy.position is not None:
                await self._manage_position()

            # 4. Circuit breaker
            if self._check_circuit_breaker():
                return

            # 5. New signal (agar pozitsiya yo'q bo'lsa)
            if self.strategy.position is None:
                await self._check_new_entry()

            # 6. Status log (har 30 tick)
            if self.tick_count % 30 == 0:
                self._log_status()

        except BitgetAPIError as e:
            logger.error(f"API xato tick: {e}")
        except Exception as e:
            logger.error(f"Tick xato: {e}", exc_info=True)

    async def _update_market_data(self) -> None:
        """Narx, candles (primary + HTF), balans yangilash."""
        try:
            self.current_price = await self.client.get_price(self.config.trading.SYMBOL)
        except Exception:
            return

        # Primary timeframe
        self.candles = await self._candle_cache.get_candles(
            client=self.client,
            symbol=self.config.trading.SYMBOL,
            timeframe=self.config.trend.timeframe,
            limit=200,
        )

        # HTF — har 60 tick (TICK_INTERVAL=1s da har minutda).
        # 4H candle 14400 sekundga yetadi, 60s polling yetarli.
        # Cache ham 300s full reload bilan ushlab turadi.
        if self.config.trend.use_htf_filter and self.tick_count % 60 == 0:
            try:
                self.htf_candles = await self._htf_cache.get_candles(
                    client=self.client,
                    symbol=self.config.trading.SYMBOL,
                    timeframe=self.config.trend.htf_timeframe,
                    limit=100,
                )
            except Exception as e:
                logger.debug(f"HTF candle xato: {e}")

        # Balance (har 5 tick)
        if self.tick_count % 5 == 0:
            try:
                self.balance = await self.client.get_balance()
            except Exception:
                pass

        # Position sync (har 3 tick)
        if self.tick_count % 3 == 0:
            await self._sync_position()

    # ─── POSITION MANAGEMENT ────────────────────────────────────────────────

    async def _manage_position(self) -> None:
        """Mavjud pozitsiyani boshqarish — partial TP, trailing, SL, opposite signal."""
        pos = self.strategy.position
        if pos is None or not self.candles:
            return

        latest = self.candles[-1]

        # 1. Partial TP (intrabar high/low)
        ptp_result = self.strategy.check_partial_tp(pos, latest.high, latest.low)
        if ptp_result:
            exit_price, close_size = ptp_result
            await self._close_partial(pos, exit_price, close_size, "PARTIAL_TP")
            if self.strategy.position is None:
                return
            pos = self.strategy.position

        # 2. Trailing stop update
        self.strategy.update_trailing_stop(pos, self.current_price)

        # 3. SL check (intrabar) — MAJBURIY EXIT, fee-aware bypass
        stop_hit = self.strategy.check_stop_hit(pos, latest.high, latest.low)
        if stop_hit is not None:
            reason = "SL" if pos.trailing_phase == TrailingPhase.NONE else "TRAIL_STOP"
            # Trailing stop allaqachon profit'da bo'lgan ekan, bu fee-aware emas.
            # Fixed SL ham — pozitsiya allaqachon zararda, exit majburiy.
            await self._close_position(pos, stop_hit, reason)
            self.strategy.set_cooldown(self.tick_count)
            return

        # 4. Opposite signal exit (yangi v2.1) — fee-aware
        # Death cross LONG'ni yopadi, golden cross SHORT'ni — agar kuchli
        # teskari signal bo'lsa va pozitsiya foydada bo'lsa.
        if self.strategy.detect_opposite_exit(pos.side):
            if self.strategy.voluntary_exit_allowed(pos, self.current_price):
                logger.info(
                    f"[OPPOSITE_SIGNAL] {pos.side.upper()} yopilmoqda — "
                    f"teskari cross detected (last_cross={self.strategy._last_cross})"
                )
                await self._close_position(pos, self.current_price, "OPPOSITE_SIGNAL")
                self.strategy.set_cooldown(self.tick_count)
                return
            else:
                logger.debug(
                    "[OPPOSITE_SIGNAL] Skipped — voluntary exit fee-not-covered"
                )

        # 5. Time-based exits (protects against stuck positions).
        #
        # Production issue observed 2026-04-23: AVAXUSDT SHORT opened at $9.196
        # sat open for 14+ hours as ADX decayed from 28 -> 22 and price drifted
        # against entry, never triggering SL (3%) or trailing (needs +1% first).
        # Unrealized PnL reached -$155 before manual intervention.
        #
        # Two rules, soft first then hard:
        #   (a) TREND_EXHAUSTION: position open > 12h AND ADX has decayed
        #       3 points below the entry threshold. Voluntary exit (fee-aware).
        #   (b) MAX_AGE: hard safety cap at 24h regardless of ADX. Majburiy.
        SOFT_HOLD_HOURS = 12
        HARD_MAX_HOURS = 24
        EXHAUSTION_MARGIN = 3.0
        opened_age_hours = (time.time() - (pos.opened_ts or time.time())) / 3600
        adx_now = self.strategy.adx.value if self.strategy.adx.initialized else 100

        if opened_age_hours > HARD_MAX_HOURS:
            logger.warning(
                f"[MAX_AGE] Force-closing {pos.side.upper()} after "
                f"{opened_age_hours:.1f}h — hard 24h safety cap reached "
                f"(ADX={adx_now:.1f})"
            )
            await self._close_position(pos, self.current_price, "MAX_AGE")
            self.strategy.set_cooldown(self.tick_count)
            return

        if (
            opened_age_hours > SOFT_HOLD_HOURS
            and adx_now < (self.strategy.cfg.adx_threshold - EXHAUSTION_MARGIN)
            and self.strategy.voluntary_exit_allowed(pos, self.current_price)
        ):
            logger.info(
                f"[TREND_EXHAUSTION] Force-closing {pos.side.upper()} after "
                f"{opened_age_hours:.1f}h — ADX={adx_now:.1f} "
                f"< threshold-{EXHAUSTION_MARGIN} ({self.strategy.cfg.adx_threshold - EXHAUSTION_MARGIN})"
            )
            await self._close_position(pos, self.current_price, "TREND_EXHAUSTION")
            self.strategy.set_cooldown(self.tick_count)

    # ─── ENTRY ───────────────────────────────────────────────────────────────

    async def _check_new_entry(self) -> None:
        if not self.strategy.can_trade_today():
            return
        if self.strategy.should_stop_trading():
            return

        signal = self.strategy.detect_signal(self.current_price, self.tick_count)
        if signal == SignalType.NONE:
            return

        # Trade amount semantikasi: MARGIN to allocate (USDT'da).
        # Notional = trade_amount × leverage (calculate_position_size'da).
        #
        # v2.0 BUG: agar _session_trade_amount=0, butun balans ishlatilardi:
        #   trade_amount = balance (1000$) × leverage (10) = $10K notional!
        # v2.1 FIX: balansning CAPITAL_ENGAGEMENT (default 15%) qismi ishlatamiz:
        #   trade_amount = $1000 × 0.15 = $150 margin × 10 leverage = $1500 notional
        if self._session_trade_amount > 0:
            trade_amount = self._session_trade_amount
        else:
            engagement = max(0.0, min(1.0, self.config.CAPITAL_ENGAGEMENT))
            trade_amount = self.balance * engagement
            if trade_amount <= 0:
                logger.debug(
                    f"Trade amount = 0 (balance=${self.balance:.2f}, "
                    f"engagement={engagement})"
                )
                return

        size = self.strategy.calculate_position_size(
            self.current_price, trade_amount, self.config.trading.LEVERAGE
        )
        if size <= 0:
            logger.debug(
                f"Position size too small: trade_amount=${trade_amount:.2f}, "
                f"price=${self.current_price:.4f}, leverage={self.config.trading.LEVERAGE}"
            )
            return

        async with self._order_lock:
            opened = await self._open_position(signal, size)
            # Signal ishlatildi — har holda consume (false signal qayta ko'tarilmasin).
            self.strategy.consume_signal()
            # Trades-per-hour rolling window — faqat haqiqiy entry'da
            if opened:
                self.strategy.register_entry()

    async def _open_position(self, signal: SignalType, size: float) -> bool:
        """
        Pozitsiya ochish. True = ochildi, False = ochilmadi (xato yoki rad).

        Margin pre-check: faqat yangi pozitsiyaning margin'ini emas, BUTUN
        exposure'ni hisoblaydi (mavjud + yangi). Bu bir necha pozitsiya bir
        vaqtda ochilsa liquidation risk'idan saqlaydi.
        """
        side = "long" if signal == SignalType.LONG else "short"
        try:
            # Yangi pozitsiya margin
            new_margin = (size * self.current_price) / self._current_leverage

            # Mavjud pozitsiya (agar bor bo'lsa) margin
            existing_margin = 0.0
            if self.strategy.position is not None:
                p = self.strategy.position
                existing_margin = (p.size * p.entry_price) / max(self._current_leverage, 1)

            # Total margin needed (mavjud + yangi) + 10% buffer
            total_required = (existing_margin + new_margin) * 1.1
            if self.balance < total_required:
                logger.warning(
                    f"Insufficient balance: total_required=${total_required:.2f} "
                    f"(existing=${existing_margin:.2f}, new=${new_margin:.2f}), "
                    f"available=${self.balance:.2f}"
                )
                return False

            if side == "long":
                result = await self.client.open_long(
                    symbol=self.config.trading.SYMBOL, size=size,
                    margin_mode=self.config.trading.MARGIN_MODE,
                )
            else:
                result = await self.client.open_short(
                    symbol=self.config.trading.SYMBOL, size=size,
                    margin_mode=self.config.trading.MARGIN_MODE,
                )

            order_id = (result.get("orderId", str(int(time.time() * 1000)))
                       if result else str(int(time.time() * 1000)))

            # Create strategy position
            pos = self.strategy.create_position(
                side=side, entry_price=self.current_price,
                size=size, order_id=order_id,
            )
            logger.info(
                f"OPEN {side.upper()}: {size:.6f} @ ${self.current_price:.4f} "
                f"(SL: ${pos.stop_price:.4f}) — ADX={self.strategy.adx.value:.1f} "
                f"ST={self.strategy.supertrend.direction}"
            )
            return True

        except BitgetAPIError as e:
            logger.error(f"Open {side} xato: {e}")
            return False

    # ─── CLOSE ───────────────────────────────────────────────────────────────

    async def _close_position(self, pos: Position, exit_price: float, reason: str) -> None:
        try:
            if pos.side == "long":
                await self.client.close_long(
                    symbol=self.config.trading.SYMBOL, size=pos.size,
                )
            else:
                await self.client.close_short(
                    symbol=self.config.trading.SYMBOL, size=pos.size,
                )
        except BitgetAPIError as e:
            if "22002" not in str(e) and "no position" not in str(e).lower():
                logger.error(f"Close {pos.side} xato: {e}")

        # Calculate net PnL
        gross = pos.pnl_at(exit_price)
        entry_fee = pos.entry_fee
        exit_fee = exit_price * pos.size * self.config.risk.TAKER_FEE_RATE
        net = gross - entry_fee - exit_fee

        self.strategy.on_trade_closed(net)
        self.strategy.position = None

        logger.info(
            f"CLOSE {pos.side.upper()} [{reason}]: exit=${exit_price:.4f}, "
            f"gross=${gross:+.2f}, net=${net:+.2f} "
            f"(WR: {self.strategy.winrate():.1f}%)"
        )

    async def _close_partial(self, pos: Position, exit_price: float,
                             close_size: float, reason: str) -> None:
        close_size = min(close_size, pos.size)
        if close_size <= 0:
            return

        try:
            if pos.side == "long":
                await self.client.close_long(
                    symbol=self.config.trading.SYMBOL, size=close_size,
                )
            else:
                await self.client.close_short(
                    symbol=self.config.trading.SYMBOL, size=close_size,
                )
        except BitgetAPIError as e:
            if "22002" not in str(e) and "no position" not in str(e).lower():
                logger.error(f"Partial close xato: {e}")

        # Proportional PnL
        pnl_delta = (exit_price - pos.entry_price) * close_size if pos.side == "long" \
            else (pos.entry_price - exit_price) * close_size
        fee_share = pos.entry_fee * (close_size / pos.initial_size)
        exit_fee = exit_price * close_size * self.config.risk.TAKER_FEE_RATE
        net = pnl_delta - fee_share - exit_fee

        self.strategy.on_trade_closed(net)

        # Reduce position size
        pos.size -= close_size
        pos.entry_fee -= fee_share
        if pos.size <= 0:
            self.strategy.position = None

        logger.info(
            f"PARTIAL {reason}: closed {close_size:.6f} @ ${exit_price:.4f}, "
            f"net=${net:+.2f}, remaining={pos.size:.6f}"
        )

    # ─── CIRCUIT BREAKER ─────────────────────────────────────────────────────

    def _check_circuit_breaker(self) -> bool:
        if self.strategy.position is None or self.initial_balance <= 0:
            return False
        unrealized = self.strategy.position.pnl_at(self.current_price)
        equity = self.balance + unrealized
        if self.strategy.check_max_drawdown(equity, self.initial_balance):
            logger.warning(
                f"[CIRCUIT BREAKER] Equity ${equity:.2f} / "
                f"Initial ${self.initial_balance:.2f} — close all"
            )
            asyncio.create_task(self._trigger_circuit_breaker())
            return True
        return False

    async def _trigger_circuit_breaker(self) -> None:
        if self.strategy.position:
            await self._close_position(self.strategy.position, self.current_price, "CIRCUIT")
        self.strategy.set_cooldown(self.tick_count)
        self.global_limit_hit = {
            "reason": "CIRCUIT_BREAKER",
            "balance": self.balance,
            "initial_balance": self.initial_balance,
        }

    # ─── POSITION SYNC ───────────────────────────────────────────────────────

    async def _sync_position(self) -> None:
        """Exchange bilan position sync (grace period bilan)."""
        if not self.strategy.position:
            return

        GRACE_SECONDS = 60.0
        REQUIRED_EMPTY = 3

        try:
            exchange_positions = await self.client.get_positions(
                symbol=self.config.trading.SYMBOL
            )
        except Exception as e:
            logger.debug(f"Position sync xato: {e}")
            return

        # Aggregate exchange size for the position's side
        ex_size = 0.0
        for ex in exchange_positions:
            side = getattr(ex, "side", "").lower()
            if side == self.strategy.position.side:
                ex_size += float(getattr(ex, "size", 0))

        now = time.time()
        if ex_size == 0:
            age = now - self.strategy.position.opened_ts
            self._empty_sync_count += 1
            if age < GRACE_SECONDS or self._empty_sync_count < REQUIRED_EMPTY:
                logger.debug(
                    f"[SYNC_RACE] Position sync miss "
                    f"(age={age:.0f}s, empty={self._empty_sync_count}/{REQUIRED_EMPTY})"
                )
                return
            # Really closed on exchange
            logger.info(
                f"[SYNC_RACE] Position confirmed closed on exchange "
                f"(age={age:.0f}s). Clearing local."
            )
            pos = self.strategy.position
            gross = pos.pnl_at(self.current_price)
            fee = pos.entry_price * pos.size * self.config.risk.TAKER_FEE_RATE
            fee += self.current_price * pos.size * self.config.risk.TAKER_FEE_RATE
            net = gross - fee
            self.strategy.on_trade_closed(net)
            self.strategy.position = None
            self._empty_sync_count = 0
        else:
            self._empty_sync_count = 0

    # ─── STATUS ──────────────────────────────────────────────────────────────

    def _log_status(self) -> None:
        s = self.strategy
        pos = s.position
        pos_str = "NONE"
        pnl = 0.0
        if pos:
            pnl = pos.pnl_at(self.current_price)
            pos_str = f"{pos.side.upper()}@${pos.entry_price:.4f} ({pos.trailing_phase.value})"
        logger.info(
            f"[Tick {self.tick_count}] {self.config.trading.SYMBOL} "
            f"${self.current_price:.4f} | EMA({s.ema.fast_value:.2f}/{s.ema.slow_value:.2f}) "
            f"ADX={s.adx.value:.1f} ATR={s.atr.value:.4f} ST={s.supertrend.direction} | "
            f"Pos={pos_str} PnL=${pnl:+.2f} | Bal=${self.balance:.2f}"
        )

    def get_status(self) -> Dict[str, Any]:
        uptime = 0
        if self.start_time:
            uptime = int((datetime.now(timezone.utc) - self.start_time).total_seconds())

        s = self.strategy
        pos = s.position if s else None
        return {
            "state": self.state.value,
            "symbol": self.config.trading.SYMBOL,
            "current_price": self.current_price,
            "balance": self.balance,
            "leverage": self.config.trading.LEVERAGE,
            "tick_count": self.tick_count,
            "uptime": uptime,
            "strategy": s.get_status() if s else {},
            "position": pos.to_dict() if pos else None,
            "stats": s.get_stats() if s else {},
            "config": {
                "symbol": self.config.trading.SYMBOL,
                "leverage": self.config.trading.LEVERAGE,
                "timeframe": self.config.trend.timeframe,
                "htf_timeframe": self.config.trend.htf_timeframe,
                "ema_fast": self.config.trend.ema_fast,
                "ema_slow": self.config.trend.ema_slow,
                "adx_threshold": self.config.trend.adx_threshold,
                "supertrend_multiplier": self.config.trend.supertrend_multiplier,
                "trailing_atr_multiplier": self.config.trend.trailing_atr_multiplier,
                "use_htf_filter": self.config.trend.use_htf_filter,
                "use_partial_tp": self.config.trend.use_partial_tp,
                "max_drawdown_percent": self.config.trend.max_drawdown_percent,
            },
        }

    # Backward compatibility
    async def force_sync_positions(self) -> bool:
        self._empty_sync_count = 0
        await self._sync_position()
        return True
