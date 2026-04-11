"""
Trend Robot - Main Robot

Bollinger Bands + RSI + Volatility Filter breakout bot.
Tick loop va lifecycle boshqaruvi.

Tick tartibi:
    1. Narx va candle olish
    2. Balans yangilash (har 5 tick)
    3. Pozitsiyalar sync (har 10 tick)
    4. Signal tekshirish (3-condition: BB + RSI + Volatility)
    5. Pozitsiya ochish (agar signal va pozitsiya yo'q)
    6. TP/SL re-check (exchange da borligini tekshirish)
    7. Partial TP tekshirish (cascade exit)
    8. Risk check (max loss / daily loss)
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional, List

from .config import RobotConfig
from .api_client import BitgetClient, BitgetAPIError
from .strategy import TrendStrategy, Position, SignalType, TPSLPrices
from .indicators import Candle

logger = logging.getLogger(__name__)


class RobotState(Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class CandleCache:
    """Shamcha keshi — API chaqiruvlarni kamaytirish"""

    def __init__(self, max_size: int = 500):
        self._candles = {}
        self._last_fetch = 0.0
        self._last_full_reload = 0.0
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get_candles(self, client, symbol, timeframe="5m", limit=200):
        async with self._lock:
            now = time.time()
            if now - self._last_fetch < 1.0 and self._candles:
                return self._get_sorted()
            try:
                if now - self._last_full_reload >= 60.0 or not self._candles:
                    raw = await client.get_candles(symbol, timeframe, limit)
                    self._merge(raw)
                    self._last_full_reload = now
                else:
                    raw = await client.get_candles(symbol, timeframe, 5)
                    self._merge(raw)
                self._last_fetch = now
            except Exception as e:
                logger.warning(f"Shamcha olishda xato: {e}")
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
            for k in keys[:len(keys) - self._max_size]:
                del self._candles[k]

    def _get_sorted(self):
        return [self._candles[k] for k in sorted(self._candles.keys())]


class TrendRobot:
    """Trend trading robot — tick loop va lifecycle"""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.state = RobotState.IDLE

        self.client: Optional[BitgetClient] = None
        self.strategy: Optional[TrendStrategy] = None
        self.candle_cache: Optional[CandleCache] = None

        self._balance = 0.0
        self._equity = 0.0
        self._current_price = 0.0
        self._initial_balance = 0.0
        self._daily_start_balance = 0.0
        self._running = False
        self._tick_count = 0
        self._start_time = 0.0
        self._symbol_info = None
        self._order_lock = asyncio.Lock()

    async def initialize(self):
        """Bot ishga tayyorlash"""
        self.state = RobotState.STARTING
        symbol = self.config.trading.SYMBOL

        try:
            self.client = BitgetClient(self.config.api)
            await self.client.connect()

            self._balance = await self.client.get_balance()
            self._equity = await self.client.get_equity()
            self._initial_balance = self._balance
            self._daily_start_balance = self._balance

            if self._balance <= 0:
                raise ValueError("Balans 0 — savdo qilib bo'lmaydi")

            # Margin mode
            try:
                await self.client.set_margin_mode(symbol, self.config.trading.MARGIN_MODE)
            except BitgetAPIError as e:
                if "already" not in str(e).lower() and "40867" not in str(e):
                    raise

            # Hedge mode
            try:
                await self.client.set_position_mode(hedge_mode=True)
            except BitgetAPIError as e:
                if "already" not in str(e).lower() and "40756" not in str(e) and "400172" not in str(e):
                    raise

            # Leverage
            try:
                await self.client.set_leverage(symbol, self.config.trading.LEVERAGE)
            except BitgetAPIError:
                try:
                    await self.client.set_leverage(symbol, self.config.trading.LEVERAGE, hold_side="long")
                    await self.client.set_leverage(symbol, self.config.trading.LEVERAGE, hold_side="short")
                except Exception:
                    pass

            self._symbol_info = await self.client.get_symbol_info(symbol)
            self.strategy = TrendStrategy(self.config)
            if self._symbol_info:
                self.strategy.set_symbol_info(self._symbol_info)
            self.candle_cache = CandleCache()

            logger.info(f"Initialized: {symbol}, balance=${self._balance:.2f}")
        except Exception as e:
            self.state = RobotState.ERROR
            raise

    async def start(self):
        self.state = RobotState.RUNNING
        self._running = True
        self._start_time = time.time()
        logger.info(f"Trading boshlandi: {self.config.trading.SYMBOL}")

        try:
            while self._running:
                if self.state in (RobotState.STOPPING, RobotState.STOPPED):
                    break
                if self.state == RobotState.PAUSED:
                    await asyncio.sleep(self.config.TICK_INTERVAL)
                    continue
                try:
                    await self._tick()
                except Exception as e:
                    logger.error(f"Tick xatosi: {e}", exc_info=True)
                await asyncio.sleep(self.config.TICK_INTERVAL)
        except asyncio.CancelledError:
            pass
        finally:
            self.state = RobotState.STOPPED

    async def stop(self):
        self.state = RobotState.STOPPING
        self._running = False
        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass
        self.state = RobotState.STOPPED

    async def pause(self):
        if self.state == RobotState.RUNNING:
            self.state = RobotState.PAUSED

    async def resume(self):
        if self.state == RobotState.PAUSED:
            self.state = RobotState.RUNNING

    async def _tick(self):
        """Asosiy tick"""
        self._tick_count += 1
        symbol = self.config.trading.SYMBOL

        # 1. Narx
        try:
            self._current_price = await self.client.get_price(symbol)
        except Exception:
            return

        # 2. Candles
        tf = self.config.entry.TIMEFRAME
        candles = await self.candle_cache.get_candles(self.client, symbol, tf, 200)

        # 3. Balans (har 5 tick)
        if self._tick_count % 5 == 0:
            try:
                self._balance = await self.client.get_balance()
                self._equity = await self.client.get_equity()
            except Exception:
                pass

        # 4. Pozitsiyalar (har 10 tick)
        long_pos = None
        short_pos = None
        if self._tick_count % 10 == 0 or self._tick_count <= 2:
            try:
                raw = await self.client.get_positions(symbol)
                for p in raw:
                    side = "long" if p.get("holdSide") == "long" else "short"
                    size = float(p.get("total", 0))
                    if size > 0:
                        pos = Position(
                            id=p.get("posId", ""),
                            side=side,
                            entry_price=float(p.get("openPriceAvg", 0)),
                            size=size,
                            unrealized_pnl=float(p.get("unrealizedPL", 0)),
                            margin=float(p.get("margin", 0)),
                            leverage=int(p.get("leverage", 0)),
                        )
                        if side == "long":
                            long_pos = pos
                        else:
                            short_pos = pos
            except Exception:
                pass

        self.strategy.long_position = long_pos
        self.strategy.short_position = short_pos
        in_position = long_pos is not None or short_pos is not None

        # 5. Risk check
        if self.strategy.check_max_loss(self._balance, self._initial_balance):
            logger.critical(f"MAX LOSS HIT! Balance: ${self._balance:.2f}")
            await self._force_close_all()
            self._running = False
            return

        if self.strategy.check_daily_loss(self._balance, self._daily_start_balance):
            logger.warning(f"DAILY LOSS HIT! Balance: ${self._balance:.2f}")
            self._running = False
            return

        # 6. Signal tekshirish (faqat pozitsiya yo'q bo'lganda)
        if not in_position and candles:
            signal = self.strategy.check_signal(candles)

            if signal != SignalType.NONE:
                async with self._order_lock:
                    await self._open_position(signal)

        # 7. Status log (har 30 tick)
        if self._tick_count % 30 == 0:
            self._log_status()

    async def _open_position(self, signal: SignalType):
        """Pozitsiya ochish"""
        symbol = self.config.trading.SYMBOL
        side = "buy" if signal == SignalType.LONG else "sell"
        size = self.strategy.calculate_position_size(self._current_price, self._balance)

        if size <= 0:
            logger.warning("Pozitsiya hajmi 0 — order qo'yilmaydi")
            self.strategy.signal_tracker.record_ignored(signal.value, "size=0")
            return

        try:
            logger.info(f"OPENING {signal.value}: {side} {size:.6f} @ ${self._current_price:.2f}")
            await self.client.place_order(symbol, side, size, "market")

            # TP/SL qo'yish
            pos_side = "long" if signal == SignalType.LONG else "short"
            tp_sl = self.strategy.calculate_tp_sl(self._current_price, pos_side)

            await self._place_tp_sl(symbol, pos_side, tp_sl)
            self.strategy.reset_for_new_position()
            self.strategy.signal_tracker.record_taken(signal.value)

        except Exception as e:
            logger.error(f"Order xatosi: {e}")

    async def _place_tp_sl(self, symbol: str, side: str, tp_sl: TPSLPrices):
        """TP/SL orderlarni exchange da qo'yish"""
        try:
            # TP
            await self.client.modify_tpsl(
                symbol=symbol,
                tp_price=tp_sl.tp_price,
                sl_price=tp_sl.sl_price,
                hold_side=side,
            )
            logger.info(
                f"TP/SL qo'yildi: TP=${tp_sl.tp_price:.2f}, SL=${tp_sl.sl_price:.2f}"
            )
        except Exception as e:
            logger.error(f"TP/SL qo'yishda xato: {e}")

    async def _force_close_all(self):
        """Barcha pozitsiyalarni yopish"""
        symbol = self.config.trading.SYMBOL
        try:
            await self.client.close_all_positions(symbol)
            logger.warning("Barcha pozitsiyalar yopildi (force close)")
        except Exception as e:
            logger.error(f"Force close xatosi: {e}")

    def _log_status(self):
        uptime = time.time() - self._start_time
        h, m = int(uptime // 3600), int((uptime % 3600) // 60)
        s = self.strategy.get_status() if self.strategy else {}
        logger.info(
            f"[{self.config.trading.SYMBOL}] "
            f"${self._current_price:.2f} | "
            f"Bal: ${self._balance:.2f} | "
            f"BB: [{s.get('bb', {}).get('lower', 0):.0f}-{s.get('bb', {}).get('upper', 0):.0f}] | "
            f"RSI: {s.get('rsi', {}).get('value', 0):.1f} | "
            f"Vol: {'YES' if s.get('volatility', {}).get('is_volatile', False) else 'no'} | "
            f"Signals: {s.get('signals', {}).get('detected', 0)}d/{s.get('signals', {}).get('taken', 0)}t | "
            f"{h}h{m}m"
        )

    def get_status(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        status = {
            "state": self.state.value,
            "symbol": self.config.trading.SYMBOL,
            "price": self._current_price,
            "balance": self._balance,
            "equity": self._equity,
            "initial_balance": self._initial_balance,
            "pnl": self._equity - self._initial_balance,
            "uptime_seconds": uptime,
            "tick_count": self._tick_count,
        }
        if self.strategy:
            status["strategy"] = self.strategy.get_status()
        return status
