"""
Trend Robot - Session Manager

Multi-user session boshqaruvi va TrendRobotWithWebhook subclass.
HEMA trend bot pattern ga mos.

SessionManager — Singleton, barcha foydalanuvchi sessionlarini boshqaradi.
TrendRobotWithWebhook — TrendRobot ni kengaytiradi:
    - Webhook integratsiya (HEMA ga eventlar yuborish)
    - State persistence (restart recovery)
    - Base lot auto-calculation
    - Allocated balance enforcement
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from .config import (
    RobotConfig, APIConfig, TradingConfig, GridConfig,
    EMAConfig, IchimokuConfig, TrendConfig, MTFConfig, ExitConfig, RiskConfig,
)
from .robot import TrendRobot, RobotState
from .api_client import BitgetClient
from .webhook_client import WebhookClient
from .state_persistence import StatePersistence, create_state_persistence

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                               SESSION STATUS
# ═══════════════════════════════════════════════════════════════════════════════

class SessionStatus(Enum):
    """Session holatlari"""
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════════════════
#                               USER SESSION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserSession:
    """Foydalanuvchi session ma'lumotlari"""

    # Identifikatsiya
    user_id: str
    user_bot_id: str
    session_key: str  # user_bot_id yoki user_id

    # Status
    status: SessionStatus = SessionStatus.REGISTERED
    created_at: float = 0.0

    # Exchange credentials
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    is_demo: bool = True

    # Trading sozlamalari
    trading_pair: str = "BTCUSDT"
    leverage: int = 10
    margin_mode: str = "crossed"
    trade_amount: float = 0.0

    # EMA sozlamalari
    ema_fast_period: int = 9
    ema_slow_period: int = 21

    # Ichimoku sozlamalari
    ichimoku_enabled: bool = True

    # Trend (ADX) sozlamalari
    adx_period: int = 14
    adx_threshold: float = 25.0

    # MTF sozlamalari
    mtf_enabled: bool = True

    # Exit sozlamalari
    use_trailing_stop: bool = True
    trailing_activate_pct: float = 1.0
    trailing_floor_pct: float = 0.3
    use_opposite_signal_exit: bool = True
    sl_percent: float = 3.0

    # Grid sozlamalari (optional)
    grid_enabled: bool = False

    # Risk
    max_loss_percent: float = 20.0
    max_daily_loss_percent: float = 10.0

    # Capital engagement
    capital_engagement: float = 0.15

    # Fee
    taker_fee_rate: float = 0.001
    maker_fee_rate: float = 0.001

    # Webhook
    webhook_url: str = ""
    webhook_secret: str = ""

    # Runtime (not serialized)
    robot: Optional[TrendRobot] = field(default=None, repr=False)
    webhook_client: Optional[WebhookClient] = field(default=None, repr=False)
    task: Optional[asyncio.Task] = field(default=None, repr=False)
    state_persistence: Optional[StatePersistence] = field(default=None, repr=False)
    symbol_info: Optional[dict] = field(default=None, repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
#                       TREND ROBOT WITH WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════════

class TrendRobotWithWebhook(TrendRobot):
    """
    TrendRobot ni kengaytiradi — webhook va state persistence qo'shadi.

    Server mode da bu klass ishlatiladi (TrendRobot emas).
    Override lar:
        - initialize() — position reconciliation
        - _open_position() — trade_opened webhook
        - _tick() — trade_closed detection + status updates
        - stop() — close all + individual trade_closed + status_changed
    """

    def __init__(self, config: RobotConfig, session: UserSession):
        super().__init__(config)
        self._session = session
        self._webhook = session.webhook_client
        self._persistence = session.state_persistence
        self._session_trade_amount = session.trade_amount
        self._status_update_counter = 0
        # Oldingi tick dagi pozitsiyalar — trade_closed detection uchun
        self._prev_long: Optional[Any] = None
        self._prev_short: Optional[Any] = None

    async def initialize(self):
        """Initialize + position reconciliation"""
        await super().initialize()

        # State persistence dan pozitsiyalarni yuklash
        if self._persistence:
            try:
                saved = self._persistence.load_positions()
                if saved:
                    logger.info(f"Saqlangan pozitsiyalar yuklandi: {len(saved)} ta")
            except Exception as e:
                logger.warning(f"State yuklashda xato: {e}")

        # Webhook — positions_synced
        if self._webhook:
            try:
                positions = await self.client.get_positions(self.config.trading.SYMBOL)
                await self._webhook.send_positions_synced(
                    user_bot_id=self._session.user_bot_id,
                    symbol=self.config.trading.SYMBOL,
                    positions=positions,
                )
            except Exception as e:
                logger.warning(f"positions_synced yuborishda xato: {e}")

    async def _open_position(self, signal):
        """Pozitsiya ochish + trade_opened webhook"""
        symbol = self.config.trading.SYMBOL
        from .strategy import SignalType
        side = "buy" if signal == SignalType.LONG else "sell"
        # Allocated balance cheklovi
        effective_balance = self._balance
        if hasattr(self, '_session_trade_amount') and self._session_trade_amount > 0:
            effective_balance = min(self._balance, self._session_trade_amount)
        size = self.strategy.calculate_position_size(self._current_price, effective_balance)

        if size <= 0:
            logger.warning("Pozitsiya hajmi 0 — order qo'yilmaydi")
            return

        try:
            logger.info(f"OPENING {signal.value}: {side} {size:.6f} @ ${self._current_price:.2f}")
            result = await self.client.place_order(symbol, side, "open", size, "market")
            order_id = ""
            if isinstance(result, dict):
                order_id = str(result.get("orderId", result.get("data", {}).get("orderId", "")))
            if not order_id:
                order_id = f"trend-{int(time.time() * 1000)}"

            # Darhol pozitsiyani belgilash — duplikat order oldini olish
            pos_side = "long" if signal == SignalType.LONG else "short"
            from .strategy import Position
            new_pos = Position(
                id=order_id, side=pos_side,
                entry_price=self._current_price, size=size,
            )
            if pos_side == "long":
                self.strategy.long_position = new_pos
            else:
                self.strategy.short_position = new_pos

            # SL qo'yish
            sl_price = self.strategy.calculate_sl_price(self._current_price, pos_side)

            if sl_price > 0:
                await self.client.modify_tpsl(
                    symbol=symbol,
                    side=pos_side,
                    sl_price=sl_price,
                )
                logger.info(f"SL qo'yildi: ${sl_price:.2f}")

            # Webhook — trade_opened
            if self._webhook:
                try:
                    await self._webhook.send_trade_opened(
                        user_bot_id=self._session.user_bot_id,
                        symbol=symbol,
                        side=side,
                        price=self._current_price,
                        quantity=size,
                        order_id=order_id,
                        leverage=self.config.trading.LEVERAGE,
                        margin_mode=self._session.margin_mode.upper(),
                    )
                except Exception as e:
                    logger.warning(f"trade_opened webhook xatosi: {e}")

            # State persistence
            if self._persistence:
                try:
                    self._persistence.save_position({
                        "order_id": order_id,
                        "side": side,
                        "price": self._current_price,
                        "size": size,
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception as e:
                    logger.warning(f"State saqlashda xato: {e}")

        except Exception as e:
            logger.error(f"Order xatosi: {e}")

    async def _tick(self):
        """Tick + trade_closed detection + webhook status updates"""
        # Oldingi pozitsiyalarni saqlash (trade_closed detection uchun)
        prev_long = self._prev_long
        prev_short = self._prev_short

        await super()._tick()

        # Hozirgi pozitsiyalar
        cur_long = self.strategy.long_position if self.strategy else None
        cur_short = self.strategy.short_position if self.strategy else None

        # Trade closed detection — pozitsiya yo'qolgan bo'lsa
        if self._webhook:
            await self._detect_closed_position(prev_long, cur_long, "long")
            await self._detect_closed_position(prev_short, cur_short, "short")

        # Hozirgi pozitsiyalarni keyingi tick uchun saqlash
        self._prev_long = cur_long
        self._prev_short = cur_short

        # Status update har 5 tick da
        self._status_update_counter += 1
        if self._status_update_counter % 5 == 0 and self._webhook:
            await self._send_status_update()

    async def _detect_closed_position(self, prev_pos, cur_pos, side: str):
        """Pozitsiya yo'qolganini aniqlash va trade_closed yuborish"""
        if prev_pos is not None and cur_pos is None:
            # Pozitsiya yopilgan
            try:
                if side == "long":
                    pnl = (self._current_price - prev_pos.entry_price) * prev_pos.size
                else:
                    pnl = (prev_pos.entry_price - self._current_price) * prev_pos.size

                await self._webhook.send_trade_closed(
                    user_bot_id=self._session.user_bot_id,
                    symbol=self.config.trading.SYMBOL,
                    side=side,
                    entry_price=prev_pos.entry_price,
                    exit_price=self._current_price,
                    quantity=prev_pos.size,
                    pnl=pnl,
                    reason="TRAILING_STOP" if self.config.exit.USE_TRAILING_STOP else "SIGNAL",
                )
                logger.info(f"trade_closed webhook: {side} pnl=${pnl:.4f}")

                # State persistence tozalash
                if self._persistence:
                    try:
                        self._persistence.clear_positions()
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"trade_closed webhook xatosi ({side}): {e}")

    async def _send_status_update(self):
        """Send status update to HEMA via _send_event (bypasses RSI-formatted send_status_update)."""
        if not self.strategy:
            return

        try:
            # Positions
            buy_positions = []
            sell_positions = []

            if self.strategy.long_position:
                p = self.strategy.long_position
                buy_positions.append({
                    "price": p.entry_price,
                    "lot": p.size,
                    "order_id": p.id,
                    "opened_at": p.opened_at,
                })

            if self.strategy.short_position:
                p = self.strategy.short_position
                sell_positions.append({
                    "price": p.entry_price,
                    "lot": p.size,
                    "order_id": p.id,
                    "opened_at": p.opened_at,
                })

            # All indicator data from strategy
            status = self.strategy.get_status()
            ema_raw = status.get("ema", {})
            ichimoku_raw = status.get("ichimoku")  # None when disabled
            adx_raw = status.get("adx", {})
            trailing_long = status.get("trailing_long", {})
            trailing_short = status.get("trailing_short", {})

            # EMA signal: compare fast vs slow
            fast_val = ema_raw.get("fast", 0)
            slow_val = ema_raw.get("slow", 0)
            if fast_val > 0 and slow_val > 0:
                spread = fast_val - slow_val
                ema_signal = "GOLDEN_CROSS" if spread > 0 else "DEATH_CROSS"
            else:
                ema_signal = "NEUTRAL"

            ema = {
                "fast": fast_val,
                "slow": slow_val,
                "spread": ema_raw.get("spread", 0),
                "spreadPct": ema_raw.get("spread_pct", 0),
                "signal": ema_signal,
            }

            # Ichimoku — None when disabled
            ichimoku = None
            if ichimoku_raw:
                cloud_top = ichimoku_raw.get("cloud_top", 0)
                cloud_bottom = ichimoku_raw.get("cloud_bottom", 0)
                price_above = self._current_price > cloud_top if cloud_top > 0 else False
                if cloud_top > 0:
                    ichi_signal = "BULLISH" if price_above else "BEARISH"
                else:
                    ichi_signal = "NEUTRAL"
                ichimoku = {
                    "tenkan": ichimoku_raw.get("tenkan", 0),
                    "kijun": ichimoku_raw.get("kijun", 0),
                    "senkouA": ichimoku_raw.get("senkou_a", 0),
                    "senkouB": ichimoku_raw.get("senkou_b", 0),
                    "chikou": ichimoku_raw.get("chikou", 0),
                    "cloudTop": cloud_top,
                    "cloudBottom": cloud_bottom,
                    "priceAboveCloud": price_above,
                    "signal": ichi_signal,
                }

            # ADX
            adx = {
                "value": adx_raw.get("value", 0),
                "plusDi": adx_raw.get("plus_di", 0),
                "minusDi": adx_raw.get("minus_di", 0),
                "trending": adx_raw.get("trending", False),
                "threshold": self.config.trend.ADX_THRESHOLD,
            }

            # Trailing stop state
            trailing = {
                "longPhase": trailing_long.get("phase", "INACTIVE"),
                "longStopPrice": trailing_long.get("stop_price", 0),
                "shortPhase": trailing_short.get("phase", "INACTIVE"),
                "shortStopPrice": trailing_short.get("stop_price", 0),
            }

            # Performance
            performance = {
                "totalTrades": 0,
                "winningTrades": 0,
                "winRate": 0,
                "profit": round(self._equity - self._initial_balance, 4),
                "unrealizedPnl": round(self._equity - self._balance, 4),
                "peakBalance": self._balance,
                "maxDrawdown": 0,
                "maxDrawdownPercent": 0,
            }

            # Settings — trend-specific config
            settings = {
                "emaFastPeriod": self.config.ema.FAST_PERIOD,
                "emaSlowPeriod": self.config.ema.SLOW_PERIOD,
                "adxPeriod": self.config.trend.ADX_PERIOD,
                "adxThreshold": self.config.trend.ADX_THRESHOLD,
                "trailingActivate": self.config.exit.TRAILING_ACTIVATE_PCT,
                "trailingSL": self.config.exit.TRAILING_FLOOR_PCT,
                "slPercent": self.config.exit.SL_PERCENT,
                "ichimokuEnabled": self.config.ichimoku.ENABLED,
                "leverage": self.config.trading.LEVERAGE,
                "timeframe": self.config.mtf.PRIMARY_TIMEFRAME,
                "baseLot": round(self.config.CAPITAL_ENGAGEMENT * 100, 1),
                "feeRate": self._session.taker_fee_rate,
            }

            # Runtime
            uptime = int(time.time() - self._start_time) if self._start_time else 0
            started_at = (
                datetime.fromtimestamp(self._start_time, tz=timezone.utc).isoformat()
                if self._start_time else ""
            )
            runtime = {
                "tick": self._tick_count,
                "uptime": uptime,
                "startedAt": started_at,
            }

            # Assemble payload — HEMA destructures: ema, ichimoku, adx, trailing, etc.
            payload = {
                "symbol": self.config.trading.SYMBOL,
                "currentPrice": self._current_price,
                "balance": self._balance,
                "positions": {
                    "buy": buy_positions,
                    "sell": sell_positions,
                },
                "performance": performance,
                "settings": settings,
                "runtime": runtime,
                "ema": ema,
                "ichimoku": ichimoku,
                "adx": adx,
                "trailing": trailing,
            }

            await self._webhook._send_event(
                event_type="status_update",
                user_bot_id=self._session.user_bot_id,
                data=payload,
            )

        except Exception as e:
            logger.warning(f"Status update xatosi: {e}", exc_info=True)

    async def stop(self):
        """Stop + har bir pozitsiya uchun trade_closed webhook + status_changed"""
        logger.info("TrendRobotWithWebhook stopping...")

        # Avval pozitsiyalarni o'qib olish (trade_closed uchun)
        positions_to_close = []
        if self.client:
            try:
                symbol = self.config.trading.SYMBOL
                raw = await self.client.get_positions(symbol)
                for p in raw:
                    side = p.side if hasattr(p, 'side') else "long"
                    size = p.size if hasattr(p, 'size') else 0
                    entry_price = p.entry_price if hasattr(p, 'entry_price') else 0
                    if size > 0:
                        positions_to_close.append({
                            "side": side,
                            "size": size,
                            "entry_price": entry_price,
                        })
            except Exception as e:
                logger.warning(f"Stop: pozitsiyalarni o'qishda xato: {e}")

        # Pozitsiyalarni yopish
        if self.client:
            try:
                symbol = self.config.trading.SYMBOL
                await self.client.close_all_positions(symbol)
                logger.info("Barcha pozitsiyalar yopildi")
            except Exception as e:
                logger.error(f"Pozitsiya yopishda xato: {e}")

        # Har bir pozitsiya uchun trade_closed webhook
        if self._webhook and positions_to_close:
            for pos in positions_to_close:
                try:
                    if pos["side"] == "long":
                        pnl = (self._current_price - pos["entry_price"]) * pos["size"]
                    else:
                        pnl = (pos["entry_price"] - self._current_price) * pos["size"]

                    await self._webhook.send_trade_closed(
                        user_bot_id=self._session.user_bot_id,
                        symbol=self.config.trading.SYMBOL,
                        side=pos["side"],
                        entry_price=pos["entry_price"],
                        exit_price=self._current_price,
                        quantity=pos["size"],
                        pnl=pnl,
                        reason="BOT_STOPPED",
                    )
                    logger.info(f"Stop trade_closed: {pos['side']} pnl=${pnl:.4f}")
                except Exception as e:
                    logger.warning(f"Stop trade_closed webhook xatosi: {e}")

        # Webhook — status_changed
        if self._webhook:
            try:
                await self._webhook.send_status_changed(
                    user_bot_id=self._session.user_bot_id,
                    status="stopped",
                    message="user_requested",
                )
            except Exception:
                pass

        # State persistence tozalash
        if self._persistence:
            try:
                self._persistence.clear_positions()
            except Exception:
                pass

        await super().stop()

# ═══════════════════════════════════════════════════════════════════════════════
#                           SESSION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Multi-user session boshqaruvchisi — Singleton.

    Har bir foydalanuvchi uchun alohida robot instance yaratadi.
    HEMA trend bot pattern ga mos API.
    """

    _instance: Optional["SessionManager"] = None

    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}
        self._sessions_by_user_id: Dict[str, str] = {}  # user_id → session_key
        self._sessions_by_bot_id: Dict[str, str] = {}   # user_bot_id → session_key
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values()
                   if s.status in (SessionStatus.RUNNING, SessionStatus.PAUSED))

    # ─── REGISTER ───────────────────────────────────────────────────────────

    async def register_user(
        self,
        user_id: str,
        user_bot_id: str,
        exchange: dict,
        settings: dict,
        custom_settings: dict,
        webhook_url: str = "",
        webhook_secret: str = "",
    ) -> str:
        """Foydalanuvchi ro'yxatdan o'tkazish"""
        session_key = user_bot_id or user_id

        async with self._lock:
            # Eski session tozalash
            if session_key in self._sessions:
                old = self._sessions[session_key]
                if old.task and not old.task.done():
                    old.task.cancel()
                if old.robot and old.robot.client:
                    try:
                        await old.robot.client.close()
                    except Exception:
                        pass
                if old.webhook_client:
                    await old.webhook_client.stop()
                del self._sessions[session_key]
                logger.info(f"Eski session tozalandi: {session_key}")

        # Custom settings dan qiymatlar olish
        cs = custom_settings or {}
        s = settings or {}

        # Demo mode
        is_demo = str(s.get("isDemo", exchange.get("isDemo", "true"))).lower() in ("true", "1")

        # Fee rate (HEMA % da yuboradi, biz decimal ga o'tkazamiz)
        fee_rate = float(cs.get("feeRate", s.get("feeRate", 0.1))) / 100

        session = UserSession(
            user_id=user_id,
            user_bot_id=user_bot_id,
            session_key=session_key,
            created_at=time.time(),
            api_key=exchange.get("apiKey", ""),
            api_secret=exchange.get("apiSecret", ""),
            passphrase=exchange.get("passphrase", ""),
            is_demo=is_demo,
            trading_pair=str(cs.get('tradingPair', s.get('tradingPair', cs.get('symbol', s.get('symbol', 'BTCUSDT'))))),
            leverage=int(cs.get("leverage", s.get("leverage", 10))),
            margin_mode=str(cs.get("marginMode", s.get("marginMode", "crossed"))),
            trade_amount=float(cs.get("tradeAmount", s.get("tradeAmount", 0))),
            # EMA
            ema_fast_period=int(cs.get("emaFastPeriod", s.get("emaFastPeriod", 9))),
            ema_slow_period=int(cs.get("emaSlowPeriod", s.get("emaSlowPeriod", 21))),
            # Ichimoku
            ichimoku_enabled=str(cs.get("ichimokuEnabled", s.get("ichimokuEnabled", "true"))).lower() in ("true", "1"),
            # Trend (ADX)
            adx_period=int(cs.get("adxPeriod", s.get("adxPeriod", 14))),
            adx_threshold=float(cs.get("adxThreshold", s.get("adxThreshold", 25.0))),
            # MTF
            mtf_enabled=str(cs.get("mtfEnabled", s.get("mtfEnabled", "true"))).lower() in ("true", "1"),
            # Exit
            use_trailing_stop=str(cs.get("useTrailingStop", s.get("useTrailingStop", "true"))).lower() in ("true", "1"),
            trailing_activate_pct=float(cs.get("trailingActivatePct", s.get("trailingActivatePct", 1.0))),
            trailing_floor_pct=float(cs.get("trailingFloorPct", s.get("trailingFloorPct", 0.3))),
            use_opposite_signal_exit=str(cs.get("useOppositeSignalExit", s.get("useOppositeSignalExit", "true"))).lower() in ("true", "1"),
            sl_percent=float(cs.get("slPercent", s.get("slPercent", 3.0))),
            # Capital engagement
            capital_engagement=float(cs.get("capitalEngagement", s.get("capitalEngagement", 15))) / 100,
            # Risk
            max_loss_percent=float(cs.get("maxLossPercent", s.get("maxLossPercent", 20))),
            max_daily_loss_percent=float(cs.get("maxDailyLossPercent", s.get("maxDailyLossPercent", 10))),
            # Fee
            taker_fee_rate=fee_rate,
            maker_fee_rate=fee_rate,
            # Webhook
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
        )

        # Webhook URL rewrite for Docker internal networking
        internal_host = os.getenv("INTERNAL_WEBHOOK_HOST", "")
        if internal_host and webhook_url:
            webhook_url = re.sub(r'https?://[^/]+', f'http://{internal_host}', webhook_url)
            logger.info(f"Webhook URL rewritten: {webhook_url}")

        # Webhook client
        if webhook_url:
            from .webhook_client import WebhookConfig
            session.webhook_client = WebhookClient(WebhookConfig(
                url=webhook_url,
                secret=webhook_secret,
                user_id=user_id,
                user_bot_id=user_bot_id,
                taker_fee_rate=fee_rate,
            ))
            await session.webhook_client.start()

        # State persistence
        state_dir = os.getenv("STATE_DIR", "/data/state")
        session.state_persistence = create_state_persistence(
            user_id=user_bot_id or user_id,
            symbol=session.trading_pair,
            state_dir=state_dir,
        )

        # Sessionni saqlash
        self._sessions[session_key] = session
        self._sessions_by_user_id[user_id] = session_key
        self._sessions_by_bot_id[user_bot_id] = session_key

        logger.info(
            f"Session yaratildi: {session_key}, pair={session.trading_pair}, "
            f"leverage={session.leverage}x, demo={session.is_demo}"
        )

        return session_key

    # ─── START / STOP / PAUSE / RESUME ──────────────────────────────────────

    async def start_session(self, user_id: str):
        """Trading boshlash"""
        session = self._find_session(user_id)
        if session.status == SessionStatus.RUNNING:
            raise ValueError("Allaqachon ishlayapti")

        session.status = SessionStatus.STARTING

        # Robot config yaratish
        config = self._create_robot_config(session)

        # Robot yaratish
        robot = TrendRobotWithWebhook(config, session)
        session.robot = robot

        # Initialize va start
        try:
            await robot.initialize()
            session.status = SessionStatus.RUNNING
            session.task = asyncio.create_task(robot.start())
            logger.info(f"Session boshlandi: {session.session_key}")
        except Exception as e:
            session.status = SessionStatus.ERROR
            logger.error(f"Start xatosi: {e}")
            raise

    async def stop_session(self, user_id: str):
        """Trading to'xtatish"""
        session = self._find_session(user_id)
        session.status = SessionStatus.STOPPING

        if session.robot:
            await session.robot.stop()

        if session.task and not session.task.done():
            session.task.cancel()

        session.status = SessionStatus.STOPPED
        logger.info(f"Session to'xtatildi: {session.session_key}")

    async def pause_session(self, user_id: str):
        """Pauza"""
        session = self._find_session(user_id)
        if session.robot:
            await session.robot.pause()
        session.status = SessionStatus.PAUSED

    async def resume_session(self, user_id: str):
        """Davom ettirish"""
        session = self._find_session(user_id)
        if session.robot:
            await session.robot.resume()
        session.status = SessionStatus.RUNNING

    # ─── STATUS / SETTINGS ──────────────────────────────────────────────────

    async def get_status(self, user_id: str) -> dict:
        """Session holati"""
        session = self._find_session(user_id)
        result = {
            "userId": session.user_id,
            "userBotId": session.user_bot_id,
            "status": session.status.value,
            "pair": session.trading_pair,
            "leverage": session.leverage,
            "isDemo": session.is_demo,
        }
        if session.robot:
            result.update(session.robot.get_status())
        return result

    async def get_settings(self, user_id: str) -> dict:
        """Session sozlamalari"""
        session = self._find_session(user_id)
        return {
            "emaFastPeriod": session.ema_fast_period,
            "emaSlowPeriod": session.ema_slow_period,
            "ichimokuEnabled": session.ichimoku_enabled,
            "adxPeriod": session.adx_period,
            "adxThreshold": session.adx_threshold,
            "mtfEnabled": session.mtf_enabled,
            "useTrailingStop": session.use_trailing_stop,
            "trailingActivatePct": session.trailing_activate_pct,
            "useOppositeSignalExit": session.use_opposite_signal_exit,
            "slPercent": session.sl_percent,
            "capitalEngagement": session.capital_engagement * 100,
            "maxLossPercent": session.max_loss_percent,
            "feeRate": session.taker_fee_rate * 100,
        }

    async def update_settings(self, user_id: str, settings: dict):
        """Sozlamalarni yangilash (HEMA dan camelCase formatda keladi)"""
        session = self._find_session(user_id)
        cs = settings or {}

        # ── Trading ──────────────────────────────────────────────────────
        if "tradingPair" in cs:
            session.trading_pair = str(cs["tradingPair"])
        elif "symbol" in cs:
            session.trading_pair = str(cs["symbol"])
        if "leverage" in cs:
            session.leverage = int(cs["leverage"])
        if "marginMode" in cs:
            session.margin_mode = str(cs["marginMode"])
        if "tradeAmount" in cs:
            session.trade_amount = float(cs["tradeAmount"])

        # ── EMA ──────────────────────────────────────────────────────────
        if "emaFastPeriod" in cs:
            session.ema_fast_period = int(cs["emaFastPeriod"])
        if "emaSlowPeriod" in cs:
            session.ema_slow_period = int(cs["emaSlowPeriod"])

        # ── Ichimoku ─────────────────────────────────────────────────────
        if "ichimokuEnabled" in cs:
            session.ichimoku_enabled = str(cs["ichimokuEnabled"]).lower() in ("true", "1")

        # ── Trend (ADX) ──────────────────────────────────────────────────
        if "adxPeriod" in cs:
            session.adx_period = int(cs["adxPeriod"])
        if "adxThreshold" in cs:
            session.adx_threshold = float(cs["adxThreshold"])

        # ── MTF ──────────────────────────────────────────────────────────
        if "mtfEnabled" in cs:
            session.mtf_enabled = str(cs["mtfEnabled"]).lower() in ("true", "1")

        # ── Exit / Trailing Stop ─────────────────────────────────────────
        if "useTrailingStop" in cs:
            session.use_trailing_stop = str(cs["useTrailingStop"]).lower() in ("true", "1")
        if "trailingActivatePct" in cs:
            session.trailing_activate_pct = float(cs["trailingActivatePct"])
        if "trailingFloorPct" in cs:
            session.trailing_floor_pct = float(cs["trailingFloorPct"])
        if "useOppositeSignalExit" in cs:
            session.use_opposite_signal_exit = str(cs["useOppositeSignalExit"]).lower() in ("true", "1")
        if "slPercent" in cs:
            session.sl_percent = float(cs["slPercent"])

        # ── Capital engagement (HEMA sends %, we store decimal) ──────────
        if "capitalEngagement" in cs:
            session.capital_engagement = float(cs["capitalEngagement"]) / 100

        # ── Risk ─────────────────────────────────────────────────────────
        if "maxLossPercent" in cs:
            session.max_loss_percent = float(cs["maxLossPercent"])
        if "maxDailyLossPercent" in cs:
            session.max_daily_loss_percent = float(cs["maxDailyLossPercent"])

        # ── Fee (HEMA sends %, we store decimal) ─────────────────────────
        if "feeRate" in cs:
            fee = float(cs["feeRate"]) / 100
            session.taker_fee_rate = fee
            session.maker_fee_rate = fee

        # ── Live config update ───────────────────────────────────────────
        if session.robot and session.status == SessionStatus.RUNNING:
            session.robot.config = self._create_robot_config(session)
            logger.info(f"Robot config updated live: {session.session_key}")

            # Leverage o'zgarganda Bitget ga ham o'rnatish
            if "leverage" in cs and session.robot.client:
                try:
                    await session.robot.client.set_leverage(
                        session.trading_pair, session.leverage
                    )
                    logger.info(f"Bitget leverage yangilandi: {session.leverage}x")
                except Exception as e:
                    logger.warning(f"Bitget leverage o'rnatishda xato: {e}")

        logger.info(f"Settings updated: {session.session_key}")

    async def close_positions(self, user_id: str):
        """Pozitsiyalarni yopish"""
        session = self._find_session(user_id)
        if session.robot and session.robot.client:
            await session.robot.client.close_all_positions(session.trading_pair)

    async def force_sync(self, user_id: str):
        """Exchange bilan sync"""
        session = self._find_session(user_id)
        if session.state_persistence:
            session.state_persistence.clear_positions()
        logger.info(f"Force sync: {session.session_key}")

    async def unregister_user(self, user_id: str):
        """Foydalanuvchini o'chirish"""
        session = self._find_session(user_id)
        await self.stop_session(user_id)

        if session.webhook_client:
            await session.webhook_client.stop()

        del self._sessions[session.session_key]
        self._sessions_by_user_id.pop(session.user_id, None)
        self._sessions_by_bot_id.pop(session.user_bot_id, None)

        logger.info(f"Session o'chirildi: {session.session_key}")

    # ─── ADMIN ──────────────────────────────────────────────────────────────

    async def get_all_sessions(self) -> list:
        """Barcha sessiyalar"""
        result = []
        for key, session in self._sessions.items():
            result.append({
                "sessionKey": key,
                "userId": session.user_id,
                "userBotId": session.user_bot_id,
                "status": session.status.value,
                "pair": session.trading_pair,
                "isDemo": session.is_demo,
                "createdAt": session.created_at,
            })
        return result

    async def stop_all(self):
        """Barcha sessionlarni to'xtatish"""
        for key in list(self._sessions.keys()):
            try:
                session = self._sessions[key]
                if session.status in (SessionStatus.RUNNING, SessionStatus.PAUSED):
                    await self.stop_session(session.user_id)
            except Exception as e:
                logger.error(f"Stop all xatosi ({key}): {e}")

    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Eski sessionlarni tozalash"""
        count = 0
        cutoff = time.time() - max_age_hours * 3600
        for key in list(self._sessions.keys()):
            session = self._sessions[key]
            if session.created_at < cutoff and session.status == SessionStatus.STOPPED:
                del self._sessions[key]
                self._sessions_by_user_id.pop(session.user_id, None)
                self._sessions_by_bot_id.pop(session.user_bot_id, None)
                count += 1
        return count

    # ─── HELPERS ────────────────────────────────────────────────────────────

    def _find_session(self, user_id: str) -> UserSession:
        """Session topish (user_id, user_bot_id yoki session_key bo'yicha)"""
        # Direct key
        if user_id in self._sessions:
            return self._sessions[user_id]
        # By user_id
        key = self._sessions_by_user_id.get(user_id)
        if key and key in self._sessions:
            return self._sessions[key]
        # By user_bot_id
        key = self._sessions_by_bot_id.get(user_id)
        if key and key in self._sessions:
            return self._sessions[key]
        raise ValueError(f"Session topilmadi: {user_id}")

    def _create_robot_config(self, session: UserSession) -> RobotConfig:
        """Session dan RobotConfig yaratish"""
        return RobotConfig(
            api=APIConfig(
                API_KEY=session.api_key,
                SECRET_KEY=session.api_secret,
                PASSPHRASE=session.passphrase,
                DEMO_MODE=session.is_demo,
            ),
            trading=TradingConfig(
                SYMBOL=session.trading_pair,
                LEVERAGE=session.leverage,
                MARGIN_MODE=session.margin_mode,
            ),
            ema=EMAConfig(
                FAST_PERIOD=session.ema_fast_period,
                SLOW_PERIOD=session.ema_slow_period,
            ),
            ichimoku=IchimokuConfig(
                ENABLED=session.ichimoku_enabled,
            ),
            trend=TrendConfig(
                ADX_PERIOD=session.adx_period,
                ADX_THRESHOLD=session.adx_threshold,
            ),
            mtf=MTFConfig(
                ENABLED=session.mtf_enabled,
            ),
            exit=ExitConfig(
                USE_TRAILING_STOP=session.use_trailing_stop,
                TRAILING_ACTIVATE_PCT=session.trailing_activate_pct,
                TRAILING_FLOOR_PCT=session.trailing_floor_pct,
                USE_OPPOSITE_SIGNAL_EXIT=session.use_opposite_signal_exit,
                SL_PERCENT=session.sl_percent,
            ),
            grid=GridConfig(
                ENABLED=session.grid_enabled,
            ),
            risk=RiskConfig(
                MAX_LOSS_PERCENT=session.max_loss_percent,
                MAX_DAILY_LOSS_PERCENT=session.max_daily_loss_percent,
                TAKER_FEE_RATE=session.taker_fee_rate,
                MAKER_FEE_RATE=session.maker_fee_rate,
            ),
            CAPITAL_ENGAGEMENT=session.capital_engagement,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                               SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

def get_session_manager() -> SessionManager:
    """SessionManager singleton olish"""
    return SessionManager.get_instance()
