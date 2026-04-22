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
    RobotConfig, APIConfig, TradingConfig, ExitConfig, RiskConfig,
    TrendConfig, apply_preset_to_config, is_supported_pair, SUPPORTED_PAIRS,
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
    tick_interval: float = 1.0

    # Exit sozlamalari (skeleton)
    sl_percent: float = 3.0

    # Risk
    max_loss_percent: float = 20.0
    max_daily_loss_percent: float = 10.0

    # Capital engagement
    capital_engagement: float = 0.15

    # Fee
    taker_fee_rate: float = 0.001
    maker_fee_rate: float = 0.001

    # ── v2.0 Smart Trend params (preset + user override) ─────────────────
    ema_fast: Optional[int] = None
    ema_slow: Optional[int] = None
    adx_threshold: Optional[float] = None
    adx_period: Optional[int] = None
    atr_period: Optional[int] = None
    supertrend_period: Optional[int] = None
    supertrend_multiplier: Optional[float] = None
    trailing_atr_multiplier: Optional[float] = None
    trailing_activation_percent: Optional[float] = None
    initial_sl_percent: Optional[float] = None
    use_htf_filter: Optional[bool] = None
    htf_ema_fast: Optional[int] = None
    htf_ema_slow: Optional[int] = None
    use_partial_tp: Optional[bool] = None
    partial_tp1_percent: Optional[float] = None
    partial_tp1_size_pct: Optional[float] = None
    partial_tp2_percent: Optional[float] = None
    partial_tp2_size_pct: Optional[float] = None
    max_drawdown_percent: Optional[float] = None
    cooldown_bars_after_sl: Optional[int] = None

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
        """Pozitsiya yo'qolganini aniqlash va trade_closed yuborish.

        BUG FIX: faqat haqiqiy yopilishlarni yuborish:
        - Robot RUNNING holatida bo'lishi kerak (STARTING/STOPPING emas)
        - prev_pos haqiqiy ma'lumotlarga ega (entry_price > 0, size > 0)
        - Bu soxta trade_closed spam ning oldini oladi.
        """
        # Guard: faqat RUNNING holatida trade_closed yuborish
        if self.state != RobotState.RUNNING:
            return

        if prev_pos is not None and cur_pos is None:
            # Prev position haqiqiy bo'lishi kerak (never-tracked = skip)
            entry = getattr(prev_pos, "entry_price", 0) or 0
            size = getattr(prev_pos, "size", 0) or 0
            if entry <= 0 or size <= 0:
                return
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
                    reason="SIGNAL",
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

            # Skeleton — indikatorlar strategiya qayta yozilganda qo'shiladi
            status = self.strategy.get_status()
            trailing_long = status.get("trailing_long", {})
            trailing_short = status.get("trailing_short", {})

            trailing = {
                "longPhase": trailing_long.get("phase", "INACTIVE"),
                "longStopPrice": trailing_long.get("stop_price", 0),
                "shortPhase": trailing_short.get("phase", "INACTIVE"),
                "shortStopPrice": trailing_short.get("stop_price", 0),
            }

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

            settings = {
                "slPercent": self.config.exit.SL_PERCENT,
                "leverage": self.config.trading.LEVERAGE,
                "baseLot": round(self.config.CAPITAL_ENGAGEMENT * 100, 1),
                "feeRate": self._session.taker_fee_rate,
                "tickInterval": self.config.TICK_INTERVAL,
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

            # Assemble payload — skeleton (indikatorlar strategiya bilan qaytadi)
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
        """Stop + pozitsiyalarni yopish + status_changed.

        BUG FIX: trade_closed webhook NO LONGER sent with reason=BOT_STOPPED.
        Previously, every restart/stop would fetch exchange positions and fire
        trade_closed for each — this caused 3000+ fake close webhooks filling
        bot_webhook_logs. Real TP/SL closes are already handled by
        _detect_closed_position() during active running state. Stop events
        are communicated via status_changed only.
        """
        logger.info("TrendRobotWithWebhook stopping...")

        # Pozitsiyalarni yopish (webhook yubormasdan)
        if self.client:
            try:
                symbol = self.config.trading.SYMBOL
                await self.client.close_all_positions(symbol)
                logger.info("Barcha pozitsiyalar yopildi")
            except Exception as e:
                logger.error(f"Pozitsiya yopishda xato: {e}")

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

        trading_pair = str(cs.get('tradingPair', s.get('tradingPair', cs.get('symbol', s.get('symbol', 'BTCUSDT')))))

        # v2.0 Smart Trend params (preset/customSettings camelCase → snake_case)
        def _opt_float(k: str) -> Optional[float]:
            v = cs.get(k)
            return float(v) if v is not None else None

        def _opt_int(k: str) -> Optional[int]:
            v = cs.get(k)
            return int(v) if v is not None else None

        def _opt_bool(k: str) -> Optional[bool]:
            v = cs.get(k)
            return bool(v) if v is not None else None

        session = UserSession(
            user_id=user_id,
            user_bot_id=user_bot_id,
            session_key=session_key,
            created_at=time.time(),
            api_key=exchange.get("apiKey", ""),
            api_secret=exchange.get("apiSecret", ""),
            passphrase=exchange.get("passphrase", ""),
            is_demo=is_demo,
            trading_pair=trading_pair,
            leverage=int(cs.get("leverage", s.get("leverage", 10))),
            margin_mode=str(cs.get("marginMode", s.get("marginMode", "crossed"))),
            trade_amount=float(cs.get("tradeAmount", s.get("tradeAmount", 0))),
            tick_interval=float(cs.get("tickInterval", s.get("tickInterval", 1.0))),
            sl_percent=float(cs.get("slPercent", s.get("slPercent", 3.0))),
            capital_engagement=float(cs.get("capitalEngagement", s.get("capitalEngagement", 15))) / 100,
            max_loss_percent=float(cs.get("maxLossPercent", s.get("maxLossPercent", 20))),
            max_daily_loss_percent=float(cs.get("maxDailyLossPercent", s.get("maxDailyLossPercent", 10))),
            taker_fee_rate=fee_rate,
            maker_fee_rate=fee_rate,
            # v2.0 Smart Trend params from customSettings
            ema_fast=_opt_int("emaFast"),
            ema_slow=_opt_int("emaSlow"),
            adx_threshold=_opt_float("adxThreshold"),
            adx_period=_opt_int("adxPeriod"),
            atr_period=_opt_int("atrPeriod"),
            supertrend_period=_opt_int("supertrendPeriod"),
            supertrend_multiplier=_opt_float("supertrendMultiplier"),
            trailing_atr_multiplier=_opt_float("trailingAtrMultiplier"),
            trailing_activation_percent=_opt_float("trailingActivationPercent"),
            initial_sl_percent=_opt_float("initialSlPercent"),
            use_htf_filter=_opt_bool("useHtfFilter"),
            htf_ema_fast=_opt_int("htfEmaFast"),
            htf_ema_slow=_opt_int("htfEmaSlow"),
            use_partial_tp=_opt_bool("usePartialTp"),
            partial_tp1_percent=_opt_float("partialTp1Percent"),
            partial_tp1_size_pct=_opt_float("partialTp1SizePct"),
            partial_tp2_percent=_opt_float("partialTp2Percent"),
            partial_tp2_size_pct=_opt_float("partialTp2SizePct"),
            max_drawdown_percent=_opt_float("maxDrawdownPercent"),
            cooldown_bars_after_sl=_opt_int("cooldownBarsAfterSl"),
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
        """Trading boshlash — thread-safe, race-condition protected"""
        # R9: Lock ichida status tekshiruv + STARTING ga o'tkazish (atomik)
        async with self._lock:
            session = self._find_session(user_id)

            if session.status == SessionStatus.RUNNING:
                raise ValueError(f"User {user_id} allaqachon ishlayapti")
            if session.status == SessionStatus.STARTING:
                raise ValueError(f"User {user_id} hozir boshlanmoqda (STARTING)")
            if session.status == SessionStatus.STOPPING:
                raise ValueError(f"User {user_id} to'xtatilmoqda, kuting")

            # Eski task bor bo'lsa tozalash (PAUSED, STOPPED, ERROR dan start)
            old_task = session.task
            old_robot = session.robot

            session.status = SessionStatus.STARTING
            session.task = None
            session.robot = None

        # Lock tashqarisida: eski task/robot cleanup (boshqa sessionlarni bloklamaslik)
        if old_task and not old_task.done():
            old_task.cancel()
            try:
                await asyncio.wait_for(old_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.warning(f"Old task cleanup xato: {e}")

        if old_robot:
            try:
                await asyncio.wait_for(old_robot.stop(), timeout=5.0)
            except Exception as e:
                logger.warning(f"Old robot cleanup xato: {e}")

        # Long-running initialization lock tashqarisida
        try:
            config = self._create_robot_config(session)
            robot = TrendRobotWithWebhook(config, session)
            await robot.initialize()

            # Atomik status flip + task create
            async with self._lock:
                session.robot = robot
                session.task = asyncio.create_task(robot.start())
                session.status = SessionStatus.RUNNING

            logger.info(f"Session boshlandi: {session.session_key}")

        except Exception as e:
            # Xatoda ham lock bilan ERROR status
            async with self._lock:
                session.status = SessionStatus.ERROR
            logger.error(f"Start xatosi: {e}")
            raise

    async def stop_session(self, user_id: str):
        """Trading to'xtatish — timeout va try/finally bilan himoyalangan.

        IDEMPOTENT: agar session topilmasa (restart'dan keyin in-memory
        yo'qolgan bo'lsa), xato qaytarmaydi — maqsad bajarilgan hisoblanadi.
        """
        # R9: Lock ichida status tekshiruv + STOPPING ga o'tkazish
        async with self._lock:
            try:
                session = self._find_session(user_id)
            except ValueError:
                logger.info(
                    f"Stop: session {user_id} topilmadi (ehtimol restart'dan "
                    f"keyin yo'qolgan) — idempotent stop, OK qaytarilmoqda"
                )
                return

            if session.status not in (SessionStatus.RUNNING, SessionStatus.PAUSED):
                # STOPPED, ERROR, REGISTERED holatlarida stop zaruri yo'q — silent ok
                if session.status in (SessionStatus.STOPPED, SessionStatus.REGISTERED):
                    return
                if session.status == SessionStatus.STOPPING:
                    raise ValueError(f"User {user_id} allaqachon to'xtatilmoqda")
                if session.status == SessionStatus.STARTING:
                    raise ValueError(f"User {user_id} hozir boshlanmoqda, keyinroq urinib ko'ring")
                if session.status == SessionStatus.ERROR:
                    # ERROR holatidan ham stop qilish mumkin — cleanup qilish
                    pass

            robot = session.robot
            task = session.task
            session.status = SessionStatus.STOPPING

        # Lock tashqarisida cleanup
        STOP_TIMEOUT = 15.0  # robot.stop() uchun maksimal vaqt
        try:
            # 1. Robot ni to'xtatish (timeout bilan)
            if robot:
                try:
                    await asyncio.wait_for(robot.stop(), timeout=STOP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"robot.stop() {STOP_TIMEOUT}s ichida tugamadi — "
                        f"majburan davom etamiz"
                    )
                except Exception as e:
                    logger.warning(f"robot.stop() xato: {e}")

            # 2. Task cancel + wait (fire-and-forget emas!)
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.warning(f"Task cancel xato: {e}")

        finally:
            # HAR DOIM STOPPED ga o'tkazish — STOPPING da muzlab qolmaslik
            async with self._lock:
                session.status = SessionStatus.STOPPED
                session.robot = None
                session.task = None
            logger.info(f"Session to'xtatildi: {session.session_key}")

    async def pause_session(self, user_id: str):
        """Pauza — idempotent (session yo'q bo'lsa ham OK)"""
        try:
            session = self._find_session(user_id)
        except ValueError:
            logger.info(f"Pause: session {user_id} yo'q — idempotent OK")
            return
        if session.robot:
            await session.robot.pause()
        session.status = SessionStatus.PAUSED

    async def resume_session(self, user_id: str):
        """Davom ettirish — idempotent"""
        try:
            session = self._find_session(user_id)
        except ValueError:
            logger.warning(f"Resume: session {user_id} yo'q — HEMA start yuborish kerak")
            raise
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
        """Session sozlamalari — skeleton"""
        session = self._find_session(user_id)
        return {
            "symbol": session.trading_pair,
            "leverage": session.leverage,
            "marginMode": session.margin_mode,
            "slPercent": session.sl_percent,
            "capitalEngagement": session.capital_engagement * 100,
            "maxLossPercent": session.max_loss_percent,
            "maxDailyLossPercent": session.max_daily_loss_percent,
            "feeRate": session.taker_fee_rate * 100,
            "tradeAmount": session.trade_amount,
            "tickInterval": session.tick_interval,
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
        if "tickInterval" in cs:
            session.tick_interval = float(cs["tickInterval"])

        # ── Exit (skeleton) ──────────────────────────────────────────────
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
        """Pozitsiyalarni yopish — idempotent"""
        try:
            session = self._find_session(user_id)
        except ValueError:
            logger.info(f"Close positions: session {user_id} yo'q — idempotent OK")
            return
        if session.robot and session.robot.client:
            await session.robot.client.close_all_positions(session.trading_pair)

    async def force_sync(self, user_id: str):
        """Exchange bilan sync — idempotent"""
        try:
            session = self._find_session(user_id)
        except ValueError:
            logger.info(f"Force sync: session {user_id} yo'q — idempotent OK")
            return
        if session.state_persistence:
            session.state_persistence.clear_positions()
        logger.info(f"Force sync: {session.session_key}")

    async def unregister_user(self, user_id: str):
        """Foydalanuvchini o'chirish — idempotent"""
        try:
            session = self._find_session(user_id)
        except ValueError:
            logger.info(f"Unregister: session {user_id} yo'q — idempotent OK")
            return

        await self.stop_session(user_id)

        if session.webhook_client:
            try:
                await asyncio.wait_for(session.webhook_client.stop(), timeout=5.0)
            except Exception as e:
                logger.warning(f"Webhook client stop xato: {e}")

        self._sessions.pop(session.session_key, None)
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
        """
        Session dan RobotConfig yaratish.

        1. Pair asosida preset yuklanadi (BotPairConfig.customSettings dan HEMA yuboradi).
        2. User customSettings preset'ni override qiladi.
        3. Strategy parametrlari (ema_fast, adx_threshold, va h.k.) shu yerdan o'rnatiladi.
        """
        # Bazaviy config yaratish
        config = RobotConfig(
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
            exit=ExitConfig(
                SL_PERCENT=getattr(session, "sl_percent", 3.0),
            ),
            risk=RiskConfig(
                MAX_LOSS_PERCENT=getattr(session, "max_loss_percent", 20.0),
                MAX_DAILY_LOSS_PERCENT=getattr(session, "max_daily_loss_percent", 10.0),
                TAKER_FEE_RATE=getattr(session, "taker_fee_rate", 0.001),
                MAKER_FEE_RATE=getattr(session, "maker_fee_rate", 0.001),
            ),
            trend=TrendConfig(),  # default, preset override qiladi
            TICK_INTERVAL=getattr(session, "tick_interval", 1.0),
            CAPITAL_ENGAGEMENT=getattr(session, "capital_engagement", 0.15),
        )

        # Per-pair preset qo'llanish + user override
        user_overrides = self._extract_user_overrides(session)
        config = apply_preset_to_config(config, session.trading_pair, user_overrides)

        logger.info(
            f"Config for {session.user_id}: symbol={config.trading.SYMBOL}, "
            f"leverage={config.trading.LEVERAGE}x, "
            f"EMA({config.trend.ema_fast}/{config.trend.ema_slow}), "
            f"ADX>{config.trend.adx_threshold}, "
            f"ST×{config.trend.supertrend_multiplier}, "
            f"HTF={config.trend.use_htf_filter}, "
            f"PTP={config.trend.use_partial_tp}"
        )
        return config

    def _extract_user_overrides(self, session: "UserSession") -> dict:
        """
        Session field'laridan user override'lar.
        HEMA customSettings orqali yuborilgan qiymatlar shu yerda yig'iladi.
        Session attribute bor bo'lsa va noldan farq qilsa — override sifatida ishlatiladi.
        """
        overrides: dict = {}
        # Leverage — har doim user tanlovi (> 0 bo'lsa)
        if getattr(session, "leverage", 0) > 0:
            overrides["leverage"] = session.leverage

        # customSettings'dagi barcha v2.0 maydonlar
        field_map = {
            "ema_fast": "ema_fast",
            "ema_slow": "ema_slow",
            "adx_threshold": "adx_threshold",
            "adx_period": "adx_period",
            "atr_period": "atr_period",
            "supertrend_period": "supertrend_period",
            "supertrend_multiplier": "supertrend_multiplier",
            "trailing_atr_multiplier": "trailing_atr_multiplier",
            "trailing_activation_percent": "trailing_activation_percent",
            "initial_sl_percent": "initial_sl_percent",
            "use_htf_filter": "use_htf_filter",
            "htf_ema_fast": "htf_ema_fast",
            "htf_ema_slow": "htf_ema_slow",
            "use_partial_tp": "use_partial_tp",
            "partial_tp1_percent": "partial_tp1_percent",
            "partial_tp1_size_pct": "partial_tp1_size_pct",
            "partial_tp2_percent": "partial_tp2_percent",
            "partial_tp2_size_pct": "partial_tp2_size_pct",
            "max_drawdown_percent": "max_drawdown_percent",
            "cooldown_bars_after_sl": "cooldown_bars_after_sl",
        }
        for attr, key in field_map.items():
            val = getattr(session, attr, None)
            if val is not None:
                overrides[key] = val
        return overrides


# ═══════════════════════════════════════════════════════════════════════════════
#                               SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

def get_session_manager() -> SessionManager:
    """SessionManager singleton olish"""
    return SessionManager.get_instance()
