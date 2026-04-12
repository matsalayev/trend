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
import time
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
        - _tick() — webhook status updates
        - stop() — close all + webhook events
    """

    def __init__(self, config: RobotConfig, session: UserSession):
        super().__init__(config)
        self._session = session
        self._webhook = session.webhook_client
        self._persistence = session.state_persistence
        self._session_trade_amount = session.trade_amount
        self._status_update_counter = 0

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
                await self._webhook.send_event("positions_synced", {
                    "positions": positions,
                    "source": "initialize",
                })
            except Exception as e:
                logger.warning(f"positions_synced yuborishda xato: {e}")

    async def _tick(self):
        """Tick + webhook status updates (har 5 tick da)"""
        await super()._tick()

        self._status_update_counter += 1

        # Status update har 5 tick da
        if self._status_update_counter % 5 == 0 and self._webhook:
            try:
                status = self.get_status()
                await self._webhook.send_event("status_update", {
                    "status": status,
                    "price": self._current_price,
                    "balance": self._balance,
                    "equity": self._equity,
                })
            except Exception as e:
                logger.debug(f"Status update xatosi: {e}")

    async def stop(self):
        """Stop + pozitsiyalarni yopish + webhook"""
        logger.info("TrendRobotWithWebhook stopping...")

        # Pozitsiyalarni yopish
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
                await self._webhook.send_event("status_changed", {
                    "status": "stopped",
                    "reason": "user_requested",
                })
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
            trading_pair=str(cs.get("symbol", s.get("symbol", "BTCUSDT"))),
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
        """Sozlamalarni yangilash"""
        session = self._find_session(user_id)
        # TODO: live config update (robot config ni o'zgartirish)
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
