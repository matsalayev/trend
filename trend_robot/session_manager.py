"""
Trend Robot - Session Manager

Multi-user session boshqaruvi va TrendRobotWithWebhook subclass.
HEMA RSI/Hedge bot pattern ga mos.

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

from .config import RobotConfig, APIConfig, TradingConfig, GridConfig, EntryConfig, ExitConfig
from .config import TrailingConfig, UnstuckConfig, RiskConfig
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

    # Grid sozlamalari (HEMA dan keladi)
    n_positions: int = 5
    grid_spacing_pct: float = 0.005
    grid_spacing_volatility_weight: float = 0.5
    ema_span_0: float = 100.0
    ema_span_1: float = 1000.0
    initial_ema_dist: float = 0.003
    initial_qty_pct: float = 0.01
    double_down_factor: float = 0.7
    close_grid_markup_start: float = 0.006
    close_grid_markup_end: float = 0.009
    close_grid_qty_pct: float = 0.5

    # Trailing
    entry_trailing_threshold: float = 0.01
    entry_trailing_retracement: float = 0.005
    trailing_grid_ratio: float = 0.5

    # Risk
    wallet_exposure_limit: float = 1.0
    max_loss_percent: float = 20.0
    hsl_red_threshold: float = 0.5

    # Unstuck
    unstuck_threshold: float = 0.03
    unstuck_close_pct: float = 0.05

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
#                       PASSIV GRID ROBOT WITH WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════════

class TrendRobotWithWebhook(TrendRobot):
    """
    TrendRobot ni kengaytiradi — webhook va state persistence qo'shadi.

    Server mode da bu klass ishlatiladi (TrendRobot emas).
    RSI/Hedge bot pattern ga mos override lar:
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
    HEMA RSI/Hedge/Sure-Fire bot pattern ga mos API.
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
            # Grid
            n_positions=int(cs.get("nPositions", s.get("nPositions", 5))),
            grid_spacing_pct=float(cs.get("gridSpacingPct", s.get("gridSpacingPct", 0.5))) / 100,
            grid_spacing_volatility_weight=float(cs.get("gridSpacingVolatilityWeight", 0.5)),
            ema_span_0=float(cs.get("emaSpan0", s.get("emaSpan0", 100))),
            ema_span_1=float(cs.get("emaSpan1", s.get("emaSpan1", 1000))),
            initial_ema_dist=float(cs.get("initialEmaDist", s.get("initialEmaDist", 0.3))) / 100,
            initial_qty_pct=float(cs.get("initialQtyPct", s.get("initialQtyPct", 1))) / 100,
            double_down_factor=float(cs.get("doubleDownFactor", s.get("doubleDownFactor", 0.7))),
            close_grid_markup_start=float(cs.get("closeGridMarkupStart", 0.6)) / 100,
            close_grid_markup_end=float(cs.get("closeGridMarkupEnd", 0.9)) / 100,
            close_grid_qty_pct=float(cs.get("closeGridQtyPct", 50)) / 100,
            # Trailing
            entry_trailing_threshold=float(cs.get("entryTrailingThreshold", 1.0)) / 100,
            entry_trailing_retracement=float(cs.get("entryTrailingRetracement", 0.5)) / 100,
            trailing_grid_ratio=float(cs.get("trailingGridRatio", 0.5)),
            # Risk
            wallet_exposure_limit=float(cs.get("walletExposureLimit", 1.0)),
            max_loss_percent=float(cs.get("maxLossPercent", 20)),
            hsl_red_threshold=float(cs.get("hslRedThreshold", 50)) / 100,
            # Unstuck
            unstuck_threshold=float(cs.get("unstuckThreshold", 3)) / 100,
            unstuck_close_pct=float(cs.get("unstuckClosePct", 5)) / 100,
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
            "nPositions": session.n_positions,
            "gridSpacingPct": session.grid_spacing_pct * 100,
            "emaSpan0": session.ema_span_0,
            "emaSpan1": session.ema_span_1,
            "initialEmaDist": session.initial_ema_dist * 100,
            "initialQtyPct": session.initial_qty_pct * 100,
            "doubleDownFactor": session.double_down_factor,
            "walletExposureLimit": session.wallet_exposure_limit,
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
            grid=GridConfig(
                N_POSITIONS=session.n_positions,
                GRID_SPACING_PCT=session.grid_spacing_pct,
                GRID_SPACING_VOLATILITY_WEIGHT=session.grid_spacing_volatility_weight,
            ),
            entry=EntryConfig(
                EMA_SPAN_0=session.ema_span_0,
                EMA_SPAN_1=session.ema_span_1,
                INITIAL_EMA_DIST=session.initial_ema_dist,
                INITIAL_QTY_PCT=session.initial_qty_pct,
                DOUBLE_DOWN_FACTOR=session.double_down_factor,
            ),
            exit=ExitConfig(
                CLOSE_GRID_MARKUP_START=session.close_grid_markup_start,
                CLOSE_GRID_MARKUP_END=session.close_grid_markup_end,
                CLOSE_GRID_QTY_PCT=session.close_grid_qty_pct,
            ),
            trailing=TrailingConfig(
                ENTRY_THRESHOLD_PCT=session.entry_trailing_threshold,
                ENTRY_RETRACEMENT_PCT=session.entry_trailing_retracement,
                ENTRY_TRAILING_GRID_RATIO=session.trailing_grid_ratio,
            ),
            unstuck=UnstuckConfig(
                THRESHOLD=session.unstuck_threshold,
                CLOSE_PCT=session.unstuck_close_pct,
            ),
            risk=RiskConfig(
                WALLET_EXPOSURE_LIMIT=session.wallet_exposure_limit,
                MAX_LOSS_PERCENT=session.max_loss_percent,
                HSL_RED_THRESHOLD=session.hsl_red_threshold,
                TAKER_FEE_RATE=session.taker_fee_rate,
                MAKER_FEE_RATE=session.maker_fee_rate,
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                               SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

def get_session_manager() -> SessionManager:
    """SessionManager singleton olish"""
    return SessionManager.get_instance()
