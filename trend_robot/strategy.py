"""
Trend Robot - Core Trading Strategy

EMA Crossover + Ichimoku Cloud + ADX trend strength + multi-timeframe.
bitget-futures-ema + HEMA Hedge bot dan ilhomlangan trend-following bot.

Signal generation (multi-layer):
    1. EMA Crossover (primary): Golden Cross = LONG, Death Cross = SHORT
    2. Ichimoku Confirmation: Price vs Cloud + Tenkan/Kijun + Chikou
    3. ADX Filter: ADX > 25 = kuchli trend
    4. Multi-timeframe: 5m signal + 15m/1H confirmation

Exit management:
    1. Trailing stop (3-phase): None → Breakeven → Trail
    2. Opposite signal: Death cross exits LONG
    3. Stop loss: Fixed SL as fallback
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import RobotConfig
from .indicators import (
    Candle, EMACrossover, IchimokuCloud, ADXIndicator, ATRIndicator,
    money_round, calculate_pnl
)

logger = logging.getLogger(__name__)


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class TrailingPhase(Enum):
    """3-phase trailing stop — HEMA RSI bot dan"""
    NONE = "none"           # Trailing faol emas
    BREAKEVEN = "breakeven" # SL break-even ga ko'tarildi
    TRAILING = "trailing"   # SL narx ortidan harakatlanadi


@dataclass
class Position:
    """Ochiq pozitsiya"""
    id: str
    side: str
    entry_price: float
    size: float
    unrealized_pnl: float = 0.0
    margin: float = 0.0
    leverage: int = 0
    opened_at: str = ""
    grid_level: int = 0


@dataclass
class TrailingState:
    """Trailing stop holati"""
    phase: TrailingPhase = TrailingPhase.NONE
    peak_price: float = 0.0    # Eng yuqori profit narxi
    stop_price: float = 0.0    # Joriy stop narxi
    activated_at: float = 0.0  # Qachon faollashdi


class TrendStrategy:
    """
    Trend-following strategiya — EMA + Ichimoku + ADX.

    Signal flow:
        1. EMA Crossover → golden_cross / death_cross
        2. Ichimoku → price above/below cloud (optional confirmation)
        3. ADX → trend kuchli (> threshold)
        4. MTF → yuqori timeframe confirmation (optional)

    Agar barcha layer lar tasdiqlasa → entry.
    """

    def __init__(self, config: RobotConfig):
        self.config = config

        # Indikatorlar
        self.ema_crossover = EMACrossover(
            fast_period=config.ema.FAST_PERIOD,
            slow_period=config.ema.SLOW_PERIOD
        )
        self.ichimoku = IchimokuCloud(
            tenkan_period=config.ichimoku.TENKAN_PERIOD,
            kijun_period=config.ichimoku.KIJUN_PERIOD,
            senkou_b_period=config.ichimoku.SENKOU_B_PERIOD,
            displacement=config.ichimoku.DISPLACEMENT,
        ) if config.ichimoku.ENABLED else None

        self.adx = ADXIndicator(period=config.trend.ADX_PERIOD)
        self.atr = ATRIndicator(period=14)

        # MTF indikatorlar (alohida timeframe)
        self.mtf_ema: Optional[EMACrossover] = None
        if config.mtf.ENABLED:
            self.mtf_ema = EMACrossover(
                fast_period=config.ema.FAST_PERIOD,
                slow_period=config.ema.SLOW_PERIOD
            )

        # Pozitsiyalar
        self.long_position: Optional[Position] = None
        self.short_position: Optional[Position] = None

        # Trailing stop
        self.trailing_long = TrailingState()
        self.trailing_short = TrailingState()

        # State
        self._last_signal: SignalType = SignalType.NONE
        self._last_signal_candle_ts: int = 0
        self._tick_count: int = 0
        self._last_processed_ts: int = 0
        self._symbol_info = None
        self._initial_state_checked: bool = False

        logger.info("TrendStrategy yaratildi")

    def check_signal(
        self,
        candles: List[Candle],
        current_price: float,
        mtf_candles: Optional[List[Candle]] = None,
    ) -> SignalType:
        """
        Multi-layer signal tekshirish.

        Layer 1: EMA Crossover
        Layer 2: Ichimoku confirmation (optional)
        Layer 3: ADX filter
        Layer 4: MTF confirmation (optional)

        Args:
            candles: Primary timeframe shamchalar
            current_price: Joriy narx
            mtf_candles: Yuqori timeframe shamchalar (optional)

        Returns:
            SignalType
        """
        if not candles or len(candles) < max(self.config.ema.SLOW_PERIOD, 52) + 5:
            return SignalType.NONE

        # Indikatorlarni yangilash
        ema_signal = None
        for candle in candles:
            if candle.timestamp <= self._last_processed_ts:
                continue
            ema_signal = self.ema_crossover.update(candle.close)
            self.adx.update(candle)
            self.atr.update(candle)
            if self.ichimoku:
                self.ichimoku.update(candle)
        if candles:
            self._last_processed_ts = candles[-1].timestamp

        if not self.ema_crossover.initialized:
            return SignalType.NONE

        # Bir xil shamchada ikki marta signal bermaslik
        if candles[-1].timestamp == self._last_signal_candle_ts:
            return SignalType.NONE

        # ─── Layer 1: EMA Crossover ─────────────────────────────────────────
        # Agar crossover hali bo'lmagan bo'lsa, lekin EMA state aniq bo'lsa — birinchi signal sifatida ishlatish
        if ema_signal is None and not self._initial_state_checked:
            self._initial_state_checked = True
            fast = self.ema_crossover.fast.value
            slow = self.ema_crossover.slow.value
            if fast is not None and slow is not None:
                if fast > slow:
                    ema_signal = 'golden_cross'
                    logger.info(f'Initial EMA state: fast({fast:.2f}) > slow({slow:.2f}) → synthetic golden_cross')
                elif slow > fast:
                    ema_signal = 'death_cross'
                    logger.info(f'Initial EMA state: slow({slow:.2f}) > fast({fast:.2f}) → synthetic death_cross')

        if ema_signal is None:
            return SignalType.NONE

        is_long = ema_signal == "golden_cross"
        is_short = ema_signal == "death_cross"

        if not is_long and not is_short:
            return SignalType.NONE

        # ─── Layer 2: Ichimoku confirmation ──────────────────────────────────
        if self.ichimoku and self.config.ichimoku.ENABLED and self.ichimoku.initialized:
            if is_long and not self.ichimoku.is_price_above_cloud(current_price):
                logger.debug(f"Ichimoku blokadi: price {current_price:.2f} cloud ichida/ostida")
                return SignalType.NONE
            if is_short and not self.ichimoku.is_price_below_cloud(current_price):
                logger.debug(f"Ichimoku blokadi: price {current_price:.2f} cloud ichida/tepasida")
                return SignalType.NONE

        # ─── Layer 3: ADX filter ─────────────────────────────────────────────
        if self.adx.initialized:
            if not self.adx.is_trending(self.config.trend.ADX_THRESHOLD):
                logger.debug(
                    f"ADX blokadi: ADX={self.adx.value:.1f} < {self.config.trend.ADX_THRESHOLD}"
                )
                return SignalType.NONE

        # ─── Layer 4: MTF confirmation ───────────────────────────────────────
        if self.config.mtf.ENABLED and self.mtf_ema and mtf_candles:
            for c in mtf_candles:
                self.mtf_ema.update(c.close)

            if self.mtf_ema.initialized:
                # MTF da ham trend yo'nalishi bir xil bo'lishi kerak
                mtf_fast = self.mtf_ema.fast.value
                mtf_slow = self.mtf_ema.slow.value
                if is_long and mtf_fast <= mtf_slow:
                    logger.debug("MTF blokadi: yuqori TF da trend pastga")
                    return SignalType.NONE
                if is_short and mtf_fast >= mtf_slow:
                    logger.debug("MTF blokadi: yuqori TF da trend tepaga")
                    return SignalType.NONE

        # ─── Signal tasdiqlandi ──────────────────────────────────────────────
        self._last_signal_candle_ts = candles[-1].timestamp

        if is_long:
            self._last_signal = SignalType.LONG
            logger.info(
                f"LONG SIGNAL: Golden Cross, "
                f"EMA fast={self.ema_crossover.fast.value:.2f} > slow={self.ema_crossover.slow.value:.2f}, "
                f"ADX={self.adx.value:.1f}"
            )
            return SignalType.LONG
        else:
            self._last_signal = SignalType.SHORT
            logger.info(
                f"SHORT SIGNAL: Death Cross, "
                f"EMA fast={self.ema_crossover.fast.value:.2f} < slow={self.ema_crossover.slow.value:.2f}, "
                f"ADX={self.adx.value:.1f}"
            )
            return SignalType.SHORT

    # ═══════════════════════════════════════════════════════════════════════════
    #                           TRAILING STOP
    # ═══════════════════════════════════════════════════════════════════════════

    def update_trailing_stop(
        self, position: Position, current_price: float
    ) -> Optional[float]:
        """
        3-phase trailing stop — HEMA RSI bot dan 1:1.

        Phase 1 (NONE): Trailing faol emas — faqat SL
        Phase 2 (BREAKEVEN): Profit ACTIVATE% ga yetdi → SL break-even ga
        Phase 3 (TRAILING): Profit davom etsa → SL narx ortidan harakatlanadi

        Args:
            position: Ochiq pozitsiya
            current_price: Joriy narx

        Returns:
            Yangi stop price (agar o'zgargan bo'lsa), None agar o'zgarmagan
        """
        if not self.config.exit.USE_TRAILING_STOP:
            return None

        trailing = self.trailing_long if position.side == "long" else self.trailing_short
        pnl_pct = abs(current_price - position.entry_price) / position.entry_price * 100

        is_long = position.side == "long"

        # Phase 1 → 2: Activation
        if trailing.phase == TrailingPhase.NONE:
            if pnl_pct >= self.config.exit.TRAILING_ACTIVATE_PCT:
                trailing.phase = TrailingPhase.BREAKEVEN
                trailing.stop_price = position.entry_price  # Break-even
                trailing.peak_price = current_price
                trailing.activated_at = time.time()
                logger.info(
                    f"Trailing ACTIVATED: {position.side} break-even @ ${position.entry_price:.2f}"
                )
                return trailing.stop_price

        # Phase 2 → 3: Trail
        elif trailing.phase == TrailingPhase.BREAKEVEN:
            # Peak tracking
            if is_long and current_price > trailing.peak_price:
                trailing.peak_price = current_price
            elif not is_long and current_price < trailing.peak_price:
                trailing.peak_price = current_price

            # Trail mode ga o'tish
            floor_pct = self.config.exit.TRAILING_FLOOR_PCT
            if is_long:
                new_stop = trailing.peak_price * (1 - floor_pct / 100)
                if new_stop > trailing.stop_price:
                    trailing.stop_price = new_stop
                    trailing.phase = TrailingPhase.TRAILING
                    return trailing.stop_price
            else:
                new_stop = trailing.peak_price * (1 + floor_pct / 100)
                if new_stop < trailing.stop_price or trailing.stop_price == position.entry_price:
                    trailing.stop_price = new_stop
                    trailing.phase = TrailingPhase.TRAILING
                    return trailing.stop_price

        # Phase 3: Continue trailing
        elif trailing.phase == TrailingPhase.TRAILING:
            if is_long and current_price > trailing.peak_price:
                trailing.peak_price = current_price
                floor_pct = self.config.exit.TRAILING_FLOOR_PCT
                new_stop = trailing.peak_price * (1 - floor_pct / 100)
                if new_stop > trailing.stop_price:
                    trailing.stop_price = new_stop
                    return trailing.stop_price
            elif not is_long and current_price < trailing.peak_price:
                trailing.peak_price = current_price
                floor_pct = self.config.exit.TRAILING_FLOOR_PCT
                new_stop = trailing.peak_price * (1 + floor_pct / 100)
                if new_stop < trailing.stop_price:
                    trailing.stop_price = new_stop
                    return trailing.stop_price

        return None

    def check_trailing_stop_hit(self, position: Position, current_price: float) -> bool:
        """Trailing stop triggerlanganmi"""
        trailing = self.trailing_long if position.side == "long" else self.trailing_short

        if trailing.phase == TrailingPhase.NONE:
            return False

        if position.side == "long":
            return current_price <= trailing.stop_price
        else:
            return current_price >= trailing.stop_price

    # ═══════════════════════════════════════════════════════════════════════════
    #                           OPPOSITE SIGNAL EXIT
    # ═══════════════════════════════════════════════════════════════════════════

    def check_opposite_signal_exit(self, position: Position) -> bool:
        """
        Qarama-qarshi signal bilan chiqish.

        LONG pozitsiyada Death Cross → exit
        SHORT pozitsiyada Golden Cross → exit
        """
        if not self.config.exit.USE_OPPOSITE_SIGNAL_EXIT:
            return False

        if position.side == "long" and self._last_signal == SignalType.SHORT:
            logger.info("OPPOSITE EXIT: Death Cross → LONG pozitsiya yopilmoqda")
            return True
        if position.side == "short" and self._last_signal == SignalType.LONG:
            logger.info("OPPOSITE EXIT: Golden Cross → SHORT pozitsiya yopilmoqda")
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════════════
    #                           POSITION SIZING
    # ═══════════════════════════════════════════════════════════════════════════

    def calculate_position_size(self, price: float, balance: float) -> float:
        """Position size = balance * capital_engagement * leverage / price"""
        if price <= 0 or balance <= 0:
            return 0.0
        margin = balance * self.config.CAPITAL_ENGAGEMENT
        size = (margin * self.config.trading.LEVERAGE) / price
        min_qty = self._get_min_qty()
        return money_round(size, 6) if size >= min_qty else 0.0

    def calculate_sl_price(self, entry_price: float, side: str) -> float:
        """Stop loss narxi"""
        sl_pct = self.config.exit.SL_PERCENT
        if side == "long":
            return entry_price * (1 - sl_pct / 100)
        return entry_price * (1 + sl_pct / 100)

    def reset_trailing(self, side: str):
        """Trailing state reset (pozitsiya yopilganda)"""
        if side == "long":
            self.trailing_long = TrailingState()
        else:
            self.trailing_short = TrailingState()

    def set_symbol_info(self, info: dict):
        self._symbol_info = info

    def _get_min_qty(self) -> float:
        if self._symbol_info and "minTradeNum" in self._symbol_info:
            return float(self._symbol_info["minTradeNum"])
        return 0.001

    def get_status(self) -> dict:
        return {
            "ema": self.ema_crossover.get_state(),
            "ichimoku": self.ichimoku.get_state() if self.ichimoku else None,
            "adx": self.adx.get_state(),
            "atr": round(self.atr.value, 2),
            "last_signal": self._last_signal.value,
            "trailing_long": {
                "phase": self.trailing_long.phase.value,
                "stop_price": round(self.trailing_long.stop_price, 2),
                "peak_price": round(self.trailing_long.peak_price, 2),
            },
            "trailing_short": {
                "phase": self.trailing_short.phase.value,
                "stop_price": round(self.trailing_short.stop_price, 2),
                "peak_price": round(self.trailing_short.peak_price, 2),
            },
        }
