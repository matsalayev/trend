"""
Trend Robot - Strategy Skeleton

Strategiya logikasi keyinchalik qayta yoziladi. Hozir check_signal() har doim
SignalType.NONE qaytaradi — bot savdo qilmaydi, faqat infratuzilma ishlaydi.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import RobotConfig
from .indicators import Candle, money_round

logger = logging.getLogger(__name__)


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class TrailingPhase(Enum):
    """3-phase trailing stop placeholder"""
    NONE = "none"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"


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
    """Trailing stop holati (placeholder)"""
    phase: TrailingPhase = TrailingPhase.NONE
    peak_price: float = 0.0
    stop_price: float = 0.0
    activated_at: float = 0.0


class TrendStrategy:
    """
    Trend strategy skeleton — hozircha signal generatsiyasi yo'q.

    check_signal() har doim SignalType.NONE qaytaradi. Strategiya qayta
    yozilganda barcha placeholder metodlar haqiqiy implementatsiya bilan
    almashtiriladi.
    """

    def __init__(self, config: RobotConfig):
        self.config = config

        # Pozitsiyalar
        self.long_position: Optional[Position] = None
        self.short_position: Optional[Position] = None

        # Trailing state (placeholder)
        self.trailing_long = TrailingState()
        self.trailing_short = TrailingState()

        # State
        self._last_signal: SignalType = SignalType.NONE
        self._symbol_info: Optional[dict] = None

        logger.info("TrendStrategy skeleton yaratildi (signal disabled)")

    # ─── Market data hook ──────────────────────────────────────────────────

    def update_candles(self, candles: List[Candle]) -> None:
        """Indikatorlarni yangilash — skeleton da no-op"""
        return None

    # ─── Signal generation ─────────────────────────────────────────────────

    def check_signal(
        self,
        candles: Optional[List[Candle]] = None,
        current_price: float = 0.0,
        mtf_candles: Optional[List[Candle]] = None,
    ) -> SignalType:
        """Placeholder — strategy qayta yozilganda real logika qo'shiladi."""
        return SignalType.NONE

    # ─── Exit placeholders ─────────────────────────────────────────────────

    def update_trailing_stop(
        self, position: Position, current_price: float
    ) -> Optional[float]:
        """Placeholder — trailing stop hozir faol emas"""
        return None

    def check_trailing_stop_hit(self, position: Position, current_price: float) -> bool:
        """Placeholder — trailing stop hit detection"""
        return False

    def check_opposite_signal_exit(self, position: Position) -> bool:
        """Placeholder — opposite signal exit"""
        return False

    def on_trade_closed(self, side: str) -> None:
        """Pozitsiya yopilgandan keyin trailing state reset (no-op placeholder)"""
        if side == "long":
            self.trailing_long = TrailingState()
        else:
            self.trailing_short = TrailingState()

    # Backward-compat alias (robot.py hali ishlatishi mumkin)
    def reset_trailing(self, side: str) -> None:
        self.on_trade_closed(side)

    # ─── Position sizing ───────────────────────────────────────────────────

    def calculate_position_size(self, price: float, balance: float) -> float:
        """Size = balance * capital_engagement * leverage / price"""
        if price <= 0 or balance <= 0:
            return 0.0
        margin = balance * self.config.CAPITAL_ENGAGEMENT
        size = (margin * self.config.trading.LEVERAGE) / price
        min_qty = self._get_min_qty()
        return money_round(size, 6) if size >= min_qty else 0.0

    def calculate_sl_price(self, entry_price: float, side: str) -> float:
        """Stop loss narxi — fixed % ichi ga qaratilgan"""
        sl_pct = self.config.exit.SL_PERCENT
        if side == "long":
            return entry_price * (1 - sl_pct / 100)
        return entry_price * (1 + sl_pct / 100)

    # ─── Helpers ───────────────────────────────────────────────────────────

    def set_symbol_info(self, info: dict) -> None:
        self._symbol_info = info

    def _get_min_qty(self) -> float:
        if self._symbol_info and "minTradeNum" in self._symbol_info:
            return float(self._symbol_info["minTradeNum"])
        return 0.001

    def get_status(self) -> dict:
        return {
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
