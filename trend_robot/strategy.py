"""
Trend Following Robot — Smart Trend Strategy v2.0

Mantiq:
1. EMA(fast) Golden/Death Cross EMA(slow) — asosiy signal
2. Supertrend direction aligned — tasdiq
3. ADX > threshold — trend kuchi filtri
4. Higher Timeframe (4H) EMA trend — kattabosib filter (optional)

Exit:
- Initial SL (fixed %)
- 3-phase trailing stop (None → Breakeven → Trailing with ATR floor)
- Partial TP (33% at 2%, 33% at 5%, trailing qoladi) — optional
- Circuit breaker (max drawdown)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .config import RobotConfig
from .indicators import (
    ADXIndicator,
    ATRIndicator,
    Candle,
    EMACrossover,
    EMAIndicator,
    SupertrendIndicator,
    calculate_pnl,
    money_round,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                             ENUMS & DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class TrailingPhase(Enum):
    NONE = "none"          # Fixed SL only
    BREAKEVEN = "breakeven"  # SL moved to entry
    TRAILING = "trailing"    # ATR trailing


@dataclass
class Position:
    """Ochiq pozitsiya"""
    id: str
    side: str                # "long" or "short"
    entry_price: float
    size: float
    opened_at: str = ""
    opened_ts: float = 0.0
    entry_fee: float = 0.0

    # Trailing stop state
    trailing_phase: TrailingPhase = TrailingPhase.NONE
    peak_price: float = 0.0
    stop_price: float = 0.0

    # Partial TP tracking
    partial_tp1_done: bool = False
    partial_tp2_done: bool = False
    initial_size: float = 0.0

    # Exchange data
    unrealized_pnl: float = 0.0
    margin: float = 0.0
    leverage: int = 0
    liquidation_price: float = 0.0
    mark_price: float = 0.0
    margin_mode: str = "crossed"
    roe: float = 0.0

    def __post_init__(self):
        if self.initial_size == 0.0:
            self.initial_size = self.size
        if self.opened_ts == 0.0:
            self.opened_ts = time.time()

    @property
    def notional(self) -> float:
        return self.entry_price * self.size

    def pnl_at(self, price: float) -> float:
        return calculate_pnl(self.entry_price, price, self.size, self.side)

    def pnl_pct_at(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "long":
            return (price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - price) / self.entry_price * 100

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "opened_at": self.opened_at,
            "opened_ts": self.opened_ts,
            "trailing_phase": self.trailing_phase.value,
            "stop_price": self.stop_price,
            "peak_price": self.peak_price,
            "partial_tp1_done": self.partial_tp1_done,
            "partial_tp2_done": self.partial_tp2_done,
            "initial_size": self.initial_size,
            "margin": self.margin,
            "leverage": self.leverage,
            "liquidation_price": self.liquidation_price,
            "mark_price": self.mark_price,
            "margin_mode": self.margin_mode,
            "roe": self.roe,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                             STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════


class TrendStrategy:
    """
    Smart Trend Following v2.0
    EMA Crossover + Supertrend + ADX + Higher-Timeframe filter
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self.cfg = config.trend  # shortcut

        # Indikatorlar
        self.ema = EMACrossover(
            fast_period=self.cfg.ema_fast,
            slow_period=self.cfg.ema_slow,
        )
        self.atr = ATRIndicator(period=self.cfg.atr_period)
        self.adx = ADXIndicator(period=self.cfg.adx_period)
        self.supertrend = SupertrendIndicator(
            period=self.cfg.supertrend_period,
            multiplier=self.cfg.supertrend_multiplier,
        )

        # HTF (Higher Timeframe) indikatorlar
        self.htf_ema_fast = EMAIndicator(period=self.cfg.htf_ema_fast)
        self.htf_ema_slow = EMAIndicator(period=self.cfg.htf_ema_slow)

        # Pozitsiya (bitta pozitsiya — trend bir tomonga)
        self.position: Optional[Position] = None

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.realized_pnl: float = 0.0
        self.today_trades = 0
        self._today_start_ts: float = 0.0
        self._peak_balance: float = 0.0
        self._max_drawdown_percent: float = 0.0
        self._stopped: bool = False

        # Cooldown after SL (tick counter)
        self._cooldown_until_tick: int = 0

        # Last ema cross info
        self._last_cross: Optional[str] = None
        self._last_signal_ts: float = 0.0

        self._symbol_info: Optional[dict] = None

        logger.info(
            f"TrendStrategy v2.0 created: EMA({self.cfg.ema_fast}/{self.cfg.ema_slow}), "
            f"ADX>{self.cfg.adx_threshold}, ST({self.cfg.supertrend_period}×{self.cfg.supertrend_multiplier}), "
            f"HTF={self.cfg.use_htf_filter}, PartialTP={self.cfg.use_partial_tp}"
        )

    def set_symbol_info(self, info: dict) -> None:
        self._symbol_info = info

    # ─── INDIKATORLAR ────────────────────────────────────────────────────────

    def update_indicators(self, candles: List[Candle]) -> None:
        """Asosiy timeframe indikatorlarini yangilash."""
        if not candles:
            return
        # Recompute indicators from scratch
        self.ema = EMACrossover(
            fast_period=self.cfg.ema_fast,
            slow_period=self.cfg.ema_slow,
        )
        self.atr = ATRIndicator(period=self.cfg.atr_period)
        self.adx = ADXIndicator(period=self.cfg.adx_period)
        self.supertrend = SupertrendIndicator(
            period=self.cfg.supertrend_period,
            multiplier=self.cfg.supertrend_multiplier,
        )

        # Update through all candles (keeps track of last cross for signal detection)
        last_cross = None
        for c in candles:
            cross = self.ema.update(c)
            self.atr.update(c)
            self.adx.update(c)
            self.supertrend.update(c)
            if cross:
                last_cross = cross
                self._last_signal_ts = c.timestamp

        self._last_cross = last_cross

    def update_htf_indicators(self, htf_candles: List[Candle]) -> None:
        """Higher Timeframe EMA fast/slow yangilash."""
        if not htf_candles:
            return
        self.htf_ema_fast = EMAIndicator(period=self.cfg.htf_ema_fast)
        self.htf_ema_slow = EMAIndicator(period=self.cfg.htf_ema_slow)
        for c in htf_candles:
            self.htf_ema_fast.update(c.close)
            self.htf_ema_slow.update(c.close)

    # ─── SIGNAL ──────────────────────────────────────────────────────────────

    def detect_signal(self, current_price: float, tick: int = 0) -> SignalType:
        """
        Signal aniqlash — barcha qatlamlar rozi bo'lishi kerak.
        EMA Crossover (moment) + Supertrend + ADX + HTF (optional)
        """
        if not self.ema.initialized:
            return SignalType.NONE
        if not self.atr.initialized or not self.adx.initialized:
            return SignalType.NONE
        if not self.supertrend.initialized:
            return SignalType.NONE

        # Cooldown check
        if tick < self._cooldown_until_tick:
            return SignalType.NONE

        # 1. EMA cross (oxirgi candle'da aniq cross bo'lgan bo'lishi kerak)
        if self._last_cross is None:
            return SignalType.NONE

        signal = SignalType.LONG if self._last_cross == "golden_cross" else SignalType.SHORT

        # Clear last_cross so we don't re-enter on same cross
        # (Will be set again on next real cross)
        # Note: we don't clear here — robot manages entry state
        # But to prevent repeated signals, check timestamp freshness

        # 2. ADX filter
        if self.adx.value < self.cfg.adx_threshold:
            return SignalType.NONE

        # 3. Supertrend alignment
        st_dir = self.supertrend.direction
        if signal == SignalType.LONG and st_dir != 1:
            return SignalType.NONE
        if signal == SignalType.SHORT and st_dir != -1:
            return SignalType.NONE

        # 4. HTF filter (optional)
        if self.cfg.use_htf_filter:
            if self.htf_ema_fast.initialized and self.htf_ema_slow.initialized:
                htf_fast = self.htf_ema_fast.value
                htf_slow = self.htf_ema_slow.value
                if signal == SignalType.LONG and htf_fast < htf_slow:
                    return SignalType.NONE
                if signal == SignalType.SHORT and htf_fast > htf_slow:
                    return SignalType.NONE

        return signal

    def consume_signal(self) -> None:
        """Signal ishlatilganidan keyin chaqiriladi — takroran trigger bo'lmasin."""
        self._last_cross = None

    # ─── POSITION SIZING ────────────────────────────────────────────────────

    def calculate_position_size(self, price: float, trade_amount: float,
                                leverage: int) -> float:
        """Position size = (trade_amount × leverage) / price."""
        if price <= 0 or trade_amount <= 0:
            return 0.0
        notional = trade_amount * leverage
        size = notional / price

        min_lot = 0.001
        max_lot = 10000.0
        if self._symbol_info:
            try:
                min_lot = max(min_lot, float(self._symbol_info.get("minTradeNum", 0)))
                max_lot = min(max_lot, float(self._symbol_info.get("maxTradeNum", max_lot)))
            except (TypeError, ValueError):
                pass

        if size < min_lot:
            return 0.0
        if size > max_lot:
            size = max_lot
        return money_round(size, 6)

    # ─── SL / TP ─────────────────────────────────────────────────────────────

    def initial_stop_loss(self, side: str, entry_price: float) -> float:
        sl_pct = self.cfg.initial_sl_percent / 100
        if side == "long":
            return entry_price * (1 - sl_pct)
        return entry_price * (1 + sl_pct)

    def update_trailing_stop(self, pos: Position, current_price: float) -> None:
        """3-phase trailing stop update."""
        if self.atr.value <= 0:
            return
        activation_pct = self.cfg.trailing_activation_percent / 100
        trail_distance = self.cfg.trailing_atr_multiplier * self.atr.value

        # Phase 1 → Phase 2: Breakeven
        if pos.trailing_phase == TrailingPhase.NONE:
            pnl_pct = pos.pnl_pct_at(current_price) / 100
            if pnl_pct >= activation_pct:
                pos.trailing_phase = TrailingPhase.BREAKEVEN
                pos.stop_price = pos.entry_price
                pos.peak_price = current_price

        # Phase 2 → Phase 3: Trailing
        if pos.trailing_phase == TrailingPhase.BREAKEVEN:
            if pos.side == "long":
                if current_price > pos.peak_price:
                    pos.peak_price = current_price
                    trail_stop = pos.peak_price - trail_distance
                    if trail_stop > pos.stop_price:
                        pos.stop_price = trail_stop
                        pos.trailing_phase = TrailingPhase.TRAILING
            else:
                if current_price < pos.peak_price or pos.peak_price == 0:
                    pos.peak_price = current_price
                    trail_stop = pos.peak_price + trail_distance
                    if trail_stop < pos.stop_price or pos.stop_price == 0:
                        pos.stop_price = trail_stop
                        pos.trailing_phase = TrailingPhase.TRAILING

        # Phase 3: Continue trailing
        if pos.trailing_phase == TrailingPhase.TRAILING:
            if pos.side == "long":
                if current_price > pos.peak_price:
                    pos.peak_price = current_price
                    trail_stop = pos.peak_price - trail_distance
                    if trail_stop > pos.stop_price:
                        pos.stop_price = trail_stop
            else:
                if current_price < pos.peak_price:
                    pos.peak_price = current_price
                    trail_stop = pos.peak_price + trail_distance
                    if trail_stop < pos.stop_price:
                        pos.stop_price = trail_stop

    def check_stop_hit(self, pos: Position, high: float, low: float) -> Optional[float]:
        """SL urilganmi? Return exit price if hit."""
        if pos.stop_price <= 0:
            return None
        if pos.side == "long" and low <= pos.stop_price:
            return pos.stop_price
        if pos.side == "short" and high >= pos.stop_price:
            return pos.stop_price
        return None

    def check_partial_tp(self, pos: Position, high: float,
                        low: float) -> Optional[Tuple[float, float]]:
        """
        Partial TP check — returns (exit_price, close_size) if hit, else None.
        """
        if not self.cfg.use_partial_tp:
            return None

        # Partial TP 1
        if not pos.partial_tp1_done:
            tp1_pct = self.cfg.partial_tp1_percent / 100
            if pos.side == "long":
                tp1_price = pos.entry_price * (1 + tp1_pct)
                if high >= tp1_price:
                    pos.partial_tp1_done = True
                    close_size = pos.initial_size * self.cfg.partial_tp1_size_pct
                    close_size = min(close_size, pos.size)
                    return (tp1_price, close_size)
            else:
                tp1_price = pos.entry_price * (1 - tp1_pct)
                if low <= tp1_price:
                    pos.partial_tp1_done = True
                    close_size = pos.initial_size * self.cfg.partial_tp1_size_pct
                    close_size = min(close_size, pos.size)
                    return (tp1_price, close_size)

        # Partial TP 2
        if pos.partial_tp1_done and not pos.partial_tp2_done:
            tp2_pct = self.cfg.partial_tp2_percent / 100
            if pos.side == "long":
                tp2_price = pos.entry_price * (1 + tp2_pct)
                if high >= tp2_price:
                    pos.partial_tp2_done = True
                    close_size = pos.initial_size * self.cfg.partial_tp2_size_pct
                    close_size = min(close_size, pos.size)
                    return (tp2_price, close_size)
            else:
                tp2_price = pos.entry_price * (1 - tp2_pct)
                if low <= tp2_price:
                    pos.partial_tp2_done = True
                    close_size = pos.initial_size * self.cfg.partial_tp2_size_pct
                    close_size = min(close_size, pos.size)
                    return (tp2_price, close_size)

        return None

    # ─── POSITION MANAGEMENT ────────────────────────────────────────────────

    def create_position(self, side: str, entry_price: float, size: float,
                       order_id: str = "") -> Position:
        pos = Position(
            id=order_id or f"{side}-{int(time.time() * 1000)}",
            side=side,
            entry_price=entry_price,
            size=size,
            initial_size=size,
            opened_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            opened_ts=time.time(),
            entry_fee=entry_price * size * self.config.risk.TAKER_FEE_RATE,
        )
        pos.stop_price = self.initial_stop_loss(side, entry_price)
        self.position = pos
        return pos

    def set_cooldown(self, current_tick: int) -> None:
        """SL urilganidan keyin cooldown o'rnatish."""
        self._cooldown_until_tick = current_tick + self.cfg.cooldown_bars_after_sl

    # ─── RISK ────────────────────────────────────────────────────────────────

    def check_max_drawdown(self, current_equity: float, initial_equity: float) -> bool:
        """Circuit breaker: equity too low."""
        if self.cfg.max_drawdown_percent <= 0 or initial_equity <= 0:
            return False
        loss_pct = (initial_equity - current_equity) / initial_equity * 100
        return loss_pct >= self.cfg.max_drawdown_percent

    def can_trade_today(self) -> bool:
        now = time.time()
        day_start = now - (now % 86400)
        if day_start > self._today_start_ts:
            self._today_start_ts = day_start
            self.today_trades = 0
        # Default 99 trades/day if not configured
        max_per_day = getattr(self.config.risk, "TRADES_PER_DAY", 99)
        return self.today_trades < max_per_day

    def increment_today_trades(self) -> None:
        self.today_trades += 1

    def stop_trading(self) -> None:
        self._stopped = True

    def should_stop_trading(self) -> bool:
        return self._stopped

    def update_drawdown(self, current_balance: float) -> None:
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        if self._peak_balance > 0:
            dd = (self._peak_balance - current_balance) / self._peak_balance * 100
            if dd > self._max_drawdown_percent:
                self._max_drawdown_percent = dd

    # ─── STATS ───────────────────────────────────────────────────────────────

    def on_trade_closed(self, net_pnl: float) -> None:
        self.total_trades += 1
        self.realized_pnl += net_pnl
        if net_pnl > 0:
            self.winning_trades += 1
        elif net_pnl < 0:
            self.losing_trades += 1
        self.increment_today_trades()

    def winrate(self) -> float:
        closed = self.winning_trades + self.losing_trades
        return (self.winning_trades / closed * 100) if closed > 0 else 0.0

    def get_status(self) -> Dict:
        return {
            "ema_fast": self.ema.fast_value,
            "ema_slow": self.ema.slow_value,
            "atr": self.atr.value,
            "adx": self.adx.value,
            "supertrend_value": self.supertrend.value,
            "supertrend_direction": self.supertrend.direction,
            "htf_ema_fast": self.htf_ema_fast.value,
            "htf_ema_slow": self.htf_ema_slow.value,
            "position": self.position.to_dict() if self.position else None,
            "last_cross": self._last_cross,
        }

    def get_stats(self) -> Dict:
        return {
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "winrate": self.winrate(),
            },
            "profit": self.realized_pnl,
            "drawdown": self._max_drawdown_percent,
            "today_trades": self.today_trades,
        }

    def reset(self) -> None:
        self.position = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.realized_pnl = 0.0
        self.today_trades = 0
        self._stopped = False


# Backward compatibility
TrendFollowingStrategy = TrendStrategy
