"""
Trend Following Robot — Smart Trend Strategy v2.1

Mantiq:
1. EMA(fast) Golden/Death Cross EMA(slow) — asosiy signal (yangi cross, eski emas)
2. Supertrend direction aligned — tasdiq
3. ADX > threshold — trend kuchi filtri
4. Higher Timeframe (4H) EMA trend — kattabosib filter (optional)

Exit:
- Initial SL (fixed % YOKI ATR-based, max(ikkalasi))
- 3-phase trailing stop (None → Breakeven → Trailing with ATR floor)
- Partial TP (33% at 2%, 33% at 5%, trailing qoladi) — optional
- Opposite signal close (death cross LONG'ni yopadi, golden cross SHORT'ni)
- Fee-aware voluntary exit (faqat net profit fee'ni qoplaganda)
- Circuit breaker (max drawdown)

v2.1 fixes:
- Position size formula: trade_amount = MARGIN, notional = margin * leverage (correct)
- Cooldown candle-bar based (timestamp), tick'da emas
- Stale EMA cross filter (max_signal_age_bars)
- ATR-adaptive stop loss
- Opposite signal exit
- Fee-aware exit threshold
- Trades-per-hour limit (loop bug himoyasi)
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
    ChoppinessIndex,
    EMACrossover,
    EMAIndicator,
    SupertrendIndicator,
    calculate_pnl,
    money_round,
)

logger = logging.getLogger(__name__)


# Timeframe → seconds (cooldown / signal-age conversion uchun)
_TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "1H": 3600, "2h": 7200, "2H": 7200,
    "4h": 14400, "4H": 14400, "6h": 21600, "6H": 21600,
    "12h": 43200, "12H": 43200, "1d": 86400, "1D": 86400,
}


def timeframe_to_seconds(tf: str) -> int:
    return _TIMEFRAME_SECONDS.get(tf, 900)  # default 15m


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
        self.chop = ChoppinessIndex(period=self.cfg.chop_period)

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

        # Cooldown after SL — TIMESTAMP asosida (ms), tick'da emas.
        # Robot timestamp asosida candle bar'ga aylantiradi.
        self._cooldown_until_ts: float = 0.0

        # Last ema cross info — timestamp ham saqlanadi (stale check uchun)
        self._last_cross: Optional[str] = None
        self._last_cross_ts: int = 0  # ms — last detected cross candle timestamp
        self._last_candle_ts: int = 0  # ms — most recent candle ts (signal age uchun)

        # Trades-per-hour tracking — fee-bleed loop himoyasi
        # Last N trade timestamp'lari (rolling window)
        self._recent_trade_ts: List[float] = []

        # Consecutive losses tracker (v2.1) — agar oxirgi N ta trade ketma-ket
        # zarar bo'lsa, qo'shimcha cooldown qo'yiladi (choppy market signal).
        self._consecutive_losses: int = 0
        self._streak_cooldown_until_ts: float = 0.0

        self._symbol_info: Optional[dict] = None

        logger.info(
            f"TrendStrategy v2.1 created: EMA({self.cfg.ema_fast}/{self.cfg.ema_slow}), "
            f"ADX>{self.cfg.adx_threshold}, ST({self.cfg.supertrend_period}×{self.cfg.supertrend_multiplier}), "
            f"HTF={self.cfg.use_htf_filter}, PartialTP={self.cfg.use_partial_tp}, "
            f"ATR_SL={self.cfg.use_atr_sl}, OppositeExit={self.cfg.use_opposite_signal_exit}, "
            f"CHOP_filter={self.cfg.use_choppiness_filter} (max={self.cfg.chop_max_for_entry}), "
            f"MaxTradesPerHour={self.cfg.max_trades_per_hour}, "
            f"ConsecLossThreshold={self.cfg.consecutive_losses_threshold}"
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
        self.chop = ChoppinessIndex(period=self.cfg.chop_period)

        # Update through all candles. Track last cross AND its timestamp for
        # stale-signal filtering. v2.0 had a bug where update_indicators would
        # re-discover an old cross (e.g. 50 candles ago) and re-trigger entry.
        # We now record cross_ts and detect_signal compares against the latest
        # candle timestamp.
        last_cross = None
        last_cross_ts = 0
        for c in candles:
            cross = self.ema.update(c)
            self.atr.update(c)
            self.adx.update(c)
            self.supertrend.update(c)
            self.chop.update(c)
            if cross:
                last_cross = cross
                last_cross_ts = c.timestamp

        self._last_cross = last_cross
        self._last_cross_ts = last_cross_ts
        self._last_candle_ts = candles[-1].timestamp

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

    # Internal: shared signal-quality gate used by both entry and opposite-exit.
    def _check_full_signal_filters(self, signal: "SignalType") -> bool:
        """ADX + Supertrend + HTF + CHOP filtrlar o'tdimi?"""
        # ADX filter
        if self.adx.value < self.cfg.adx_threshold:
            return False

        # Choppiness Index filter (v2.1) — choppy market'da entry rad etiladi
        # Diagnostika ko'rsatdi: yo'qotuvchi oynalardagi SL'lar
        # CHOP yuqori bo'lgan davrlarda (false trend signal) ko'p bo'ladi.
        if self.cfg.use_choppiness_filter and self.chop.initialized:
            if self.chop.value > self.cfg.chop_max_for_entry:
                return False

        # Supertrend alignment
        st_dir = self.supertrend.direction
        if signal == SignalType.LONG and st_dir != 1:
            return False
        if signal == SignalType.SHORT and st_dir != -1:
            return False
        # HTF filter (optional)
        if self.cfg.use_htf_filter:
            if self.htf_ema_fast.initialized and self.htf_ema_slow.initialized:
                htf_fast = self.htf_ema_fast.value
                htf_slow = self.htf_ema_slow.value
                if signal == SignalType.LONG and htf_fast < htf_slow:
                    return False
                if signal == SignalType.SHORT and htf_fast > htf_slow:
                    return False
        return True

    def _signal_age_bars(self) -> int:
        """Last cross necha candle bar oldin sodir bo'lgan."""
        if self._last_cross_ts <= 0 or self._last_candle_ts <= 0:
            return 999  # unknown — treat as stale
        bar_ms = timeframe_to_seconds(self.cfg.timeframe) * 1000
        if bar_ms <= 0:
            return 0
        return max(0, (self._last_candle_ts - self._last_cross_ts) // bar_ms)

    def _is_in_cooldown(self, now_ts: Optional[float] = None) -> bool:
        """Timestamp asosida cooldown — tick'lar emas. Streak cooldown ham hisoblanadi."""
        now = now_ts if now_ts is not None else time.time()
        if self._cooldown_until_ts > 0 and now < self._cooldown_until_ts:
            return True
        if self._streak_cooldown_until_ts > 0 and now < self._streak_cooldown_until_ts:
            return True
        return False

    def _is_rate_limited(self, now_ts: Optional[float] = None) -> bool:
        """Trades-per-hour limit (loop-bug himoyasi)."""
        if self.cfg.max_trades_per_hour <= 0:
            return False
        now = now_ts if now_ts is not None else time.time()
        # Rolling window: oxirgi soatda ochilgan trade'lar soni
        cutoff = now - 3600
        self._recent_trade_ts = [t for t in self._recent_trade_ts if t >= cutoff]
        return len(self._recent_trade_ts) >= self.cfg.max_trades_per_hour

    def detect_signal(self, current_price: float, tick: int = 0,
                      now_ts: Optional[float] = None) -> SignalType:
        """
        Signal aniqlash — barcha qatlamlar rozi bo'lishi kerak.
        EMA Crossover (moment, < max_signal_age_bars) + Supertrend + ADX + HTF (optional).

        Cooldown va trades-per-hour limit ham bu yerda enforce qilinadi.
        """
        if not self.ema.initialized:
            return SignalType.NONE
        if not self.atr.initialized or not self.adx.initialized:
            return SignalType.NONE
        if not self.supertrend.initialized:
            return SignalType.NONE

        # Cooldown check (timestamp asosida — tick'da emas)
        if self._is_in_cooldown(now_ts):
            return SignalType.NONE

        # Trades-per-hour limit (loop-bug guard)
        if self._is_rate_limited(now_ts):
            return SignalType.NONE

        # 1. EMA cross (oxirgi candle'da aniq cross bo'lgan bo'lishi kerak)
        if self._last_cross is None:
            return SignalType.NONE

        # 1a. Stale signal check — cross max_signal_age_bars dan eski emas.
        # update_indicators recompute qilganda eski cross ham topilishi mumkin,
        # bu esa noto'g'ri "yangi entry" signaliga olib keladi.
        age = self._signal_age_bars()
        if age > self.cfg.max_signal_age_bars:
            # Eski cross — endi rad etamiz va clear qilamiz, qayta ko'tarilmasin.
            self._last_cross = None
            return SignalType.NONE

        signal = SignalType.LONG if self._last_cross == "golden_cross" else SignalType.SHORT

        # 2-4. ADX + Supertrend + HTF filtrlar
        if not self._check_full_signal_filters(signal):
            return SignalType.NONE

        return signal

    def detect_opposite_exit(self, current_side: str) -> bool:
        """
        Opposite signal exit — ochiq pozitsiyani kuchli teskari signal'da yopish.

        LONG ochiq + death cross + (full filter confirm if enabled) => True (yopish kerak)
        SHORT ochiq + golden cross + ... => True
        """
        if not self.cfg.use_opposite_signal_exit:
            return False
        if not self.ema.initialized or self._last_cross is None:
            return False

        # Cross signal'i juda eski bo'lmasligi kerak
        if self._signal_age_bars() > self.cfg.max_signal_age_bars:
            return False

        # Opposite check
        is_opposite = (
            (current_side == "long" and self._last_cross == "death_cross")
            or (current_side == "short" and self._last_cross == "golden_cross")
        )
        if not is_opposite:
            return False

        # Optional: ADX/Supertrend/HTF ham tasdiqlashi shart
        if self.cfg.opposite_signal_requires_full_confirm:
            opposite_signal = (
                SignalType.SHORT if current_side == "long" else SignalType.LONG
            )
            if not self._check_full_signal_filters(opposite_signal):
                return False

        return True

    def consume_signal(self) -> None:
        """Signal ishlatilganidan keyin chaqiriladi — takroran trigger bo'lmasin."""
        self._last_cross = None
        self._last_cross_ts = 0

    # ─── POSITION SIZING ────────────────────────────────────────────────────

    def calculate_position_size(self, price: float, trade_amount: float,
                                leverage: int) -> float:
        """
        Position size hisoblash.

        SEMANTIKA (v2.1 — aniqlashtirilgan):
        - `trade_amount` = MARGIN to allocate (foydalanuvchi pozitsiyaga ajratayotgan kapital, USDT'da)
        - `notional` = trade_amount × leverage (real kontrakt qiymati)
        - `size` = notional / price (qancha unit/kontrakt)

        Misol: trade_amount=$50, leverage=10, AVAX@$9.24
          notional = $500, size = 54.1 AVAX, fee per side = $0.50

        KRITIK: Robot.py BU funksiyaga `_session_trade_amount` ni uzatadi.
        Agar u 0 bo'lsa, robot CAPITAL_ENGAGEMENT × balance ishlatadi
        (entire balance EMAS — bu eski v2.0 bug'i).
        """
        if price <= 0 or trade_amount <= 0:
            return 0.0
        notional = trade_amount * leverage
        size = notional / price

        # Symbol info'dan precision va min/max chegaralar
        min_lot = 0.001
        max_lot = 10000.0
        size_precision: Optional[int] = None
        if self._symbol_info:
            try:
                min_lot = max(min_lot, float(self._symbol_info.get("minTradeNum", 0) or 0))
                raw_max = self._symbol_info.get("maxTradeNum", max_lot) or max_lot
                max_lot = min(max_lot, float(raw_max))
            except (TypeError, ValueError):
                pass
            try:
                vp = self._symbol_info.get("volumePlace")
                if vp is not None:
                    size_precision = int(vp)
            except (TypeError, ValueError):
                pass

        if size < min_lot:
            return 0.0
        if size > max_lot:
            size = max_lot
        # Symbol-specific precision (e.g. BTC=4 decimals, AVAX=1)
        precision = size_precision if size_precision is not None else 6
        return money_round(size, precision)

    # ─── SL / TP ─────────────────────────────────────────────────────────────

    def initial_stop_loss(self, side: str, entry_price: float) -> float:
        """
        Initial SL — fixed % VA optionally ATR-based ning eng wide'i.

        v2.0: faqat fixed %.
        v2.1: agar use_atr_sl=True, SL = max(fixed%, sl_atr_mult * ATR / price * 100)
        Bu past-vol pair'larda fixed %, yuqori-vol'da ATR'ga moslashadi.
        """
        sl_pct = self.cfg.initial_sl_percent / 100

        if self.cfg.use_atr_sl and self.atr.initialized and self.atr.value > 0 and entry_price > 0:
            atr_pct = (self.cfg.sl_atr_multiplier * self.atr.value) / entry_price
            # Wider'ini olamiz — ya'ni ko'proq nafas oladigan SL
            sl_pct = max(sl_pct, atr_pct)

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

    def set_cooldown(self, _current_tick: int = 0, now_ts: Optional[float] = None) -> None:
        """
        SL urilganidan keyin cooldown o'rnatish.

        v2.0: tick'da bo'lgan (1s × 5 = 5 sek — juda qisqa).
        v2.1: candle bar'da. cooldown_bars_after_sl × timeframe_seconds = haqiqiy vaqt.
        """
        now = now_ts if now_ts is not None else time.time()
        bar_seconds = timeframe_to_seconds(self.cfg.timeframe)
        cooldown_seconds = self.cfg.cooldown_bars_after_sl * bar_seconds
        self._cooldown_until_ts = now + cooldown_seconds

    def cooldown_remaining_seconds(self, now_ts: Optional[float] = None) -> float:
        """Diagnostika uchun — cooldown yana qancha?"""
        if self._cooldown_until_ts <= 0:
            return 0.0
        now = now_ts if now_ts is not None else time.time()
        return max(0.0, self._cooldown_until_ts - now)

    # ─── FEE-AWARE EXIT ──────────────────────────────────────────────────────

    def estimated_round_trip_fee(self, position: Position, exit_price: float) -> float:
        """Pozitsiya uchun open+close fee taxminiy hisobi (Taker rate)."""
        rate = self.config.risk.TAKER_FEE_RATE
        entry_fee = position.entry_price * position.size * rate
        exit_fee = exit_price * position.size * rate
        return entry_fee + exit_fee

    def voluntary_exit_allowed(self, position: Position, exit_price: float) -> bool:
        """
        Voluntary exit (trailing/opposite/exhaustion) faqat gross profit
        fee'larni qoplagandagina ruxsat etiladi.

        SL va MAX_AGE bunga taalluqli emas — ularni majburan yopish kerak.

        factor=1.0 (default): gross >= round-trip fee (net >= 0)
        factor=2.0: gross >= 2× fee (net >= fee — yaxshi buffer)
        """
        factor = self.cfg.min_net_profit_fee_factor
        if factor <= 0:
            return True  # Disabled — har qanday voluntary exit ruxsat
        gross = position.pnl_at(exit_price)
        fees = self.estimated_round_trip_fee(position, exit_price)
        return gross >= factor * fees

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

    def register_entry(self, now_ts: Optional[float] = None) -> None:
        """Yangi entry ochilganda — trades-per-hour rolling window'ga qo'shamiz."""
        now = now_ts if now_ts is not None else time.time()
        self._recent_trade_ts.append(now)
        # Ortiqcha eski entry'larni tozalash
        cutoff = now - 3600
        self._recent_trade_ts = [t for t in self._recent_trade_ts if t >= cutoff]

    def on_trade_closed(self, net_pnl: float, now_ts: Optional[float] = None) -> None:
        self.total_trades += 1
        self.realized_pnl += net_pnl
        if net_pnl > 0:
            self.winning_trades += 1
            self._consecutive_losses = 0  # streak reset
        elif net_pnl < 0:
            self.losing_trades += 1
            self._consecutive_losses += 1
            # Consecutive loss streak — qo'shimcha cooldown
            if (
                self.cfg.consecutive_losses_threshold > 0
                and self._consecutive_losses >= self.cfg.consecutive_losses_threshold
            ):
                now = now_ts if now_ts is not None else time.time()
                bar_seconds = timeframe_to_seconds(self.cfg.timeframe)
                cooldown_seconds = (
                    self.cfg.consecutive_losses_cooldown_bars * bar_seconds
                )
                self._streak_cooldown_until_ts = now + cooldown_seconds
                logger.warning(
                    f"[CONSECUTIVE_LOSSES] {self._consecutive_losses} ta zarara "
                    f"trade qatoriy → {self.cfg.consecutive_losses_cooldown_bars} bar "
                    f"({cooldown_seconds}s) cooldown qo'yildi"
                )
                # Streakni reset qilamiz (cooldown'dan keyin yangi sanovdan boshlanadi)
                self._consecutive_losses = 0
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
            "chop": self.chop.value,
            "chop_is_trending": self.chop.is_trending(self.cfg.chop_max_for_entry),
            "supertrend_value": self.supertrend.value,
            "supertrend_direction": self.supertrend.direction,
            "htf_ema_fast": self.htf_ema_fast.value,
            "htf_ema_slow": self.htf_ema_slow.value,
            "position": self.position.to_dict() if self.position else None,
            "last_cross": self._last_cross,
            "signal_age_bars": self._signal_age_bars(),
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds(),
            "trades_last_hour": len(self._recent_trade_ts),
            "consecutive_losses": self._consecutive_losses,
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
        self._cooldown_until_ts = 0.0
        self._streak_cooldown_until_ts = 0.0
        self._consecutive_losses = 0
        self._last_cross = None
        self._last_cross_ts = 0
        self._recent_trade_ts = []


# Backward compatibility
TrendFollowingStrategy = TrendStrategy
