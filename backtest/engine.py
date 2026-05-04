"""
Trend Following Backtest Engine

Strategy: Smart Trend Following v2.0
- EMA(fast) crossover EMA(slow) + Supertrend direction aligned + ADX > threshold
- Higher timeframe (4H) trend filter (optional)
- Exit: ATR-based trailing stop + partial TP + opposite signal
- Position sizing: capital × leverage / (price × N pyramids)
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .data_loader import Candle

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                             DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


class Signal(Enum):
    NONE = "none"
    LONG = "long"
    SHORT = "short"


class TrailingPhase(Enum):
    NONE = "none"         # Fixed SL
    BREAKEVEN = "breakeven"  # SL at entry
    TRAILING = "trailing"    # ATR-based trailing


@dataclass
class Position:
    side: str            # "long" or "short"
    entry_price: float
    size: float
    opened_at: int
    entry_fee: float = 0.0
    # Trailing stop state
    trailing_phase: TrailingPhase = TrailingPhase.NONE
    peak_price: float = 0.0      # best price seen after entry
    stop_price: float = 0.0      # current stop level
    # Partial TP tracking
    partial_tp1_done: bool = False  # 2% TP taken
    partial_tp2_done: bool = False  # 5% TP taken
    initial_size: float = 0.0

    def __post_init__(self):
        if self.initial_size == 0.0:
            self.initial_size = self.size

    @property
    def notional(self) -> float:
        return self.entry_price * self.size

    def unrealized_pnl(self, price: float) -> float:
        if self.side == "long":
            return (price - self.entry_price) * self.size
        return (self.entry_price - price) * self.size

    def pnl_percent(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "long":
            return (price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - price) / self.entry_price * 100


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    size: float
    opened_at: int
    closed_at: int
    gross_pnl: float
    fees: float
    net_pnl: float
    reason: str


@dataclass
class BacktestConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    htf_timeframe: str = "4H"

    leverage: int = 10
    # Trade amount semantikasi (v2.1): MARGIN to allocate per trade.
    # Notional = trade_amount × leverage.
    trade_amount: float = 1000.0
    # Fallback agar `trade_amount=0`: balansning shu qismidan margin olinadi.
    # robot.py:_check_new_entry da CAPITAL_ENGAGEMENT bilan mos.
    capital_engagement: float = 0.15

    taker_fee: float = 0.0006
    funding_rate_per_8h: float = 0.0001

    # Strategy params
    ema_fast: int = 9
    ema_slow: int = 21
    max_signal_age_bars: int = 3
    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 20.0
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # Choppiness Index filter (v2.1) — choppy market'larda entry rad etiladi
    use_choppiness_filter: bool = True
    chop_period: int = 14
    chop_max_for_entry: float = 50.0

    # Consecutive losses cooldown (loop guard / regime change detector)
    consecutive_losses_threshold: int = 3
    consecutive_losses_cooldown_bars: int = 20

    # HTF filter (4H candles)
    use_htf_filter: bool = True
    htf_ema_fast: int = 21
    htf_ema_slow: int = 50

    # Exit — fixed % + ATR-adaptive
    initial_sl_percent: float = 3.0
    use_atr_sl: bool = True
    sl_atr_multiplier: float = 2.0
    trailing_activation_percent: float = 1.0  # move to breakeven at +1%
    trailing_atr_multiplier: float = 1.5       # trailing distance = 1.5 × ATR

    # Partial TP
    use_partial_tp: bool = True
    partial_tp1_percent: float = 2.0
    partial_tp1_size_pct: float = 0.33  # close 33%
    partial_tp2_percent: float = 5.0
    partial_tp2_size_pct: float = 0.33  # close 33%

    # Opposite signal exit
    use_opposite_signal_exit: bool = True
    opposite_signal_requires_full_confirm: bool = True

    # Fee-aware voluntary exit (1.0 = exit must AT LEAST cover fees)
    min_net_profit_fee_factor: float = 1.0

    # Risk
    max_drawdown_percent: float = 20.0
    cooldown_bars_after_sl: int = 5

    # Trades-per-hour limit (loop-bug guard)
    max_trades_per_hour: int = 4

    initial_balance: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#                             INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


def calc_ema(candles: List[Candle], period: int) -> List[float]:
    n = len(candles)
    if n < period:
        return [0.0] * n
    alpha = 2 / (period + 1)
    ema = [0.0] * n
    # Initial SMA
    sma = sum(c.close for c in candles[:period]) / period
    ema[period - 1] = sma
    for i in range(period, n):
        ema[i] = candles[i].close * alpha + ema[i - 1] * (1 - alpha)
    return ema


def calc_atr(candles: List[Candle], period: int = 14) -> List[float]:
    n = len(candles)
    if n < period + 1:
        return [0.0] * n
    trs = []
    atrs = [0.0]
    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        trs.append(tr)
        if i < period:
            atrs.append(0.0)
        elif i == period:
            atrs.append(sum(trs[:period]) / period)
        else:
            atrs.append((atrs[-1] * (period - 1) + tr) / period)
    return atrs


def calc_adx(candles: List[Candle], period: int = 14) -> List[float]:
    n = len(candles)
    if n < period * 2:
        return [0.0] * n
    plus_dm = [0.0]
    minus_dm = [0.0]
    trs = [0.0]
    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        up = c.high - p.high
        down = p.low - c.low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        trs.append(tr)

    def ws(v, pp):
        out = [0.0] * len(v)
        if len(v) <= pp:
            return out
        out[pp] = sum(v[1:pp + 1])
        for i in range(pp + 1, len(v)):
            out[i] = out[i - 1] - (out[i - 1] / pp) + v[i]
        return out

    sm_plus, sm_minus, sm_tr = ws(plus_dm, period), ws(minus_dm, period), ws(trs, period)
    plus_di = [100 * sm_plus[i] / sm_tr[i] if sm_tr[i] > 0 else 0.0 for i in range(n)]
    minus_di = [100 * sm_minus[i] / sm_tr[i] if sm_tr[i] > 0 else 0.0 for i in range(n)]
    dx = []
    for i in range(n):
        s = plus_di[i] + minus_di[i]
        dx.append(100 * abs(plus_di[i] - minus_di[i]) / s if s > 0 else 0.0)
    adx = [0.0] * n
    start = period * 2
    if start < n:
        adx[start] = sum(dx[period + 1:start + 1]) / period
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def calc_chop(candles: List[Candle], period: int = 14) -> List[float]:
    """
    Choppiness Index — bar bo'yicha. Returns list of float same length as candles.
    Past CHOP (~< 38) = trending, yuqori CHOP (~> 62) = sideways.
    """
    n = len(candles)
    if n < period + 1:
        return [50.0] * n
    log_period = math.log10(period)
    out = [50.0] * n
    for i in range(period, n):
        window = candles[i - period + 1:i + 1]  # period bars (inclusive of i)
        sum_tr = 0.0
        for j in range(len(window)):
            c = window[j]
            pc = candles[i - period + j].close if (i - period + j) >= 0 else c.close
            tr = max(c.high - c.low, abs(c.high - pc), abs(c.low - pc))
            sum_tr += tr
        max_high = max(c.high for c in window)
        min_low = min(c.low for c in window)
        rng = max_high - min_low
        if rng <= 0 or sum_tr <= 0:
            out[i] = out[i - 1] if i > 0 else 50.0
            continue
        try:
            out[i] = 100.0 * math.log10(sum_tr / rng) / log_period
        except (ValueError, ZeroDivisionError):
            out[i] = 50.0
    return out


def calc_supertrend(candles: List[Candle], period: int = 10,
                    multiplier: float = 3.0) -> List[tuple]:
    """
    Supertrend — ATR-based trend indicator.
    Returns list of (supertrend_value, direction): direction 1=uptrend, -1=downtrend
    """
    n = len(candles)
    if n < period + 1:
        return [(0.0, 0)] * n

    atrs = calc_atr(candles, period)
    supertrend = [(0.0, 0)] * n

    for i in range(period, n):
        c = candles[i]
        hl2 = (c.high + c.low) / 2
        atr = atrs[i]
        if atr <= 0:
            continue

        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        if i == period:
            # First bar: initialize
            direction = 1 if c.close > basic_upper else -1
            st_value = basic_lower if direction == 1 else basic_upper
        else:
            prev_st, prev_dir = supertrend[i - 1]
            prev_close = candles[i - 1].close

            # Final upper/lower band logic
            final_upper = basic_upper
            if prev_dir == -1 and basic_upper > prev_st:
                final_upper = prev_st
            final_lower = basic_lower
            if prev_dir == 1 and basic_lower < prev_st:
                final_lower = prev_st

            # Flip logic
            if prev_dir == 1:
                if c.close < final_lower:
                    direction = -1
                    st_value = final_upper
                else:
                    direction = 1
                    st_value = final_lower
            else:  # prev_dir == -1
                if c.close > final_upper:
                    direction = 1
                    st_value = final_lower
                else:
                    direction = -1
                    st_value = final_upper

        supertrend[i] = (st_value, direction)

    return supertrend


# ═══════════════════════════════════════════════════════════════════════════════
#                             STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════


class SmartTrendStrategy:
    """
    Smart Trend Following v2.1

    Entry signal (barcha kerak):
    1. EMA(fast) Golden/Death Cross EMA(slow) — yangi (max_signal_age_bars ichida)
    2. Supertrend direction aligned
    3. ADX > threshold
    4. Optional: HTF trend filter

    Exit:
    - Initial SL: max(fixed%, ATR*sl_atr_multiplier)
    - 3-phase trailing
    - Partial TP (optional)
    - Opposite signal exit (fee-aware)
    """

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self.position: Optional[Position] = None
        # Last detected cross + bar index (stale check)
        self.last_cross: Optional[str] = None  # "golden_cross" | "death_cross" | None
        self.last_cross_bar: int = -1
        self.cooldown_until_bar: int = 0
        self.atr_at_entry: float = 0.0
        # Trades-per-hour rolling window (bar indices)
        self.recent_trade_bars: List[int] = []

    # --- Signal helpers ---
    def _check_full_filters(self, signal: Signal, i: int,
                            supertrend: List[tuple], adx: List[float],
                            htf_ema_fast: Optional[float],
                            htf_ema_slow: Optional[float],
                            chops: Optional[List[float]] = None) -> bool:
        if adx[i] < self.cfg.adx_threshold:
            return False
        # Choppiness filter (v2.1) — choppy market'da entry blok
        if self.cfg.use_choppiness_filter and chops is not None:
            chop_val = chops[i]
            if chop_val > self.cfg.chop_max_for_entry:
                return False
        st_val, st_dir = supertrend[i]
        if signal == Signal.LONG and st_dir != 1:
            return False
        if signal == Signal.SHORT and st_dir != -1:
            return False
        if self.cfg.use_htf_filter and htf_ema_fast is not None and htf_ema_slow is not None:
            if htf_ema_fast > 0 and htf_ema_slow > 0:
                if signal == Signal.LONG and htf_ema_fast < htf_ema_slow:
                    return False
                if signal == Signal.SHORT and htf_ema_fast > htf_ema_slow:
                    return False
        return True

    def _is_rate_limited(self, i: int, bars_per_hour: int) -> bool:
        """Trades-per-hour limit check (bar indices'ga asoslanadi)."""
        if self.cfg.max_trades_per_hour <= 0:
            return False
        cutoff = i - bars_per_hour
        self.recent_trade_bars = [b for b in self.recent_trade_bars if b > cutoff]
        return len(self.recent_trade_bars) >= self.cfg.max_trades_per_hour

    def detect_signal(self, i: int, ema_fast: List[float], ema_slow: List[float],
                      supertrend: List[tuple], adx: List[float],
                      htf_ema_fast: Optional[float] = None,
                      htf_ema_slow: Optional[float] = None,
                      bars_per_hour: int = 4,
                      chops: Optional[List[float]] = None) -> Signal:
        """Signal detection — all layers must agree, fresh cross required."""
        if i < self.cfg.ema_slow + 5:
            return Signal.NONE

        # Cooldown after SL (default + streak)
        if i < self.cooldown_until_bar:
            return Signal.NONE

        # Trades-per-hour limit
        if self._is_rate_limited(i, bars_per_hour):
            return Signal.NONE

        # 1. EMA crossover (exact moment) — track latest cross
        prev_fast, prev_slow = ema_fast[i - 1], ema_slow[i - 1]
        cur_fast, cur_slow = ema_fast[i], ema_slow[i]
        if prev_fast <= prev_slow and cur_fast > cur_slow:
            self.last_cross = "golden_cross"
            self.last_cross_bar = i
        elif prev_fast >= prev_slow and cur_fast < cur_slow:
            self.last_cross = "death_cross"
            self.last_cross_bar = i

        if self.last_cross is None:
            return Signal.NONE

        # Stale check — cross max_signal_age_bars dan eski emas
        age = i - self.last_cross_bar
        if age > self.cfg.max_signal_age_bars:
            self.last_cross = None
            return Signal.NONE

        signal = Signal.LONG if self.last_cross == "golden_cross" else Signal.SHORT

        # 2-4. Full filter (CHOP + ADX + ST + HTF)
        if not self._check_full_filters(signal, i, supertrend, adx,
                                         htf_ema_fast, htf_ema_slow, chops=chops):
            return Signal.NONE

        return signal

    def detect_opposite_exit(self, current_side: str, i: int,
                             supertrend: List[tuple], adx: List[float],
                             htf_ema_fast: Optional[float] = None,
                             htf_ema_slow: Optional[float] = None,
                             chops: Optional[List[float]] = None) -> bool:
        """LONG + death_cross => True (exit), SHORT + golden_cross => True."""
        if not self.cfg.use_opposite_signal_exit:
            return False
        if self.last_cross is None:
            return False
        # Stale check
        if (i - self.last_cross_bar) > self.cfg.max_signal_age_bars:
            return False
        is_opposite = (
            (current_side == "long" and self.last_cross == "death_cross")
            or (current_side == "short" and self.last_cross == "golden_cross")
        )
        if not is_opposite:
            return False
        if self.cfg.opposite_signal_requires_full_confirm:
            opp = Signal.SHORT if current_side == "long" else Signal.LONG
            if not self._check_full_filters(opp, i, supertrend, adx,
                                             htf_ema_fast, htf_ema_slow, chops=chops):
                return False
        return True

    def calculate_position_size(self, price: float, balance: Optional[float] = None) -> float:
        """
        Position size — margin × leverage / price.

        v2.1 semantikasi: trade_amount = MARGIN to allocate (not the full notional).
        notional = trade_amount × leverage; size = notional / price.

        Agar trade_amount=0 va balance berilgan bo'lsa: balance × capital_engagement.
        """
        if price <= 0:
            return 0.0
        margin = self.cfg.trade_amount
        if margin <= 0 and balance is not None:
            margin = balance * max(0.0, min(1.0, self.cfg.capital_engagement))
        if margin <= 0:
            return 0.0
        return (margin * self.cfg.leverage) / price

    def initial_stop_loss(self, side: str, entry: float, atr: float = 0.0) -> float:
        """SL = max(fixed%, ATR*sl_atr_multiplier) (agar use_atr_sl=True)."""
        sl_pct = self.cfg.initial_sl_percent / 100
        if self.cfg.use_atr_sl and atr > 0 and entry > 0:
            atr_pct = (self.cfg.sl_atr_multiplier * atr) / entry
            sl_pct = max(sl_pct, atr_pct)
        if side == "long":
            return entry * (1 - sl_pct)
        return entry * (1 + sl_pct)

    def voluntary_exit_allowed(self, pos: Position, exit_price: float) -> bool:
        """Fee-aware: gross >= factor × round-trip fee."""
        factor = self.cfg.min_net_profit_fee_factor
        if factor <= 0:
            return True
        gross = pos.unrealized_pnl(exit_price)
        entry_fee = pos.entry_price * pos.size * self.cfg.taker_fee
        exit_fee = exit_price * pos.size * self.cfg.taker_fee
        return gross >= factor * (entry_fee + exit_fee)

    def update_trailing_stop(self, pos: Position, price: float, atr: float) -> None:
        """3-phase trailing stop."""
        activation_pct = self.cfg.trailing_activation_percent / 100
        trail_distance = self.cfg.trailing_atr_multiplier * atr

        # Phase 1 → Phase 2: Breakeven
        if pos.trailing_phase == TrailingPhase.NONE:
            profit_pct = pos.pnl_percent(price) / 100
            if profit_pct >= activation_pct:
                pos.trailing_phase = TrailingPhase.BREAKEVEN
                pos.stop_price = pos.entry_price
                pos.peak_price = price

        # Phase 2 → Phase 3: Trailing
        if pos.trailing_phase == TrailingPhase.BREAKEVEN:
            if pos.side == "long":
                if price > pos.peak_price:
                    pos.peak_price = price
                    trail_stop = pos.peak_price - trail_distance
                    if trail_stop > pos.stop_price:
                        pos.stop_price = trail_stop
                        pos.trailing_phase = TrailingPhase.TRAILING
            else:
                if price < pos.peak_price or pos.peak_price == 0:
                    pos.peak_price = price
                    trail_stop = pos.peak_price + trail_distance
                    if trail_stop < pos.stop_price or pos.stop_price == 0:
                        pos.stop_price = trail_stop
                        pos.trailing_phase = TrailingPhase.TRAILING

        # Phase 3: Continue trailing
        if pos.trailing_phase == TrailingPhase.TRAILING:
            if pos.side == "long":
                if price > pos.peak_price:
                    pos.peak_price = price
                    trail_stop = pos.peak_price - trail_distance
                    if trail_stop > pos.stop_price:
                        pos.stop_price = trail_stop
            else:
                if price < pos.peak_price:
                    pos.peak_price = price
                    trail_stop = pos.peak_price + trail_distance
                    if trail_stop < pos.stop_price:
                        pos.stop_price = trail_stop

    def check_stop_hit(self, pos: Position, candle: Candle) -> Optional[float]:
        """SL urilganmi? Return exit price if hit."""
        if pos.stop_price <= 0:
            return None
        if pos.side == "long" and candle.low <= pos.stop_price:
            return pos.stop_price
        if pos.side == "short" and candle.high >= pos.stop_price:
            return pos.stop_price
        return None

    def check_partial_tp(self, pos: Position, candle: Candle) -> Optional[tuple]:
        """Partial TP check — returns (exit_price, close_fraction) if hit."""
        if not self.cfg.use_partial_tp:
            return None

        # Partial TP 1 (2% default)
        if not pos.partial_tp1_done:
            tp1_pct = self.cfg.partial_tp1_percent / 100
            if pos.side == "long":
                tp1_price = pos.entry_price * (1 + tp1_pct)
                if candle.high >= tp1_price:
                    pos.partial_tp1_done = True
                    return (tp1_price, self.cfg.partial_tp1_size_pct)
            else:
                tp1_price = pos.entry_price * (1 - tp1_pct)
                if candle.low <= tp1_price:
                    pos.partial_tp1_done = True
                    return (tp1_price, self.cfg.partial_tp1_size_pct)

        # Partial TP 2 (5% default)
        if pos.partial_tp1_done and not pos.partial_tp2_done:
            tp2_pct = self.cfg.partial_tp2_percent / 100
            if pos.side == "long":
                tp2_price = pos.entry_price * (1 + tp2_pct)
                if candle.high >= tp2_price:
                    pos.partial_tp2_done = True
                    # 33% of INITIAL size (proportional)
                    return (tp2_price, self.cfg.partial_tp2_size_pct * pos.initial_size / pos.size)
            else:
                tp2_price = pos.entry_price * (1 - tp2_pct)
                if candle.low <= tp2_price:
                    pos.partial_tp2_done = True
                    return (tp2_price, self.cfg.partial_tp2_size_pct * pos.initial_size / pos.size)

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#                             BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BacktestResult:
    config: BacktestConfig
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    max_drawdown_percent: float
    total_fees: float
    total_funding: float
    trades: List[Trade] = field(default_factory=list)
    balance_history: List[tuple] = field(default_factory=list)

    @property
    def winrate(self) -> float:
        c = self.winning_trades + self.losing_trades
        return (self.winning_trades / c * 100) if c > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        w = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        l = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        return w / l if l > 0 else float('inf')

    @property
    def return_percent(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100


class Backtester:
    def __init__(self, config: BacktestConfig, candles: List[Candle],
                 htf_candles: Optional[List[Candle]] = None):
        self.cfg = config
        self.candles = candles
        self.htf_candles = htf_candles or []
        self.strategy = SmartTrendStrategy(config)

        self.balance = config.initial_balance if config.initial_balance > 0 else config.trade_amount
        self.initial_balance = self.balance
        self.peak_balance = self.balance

        self.trades: List[Trade] = []
        self.balance_history: List[tuple] = []
        self.total_fees = 0.0
        self.total_funding = 0.0

    def run(self) -> BacktestResult:
        ema_fast = calc_ema(self.candles, self.cfg.ema_fast)
        ema_slow = calc_ema(self.candles, self.cfg.ema_slow)
        atrs = calc_atr(self.candles, self.cfg.atr_period)
        adxs = calc_adx(self.candles, self.cfg.adx_period)
        supertrend = calc_supertrend(self.candles, self.cfg.supertrend_period,
                                     self.cfg.supertrend_multiplier)
        # Choppiness Index — choppy market filtri uchun (v2.1)
        chops = calc_chop(self.candles, self.cfg.chop_period)
        # Consecutive losses tracker (engine local — doesn't need state across runs)
        consecutive_losses = 0

        # HTF EMAs (mapped to each candle by timestamp)
        htf_ema_fast_map = {}
        htf_ema_slow_map = {}
        if self.htf_candles and self.cfg.use_htf_filter:
            htf_ef = calc_ema(self.htf_candles, self.cfg.htf_ema_fast)
            htf_es = calc_ema(self.htf_candles, self.cfg.htf_ema_slow)
            for i, c in enumerate(self.htf_candles):
                htf_ema_fast_map[c.timestamp] = htf_ef[i]
                htf_ema_slow_map[c.timestamp] = htf_es[i]

        def get_htf_emas(ts: int) -> tuple:
            if not htf_ema_fast_map:
                return (None, None)
            # Find latest HTF candle before this timestamp
            htf_ts = [t for t in htf_ema_fast_map if t <= ts]
            if not htf_ts:
                return (None, None)
            latest = max(htf_ts)
            return (htf_ema_fast_map[latest], htf_ema_slow_map[latest])

        warmup = max(self.cfg.ema_slow, self.cfg.adx_period * 2, self.cfg.supertrend_period) + 5

        # Bars per hour (timeframe asosida) — trades-per-hour limit uchun
        from trend_robot.strategy import timeframe_to_seconds
        tf_seconds = timeframe_to_seconds(self.cfg.timeframe)
        bars_per_hour = max(1, 3600 // max(tf_seconds, 1))

        for i in range(warmup, len(self.candles)):
            c = self.candles[i]
            atr = atrs[i]

            # 1. Manage existing position
            if self.strategy.position is not None:
                pos = self.strategy.position

                # Partial TP (intrabar)
                ptp_result = self.strategy.check_partial_tp(pos, c)
                if ptp_result:
                    exit_price, close_frac = ptp_result
                    close_size = pos.size * close_frac if close_frac <= 1.0 else close_frac
                    close_size = min(close_size, pos.size)
                    if close_size > 0:
                        self._close_partial(pos, exit_price, close_size, c.timestamp, "PARTIAL_TP")

                # If position fully closed, skip
                if self.strategy.position is None:
                    continue

                pos = self.strategy.position

                # Update trailing stop
                self.strategy.update_trailing_stop(pos, c.close, atr)

                # Check SL (intrabar using high/low) — MAJBURIY exit
                stop_hit = self.strategy.check_stop_hit(pos, c)
                if stop_hit is not None:
                    is_real_sl = pos.trailing_phase == TrailingPhase.NONE
                    self._close_position(pos, stop_hit, c.timestamp,
                                         "SL" if is_real_sl else "TRAIL_STOP")
                    self.strategy.cooldown_until_bar = i + self.cfg.cooldown_bars_after_sl
                    # Consecutive losses tracking (faqat haqiqiy SL — TRAIL_STOP odatda profitable)
                    if is_real_sl:
                        consecutive_losses += 1
                        if (
                            self.cfg.consecutive_losses_threshold > 0
                            and consecutive_losses >= self.cfg.consecutive_losses_threshold
                        ):
                            extra = self.cfg.consecutive_losses_cooldown_bars
                            self.strategy.cooldown_until_bar = max(
                                self.strategy.cooldown_until_bar, i + extra
                            )
                            consecutive_losses = 0  # reset, qaytadan sanovni boshlaymiz
                    else:
                        consecutive_losses = 0
                    # Continue loop — pozitsiya yopildi
                    unrealized = 0.0
                    equity = self.balance
                    self.peak_balance = max(self.peak_balance, equity)
                    self.balance_history.append((c.timestamp, equity))
                    continue

                # Opposite signal exit (yangi v2.1) — fee-aware
                # First update last_cross by re-running detect_signal logic on this bar
                # (without consuming): we update_indicators-style track of cross
                prev_fast, prev_slow = ema_fast[i - 1], ema_slow[i - 1]
                cur_fast, cur_slow = ema_fast[i], ema_slow[i]
                if prev_fast <= prev_slow and cur_fast > cur_slow:
                    self.strategy.last_cross = "golden_cross"
                    self.strategy.last_cross_bar = i
                elif prev_fast >= prev_slow and cur_fast < cur_slow:
                    self.strategy.last_cross = "death_cross"
                    self.strategy.last_cross_bar = i

                htf_ef, htf_es = get_htf_emas(c.timestamp)
                if self.strategy.detect_opposite_exit(
                    pos.side, i, supertrend, adxs, htf_ef, htf_es, chops=chops
                ) and self.strategy.voluntary_exit_allowed(pos, c.close):
                    self._close_position(pos, c.close, c.timestamp, "OPPOSITE_SIGNAL")
                    self.strategy.cooldown_until_bar = i + self.cfg.cooldown_bars_after_sl

            # 2. Entry check (no position)
            if self.strategy.position is None:
                htf_ef, htf_es = get_htf_emas(c.timestamp)
                signal = self.strategy.detect_signal(
                    i, ema_fast, ema_slow, supertrend, adxs,
                    htf_ef, htf_es, bars_per_hour=bars_per_hour, chops=chops,
                )
                if signal != Signal.NONE:
                    side = "long" if signal == Signal.LONG else "short"
                    size = self.strategy.calculate_position_size(c.close, balance=self.balance)
                    if size > 0:
                        entry_fee = c.close * size * self.cfg.taker_fee
                        self.total_fees += entry_fee
                        self.strategy.position = Position(
                            side=side, entry_price=c.close, size=size,
                            opened_at=c.timestamp, entry_fee=entry_fee,
                            initial_size=size,
                        )
                        # ATR-based SL ham hisobga olamiz
                        self.strategy.position.stop_price = self.strategy.initial_stop_loss(
                            side, c.close, atr=atr
                        )
                        self.strategy.atr_at_entry = atr
                        # Trades-per-hour rolling window'ga qo'shamiz
                        self.strategy.recent_trade_bars.append(i)
                        # Signal consume — bir cross'da bir entry
                        self.strategy.last_cross = None

            # Track balance history
            unrealized = self.strategy.position.unrealized_pnl(c.close) if self.strategy.position else 0.0
            equity = self.balance + unrealized
            self.peak_balance = max(self.peak_balance, equity)
            self.balance_history.append((c.timestamp, equity))

            # Circuit breaker: max drawdown
            dd_pct = (self.initial_balance - equity) / self.initial_balance * 100
            if dd_pct >= self.cfg.max_drawdown_percent and self.strategy.position:
                self._close_position(self.strategy.position, c.close, c.timestamp, "CIRCUIT")

        # Close remaining at end
        if self.strategy.position:
            last = self.candles[-1]
            self._close_position(self.strategy.position, last.close, last.timestamp, "END")

        return self._make_result()

    def _close_position(self, pos: Position, exit_price: float, ts: int, reason: str):
        gross = pos.unrealized_pnl(exit_price)
        exit_fee = exit_price * pos.size * self.cfg.taker_fee
        self.total_fees += exit_fee
        hours = (ts - pos.opened_at) / 1000 / 3600
        funding = pos.notional * self.cfg.funding_rate_per_8h * (hours / 8)
        self.total_funding += funding
        net = gross - pos.entry_fee - exit_fee - funding
        self.trades.append(Trade(
            side=pos.side, entry_price=pos.entry_price, exit_price=exit_price,
            size=pos.size, opened_at=pos.opened_at, closed_at=ts,
            gross_pnl=gross, fees=pos.entry_fee + exit_fee,
            net_pnl=net, reason=reason,
        ))
        self.balance += net
        self.strategy.position = None

    def _close_partial(self, pos: Position, exit_price: float, close_size: float,
                      ts: int, reason: str):
        # Proportional close
        partial_entry_fee = pos.entry_fee * (close_size / pos.initial_size)
        gross = (exit_price - pos.entry_price) * close_size if pos.side == "long" \
            else (pos.entry_price - exit_price) * close_size
        exit_fee = exit_price * close_size * self.cfg.taker_fee
        self.total_fees += exit_fee
        net = gross - partial_entry_fee - exit_fee
        self.trades.append(Trade(
            side=pos.side, entry_price=pos.entry_price, exit_price=exit_price,
            size=close_size, opened_at=pos.opened_at, closed_at=ts,
            gross_pnl=gross, fees=partial_entry_fee + exit_fee,
            net_pnl=net, reason=reason,
        ))
        self.balance += net

        # Reduce position
        pos.size -= close_size
        pos.entry_fee -= partial_entry_fee
        if pos.size <= 0:
            self.strategy.position = None

    def _make_result(self) -> BacktestResult:
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        losses = sum(1 for t in self.trades if t.net_pnl < 0)
        total_pnl = sum(t.net_pnl for t in self.trades)
        min_eq = min((b for _, b in self.balance_history), default=self.peak_balance)
        max_dd = self.peak_balance - min_eq
        max_dd_pct = (max_dd / self.peak_balance * 100) if self.peak_balance > 0 else 0
        return BacktestResult(
            config=self.cfg,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_trades=len(self.trades),
            winning_trades=wins,
            losing_trades=losses,
            total_pnl=total_pnl,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct,
            total_fees=self.total_fees,
            total_funding=self.total_funding,
            trades=self.trades,
            balance_history=self.balance_history,
        )
