"""
Trend Following Robot — Technical Indicators (v2.0)

- EMA: trend anchor, crossover signal
- ATR: volatility, trailing stop, Supertrend
- ADX: trend strength filter
- Supertrend: ATR-based trend direction
- EMACrossover: Golden/Death cross detector
"""

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_bitget(cls, data: List) -> "Candle":
        return cls(
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]) if len(data) > 5 else 0.0,
        )


def money_round(value: float, places: int = 6) -> float:
    q = Decimal(10) ** -places
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def calculate_pnl(entry: float, exit_price: float, qty: float, side: str) -> float:
    if side.lower() in ("long", "buy"):
        return (exit_price - entry) * qty
    return (entry - exit_price) * qty


# ═══════════════════════════════════════════════════════════════════════════════
#                             EMA
# ═══════════════════════════════════════════════════════════════════════════════


class EMAIndicator:
    def __init__(self, period: int = 20):
        self.period = period
        self._ema: float = 0.0
        self._values: List[float] = []
        self._initialized = False
        self._alpha = 2 / (period + 1)

    @property
    def value(self) -> float:
        return self._ema

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, price: float) -> float:
        self._values.append(price)
        if len(self._values) < self.period:
            return 0.0
        if not self._initialized:
            self._ema = sum(self._values[-self.period:]) / self.period
            self._initialized = True
        else:
            self._ema = price * self._alpha + self._ema * (1 - self._alpha)
        return self._ema

    def calculate_from_candles(self, candles: List[Candle]) -> float:
        self._values.clear()
        self._ema = 0.0
        self._initialized = False
        for c in candles:
            self.update(c.close)
        return self._ema


# ═══════════════════════════════════════════════════════════════════════════════
#                             ATR
# ═══════════════════════════════════════════════════════════════════════════════


class ATRIndicator:
    def __init__(self, period: int = 14):
        self.period = period
        self._trs: List[float] = []
        self._atr: float = 0.0
        self._initialized = False
        self._prev_close: Optional[float] = None

    @property
    def value(self) -> float:
        return self._atr

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, candle: Candle) -> float:
        if self._prev_close is None:
            self._prev_close = candle.close
            return 0.0
        tr = max(
            candle.high - candle.low,
            abs(candle.high - self._prev_close),
            abs(candle.low - self._prev_close),
        )
        self._trs.append(tr)
        if len(self._trs) < self.period:
            self._atr = sum(self._trs) / len(self._trs)
        elif len(self._trs) == self.period:
            self._atr = sum(self._trs) / self.period
            self._initialized = True
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period
        self._prev_close = candle.close
        return self._atr

    def calculate_from_candles(self, candles: List[Candle]) -> float:
        self._trs.clear()
        self._atr = 0.0
        self._initialized = False
        self._prev_close = None
        for c in candles:
            self.update(c)
        return self._atr

    def percent_of_price(self, price: float) -> float:
        if price <= 0:
            return 0.0
        return (self._atr / price) * 100


# ═══════════════════════════════════════════════════════════════════════════════
#                             ADX
# ═══════════════════════════════════════════════════════════════════════════════


class ADXIndicator:
    def __init__(self, period: int = 14):
        self.period = period
        self._plus_dm: List[float] = []
        self._minus_dm: List[float] = []
        self._trs: List[float] = []
        self._sm_plus: float = 0.0
        self._sm_minus: float = 0.0
        self._sm_tr: float = 0.0
        self._dx_history: List[float] = []
        self._adx: float = 0.0
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._initialized = False

    @property
    def value(self) -> float:
        return self._adx

    @property
    def initialized(self) -> bool:
        return self._initialized

    def is_trending(self, threshold: float = 25.0) -> bool:
        return self._adx >= threshold

    def update(self, candle: Candle) -> float:
        if self._prev_close is None:
            self._prev_high = candle.high
            self._prev_low = candle.low
            self._prev_close = candle.close
            return 0.0

        up = candle.high - self._prev_high
        down = self._prev_low - candle.low
        plus_dm = up if up > down and up > 0 else 0.0
        minus_dm = down if down > up and down > 0 else 0.0
        tr = max(
            candle.high - candle.low,
            abs(candle.high - self._prev_close),
            abs(candle.low - self._prev_close),
        )
        self._plus_dm.append(plus_dm)
        self._minus_dm.append(minus_dm)
        self._trs.append(tr)

        p = self.period
        if len(self._trs) == p:
            self._sm_plus = sum(self._plus_dm[:p])
            self._sm_minus = sum(self._minus_dm[:p])
            self._sm_tr = sum(self._trs[:p])
        elif len(self._trs) > p:
            self._sm_plus = self._sm_plus - (self._sm_plus / p) + plus_dm
            self._sm_minus = self._sm_minus - (self._sm_minus / p) + minus_dm
            self._sm_tr = self._sm_tr - (self._sm_tr / p) + tr

        if len(self._trs) >= p and self._sm_tr > 0:
            plus_di = 100 * self._sm_plus / self._sm_tr
            minus_di = 100 * self._sm_minus / self._sm_tr
            di_sum = plus_di + minus_di
            dx = (100 * abs(plus_di - minus_di) / di_sum) if di_sum > 0 else 0.0
            self._dx_history.append(dx)
            if len(self._dx_history) == p:
                self._adx = sum(self._dx_history) / p
                self._initialized = True
            elif len(self._dx_history) > p:
                self._adx = (self._adx * (p - 1) + dx) / p

        self._prev_high = candle.high
        self._prev_low = candle.low
        self._prev_close = candle.close
        return self._adx

    def calculate_from_candles(self, candles: List[Candle]) -> float:
        self.__init__(self.period)
        for c in candles:
            self.update(c)
        return self._adx


# ═══════════════════════════════════════════════════════════════════════════════
#                             SUPERTREND
# ═══════════════════════════════════════════════════════════════════════════════


class SupertrendIndicator:
    """
    Supertrend — ATR-based trend direction indicator.
    direction: 1 = uptrend, -1 = downtrend
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self._atr = ATRIndicator(period=period)
        self._values: List[Tuple[float, int]] = []
        self._prev_candle: Optional[Candle] = None

    @property
    def value(self) -> float:
        return self._values[-1][0] if self._values else 0.0

    @property
    def direction(self) -> int:
        return self._values[-1][1] if self._values else 0

    @property
    def initialized(self) -> bool:
        return len(self._values) > 0 and self._atr.initialized

    def update(self, candle: Candle) -> Tuple[float, int]:
        self._atr.update(candle)
        atr = self._atr.value
        if atr <= 0:
            self._prev_candle = candle
            return (0.0, 0)

        hl2 = (candle.high + candle.low) / 2
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        if not self._values:
            direction = 1 if candle.close > basic_upper else -1
            st_value = basic_lower if direction == 1 else basic_upper
            self._values.append((st_value, direction))
            self._prev_candle = candle
            return (st_value, direction)

        prev_st, prev_dir = self._values[-1]
        final_upper = basic_upper
        if prev_dir == -1 and basic_upper > prev_st:
            final_upper = prev_st
        final_lower = basic_lower
        if prev_dir == 1 and basic_lower < prev_st:
            final_lower = prev_st

        if prev_dir == 1:
            if candle.close < final_lower:
                direction = -1
                st_value = final_upper
            else:
                direction = 1
                st_value = final_lower
        else:
            if candle.close > final_upper:
                direction = 1
                st_value = final_lower
            else:
                direction = -1
                st_value = final_upper

        self._values.append((st_value, direction))
        if len(self._values) > 200:
            self._values = self._values[-200:]
        self._prev_candle = candle
        return (st_value, direction)

    def calculate_from_candles(self, candles: List[Candle]) -> Tuple[float, int]:
        self.__init__(self.period, self.multiplier)
        for c in candles:
            self.update(c)
        return self._values[-1] if self._values else (0.0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#                             EMA CROSSOVER
# ═══════════════════════════════════════════════════════════════════════════════


class EMACrossover:
    """EMA crossover detector — Golden/Death cross moments."""

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        self.fast = EMAIndicator(period=fast_period)
        self.slow = EMAIndicator(period=slow_period)
        self._prev_fast = 0.0
        self._prev_slow = 0.0

    @property
    def initialized(self) -> bool:
        return self.fast.initialized and self.slow.initialized

    @property
    def fast_value(self) -> float:
        return self.fast.value

    @property
    def slow_value(self) -> float:
        return self.slow.value

    def update(self, candle: Candle) -> Optional[str]:
        """Returns 'golden_cross', 'death_cross' or None."""
        old_fast, old_slow = self._prev_fast, self._prev_slow
        new_fast = self.fast.update(candle.close)
        new_slow = self.slow.update(candle.close)
        self._prev_fast, self._prev_slow = new_fast, new_slow

        if not self.initialized:
            return None
        if old_fast == 0 or old_slow == 0:
            return None

        if old_fast <= old_slow and new_fast > new_slow:
            return "golden_cross"
        if old_fast >= old_slow and new_fast < new_slow:
            return "death_cross"
        return None

    def calculate_from_candles(self, candles: List[Candle]) -> None:
        self.fast = EMAIndicator(period=self.fast.period)
        self.slow = EMAIndicator(period=self.slow.period)
        self._prev_fast = 0.0
        self._prev_slow = 0.0
        for c in candles[:-1]:
            self.fast.update(c.close)
            self.slow.update(c.close)
        self._prev_fast = self.fast.value
        self._prev_slow = self.slow.value
        if candles:
            self.fast.update(candles[-1].close)
            self.slow.update(candles[-1].close)
