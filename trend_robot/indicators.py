"""
Trend Robot - Texnik Indikatorlar

EMA Crossover, Ichimoku Cloud, ADX — trend-following indikatorlar.

Manbalar:
    EMA — bitget-futures-ema dan (vanilla + streaming)
    Ichimoku — bitget-futures-ema dan (5 komponent)
    ADX — HEMA Hedge bot dan (Wilder's smoothing)
    ATR — HEMA RSI/Hedge bot dan
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV shamcha"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#                           EMA INDIKATORI
# ═══════════════════════════════════════════════════════════════════════════════

class EMAIndicator:
    """
    Eksponensial o'rtacha — bitget-futures-ema dan.

    Streaming mode (har bir narxda yangilanadi):
        multiplier = 2 / (period + 1)
        EMA(t) = (price - EMA(t-1)) * multiplier + EMA(t-1)

    Crossover detection uchun oldingi va joriy qiymatlar saqlanadi.
    """

    def __init__(self, period: int):
        self.period = period
        self._multiplier = 2.0 / (period + 1.0)
        self._value: float = 0.0
        self._prev_value: float = 0.0
        self._initialized: bool = False
        self._count: int = 0
        self._initial_sum: float = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def prev_value(self) -> float:
        return self._prev_value

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, price: float) -> float:
        """
        EMA yangilash.

        bitget-futures-ema dan:
            EMA = ((Price - EMA_prev) × multiplier) + EMA_prev

        Birinchi `period` ta narxdan SMA hisoblab, keyin EMA ga o'tadi.
        """
        self._count += 1
        self._prev_value = self._value

        if not self._initialized:
            self._initial_sum += price
            if self._count >= self.period:
                # Birinchi EMA = SMA
                self._value = self._initial_sum / self.period
                self._initialized = True
        else:
            self._value = (price - self._value) * self._multiplier + self._value

        return self._value

    def get_state(self) -> dict:
        return {"value": self._value, "prev": self._prev_value, "initialized": self._initialized}

    def set_state(self, state: dict):
        self._value = state.get("value", 0.0)
        self._prev_value = state.get("prev", 0.0)
        self._initialized = state.get("initialized", False)


# ═══════════════════════════════════════════════════════════════════════════════
#                           EMA CROSSOVER
# ═══════════════════════════════════════════════════════════════════════════════

class EMACrossover:
    """
    EMA Crossover — bitget-futures-ema dan 1:1.

    Golden Cross: fast EMA > slow EMA (oldingi tick da fast <= slow edi)
    Death Cross: slow EMA > fast EMA (oldingi tick da slow <= fast edi)

    bitget-futures-ema kodi:
        if(emaShort > emaLong and last_emaShort and not buy):
            if(last_emaShort <= last_emaLong):
                # BUY
        if(emaLong > emaShort and last_emaShort and not sell):
            if(last_emaLong <= last_emaShort):
                # SELL
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        self.fast = EMAIndicator(fast_period)
        self.slow = EMAIndicator(slow_period)
        self._last_fast: float = 0.0
        self._last_slow: float = 0.0

    @property
    def initialized(self) -> bool:
        return self.fast.initialized and self.slow.initialized

    def update(self, price: float) -> Optional[str]:
        """
        EMA lar yangilash va crossover tekshirish.

        Returns:
            "golden_cross" — LONG signal (fast > slow, oldin fast <= slow)
            "death_cross" — SHORT signal (slow > fast, oldin slow <= fast)
            None — crossover yo'q
        """
        # Oldingi qiymatlarni saqlash
        prev_fast = self.fast.value
        prev_slow = self.slow.value

        # Yangilash
        fast_val = self.fast.update(price)
        slow_val = self.slow.update(price)

        if not self.initialized:
            self._last_fast = fast_val
            self._last_slow = slow_val
            return None

        signal = None

        # Golden Cross: fast EMA tepaga kesib o'tdi
        if fast_val > slow_val and self._last_fast <= self._last_slow:
            signal = "golden_cross"

        # Death Cross: fast EMA pastga kesib o'tdi
        elif slow_val > fast_val and self._last_slow <= self._last_fast:
            signal = "death_cross"

        self._last_fast = fast_val
        self._last_slow = slow_val

        return signal

    def get_state(self) -> dict:
        return {
            "fast": round(self.fast.value, 2),
            "slow": round(self.slow.value, 2),
            "spread": round(self.fast.value - self.slow.value, 2),
            "spread_pct": round(
                (self.fast.value - self.slow.value) / self.slow.value * 100, 4
            ) if self.slow.value > 0 else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           ICHIMOKU CLOUD
# ═══════════════════════════════════════════════════════════════════════════════

class IchimokuCloud:
    """
    Ichimoku Cloud — bitget-futures-ema dan 1:1.

    5 komponent:
        1. Tenkan-sen (Conversion): (N-period high + low) / 2, N=9
        2. Kijun-sen (Base): (N-period high + low) / 2, N=26
        3. Chikou Span (Lagging): close, 26 period o'tmishga
        4. Senkou Span A: (Tenkan + Kijun) / 2, 26 period kelajakka
        5. Senkou Span B: (52-period high + low) / 2, 26 period kelajakka

    LONG signal (bitget-futures-ema dan):
        1. Tenkan > Kijun (crossover)
        2. Price > Cloud (Span A va B)
        3. Chikou > price[26 ago]

    SHORT signal:
        1. Kijun > Tenkan (crossover)
        2. Price < Cloud (Span A va B)
        3. Chikou < price[26 ago]
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

        # Shamcha tarixi
        self._candles: List[Candle] = []

        # Hisoblangan qiymatlar
        self._tenkan: float = 0.0
        self._kijun: float = 0.0
        self._chikou: float = 0.0
        self._senkou_a: float = 0.0
        self._senkou_b: float = 0.0

        # Oldingi qiymatlar (crossover uchun)
        self._prev_tenkan: float = 0.0
        self._prev_kijun: float = 0.0

    @property
    def tenkan(self) -> float:
        return self._tenkan

    @property
    def kijun(self) -> float:
        return self._kijun

    @property
    def chikou(self) -> float:
        return self._chikou

    @property
    def senkou_a(self) -> float:
        return self._senkou_a

    @property
    def senkou_b(self) -> float:
        return self._senkou_b

    @property
    def cloud_top(self) -> float:
        return max(self._senkou_a, self._senkou_b)

    @property
    def cloud_bottom(self) -> float:
        return min(self._senkou_a, self._senkou_b)

    @property
    def initialized(self) -> bool:
        return len(self._candles) >= self.senkou_b_period + self.displacement

    def _period_midpoint(self, period: int) -> float:
        """N-period (high + low) / 2 hisoblash"""
        if len(self._candles) < period:
            return 0.0
        window = self._candles[-period:]
        highest = max(c.high for c in window)
        lowest = min(c.low for c in window)
        return (highest + lowest) / 2

    def update(self, candle: Candle):
        """
        Yangi shamcha bilan Ichimoku yangilash.

        bitget-futures-ema dan:
            tenkan = (9-period high + low) / 2
            kijun = (26-period high + low) / 2
            chikou = close (26 period oldin joylashgan)
            senkou_a = (tenkan + kijun) / 2
            senkou_b = (52-period high + low) / 2
        """
        self._candles.append(candle)

        # Faqat kerakli tarix saqlash
        max_history = self.senkou_b_period + self.displacement + 10
        if len(self._candles) > max_history:
            self._candles = self._candles[-max_history:]

        # Oldingi qiymatlar
        self._prev_tenkan = self._tenkan
        self._prev_kijun = self._kijun

        # Tenkan-sen (Conversion Line)
        self._tenkan = self._period_midpoint(self.tenkan_period)

        # Kijun-sen (Base Line)
        self._kijun = self._period_midpoint(self.kijun_period)

        # Senkou Span A (26 period kelajakka — hozirgi qiymat sifatida)
        self._senkou_a = (self._tenkan + self._kijun) / 2

        # Senkou Span B
        self._senkou_b = self._period_midpoint(self.senkou_b_period)

        # Chikou Span — 26 period oldingi close
        if len(self._candles) > self.displacement:
            self._chikou = self._candles[-(self.displacement + 1)].close
        else:
            self._chikou = candle.close

    def check_signal(self, current_price: float) -> Optional[str]:
        """
        Ichimoku signal tekshirish — bitget-futures-ema dan 1:1.

        LONG: Tenkan > Kijun crossover + price > cloud + chikou > price[26 ago]
        SHORT: Kijun > Tenkan crossover + price < cloud + chikou < price[26 ago]

        Returns:
            "ichimoku_long", "ichimoku_short", yoki None
        """
        if not self.initialized:
            return None

        # Cloud qiymatlari
        cloud_top = self.cloud_top
        cloud_bottom = self.cloud_bottom

        # Price 26 period oldin (chikou comparison uchun)
        price_26_ago = 0.0
        if len(self._candles) > self.displacement:
            price_26_ago = self._candles[-(self.displacement + 1)].close

        # LONG signal
        if (self._tenkan > self._kijun and self._prev_tenkan <= self._prev_kijun):
            # Tenkan > Kijun crossover
            if current_price > cloud_top:
                # Price above cloud
                if price_26_ago > 0 and self._chikou > price_26_ago:
                    # Chikou > price 26 ago
                    return "ichimoku_long"

        # SHORT signal
        if (self._kijun > self._tenkan and self._prev_kijun <= self._prev_tenkan):
            # Kijun > Tenkan crossover
            if current_price < cloud_bottom:
                # Price below cloud
                if price_26_ago > 0 and self._chikou < price_26_ago:
                    return "ichimoku_short"

        return None

    def is_price_above_cloud(self, price: float) -> bool:
        """Narx cloud tepasida-mi"""
        return price > self.cloud_top

    def is_price_below_cloud(self, price: float) -> bool:
        """Narx cloud ostida-mi"""
        return price < self.cloud_bottom

    def get_state(self) -> dict:
        return {
            "tenkan": round(self._tenkan, 2),
            "kijun": round(self._kijun, 2),
            "senkou_a": round(self._senkou_a, 2),
            "senkou_b": round(self._senkou_b, 2),
            "chikou": round(self._chikou, 2),
            "cloud_top": round(self.cloud_top, 2),
            "cloud_bottom": round(self.cloud_bottom, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           ADX INDIKATORI
# ═══════════════════════════════════════════════════════════════════════════════

class ADXIndicator:
    """
    ADX (Average Directional Index) — HEMA Hedge bot dan 1:1.

    Wilder's smoothing:
        +DM / -DM — yo'nalish harakatlari
        +DI = smoothed(+DM) / smoothed(TR) × 100
        -DI = smoothed(-DM) / smoothed(TR) × 100
        DX = |+DI - -DI| / (+DI + -DI) × 100
        ADX = smoothed(DX)

    > 25 = kuchli trend
    > 35 = juda kuchli
    < 20 = flat (signal bloklanadi)
    """

    def __init__(self, period: int = 14):
        self.period = period
        self._prev_candle: Optional[Candle] = None
        self._plus_dm_sum: float = 0.0
        self._minus_dm_sum: float = 0.0
        self._tr_sum: float = 0.0
        self._adx_sum: float = 0.0
        self._value: float = 0.0
        self._plus_di: float = 0.0
        self._minus_di: float = 0.0
        self._count: int = 0
        self._dx_values: List[float] = []
        self._initialized: bool = False

    @property
    def value(self) -> float:
        return self._value

    @property
    def initialized(self) -> bool:
        return self._initialized

    def is_trending(self, threshold: float = 25.0) -> bool:
        """Bozor trending-mi"""
        return self._initialized and self._value >= threshold

    def update(self, candle: Candle) -> float:
        """ADX yangilash — Wilder's smoothing"""
        if self._prev_candle is None:
            self._prev_candle = candle
            return 0.0

        # True Range
        tr = max(
            candle.high - candle.low,
            abs(candle.high - self._prev_candle.close),
            abs(candle.low - self._prev_candle.close)
        )

        # Directional Movement
        up_move = candle.high - self._prev_candle.high
        down_move = self._prev_candle.low - candle.low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        self._prev_candle = candle
        self._count += 1

        if self._count <= self.period:
            self._plus_dm_sum += plus_dm
            self._minus_dm_sum += minus_dm
            self._tr_sum += tr

            if self._count == self.period:
                # Birinchi DI hisoblash
                if self._tr_sum > 0:
                    self._plus_di = (self._plus_dm_sum / self._tr_sum) * 100
                    self._minus_di = (self._minus_dm_sum / self._tr_sum) * 100

                    di_sum = self._plus_di + self._minus_di
                    if di_sum > 0:
                        dx = abs(self._plus_di - self._minus_di) / di_sum * 100
                        self._dx_values.append(dx)
            return 0.0

        # Wilder's smoothing
        self._plus_dm_sum = self._plus_dm_sum - (self._plus_dm_sum / self.period) + plus_dm
        self._minus_dm_sum = self._minus_dm_sum - (self._minus_dm_sum / self.period) + minus_dm
        self._tr_sum = self._tr_sum - (self._tr_sum / self.period) + tr

        if self._tr_sum > 0:
            self._plus_di = (self._plus_dm_sum / self._tr_sum) * 100
            self._minus_di = (self._minus_dm_sum / self._tr_sum) * 100

        di_sum = self._plus_di + self._minus_di
        if di_sum > 0:
            dx = abs(self._plus_di - self._minus_di) / di_sum * 100
            self._dx_values.append(dx)

        # ADX = smoothed DX
        if len(self._dx_values) >= self.period:
            if not self._initialized:
                self._value = sum(self._dx_values[-self.period:]) / self.period
                self._initialized = True
            else:
                self._value = (self._value * (self.period - 1) + dx) / self.period

        return self._value

    def get_state(self) -> dict:
        return {
            "value": round(self._value, 2),
            "plus_di": round(self._plus_di, 2),
            "minus_di": round(self._minus_di, 2),
            "trending": self.is_trending(),
            "initialized": self._initialized,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           ATR INDIKATORI
# ═══════════════════════════════════════════════════════════════════════════════

class ATRIndicator:
    """ATR — stop distance uchun"""

    def __init__(self, period: int = 14):
        self.period = period
        self._prev_close: Optional[float] = None
        self._atr: float = 0.0
        self._tr_values: List[float] = []
        self._initialized: bool = False

    @property
    def value(self) -> float:
        return self._atr

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, candle: Candle) -> float:
        if self._prev_close is None:
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self._prev_close),
                abs(candle.low - self._prev_close)
            )
        self._prev_close = candle.close
        self._tr_values.append(tr)

        if len(self._tr_values) >= self.period:
            if not self._initialized:
                self._atr = sum(self._tr_values[-self.period:]) / self.period
                self._initialized = True
            else:
                self._atr = (self._atr * (self.period - 1) + tr) / self.period
        return self._atr


# ═══════════════════════════════════════════════════════════════════════════════
#                           HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def money_round(value: float, precision: int = 8) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal(10) ** -precision, rounding=ROUND_HALF_UP))

def calculate_pnl(entry: float, exit: float, qty: float, side: str) -> float:
    if side.lower() in ("long", "buy"):
        return (exit - entry) * qty
    return (entry - exit) * qty
