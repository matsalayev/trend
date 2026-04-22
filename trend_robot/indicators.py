"""
Trend Robot - Indicators (Skeleton)

Strategiya logikasi qayta yozilishida indikatorlar shu yerga qo'shiladi.
Hozircha faqat umumiy Candle dataclass va helper funksiyalar saqlanadi.
"""

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


@dataclass
class Candle:
    """OHLCV shamcha"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def money_round(value: float, precision: int = 8) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal(10) ** -precision, rounding=ROUND_HALF_UP))


def calculate_pnl(entry: float, exit: float, qty: float, side: str) -> float:
    if side.lower() in ("long", "buy"):
        return (exit - entry) * qty
    return (entry - exit) * qty
