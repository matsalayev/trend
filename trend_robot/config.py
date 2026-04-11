"""
Trend Robot - Konfiguratsiya

EMA Crossover + Ichimoku Cloud confirmation + ADX trend strength filter.
bitget-futures-ema dan EMA/Ichimoku, HEMA Hedge bot dan ADX olingan.

Asosiy tarkib:
    APIConfig       — Bitget API credentials
    TradingConfig   — Symbol, leverage, margin mode
    EMAConfig       — EMA Crossover parametrlari
    IchimokuConfig  — Ichimoku Cloud parametrlari
    TrendConfig     — ADX filter, multi-timeframe
    MTFConfig       — Multi-timeframe sozlamalari
    ExitConfig      — Trailing stop, opposite signal exit, SL
    GridConfig      — Trend grid (optional martingale)
    RiskConfig      — Max loss, daily loss, fee
    RobotConfig     — Master config
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def _get_env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

def _get_env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

def _get_env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


@dataclass(frozen=True)
class APIConfig:
    API_KEY: str = ""
    SECRET_KEY: str = ""
    PASSPHRASE: str = ""
    BASE_URL: str = "https://api.bitget.com"
    DEMO_MODE: bool = True
    REQUEST_TIMEOUT: float = 10.0

    @classmethod
    def from_env(cls) -> "APIConfig":
        return cls(
            API_KEY=_get_env("BITGET_API_KEY"),
            SECRET_KEY=_get_env("BITGET_SECRET_KEY"),
            PASSPHRASE=_get_env("BITGET_PASSPHRASE"),
            DEMO_MODE=_get_env_bool("DEMO_MODE", True),
        )


@dataclass(frozen=True)
class TradingConfig:
    SYMBOL: str = "BTCUSDT"
    PRODUCT_TYPE: str = "USDT-FUTURES"
    MARGIN_COIN: str = "USDT"
    LEVERAGE: int = 10
    MARGIN_MODE: str = "crossed"

    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            SYMBOL=_get_env("TRADING_SYMBOL", "BTCUSDT"),
            LEVERAGE=_get_env_int("LEVERAGE", 10),
            MARGIN_MODE=_get_env("MARGIN_MODE", "crossed"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           EMA KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EMAConfig:
    """
    EMA Crossover parametrlari — bitget-futures-ema dan.

    Signal:
        Golden Cross: fast EMA > slow EMA (LONG)
        Death Cross: fast EMA < slow EMA (SHORT)

    Crossover detection: oldingi tick da fast <= slow, hozir fast > slow.
    """
    FAST_PERIOD: int = 9      # Tez EMA (qisqa muddat)
    SLOW_PERIOD: int = 21     # Sekin EMA (uzun muddat)

    @classmethod
    def from_env(cls) -> "EMAConfig":
        return cls(
            FAST_PERIOD=_get_env_int("EMA_FAST_PERIOD", 9),
            SLOW_PERIOD=_get_env_int("EMA_SLOW_PERIOD", 21),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           ICHIMOKU KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IchimokuConfig:
    """
    Ichimoku Cloud parametrlari — bitget-futures-ema dan.

    5 ta komponent:
        Tenkan-sen (Conversion): (9-period high + low) / 2
        Kijun-sen (Base): (26-period high + low) / 2
        Senkou Span A: (Tenkan + Kijun) / 2, 26 period kelajakka
        Senkou Span B: (52-period high + low) / 2, 26 period kelajakka
        Chikou Span: close, 26 period o'tmishga

    LONG signal:
        1. Tenkan > Kijun (crossover)
        2. Price > Cloud (Span A va B)
        3. Chikou > price[26 periods ago]
    """
    ENABLED: bool = True
    TENKAN_PERIOD: int = 9
    KIJUN_PERIOD: int = 26
    SENKOU_B_PERIOD: int = 52
    DISPLACEMENT: int = 26

    @classmethod
    def from_env(cls) -> "IchimokuConfig":
        return cls(
            ENABLED=_get_env_bool("USE_ICHIMOKU", True),
            TENKAN_PERIOD=_get_env_int("TENKAN_PERIOD", 9),
            KIJUN_PERIOD=_get_env_int("KIJUN_PERIOD", 26),
            SENKOU_B_PERIOD=_get_env_int("SENKOU_B_PERIOD", 52),
            DISPLACEMENT=_get_env_int("DISPLACEMENT", 26),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           TREND KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrendConfig:
    """
    Trend strength filter — HEMA Hedge bot dan ADX.

    ADX (Average Directional Index):
        > 25 = kuchli trend
        > 35 = juda kuchli trend
        < 20 = flat bozor (signal bloklanadi)
    """
    ADX_PERIOD: int = 14
    ADX_THRESHOLD: float = 25.0

    @classmethod
    def from_env(cls) -> "TrendConfig":
        return cls(
            ADX_PERIOD=_get_env_int("ADX_PERIOD", 14),
            ADX_THRESHOLD=_get_env_float("ADX_THRESHOLD", 25.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           MULTI-TIMEFRAME KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MTFConfig:
    """
    Multi-timeframe confirmation — HEMA RSI bot dan pattern.

    Primary signal: PRIMARY_TIMEFRAME da aniqlash (5m)
    Confirmation: CONFIRM_TIMEFRAME da tasdiqlash (15m)
    Big picture: HTF_TIMEFRAME da umumiy trend (1H)
    """
    ENABLED: bool = True
    PRIMARY_TIMEFRAME: str = "5m"
    CONFIRM_TIMEFRAME: str = "15m"
    HTF_TIMEFRAME: str = "1H"

    @classmethod
    def from_env(cls) -> "MTFConfig":
        return cls(
            ENABLED=_get_env_bool("USE_MTF", True),
            PRIMARY_TIMEFRAME=_get_env("PRIMARY_TIMEFRAME", "5m"),
            CONFIRM_TIMEFRAME=_get_env("CONFIRM_TIMEFRAME", "15m"),
            HTF_TIMEFRAME=_get_env("HTF_TIMEFRAME", "1H"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           EXIT KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExitConfig:
    """
    Exit sozlamalari — trailing stop + opposite signal.

    3-phase trailing stop (HEMA RSI bot dan):
        1. None — trailing faol emas
        2. Breakeven — profit ACTIVATE% da SL break-even ga ko'tariladi
        3. Trail — SL narx ortidan harakatlanadi

    Opposite signal exit:
        LONG pozitsiyada Death Cross → exit
        SHORT pozitsiyada Golden Cross → exit
    """
    USE_TRAILING_STOP: bool = True
    TRAILING_ACTIVATE_PCT: float = 1.0   # Qaysi % da trailing boshlanadi
    TRAILING_FLOOR_PCT: float = 0.3      # Minimal saqlanadigan profit

    USE_OPPOSITE_SIGNAL_EXIT: bool = True  # Qarama-qarshi signal bilan chiqish

    SL_PERCENT: float = 3.0  # Stop loss (fallback)

    @classmethod
    def from_env(cls) -> "ExitConfig":
        return cls(
            USE_TRAILING_STOP=_get_env_bool("USE_TRAILING_STOP", True),
            TRAILING_ACTIVATE_PCT=_get_env_float("TRAILING_ACTIVATE_PCT", 1.0),
            TRAILING_FLOOR_PCT=_get_env_float("TRAILING_FLOOR_PCT", 0.3),
            USE_OPPOSITE_SIGNAL_EXIT=_get_env_bool("USE_OPPOSITE_SIGNAL_EXIT", True),
            SL_PERCENT=_get_env_float("SL_PERCENT", 3.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           GRID KONFIGURATSIYA (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GridConfig:
    """Trend grid — kuchli trend da martingale qo'shish (ixtiyoriy)"""
    ENABLED: bool = False
    MULTIPLIER: float = 1.5
    MAX_ORDERS: int = 5
    MIN_ADX: float = 35.0  # Faqat kuchli trend da grid

    @classmethod
    def from_env(cls) -> "GridConfig":
        return cls(
            ENABLED=_get_env_bool("USE_TREND_GRID", False),
            MULTIPLIER=_get_env_float("GRID_MULTIPLIER", 1.5),
            MAX_ORDERS=_get_env_int("GRID_MAX_ORDERS", 5),
            MIN_ADX=_get_env_float("GRID_MIN_ADX", 35.0),
        )


@dataclass(frozen=True)
class RiskConfig:
    MAX_LOSS_PERCENT: float = 20.0
    MAX_DAILY_LOSS_PERCENT: float = 10.0
    TAKER_FEE_RATE: float = 0.001
    MAKER_FEE_RATE: float = 0.001

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            MAX_LOSS_PERCENT=_get_env_float("MAX_LOSS_PERCENT", 20.0),
            MAX_DAILY_LOSS_PERCENT=_get_env_float("MAX_DAILY_LOSS_PERCENT", 10.0),
            TAKER_FEE_RATE=_get_env_float("TAKER_FEE_RATE", 0.001),
            MAKER_FEE_RATE=_get_env_float("MAKER_FEE_RATE", 0.001),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           ROBOT KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RobotConfig:
    """Master konfiguratsiya"""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    ichimoku: IchimokuConfig = field(default_factory=IchimokuConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    mtf: MTFConfig = field(default_factory=MTFConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    TICK_INTERVAL: float = 1.0
    CAPITAL_ENGAGEMENT: float = 0.15  # 15% per trade (trend bot uchun)
    STATE_DIR: str = "/data/state"
    LOG_LEVEL: str = "INFO"

    @classmethod
    def from_env(cls) -> "RobotConfig":
        return cls(
            api=APIConfig.from_env(),
            trading=TradingConfig.from_env(),
            ema=EMAConfig.from_env(),
            ichimoku=IchimokuConfig.from_env(),
            trend=TrendConfig.from_env(),
            mtf=MTFConfig.from_env(),
            exit=ExitConfig.from_env(),
            grid=GridConfig.from_env(),
            risk=RiskConfig.from_env(),
            TICK_INTERVAL=_get_env_float("TICK_INTERVAL", 1.0),
            CAPITAL_ENGAGEMENT=_get_env_float("CAPITAL_ENGAGEMENT", 0.15),
            STATE_DIR=_get_env("STATE_DIR", "/data/state"),
            LOG_LEVEL=_get_env("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        errors = []
        if self.ema.FAST_PERIOD >= self.ema.SLOW_PERIOD:
            errors.append(f"EMA fast ({self.ema.FAST_PERIOD}) < slow ({self.ema.SLOW_PERIOD}) bo'lishi kerak")
        if self.trading.LEVERAGE < 1 or self.trading.LEVERAGE > 125:
            errors.append(f"LEVERAGE {self.trading.LEVERAGE} — 1-125 orasida")
        if self.trend.ADX_THRESHOLD < 0 or self.trend.ADX_THRESHOLD > 100:
            errors.append(f"ADX_THRESHOLD {self.trend.ADX_THRESHOLD} — 0-100 orasida")
        if self.exit.SL_PERCENT <= 0:
            errors.append(f"SL_PERCENT {self.exit.SL_PERCENT} — 0 dan katta")
        if errors:
            raise ValueError(f"Config xatolari: {'; '.join(errors)}")
        logger.info("Konfiguratsiya tekshirildi")


def validate_trading_settings(settings: dict) -> Optional[str]:
    """HEMA customSettings validation"""
    checks = [
        ("emaFastPeriod", 1, 200, "int"),
        ("emaSlowPeriod", 1, 500, "int"),
        ("adxPeriod", 1, 100, "int"),
        ("adxThreshold", 0, 100, "float"),
        ("trailingActivatePct", 0, 50, "float"),
        ("trailingFloorPct", 0, 50, "float"),
        ("slPercent", 0, 50, "float"),
        ("leverage", 1, 125, "int"),
        ("capitalEngagement", 1, 100, "float"),
        ("feeRate", 0, 1, "float"),
    ]
    for name, mn, mx, vt in checks:
        if name in settings:
            try:
                v = float(settings[name]) if vt == "float" else int(settings[name])
                if v < mn or v > mx:
                    return f"{name}={v} — {mn}-{mx} orasida"
            except (ValueError, TypeError):
                return f"{name} — {vt} bo'lishi kerak"
    if "emaFastPeriod" in settings and "emaSlowPeriod" in settings:
        if int(settings["emaFastPeriod"]) >= int(settings["emaSlowPeriod"]):
            return "emaFastPeriod < emaSlowPeriod bo'lishi kerak"
    return None
