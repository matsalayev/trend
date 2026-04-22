"""
Trend Robot - Configuration (Skeleton)

Strategy-specific konfiguratsiyalar (EMA, Ichimoku, ADX, MTF, Grid) olib
tashlandi — ular qayta yoziladigan strategiya bilan birga qo'shiladi.
Infratuzilma uchun zarur konfiglar saqlangan.
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


@dataclass(frozen=True)
class ExitConfig:
    """Generic exit config — strategy-specific (trailing) olib tashlandi"""
    SL_PERCENT: float = 3.0

    @classmethod
    def from_env(cls) -> "ExitConfig":
        return cls(
            SL_PERCENT=_get_env_float("SL_PERCENT", 3.0),
        )


@dataclass
class RobotConfig:
    """Master konfiguratsiya — skeleton"""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    TICK_INTERVAL: float = 1.0
    CAPITAL_ENGAGEMENT: float = 0.15
    STATE_DIR: str = "/data/state"
    LOG_LEVEL: str = "INFO"

    @classmethod
    def from_env(cls) -> "RobotConfig":
        return cls(
            api=APIConfig.from_env(),
            trading=TradingConfig.from_env(),
            exit=ExitConfig.from_env(),
            risk=RiskConfig.from_env(),
            TICK_INTERVAL=_get_env_float("TICK_INTERVAL", 1.0),
            CAPITAL_ENGAGEMENT=_get_env_float("CAPITAL_ENGAGEMENT", 0.15),
            STATE_DIR=_get_env("STATE_DIR", "/data/state"),
            LOG_LEVEL=_get_env("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        errors = []
        if self.trading.LEVERAGE < 1 or self.trading.LEVERAGE > 125:
            errors.append(f"LEVERAGE {self.trading.LEVERAGE} — 1-125 orasida")
        if self.exit.SL_PERCENT <= 0:
            errors.append(f"SL_PERCENT {self.exit.SL_PERCENT} — 0 dan katta")
        if errors:
            raise ValueError(f"Config xatolari: {'; '.join(errors)}")
        logger.info("Konfiguratsiya tekshirildi")


def validate_trading_settings(settings: dict) -> Optional[str]:
    """HEMA customSettings validation — skeleton"""
    checks = [
        ("slPercent", 0, 50, "float"),
        ("leverage", 1, 125, "int"),
        ("capitalEngagement", 1, 100, "float"),
        ("feeRate", 0, 1, "float"),
        ("maxLossPercent", 0, 100, "float"),
        ("maxDailyLossPercent", 0, 100, "float"),
    ]
    for name, mn, mx, vt in checks:
        if name in settings:
            try:
                v = float(settings[name]) if vt == "float" else int(settings[name])
                if v < mn or v > mx:
                    return f"{name}={v} — {mn}-{mx} orasida"
            except (ValueError, TypeError):
                return f"{name} — {vt} bo'lishi kerak"
    return None
