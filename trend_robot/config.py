"""
Trend Robot - Configuration (v2.0 Smart Trend Following)

EMA crossover + ADX + Supertrend + HTF filter bilan ishlaydigan v2.0
strategiyasi uchun config. Per-pair preset'lar presets.json fayldan
yuklanadi (backtest tomonidan optimallashtirilgan).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                               ENV HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
#                               CORE CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════


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


@dataclass
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
    SL_PERCENT: float = 3.0

    @classmethod
    def from_env(cls) -> "ExitConfig":
        return cls(
            SL_PERCENT=_get_env_float("SL_PERCENT", 3.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                           TREND STRATEGY CONFIG (v2.0)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrendConfig:
    """
    Smart Trend Following v2.0 strategiya parametrlari.

    Default qiymatlar — konservativ. Har bir pair uchun haqiqiy optimal
    qiymatlar presets.json'dan yuklanadi (apply_preset_to_config orqali).
    """

    # ── Timeframes ─────────────────────────────────────────────────────────
    timeframe: str = "15m"
    htf_timeframe: str = "4H"

    # ── EMA crossover ──────────────────────────────────────────────────────
    ema_fast: int = 9
    ema_slow: int = 21

    # Cross signal stale check — EMA cross signal'i bitta candle'dan eski
    # bo'lmasa qabul qilinadi. Eski cross'lar (re-init paytida history'dan
    # topilgan) noto'g'ri "ushlangan" bo'lishi mumkin, shuning uchun
    # max_signal_age_bars dan ortiq cross'lar rad etiladi.
    max_signal_age_bars: int = 3

    # ── Volatility / trend strength ────────────────────────────────────────
    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 25.0

    # ── Choppiness Index filter (v2.1) ─────────────────────────────────────
    # CHOP < 38.2 => kuchli trend (entry uchun yaxshi)
    # CHOP > 61.8 => choppy/sideways (entry'dan saqlanish)
    # Bu filter trend strategiyaning eng katta zaifligini hal qiladi:
    # choppy market'da false signal'lar SL'da yopilib bot zarara qiladi.
    use_choppiness_filter: bool = True
    chop_period: int = 14
    chop_max_for_entry: float = 50.0  # CHOP > shu qiymat bo'lsa entry rad etiladi

    # Pair-level "consecutive losses" cooldown (loop guard)
    # Agar oxirgi N ta trade ketma-ket SL'da yopilgan bo'lsa, qo'shimcha cooldown
    consecutive_losses_threshold: int = 3
    consecutive_losses_cooldown_bars: int = 20  # 5 candle SL cooldown'dan tashqari

    # ── Supertrend ─────────────────────────────────────────────────────────
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # ── HTF filter ─────────────────────────────────────────────────────────
    use_htf_filter: bool = True
    htf_ema_fast: int = 21
    htf_ema_slow: int = 50

    # ── Stop loss / trailing ───────────────────────────────────────────────
    initial_sl_percent: float = 3.0
    # ATR-based SL: agar yoqilgan bo'lsa, SL = max(fixed_sl%, atr_multiplier*ATR)
    # Bu past-volatility pair'larda tighter SL beradi, yuqori-volatility'da wider.
    use_atr_sl: bool = True
    sl_atr_multiplier: float = 2.0

    trailing_activation_percent: float = 1.0
    trailing_atr_multiplier: float = 1.5

    # ── Partial TP ─────────────────────────────────────────────────────────
    use_partial_tp: bool = True
    partial_tp1_percent: float = 2.0
    partial_tp1_size_pct: float = 0.33
    partial_tp2_percent: float = 5.0
    partial_tp2_size_pct: float = 0.33

    # ── Opposite signal close ──────────────────────────────────────────────
    # CLAUDE.md: "Opposite signal: EMA death cross exits LONG"
    # Hozir kod'da yo'q. Yoqilsa: LONG ochiq + death cross => yopiladi
    # (cooldown bilan, false signal'larni filter qilish uchun).
    use_opposite_signal_exit: bool = True
    # Opposite signal triggerga ishonch uchun ADX/Supertrend ham tasdiqlashi
    # kerak (oddiy EMA cross emas, to'liq signal).
    opposite_signal_requires_full_confirm: bool = True

    # ── Fee-aware exit ─────────────────────────────────────────────────────
    # Voluntary exit (trailing/opposite/exhaustion) faqat net profit kamida
    # min_net_profit_fee_factor × estimated_fees bo'lganda amalga oshiriladi.
    # SL/MAX_AGE bunga taalluqli emas (ularni bypass qilolmaymiz).
    min_net_profit_fee_factor: float = 1.0

    # ── Risk / cooldown ────────────────────────────────────────────────────
    max_drawdown_percent: float = 20.0
    # Cooldown after SL — CANDLE BAR'larda (sekund yoki tick'da emas).
    # Robot timestamp asosida candle bar'ga aylantiradi.
    cooldown_bars_after_sl: int = 5

    # Trades-per-hour limit — fee-bleed loop himoyasi.
    # 0 = limit yo'q. 4 = soatiga maksimum 4 ta trade (15m candle, 1 per candle).
    max_trades_per_hour: int = 4

    @classmethod
    def from_env(cls) -> "TrendConfig":
        return cls(
            timeframe=_get_env("TIMEFRAME", "15m"),
            htf_timeframe=_get_env("HTF_TIMEFRAME", "4H"),
            ema_fast=_get_env_int("EMA_FAST", 9),
            ema_slow=_get_env_int("EMA_SLOW", 21),
            max_signal_age_bars=_get_env_int("MAX_SIGNAL_AGE_BARS", 3),
            atr_period=_get_env_int("ATR_PERIOD", 14),
            adx_period=_get_env_int("ADX_PERIOD", 14),
            adx_threshold=_get_env_float("ADX_THRESHOLD", 25.0),
            use_choppiness_filter=_get_env_bool("USE_CHOPPINESS_FILTER", True),
            chop_period=_get_env_int("CHOP_PERIOD", 14),
            chop_max_for_entry=_get_env_float("CHOP_MAX_FOR_ENTRY", 50.0),
            consecutive_losses_threshold=_get_env_int("CONSECUTIVE_LOSSES_THRESHOLD", 3),
            consecutive_losses_cooldown_bars=_get_env_int("CONSECUTIVE_LOSSES_COOLDOWN_BARS", 20),
            supertrend_period=_get_env_int("SUPERTREND_PERIOD", 10),
            supertrend_multiplier=_get_env_float("SUPERTREND_MULTIPLIER", 3.0),
            use_htf_filter=_get_env_bool("USE_HTF_FILTER", True),
            htf_ema_fast=_get_env_int("HTF_EMA_FAST", 21),
            htf_ema_slow=_get_env_int("HTF_EMA_SLOW", 50),
            initial_sl_percent=_get_env_float("INITIAL_SL_PERCENT", 3.0),
            use_atr_sl=_get_env_bool("USE_ATR_SL", True),
            sl_atr_multiplier=_get_env_float("SL_ATR_MULTIPLIER", 2.0),
            trailing_activation_percent=_get_env_float("TRAILING_ACTIVATION_PERCENT", 1.0),
            trailing_atr_multiplier=_get_env_float("TRAILING_ATR_MULTIPLIER", 1.5),
            use_partial_tp=_get_env_bool("USE_PARTIAL_TP", True),
            partial_tp1_percent=_get_env_float("PARTIAL_TP1_PERCENT", 2.0),
            partial_tp1_size_pct=_get_env_float("PARTIAL_TP1_SIZE_PCT", 0.33),
            partial_tp2_percent=_get_env_float("PARTIAL_TP2_PERCENT", 5.0),
            partial_tp2_size_pct=_get_env_float("PARTIAL_TP2_SIZE_PCT", 0.33),
            use_opposite_signal_exit=_get_env_bool("USE_OPPOSITE_SIGNAL_EXIT", True),
            opposite_signal_requires_full_confirm=_get_env_bool(
                "OPPOSITE_SIGNAL_REQUIRES_FULL_CONFIRM", True
            ),
            min_net_profit_fee_factor=_get_env_float("MIN_NET_PROFIT_FEE_FACTOR", 1.0),
            max_drawdown_percent=_get_env_float("MAX_DRAWDOWN_PERCENT", 20.0),
            cooldown_bars_after_sl=_get_env_int("COOLDOWN_BARS_AFTER_SL", 5),
            max_trades_per_hour=_get_env_int("MAX_TRADES_PER_HOUR", 4),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                               ROBOT CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RobotConfig:
    """Master konfiguratsiya"""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)

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
            trend=TrendConfig.from_env(),
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
        if self.trend.ema_fast >= self.trend.ema_slow:
            errors.append(
                f"ema_fast ({self.trend.ema_fast}) >= ema_slow ({self.trend.ema_slow})"
            )
        if errors:
            raise ValueError(f"Config xatolari: {'; '.join(errors)}")
        logger.info("Konfiguratsiya tekshirildi")


def validate_trading_settings(settings: dict) -> Optional[str]:
    """HEMA customSettings validation"""
    checks = [
        ("slPercent", 0, 50, "float"),
        ("leverage", 1, 125, "int"),
        ("capitalEngagement", 1, 100, "float"),
        ("feeRate", 0, 1, "float"),
        ("maxLossPercent", 0, 100, "float"),
        ("maxDailyLossPercent", 0, 100, "float"),
        ("emaFast", 2, 200, "int"),
        ("emaSlow", 3, 500, "int"),
        ("adxThreshold", 0, 100, "float"),
        ("adxPeriod", 2, 100, "int"),
        ("atrPeriod", 2, 100, "int"),
        ("supertrendPeriod", 2, 100, "int"),
        ("supertrendMultiplier", 0.1, 20, "float"),
        ("trailingAtrMultiplier", 0.1, 20, "float"),
        ("trailingActivationPercent", 0, 50, "float"),
        ("initialSlPercent", 0, 50, "float"),
        ("htfEmaFast", 2, 500, "int"),
        ("htfEmaSlow", 3, 500, "int"),
        ("partialTp1Percent", 0, 100, "float"),
        ("partialTp2Percent", 0, 100, "float"),
        ("partialTp1SizePct", 0, 1, "float"),
        ("partialTp2SizePct", 0, 1, "float"),
        ("maxDrawdownPercent", 0, 100, "float"),
        ("cooldownBarsAfterSl", 0, 1000, "int"),
        ("maxSignalAgeBars", 1, 100, "int"),
        ("slAtrMultiplier", 0.1, 10, "float"),
        ("minNetProfitFeeFactor", 0, 100, "float"),
        ("maxTradesPerHour", 0, 60, "int"),
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


# ═══════════════════════════════════════════════════════════════════════════════
#                           SYMBOL PRESETS (v2.0)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Per-pair optimal parametrlar backtest asosida topilgan va presets.json
# fayliga yozilgan. User faqat supported pair'lar orasidan tanlashi mumkin.
# Preset avtomatik qo'llaniladi; user customSettings orqali override qila oladi.
# ═══════════════════════════════════════════════════════════════════════════════


# v2.1 v2 walk-forward audit (2026-05-04): faqat 3 pair test'da profitable.
# Frequency-aware optimization (ADX 30-33, CHOP 50-65, SL atr 1.0, Trail atr 2.5):
#   ETHUSDT:  12 test trades, +$56.51 (+5.65%), 83% WR
#   AVAXUSDT: 10 test trades, +$209.83 (+20.98%), 90% WR
#   DOGEUSDT:  8 test trades, +$74.98 (+7.50%), 87.5% WR
# Jami: 30 trade / 60 kun = 0.5 trade/kun (maqsad), +$341 = +11.38% / 60 kun = +69.21% APR
# Drop pair'lar (test'da yo'qotgan yoki frequency past): BTC, SOL, BNB, XRP, ADA, LINK
SUPPORTED_PAIRS: List[str] = [
    "ETHUSDT", "AVAXUSDT", "DOGEUSDT",
]


_DEFAULT_PRESET = {
    "leverage": 10,
    "ema_fast": 9,
    "ema_slow": 21,
    "adx_threshold": 25.0,
    "adx_period": 14,
    "atr_period": 14,
    "supertrend_period": 10,
    "supertrend_multiplier": 3.0,
    "trailing_atr_multiplier": 1.5,
    "trailing_activation_percent": 1.0,
    "initial_sl_percent": 3.0,
    "use_htf_filter": True,
    "htf_ema_fast": 21,
    "htf_ema_slow": 50,
    "use_partial_tp": True,
    "partial_tp1_percent": 2.0,
    "partial_tp1_size_pct": 0.33,
    "partial_tp2_percent": 5.0,
    "partial_tp2_size_pct": 0.33,
    "max_drawdown_percent": 20.0,
    "cooldown_bars_after_sl": 5,
}


def _load_presets_from_file() -> dict:
    """presets.json faylidan yuklash (backtest tomonidan yaratiladi)."""
    paths = [
        Path(__file__).parent / "presets.json",
        Path(__file__).parent.parent / "backtest" / "presets.json",
    ]
    for p in paths:
        if not p.exists():
            continue
        try:
            with p.open() as f:
                data = json.load(f)
            result = {}
            for sym, r in data.items():
                cfg = r.get("config", r)
                result[sym] = {
                    "leverage": int(cfg.get("leverage", 10)),
                    "ema_fast": int(cfg.get("ema_fast", 9)),
                    "ema_slow": int(cfg.get("ema_slow", 21)),
                    "adx_threshold": float(cfg.get("adx_threshold", 25.0)),
                    "adx_period": int(cfg.get("adx_period", 14)),
                    "atr_period": int(cfg.get("atr_period", 14)),
                    "supertrend_period": int(cfg.get("supertrend_period", 10)),
                    "supertrend_multiplier": float(cfg.get("supertrend_multiplier", 3.0)),
                    "trailing_atr_multiplier": float(cfg.get("trailing_atr_multiplier", 1.5)),
                    "trailing_activation_percent": float(
                        cfg.get("trailing_activation_percent", 1.0)
                    ),
                    "initial_sl_percent": float(cfg.get("initial_sl_percent", 3.0)),
                    "use_htf_filter": bool(cfg.get("use_htf_filter", True)),
                    "htf_ema_fast": int(cfg.get("htf_ema_fast", 21)),
                    "htf_ema_slow": int(cfg.get("htf_ema_slow", 50)),
                    "use_partial_tp": bool(cfg.get("use_partial_tp", True)),
                    "partial_tp1_percent": float(cfg.get("partial_tp1_percent", 2.0)),
                    "partial_tp1_size_pct": float(cfg.get("partial_tp1_size_pct", 0.33)),
                    "partial_tp2_percent": float(cfg.get("partial_tp2_percent", 5.0)),
                    "partial_tp2_size_pct": float(cfg.get("partial_tp2_size_pct", 0.33)),
                    "max_drawdown_percent": float(cfg.get("max_drawdown_percent", 20.0)),
                    "cooldown_bars_after_sl": int(cfg.get("cooldown_bars_after_sl", 5)),
                    # Backtest metadatasi (UI da ko'rsatish uchun)
                    "_backtest_return": r.get("return_percent", 0.0),
                    "_backtest_winrate": r.get("winrate", 0.0),
                    "_backtest_trades": r.get("total_trades", 0),
                    "_backtest_drawdown": r.get("max_drawdown_percent", 0.0),
                }
            logger.info(f"Loaded {len(result)} pair presets from {p}")
            return result
        except Exception as e:
            logger.warning(f"Preset load xato ({p}): {e}")
    logger.info("presets.json topilmadi — default preset ishlatiladi")
    return {}


SYMBOL_PRESETS: dict = _load_presets_from_file()


def get_preset(symbol: str) -> dict:
    """Symbol uchun preset — agar topilmasa default."""
    return SYMBOL_PRESETS.get(symbol.upper(), _DEFAULT_PRESET).copy()


def is_supported_pair(symbol: str) -> bool:
    sym = symbol.upper()
    return sym in SUPPORTED_PAIRS or sym in SYMBOL_PRESETS


def apply_preset_to_config(
    config: RobotConfig, symbol: str, user_overrides: Optional[dict] = None
) -> RobotConfig:
    """
    Preset'ni config'ga qo'llash.

    Priority: user_overrides > preset > hardcoded defaults
    user_overrides — snake_case kalitlar bilan dict.
    """
    preset = get_preset(symbol)
    merged = preset.copy()
    if user_overrides:
        for k, v in user_overrides.items():
            if v is None:
                continue
            merged[k] = v

    config.trading.SYMBOL = symbol

    if "leverage" in merged:
        config.trading.LEVERAGE = int(merged["leverage"])

    t = config.trend
    for key in (
        "ema_fast", "ema_slow", "atr_period", "adx_period",
        "supertrend_period", "htf_ema_fast", "htf_ema_slow",
        "cooldown_bars_after_sl", "max_signal_age_bars",
        "max_trades_per_hour", "chop_period",
        "consecutive_losses_threshold", "consecutive_losses_cooldown_bars",
    ):
        if key in merged:
            setattr(t, key, int(merged[key]))

    for key in (
        "adx_threshold", "supertrend_multiplier", "trailing_atr_multiplier",
        "trailing_activation_percent", "initial_sl_percent",
        "sl_atr_multiplier",
        "partial_tp1_percent", "partial_tp1_size_pct",
        "partial_tp2_percent", "partial_tp2_size_pct",
        "max_drawdown_percent", "min_net_profit_fee_factor",
        "chop_max_for_entry",
    ):
        if key in merged:
            setattr(t, key, float(merged[key]))

    for key in (
        "use_htf_filter", "use_partial_tp", "use_atr_sl",
        "use_opposite_signal_exit", "opposite_signal_requires_full_confirm",
        "use_choppiness_filter",
    ):
        if key in merged:
            setattr(t, key, bool(merged[key]))

    if "timeframe" in merged:
        t.timeframe = str(merged["timeframe"])
    if "htf_timeframe" in merged:
        t.htf_timeframe = str(merged["htf_timeframe"])

    # initial_sl_percent → ExitConfig.SL_PERCENT (tickda ishlatish uchun)
    # ExitConfig frozen, shuning uchun yangi instance yaratamiz
    if "initial_sl_percent" in merged:
        config.exit = ExitConfig(SL_PERCENT=float(merged["initial_sl_percent"]))

    return config
