"""
Trend Following Robot — Strategy v2.1 Tests
"""
import time

import pytest

from trend_robot.strategy import (
    TrendStrategy, SignalType, Position, TrailingPhase, timeframe_to_seconds,
)
from trend_robot.indicators import (
    Candle, money_round, calculate_pnl,
    EMAIndicator, ATRIndicator, ADXIndicator, SupertrendIndicator, EMACrossover,
)
from trend_robot.config import (
    RobotConfig, apply_preset_to_config, SYMBOL_PRESETS,
    is_supported_pair, get_preset,
)


def _make_candle(ts, o, h, l, c, v=100.0):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)


def _uptrend_candles(n=100, start=100.0):
    # 15m bars (900_000ms) — TimeframeToSeconds bilan mos bo'lishi uchun
    return [
        _make_candle(1_700_000_000_000 + i * 900_000,
                     start + i * 0.5, start + i * 0.5 + 1.0,
                     start + i * 0.5 - 0.5, start + i * 0.5 + 0.8)
        for i in range(n)
    ]


@pytest.fixture
def config():
    return apply_preset_to_config(RobotConfig(), "ETHUSDT", {})


@pytest.fixture
def strategy(config):
    return TrendStrategy(config)


class TestIndicators:
    def test_ema_not_initialized_early(self):
        ema = EMAIndicator(period=20)
        ema.update(100.0)
        assert not ema.initialized

    def test_ema_initialized_after_period(self):
        ema = EMAIndicator(period=5)
        for p in [100, 101, 102, 103, 104]:
            ema.update(p)
        assert ema.initialized

    def test_atr_initialized(self):
        atr = ATRIndicator(period=5)
        atr.calculate_from_candles(_uptrend_candles(10))
        assert atr.initialized
        assert atr.value > 0

    def test_adx_initialized(self):
        adx = ADXIndicator(period=7)
        adx.calculate_from_candles(_uptrend_candles(30))
        assert adx.initialized

    def test_supertrend_has_direction(self):
        st = SupertrendIndicator(period=5, multiplier=3.0)
        st.calculate_from_candles(_uptrend_candles(30))
        assert st.direction in (1, -1)

    def test_ema_crossover_can_detect(self):
        cross = EMACrossover(fast_period=3, slow_period=5)
        for p in [100, 99, 98, 97, 98, 100, 102, 105, 108, 110, 112]:
            cross.update(_make_candle(0, p, p + 0.5, p - 0.5, p))
        assert cross.initialized


class TestPosition:
    def test_long_pnl_profit(self):
        p = Position(id="1", side="long", entry_price=100.0, size=1.0)
        assert p.pnl_at(110.0) == pytest.approx(10.0)

    def test_long_pnl_loss(self):
        p = Position(id="1", side="long", entry_price=100.0, size=1.0)
        assert p.pnl_at(90.0) == pytest.approx(-10.0)

    def test_short_pnl_profit(self):
        p = Position(id="1", side="short", entry_price=100.0, size=1.0)
        assert p.pnl_at(90.0) == pytest.approx(10.0)

    def test_pnl_pct(self):
        p = Position(id="1", side="long", entry_price=100.0, size=1.0)
        assert p.pnl_pct_at(105.0) == pytest.approx(5.0)

    def test_notional(self):
        p = Position(id="1", side="long", entry_price=100.0, size=2.5)
        assert p.notional == pytest.approx(250.0)

    def test_initial_size_defaults(self):
        p = Position(id="1", side="long", entry_price=100.0, size=1.0)
        assert p.initial_size == 1.0

    def test_to_dict(self):
        p = Position(id="1", side="long", entry_price=100.0, size=1.0)
        d = p.to_dict()
        assert d["side"] == "long"
        assert d["trailing_phase"] == "none"


class TestStrategy:
    def test_initial_state(self, strategy):
        assert strategy.position is None
        assert strategy.total_trades == 0

    def test_no_signal_when_indicators_uninitialized(self, strategy):
        assert strategy.detect_signal(100.0, 0) == SignalType.NONE

    def test_update_indicators(self, strategy):
        candles = _uptrend_candles(60)
        strategy.update_indicators(candles)
        assert strategy.atr.initialized

    def test_create_position_long(self, strategy):
        pos = strategy.create_position("long", 100.0, 1.0, "o1")
        assert strategy.position == pos
        assert pos.stop_price < pos.entry_price

    def test_create_position_short(self, strategy):
        pos = strategy.create_position("short", 100.0, 1.0, "o2")
        assert pos.stop_price > pos.entry_price

    def test_initial_stop_loss_long(self, strategy):
        sl = strategy.initial_stop_loss("long", 100.0)
        expected = 100.0 * (1 - strategy.cfg.initial_sl_percent / 100)
        assert sl == pytest.approx(expected)

    def test_initial_stop_loss_short(self, strategy):
        sl = strategy.initial_stop_loss("short", 100.0)
        expected = 100.0 * (1 + strategy.cfg.initial_sl_percent / 100)
        assert sl == pytest.approx(expected)

    def test_trailing_stop_activation(self, strategy):
        strategy.atr._atr = 1.0
        strategy.atr._initialized = True
        pos = strategy.create_position("long", 100.0, 1.0)
        activation_price = 100.0 * (1 + strategy.cfg.trailing_activation_percent / 100)
        strategy.update_trailing_stop(pos, activation_price + 0.5)
        assert pos.trailing_phase != TrailingPhase.NONE

    def test_stop_hit_long(self, strategy):
        pos = strategy.create_position("long", 100.0, 1.0)
        pos.stop_price = 95.0
        hit = strategy.check_stop_hit(pos, high=96.0, low=94.0)
        assert hit == pytest.approx(95.0)

    def test_stop_hit_short(self, strategy):
        pos = strategy.create_position("short", 100.0, 1.0)
        pos.stop_price = 105.0
        hit = strategy.check_stop_hit(pos, high=106.0, low=103.0)
        assert hit == pytest.approx(105.0)

    def test_stop_not_hit(self, strategy):
        pos = strategy.create_position("long", 100.0, 1.0)
        pos.stop_price = 95.0
        hit = strategy.check_stop_hit(pos, high=101.0, low=96.0)
        assert hit is None

    def test_partial_tp_long_hit(self, strategy):
        strategy.cfg.use_partial_tp = True
        strategy.cfg.partial_tp1_percent = 2.0
        strategy.cfg.partial_tp1_size_pct = 0.33
        pos = strategy.create_position("long", 100.0, 1.0)
        res = strategy.check_partial_tp(pos, high=102.5, low=100.0)
        assert res is not None
        exit_price, close_size = res
        assert exit_price == pytest.approx(102.0)
        assert pos.partial_tp1_done

    def test_partial_tp_disabled(self, strategy):
        strategy.cfg.use_partial_tp = False
        pos = strategy.create_position("long", 100.0, 1.0)
        assert strategy.check_partial_tp(pos, high=105.0, low=100.0) is None

    def test_calculate_position_size(self, strategy):
        # v2.1 semantikasi: trade_amount = MARGIN, notional = margin × leverage
        # margin=$1000, leverage=10x, price=$100 → notional=$10K → size=100
        size = strategy.calculate_position_size(100.0, 1000.0, 10)
        assert size == pytest.approx(100.0)

    def test_calculate_position_size_zero_amount(self, strategy):
        assert strategy.calculate_position_size(100.0, 0.0, 10) == 0.0
        assert strategy.calculate_position_size(100.0, -5.0, 10) == 0.0

    def test_calculate_position_size_zero_price(self, strategy):
        assert strategy.calculate_position_size(0.0, 100.0, 10) == 0.0

    def test_calculate_position_size_below_min_lot(self, strategy):
        # min_lot 0.001, price=$100K, $0.05 margin × 1x = 0.0000005 — too small
        size = strategy.calculate_position_size(100_000.0, 0.05, 1)
        assert size == 0.0

    def test_can_trade_today(self, strategy):
        assert strategy.can_trade_today()

    def test_on_trade_closed_win(self, strategy):
        strategy.on_trade_closed(10.0)
        assert strategy.winning_trades == 1
        assert strategy.realized_pnl == pytest.approx(10.0)

    def test_on_trade_closed_loss(self, strategy):
        strategy.on_trade_closed(-5.0)
        assert strategy.losing_trades == 1

    def test_get_stats(self, strategy):
        strategy.on_trade_closed(10.0)
        strategy.on_trade_closed(-3.0)
        stats = strategy.get_stats()
        assert stats["trades"]["total"] == 2
        assert stats["trades"]["winrate"] == 50.0


class TestPresets:
    def test_symbol_presets_loaded(self):
        assert len(SYMBOL_PRESETS) > 0

    def test_supported_pair_true(self):
        assert is_supported_pair("ETHUSDT")

    def test_supported_pair_false(self):
        assert not is_supported_pair("UNKNOWN_PAIR")

    def test_get_preset_returns_dict(self):
        p = get_preset("ETHUSDT")
        assert "leverage" in p
        assert "ema_fast" in p

    def test_apply_preset_applies_leverage(self):
        result = apply_preset_to_config(RobotConfig(), "ETHUSDT", {})
        assert result.trading.LEVERAGE > 0

    def test_user_override_priority(self):
        result = apply_preset_to_config(RobotConfig(), "ETHUSDT", {"leverage": 3})
        assert result.trading.LEVERAGE == 3


class TestUtility:
    def test_calculate_pnl_long(self):
        assert calculate_pnl(100.0, 110.0, 1.0, "long") == pytest.approx(10.0)

    def test_calculate_pnl_short(self):
        assert calculate_pnl(100.0, 90.0, 1.0, "short") == pytest.approx(10.0)

    def test_money_round(self):
        assert money_round(1.23456789, 4) == pytest.approx(1.2346)

    def test_timeframe_to_seconds(self):
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("15m") == 900
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("4H") == 14400
        assert timeframe_to_seconds("1d") == 86400
        # Default fallback
        assert timeframe_to_seconds("unknown") == 900


class TestATRStopLoss:
    """v2.1: ATR-adaptive SL — fixed % vs ATR-based ning kattasi."""

    def test_atr_sl_disabled_uses_fixed(self, strategy):
        strategy.cfg.use_atr_sl = False
        strategy.cfg.initial_sl_percent = 3.0
        # ATR initialized lekin ignore qilinishi kerak
        strategy.atr._atr = 5.0
        strategy.atr._initialized = True
        sl = strategy.initial_stop_loss("long", 100.0)
        # Faqat fixed 3% SL
        assert sl == pytest.approx(97.0)

    def test_atr_sl_high_volatility_widens(self, strategy):
        strategy.cfg.use_atr_sl = True
        strategy.cfg.initial_sl_percent = 1.0  # past
        strategy.cfg.sl_atr_multiplier = 2.0
        strategy.atr._atr = 10.0  # ATR 10% of price
        strategy.atr._initialized = True
        sl = strategy.initial_stop_loss("long", 100.0)
        # ATR-based: 100 - 2 × 10 = 80, fixed: 100 - 1 = 99
        # max(fixed_pct=1%, atr_pct=20%) = 20% wider SL = 80
        assert sl == pytest.approx(80.0)

    def test_atr_sl_low_volatility_uses_fixed(self, strategy):
        strategy.cfg.use_atr_sl = True
        strategy.cfg.initial_sl_percent = 3.0
        strategy.cfg.sl_atr_multiplier = 2.0
        strategy.atr._atr = 0.5  # ATR small, price 100 → 1% < fixed 3%
        strategy.atr._initialized = True
        sl = strategy.initial_stop_loss("long", 100.0)
        assert sl == pytest.approx(97.0)  # fixed wins

    def test_atr_sl_short_side(self, strategy):
        strategy.cfg.use_atr_sl = True
        strategy.cfg.initial_sl_percent = 1.0
        strategy.cfg.sl_atr_multiplier = 2.0
        strategy.atr._atr = 5.0
        strategy.atr._initialized = True
        sl = strategy.initial_stop_loss("short", 100.0)
        # SHORT: SL is above entry. ATR pct = 10%, so SL = 110
        assert sl == pytest.approx(110.0)


class TestCooldown:
    """v2.1: cooldown timestamp asosida (bar count × candle duration), tick'da emas."""

    def test_cooldown_timestamp_based(self, strategy):
        strategy.cfg.timeframe = "15m"
        strategy.cfg.cooldown_bars_after_sl = 5
        now = 1_700_000_000.0
        strategy.set_cooldown(now_ts=now)
        # 5 bars × 900 sec = 4500 sec
        assert strategy._cooldown_until_ts == pytest.approx(now + 4500)

    def test_in_cooldown_blocks_signal(self, strategy):
        strategy.cfg.timeframe = "15m"
        strategy.cfg.cooldown_bars_after_sl = 5
        now = 1_700_000_000.0
        strategy.set_cooldown(now_ts=now)
        # 100 sec keyin — hali cooldown
        assert strategy._is_in_cooldown(now_ts=now + 100)
        # 5000 sec keyin — cooldown tugagan
        assert not strategy._is_in_cooldown(now_ts=now + 5000)

    def test_cooldown_remaining(self, strategy):
        strategy.cfg.timeframe = "15m"
        strategy.cfg.cooldown_bars_after_sl = 5
        now = 1_700_000_000.0
        strategy.set_cooldown(now_ts=now)
        assert strategy.cooldown_remaining_seconds(now_ts=now + 1000) == pytest.approx(3500)


class TestStaleSignal:
    """v2.1: EMA cross signal max_signal_age_bars dan eski emas."""

    def test_signal_age_bars_calculation(self, strategy):
        strategy.cfg.timeframe = "15m"
        strategy._last_cross_ts = 1_700_000_000_000
        strategy._last_candle_ts = 1_700_000_000_000 + 3 * 900_000  # 3 bars later
        assert strategy._signal_age_bars() == 3

    def test_no_cross_means_stale(self, strategy):
        strategy._last_cross_ts = 0
        strategy._last_candle_ts = 1_700_000_000_000
        assert strategy._signal_age_bars() >= 100

    def test_stale_cross_rejected_in_detect_signal(self, strategy):
        # Setup a stale cross
        strategy.cfg.timeframe = "15m"
        strategy.cfg.max_signal_age_bars = 3
        strategy._last_cross = "golden_cross"
        strategy._last_cross_ts = 1_700_000_000_000
        strategy._last_candle_ts = 1_700_000_000_000 + 10 * 900_000  # 10 bars later
        # Force-init indicators
        strategy.ema.fast._initialized = True
        strategy.ema.slow._initialized = True
        strategy.atr._initialized = True
        strategy.adx._initialized = True
        strategy.adx._adx = 30.0
        strategy.supertrend._values = [(0.0, 1)]
        strategy.supertrend._atr._initialized = True
        sig = strategy.detect_signal(100.0)
        assert sig == SignalType.NONE
        # Stale cross ham clear bo'ladi
        assert strategy._last_cross is None


class TestRateLimit:
    """v2.1: trades-per-hour limit — fee-bleed loop himoyasi."""

    def test_rate_limit_blocks_after_threshold(self, strategy):
        strategy.cfg.max_trades_per_hour = 4
        now = 1_700_000_000.0
        # 4 ta entry oxirgi soatda
        for offset in (0, 100, 200, 300):
            strategy.register_entry(now_ts=now + offset)
        # 5-chi entry — limit'ga yetdi
        assert strategy._is_rate_limited(now_ts=now + 400)

    def test_rate_limit_resets_after_hour(self, strategy):
        strategy.cfg.max_trades_per_hour = 2
        now = 1_700_000_000.0
        strategy.register_entry(now_ts=now)
        strategy.register_entry(now_ts=now + 100)
        assert strategy._is_rate_limited(now_ts=now + 200)
        # 1 soatdan keyin — barcha eski entry'lar tushib ketdi
        assert not strategy._is_rate_limited(now_ts=now + 3700)

    def test_rate_limit_disabled(self, strategy):
        strategy.cfg.max_trades_per_hour = 0
        now = 1_700_000_000.0
        for offset in range(100):
            strategy.register_entry(now_ts=now + offset)
        # Limit yo'q — hech qachon rate-limited bo'lmaydi
        assert not strategy._is_rate_limited(now_ts=now + 200)


class TestOppositeSignalExit:
    """v2.1: opposite EMA cross pozitsiyani yopadi."""

    def _setup_indicators(self, strategy, adx_value=30.0, st_dir=1, htf=False):
        strategy.ema.fast._initialized = True
        strategy.ema.slow._initialized = True
        strategy.atr._initialized = True
        strategy.adx._initialized = True
        strategy.adx._adx = adx_value
        strategy.supertrend._values = [(0.0, st_dir)]
        strategy.supertrend._atr._initialized = True
        strategy.cfg.use_htf_filter = htf
        strategy._last_candle_ts = 1_700_000_000_000

    def test_opposite_exit_disabled(self, strategy):
        strategy.cfg.use_opposite_signal_exit = False
        strategy._last_cross = "death_cross"
        assert not strategy.detect_opposite_exit("long")

    def test_long_exits_on_death_cross(self, strategy):
        strategy.cfg.use_opposite_signal_exit = True
        strategy.cfg.opposite_signal_requires_full_confirm = False
        strategy.ema.fast._initialized = True
        strategy.ema.slow._initialized = True
        strategy._last_cross = "death_cross"
        strategy._last_cross_ts = 1_700_000_000_000
        strategy.cfg.timeframe = "15m"
        strategy._last_candle_ts = 1_700_000_000_000  # same bar
        assert strategy.detect_opposite_exit("long")

    def test_short_exits_on_golden_cross(self, strategy):
        strategy.cfg.use_opposite_signal_exit = True
        strategy.cfg.opposite_signal_requires_full_confirm = False
        strategy.ema.fast._initialized = True
        strategy.ema.slow._initialized = True
        strategy._last_cross = "golden_cross"
        strategy._last_cross_ts = 1_700_000_000_000
        strategy.cfg.timeframe = "15m"
        strategy._last_candle_ts = 1_700_000_000_000
        assert strategy.detect_opposite_exit("short")

    def test_no_opposite_for_same_direction(self, strategy):
        strategy.cfg.use_opposite_signal_exit = True
        strategy.cfg.opposite_signal_requires_full_confirm = False
        strategy._last_cross = "golden_cross"
        strategy._last_cross_ts = 1_700_000_000_000
        strategy._last_candle_ts = 1_700_000_000_000
        # LONG va golden_cross — bir tomonga, exit emas
        assert not strategy.detect_opposite_exit("long")

    def test_full_confirm_blocks_weak_opposite(self, strategy):
        strategy.cfg.timeframe = "15m"
        strategy.cfg.use_opposite_signal_exit = True
        strategy.cfg.opposite_signal_requires_full_confirm = True
        # ADX past — full confirm pass etmaydi
        self._setup_indicators(strategy, adx_value=10.0, st_dir=-1)  # ADX 10 < 25
        strategy._last_cross = "death_cross"
        strategy._last_cross_ts = 1_700_000_000_000
        strategy._last_candle_ts = 1_700_000_000_000
        assert not strategy.detect_opposite_exit("long")


class TestFeeAwareExit:
    """v2.1: voluntary exit faqat gross profit fee'ni qoplaganda."""

    def test_fee_aware_disabled(self, strategy):
        strategy.cfg.min_net_profit_fee_factor = 0.0
        pos = strategy.create_position("long", 100.0, 1.0)
        # Hatto zararda ham voluntary exit ruxsat
        assert strategy.voluntary_exit_allowed(pos, 99.0)

    def test_fee_aware_blocks_below_threshold(self, strategy):
        strategy.cfg.min_net_profit_fee_factor = 1.0
        # Taker fee rate
        strategy.config.risk = strategy.config.risk.__class__(TAKER_FEE_RATE=0.001)
        pos = strategy.create_position("long", 100.0, 1.0)
        # Entry fee = 100 × 1 × 0.001 = $0.10
        # Exit fee at 100.05 = $0.10 → round trip fees = $0.20
        # Gross at 100.05 = $0.05 < $0.20 → exit BLOCKED
        assert not strategy.voluntary_exit_allowed(pos, 100.05)

    def test_fee_aware_allows_above_threshold(self, strategy):
        strategy.cfg.min_net_profit_fee_factor = 1.0
        strategy.config.risk = strategy.config.risk.__class__(TAKER_FEE_RATE=0.001)
        pos = strategy.create_position("long", 100.0, 1.0)
        # Gross at 101 = $1.00, fees ~= $0.20 → ALLOWED
        assert strategy.voluntary_exit_allowed(pos, 101.0)


class TestEntryRegistration:
    """v2.1: register_entry rolling-window'ga qo'shadi."""

    def test_register_entry_appends(self, strategy):
        strategy.cfg.max_trades_per_hour = 10
        now = 1_700_000_000.0
        strategy.register_entry(now_ts=now)
        strategy.register_entry(now_ts=now + 60)
        assert len(strategy._recent_trade_ts) == 2

    def test_register_entry_prunes_old(self, strategy):
        strategy.cfg.max_trades_per_hour = 10
        now = 1_700_000_000.0
        strategy.register_entry(now_ts=now)
        # Yangi entry 2 soat keyin — birinchi entry tushib ketishi kerak
        strategy.register_entry(now_ts=now + 7200)
        assert len(strategy._recent_trade_ts) == 1


class TestResetClears:
    """reset() yangi v2.1 fields'ni ham tozalashi kerak."""

    def test_reset_clears_cooldown_and_cross(self, strategy):
        strategy._cooldown_until_ts = 9999999.0
        strategy._last_cross = "golden_cross"
        strategy._last_cross_ts = 1234567890
        strategy._recent_trade_ts = [1.0, 2.0, 3.0]
        strategy.reset()
        assert strategy._cooldown_until_ts == 0.0
        assert strategy._last_cross is None
        assert strategy._last_cross_ts == 0
        assert strategy._recent_trade_ts == []
