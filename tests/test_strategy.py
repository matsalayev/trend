"""
Trend Following Robot — Strategy v2.0 Tests
"""
import pytest

from trend_robot.strategy import (
    TrendStrategy, SignalType, Position, TrailingPhase,
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
    return [
        _make_candle(1_700_000_000_000 + i * 60_000,
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
        size = strategy.calculate_position_size(100.0, 1000.0, 10)
        assert size == pytest.approx(100.0)

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
