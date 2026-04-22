"""
Trend Robot - Strategy Skeleton Tests

Strategiya qayta yozilgandan keyin bu testlar almashtiriladi. Hozircha faqat
skeleton kontrakti tekshiriladi: check_signal har doim NONE, placeholderlar
to'g'ri default qiymatlar qaytaradi.
"""

import pytest

from trend_robot.strategy import (
    TrendStrategy, SignalType, Position, TrailingPhase, TrailingState,
)
from trend_robot.indicators import Candle, money_round, calculate_pnl
from trend_robot.config import (
    RobotConfig, TradingConfig, ExitConfig, RiskConfig,
)


def _make_config():
    return RobotConfig(
        trading=TradingConfig(LEVERAGE=10),
        exit=ExitConfig(SL_PERCENT=3.0),
        risk=RiskConfig(),
        CAPITAL_ENGAGEMENT=0.15,
    )


class TestSkeletonSignal:
    def test_check_signal_always_none(self):
        s = TrendStrategy(_make_config())
        assert s.check_signal(candles=[], current_price=50000) == SignalType.NONE

    def test_update_candles_noop(self):
        s = TrendStrategy(_make_config())
        assert s.update_candles([]) is None

    def test_last_signal_default(self):
        s = TrendStrategy(_make_config())
        assert s.get_status()["last_signal"] == "NONE"


class TestSkeletonExits:
    def test_update_trailing_stop_returns_none(self):
        s = TrendStrategy(_make_config())
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)
        assert s.update_trailing_stop(pos, 51000) is None

    def test_check_trailing_stop_hit_returns_false(self):
        s = TrendStrategy(_make_config())
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)
        assert s.check_trailing_stop_hit(pos, 49000) is False

    def test_check_opposite_signal_exit_returns_false(self):
        s = TrendStrategy(_make_config())
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)
        assert s.check_opposite_signal_exit(pos) is False

    def test_on_trade_closed_resets_trailing(self):
        s = TrendStrategy(_make_config())
        s.trailing_long.phase = TrailingPhase.BREAKEVEN
        s.on_trade_closed("long")
        assert s.trailing_long.phase == TrailingPhase.NONE


class TestPositionSizing:
    def test_size_calculation(self):
        s = TrendStrategy(_make_config())
        s.set_symbol_info({"minTradeNum": 0.001})
        # size = 1000 * 0.15 * 10 / 50000 = 0.03
        assert s.calculate_position_size(50000, 1000) == pytest.approx(0.03, rel=0.01)

    def test_size_zero_when_below_min(self):
        s = TrendStrategy(_make_config())
        s.set_symbol_info({"minTradeNum": 1.0})
        # juda kichik hajm → 0
        assert s.calculate_position_size(50000, 10) == 0.0

    def test_sl_price_long(self):
        s = TrendStrategy(_make_config())
        assert s.calculate_sl_price(50000, "long") == pytest.approx(50000 * 0.97, rel=0.001)

    def test_sl_price_short(self):
        s = TrendStrategy(_make_config())
        assert s.calculate_sl_price(50000, "short") == pytest.approx(50000 * 1.03, rel=0.001)


class TestHelpers:
    def test_money_round(self):
        assert money_round(1.234567891, 2) == 1.23

    def test_calculate_pnl_long(self):
        assert calculate_pnl(100, 110, 2, "long") == pytest.approx(20)

    def test_calculate_pnl_short(self):
        assert calculate_pnl(110, 100, 2, "short") == pytest.approx(20)


class TestCandleDataclass:
    def test_candle_fields(self):
        c = Candle(timestamp=1000, open=1, high=2, low=0.5, close=1.5, volume=10)
        assert c.close == 1.5
        assert c.volume == 10


class TestTrailingStateDefaults:
    def test_default_phase_is_none(self):
        st = TrailingState()
        assert st.phase == TrailingPhase.NONE
        assert st.stop_price == 0.0
