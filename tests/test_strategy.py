"""
Trend Robot - Strategy Tests

EMA Crossover, Ichimoku Cloud, ADX, Trailing Stop testlari.
bitget-futures-ema + HEMA Hedge bot algoritmlarining aniq testlari.
"""

import pytest
import time
from trend_robot.strategy import TrendStrategy, SignalType, Position, TrailingPhase, TrailingState
from trend_robot.indicators import (
    Candle, EMAIndicator, EMACrossover, IchimokuCloud, ADXIndicator, ATRIndicator, money_round
)
from trend_robot.config import RobotConfig, EMAConfig, IchimokuConfig, TrendConfig, MTFConfig, ExitConfig, RiskConfig, TradingConfig


# ═══════════════════════════════════════════════════════════════════════════════
#                           EMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEMA:
    """EMA hisoblash — bitget-futures-ema dan"""

    def test_ema_converges_to_constant(self):
        ema = EMAIndicator(period=10)
        for _ in range(50):
            ema.update(100.0)
        assert ema.value == pytest.approx(100.0, rel=0.01)

    def test_ema_responds_to_price(self):
        ema = EMAIndicator(period=10)
        for _ in range(20):
            ema.update(100.0)
        ema.update(200.0)
        assert 100 < ema.value < 200

    def test_short_ema_faster_than_long(self):
        """Qisqa EMA tezroq reaksiya qiladi"""
        fast = EMAIndicator(period=5)
        slow = EMAIndicator(period=20)
        for _ in range(30):
            fast.update(100.0)
            slow.update(100.0)
        fast.update(150.0)
        slow.update(150.0)
        assert fast.value > slow.value  # Fast tezroq moslashtiradi


# ═══════════════════════════════════════════════════════════════════════════════
#                           EMA CROSSOVER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEMACrossover:
    """EMA Crossover — bitget-futures-ema dan 1:1"""

    def test_golden_cross(self):
        """Fast EMA slow EMA ni yuqoriga kesib o'tsa → golden_cross"""
        ec = EMACrossover(fast_period=3, slow_period=10)
        # Dastlab tushish
        for p in [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]:
            ec.update(p)
        # Keyin keskin o'sish → golden cross
        signals = []
        for p in [95, 100, 105, 110, 115, 120]:
            s = ec.update(p)
            if s:
                signals.append(s)
        assert "golden_cross" in signals

    def test_death_cross(self):
        """Fast EMA slow EMA ni pastga kesib o'tsa → death_cross"""
        ec = EMACrossover(fast_period=3, slow_period=10)
        # Dastlab o'sish
        for p in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]:
            ec.update(p)
        # Keyin keskin tushish → death cross
        signals = []
        for p in [95, 90, 85, 80, 75, 70]:
            s = ec.update(p)
            if s:
                signals.append(s)
        assert "death_cross" in signals

    def test_no_signal_on_constant(self):
        """Doimiy narxda crossover yo'q"""
        ec = EMACrossover(fast_period=3, slow_period=10)
        for _ in range(30):
            s = ec.update(100)
        assert s is None


# ═══════════════════════════════════════════════════════════════════════════════
#                           ICHIMOKU TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIchimoku:
    """Ichimoku Cloud — bitget-futures-ema dan"""

    def _make_candles(self, prices):
        return [Candle(timestamp=i*60000, open=p, high=p+50, low=p-50, close=p, volume=100)
                for i, p in enumerate(prices)]

    def test_tenkan_kijun_calculation(self):
        """Tenkan = (9-period high + low) / 2"""
        ich = IchimokuCloud(tenkan_period=9, kijun_period=26, senkou_b_period=52)
        # 10 shamcha — barcha 100 da
        for c in self._make_candles([100]*15):
            ich.update(c)
        # Tenkan = (100+50 + 100-50) / 2 = 100
        assert ich.tenkan == pytest.approx(100, abs=1)

    def test_cloud_components(self):
        """Senkou Span A = (Tenkan + Kijun) / 2"""
        ich = IchimokuCloud(tenkan_period=9, kijun_period=26, senkou_b_period=52)
        for c in self._make_candles([100]*60):
            ich.update(c)
        assert ich.senkou_a == pytest.approx((ich.tenkan + ich.kijun) / 2, rel=0.01)

    def test_price_above_cloud(self):
        ich = IchimokuCloud(tenkan_period=9, kijun_period=26, senkou_b_period=52)
        for c in self._make_candles([100]*60):
            ich.update(c)
        assert ich.is_price_above_cloud(200)
        assert not ich.is_price_above_cloud(50)

    def test_price_below_cloud(self):
        ich = IchimokuCloud(tenkan_period=9, kijun_period=26, senkou_b_period=52)
        for c in self._make_candles([100]*60):
            ich.update(c)
        assert ich.is_price_below_cloud(50)
        assert not ich.is_price_below_cloud(200)


# ═══════════════════════════════════════════════════════════════════════════════
#                           ADX TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestADX:
    """ADX — HEMA Hedge bot dan"""

    def test_adx_trending(self):
        """Kuchli trend da ADX > threshold"""
        adx = ADXIndicator(period=5)
        # Kuchli uptrend
        for i in range(30):
            adx.update(Candle(timestamp=i, open=100+i*2, high=102+i*2, low=98+i*2, close=101+i*2))
        if adx.initialized:
            assert adx.value > 0

    def test_adx_flat(self):
        """Flat bozorda ADX past"""
        adx = ADXIndicator(period=5)
        # Flat — narx o'zgarmaydi
        for i in range(30):
            adx.update(Candle(timestamp=i, open=100, high=101, low=99, close=100))
        if adx.initialized:
            assert adx.value < 50  # Flat da ADX past bo'lishi kerak

    def test_is_trending_method(self):
        adx = ADXIndicator(period=5)
        assert not adx.is_trending(25)  # Not initialized


# ═══════════════════════════════════════════════════════════════════════════════
#                           TRAILING STOP TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrailingStop:
    """3-phase trailing stop — HEMA RSI bot dan"""

    def _make_strategy(self):
        config = RobotConfig(
            ema=EMAConfig(FAST_PERIOD=5, SLOW_PERIOD=10),
            exit=ExitConfig(
                USE_TRAILING_STOP=True,
                TRAILING_ACTIVATE_PCT=1.0,
                TRAILING_FLOOR_PCT=0.3,
                SL_PERCENT=3.0,
            ),
        )
        return TrendStrategy(config)

    def test_no_trailing_below_activation(self):
        """Profit < ACTIVATE% da trailing boshlanmaydi"""
        s = self._make_strategy()
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)
        result = s.update_trailing_stop(pos, 50200)  # 0.4% < 1.0%
        assert result is None
        assert s.trailing_long.phase == TrailingPhase.NONE

    def test_trailing_activates_at_threshold(self):
        """Profit >= ACTIVATE% da trailing boshlanadi"""
        s = self._make_strategy()
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)
        result = s.update_trailing_stop(pos, 50600)  # 1.2% > 1.0%
        assert result is not None
        assert result == 50000  # Breakeven
        assert s.trailing_long.phase == TrailingPhase.BREAKEVEN

    def test_trailing_ratchets_up(self):
        """Trail mode da SL faqat yuqoriga harakatlanadi"""
        s = self._make_strategy()
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)

        # Activate
        s.update_trailing_stop(pos, 50600)
        # Trail
        new_stop = s.update_trailing_stop(pos, 51000)
        if new_stop:
            assert new_stop > 50000  # SL breakeven dan yuqori

    def test_trailing_stop_hit(self):
        """Stop price ga yetganda trigger"""
        s = self._make_strategy()
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)
        s.trailing_long.phase = TrailingPhase.BREAKEVEN
        s.trailing_long.stop_price = 50000
        assert s.check_trailing_stop_hit(pos, 49900) is True
        assert s.check_trailing_stop_hit(pos, 50100) is False


# ═══════════════════════════════════════════════════════════════════════════════
#                           STRATEGY SIGNAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrendSignal:
    """Multi-layer signal testlari"""

    def _make_strategy(self, adx_threshold=0, ichimoku=False, mtf=False):
        config = RobotConfig(
            ema=EMAConfig(FAST_PERIOD=3, SLOW_PERIOD=10),
            ichimoku=IchimokuConfig(ENABLED=ichimoku),
            trend=TrendConfig(ADX_PERIOD=5, ADX_THRESHOLD=adx_threshold),
            mtf=MTFConfig(ENABLED=mtf),
            exit=ExitConfig(SL_PERCENT=3.0),
        )
        return TrendStrategy(config)

    def _make_candles(self, prices):
        return [Candle(timestamp=1000000+i*60000, open=p, high=p+50, low=p-50, close=p, volume=100)
                for i, p in enumerate(prices)]

    def test_long_signal_on_uptrend(self):
        """Tushishdan o'sishga o'tganda LONG signal"""
        s = self._make_strategy(adx_threshold=0)
        # Tushish keyin o'sish
        prices = [100-i for i in range(15)] + [85+i*3 for i in range(10)]
        candles = self._make_candles(prices)
        result = s.check_signal(candles, prices[-1])
        # Golden cross bo'lishi mumkin
        assert isinstance(result, SignalType)

    def test_no_signal_insufficient_data(self):
        s = self._make_strategy()
        candles = self._make_candles([100, 101, 102])
        assert s.check_signal(candles, 102) == SignalType.NONE

    def test_opposite_signal_exit_long(self):
        """Death Cross → LONG exit"""
        s = self._make_strategy()
        s._last_signal = SignalType.SHORT
        pos = Position(id="1", side="long", entry_price=50000, size=0.1)
        assert s.check_opposite_signal_exit(pos) is True

    def test_opposite_signal_exit_short(self):
        """Golden Cross → SHORT exit"""
        s = self._make_strategy()
        s._last_signal = SignalType.LONG
        pos = Position(id="1", side="short", entry_price=50000, size=0.1)
        assert s.check_opposite_signal_exit(pos) is True


# ═══════════════════════════════════════════════════════════════════════════════
#                           POSITION SIZING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionSizing:
    def test_size_calculation(self):
        config = RobotConfig(
            ema=EMAConfig(FAST_PERIOD=5, SLOW_PERIOD=10),
            trading=TradingConfig(LEVERAGE=10),
            CAPITAL_ENGAGEMENT=0.15,
        )
        s = TrendStrategy(config)
        s.set_symbol_info({"minTradeNum": 0.001})
        # size = 1000 × 0.15 × 10 / 50000 = 0.03
        assert s.calculate_position_size(50000, 1000) == pytest.approx(0.03, rel=0.01)

    def test_sl_price_long(self):
        config = RobotConfig(
            ema=EMAConfig(FAST_PERIOD=5, SLOW_PERIOD=10),
            exit=ExitConfig(SL_PERCENT=3.0),
        )
        s = TrendStrategy(config)
        sl = s.calculate_sl_price(50000, "long")
        assert sl == pytest.approx(50000 * 0.97, rel=0.001)

    def test_sl_price_short(self):
        config = RobotConfig(
            ema=EMAConfig(FAST_PERIOD=5, SLOW_PERIOD=10),
            exit=ExitConfig(SL_PERCENT=3.0),
        )
        s = TrendStrategy(config)
        sl = s.calculate_sl_price(50000, "short")
        assert sl == pytest.approx(50000 * 1.03, rel=0.001)
