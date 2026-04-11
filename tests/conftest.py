"""Trend Robot - Shared Test Fixtures"""

import pytest
from unittest.mock import AsyncMock

from trend_robot.config import (
    APIConfig, TradingConfig, EMAConfig, IchimokuConfig, TrendConfig,
    MTFConfig, ExitConfig, GridConfig, RiskConfig, RobotConfig
)


@pytest.fixture
def default_robot_config():
    return RobotConfig(
        api=APIConfig(DEMO_MODE=True, API_KEY="test", SECRET_KEY="test", PASSPHRASE="test"),
        trading=TradingConfig(SYMBOL="BTCUSDT", LEVERAGE=10),
        ema=EMAConfig(FAST_PERIOD=9, SLOW_PERIOD=21),
        ichimoku=IchimokuConfig(ENABLED=True, TENKAN_PERIOD=9, KIJUN_PERIOD=26),
        trend=TrendConfig(ADX_PERIOD=14, ADX_THRESHOLD=25),
        mtf=MTFConfig(ENABLED=False),
        exit=ExitConfig(USE_TRAILING_STOP=True, TRAILING_ACTIVATE_PCT=1.0, TRAILING_FLOOR_PCT=0.3, SL_PERCENT=3.0),
        grid=GridConfig(ENABLED=False),
        risk=RiskConfig(MAX_LOSS_PERCENT=20, TAKER_FEE_RATE=0.001),
        CAPITAL_ENGAGEMENT=0.15,
    )


@pytest.fixture
def mock_bitget_client():
    client = AsyncMock()
    client.get_balance.return_value = 1000.0
    client.get_equity.return_value = 1000.0
    client.get_price.return_value = 50000.0
    client.get_positions.return_value = []
    client.get_candles.return_value = []
    client.place_order.return_value = {"orderId": "test_1"}
    client.get_symbol_info.return_value = {"symbol": "BTCUSDT", "minTradeNum": 0.001}
    client.close.return_value = None
    return client
