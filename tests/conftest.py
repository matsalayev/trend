"""Trend Robot - Shared Test Fixtures (Skeleton)"""

import pytest
from unittest.mock import AsyncMock

from trend_robot.config import (
    APIConfig, TradingConfig, ExitConfig, RiskConfig, RobotConfig,
)


@pytest.fixture
def default_robot_config():
    return RobotConfig(
        api=APIConfig(DEMO_MODE=True, API_KEY="test", SECRET_KEY="test", PASSPHRASE="test"),
        trading=TradingConfig(SYMBOL="BTCUSDT", LEVERAGE=10),
        exit=ExitConfig(SL_PERCENT=3.0),
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
