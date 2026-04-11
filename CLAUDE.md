# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Trend Robot is a trend-following trading bot for Bitget Futures, inspired by bitget-futures-ema and Ichimoku Cloud analysis. It uses EMA Crossover as primary signal with Ichimoku Cloud confirmation, multi-timeframe validation, and ADX trend strength filtering.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Standalone CLI mode
python run.py
python run.py --symbol BTCUSDT --leverage 10 --demo

# Server mode (HEMA integration)
python run_server.py
python run_server.py --host 0.0.0.0 --port 8090

# Run tests
pytest tests/
pytest tests/test_strategy.py -v

# Docker
docker-compose up -d
```

## Architecture

### Core Components (`trend_robot/`)

| File | Responsibility |
|------|---------------|
| `config.py` | EMAConfig, IchimokuConfig, TrendConfig, MTFConfig, ExitConfig, GridConfig, RiskConfig, RobotConfig |
| `indicators.py` | EMA, IchimokuCloud (Tenkan/Kijun/Senkou/Chikou), ADX, ATR |
| `strategy.py` | TrendStrategy — EMA cross, Ichimoku confirm, MTF validation, trailing stop, opposite signal exit |
| `robot.py` | Main tick loop, state machine |
| `api_client.py` | Async Bitget REST client with HMAC-SHA256 auth, circuit breaker |
| `server.py` | FastAPI endpoints for HEMA integration |
| `session_manager.py` | Multi-user sessions, TrendRobotWithWebhook subclass |
| `webhook_client.py` | Async event queue with priority events |
| `state_persistence.py` | JSON atomic write for position recovery |

### Trading Strategy

**Signal Generation (multi-layer):**
1. **EMA Crossover** (fast=9, slow=21): Golden cross = LONG, Death cross = SHORT
2. **Ichimoku Confirmation**: Price above/below cloud + Tenkan/Kijun cross + Chikou Span
3. **Multi-Timeframe**: Primary signal on 5m, confirmation on 15m or 1H
4. **ADX Filter**: Only trade when ADX > 25 (strong trend)

All layers must confirm for entry. Entry on primary timeframe only.

**Exit Management:**
1. Trailing stop (3-phase): None → Breakeven (at ACTIVATION%) → Trail (ratcheting SL)
2. Opposite signal: EMA death cross exits LONG (golden cross exits SHORT)
3. Stop loss: Fixed SL as fallback

**Optional Trend Grid:**
- In strong trends (ADX > 35), add martingale grid orders in trend direction
- Disabled by default

### Key Patterns

- **Multi-timeframe**: Separate candle fetches for each timeframe (5m, 15m, 1H)
- **Ichimoku Cloud**: Full 5-component calculation with proper displacement
- **ADX trend filter**: Blocks signals in flat/choppy markets
- **3-phase trailing**: Progressive SL management (RSI bot pattern)
- **Code comments in Uzbek**

## Production Deployment

- **URL**: https://trend.hema.trading
- **Port**: 8090
- **BOT_ID**: trend-v1
- **Mode**: AUTOMATIC

## Configuration

Key groups via `.env`:
- **Server**: SERVER_PORT, BOT_ID, BOT_SECRET
- **EMA**: FAST_PERIOD (9), SLOW_PERIOD (21)
- **Ichimoku**: TENKAN (9), KIJUN (26), SENKOU_B (52), DISPLACEMENT (26)
- **Trend**: ADX_PERIOD (14), ADX_THRESHOLD (25), USE_ICHIMOKU (True), USE_MTF (True)
- **MTF**: PRIMARY_TF (5m), CONFIRM_TF (15m), HTF (1H)
- **Exit**: TRAILING_ACTIVATE (1.0%), TRAILING_FLOOR (0.3%), USE_OPPOSITE_SIGNAL (True)
- **Grid**: USE_TREND_GRID (False), MULTIPLIER, MAX_ORDERS
- **Risk**: MAX_LOSS_PERCENT, SL_PERCENT, MAX_DAILY_LOSS
