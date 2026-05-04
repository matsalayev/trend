"""
Trend Robot v2.1 — Stress Test (multi-period + slippage + sensitivity).

Uchta jiddiy validatsiya:

1. **Multi-period**: 6 ta turli 30-kunlik oyna ustida backtest. Agar
   strategiya faqat bitta davrda yaxshi ishlasa — overfit signal.

2. **Slippage**: Real bozarda har trade'da kichik slippage. Past-liquidity
   pair'lar (AVAX, ADA) uchun 0.05-0.10% slippage qo'shamiz.

3. **Sensitivity**: `min_net_profit_fee_factor` (1.0 vs 2.0) va
   `max_trades_per_hour` (4 vs 2) bilan natija qanday o'zgaradi.

Ishlatish:
    PYTHONIOENCODING=utf-8 python -X utf8 -m backtest.stress_test_v21
"""
import logging
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# Ko'proq pair (production'da ishlatilgan + boshqalar)
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT",
         "DOGEUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT"]

# 6 ta 30-kunlik oyna — turli market regime'larni qoplash
TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
WINDOWS = [
    (TODAY - timedelta(days=180), TODAY - timedelta(days=150)),  # ~6 oy oldingi
    (TODAY - timedelta(days=150), TODAY - timedelta(days=120)),
    (TODAY - timedelta(days=120), TODAY - timedelta(days=90)),
    (TODAY - timedelta(days=90),  TODAY - timedelta(days=60)),
    (TODAY - timedelta(days=60),  TODAY - timedelta(days=30)),
    (TODAY - timedelta(days=30),  TODAY),                        # Eng so'nggi
]

INITIAL_BALANCE = 1000.0


def _v20_config(symbol: str) -> BacktestConfig:
    """v2.0 baseline — barcha bug'lar bilan."""
    cfg = _base_config(symbol)
    cfg.trade_amount = INITIAL_BALANCE  # full balance bug
    cfg.capital_engagement = 1.0
    cfg.use_atr_sl = False
    cfg.max_signal_age_bars = 9999
    cfg.use_opposite_signal_exit = False
    cfg.min_net_profit_fee_factor = 0.0
    cfg.max_trades_per_hour = 0
    return cfg


def _v21_config(symbol: str, fee_factor: float = 1.0,
                trades_per_hour: int = 4,
                slippage_bps: float = 0.0) -> BacktestConfig:
    """v2.1 yangi — sozlanadigan parametrlar bilan."""
    cfg = _base_config(symbol)
    cfg.trade_amount = INITIAL_BALANCE * 0.15
    cfg.capital_engagement = 0.15
    cfg.use_atr_sl = True
    cfg.sl_atr_multiplier = 2.0
    cfg.max_signal_age_bars = 3
    cfg.use_opposite_signal_exit = True
    cfg.opposite_signal_requires_full_confirm = True
    cfg.min_net_profit_fee_factor = fee_factor
    cfg.max_trades_per_hour = trades_per_hour

    # Slippage'ni effective fee'ga qo'shamiz (har side uchun yarmi)
    # 5 bps slippage = 0.05% — har trade ham open ham close'da
    slip_decimal = slippage_bps / 10000.0
    cfg.taker_fee = cfg.taker_fee + slip_decimal
    return cfg


def _base_config(symbol: str) -> BacktestConfig:
    from trend_robot.config import get_preset
    p = get_preset(symbol)
    return BacktestConfig(
        symbol=symbol,
        timeframe="15m",
        htf_timeframe="4H",
        leverage=int(p.get("leverage", 10)),
        trade_amount=0,
        taker_fee=0.0006,
        funding_rate_per_8h=0.0001,
        ema_fast=int(p.get("ema_fast", 9)),
        ema_slow=int(p.get("ema_slow", 21)),
        atr_period=int(p.get("atr_period", 14)),
        adx_period=int(p.get("adx_period", 14)),
        adx_threshold=float(p.get("adx_threshold", 25.0)),
        supertrend_period=int(p.get("supertrend_period", 10)),
        supertrend_multiplier=float(p.get("supertrend_multiplier", 3.0)),
        use_htf_filter=bool(p.get("use_htf_filter", True)),
        htf_ema_fast=int(p.get("htf_ema_fast", 21)),
        htf_ema_slow=int(p.get("htf_ema_slow", 50)),
        initial_sl_percent=float(p.get("initial_sl_percent", 3.0)),
        trailing_activation_percent=float(p.get("trailing_activation_percent", 1.0)),
        trailing_atr_multiplier=float(p.get("trailing_atr_multiplier", 1.5)),
        use_partial_tp=bool(p.get("use_partial_tp", True)),
        partial_tp1_percent=float(p.get("partial_tp1_percent", 2.0)),
        partial_tp1_size_pct=float(p.get("partial_tp1_size_pct", 0.33)),
        partial_tp2_percent=float(p.get("partial_tp2_percent", 5.0)),
        partial_tp2_size_pct=float(p.get("partial_tp2_size_pct", 0.33)),
        max_drawdown_percent=float(p.get("max_drawdown_percent", 20.0)),
        cooldown_bars_after_sl=int(p.get("cooldown_bars_after_sl", 5)),
        initial_balance=INITIAL_BALANCE,
    )


def _run(cfg: BacktestConfig, start: datetime, end: datetime) -> Dict:
    htf_start = start - timedelta(days=15)  # HTF warmup
    candles = load_candles(cfg.symbol, "15m", start, end)
    htf_candles = load_candles(cfg.symbol, "4H", htf_start, end)
    if not candles:
        return {}
    bt = Backtester(cfg, candles, htf_candles=htf_candles)
    res = bt.run()
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "max_dd_pct": res.max_drawdown_percent,
        "return_pct": res.return_percent,
        "profit_factor": res.profit_factor if res.profit_factor != float('inf') else 99.99,
    }


# ──────────────────────────────────────────────────────────────────────────
# TEST 1: MULTI-PERIOD (overfit check)
# ──────────────────────────────────────────────────────────────────────────


def test_multi_period():
    print("=" * 100)
    print("TEST 1: MULTI-PERIOD (6 ta 30-kunlik oyna, overfit detector)")
    print("=" * 100)

    grand_total = {"v2.0": [0.0, 0, 0], "v2.1": [0.0, 0, 0]}  # [net, trades, wins]
    period_summary = {}

    for win_idx, (start, end) in enumerate(WINDOWS, 1):
        print(f"\n--- OYNA #{win_idx}: {start.date()} → {end.date()} ---")
        period_v20 = {"net": 0.0, "trades": 0, "wins": 0, "fees": 0.0}
        period_v21 = {"net": 0.0, "trades": 0, "wins": 0, "fees": 0.0}

        for symbol in PAIRS:
            v20 = _run(_v20_config(symbol), start, end)
            v21 = _run(_v21_config(symbol), start, end)
            if v20:
                period_v20["net"] += v20["net_pnl"]
                period_v20["fees"] += v20["fees"]
                period_v20["trades"] += v20["trades"]
                period_v20["wins"] += v20["wins"]
            if v21:
                period_v21["net"] += v21["net_pnl"]
                period_v21["fees"] += v21["fees"]
                period_v21["trades"] += v21["trades"]
                period_v21["wins"] += v21["wins"]

        wr20 = (period_v20["wins"] / period_v20["trades"] * 100) if period_v20["trades"] else 0
        wr21 = (period_v21["wins"] / period_v21["trades"] * 100) if period_v21["trades"] else 0
        net20_after = period_v20["net"] - period_v20["fees"]
        net21_after = period_v21["net"] - period_v21["fees"]

        print(f"  v2.0:  trades={period_v20['trades']:4} WR={wr20:5.1f}% "
              f"net=${period_v20['net']:+8.2f} fees=${period_v20['fees']:7.2f} "
              f"NET_AFTER=${net20_after:+8.2f}")
        print(f"  v2.1:  trades={period_v21['trades']:4} WR={wr21:5.1f}% "
              f"net=${period_v21['net']:+8.2f} fees=${period_v21['fees']:7.2f} "
              f"NET_AFTER=${net21_after:+8.2f}")
        print(f"  DELTA: ${net21_after - net20_after:+8.2f}")

        period_summary[win_idx] = (net20_after, net21_after)
        grand_total["v2.0"] = [grand_total["v2.0"][0] + net20_after,
                               grand_total["v2.0"][1] + period_v20["trades"],
                               grand_total["v2.0"][2] + period_v20["wins"]]
        grand_total["v2.1"] = [grand_total["v2.1"][0] + net21_after,
                               grand_total["v2.1"][1] + period_v21["trades"],
                               grand_total["v2.1"][2] + period_v21["wins"]]

    print("\n" + "=" * 100)
    print("MULTI-PERIOD UMUMIY (6 oyna × 9 pair = 54 backtest):")
    for v, d in grand_total.items():
        wr = (d[2] / d[1] * 100) if d[1] else 0
        print(f"  {v}: NET_AFTER_FEES=${d[0]:+9.2f}, jami {d[1]} trade, WR={wr:.1f}%")

    # Overfit check: v2.1 har bir oynada profit qildimi?
    losing_periods = [i for i, (_, n21) in period_summary.items() if n21 < 0]
    if losing_periods:
        print(f"\n  ⚠ v2.1 zarara qilgan oynalar: {losing_periods} ({len(losing_periods)}/{len(period_summary)})")
    else:
        print(f"\n  ✓ v2.1 BARCHA {len(period_summary)} oynada profit qildi")

    return period_summary


# ──────────────────────────────────────────────────────────────────────────
# TEST 2: SLIPPAGE SENSITIVITY
# ──────────────────────────────────────────────────────────────────────────


def test_slippage():
    print("\n" + "=" * 100)
    print("TEST 2: SLIPPAGE SENSITIVITY (oxirgi 60 kun, 9 pair)")
    print("=" * 100)

    start = TODAY - timedelta(days=60)
    end = TODAY

    for slip_bps in [0, 2, 5, 10]:
        period = {"net": 0.0, "trades": 0, "wins": 0, "fees": 0.0}
        for symbol in PAIRS:
            r = _run(_v21_config(symbol, slippage_bps=slip_bps), start, end)
            if r:
                period["net"] += r["net_pnl"]
                period["fees"] += r["fees"]
                period["trades"] += r["trades"]
                period["wins"] += r["wins"]
        wr = (period["wins"] / period["trades"] * 100) if period["trades"] else 0
        net_after = period["net"] - period["fees"]
        print(f"  Slippage {slip_bps:>2} bps: "
              f"trades={period['trades']:3} WR={wr:5.1f}% "
              f"net=${period['net']:+8.2f} fees=${period['fees']:7.2f} "
              f"NET_AFTER=${net_after:+8.2f}")


# ──────────────────────────────────────────────────────────────────────────
# TEST 3: PARAMETER SENSITIVITY (fee_factor & trades_per_hour)
# ──────────────────────────────────────────────────────────────────────────


def test_param_sensitivity():
    print("\n" + "=" * 100)
    print("TEST 3: PARAMETER SENSITIVITY (oxirgi 60 kun, 9 pair, 5 bps slippage)")
    print("=" * 100)

    start = TODAY - timedelta(days=60)
    end = TODAY

    print("\n  fee_factor (voluntary exit gross >= factor × fee):")
    for ff in [1.0, 1.5, 2.0, 3.0]:
        period = {"net": 0.0, "trades": 0, "wins": 0, "fees": 0.0}
        for symbol in PAIRS:
            r = _run(_v21_config(symbol, fee_factor=ff, slippage_bps=5), start, end)
            if r:
                period["net"] += r["net_pnl"]
                period["fees"] += r["fees"]
                period["trades"] += r["trades"]
                period["wins"] += r["wins"]
        wr = (period["wins"] / period["trades"] * 100) if period["trades"] else 0
        net_after = period["net"] - period["fees"]
        print(f"    factor={ff}: trades={period['trades']:3} WR={wr:5.1f}% "
              f"NET_AFTER=${net_after:+8.2f}")

    print("\n  max_trades_per_hour (loop guard):")
    for tph in [0, 2, 4, 8]:
        period = {"net": 0.0, "trades": 0, "wins": 0, "fees": 0.0}
        for symbol in PAIRS:
            r = _run(_v21_config(symbol, trades_per_hour=tph, slippage_bps=5), start, end)
            if r:
                period["net"] += r["net_pnl"]
                period["fees"] += r["fees"]
                period["trades"] += r["trades"]
                period["wins"] += r["wins"]
        wr = (period["wins"] / period["trades"] * 100) if period["trades"] else 0
        net_after = period["net"] - period["fees"]
        label = "unlimited" if tph == 0 else str(tph)
        print(f"    tph={label:>9}: trades={period['trades']:3} WR={wr:5.1f}% "
              f"NET_AFTER=${net_after:+8.2f}")


# ──────────────────────────────────────────────────────────────────────────
# TEST 4: PRODUCTION REPLAY (2026-04-23 → 05-04 — haqiqiy production davri)
# ──────────────────────────────────────────────────────────────────────────


def test_production_replay():
    print("\n" + "=" * 100)
    print("TEST 4: PRODUCTION DAVR REPLAY (2026-04-23 → 05-04)")
    print("Production: 27 trade, NET -$307.22 (AVAX 17, ADA 10)")
    print("=" * 100)

    start = datetime(2026, 4, 23, tzinfo=timezone.utc)
    end = datetime(2026, 5, 4, tzinfo=timezone.utc)

    for symbol in ["AVAXUSDT", "ADAUSDT"]:
        v20 = _run(_v20_config(symbol), start, end)
        v21_clean = _run(_v21_config(symbol, slippage_bps=0), start, end)
        v21_real = _run(_v21_config(symbol, fee_factor=2.0, slippage_bps=5), start, end)

        print(f"\n  {symbol}:")
        if v20:
            print(f"    v2.0  (eski bug):   trades={v20['trades']:3} WR={v20['winrate']:5.1f}% "
                  f"net=${v20['net_pnl']:+7.2f} fees=${v20['fees']:6.2f}")
        if v21_clean:
            print(f"    v2.1  (clean):      trades={v21_clean['trades']:3} WR={v21_clean['winrate']:5.1f}% "
                  f"net=${v21_clean['net_pnl']:+7.2f} fees=${v21_clean['fees']:6.2f}")
        if v21_real:
            print(f"    v2.1  (real cfg):   trades={v21_real['trades']:3} WR={v21_real['winrate']:5.1f}% "
                  f"net=${v21_real['net_pnl']:+7.2f} fees=${v21_real['fees']:6.2f}  "
                  f"(factor=2.0, 5bps slip)")


def main():
    t0 = time.time()
    test_multi_period()
    test_slippage()
    test_param_sensitivity()
    test_production_replay()
    print(f"\n[Total: {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
