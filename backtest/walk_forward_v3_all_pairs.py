"""
Walk-Forward v3 — barcha 9 pair uchun INCLUSIVE optimizatsiya.

User talab: har 9 pair'ni o'z optimal parametri bilan ishlash kerak.
Hatto profitable bo'lmasa ham, eng yaxshi mumkin bo'lgan parametr bilan.

Strategiya:
1. TRAIN 4 oy (Nov 5 → Mar 5)
2. TEST 2 oy (Mar 5 → May 4) — out-of-sample
3. Score: profit (asosiy) + frequency bonus + drawdown penalty
4. ALL 9 pair uchun ENABLED (eng yaxshi possible params)
5. UI'da TEST + FULL 6mo statistika ham

Goal: foydalanuvchi tanlovi maksimal — har kim o'z pair'ini tanlab ishlatsa bo'lsin.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester


PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
         "AVAXUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]

TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
TRAIN_START = TODAY - timedelta(days=180)
TRAIN_END = TODAY - timedelta(days=60)
TEST_START = TODAY - timedelta(days=60)
TEST_END = TODAY
FULL_START = TODAY - timedelta(days=180)
HTF_OFFSET = timedelta(days=20)
INITIAL_BALANCE = 1000.0

# v3 grid — yangi yo'lda yana keng (har pair uchun individual)
ADX_GRID = [20, 25, 28, 30, 33, 35]
CHOP_GRID = [42, 50, 55, 60, 65, 70]
SL_ATR_GRID = [1.0, 1.5, 2.0]
TRAIL_ATR_GRID = [1.5, 2.5, 3.5]


def _config(symbol: str, **overrides) -> BacktestConfig:
    from trend_robot.config import get_preset
    p = get_preset(symbol)
    cfg = BacktestConfig(
        symbol=symbol, timeframe="15m", htf_timeframe="4H",
        leverage=int(p.get("leverage", 10)),
        trade_amount=150.0, capital_engagement=0.15,
        taker_fee=0.0006, funding_rate_per_8h=0.0001,
        ema_fast=int(p.get("ema_fast", 9)),
        ema_slow=int(p.get("ema_slow", 21)),
        atr_period=14, adx_period=14,
        adx_threshold=25.0,
        supertrend_period=10,
        supertrend_multiplier=float(p.get("supertrend_multiplier", 3.0)),
        use_htf_filter=bool(p.get("use_htf_filter", True)),
        htf_ema_fast=21, htf_ema_slow=50,
        initial_sl_percent=float(p.get("initial_sl_percent", 3.0)),
        use_atr_sl=True, sl_atr_multiplier=2.0,
        max_signal_age_bars=3,
        trailing_activation_percent=1.0,
        trailing_atr_multiplier=1.5,
        use_partial_tp=bool(p.get("use_partial_tp", True)),
        partial_tp1_percent=2.0, partial_tp1_size_pct=0.33,
        partial_tp2_percent=5.0, partial_tp2_size_pct=0.33,
        use_opposite_signal_exit=True,
        opposite_signal_requires_full_confirm=True,
        min_net_profit_fee_factor=1.0,
        max_drawdown_percent=20.0,
        cooldown_bars_after_sl=5,
        max_trades_per_hour=4,
        use_choppiness_filter=True,
        chop_max_for_entry=50.0,
        consecutive_losses_threshold=3,
        consecutive_losses_cooldown_bars=20,
        initial_balance=INITIAL_BALANCE,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _run(symbol: str, start, end, **overrides) -> Dict:
    candles = load_candles(symbol, "15m", start, end)
    htf = load_candles(symbol, "4H", start - HTF_OFFSET, end)
    if not candles or len(candles) < 200:
        return {}
    cfg = _config(symbol, **overrides)
    bt = Backtester(cfg, candles, htf_candles=htf)
    res = bt.run()
    sl_count = sum(1 for t in res.trades if t.reason == "SL")
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "net_after": res.total_pnl - res.total_fees,
        "max_dd_pct": res.max_drawdown_percent,
        "sl_count": sl_count,
    }


def _score(stats: Dict, period_days: int) -> float:
    """v3 score: profit (asosiy) + frequency bonus + DD penalty."""
    if not stats or stats["trades"] < 2:
        return -1e9
    net = stats["net_after"]
    # Frequency bonus: max +20 if 0.3+ trade/day
    target_per_day = 0.2
    actual_per_day = stats["trades"] / period_days
    if actual_per_day < target_per_day:
        net -= 20 * (target_per_day - actual_per_day) / target_per_day
    else:
        net += min(15, (actual_per_day - target_per_day) * 50)
    # DD penalty: -5 per 10% DD over 20%
    if stats["max_dd_pct"] > 20:
        net -= (stats["max_dd_pct"] - 20) * 0.5
    return net


def optimize_pair_train(symbol: str) -> Dict:
    best = {"score": -1e9, "params": None, "stats": None}
    train_days = (TRAIN_END - TRAIN_START).days
    for adx_t in ADX_GRID:
        for chop_t in CHOP_GRID:
            for sl_atr in SL_ATR_GRID:
                for trail_atr in TRAIL_ATR_GRID:
                    stats = _run(
                        symbol, TRAIN_START, TRAIN_END,
                        adx_threshold=adx_t,
                        chop_max_for_entry=chop_t,
                        sl_atr_multiplier=sl_atr,
                        trailing_atr_multiplier=trail_atr,
                    )
                    s = _score(stats, train_days)
                    if s > best["score"]:
                        best = {
                            "score": s,
                            "params": {
                                "adx_threshold": adx_t,
                                "chop_max_for_entry": chop_t,
                                "sl_atr_multiplier": sl_atr,
                                "trailing_atr_multiplier": trail_atr,
                            },
                            "stats": stats,
                        }
    return best


def main():
    print("=" * 130)
    print("WALK-FORWARD v3 — BARCHA 9 PAIR (har biri o'z optimal parametri bilan)")
    print(f"  TRAIN: {TRAIN_START.date()} -> {TRAIN_END.date()} (120 kun)")
    print(f"  TEST:  {TEST_START.date()}  -> {TEST_END.date()}  (60 kun)")
    print(f"  Grid: 6 ADX × 6 CHOP × 3 SL × 3 Trail = 324 combos × 9 pair = 2916 backtests")
    print("=" * 130)

    print(f"\n{'Pair':<10} {'ADX':>4} {'CHOP':>5} {'SL':>4} {'Trail':>5}  ||  "
          f"{'TRAIN':>5} {'tNET':>9} {'tWR%':>5} {'t/d':>5}  ||  "
          f"{'TEST':>5} {'TEST_NET':>10} {'tWR%':>5} {'t/d':>5} {'tDD%':>5}  ||  "
          f"{'FULL':>5} {'FULL_NET':>10} {'fWR%':>5} {'fDD%':>5}")
    print("-" * 130)

    results = {}
    for symbol in PAIRS:
        opt = optimize_pair_train(symbol)
        if not opt["params"]:
            print(f"{symbol:<10} -- skip --")
            continue

        # Test on out-of-sample
        test_stats = _run(symbol, TEST_START, TEST_END, **opt["params"])
        # Full 6-month
        full_stats = _run(symbol, FULL_START, TEST_END, **opt["params"])

        if not test_stats:
            test_stats = {"trades": 0, "wins": 0, "winrate": 0, "net_after": 0,
                         "max_dd_pct": 0}
        if not full_stats:
            full_stats = {"trades": 0, "winrate": 0, "net_after": 0, "max_dd_pct": 0}

        train = opt["stats"]
        p = opt["params"]
        train_per_day = train["trades"] / 120
        test_per_day = test_stats["trades"] / 60

        print(f"{symbol:<10} {p['adx_threshold']:>4} {p['chop_max_for_entry']:>5} "
              f"{p['sl_atr_multiplier']:>4} {p['trailing_atr_multiplier']:>5}  ||  "
              f"{train['trades']:>5} ${train['net_after']:>+8.2f} "
              f"{train['winrate']:>4.1f}% {train_per_day:>4.2f}  ||  "
              f"{test_stats['trades']:>5} ${test_stats['net_after']:>+9.2f} "
              f"{test_stats['winrate']:>4.1f}% {test_per_day:>4.2f} "
              f"{test_stats['max_dd_pct']:>4.1f}%  ||  "
              f"{full_stats['trades']:>5} ${full_stats['net_after']:>+9.2f} "
              f"{full_stats['winrate']:>4.1f}% {full_stats['max_dd_pct']:>4.1f}%")

        results[symbol] = {
            "params": p,
            "train": train,
            "test": test_stats,
            "full": full_stats,
        }

    # Save
    out = Path("backtest/walk_forward_v3_all_pairs_results.json")
    payload = {
        "version": "v3_all_pairs",
        "train_period": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
        "test_period": [TEST_START.isoformat(), TEST_END.isoformat()],
        "full_period": [FULL_START.isoformat(), TEST_END.isoformat()],
        "all_pairs_enabled": True,
        "per_pair": {s: {**d["params"], "train": d["train"], "test": d["test"], "full": d["full"]}
                     for s, d in results.items()},
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaqlandi: {out}")

    # Summary
    print("\n" + "=" * 130)
    print("YAKUNIY SUMMARY (TEST out-of-sample, 60 kun):")
    print("-" * 130)
    test_total = sum(d["test"]["net_after"] for d in results.values())
    test_trades = sum(d["test"]["trades"] for d in results.values())
    test_profitable = sum(1 for d in results.values() if d["test"]["net_after"] > 0)
    print(f"  TEST jami: {test_trades} trade ({test_trades/60:.2f}/kun), "
          f"NET ${test_total:+.2f} ({test_profitable}/{len(results)} profitable)")
    full_total = sum(d["full"]["net_after"] for d in results.values())
    full_trades = sum(d["full"]["trades"] for d in results.values())
    print(f"  FULL 6oy: {full_trades} trade ({full_trades/180:.2f}/kun), "
          f"NET ${full_total:+.2f}")


if __name__ == "__main__":
    main()
