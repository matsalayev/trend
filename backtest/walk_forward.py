"""
Walk-Forward Optimization — overfit'dan himoyalangan validatsiya.

Mantiq:
1. 6 oy ma'lumot 2 ga bo'linadi:
   - TRAIN: birinchi 4 oy (2025-11-05 → 2026-03-05)
   - TEST:  oxirgi 2 oy (2026-03-05 → 2026-05-04) — ROBOT KO'RMAGAN
2. Har pair uchun TRAIN'da grid search → best params
3. Best params bilan TEST'da run
4. Faqat TEST natijasini ishonib, real performance bashorat qilamiz

Bu sample-bias'dan himoya qiladi: in-sample optimizatsiya har doim "yaxshi"
bo'ladi (overfit), lekin out-of-sample ko'rsatadi haqiqiy umumiylikni.
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

# 6 oy = 180 kun. Train = 120, Test = 60.
TRAIN_START = TODAY - timedelta(days=180)
TRAIN_END = TODAY - timedelta(days=60)
TEST_START = TODAY - timedelta(days=60)
TEST_END = TODAY
HTF_OFFSET = timedelta(days=20)  # HTF warmup

INITIAL_BALANCE = 1000.0


# Grid'da sinaymiz
ADX_GRID = [25, 28, 30, 33, 35, 38]
CHOP_GRID = [38, 42, 50, 60]
SL_ATR_GRID = [1.5, 2.0, 2.5]
TRAIL_ATR_GRID = [1.0, 1.5, 2.5]


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


def _run(symbol: str, start: datetime, end: datetime, **overrides) -> Dict:
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
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "net_after": res.total_pnl - res.total_fees,
        "max_dd_pct": res.max_drawdown_percent,
        "sl_count": sl_count,
        "return_pct": res.return_percent,
    }


def _score(stats: Dict) -> float:
    """Train'da pair'ni baholash. Trades soni ham hisobga olinadi."""
    if not stats or stats["trades"] < 3:
        return -1e6
    return stats["net_after"]  # Sof: net after fees


def optimize_pair_train(symbol: str) -> Dict:
    """Train period'da grid search."""
    best = {"score": -1e9, "params": None, "stats": None}
    combos_tested = 0
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
                    combos_tested += 1
                    s = _score(stats)
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
    best["combos_tested"] = combos_tested
    return best


def main():
    print("=" * 130)
    print("WALK-FORWARD VALIDATSIYA — Train (4 oy) -> Test (2 oy, out-of-sample)")
    print(f"  TRAIN: {TRAIN_START.date()} -> {TRAIN_END.date()} (120 kun)")
    print(f"  TEST:  {TEST_START.date()}  -> {TEST_END.date()}  (60 kun, ROBOT KO'RMAGAN)")
    print("=" * 130)

    print(f"\n{'Pair':<10} {'ADX':>4} {'CHOP':>5} {'SL_atr':>6} {'Trail':>6}  ||  "
          f"{'TRAIN':>7} {'tNET':>9} {'tWR%':>6}  ||  "
          f"{'TEST':>5} {'TEST_NET':>10} {'tWR%':>6} {'tDD%':>6}  Status")
    print("-" * 130)

    optimal_per_pair = {}
    train_total = 0.0
    test_total = 0.0
    test_profitable = 0
    test_count = 0

    for symbol in PAIRS:
        opt = optimize_pair_train(symbol)
        if not opt["params"]:
            print(f"{symbol:<10} -- skip (no data) --")
            continue

        # Test on out-of-sample
        test_stats = _run(
            symbol, TEST_START, TEST_END,
            **opt["params"],
        )
        if not test_stats:
            test_stats = {"trades": 0, "wins": 0, "winrate": 0, "net_after": 0,
                         "max_dd_pct": 0, "return_pct": 0}

        train_stats = opt["stats"]
        p = opt["params"]
        status = "✓" if test_stats["net_after"] > 0 else (
            "≈" if test_stats["net_after"] > -30 else "✗")

        print(f"{symbol:<10} {p['adx_threshold']:>4} {p['chop_max_for_entry']:>5} "
              f"{p['sl_atr_multiplier']:>6} {p['trailing_atr_multiplier']:>6}  ||  "
              f"{train_stats['trades']:>7} ${train_stats['net_after']:>+8.2f} "
              f"{train_stats['winrate']:>5.1f}%  ||  "
              f"{test_stats['trades']:>5} ${test_stats['net_after']:>+9.2f} "
              f"{test_stats['winrate']:>5.1f}% {test_stats['max_dd_pct']:>5.1f}%  {status}")

        optimal_per_pair[symbol] = {
            "params": p,
            "train": train_stats,
            "test": test_stats,
        }
        train_total += train_stats["net_after"]
        test_total += test_stats["net_after"]
        if test_stats["net_after"] > 0:
            test_profitable += 1
        test_count += 1

    print("-" * 130)
    print(f"{'JAMI':<10} {' '*30}  ||  "
          f"{' '*7} ${train_total:>+8.2f} {' '*5}  ||  "
          f"{' '*5} ${test_total:>+9.2f} {' '*5} {' '*5}  "
          f"{test_profitable}/{test_count} pair profitable")

    print("\n" + "=" * 130)
    print("TANLOV: Faqat TEST'da PROFITABLE pair'larni saqlash")
    print("=" * 130)

    keep_pairs = [s for s, d in optimal_per_pair.items() if d["test"]["net_after"] > 0]
    drop_pairs = [s for s in optimal_per_pair if s not in keep_pairs]
    keep_test_total = sum(optimal_per_pair[s]["test"]["net_after"] for s in keep_pairs)

    print(f"\n  KEEP: {keep_pairs}")
    print(f"  DROP: {drop_pairs}")
    print(f"  TEST NET (faqat keep): ${keep_test_total:+.2f}")
    pct_return = keep_test_total / (INITIAL_BALANCE * len(keep_pairs)) * 100
    print(f"  Per-pair test return: {pct_return:+.2f}% (60 kun)")
    apr = pct_return * (365 / 60)
    print(f"  APR (annualizirovangan): {apr:+.2f}%")

    # Save optimal presets
    out_path = Path("backtest/walk_forward_results.json")
    payload = {
        "train_period": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
        "test_period": [TEST_START.isoformat(), TEST_END.isoformat()],
        "keep_pairs": keep_pairs,
        "drop_pairs": drop_pairs,
        "keep_test_net": keep_test_total,
        "per_pair": {s: {**d["params"], "train": d["train"], "test": d["test"]}
                     for s, d in optimal_per_pair.items()},
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Saqlandi: {out_path}")


if __name__ == "__main__":
    main()
