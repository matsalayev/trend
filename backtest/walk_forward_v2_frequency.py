"""
Walk-Forward v2: Frequency-Aware Optimization.

Eski v1 muammolari:
- ADX threshold 30-38 juda baland → kam signal
- CHOP threshold 38-50 juda strict → kam signal
- 60 kunda 14 trade = 0.23/kun (juda kam)

Yangi v2 maqsadi:
- Kamida 30 trade / 60 kun (0.5+/kun)
- Profitable bo'lib qolish (test'da)
- Looser filters + tighter SL (kompensatsiya)
- 9 pair'ga qaytish (drop'larni qaytadan sinash)

Strategiya:
1. Looser ADX (20-30) va CHOP (50-70)
2. Tighter SL (sl_atr 1.0-2.0) — kichikroq zarar
3. Wider trail (1.5-3.0) — kattaroq winner
4. Score: profit + frequency bonus (kamida 6 trade/pair → bonus)
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
HTF_OFFSET = timedelta(days=20)
INITIAL_BALANCE = 1000.0

# Maqsad: per-pair test'da kamida 6 trade (0.1/kun) — portfolio jami 30+
MIN_TRADES_PER_PAIR_TEST = 6
TARGET_TRADES_PER_PAIR_TEST = 12  # ideal: 0.2/kun

# Yangi grid — ANCHA looser, frequency uchun
ADX_GRID = [20, 22, 25, 28, 30, 33]   # 25 dan past ham qo'shildi
CHOP_GRID = [50, 55, 60, 65, 70]      # 70 gacha — choppy market'da ham
SL_ATR_GRID = [1.0, 1.5, 2.0]         # tighter SL — kichikroq zarar
TRAIL_ATR_GRID = [1.5, 2.5, 3.5]      # wider trail — kattaroq winner


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
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "net_after": res.total_pnl - res.total_fees,
        "max_dd_pct": res.max_drawdown_percent,
    }


def _score(stats: Dict, period_days: int) -> float:
    """
    Frequency-aware scoring:
    - Asosiy: net profit
    - Bonus: agar trade soni maqsadga yetsa
    - Penalty: agar trade <2 (statistik ahamiyat yo'q)
    """
    if not stats or stats["trades"] < 2:
        return -1e9

    # Asosiy: net profit
    score = stats["net_after"]

    # Frequency bonus: target 0.2 trade/kun (12 trade per 60 kun TRAIN'da 24 trade)
    target_trades = period_days * 0.2
    actual = stats["trades"]
    if actual < target_trades:
        # Linear penalty: kam frequency = ko'p penalty
        deficit_ratio = (target_trades - actual) / target_trades
        score -= 30 * deficit_ratio  # max $30 penalty
    else:
        # Bonus: maqsadga yetdi
        score += min(10, (actual - target_trades))  # max $10 bonus

    return score


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
    print("WALK-FORWARD v2 — FREQUENCY-AWARE (target: 0.2+ trade/kun per pair)")
    print(f"  TRAIN: {TRAIN_START.date()} -> {TRAIN_END.date()} (120 kun)")
    print(f"  TEST:  {TEST_START.date()}  -> {TEST_END.date()}  (60 kun)")
    print(f"  Grid: 6 ADX × 5 CHOP × 3 SL × 3 Trail = 270 combos × 9 pair = 2430 backtests")
    print("=" * 130)

    print(f"\n{'Pair':<10} {'ADX':>4} {'CHOP':>5} {'SL':>4} {'Trail':>5}  ||  "
          f"{'TRAIN':>5} {'tNET':>9} {'tWR%':>5} {'t/d':>5}  ||  "
          f"{'TEST':>5} {'TEST_NET':>10} {'tWR%':>5} {'t/d':>5} {'tDD%':>5}  Status")
    print("-" * 130)

    results = {}
    test_total = 0.0
    test_trades_total = 0
    test_profitable_pairs = 0

    for symbol in PAIRS:
        opt = optimize_pair_train(symbol)
        if not opt["params"]:
            print(f"{symbol:<10} -- skip --")
            continue

        test_stats = _run(symbol, TEST_START, TEST_END, **opt["params"])
        if not test_stats:
            test_stats = {"trades": 0, "wins": 0, "winrate": 0, "net_after": 0,
                         "max_dd_pct": 0}

        train = opt["stats"]
        p = opt["params"]

        train_per_day = train["trades"] / 120
        test_per_day = test_stats["trades"] / 60

        # Status
        if test_stats["net_after"] > 0 and test_per_day >= 0.1:
            status = "✓ KEEP"
        elif test_stats["net_after"] > -30 and test_per_day >= 0.1:
            status = "≈ MARGINAL"
        elif test_per_day < 0.05:
            status = "✗ FREQ_LOW"
        else:
            status = "✗ DROP"

        print(f"{symbol:<10} {p['adx_threshold']:>4} {p['chop_max_for_entry']:>5} "
              f"{p['sl_atr_multiplier']:>4} {p['trailing_atr_multiplier']:>5}  ||  "
              f"{train['trades']:>5} ${train['net_after']:>+8.2f} "
              f"{train['winrate']:>4.1f}% {train_per_day:>4.2f}  ||  "
              f"{test_stats['trades']:>5} ${test_stats['net_after']:>+9.2f} "
              f"{test_stats['winrate']:>4.1f}% {test_per_day:>4.2f} "
              f"{test_stats['max_dd_pct']:>4.1f}%  {status}")

        results[symbol] = {
            "params": p,
            "train": train,
            "test": test_stats,
            "test_per_day": test_per_day,
            "status": status,
        }
        if "KEEP" in status or "MARGINAL" in status:
            test_total += test_stats["net_after"]
            test_trades_total += test_stats["trades"]
            if test_stats["net_after"] > 0:
                test_profitable_pairs += 1

    print("-" * 130)
    keep_pairs = [s for s, d in results.items() if "KEEP" in d["status"] or "MARGINAL" in d["status"]]
    drop_pairs = [s for s in results if s not in keep_pairs]

    print(f"\n  KEEP/MARGINAL pair'lar: {keep_pairs}")
    print(f"  DROP pair'lar:          {drop_pairs}")
    print(f"\n  TEST jami trade:       {test_trades_total}")
    print(f"  TEST trade/kun:        {test_trades_total/60:.2f}  (maqsad: 0.5+)")
    print(f"  TEST NET (keep only):  ${test_total:+.2f}")

    if keep_pairs:
        keep_capital = INITIAL_BALANCE * len(keep_pairs)
        ret_pct = test_total / keep_capital * 100
        apr = ret_pct * (365 / 60)
        print(f"  TEST RETURN:           {ret_pct:+.2f}% (60 kun)")
        print(f"  APR:                   {apr:+.2f}%")
        print(f"  Profitable pair count: {test_profitable_pairs}/{len(keep_pairs)}")

    # Save
    out = Path("backtest/walk_forward_v2_results.json")
    payload = {
        "version": "v2_frequency_aware",
        "train_period": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
        "test_period": [TEST_START.isoformat(), TEST_END.isoformat()],
        "keep_pairs": keep_pairs,
        "drop_pairs": drop_pairs,
        "test_total_trades": test_trades_total,
        "test_total_net": test_total,
        "per_pair": {
            s: {**d["params"], "train": d["train"], "test": d["test"],
                "status": d["status"]}
            for s, d in results.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  Saqlandi: {out}")


if __name__ == "__main__":
    main()
