"""
Yomon ishlagan 3 pair (BNB/SOL/ADA) uchun WIDER grid optimization.

Standart v3 walk-forward 4 parametr (ADX/CHOP/SL/Trail) varied. Bu yerda:
- Yana 3 ta param qo'shamiz: supertrend_multiplier, ema_fast/slow, htf_filter on/off
- Maqsad: 6-oy FULL davrida profitable parametr topish
- Agar topilmasa — pair drop qilinadi
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester


PAIRS = ["BNBUSDT", "SOLUSDT", "ADAUSDT"]

TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
TRAIN_START = TODAY - timedelta(days=180)
TRAIN_END = TODAY - timedelta(days=60)
TEST_START = TODAY - timedelta(days=60)
TEST_END = TODAY
FULL_START = TODAY - timedelta(days=180)
HTF_OFFSET = timedelta(days=20)
INITIAL_BALANCE = 1000.0

# Yangi WIDER grid — 7 parametr varied
ADX_GRID = [25, 30, 35, 40]          # 4
CHOP_GRID = [42, 50, 60]             # 3
SL_ATR_GRID = [1.0, 1.5]              # 2
TRAIL_ATR_GRID = [2.0, 3.0, 4.0]     # 3
SUPERTREND_MULT_GRID = [2.0, 3.0]    # 2
EMA_PAIRS = [(9, 21), (12, 26)]       # 2
HTF_FILTER = [True, False]            # 2
# Total: 4×3×2×3×2×2×2 = 576 combos per pair


def _config(symbol: str, **overrides) -> BacktestConfig:
    cfg = BacktestConfig(
        symbol=symbol, timeframe="15m", htf_timeframe="4H",
        leverage=10,
        trade_amount=150.0, capital_engagement=0.15,
        taker_fee=0.0006, funding_rate_per_8h=0.0001,
        ema_fast=9, ema_slow=21,
        atr_period=14, adx_period=14,
        adx_threshold=25.0,
        supertrend_period=10, supertrend_multiplier=2.0,
        use_htf_filter=True, htf_ema_fast=21, htf_ema_slow=50,
        initial_sl_percent=3.0,
        use_atr_sl=True, sl_atr_multiplier=2.0,
        max_signal_age_bars=3,
        trailing_activation_percent=1.0,
        trailing_atr_multiplier=1.5,
        use_partial_tp=True,
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


def optimize_pair(symbol: str) -> Dict:
    """Wider grid — 576 combos per pair, score on TRAIN."""
    best = {"score": -1e9, "params": None, "stats": None}
    train_days = (TRAIN_END - TRAIN_START).days
    combos = 0
    for adx_t in ADX_GRID:
        for chop_t in CHOP_GRID:
            for sl_atr in SL_ATR_GRID:
                for trail_atr in TRAIL_ATR_GRID:
                    for st_mult in SUPERTREND_MULT_GRID:
                        for (ef, es) in EMA_PAIRS:
                            for htf_on in HTF_FILTER:
                                combos += 1
                                stats = _run(
                                    symbol, TRAIN_START, TRAIN_END,
                                    adx_threshold=adx_t,
                                    chop_max_for_entry=chop_t,
                                    sl_atr_multiplier=sl_atr,
                                    trailing_atr_multiplier=trail_atr,
                                    supertrend_multiplier=st_mult,
                                    ema_fast=ef,
                                    ema_slow=es,
                                    use_htf_filter=htf_on,
                                )
                                if not stats or stats["trades"] < 2:
                                    continue
                                # Score: profit + small frequency bonus
                                score = stats["net_after"]
                                per_day = stats["trades"] / train_days
                                if per_day < 0.1:
                                    score -= 20 * (0.1 - per_day) / 0.1
                                if score > best["score"]:
                                    best = {
                                        "score": score,
                                        "params": {
                                            "adx_threshold": adx_t,
                                            "chop_max_for_entry": chop_t,
                                            "sl_atr_multiplier": sl_atr,
                                            "trailing_atr_multiplier": trail_atr,
                                            "supertrend_multiplier": st_mult,
                                            "ema_fast": ef,
                                            "ema_slow": es,
                                            "use_htf_filter": htf_on,
                                        },
                                        "stats": stats,
                                    }
    best["combos_tested"] = combos
    return best


def main():
    print("=" * 130)
    print("OPTIMIZATION YOMON PAIR'LAR UCHUN — wider grid (576 combos × 3 pair = 1728 backtests)")
    print("=" * 130)
    print(f"\n{'Pair':<10} {'ADX':>4} {'CHOP':>5} {'SL':>4} {'Trail':>5} {'ST':>4} {'EMA':>7} {'HTF':>4} || "
          f"{'TRAIN':>5} {'tNET':>9} || {'TEST':>5} {'tNET':>9} {'tWR':>5} {'tDD':>5} || "
          f"{'FULL':>5} {'fNET':>9} {'fWR':>5} {'Status'}")
    print("-" * 130)

    results = {}
    for symbol in PAIRS:
        opt = optimize_pair(symbol)
        if not opt["params"]:
            print(f"{symbol:<10} -- yetarli signal yo'q --")
            continue

        test_stats = _run(symbol, TEST_START, TEST_END, **opt["params"])
        full_stats = _run(symbol, FULL_START, TEST_END, **opt["params"])

        if not test_stats:
            test_stats = {"trades": 0, "winrate": 0, "net_after": 0, "max_dd_pct": 0}
        if not full_stats:
            full_stats = {"trades": 0, "winrate": 0, "net_after": 0, "max_dd_pct": 0}

        train = opt["stats"]
        p = opt["params"]
        full_profitable = full_stats["net_after"] > 0
        test_marginal = test_stats["net_after"] > -30

        if full_profitable and test_marginal:
            status = "✓ KEEP"
        elif full_profitable:
            status = "≈ MARGINAL"
        else:
            status = "✗ DROP"

        ema_str = f"{p['ema_fast']}/{p['ema_slow']}"
        htf_str = "Y" if p['use_htf_filter'] else "N"
        print(f"{symbol:<10} {p['adx_threshold']:>4} {p['chop_max_for_entry']:>5} "
              f"{p['sl_atr_multiplier']:>4} {p['trailing_atr_multiplier']:>5} "
              f"{p['supertrend_multiplier']:>4} {ema_str:>7} {htf_str:>4} || "
              f"{train['trades']:>5} ${train['net_after']:>+8.2f} || "
              f"{test_stats['trades']:>5} ${test_stats['net_after']:>+8.2f} "
              f"{test_stats['winrate']:>4.1f}% {test_stats['max_dd_pct']:>4.1f}% || "
              f"{full_stats['trades']:>5} ${full_stats['net_after']:>+8.2f} "
              f"{full_stats['winrate']:>4.1f}%  {status}")

        results[symbol] = {
            "params": p,
            "train": train,
            "test": test_stats,
            "full": full_stats,
            "status": status,
        }

    out = Path("backtest/optimize_bad_pairs_results.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaqlandi: {out}")

    # Summary
    keep = [s for s, d in results.items() if "KEEP" in d["status"] or "MARGINAL" in d["status"]]
    drop = [s for s in results if s not in keep]
    print(f"\n  KEEP: {keep}")
    print(f"  DROP: {drop}")


if __name__ == "__main__":
    main()
