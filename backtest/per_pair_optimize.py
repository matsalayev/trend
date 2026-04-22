"""Per-pair grid search for Smart Trend Following strategy."""

import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

from backtest.data_loader import load_candles
from backtest.engine import Backtester, BacktestConfig, BacktestResult

logging.basicConfig(level=logging.WARNING)

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "ADAUSDT",
]

PARAM_GRID = {
    "ema_fast": [9, 12],
    "ema_slow": [21, 26],
    "adx_threshold": [18, 22, 28],
    "supertrend_multiplier": [2.0, 3.0, 4.0],
    "trailing_atr_multiplier": [1.0, 1.5, 2.5],
    "initial_sl_percent": [2.0, 3.0, 4.0],
    "use_htf_filter": [True, False],
    "use_partial_tp": [True, False],
    "leverage": [5, 10],
}


def score(r: BacktestResult) -> float:
    if r.total_trades == 0:
        return -1000
    s = r.return_percent
    s -= r.max_drawdown_percent * 0.3
    if r.profit_factor != float('inf'):
        s += r.profit_factor * 5
    s += r.winrate * 0.1
    return s


def optimize_pair(symbol: str, candles, htf_candles, base: BacktestConfig):
    combos = list(product(*PARAM_GRID.values()))
    best_s, best_r, best_c = float('-inf'), None, None
    keys = list(PARAM_GRID.keys())

    for combo in combos:
        overrides = dict(zip(keys, combo))
        cfg = replace(base, symbol=symbol, **overrides)
        r = Backtester(cfg, candles, htf_candles).run()
        s = score(r)
        if s > best_s:
            best_s, best_r, best_c = s, r, cfg

    return {
        "symbol": symbol,
        "score": best_s,
        "return_percent": best_r.return_percent,
        "winrate": best_r.winrate,
        "profit_factor": best_r.profit_factor if best_r.profit_factor != float('inf') else 99.99,
        "max_drawdown_percent": best_r.max_drawdown_percent,
        "total_trades": best_r.total_trades,
        "config": {
            "leverage": best_c.leverage,
            "ema_fast": best_c.ema_fast,
            "ema_slow": best_c.ema_slow,
            "adx_threshold": best_c.adx_threshold,
            "adx_period": best_c.adx_period,
            "atr_period": best_c.atr_period,
            "supertrend_period": best_c.supertrend_period,
            "supertrend_multiplier": best_c.supertrend_multiplier,
            "trailing_atr_multiplier": best_c.trailing_atr_multiplier,
            "trailing_activation_percent": best_c.trailing_activation_percent,
            "initial_sl_percent": best_c.initial_sl_percent,
            "use_htf_filter": best_c.use_htf_filter,
            "htf_ema_fast": best_c.htf_ema_fast,
            "htf_ema_slow": best_c.htf_ema_slow,
            "use_partial_tp": best_c.use_partial_tp,
            "partial_tp1_percent": best_c.partial_tp1_percent,
            "partial_tp1_size_pct": best_c.partial_tp1_size_pct,
            "partial_tp2_percent": best_c.partial_tp2_percent,
            "partial_tp2_size_pct": best_c.partial_tp2_size_pct,
            "max_drawdown_percent": best_c.max_drawdown_percent,
            "cooldown_bars_after_sl": best_c.cooldown_bars_after_sl,
        },
    }


def main():
    base = BacktestConfig(timeframe="15m", trade_amount=1000)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)

    results = {}
    combos_count = 1
    for v in PARAM_GRID.values():
        combos_count *= len(v)
    print(f"Testing {combos_count} combos × {len(PAIRS)} pairs")
    print("=" * 90)

    for sym in PAIRS:
        print(f"\n[{sym}] Loading 15m...")
        candles = load_candles(sym, "15m", start, end)
        if not candles:
            print(f"  SKIP")
            continue
        print(f"  Loading 4H (HTF)...")
        htf_start = end - timedelta(days=60)  # 60 days = 360 4H candles (enough for EMA200)
        htf_candles = load_candles(sym, "4H", htf_start, end)
        print(f"  15m: {len(candles)}, 4H: {len(htf_candles)}. Optimizing...")

        r = optimize_pair(sym, candles, htf_candles, base)
        results[sym] = r
        c = r['config']
        print(f"  BEST: return={r['return_percent']:+.2f}%  "
              f"WR={r['winrate']:.1f}%  PF={r['profit_factor']:.2f}  "
              f"DD={r['max_drawdown_percent']:.1f}%  trades={r['total_trades']}")
        print(f"  PARAMS: EMA {c['ema_fast']}/{c['ema_slow']}  "
              f"ADX>{c['adx_threshold']}  ST×{c['supertrend_multiplier']}  "
              f"Trail×{c['trailing_atr_multiplier']}  SL={c['initial_sl_percent']}%  "
              f"HTF={c['use_htf_filter']}  PTP={c['use_partial_tp']}  Lev={c['leverage']}")

    out = Path(__file__).parent / "presets.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    # Summary
    print()
    print("=" * 95)
    print(f"{'Pair':<10}{'Return':<11}{'WR':<8}{'PF':<6}{'DD':<7}{'Trades':<8}"
          f"{'EMA':<8}{'ADX':<5}{'ST':<4}{'Trail':<6}{'SL':<5}{'HTF':<5}{'PTP':<5}{'Lev':<4}")
    print("-" * 95)
    for sym, r in sorted(results.items(), key=lambda x: -x[1]['return_percent']):
        c = r['config']
        print(f"{sym:<10}"
              f"{r['return_percent']:+8.2f}%  "
              f"{r['winrate']:5.1f}%  "
              f"{r['profit_factor']:4.2f}  "
              f"{r['max_drawdown_percent']:5.1f}%  "
              f"{r['total_trades']:<8}"
              f"{c['ema_fast']}/{c['ema_slow']:<5}"
              f"{c['adx_threshold']:<5}"
              f"{c['supertrend_multiplier']:<4}"
              f"{c['trailing_atr_multiplier']:<6}"
              f"{c['initial_sl_percent']:<5}"
              f"{'Y' if c['use_htf_filter'] else 'N':<5}"
              f"{'Y' if c['use_partial_tp'] else 'N':<5}"
              f"{c['leverage']:<4}")
    print("=" * 95)


if __name__ == "__main__":
    main()
