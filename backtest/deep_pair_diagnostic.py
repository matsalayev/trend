"""
Per-pair chuqur diagnostika — har pair'ga nima to'sqinlik qilayapti?

Har pair uchun 6 oyda hisoblaymiz:
- Avg ADX (overall vs at trade entries)
- Avg CHOP (overall vs at trade entries)
- Avg ATR % (volatility)
- Trend periods % (kuni nechta candle ADX>30)
- Best vs worst trade analizi
- Optimal ADX threshold qidirish (per-pair)
- Optimal CHOP threshold qidirish
- Optimal SL multiplier
- Pair'ning trend "natural" qobiliyati: long-term EMA slope

Maqsad: har pair'ga bir xil universal parametr o'rniga shaxsiy tuning topish.
"""
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Tuple

from backtest.data_loader import load_candles
from backtest.engine import (
    BacktestConfig, Backtester, calc_adx, calc_atr, calc_chop, calc_ema,
)


PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
         "AVAXUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]

TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
START = TODAY - timedelta(days=180)
HTF_START = TODAY - timedelta(days=200)


def market_profile(symbol: str) -> Dict:
    """Pair'ning 6 oylik market profilini hisoblash."""
    candles = load_candles(symbol, "15m", START, TODAY)
    if not candles:
        return {}

    adxs = calc_adx(candles, 14)
    atrs = calc_atr(candles, 14)
    chops = calc_chop(candles, 14)
    ema_fast = calc_ema(candles, 9)
    ema_slow = calc_ema(candles, 21)
    ema_long = calc_ema(candles, 200)

    # Filter to valid (post-warmup)
    valid_adx = [a for a in adxs[50:] if a > 0]
    valid_atr_pct = [
        atrs[i] / candles[i].close * 100
        for i in range(50, len(candles))
        if atrs[i] > 0 and candles[i].close > 0
    ]
    valid_chop = [c for c in chops[50:] if 1 < c < 99]

    # Trend regime stats
    trending_30 = sum(1 for a in valid_adx if a > 30) / len(valid_adx) * 100
    trending_35 = sum(1 for a in valid_adx if a > 35) / len(valid_adx) * 100
    chop_low = sum(1 for c in valid_chop if c < 38) / len(valid_chop) * 100
    chop_low_60 = sum(1 for c in valid_chop if c < 60) / len(valid_chop) * 100
    sweet_spot = sum(
        1 for a, c in zip(valid_adx, valid_chop) if a > 30 and c < 45
    ) / min(len(valid_adx), len(valid_chop)) * 100

    # Long-term EMA slope (200 EMA over 6 months) — trend "natural-ness" indikatori
    if len(ema_long) > 1000 and ema_long[1000] > 0 and ema_long[-1] > 0:
        ema_change_pct = (ema_long[-1] - ema_long[1000]) / ema_long[1000] * 100
    else:
        ema_change_pct = 0

    # Price range
    closes = [c.close for c in candles]
    price_range_pct = (max(closes) - min(closes)) / closes[0] * 100

    return {
        "symbol": symbol,
        "candles": len(candles),
        "avg_adx": sum(valid_adx) / len(valid_adx) if valid_adx else 0,
        "avg_atr_pct": sum(valid_atr_pct) / len(valid_atr_pct) if valid_atr_pct else 0,
        "avg_chop": sum(valid_chop) / len(valid_chop) if valid_chop else 0,
        "trending_30_pct": trending_30,
        "trending_35_pct": trending_35,
        "chop_low_pct": chop_low,
        "chop_low_60_pct": chop_low_60,
        "sweet_spot_pct": sweet_spot,  # ADX>30 + CHOP<45 (entry uchun ideal)
        "ema_long_change_pct": ema_change_pct,
        "price_range_pct": price_range_pct,
    }


def trade_analysis(symbol: str, **overrides) -> Dict:
    """Pair'ni backtest qilib, trade-level statistika."""
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
        adx_threshold=float(p.get("adx_threshold", 25.0)),
        supertrend_period=10,
        supertrend_multiplier=float(p.get("supertrend_multiplier", 3.0)),
        use_htf_filter=bool(p.get("use_htf_filter", True)),
        htf_ema_fast=21, htf_ema_slow=50,
        initial_sl_percent=float(p.get("initial_sl_percent", 3.0)),
        use_atr_sl=True, sl_atr_multiplier=2.0,
        max_signal_age_bars=3,
        trailing_activation_percent=1.0,
        trailing_atr_multiplier=float(p.get("trailing_atr_multiplier", 1.5)),
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
        initial_balance=1000.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)

    candles = load_candles(symbol, "15m", START, TODAY)
    htf = load_candles(symbol, "4H", HTF_START, TODAY)
    if not candles:
        return {}

    bt = Backtester(cfg, candles, htf)
    res = bt.run()

    wins = [t for t in res.trades if t.net_pnl > 0]
    losses = [t for t in res.trades if t.net_pnl < 0]

    return {
        "trades": res.total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "max_dd_pct": res.max_drawdown_percent,
        "avg_win": sum(t.net_pnl for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t.net_pnl for t in losses) / len(losses) if losses else 0,
        "best_trade": max((t.net_pnl for t in res.trades), default=0),
        "worst_trade": min((t.net_pnl for t in res.trades), default=0),
        "config": cfg,
    }


def find_optimal_per_pair(symbol: str) -> Dict:
    """Per-pair: ADX threshold va CHOP max ni grid search."""
    profile = market_profile(symbol)
    if not profile:
        return {}

    best = {"net_after": -1e9, "params": None, "stats": None}

    # Mini grid (tez)
    adx_options = [25, 28, 30, 33, 35, 38]
    chop_options = [38, 42, 50, 60]

    for adx_t in adx_options:
        for chop_t in chop_options:
            stats = trade_analysis(
                symbol,
                adx_threshold=adx_t,
                chop_max_for_entry=chop_t,
            )
            if not stats:
                continue
            net_after = stats["net_pnl"] - stats["fees"]
            # Score: prioritize positive net + reasonable trade count
            score = net_after - max(0, 5 - stats["trades"]) * 50  # penalize <5 trades
            if score > best["net_after"]:
                best = {
                    "net_after": net_after,
                    "score": score,
                    "params": {"adx_threshold": adx_t, "chop_max": chop_t},
                    "stats": stats,
                }

    return {"profile": profile, "best": best}


def main():
    print("=" * 130)
    print("DEEP PER-PAIR DIAGNOSTIKA + Per-pair optimal parametrlarni qidirish")
    print(f"Davr: {START.date()} -> {TODAY.date()} (180 kun, 6 oy)")
    print("=" * 130)

    print(f"\n{'PAIR':<10} {'avgADX':>7} {'avgATR%':>8} {'avgCHOP':>8} "
          f"{'Trnd30%':>8} {'Trnd35%':>8} {'Sweet%':>7} {'EMAchg%':>8} {'Range%':>8}")
    print("-" * 130)

    profiles = {}
    for symbol in PAIRS:
        p = market_profile(symbol)
        profiles[symbol] = p
        if not p:
            continue
        print(f"{symbol:<10} "
              f"{p['avg_adx']:>6.1f}  "
              f"{p['avg_atr_pct']:>7.2f}% "
              f"{p['avg_chop']:>7.1f}  "
              f"{p['trending_30_pct']:>7.1f}% "
              f"{p['trending_35_pct']:>7.1f}% "
              f"{p['sweet_spot_pct']:>6.1f}% "
              f"{p['ema_long_change_pct']:>+7.1f}% "
              f"{p['price_range_pct']:>7.1f}%")

    print("\n" + "=" * 130)
    print("PER-PAIR OPTIMAL PARAMETRLARNI QIDIRISH (grid search 24 combo per pair)")
    print("=" * 130)

    results = {}
    for symbol in PAIRS:
        print(f"\n>>> {symbol}")
        opt = find_optimal_per_pair(symbol)
        if not opt:
            continue
        results[symbol] = opt
        b = opt["best"]
        s = b["stats"]
        p = b["params"]
        print(f"  Best params: ADX>{p['adx_threshold']:>4}  CHOP<{p['chop_max']:>4}")
        print(f"  Result:  trades={s['trades']:>3}  WR={s['winrate']:>5.1f}%  "
              f"net_after=${b['net_after']:>+8.2f}  "
              f"avg_win=${s['avg_win']:>+6.2f}  avg_loss=${s['avg_loss']:>+6.2f}  "
              f"DD={s['max_dd_pct']:>5.1f}%")
        print(f"  Best/Worst trade: ${s['best_trade']:>+7.2f} / ${s['worst_trade']:>+7.2f}")

    print("\n" + "=" * 130)
    print("YAKUNIY: HAR PAIR'NING OPTIMAL ADX/CHOP")
    print("=" * 130)
    print(f"\n{'Pair':<10} {'ADX_threshold':>14} {'CHOP_max':>10} "
          f"{'Trades':>7} {'WR':>6} {'NET':>9} {'Status'}")
    print("-" * 100)
    for symbol, r in results.items():
        b = r["best"]
        s = b["stats"]
        p = b["params"]
        net_after = b["net_after"]
        status = "✓" if net_after > 0 else ("⚠" if net_after > -50 else "✗")
        print(f"{symbol:<10} {p['adx_threshold']:>14} {p['chop_max']:>10} "
              f"{s['trades']:>7} {s['winrate']:>5.1f}% ${net_after:>+8.2f}  {status}")


if __name__ == "__main__":
    main()
