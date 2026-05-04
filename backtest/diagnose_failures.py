"""
Trend v2.1 — failure diagnostikasi.

Maqsad: Stress test'da 5/6 oynada zarar bo'ldi. Sababini aniqlash uchun:
1. Har exit reason'dan qancha PnL keladi (SL vs TRAIL vs PARTIAL_TP vs OPPOSITE)
2. Average ADX, ATR pair va davr bo'yicha
3. Per-pair PnL har oynada — qaysi pair toza yo'qotyapti
4. Trade-level analiz — qancha trade'lar SL'da yopilmoqda

Ishlatish:
    PYTHONIOENCODING=utf-8 python -X utf8 -m backtest.diagnose_failures
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester


PAIRS = ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "ADAUSDT",
         "SOLUSDT", "DOGEUSDT", "LINKUSDT"]

TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
WINDOWS = [
    ("Win1 (Nov)", TODAY - timedelta(days=180), TODAY - timedelta(days=150)),
    ("Win2 (Dec)", TODAY - timedelta(days=150), TODAY - timedelta(days=120)),
    ("Win3 (Jan)", TODAY - timedelta(days=120), TODAY - timedelta(days=90)),
    ("Win4 (Feb)", TODAY - timedelta(days=90),  TODAY - timedelta(days=60)),
    ("Win5 (Mar)", TODAY - timedelta(days=60),  TODAY - timedelta(days=30)),
    ("Win6 (Apr)", TODAY - timedelta(days=30),  TODAY),
]

INITIAL_BALANCE = 1000.0


def _v21_config(symbol: str) -> BacktestConfig:
    from trend_robot.config import get_preset
    p = get_preset(symbol)
    return BacktestConfig(
        symbol=symbol, timeframe="15m", htf_timeframe="4H",
        leverage=int(p.get("leverage", 10)),
        trade_amount=INITIAL_BALANCE * 0.15,
        capital_engagement=0.15,
        taker_fee=0.0006, funding_rate_per_8h=0.0001,
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
        use_atr_sl=True, sl_atr_multiplier=2.0,
        max_signal_age_bars=3,
        trailing_activation_percent=float(p.get("trailing_activation_percent", 1.0)),
        trailing_atr_multiplier=float(p.get("trailing_atr_multiplier", 1.5)),
        use_partial_tp=bool(p.get("use_partial_tp", True)),
        partial_tp1_percent=float(p.get("partial_tp1_percent", 2.0)),
        partial_tp1_size_pct=float(p.get("partial_tp1_size_pct", 0.33)),
        partial_tp2_percent=float(p.get("partial_tp2_percent", 5.0)),
        partial_tp2_size_pct=float(p.get("partial_tp2_size_pct", 0.33)),
        use_opposite_signal_exit=True,
        opposite_signal_requires_full_confirm=True,
        min_net_profit_fee_factor=1.0,
        max_drawdown_percent=float(p.get("max_drawdown_percent", 20.0)),
        cooldown_bars_after_sl=int(p.get("cooldown_bars_after_sl", 5)),
        max_trades_per_hour=4,
        initial_balance=INITIAL_BALANCE,
    )


def _measure_market_regime(candles, htf_candles) -> Dict[str, float]:
    """Davr xarakteristikasini hisoblash — ADX, volatility, trend strength."""
    from backtest.engine import calc_adx, calc_atr
    if not candles:
        return {}
    adxs = calc_adx(candles, 14)
    atrs = calc_atr(candles, 14)
    valid_adx = [a for a in adxs if a > 0]
    avg_adx = sum(valid_adx) / len(valid_adx) if valid_adx else 0
    avg_atr_pct = (sum(atrs[-30:]) / 30) / candles[-1].close * 100 if candles else 0

    # Trend strength: O'rtacha ADX > 25 bo'lsa "trending", aks holda "choppy"
    trending_share = sum(1 for a in valid_adx if a > 25) / len(valid_adx) if valid_adx else 0

    # Range / Volatility
    closes = [c.close for c in candles]
    price_range = (max(closes) - min(closes)) / closes[0] * 100 if closes else 0

    return {
        "avg_adx": avg_adx,
        "avg_atr_pct": avg_atr_pct,
        "trending_share": trending_share * 100,  # foiz
        "price_range_pct": price_range,
        "candles": len(candles),
    }


def diagnose_window(label: str, start, end):
    print(f"\n{'='*100}")
    print(f"{label}: {start.date()} → {end.date()}")
    print('='*100)

    htf_start = start - timedelta(days=15)

    print(f"\n  {'Pair':<10} {'AvgADX':>8} {'ATR%':>7} {'Trend%':>8} {'Range%':>8}  ||  "
          f"{'Trades':>7} {'Wins':>5} {'WR%':>6} {'NetPnL':>9} {'Exits':<35}")
    print("  " + "-"*120)

    window_summary = {
        "trades": 0, "wins": 0, "net": 0.0, "fees": 0.0,
        "exit_breakdown": defaultdict(lambda: {"count": 0, "pnl": 0.0}),
    }

    for symbol in PAIRS:
        candles = load_candles(symbol, "15m", start, end)
        htf = load_candles(symbol, "4H", htf_start, end)
        if not candles:
            continue

        regime = _measure_market_regime(candles, htf)

        cfg = _v21_config(symbol)
        bt = Backtester(cfg, candles, htf_candles=htf)
        res = bt.run()

        # Exit reason breakdown
        exit_pnl: Dict[str, float] = defaultdict(float)
        exit_count: Dict[str, int] = defaultdict(int)
        for t in res.trades:
            exit_pnl[t.reason] += t.net_pnl
            exit_count[t.reason] += 1
            window_summary["exit_breakdown"][t.reason]["count"] += 1
            window_summary["exit_breakdown"][t.reason]["pnl"] += t.net_pnl

        exits_str = ", ".join(
            f"{k}({c}/{exit_pnl[k]:+.0f})"
            for k, c in sorted(exit_count.items(), key=lambda x: -x[1])
        ) if exit_count else "-"

        print(f"  {symbol:<10} {regime.get('avg_adx', 0):>7.1f} "
              f"{regime.get('avg_atr_pct', 0):>6.2f}% "
              f"{regime.get('trending_share', 0):>7.1f}% "
              f"{regime.get('price_range_pct', 0):>7.1f}%  ||  "
              f"{res.total_trades:>7} {res.winning_trades:>5} "
              f"{res.winrate:>5.1f}% "
              f"{res.total_pnl:>+8.2f}  {exits_str:<35}")

        window_summary["trades"] += res.total_trades
        window_summary["wins"] += res.winning_trades
        window_summary["net"] += res.total_pnl
        window_summary["fees"] += res.total_fees

    print("  " + "-"*120)
    wr = window_summary["wins"] / window_summary["trades"] * 100 if window_summary["trades"] else 0
    net_after = window_summary["net"] - window_summary["fees"]
    print(f"  {'JAMI':<10} {' ':>7} {' ':>6}  {' ':>7}  {' ':>7}   ||  "
          f"{window_summary['trades']:>7} {window_summary['wins']:>5} "
          f"{wr:>5.1f}% net=${window_summary['net']:>+8.2f}  fees=${window_summary['fees']:.2f}  "
          f"NET_AFTER=${net_after:+.2f}")

    print(f"\n  Window'dagi exit breakdown:")
    total_pnl = sum(d["pnl"] for d in window_summary["exit_breakdown"].values())
    for reason, d in sorted(window_summary["exit_breakdown"].items(),
                            key=lambda x: x[1]["pnl"]):
        share = (d["pnl"] / total_pnl * 100) if total_pnl else 0
        print(f"    {reason:<20} count={d['count']:>4}  pnl=${d['pnl']:>+9.2f}  share={share:>5.1f}%")


def main():
    print("\n" + "█"*100)
    print("█" + " "*40 + "DIAGNOSTIKA" + " "*47 + "█")
    print("█"*100)

    for label, start, end in WINDOWS:
        diagnose_window(label, start, end)

    print("\n" + "█"*100)
    print("█" + " "*30 + "OXIRGI XULOSA / GIPOTEZA" + " "*44 + "█")
    print("█"*100)
    print("""
    1. Yo'qotuvchi oynalarda avg ADX past bo'ladi (~20-25) → choppy market signal
    2. SL eng ko'p zarar keltiruvchi exit bo'lishi kutiladi (false trend signals)
    3. Trending share past pair'lar (<40%) yomon ishlaydi
    4. Yechim: ADX-based pair filter + Choppiness Index gate + per-pair regime detection
    """)


if __name__ == "__main__":
    main()
