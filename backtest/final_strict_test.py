"""
Final test — eng yaxshi sozlama bilan har bir pair'ni ko'rish.
"Very Strict" rejim: ADX>35, CHOP<38, qolgan v2.1 fixes.

Maqsad: qaysi pair'lar profitable, qaysilari emas — pair selection uchun.
"""
from datetime import datetime, timedelta, timezone
from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester


# v2.1 audit (2026-05-04): ADAUSDT olib tashlandi
PAIRS = ["BTCUSDT", "ETHUSDT", "AVAXUSDT",
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


def _config(symbol: str, **overrides) -> BacktestConfig:
    from trend_robot.config import get_preset
    p = get_preset(symbol)
    cfg = BacktestConfig(
        symbol=symbol, timeframe="15m", htf_timeframe="4H",
        leverage=int(p.get("leverage", 10)),
        trade_amount=INITIAL_BALANCE * 0.15,
        capital_engagement=0.15,
        taker_fee=0.0006, funding_rate_per_8h=0.0001,
        ema_fast=int(p.get("ema_fast", 9)),
        ema_slow=int(p.get("ema_slow", 21)),
        atr_period=14, adx_period=14,
        adx_threshold=35.0,  # STRICT
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
        max_drawdown_percent=float(p.get("max_drawdown_percent", 20.0)),
        cooldown_bars_after_sl=5,
        max_trades_per_hour=4,
        use_choppiness_filter=True,
        chop_max_for_entry=38.0,  # STRICT
        consecutive_losses_threshold=3,
        consecutive_losses_cooldown_bars=20,
        initial_balance=INITIAL_BALANCE,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _run(cfg, start, end):
    htf_start = start - timedelta(days=15)
    candles = load_candles(cfg.symbol, "15m", start, end)
    htf = load_candles(cfg.symbol, "4H", htf_start, end)
    if not candles:
        return None
    bt = Backtester(cfg, candles, htf_candles=htf)
    res = bt.run()
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
    }


def main():
    print("=" * 110)
    print("VERY STRICT REJIM (ADX>35 + CHOP<38) — Per-pair break")
    print("=" * 110)
    print(f"\n  {'Pair':<10} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'NetPnL':>9} {'Fees':>7} "
          f"{'NET_AFTER':>10} {'Status'}")
    print("  " + "-" * 100)

    pair_totals = {}
    for symbol in PAIRS:
        ptotal = {"trades": 0, "wins": 0, "net": 0.0, "fees": 0.0, "windows_won": 0}
        for _, start, end in WINDOWS:
            r = _run(_config(symbol), start, end)
            if not r:
                continue
            ptotal["trades"] += r["trades"]
            ptotal["wins"] += r["wins"]
            ptotal["net"] += r["net_pnl"]
            ptotal["fees"] += r["fees"]
            if (r["net_pnl"] - r["fees"]) > 0:
                ptotal["windows_won"] += 1

        wr = ptotal["wins"] / ptotal["trades"] * 100 if ptotal["trades"] else 0
        net_after = ptotal["net"] - ptotal["fees"]
        status = "✓ KEEP" if net_after >= 0 else ("⚠ MARGINAL" if net_after > -50 else "✗ DROP")
        print(f"  {symbol:<10} {ptotal['trades']:>7} {ptotal['wins']:>5} {wr:>5.1f}% "
              f"${ptotal['net']:>+8.2f} ${ptotal['fees']:>6.2f} ${net_after:>+9.2f}  {status} "
              f"({ptotal['windows_won']}/6 win)")
        pair_totals[symbol] = (net_after, ptotal["windows_won"], ptotal["trades"])

    # Aggregate
    total_net = sum(v[0] for v in pair_totals.values())
    total_trades = sum(v[2] for v in pair_totals.values())
    print("  " + "-" * 100)
    print(f"  {'JAMI':<10} {total_trades:>7}  {' ':>5}  {' ':>5}  ")
    print(f"\n  Total NET (all pairs): ${total_net:+.2f}")

    # Best subset
    keep_pairs = [k for k, v in pair_totals.items() if v[0] > -50]
    keep_net = sum(pair_totals[k][0] for k in keep_pairs)
    drop_pairs = [k for k in pair_totals if k not in keep_pairs]
    print(f"\n  TANLOV: Faqat KEEP/MARGINAL pair'lar bilan ({', '.join(keep_pairs)}):")
    drop_str = ", ".join(drop_pairs) if drop_pairs else "yo'q"
    print(f"     NET = ${keep_net:+.2f} (drop qilingan: {drop_str})")


if __name__ == "__main__":
    main()
