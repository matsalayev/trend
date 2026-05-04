"""
Quick sanity check — CHOP filter samarasini 6 oyna'da ko'rish.

Avval (CHOP yo'q) → keyin (CHOP yoqilgan) — quyidagilarni taqqoslaymiz:
- Total trades
- Win rate
- Net PnL
- SL hit count (asosiy zarar manbai)
"""
from datetime import datetime, timedelta, timezone
from collections import defaultdict

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


def _config(symbol: str, use_chop: bool, chop_max: float = 50.0) -> BacktestConfig:
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
        # YANGI v2.1 fields:
        use_choppiness_filter=use_chop,
        chop_max_for_entry=chop_max,
        consecutive_losses_threshold=3,
        consecutive_losses_cooldown_bars=20,
        initial_balance=INITIAL_BALANCE,
    )


def _run(cfg: BacktestConfig, start, end) -> dict:
    htf_start = start - timedelta(days=15)
    candles = load_candles(cfg.symbol, "15m", start, end)
    htf = load_candles(cfg.symbol, "4H", htf_start, end)
    if not candles:
        return {}
    bt = Backtester(cfg, candles, htf_candles=htf)
    res = bt.run()
    sl_count = sum(1 for t in res.trades if t.reason == "SL")
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "sl_count": sl_count,
        "max_dd_pct": res.max_drawdown_percent,
    }


def main():
    print("=" * 110)
    print(f"CHOP FILTER SANITY TEST (6 oyna × 7 pair)")
    print(f"Avval (CHOP off) vs Keyin (CHOP on, max=50) — diagnostika asosida 50 tanlandi")
    print("=" * 110)

    for chop_max in [None, 60.0, 50.0, 40.0]:
        if chop_max is None:
            label = "OFF"
            use_chop = False
            cm = 100
        else:
            label = f"ON (max={chop_max})"
            use_chop = True
            cm = chop_max
        print(f"\n--- CHOP filter: {label} ---")
        gtotal = {"trades": 0, "wins": 0, "net": 0.0, "fees": 0.0, "sl": 0}
        win_count = 0  # profit qilgan oynalar soni

        for win_name, start, end in WINDOWS:
            ptotal = {"trades": 0, "wins": 0, "net": 0.0, "fees": 0.0, "sl": 0}
            for symbol in PAIRS:
                r = _run(_config(symbol, use_chop, cm), start, end)
                if not r:
                    continue
                ptotal["trades"] += r["trades"]
                ptotal["wins"] += r["wins"]
                ptotal["net"] += r["net_pnl"]
                ptotal["fees"] += r["fees"]
                ptotal["sl"] += r["sl_count"]
            wr = ptotal["wins"] / ptotal["trades"] * 100 if ptotal["trades"] else 0
            net_after = ptotal["net"] - ptotal["fees"]
            print(f"  {win_name:<14} trades={ptotal['trades']:>4} WR={wr:>5.1f}% "
                  f"SL={ptotal['sl']:>3}  net=${ptotal['net']:>+8.2f} fees=${ptotal['fees']:>6.2f} "
                  f"NET_AFTER=${net_after:>+8.2f}")
            for k in ("trades", "wins", "net", "fees", "sl"):
                gtotal[k] += ptotal[k]
            if net_after > 0:
                win_count += 1

        wr = gtotal["wins"] / gtotal["trades"] * 100 if gtotal["trades"] else 0
        net_after = gtotal["net"] - gtotal["fees"]
        print(f"  {'JAMI':<14} trades={gtotal['trades']:>4} WR={wr:>5.1f}% "
              f"SL={gtotal['sl']:>3}  net=${gtotal['net']:>+8.2f} fees=${gtotal['fees']:>6.2f} "
              f"NET_AFTER=${net_after:>+9.2f}  | profit oynalari: {win_count}/6")


if __name__ == "__main__":
    main()
