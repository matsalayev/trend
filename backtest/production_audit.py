"""
Production Audit Backtest — Per-pair 6mo backtest with CURRENT presets.json,
plus sensitivity scans (ADX/CHOP/HTF) for any pair with too few trades.

Goal: figure out whether production presets are too restrictive, especially
for AVAXUSDT which has been live 27h+ with 0 trades.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester
from trend_robot.config import SYMBOL_PRESETS

# ── Time window ────────────────────────────────────────────────────────────
TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
START = TODAY - timedelta(days=180)   # 6 months — cache available
END = TODAY
HTF_OFFSET = timedelta(days=20)
PERIOD_DAYS = (END - START).days

INITIAL_BALANCE = 1000.0
TRADE_AMOUNT = 150.0


def _build_config(symbol: str, preset: dict, **overrides) -> BacktestConfig:
    """Build BacktestConfig from preset, with overrides (for sensitivity scan)."""
    cfg = BacktestConfig(
        symbol=symbol,
        timeframe="15m",
        htf_timeframe="4H",
        leverage=int(preset.get("leverage", 10)),
        trade_amount=TRADE_AMOUNT,
        capital_engagement=0.15,
        taker_fee=0.0006,
        funding_rate_per_8h=0.0001,
        ema_fast=int(preset.get("ema_fast", 9)),
        ema_slow=int(preset.get("ema_slow", 21)),
        max_signal_age_bars=3,
        atr_period=int(preset.get("atr_period", 14)),
        adx_period=int(preset.get("adx_period", 14)),
        adx_threshold=float(preset.get("adx_threshold", 25.0)),
        supertrend_period=int(preset.get("supertrend_period", 10)),
        supertrend_multiplier=float(preset.get("supertrend_multiplier", 3.0)),
        use_choppiness_filter=True,
        chop_period=14,
        chop_max_for_entry=50.0,
        consecutive_losses_threshold=3,
        consecutive_losses_cooldown_bars=20,
        use_htf_filter=bool(preset.get("use_htf_filter", True)),
        htf_ema_fast=int(preset.get("htf_ema_fast", 21)),
        htf_ema_slow=int(preset.get("htf_ema_slow", 50)),
        initial_sl_percent=float(preset.get("initial_sl_percent", 3.0)),
        use_atr_sl=True,
        sl_atr_multiplier=2.0,
        trailing_activation_percent=float(preset.get("trailing_activation_percent", 1.0)),
        trailing_atr_multiplier=float(preset.get("trailing_atr_multiplier", 1.5)),
        use_partial_tp=bool(preset.get("use_partial_tp", True)),
        partial_tp1_percent=float(preset.get("partial_tp1_percent", 2.0)),
        partial_tp1_size_pct=float(preset.get("partial_tp1_size_pct", 0.33)),
        partial_tp2_percent=float(preset.get("partial_tp2_percent", 5.0)),
        partial_tp2_size_pct=float(preset.get("partial_tp2_size_pct", 0.33)),
        use_opposite_signal_exit=True,
        opposite_signal_requires_full_confirm=True,
        min_net_profit_fee_factor=1.0,
        max_drawdown_percent=20.0,
        cooldown_bars_after_sl=int(preset.get("cooldown_bars_after_sl", 5)),
        max_trades_per_hour=4,
        initial_balance=INITIAL_BALANCE,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _avg_duration_hours(trades) -> float:
    if not trades:
        return 0.0
    durations = [(t.closed_at - t.opened_at) / 1000 / 3600 for t in trades]
    return sum(durations) / len(durations)


def _run_one(symbol: str, preset: dict, **overrides) -> dict:
    """Run backtest, return summary dict."""
    candles = load_candles(symbol, "15m", START, END)
    htf = load_candles(symbol, "4H", START - HTF_OFFSET, END)
    if not candles or len(candles) < 200:
        return {"error": "no data"}

    cfg = _build_config(symbol, preset, **overrides)
    bt = Backtester(cfg, candles, htf_candles=htf)
    res = bt.run()

    avg_dur_h = _avg_duration_hours(res.trades)
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winrate": res.winrate,
        "return_pct": res.return_percent,
        "max_dd_pct": res.max_drawdown_percent,
        "profit_factor": res.profit_factor,
        "net_pnl": res.total_pnl,
        "avg_dur_h": avg_dur_h,
        "fees": res.total_fees,
    }


def _fmt_pf(pf: float) -> str:
    if pf == float("inf"):
        return "  inf"
    return f"{pf:>5.2f}"


def main():
    print("=" * 110)
    print(f"PRODUCTION AUDIT - Per-pair {PERIOD_DAYS}-day backtest with CURRENT presets")
    print(f"Period: {START.date()} -> {END.date()}")
    print(f"Initial balance: ${INITIAL_BALANCE}, trade margin: ${TRADE_AMOUNT}, taker fee: 0.06%")
    print("=" * 110)

    pairs = sorted(SYMBOL_PRESETS.keys())
    print(f"Pairs in presets.json: {pairs}\n")

    print(f"{'PAIR':<10} {'trades':>7} {'WR%':>6} {'ret%':>7} {'maxDD%':>7} "
          f"{'PF':>6} {'avgDur':>7} {'preset':<32}")
    print("-" * 110)

    prod_results = {}
    for sym in pairs:
        p = SYMBOL_PRESETS[sym]
        r = _run_one(sym, p)
        if "error" in r:
            print(f"{sym:<10} ERROR: {r['error']}")
            continue
        prod_results[sym] = r
        preset_label = (f"adx={p['adx_threshold']:.0f} "
                        f"htf={'Y' if p['use_htf_filter'] else 'N'} "
                        f"st_mult={p['supertrend_multiplier']:.1f} "
                        f"ema={p['ema_fast']}/{p['ema_slow']}")
        flag = ""
        if r["trades"] == 0:
            flag = " <- 0 TRADES!"
        elif r["trades"] < 10:
            flag = " <- few"
        print(f"{sym:<10} {r['trades']:>7} {r['winrate']:>5.1f}% "
              f"{r['return_pct']:>+6.1f}% {r['max_dd_pct']:>6.1f}% "
              f"{_fmt_pf(r['profit_factor'])} {r['avg_dur_h']:>6.1f}h "
              f"{preset_label}{flag}")

    # ── Sensitivity scan for low-trade pairs ───────────────────────────────────
    weak = [s for s, r in prod_results.items() if r["trades"] < 15]
    print()
    print("=" * 110)
    print(f"SENSITIVITY SCAN - pairs with <15 trades over {PERIOD_DAYS}d: {weak}")
    print("=" * 110)

    scan_results = {}
    for sym in weak:
        p = SYMBOL_PRESETS[sym]
        scan_results[sym] = {}

        print(f"\n{sym}  (current preset: adx={p['adx_threshold']}, "
              f"htf={'Y' if p['use_htf_filter'] else 'N'}, "
              f"chop_max=50, st_mult={p['supertrend_multiplier']})")

        # ADX scan
        print(f"  ADX scan       :", end="")
        for adx_t in [15, 18, 20, 22, 25]:
            r = _run_one(sym, p, adx_threshold=adx_t)
            scan_results[sym][f"adx_{adx_t}"] = r
            print(f"  {adx_t}=>{r['trades']:>3}t/{r['return_pct']:+5.1f}%", end="")
        print()

        # CHOP scan
        print(f"  CHOP scan      :", end="")
        for chop_t in [50, 55, 60, 65, 70]:
            r = _run_one(sym, p, chop_max_for_entry=chop_t)
            scan_results[sym][f"chop_{chop_t}"] = r
            print(f"  {chop_t}=>{r['trades']:>3}t/{r['return_pct']:+5.1f}%", end="")
        print()

        # HTF scan
        print(f"  HTF filter     :", end="")
        for htf in [True, False]:
            r = _run_one(sym, p, use_htf_filter=htf)
            scan_results[sym][f"htf_{htf}"] = r
            print(f"  {'ON' if htf else 'OFF':<3}=>{r['trades']:>3}t/{r['return_pct']:+5.1f}% (PF={_fmt_pf(r['profit_factor'])})", end="")
        print()

        # Combined alternatives — find best tuned variant
        print(f"  ALTERNATIVES   :")
        candidates = [
            ("adx30_htfOFF", {"adx_threshold": 30, "use_htf_filter": False}),
            ("adx28_htfOFF", {"adx_threshold": 28, "use_htf_filter": False}),
            ("adx25_htfON ", {"adx_threshold": 25, "use_htf_filter": True}),
            ("adx25_htfOFF", {"adx_threshold": 25, "use_htf_filter": False}),
            ("adx30_chop60", {"adx_threshold": 30, "chop_max_for_entry": 60}),
            ("adx28_htfON ", {"adx_threshold": 28, "use_htf_filter": True}),
        ]
        for label, ovr in candidates:
            r = _run_one(sym, p, **ovr)
            scan_results[sym][f"alt_{label}"] = r
            tag = " <- candidate" if (r["trades"] >= 20 and r["return_pct"] > 0
                                      and r["profit_factor"] >= 1.2) else ""
            print(f"    {label}: {r['trades']:>3}t  ret={r['return_pct']:+5.1f}%  "
                  f"WR={r['winrate']:>4.1f}%  DD={r['max_dd_pct']:>4.1f}%  "
                  f"PF={_fmt_pf(r['profit_factor'])}{tag}")

    # ── Save full results ─────────────────────────────────────────────────────
    out = Path(__file__).parent / "production_audit_results.json"
    payload = {
        "period_days": PERIOD_DAYS,
        "period_start": START.isoformat(),
        "period_end": END.isoformat(),
        "initial_balance": INITIAL_BALANCE,
        "trade_amount": TRADE_AMOUNT,
        "production_presets": prod_results,
        "sensitivity_scan": {k: {kk: vv for kk, vv in v.items()} for k, v in scan_results.items()},
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nFull results saved: {out}")


if __name__ == "__main__":
    main()
