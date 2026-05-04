"""
Multi-parameter sensitivity — qaysi o'zgartirish eng katta yaxshilanish beradi?

Diagnostika ko'rsatdi:
- SL avg loss = -$45 per trade
- TRAIL avg win = +$10-15 per trade
- Asimmetriya: zarar wins'dan 3x katta

Sinaladigan kombinatsiyalar:
1. Baseline (hozirgi v2.1)
2. Tighter SL (sl_atr_multiplier=1.0 vs 2.0) — loss size'ni kichik
3. Wider trail (trailing_atr=3.0) — winners'ga ko'proq joy
4. ATR-vol filter (faqat past-vol pair'larda entry — Win 6 patterni)
5. Tighter MAX_DD (max_drawdown=10% — bad streak'larda erta to'xtash)
6. CHOP=40 + tight SL (combo)
7. Strict regime: ADX>30 + CHOP<40 (faqat juda kuchli trend)
"""
from datetime import datetime, timedelta, timezone
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
        max_drawdown_percent=float(p.get("max_drawdown_percent", 20.0)),
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


def _run(cfg: BacktestConfig, start, end):
    htf_start = start - timedelta(days=15)
    candles = load_candles(cfg.symbol, "15m", start, end)
    htf = load_candles(cfg.symbol, "4H", htf_start, end)
    if not candles:
        return None
    bt = Backtester(cfg, candles, htf_candles=htf)
    res = bt.run()
    sl_count = sum(1 for t in res.trades if t.reason == "SL")
    sl_pnl = sum(t.net_pnl for t in res.trades if t.reason == "SL")
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "sl_count": sl_count,
        "sl_pnl": sl_pnl,
        "max_dd_pct": res.max_drawdown_percent,
    }


def aggregate(overrides: dict, label: str):
    """Run config across all windows + pairs, return aggregate."""
    g = {"trades": 0, "wins": 0, "net": 0.0, "fees": 0.0, "sl": 0, "sl_pnl": 0.0}
    profit_windows = 0
    for _, start, end in WINDOWS:
        win_total = {"net": 0.0, "fees": 0.0}
        for symbol in PAIRS:
            r = _run(_config(symbol, **overrides), start, end)
            if not r:
                continue
            g["trades"] += r["trades"]
            g["wins"] += r["wins"]
            g["net"] += r["net_pnl"]
            g["fees"] += r["fees"]
            g["sl"] += r["sl_count"]
            g["sl_pnl"] += r["sl_pnl"]
            win_total["net"] += r["net_pnl"]
            win_total["fees"] += r["fees"]
        if win_total["net"] - win_total["fees"] > 0:
            profit_windows += 1
    wr = g["wins"] / g["trades"] * 100 if g["trades"] else 0
    avg_sl_loss = g["sl_pnl"] / g["sl"] if g["sl"] else 0
    avg_win = (g["net"] - g["sl_pnl"]) / max(g["wins"], 1)
    net_after = g["net"] - g["fees"]
    print(f"  {label:<55} trades={g['trades']:>4} WR={wr:>5.1f}% "
          f"SL={g['sl']:>3} avg_SL=${avg_sl_loss:>+6.2f} avg_win=${avg_win:>+6.2f}  "
          f"NET=${net_after:>+8.2f} ({profit_windows}/6 profit)")
    return net_after, profit_windows


def main():
    print("=" * 130)
    print("PARAMETER SENSITIVITY (6 oyna × 7 pair = 42 backtest har bir variant)")
    print("=" * 130)

    # Baseline
    print("\n--- BASELINE ---")
    aggregate({}, "Baseline (hozirgi v2.1)")

    print("\n--- SL SIZE FIXES (zarar size'ni kichiklash) ---")
    aggregate({"sl_atr_multiplier": 1.0}, "Tight SL (atr_mult=1.0)")
    aggregate({"sl_atr_multiplier": 1.5}, "Medium SL (atr_mult=1.5)")
    aggregate({"initial_sl_percent": 2.0, "use_atr_sl": False}, "Fixed 2% SL")
    aggregate({"initial_sl_percent": 1.5, "use_atr_sl": False}, "Fixed 1.5% SL")

    print("\n--- TRAILING (winners'ga joy berish) ---")
    aggregate({"trailing_atr_multiplier": 0.8}, "Tight trail (atr=0.8)")
    aggregate({"trailing_atr_multiplier": 2.5}, "Wide trail (atr=2.5)")
    aggregate({"trailing_atr_multiplier": 3.5}, "Very wide trail (atr=3.5)")

    print("\n--- DRAWDOWN GUARD ---")
    aggregate({"max_drawdown_percent": 10.0}, "Tight DD circuit (10%)")
    aggregate({"max_drawdown_percent": 5.0}, "Very tight DD (5%)")
    aggregate({"max_drawdown_percent": 50.0}, "Loose DD (50%)")

    print("\n--- STRICT REGIME ---")
    aggregate({"adx_threshold": 30.0, "chop_max_for_entry": 40.0},
              "Strict: ADX>30 + CHOP<40")
    aggregate({"adx_threshold": 35.0, "chop_max_for_entry": 38.0},
              "Very strict: ADX>35 + CHOP<38")

    print("\n--- COMBO FIXES ---")
    aggregate({"sl_atr_multiplier": 1.0, "trailing_atr_multiplier": 3.0,
               "max_drawdown_percent": 10.0},
              "Combo: tight SL + wide trail + tight DD")
    aggregate({"sl_atr_multiplier": 1.0, "trailing_atr_multiplier": 3.0,
               "adx_threshold": 30, "chop_max_for_entry": 40,
               "max_drawdown_percent": 10},
              "Combo: tight SL + wide trail + strict regime")
    aggregate({"sl_atr_multiplier": 0.8, "trailing_atr_multiplier": 4.0,
               "adx_threshold": 30, "chop_max_for_entry": 38,
               "use_partial_tp": False, "max_drawdown_percent": 10},
              "Combo: ultra-tight SL + huge trail + strict + no PTP")

    print("\n--- LEVERAGE/SIZE CONSERVATIVE ---")
    aggregate({"capital_engagement": 0.05, "trade_amount": INITIAL_BALANCE * 0.05},
              "Lower size (5% engagement)")


if __name__ == "__main__":
    main()
