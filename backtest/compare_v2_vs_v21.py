"""
Trend Robot — v2.0 (eski) vs v2.1 (yangi) backtest comparator.

Maqsad: 2026-05-04 incidentdan keyin qo'llanilgan barcha fix'lar samarasini
har bir pair uchun raqamlar bilan ko'rsatish.

Eski (v2.0) baseline:
- trade_amount = balans (CAPITAL_ENGAGEMENT bypass)
- Fixed % SL faqat
- ATR-based SL yo'q
- Stale EMA cross qabul qilinishi mumkin (max_signal_age_bars yo'q)
- Cooldown tick'da (5 sek)
- Opposite signal exit yo'q
- Fee-aware exit yo'q
- Trades-per-hour limit yo'q

Yangi (v2.1) — barcha bug-fixes yoqilgan:
- trade_amount = MARGIN, default = balans × CAPITAL_ENGAGEMENT
- ATR-adaptive SL
- Max signal age 3 bars
- Cooldown candle bar'da
- Opposite signal exit (fee-aware)
- Voluntary exit fee-aware
- Trades-per-hour limit

Ishlatish:
    python -m backtest.compare_v2_vs_v21
"""
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# Production'dagi kunlardan foydalanamiz: 30 kun (2026-03-23 → 2026-04-22).
# Bu cached CSV'lar mavjud davr.
START = datetime(2026, 3, 23, tzinfo=timezone.utc)
END = datetime(2026, 4, 22, tzinfo=timezone.utc)
HTF_START = datetime(2026, 2, 21, tzinfo=timezone.utc)  # 4H uchun warmup ko'p kerak

# Test pair'lar — production'da ishlatilganlar (AVAX, ADA — eng zararli)
PAIRS = ["AVAXUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT", "DOGEUSDT", "LINKUSDT"]

INITIAL_BALANCE = 1000.0


def _v20_config(symbol: str) -> BacktestConfig:
    """v2.0 baseline: yangi fix'lar O'CHIQ.
    BUG simulation: trade_amount = balance (capital_engagement bypass).
    """
    cfg = _base_config(symbol)
    # v2.0 BUG: trade_amount = full balance (CAPITAL_ENGAGEMENT ignored)
    cfg.trade_amount = INITIAL_BALANCE  # entire balance — this is the v2.0 bug
    cfg.capital_engagement = 1.0  # not used in v2.0 anyway

    # v2.0: barcha yangi fix'lar disabled
    cfg.use_atr_sl = False
    cfg.max_signal_age_bars = 999  # disable stale check
    cfg.use_opposite_signal_exit = False
    cfg.min_net_profit_fee_factor = 0.0  # no fee-aware
    cfg.max_trades_per_hour = 0  # no limit
    return cfg


def _v21_config(symbol: str) -> BacktestConfig:
    """v2.1 yangi: barcha fix'lar YOQILGAN, CAPITAL_ENGAGEMENT to'g'ri."""
    cfg = _base_config(symbol)
    # v2.1 FIX: trade_amount = balance × engagement (proper margin allocation)
    cfg.trade_amount = INITIAL_BALANCE * 0.15  # 15% of balance as margin
    cfg.capital_engagement = 0.15

    # v2.1: barcha fix'lar yoqilgan
    cfg.use_atr_sl = True
    cfg.sl_atr_multiplier = 2.0
    cfg.max_signal_age_bars = 3
    cfg.use_opposite_signal_exit = True
    cfg.opposite_signal_requires_full_confirm = True
    cfg.min_net_profit_fee_factor = 1.0
    cfg.max_trades_per_hour = 4
    return cfg


def _base_config(symbol: str) -> BacktestConfig:
    """Pair preset asosida (presets.json'dan) base config."""
    from trend_robot.config import get_preset
    p = get_preset(symbol)
    return BacktestConfig(
        symbol=symbol,
        timeframe="15m",
        htf_timeframe="4H",
        leverage=int(p.get("leverage", 10)),
        trade_amount=0,  # _v20/_v21 set qiladi
        taker_fee=0.0006,
        funding_rate_per_8h=0.0001,
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
        trailing_activation_percent=float(p.get("trailing_activation_percent", 1.0)),
        trailing_atr_multiplier=float(p.get("trailing_atr_multiplier", 1.5)),
        use_partial_tp=bool(p.get("use_partial_tp", True)),
        partial_tp1_percent=float(p.get("partial_tp1_percent", 2.0)),
        partial_tp1_size_pct=float(p.get("partial_tp1_size_pct", 0.33)),
        partial_tp2_percent=float(p.get("partial_tp2_percent", 5.0)),
        partial_tp2_size_pct=float(p.get("partial_tp2_size_pct", 0.33)),
        max_drawdown_percent=float(p.get("max_drawdown_percent", 20.0)),
        cooldown_bars_after_sl=int(p.get("cooldown_bars_after_sl", 5)),
        initial_balance=INITIAL_BALANCE,
    )


def run_backtest(cfg: BacktestConfig) -> Dict:
    candles = load_candles(cfg.symbol, "15m", START, END)
    htf_candles = load_candles(cfg.symbol, "4H", HTF_START, END)
    if not candles:
        return {}
    bt = Backtester(cfg, candles, htf_candles=htf_candles)
    res = bt.run()
    return {
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winrate": res.winrate,
        "net_pnl": res.total_pnl,
        "fees": res.total_fees,
        "max_dd_pct": res.max_drawdown_percent,
        "return_pct": res.return_percent,
        "final_balance": res.final_balance,
        "profit_factor": res.profit_factor if res.profit_factor != float('inf') else 99.99,
    }


def _print_row(symbol: str, label: str, r: Dict):
    if not r:
        print(f"  {label:6} {symbol:10}  NO DATA")
        return
    pf = "inf" if r["profit_factor"] >= 99.99 else f"{r['profit_factor']:.2f}"
    print(
        f"  {label:6} {symbol:10}  "
        f"trades={r['trades']:3}  "
        f"wins={r['wins']:3}  "
        f"WR={r['winrate']:5.1f}%  "
        f"net=${r['net_pnl']:+8.2f}  "
        f"fees=${r['fees']:6.2f}  "
        f"DD={r['max_dd_pct']:5.1f}%  "
        f"ret={r['return_pct']:+6.2f}%  "
        f"PF={pf:>5}"
    )


def main():
    print("=" * 110)
    print(f"Trend Bot — v2.0 vs v2.1 BACKTEST TAQQOSLASH")
    print(f"Davr: {START.date()} → {END.date()} (30 kun)")
    print(f"Initial balance: ${INITIAL_BALANCE}")
    print(f"v2.0 = eski (barcha bug'lar bilan), v2.1 = yangi (fixed)")
    print("=" * 110)

    summary = {"v2.0": {"net": 0.0, "fees": 0.0, "trades": 0, "wins": 0, "losses": 0},
               "v2.1": {"net": 0.0, "fees": 0.0, "trades": 0, "wins": 0, "losses": 0}}

    for symbol in PAIRS:
        print(f"\n--- {symbol} ---")
        v20 = run_backtest(_v20_config(symbol))
        v21 = run_backtest(_v21_config(symbol))

        _print_row(symbol, "v2.0", v20)
        _print_row(symbol, "v2.1", v21)

        # Pair-level delta
        if v20 and v21:
            delta = v21["net_pnl"] - v20["net_pnl"]
            print(f"  DELTA  {symbol:10}  net = ${delta:+8.2f}  "
                  f"({'+' if delta >= 0 else ''}{(delta/abs(v20['net_pnl'])*100 if v20['net_pnl'] != 0 else 0):+.0f}%)")
            for k in ("net_pnl", "fees", "trades", "wins", "losses"):
                short_k = "net" if k == "net_pnl" else k
                summary["v2.0"][short_k] += v20.get(k, 0)
                summary["v2.1"][short_k] += v21.get(k, 0)

    print("\n" + "=" * 110)
    print("UMUMIY YAKUN (barcha pair'lar bo'yicha):")
    for v, s in summary.items():
        wr = (s["wins"] / (s["wins"] + s["losses"]) * 100) if (s["wins"] + s["losses"]) else 0
        net_after_fees = s["net"] - s["fees"]
        print(f"  {v}:  trades={s['trades']:4}  WR={wr:5.1f}%  "
              f"net_pnl=${s['net']:+9.2f}  fees=${s['fees']:7.2f}  "
              f"NET_AFTER_FEES=${net_after_fees:+9.2f}")
    delta = (summary["v2.1"]["net"] - summary["v2.1"]["fees"]) - (summary["v2.0"]["net"] - summary["v2.0"]["fees"])
    print(f"\n  YAXSHILANISH: ${delta:+9.2f} (net after fees, jami)")
    print("=" * 110)


if __name__ == "__main__":
    main()
