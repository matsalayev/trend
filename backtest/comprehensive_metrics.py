"""
TREND BOT v2.1 - COMPREHENSIVE METRICS REPORT

Walk-forward'da tasdiqlangan presets'ni ishlatib, har pair va davr uchun
TO'LIQ statistika beradi:
- Trade soni, win rate, profit factor
- NET PnL (USD) va RETURN % (foiz)
- Max balance, Min balance
- Total volume (turnover)
- Max drawdown ($ va %)
- Avg win, avg loss, expectancy
- Best trade, worst trade
- Sharpe ratio (taxminiy)
- Avg trade duration
- Per-pair, per-month breakdown

TEST davri: 2026-03-05 → 2026-05-04 (out-of-sample, 60 kun)
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester


TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
TEST_START = TODAY - timedelta(days=60)  # walk-forward TEST davri
TEST_END = TODAY
INITIAL_BALANCE = 1000.0
HTF_OFFSET = timedelta(days=20)

# Walk-forward v2 frequency-aware natijalari
with open("backtest/walk_forward_v2_results.json") as f:
    WF = json.load(f)

KEEP_PAIRS = WF["keep_pairs"]
PER_PAIR_PARAMS = {k: v for k, v in WF["per_pair"].items() if k in KEEP_PAIRS}


def _config(symbol: str) -> BacktestConfig:
    """v2.1 + walk-forward optimal params."""
    from trend_robot.config import get_preset
    p = get_preset(symbol)  # eski preset asosi
    pp = PER_PAIR_PARAMS[symbol]  # walk-forward optimal

    cfg = BacktestConfig(
        symbol=symbol, timeframe="15m", htf_timeframe="4H",
        leverage=int(p.get("leverage", 10)),
        trade_amount=INITIAL_BALANCE * 0.15,
        capital_engagement=0.15,
        taker_fee=0.0006, funding_rate_per_8h=0.0001,
        ema_fast=int(p.get("ema_fast", 9)),
        ema_slow=int(p.get("ema_slow", 21)),
        atr_period=14, adx_period=14,
        adx_threshold=float(pp["adx_threshold"]),
        supertrend_period=10,
        supertrend_multiplier=float(p.get("supertrend_multiplier", 3.0)),
        use_htf_filter=bool(p.get("use_htf_filter", True)),
        htf_ema_fast=21, htf_ema_slow=50,
        initial_sl_percent=float(p.get("initial_sl_percent", 3.0)),
        use_atr_sl=True, sl_atr_multiplier=float(pp["sl_atr_multiplier"]),
        max_signal_age_bars=3,
        trailing_activation_percent=1.0,
        trailing_atr_multiplier=float(pp["trailing_atr_multiplier"]),
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
        chop_max_for_entry=float(pp["chop_max_for_entry"]),
        consecutive_losses_threshold=3,
        consecutive_losses_cooldown_bars=20,
        initial_balance=INITIAL_BALANCE,
    )
    return cfg


def _full_metrics(symbol: str, start: datetime, end: datetime) -> Dict:
    """To'liq metrics — har possible statistik."""
    candles = load_candles(symbol, "15m", start, end)
    htf = load_candles(symbol, "4H", start - HTF_OFFSET, end)
    if not candles:
        return {}
    cfg = _config(symbol)
    bt = Backtester(cfg, candles, htf_candles=htf)
    res = bt.run()

    # Balance history
    balances = [b for _, b in res.balance_history]
    max_balance = max(balances) if balances else INITIAL_BALANCE
    min_balance = min(balances) if balances else INITIAL_BALANCE

    # Volume (turnover) — sum of all entry+exit notionals
    total_volume = 0.0
    for t in res.trades:
        total_volume += t.entry_price * t.size  # entry
        total_volume += t.exit_price * t.size   # exit

    # Trade metrics
    wins = [t for t in res.trades if t.net_pnl > 0]
    losses = [t for t in res.trades if t.net_pnl < 0]

    avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0

    # Expectancy = (wr × avg_win) + ((1-wr) × avg_loss)
    wr = res.winrate / 100
    expectancy = wr * avg_win + (1 - wr) * avg_loss

    # Profit factor
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown ($ va %)
    peak = INITIAL_BALANCE
    max_dd_usd = 0
    for b in balances:
        if b > peak:
            peak = b
        dd = peak - b
        if dd > max_dd_usd:
            max_dd_usd = dd
    max_dd_pct = max_dd_usd / peak * 100 if peak > 0 else 0

    # Sharpe ratio (kunlik returns asosida, taxminiy)
    if len(balances) > 1:
        # Group by day and calc daily returns
        daily_returns = []
        prev_b = INITIAL_BALANCE
        # ts_per_bar = 15min * 60sec = 900s, 96 bars/day
        chunk_size = 96  # 1 kun = 96 ta 15min candle
        for i in range(0, len(balances), chunk_size):
            chunk_end = min(i + chunk_size, len(balances))
            day_b = balances[chunk_end - 1]
            ret = (day_b - prev_b) / prev_b if prev_b > 0 else 0
            daily_returns.append(ret)
            prev_b = day_b
        if len(daily_returns) > 1:
            mean_ret = sum(daily_returns) / len(daily_returns)
            var = sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)
            std = math.sqrt(var)
            sharpe = (mean_ret / std * math.sqrt(365)) if std > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Trade durations
    durations_hours = [(t.closed_at - t.opened_at) / 1000 / 3600 for t in res.trades]
    avg_duration_h = sum(durations_hours) / len(durations_hours) if durations_hours else 0

    # Exit reason breakdown
    exit_breakdown = {}
    for t in res.trades:
        if t.reason not in exit_breakdown:
            exit_breakdown[t.reason] = {"count": 0, "pnl": 0.0}
        exit_breakdown[t.reason]["count"] += 1
        exit_breakdown[t.reason]["pnl"] += t.net_pnl

    # Final balance
    final_balance = res.final_balance
    net_pct = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    return {
        "symbol": symbol,
        "period_days": (end - start).days,
        # Trades
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winrate_pct": res.winrate,
        # Volume + sizing
        "total_volume_usd": total_volume,
        "avg_position_size_usd": (total_volume / 2 / res.total_trades) if res.total_trades else 0,
        # PnL
        "gross_pnl_usd": res.total_pnl + res.total_fees,  # before fees
        "fees_usd": res.total_fees,
        "net_pnl_usd": res.total_pnl,  # after fees? actually net_pnl already includes
        "final_balance_usd": final_balance,
        "max_balance_usd": max(max_balance, final_balance),
        "min_balance_usd": min(min_balance, INITIAL_BALANCE),
        "return_pct": net_pct,
        # Risk
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        # Trade quality
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "best_trade_usd": max((t.net_pnl for t in res.trades), default=0),
        "worst_trade_usd": min((t.net_pnl for t in res.trades), default=0),
        "expectancy_usd": expectancy,
        "profit_factor": pf if pf != float('inf') else 99.99,
        "sharpe_ratio": sharpe,
        # Time
        "avg_trade_duration_hours": avg_duration_h,
        # Exit reasons
        "exit_breakdown": exit_breakdown,
    }


def print_pair_report(m: Dict, label: str = ""):
    if not m:
        print(f"  No data")
        return
    print(f"\n  {'='*100}")
    print(f"  {label or m['symbol']:<88}{m['period_days']:>3} kun")
    print(f"  {'='*100}")
    print(f"    TRADE STATISTIKASI:")
    print(f"      Jami trade:           {m['trades']:>10}")
    print(f"      Win / Loss:           {m['wins']:>3} / {m['losses']:>3}  "
          f"({m['winrate_pct']:.1f}% WR)")
    print(f"      Avg trade duration:   {m['avg_trade_duration_hours']:>8.1f} soat")
    print(f"\n    BALANS / VOLUME:")
    print(f"      Initial balance:      ${INITIAL_BALANCE:>9.2f}")
    print(f"      Final balance:        ${m['final_balance_usd']:>9.2f}")
    print(f"      Max balance:          ${m['max_balance_usd']:>9.2f}")
    print(f"      Min balance:          ${m['min_balance_usd']:>9.2f}")
    print(f"      Total volume traded:  ${m['total_volume_usd']:>9.2f}")
    print(f"      Avg position size:    ${m['avg_position_size_usd']:>9.2f}")
    print(f"\n    DAROMAD:")
    print(f"      Gross PnL:            ${m['gross_pnl_usd']:>+9.2f}")
    print(f"      Total fees:           ${m['fees_usd']:>9.2f}")
    print(f"      NET PnL:              ${m['net_pnl_usd']:>+9.2f}")
    print(f"      RETURN:               {m['return_pct']:>+9.2f}%")
    if m['period_days'] > 0:
        apr = m['return_pct'] * 365 / m['period_days']
        print(f"      APR (annualized):     {apr:>+9.2f}%")
    print(f"\n    RISK:")
    print(f"      Max drawdown:         ${m['max_drawdown_usd']:>9.2f}  ({m['max_drawdown_pct']:.2f}%)")
    print(f"      Sharpe ratio:         {m['sharpe_ratio']:>9.2f}")
    print(f"      Profit factor:        {m['profit_factor']:>9.2f}")
    print(f"      Expectancy/trade:     ${m['expectancy_usd']:>+9.2f}")
    print(f"\n    TRADE SIFATI:")
    print(f"      Avg win:              ${m['avg_win_usd']:>+9.2f}")
    print(f"      Avg loss:             ${m['avg_loss_usd']:>+9.2f}")
    print(f"      Best trade:           ${m['best_trade_usd']:>+9.2f}")
    print(f"      Worst trade:          ${m['worst_trade_usd']:>+9.2f}")
    print(f"\n    EXIT BREAKDOWN:")
    for reason, d in sorted(m['exit_breakdown'].items(), key=lambda x: -x[1]['pnl']):
        avg = d['pnl'] / d['count'] if d['count'] else 0
        print(f"      {reason:<20} count={d['count']:>3}  total=${d['pnl']:>+8.2f}  avg=${avg:>+6.2f}")


def print_portfolio_summary(per_pair_metrics: Dict[str, Dict]):
    """Portfolio-level — agar har pair'ga $1000 ajratsangiz."""
    print("\n" + "█" * 100)
    print("█" + " " * 28 + "PORTFOLIO YAKUN (har pair'ga $1000)" + " " * 35 + "█")
    print("█" * 100)

    n_pairs = len(per_pair_metrics)
    init_capital = INITIAL_BALANCE * n_pairs
    final_capital = sum(m['final_balance_usd'] for m in per_pair_metrics.values())

    total_trades = sum(m['trades'] for m in per_pair_metrics.values())
    total_wins = sum(m['wins'] for m in per_pair_metrics.values())
    total_volume = sum(m['total_volume_usd'] for m in per_pair_metrics.values())
    total_fees = sum(m['fees_usd'] for m in per_pair_metrics.values())
    total_net = sum(m['net_pnl_usd'] for m in per_pair_metrics.values())

    portfolio_return = (final_capital - init_capital) / init_capital * 100
    period_days = next(iter(per_pair_metrics.values()))['period_days']
    apr = portfolio_return * 365 / period_days if period_days else 0

    # Worst max DD across pairs (portfolio max DD taxminiy)
    max_dd_pct = max((m['max_drawdown_pct'] for m in per_pair_metrics.values()), default=0)

    print(f"\n  Pair count:           {n_pairs}")
    print(f"  Test period:          {period_days} kun")
    print(f"  Initial portfolio:    ${init_capital:>10.2f}")
    print(f"  Final portfolio:      ${final_capital:>10.2f}")
    print(f"  PORTFOLIO RETURN:     {portfolio_return:>+10.2f}%  ({period_days} kun)")
    print(f"  APR:                  {apr:>+10.2f}%")
    print(f"  Total trades:         {total_trades}")
    print(f"  Total wins:           {total_wins}  ({total_wins/total_trades*100:.1f}% WR)")
    print(f"  Total volume:         ${total_volume:>10.2f}")
    print(f"  Total fees:           ${total_fees:>10.2f}")
    print(f"  Net PnL:              ${total_net:>+10.2f}")
    print(f"  Worst per-pair DD:    {max_dd_pct:.2f}% (portfolio jami pastroq bo'ladi diversifikatsiyadan)")


def run_period(label: str, start: datetime, end: datetime, verbose: bool = True):
    print("\n" + "█" * 100)
    print(f"█ {label:<96} █")
    print(f"█ Davr: {start.date()} → {end.date()} ({(end-start).days} kun){'':<60} █")
    print("█" * 100)

    per_pair = {}
    for symbol in KEEP_PAIRS:
        m = _full_metrics(symbol, start, end)
        if m:
            per_pair[symbol] = m
            if verbose:
                print_pair_report(m)

    if per_pair:
        print_portfolio_summary(per_pair)
    return per_pair


def main():
    print("█" * 100)
    print("█" + " " * 25 + "TREND BOT v2.1 - COMPREHENSIVE METRICS" + " " * 31 + "█")
    print("█" * 100)
    print(f"\n  Initial balance: ${INITIAL_BALANCE:.0f} per pair")
    print(f"  Strategy: v2.1 (CHOP filter, ATR SL, opposite exit, fee-aware, walk-forward optimal params)")
    print(f"  Pair count: {len(KEEP_PAIRS)}")
    print(f"  Pairs: {', '.join(KEEP_PAIRS)}")

    # OUT-OF-SAMPLE TEST (60 kun) — BU ASOSIY metrik
    test_metrics = run_period(
        "1️⃣  OUT-OF-SAMPLE TEST (haqiqiy ko'rsatkich — robot ko'rmagan davr)",
        TEST_START, TEST_END, verbose=True,
    )

    # Full 6-month (train + test) — for context
    full_start = TEST_END - timedelta(days=180)
    full_metrics = run_period(
        "2️⃣  TO'LIQ 6 OYLIK PICTURE (informatsion — train ham bor, slightly overfit)",
        full_start, TEST_END, verbose=False,
    )


if __name__ == "__main__":
    main()
