"""
Per-pair detailed stats — 60 kun TEST + 180 kun FULL.

Har pair uchun comprehensive statistika hisoblaydi va HEMA bot_pair_configs.customSettings
ga "detailedStats" ob'ekti sifatida qo'shadi. Bu HEMA dashboard'da batafsil
ko'rsatish uchun ishlatiladi.

Statistika:
- Trade soni, win rate, profit factor
- NET PnL ($ va %), APR
- Volume (turnover)
- Max/Min balance
- Max drawdown ($ va %)
- Avg win, avg loss, expectancy
- Best/worst trade
- Sharpe ratio
- Per-month breakdown (oylik PnL/trade)
"""
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester


TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
TEST_START = TODAY - timedelta(days=60)
FULL_START = TODAY - timedelta(days=180)
HTF_OFFSET = timedelta(days=20)
INITIAL_BALANCE = 1000.0

with open("backtest/walk_forward_v2_results.json") as f:
    WF = json.load(f)

KEEP = WF["keep_pairs"]
PER_PAIR_PARAMS = WF["per_pair"]


def _config(symbol: str) -> BacktestConfig:
    from trend_robot.config import get_preset
    p = get_preset(symbol)
    pp = PER_PAIR_PARAMS[symbol]
    return BacktestConfig(
        symbol=symbol, timeframe="15m", htf_timeframe="4H",
        leverage=int(p.get("leverage", 10)),
        trade_amount=INITIAL_BALANCE * 0.15, capital_engagement=0.15,
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
        max_drawdown_percent=20.0, cooldown_bars_after_sl=5,
        max_trades_per_hour=4,
        use_choppiness_filter=True,
        chop_max_for_entry=float(pp["chop_max_for_entry"]),
        consecutive_losses_threshold=3,
        consecutive_losses_cooldown_bars=20,
        initial_balance=INITIAL_BALANCE,
    )


def _full_metrics(symbol: str, start: datetime, end: datetime) -> Dict:
    candles = load_candles(symbol, "15m", start, end)
    htf = load_candles(symbol, "4H", start - HTF_OFFSET, end)
    if not candles:
        return {}
    cfg = _config(symbol)
    bt = Backtester(cfg, candles, htf_candles=htf)
    res = bt.run()

    balances = [b for _, b in res.balance_history]
    max_balance = max(balances) if balances else INITIAL_BALANCE
    min_balance = min(balances) if balances else INITIAL_BALANCE

    total_volume = sum(t.entry_price * t.size + t.exit_price * t.size for t in res.trades)

    wins = [t for t in res.trades if t.net_pnl > 0]
    losses = [t for t in res.trades if t.net_pnl < 0]
    avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0

    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 99.99

    # Max DD
    peak = INITIAL_BALANCE
    max_dd_usd = 0
    for b in balances:
        if b > peak:
            peak = b
        if peak - b > max_dd_usd:
            max_dd_usd = peak - b
    max_dd_pct = max_dd_usd / peak * 100 if peak else 0

    # Sharpe
    if len(balances) > 1:
        daily = []
        prev = INITIAL_BALANCE
        for i in range(0, len(balances), 96):
            day_b = balances[min(i + 95, len(balances) - 1)]
            daily.append((day_b - prev) / prev if prev else 0)
            prev = day_b
        if len(daily) > 1:
            mean = sum(daily) / len(daily)
            std = math.sqrt(sum((r - mean) ** 2 for r in daily) / len(daily))
            sharpe = (mean / std * math.sqrt(365)) if std > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Per-month breakdown
    monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "fees": 0.0})
    for t in res.trades:
        dt = datetime.fromtimestamp(t.opened_at / 1000, tz=timezone.utc)
        month_key = dt.strftime("%Y-%m")
        monthly[month_key]["trades"] += 1
        monthly[month_key]["pnl"] += t.net_pnl
        monthly[month_key]["fees"] += t.fees
        if t.net_pnl > 0:
            monthly[month_key]["wins"] += 1

    period_days = (end - start).days
    return {
        "periodDays": period_days,
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winratePct": round(res.winrate, 2),
        "totalVolumeUsd": round(total_volume, 2),
        "avgPositionSizeUsd": round(total_volume / 2 / res.total_trades, 2) if res.total_trades else 0,
        "grossPnlUsd": round(res.total_pnl + res.total_fees, 2),
        "feesUsd": round(res.total_fees, 2),
        "netPnlUsd": round(res.total_pnl, 2),
        "finalBalanceUsd": round(res.final_balance, 2),
        "maxBalanceUsd": round(max(max_balance, res.final_balance), 2),
        "minBalanceUsd": round(min(min_balance, INITIAL_BALANCE), 2),
        "returnPct": round((res.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
        "aprPct": round((res.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100 * 365 / period_days, 2) if period_days else 0,
        "maxDrawdownUsd": round(max_dd_usd, 2),
        "maxDrawdownPct": round(max_dd_pct, 2),
        "avgWinUsd": round(avg_win, 2),
        "avgLossUsd": round(avg_loss, 2),
        "bestTradeUsd": round(max((t.net_pnl for t in res.trades), default=0), 2),
        "worstTradeUsd": round(min((t.net_pnl for t in res.trades), default=0), 2),
        "expectancyUsd": round((res.winrate / 100) * avg_win + (1 - res.winrate / 100) * avg_loss, 2),
        "profitFactor": round(pf, 2),
        "sharpeRatio": round(sharpe, 2),
        "monthlyBreakdown": {k: {**v, "pnl": round(v["pnl"], 2), "fees": round(v["fees"], 2)} for k, v in monthly.items()},
    }


def main():
    print("=" * 100)
    print("PER-PAIR DETAILED STATS — TEST (60 kun) + FULL (180 kun)")
    print("=" * 100)

    all_stats = {}
    for symbol in KEEP:
        print(f"\n--- {symbol} ---")
        test = _full_metrics(symbol, TEST_START, TODAY)
        full = _full_metrics(symbol, FULL_START, TODAY)
        all_stats[symbol] = {"test60d": test, "full180d": full}

        print(f"  TEST 60d:  trades={test['trades']}, WR={test['winratePct']}%, "
              f"return={test['returnPct']}%, APR={test['aprPct']}%, "
              f"DD={test['maxDrawdownPct']}%, sharpe={test['sharpeRatio']}")
        print(f"  FULL 180d: trades={full['trades']}, WR={full['winratePct']}%, "
              f"return={full['returnPct']}%, APR={full['aprPct']}%, "
              f"DD={full['maxDrawdownPct']}%, sharpe={full['sharpeRatio']}")
        print(f"  Volume:    test=${test['totalVolumeUsd']:,.0f}, full=${full['totalVolumeUsd']:,.0f}")
        print(f"  Balance:   test min/max=${test['minBalanceUsd']}/${test['maxBalanceUsd']}, "
              f"final=${test['finalBalanceUsd']}")
        # Monthly
        for month, m in sorted(full["monthlyBreakdown"].items()):
            print(f"    {month}: {m['trades']} trade, {m['wins']} win, "
                  f"PnL ${m['pnl']:+.2f}, fees ${m['fees']:.2f}")

    # Generate SQL to update HEMA customSettings with detailedStats
    sql = ["BEGIN;"]
    sql.append("DO $$ DECLARE bot_id text;")
    sql.append("BEGIN")
    sql.append("  SELECT id INTO bot_id FROM bots WHERE slug='trend-following-robot';")

    for symbol in KEEP:
        # Read existing customSettings, merge detailedStats
        stats_obj = all_stats[symbol]
        stats_json = json.dumps({"detailedStats": stats_obj}).replace("'", "''")
        sql.append(f"  -- {symbol}: add detailedStats")
        sql.append(f"  UPDATE bot_pair_configs")
        sql.append(f"  SET \"customSettings\" = COALESCE(\"customSettings\",'{{}}'::jsonb) || '{stats_json}'::jsonb,")
        sql.append(f"      \"updatedAt\" = NOW()")
        sql.append(f"  WHERE \"botId\" = bot_id AND \"tradingPair\" = '{symbol}';")

    sql.append("END $$;")
    sql.append("COMMIT;")

    out = Path("backtest/update_pair_detailed_stats.sql")
    out.write_text("\n".join(sql))
    print(f"\n\nSQL saqlandi: {out}")

    # Also save as JSON for documentation
    json_out = Path("backtest/per_pair_detailed_stats.json")
    json_out.write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"JSON saqlandi: {json_out}")


if __name__ == "__main__":
    main()
