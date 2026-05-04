"""
HEMA bot_pair_configs ni v3 walk-forward natijalari bilan yangilash.

User talab:
1. BARCHA 9 pair ENABLED (kim qaysi pair'ni xohlasa o'shasi)
2. Har pair uchun o'z optimal parametri (walk-forward asosida)
3. 6-OYLIK statistika ko'rsatish (60-day TEST emas)
4. Detailed stats (per-month, volume, max/min balance) customSettings'da

Statistika prioriteti:
- backtestReturn / Winrate / DD / Trades = FULL 6-month (180 kun) data
- backtestPeriodDays = 180
- detailedStats.test60d = out-of-sample
- detailedStats.full180d = full 6-month (asosiy)
- detailedStats.monthly = per-month breakdown
"""
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import load_candles
from backtest.engine import BacktestConfig, Backtester


TODAY = datetime(2026, 5, 4, tzinfo=timezone.utc)
TEST_START = TODAY - timedelta(days=60)
FULL_START = TODAY - timedelta(days=180)
HTF_OFFSET = timedelta(days=20)
INITIAL_BALANCE = 1000.0

with open("backtest/walk_forward_v3_all_pairs_results.json") as f:
    WF = json.load(f)

ALL_PAIRS = list(WF["per_pair"].keys())


def _config(symbol: str) -> BacktestConfig:
    pp = WF["per_pair"][symbol]
    return BacktestConfig(
        symbol=symbol, timeframe="15m", htf_timeframe="4H",
        leverage=10,
        trade_amount=INITIAL_BALANCE * 0.15,
        capital_engagement=0.15,
        taker_fee=0.0006, funding_rate_per_8h=0.0001,
        ema_fast=9, ema_slow=21,
        atr_period=14, adx_period=14,
        adx_threshold=float(pp["adx_threshold"]),
        supertrend_period=10, supertrend_multiplier=2.0,
        use_htf_filter=True, htf_ema_fast=21, htf_ema_slow=50,
        initial_sl_percent=3.0,
        use_atr_sl=True, sl_atr_multiplier=float(pp["sl_atr_multiplier"]),
        max_signal_age_bars=3,
        trailing_activation_percent=1.0,
        trailing_atr_multiplier=float(pp["trailing_atr_multiplier"]),
        use_partial_tp=True,
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


def _full_metrics(symbol: str, start, end) -> dict:
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
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.99

    # Max DD
    peak = INITIAL_BALANCE; max_dd_usd = 0
    for b in balances:
        if b > peak: peak = b
        if peak - b > max_dd_usd: max_dd_usd = peak - b
    max_dd_pct = max_dd_usd / peak * 100 if peak else 0

    # Sharpe (taxminiy)
    sharpe = 0
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
            sharpe = round((mean / std * math.sqrt(365)) if std > 0 else 0, 2)

    # Per-month
    monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "fees": 0.0})
    for t in res.trades:
        dt = datetime.fromtimestamp(t.opened_at / 1000, tz=timezone.utc)
        m = dt.strftime("%Y-%m")
        monthly[m]["trades"] += 1
        monthly[m]["pnl"] += t.net_pnl
        monthly[m]["fees"] += t.fees
        if t.net_pnl > 0:
            monthly[m]["wins"] += 1

    period_days = (end - start).days
    return {
        "periodDays": period_days,
        "trades": res.total_trades,
        "wins": res.winning_trades,
        "losses": res.losing_trades,
        "winratePct": round(res.winrate, 2),
        "totalVolumeUsd": round(total_volume, 2),
        "avgPositionSizeUsd": round(total_volume / 2 / res.total_trades, 2) if res.total_trades else 0,
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
        "profitFactor": pf,
        "sharpeRatio": sharpe,
        "monthlyBreakdown": {k: {**v, "pnl": round(v["pnl"], 2), "fees": round(v["fees"], 2)} for k, v in monthly.items()},
    }


def main():
    sql_parts = ["BEGIN;",
                 "DO $$ DECLARE bot_id text;",
                 "BEGIN",
                 "  SELECT id INTO bot_id FROM bots WHERE slug='trend-following-robot';"]

    for symbol in ALL_PAIRS:
        print(f"--- {symbol} ---")
        pp = WF["per_pair"][symbol]
        # 60-day TEST + 180-day FULL
        test60 = _full_metrics(symbol, TEST_START, TODAY)
        full180 = _full_metrics(symbol, FULL_START, TODAY)
        print(f"  TEST 60d:  return={test60['returnPct']}%, WR={test60['winratePct']}%, trades={test60['trades']}, DD={test60['maxDrawdownPct']}%")
        print(f"  FULL 180d: return={full180['returnPct']}%, WR={full180['winratePct']}%, trades={full180['trades']}, DD={full180['maxDrawdownPct']}%")

        # customSettings — yangi v3 + detailedStats
        custom = {
            "adxThreshold": float(pp["adx_threshold"]),
            "chopMaxForEntry": float(pp["chop_max_for_entry"]),
            "slAtrMultiplier": float(pp["sl_atr_multiplier"]),
            "trailingAtrMultiplier": float(pp["trailing_atr_multiplier"]),
            "useChoppinessFilter": True,
            "useAtrSl": True,
            "useOppositeSignalExit": True,
            "oppositeSignalRequiresFullConfirm": True,
            "maxSignalAgeBars": 3,
            "minNetProfitFeeFactor": 1.0,
            "maxTradesPerHour": 4,
            "consecutiveLossesThreshold": 3,
            "consecutiveLossesCooldownBars": 20,
            "chopPeriod": 14,
            "adxPeriod": 14,
            "atrPeriod": 14,
            "supertrendPeriod": 10,
            "supertrendMultiplier": 2.0,
            "useHtfFilter": True,
            "htfEmaFast": 21,
            "htfEmaSlow": 50,
            "trailingActivationPercent": 1.0,
            "usePartialTp": True,
            "partialTp1Percent": 2.0,
            "partialTp1SizePct": 0.33,
            "partialTp2Percent": 5.0,
            "partialTp2SizePct": 0.33,
            "maxDrawdownPercent": 20.0,
            "cooldownBarsAfterSl": 5,
            "_v3WalkForward": {
                "version": "v3_all_pairs",
                "trainPeriod": "2025-11-05 -> 2026-03-05 (120 days)",
                "testPeriod": "2026-03-05 -> 2026-05-04 (60 days, OUT-OF-SAMPLE)",
                "validatedAt": "2026-05-04",
            },
            "detailedStats": {
                "test60d": test60,
                "full180d": full180,
            },
        }
        cs_json = json.dumps(custom).replace("'", "''")

        # Asosiy maydonlarda 180-day FULL data
        sql_parts.append(f"  -- {symbol}: enable + v3 walk-forward params + 180-day stats")
        sql_parts.append(f"  UPDATE bot_pair_configs")
        sql_parts.append(f"  SET enabled = true,")
        sql_parts.append(f"      \"customSettings\" = '{cs_json}'::jsonb,")
        sql_parts.append(f"      leverage = 10,")
        sql_parts.append(f"      \"backtestReturn\" = {full180['returnPct']},")
        sql_parts.append(f"      \"backtestWinrate\" = {full180['winratePct']},")
        sql_parts.append(f"      \"backtestTrades\" = {full180['trades']},")
        sql_parts.append(f"      \"backtestDrawdown\" = {full180['maxDrawdownPct']},")
        sql_parts.append(f"      \"backtestProfitFactor\" = {full180['profitFactor']},")
        sql_parts.append(f"      \"backtestPeriodDays\" = 180,")
        sql_parts.append(f"      \"backtestUpdatedAt\" = NOW(),")
        sql_parts.append(f"      \"updatedAt\" = NOW()")
        sql_parts.append(f"  WHERE \"botId\" = bot_id AND \"tradingPair\" = '{symbol}';")

    # Update bot's supportedPairs
    sql_parts.append(f"  UPDATE bots SET \"supportedPairs\" = ARRAY{ALL_PAIRS}::text[] WHERE id = bot_id;")
    sql_parts.append("END $$;")
    sql_parts.append("")
    sql_parts.append("-- Verify")
    sql_parts.append("SELECT bpc.\"tradingPair\", bpc.enabled, bpc.\"backtestReturn\" AS bk_ret, bpc.\"backtestWinrate\" AS bk_wr, bpc.\"backtestTrades\" AS bk_t, bpc.\"backtestDrawdown\" AS bk_dd, bpc.\"backtestPeriodDays\" AS bk_days FROM bot_pair_configs bpc JOIN bots b ON b.id = bpc.\"botId\" WHERE b.slug = 'trend-following-robot' ORDER BY bpc.\"backtestReturn\" DESC NULLS LAST;")
    sql_parts.append("COMMIT;")

    out = Path("backtest/update_hema_v3.sql")
    out.write_text("\n".join(sql_parts), encoding='utf-8')
    print(f"\nSQL saqlandi: {out}")


if __name__ == "__main__":
    main()
