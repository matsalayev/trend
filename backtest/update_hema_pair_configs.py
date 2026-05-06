"""
HEMA bot_pair_configs jadvalida trend bot pair'larini yangilash.

Walk-forward v2 frequency-aware natijalari asosida:
- 3 pair (ETH/AVAX/DOGE) ENABLED — yangi customSettings + backtest stats
- 6 pair (BTC/SOL/BNB/XRP/ADA/LINK) DISABLED — test'da yo'qotgan
"""
import json
import subprocess
from pathlib import Path

# Walk-forward v2 natijalari
with open("backtest/walk_forward_v2_results.json") as f:
    WF = json.load(f)

KEEP_PAIRS = WF["keep_pairs"]
DROP_PAIRS = WF["drop_pairs"]


def generate_sql():
    """SQL generate qilish."""
    sql_parts = [
        "-- HEMA bot_pair_configs yangilash (trend bot v2.1 v2)",
        "-- Walk-forward TEST davri: 2026-03-05 → 2026-05-04 (60 kun)",
        "",
        "BEGIN;",
        "",
        "-- Trend bot ID olish",
        "DO $$ DECLARE bot_id text;",
        "BEGIN",
        "  SELECT id INTO bot_id FROM bots WHERE slug = 'trend-following-robot';",
        "",
    ]

    # 1) DROP pair'larni disable qilish
    for sym in DROP_PAIRS:
        sql_parts.extend([
            f"  -- Disable {sym}",
            f"  UPDATE bot_pair_configs",
            f"  SET enabled = false,",
            f"      \"updatedAt\" = NOW()",
            f"  WHERE \"botId\" = bot_id AND \"tradingPair\" = '{sym}';",
            "",
        ])

    # 2) KEEP pair'larni yangilash (customSettings + stats)
    for sym in KEEP_PAIRS:
        pp = WF["per_pair"][sym]
        cs = {
            # v2.1 v2 walk-forward optimal
            "adxThreshold": float(pp["adx_threshold"]),
            "chopMaxForEntry": float(pp["chop_max_for_entry"]),
            "slAtrMultiplier": float(pp["sl_atr_multiplier"]),
            "trailingAtrMultiplier": float(pp["trailing_atr_multiplier"]),
            # v2.1 enable flags
            "useChoppinessFilter": True,
            "useAtrSl": True,
            "useOppositeSignalExit": True,
            "oppositeSignalRequiresFullConfirm": True,
            # Fixed v2.1 params
            "maxSignalAgeBars": 3,
            "minNetProfitFeeFactor": 1.0,
            "maxTradesPerHour": 4,
            "consecutiveLossesThreshold": 3,
            "consecutiveLossesCooldownBars": 20,
            # Standard
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
            # Audit metadata
            "_v21WalkForward": {
                "version": "v2_frequency_aware",
                "trainPeriod": "2025-11-05 → 2026-03-05 (120 kun)",
                "testPeriod": "2026-03-05 → 2026-05-04 (60 kun, OUT-OF-SAMPLE)",
                "validatedAt": "2026-05-04",
            },
        }
        cs_json = json.dumps(cs).replace("'", "''")  # SQL escape

        # Test stats — out-of-sample (haqiqiy ko'rsatkich)
        test_stats = pp["test"]
        # Backtest return % per-pair per-60-day initial $1000
        return_pct = test_stats["net_after"] / 1000 * 100  # 60-day return
        # PF approximation
        pf = 99.0 if test_stats["losses"] == 0 else round(
            test_stats["net_after"] / abs(test_stats["losses"] * 50), 2
        )

        sql_parts.extend([
            f"  -- Enable {sym} with v2.1 v2 params + walk-forward stats",
            f"  UPDATE bot_pair_configs",
            f"  SET enabled = true,",
            f"      \"customSettings\" = '{cs_json}'::jsonb,",
            f"      leverage = 10,",
            f"      \"backtestReturn\" = {round(return_pct, 2)},",
            f"      \"backtestWinrate\" = {round(test_stats['winrate'], 2)},",
            f"      \"backtestTrades\" = {test_stats['trades']},",
            f"      \"backtestDrawdown\" = {round(test_stats['max_dd_pct'], 2)},",
            f"      \"backtestProfitFactor\" = {pf},",
            f"      \"backtestPeriodDays\" = 60,",
            f"      \"backtestUpdatedAt\" = NOW(),",
            f"      \"updatedAt\" = NOW()",
            f"  WHERE \"botId\" = bot_id AND \"tradingPair\" = '{sym}';",
            "",
        ])

    sql_parts.extend([
        "END $$;",
        "",
        "-- Verify",
        "SELECT bpc.\"tradingPair\", bpc.enabled, bpc.leverage,",
        "  bpc.\"backtestReturn\" AS bk_ret,",
        "  bpc.\"backtestWinrate\" AS bk_wr,",
        "  bpc.\"backtestTrades\" AS bk_t,",
        "  bpc.\"backtestDrawdown\" AS bk_dd,",
        "  bpc.\"backtestPeriodDays\" AS bk_days",
        "FROM bot_pair_configs bpc",
        "JOIN bots b ON b.id = bpc.\"botId\"",
        "WHERE b.slug = 'trend-following-robot'",
        "ORDER BY bpc.enabled DESC, bpc.\"backtestReturn\" DESC NULLS LAST;",
        "",
        "-- Bot.supportedPairs ham yangilash (faqat aktive pairs)",
        "UPDATE bots",
        f"SET \"supportedPairs\" = ARRAY{KEEP_PAIRS}::text[]",
        "WHERE slug = 'trend-following-robot';",
        "",
        "COMMIT;",
    ])

    out = Path("backtest/update_hema_pair_configs.sql")
    out.write_text("\n".join(sql_parts))
    print(f"SQL yozildi: {out}")
    print(f"Keep pairs (enabled): {KEEP_PAIRS}")
    print(f"Drop pairs (disabled): {DROP_PAIRS}")


if __name__ == "__main__":
    generate_sql()
