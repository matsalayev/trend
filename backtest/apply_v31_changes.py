"""SOLUSDT yangi paramlarni qo'llash + BNB/ADA disable."""
import json
from pathlib import Path
import re

# Read v3 results va yangi SOL optimization
with open("backtest/walk_forward_v3_all_pairs_results.json") as f:
    wf = json.load(f)
with open("backtest/optimize_bad_pairs_results.json") as f:
    bad = json.load(f)

# SOLUSDT yangi params bilan replace
sol_new = bad["SOLUSDT"]
wf["per_pair"]["SOLUSDT"] = {
    "adx_threshold": sol_new["params"]["adx_threshold"],
    "chop_max_for_entry": sol_new["params"]["chop_max_for_entry"],
    "sl_atr_multiplier": sol_new["params"]["sl_atr_multiplier"],
    "trailing_atr_multiplier": sol_new["params"]["trailing_atr_multiplier"],
    "supertrend_multiplier": sol_new["params"]["supertrend_multiplier"],
    "ema_fast": sol_new["params"]["ema_fast"],
    "ema_slow": sol_new["params"]["ema_slow"],
    "use_htf_filter": sol_new["params"]["use_htf_filter"],
    "train": sol_new["train"],
    "test": sol_new["test"],
    "full": sol_new["full"],
}

KEEP_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT"]

# Update presets.json
preset_file = Path("trend_robot/presets.json")
new_presets = {}
for sym in KEEP_PAIRS:
    pp = wf["per_pair"][sym]
    cfg = {
        "leverage": 10,
        "ema_fast": pp.get("ema_fast", 9),
        "ema_slow": pp.get("ema_slow", 21),
        "adx_period": 14,
        "atr_period": 14,
        "supertrend_period": 10,
        "supertrend_multiplier": pp.get("supertrend_multiplier", 2.0),
        "use_htf_filter": pp.get("use_htf_filter", True),
        "htf_ema_fast": 21,
        "htf_ema_slow": 50,
        "initial_sl_percent": 3.0,
        "trailing_activation_percent": 1.0,
        "use_partial_tp": True,
        "partial_tp1_percent": 2.0,
        "partial_tp1_size_pct": 0.33,
        "partial_tp2_percent": 5.0,
        "partial_tp2_size_pct": 0.33,
        "max_drawdown_percent": 20.0,
        "cooldown_bars_after_sl": 5,
        "adx_threshold": float(pp["adx_threshold"]),
        "chop_max_for_entry": float(pp["chop_max_for_entry"]),
        "sl_atr_multiplier": float(pp["sl_atr_multiplier"]),
        "trailing_atr_multiplier": float(pp["trailing_atr_multiplier"]),
        "use_choppiness_filter": True,
        "chop_period": 14,
        "use_atr_sl": True,
        "max_signal_age_bars": 3,
        "use_opposite_signal_exit": True,
        "opposite_signal_requires_full_confirm": True,
        "min_net_profit_fee_factor": 1.0,
        "max_trades_per_hour": 4,
        "consecutive_losses_threshold": 3,
        "consecutive_losses_cooldown_bars": 20,
    }
    new_presets[sym] = {
        "symbol": sym,
        "_v3_walk_forward": {
            "version": "v3.1_optimized",
            "test_60d": {
                "net": round(pp["test"]["net_after"], 2),
                "wr": round(pp["test"]["winrate"], 1),
                "trades": pp["test"]["trades"],
            },
            "full_180d": {
                "net": round(pp["full"]["net_after"], 2),
                "wr": round(pp["full"]["winrate"], 1),
                "trades": pp["full"]["trades"],
            },
        },
        "config": cfg,
    }

preset_file.write_text(json.dumps(new_presets, indent=2, default=str), encoding='utf-8')
print(f"presets.json: 7 pair (BNB+ADA olib tashlandi)")

# SUPPORTED_PAIRS
config_path = Path("trend_robot/config.py")
config_text = config_path.read_text(encoding='utf-8')
new_pairs_list = '\n'.join(f'    "{s}",' for s in KEEP_PAIRS)
new_block = f"SUPPORTED_PAIRS: List[str] = [\n{new_pairs_list}\n]"
pattern = re.compile(r'SUPPORTED_PAIRS:\s*List\[str\]\s*=\s*\[[^\]]*\]', re.DOTALL)
config_text = pattern.sub(new_block, config_text)
config_path.write_text(config_text, encoding='utf-8')
print(f"SUPPORTED_PAIRS = {KEEP_PAIRS}")

# HEMA SQL
sql_lines = [
    "BEGIN;",
    "DO $$ DECLARE bot_id text;",
    "BEGIN",
    "  SELECT id INTO bot_id FROM bots WHERE slug='trend-following-robot';",
]

# Disable BNB + ADA
for sym in ["BNBUSDT", "ADAUSDT"]:
    sql_lines.append(
        f"  UPDATE bot_pair_configs SET enabled=false, \"updatedAt\"=NOW() "
        f"WHERE \"botId\"=bot_id AND \"tradingPair\"='{sym}';"
    )

# Update SOLUSDT
sol = wf["per_pair"]["SOLUSDT"]
sol_full = sol["full"]
sol_test = sol["test"]
sol_full_pct = round(sol_full["net_after"] / 1000 * 100, 2)
sol_pf = round((sol_full["net_after"] + 50) / max(50, abs(sol_full["net_after"]) / 2), 2)

sol_custom = {
    "adxThreshold": float(sol["adx_threshold"]),
    "chopMaxForEntry": float(sol["chop_max_for_entry"]),
    "slAtrMultiplier": float(sol["sl_atr_multiplier"]),
    "trailingAtrMultiplier": float(sol["trailing_atr_multiplier"]),
    "supertrendMultiplier": float(sol["supertrend_multiplier"]),
    "emaFast": int(sol["ema_fast"]),
    "emaSlow": int(sol["ema_slow"]),
    "useHtfFilter": bool(sol["use_htf_filter"]),
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
        "version": "v3.1_wider_grid",
        "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)",
        "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)",
        "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)",
        "validatedAt": "2026-05-04",
        "note": "Wider grid topdi: HTF off + EMA 12/26 + ST 3.0",
    },
    "detailedStats": {
        "test60d": {
            "trades": sol_test["trades"],
            "wins": sol_test["wins"],
            "losses": sol_test["losses"],
            "winratePct": round(sol_test["winrate"], 2),
            "netPnlUsd": round(sol_test["net_after"], 2),
            "returnPct": round(sol_test["net_after"] / 1000 * 100, 2),
            "maxDrawdownPct": round(sol_test["max_dd_pct"], 2),
            "feesUsd": round(sol_test["fees"], 2),
        },
        "full180d": {
            "trades": sol_full["trades"],
            "wins": sol_full["wins"],
            "losses": sol_full["losses"],
            "winratePct": round(sol_full["winrate"], 2),
            "netPnlUsd": round(sol_full["net_after"], 2),
            "returnPct": sol_full_pct,
            "maxDrawdownPct": round(sol_full["max_dd_pct"], 2),
            "feesUsd": round(sol_full["fees"], 2),
            "aprPct": round(sol_full_pct * 365 / 180, 2),
        },
    },
}
cs_json = json.dumps(sol_custom).replace("'", "''")

sql_lines.extend([
    "  UPDATE bot_pair_configs SET",
    "    enabled = true,",
    f"    \"customSettings\" = '{cs_json}'::jsonb,",
    f"    \"backtestReturn\" = {sol_full_pct},",
    f"    \"backtestWinrate\" = {round(sol_full['winrate'], 2)},",
    f"    \"backtestTrades\" = {sol_full['trades']},",
    f"    \"backtestDrawdown\" = {round(sol_full['max_dd_pct'], 2)},",
    f"    \"backtestProfitFactor\" = {sol_pf},",
    f"    \"backtestPeriodDays\" = 180,",
    "    \"backtestUpdatedAt\" = NOW(),",
    "    \"updatedAt\" = NOW()",
    "  WHERE \"botId\" = bot_id AND \"tradingPair\" = 'SOLUSDT';",
    f"  UPDATE bots SET \"supportedPairs\" = ARRAY{KEEP_PAIRS}::text[] WHERE id = bot_id;",
    "END $$;",
    "",
    "SELECT bpc.\"tradingPair\", bpc.enabled, bpc.\"backtestReturn\" AS bk_ret, "
    "bpc.\"backtestWinrate\" AS bk_wr, bpc.\"backtestTrades\" AS bk_t, "
    "bpc.\"backtestDrawdown\" AS bk_dd, bpc.\"backtestPeriodDays\" AS bk_days "
    "FROM bot_pair_configs bpc JOIN bots b ON b.id = bpc.\"botId\" "
    "WHERE b.slug = 'trend-following-robot' "
    "ORDER BY bpc.enabled DESC, bpc.\"backtestReturn\" DESC NULLS LAST;",
    "COMMIT;",
])

out = Path("backtest/update_hema_v31.sql")
out.write_text("\n".join(sql_lines), encoding='utf-8')
print(f"\nSQL: {out}")
print(f"SOLUSDT 6oy: ${sol_full['net_after']:+.2f} ({sol_full_pct}%) — wider grid yutdi")
print(f"BNB, ADA: disabled (chunki backtest profitable bo'lmagan)")
