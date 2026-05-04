BEGIN;
DO $$ DECLARE bot_id text;
BEGIN
  SELECT id INTO bot_id FROM bots WHERE slug='trend-following-robot';
  UPDATE bot_pair_configs SET enabled=false, "updatedAt"=NOW() WHERE "botId"=bot_id AND "tradingPair"='BNBUSDT';
  UPDATE bot_pair_configs SET enabled=false, "updatedAt"=NOW() WHERE "botId"=bot_id AND "tradingPair"='ADAUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 30.0, "chopMaxForEntry": 42.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 4.0, "supertrendMultiplier": 3.0, "emaFast": 12, "emaSlow": 26, "useHtfFilter": false, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3.1_wider_grid", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04", "note": "Wider grid topdi: HTF off + EMA 12/26 + ST 3.0"}, "detailedStats": {"test60d": {"trades": 15, "wins": 11, "losses": 4, "winratePct": 73.33, "netPnlUsd": 129.6, "returnPct": 12.96, "maxDrawdownPct": 14.76, "feesUsd": 16.2}, "full180d": {"trades": 43, "wins": 27, "losses": 16, "winratePct": 62.79, "netPnlUsd": 213.25, "returnPct": 21.33, "maxDrawdownPct": 32.83, "feesUsd": 46.97, "aprPct": 43.25}}}'::jsonb,
    "backtestReturn" = 21.33,
    "backtestWinrate" = 62.79,
    "backtestTrades" = 43,
    "backtestDrawdown" = 32.83,
    "backtestProfitFactor" = 2.47,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'SOLUSDT';
  UPDATE bots SET "supportedPairs" = ARRAY['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'AVAXUSDT', 'DOGEUSDT', 'LINKUSDT']::text[] WHERE id = bot_id;
END $$;

SELECT bpc."tradingPair", bpc.enabled, bpc."backtestReturn" AS bk_ret, bpc."backtestWinrate" AS bk_wr, bpc."backtestTrades" AS bk_t, bpc."backtestDrawdown" AS bk_dd, bpc."backtestPeriodDays" AS bk_days FROM bot_pair_configs bpc JOIN bots b ON b.id = bpc."botId" WHERE b.slug = 'trend-following-robot' ORDER BY bpc.enabled DESC, bpc."backtestReturn" DESC NULLS LAST;
COMMIT;