-- HEMA bot_pair_configs yangilash (trend bot v2.1 v2)
-- Walk-forward TEST davri: 2026-03-05 → 2026-05-04 (60 kun)

BEGIN;

-- Trend bot ID olish
DO $$ DECLARE bot_id text;
BEGIN
  SELECT id INTO bot_id FROM bots WHERE slug = 'trend-following-robot';

  -- Disable BTCUSDT
  UPDATE bot_pair_configs
  SET enabled = false,
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'BTCUSDT';

  -- Disable SOLUSDT
  UPDATE bot_pair_configs
  SET enabled = false,
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'SOLUSDT';

  -- Disable BNBUSDT
  UPDATE bot_pair_configs
  SET enabled = false,
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'BNBUSDT';

  -- Disable XRPUSDT
  UPDATE bot_pair_configs
  SET enabled = false,
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'XRPUSDT';

  -- Disable ADAUSDT
  UPDATE bot_pair_configs
  SET enabled = false,
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'ADAUSDT';

  -- Disable LINKUSDT
  UPDATE bot_pair_configs
  SET enabled = false,
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'LINKUSDT';

  -- Enable ETHUSDT with v2.1 v2 params + walk-forward stats
  UPDATE bot_pair_configs
  SET enabled = true,
      "customSettings" = '{"adxThreshold": 30.0, "chopMaxForEntry": 50.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 2.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v21WalkForward": {"version": "v2_frequency_aware", "trainPeriod": "2025-11-05 \u2192 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 \u2192 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "validatedAt": "2026-05-04"}}'::jsonb,
      leverage = 10,
      "backtestReturn" = 5.65,
      "backtestWinrate" = 83.33,
      "backtestTrades" = 12,
      "backtestDrawdown" = 14.14,
      "backtestProfitFactor" = 0.57,
      "backtestPeriodDays" = 60,
      "backtestUpdatedAt" = NOW(),
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'ETHUSDT';

  -- Enable AVAXUSDT with v2.1 v2 params + walk-forward stats
  UPDATE bot_pair_configs
  SET enabled = true,
      "customSettings" = '{"adxThreshold": 33.0, "chopMaxForEntry": 65.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 2.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v21WalkForward": {"version": "v2_frequency_aware", "trainPeriod": "2025-11-05 \u2192 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 \u2192 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "validatedAt": "2026-05-04"}}'::jsonb,
      leverage = 10,
      "backtestReturn" = 20.98,
      "backtestWinrate" = 90.0,
      "backtestTrades" = 10,
      "backtestDrawdown" = 23.5,
      "backtestProfitFactor" = 4.2,
      "backtestPeriodDays" = 60,
      "backtestUpdatedAt" = NOW(),
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'AVAXUSDT';

  -- Enable DOGEUSDT with v2.1 v2 params + walk-forward stats
  UPDATE bot_pair_configs
  SET enabled = true,
      "customSettings" = '{"adxThreshold": 33.0, "chopMaxForEntry": 55.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 2.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v21WalkForward": {"version": "v2_frequency_aware", "trainPeriod": "2025-11-05 \u2192 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 \u2192 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "validatedAt": "2026-05-04"}}'::jsonb,
      leverage = 10,
      "backtestReturn" = 7.5,
      "backtestWinrate" = 87.5,
      "backtestTrades" = 8,
      "backtestDrawdown" = 8.74,
      "backtestProfitFactor" = 1.5,
      "backtestPeriodDays" = 60,
      "backtestUpdatedAt" = NOW(),
      "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'DOGEUSDT';

END $$;

-- Verify
SELECT bpc."tradingPair", bpc.enabled, bpc.leverage,
  bpc."backtestReturn" AS bk_ret,
  bpc."backtestWinrate" AS bk_wr,
  bpc."backtestTrades" AS bk_t,
  bpc."backtestDrawdown" AS bk_dd,
  bpc."backtestPeriodDays" AS bk_days
FROM bot_pair_configs bpc
JOIN bots b ON b.id = bpc."botId"
WHERE b.slug = 'trend-following-robot'
ORDER BY bpc.enabled DESC, bpc."backtestReturn" DESC NULLS LAST;

-- Bot.supportedPairs ham yangilash (faqat aktive pairs)
UPDATE bots
SET "supportedPairs" = ARRAY['ETHUSDT', 'AVAXUSDT', 'DOGEUSDT']::text[]
WHERE slug = 'trend-following-robot';

COMMIT;