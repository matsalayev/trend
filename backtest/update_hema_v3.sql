BEGIN;
DO $$ DECLARE bot_id text;
BEGIN
  SELECT id INTO bot_id FROM bots WHERE slug='trend-following-robot';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 30.0, "chopMaxForEntry": 50.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 2.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 6, "wins": 2, "losses": 4, "winratePct": 33.33, "netPnlUsd": -60.98, "returnPct": -6.1, "maxDrawdownPct": 10.57, "feesUsd": 10.78, "tradesPerDay": 0.1}, "full180d": {"trades": 33, "wins": 23, "losses": 10, "winratePct": 69.7, "netPnlUsd": 3.28, "returnPct": 0.33, "maxDrawdownPct": 11.6, "feesUsd": 41.27, "tradesPerDay": 0.18, "aprPct": 0.67}, "train120d": {"trades": 27, "winratePct": 77.78, "returnPct": 6.43}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = 0.33,
    "backtestWinrate" = 69.7,
    "backtestTrades" = 33,
    "backtestDrawdown" = 11.6,
    "backtestProfitFactor" = 1.07,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'BTCUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 33.0, "chopMaxForEntry": 42.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 1.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 5, "wins": 5, "losses": 0, "winratePct": 100.0, "netPnlUsd": 54.51, "returnPct": 5.45, "maxDrawdownPct": 8.39, "feesUsd": 7.22, "tradesPerDay": 0.08}, "full180d": {"trades": 17, "wins": 15, "losses": 2, "winratePct": 88.24, "netPnlUsd": 91.69, "returnPct": 9.17, "maxDrawdownPct": 10.24, "feesUsd": 17.99, "tradesPerDay": 0.09, "aprPct": 18.59}, "train120d": {"trades": 12, "winratePct": 83.33, "returnPct": 3.72}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = 9.17,
    "backtestWinrate" = 88.24,
    "backtestTrades" = 17,
    "backtestDrawdown" = 10.24,
    "backtestProfitFactor" = 2.83,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'ETHUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 35.0, "chopMaxForEntry": 50.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 1.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 1, "wins": 0, "losses": 1, "winratePct": 0.0, "netPnlUsd": -49.75, "returnPct": -4.97, "maxDrawdownPct": 6.11, "feesUsd": 1.83, "tradesPerDay": 0.02}, "full180d": {"trades": 9, "wins": 6, "losses": 3, "winratePct": 66.67, "netPnlUsd": -15.94, "returnPct": -1.59, "maxDrawdownPct": 11.81, "feesUsd": 10.8, "tradesPerDay": 0.05, "aprPct": -3.22}, "train120d": {"trades": 8, "winratePct": 75.0, "returnPct": 3.38}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = -1.59,
    "backtestWinrate" = 66.67,
    "backtestTrades" = 9,
    "backtestDrawdown" = 11.81,
    "backtestProfitFactor" = 0.0,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'SOLUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 30.0, "chopMaxForEntry": 50.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 3.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 4, "wins": 3, "losses": 1, "winratePct": 75.0, "netPnlUsd": -35.49, "returnPct": -3.55, "maxDrawdownPct": 6.92, "feesUsd": 5.36, "tradesPerDay": 0.07}, "full180d": {"trades": 19, "wins": 12, "losses": 7, "winratePct": 63.16, "netPnlUsd": -77.06, "returnPct": -7.71, "maxDrawdownPct": 12.88, "feesUsd": 21.56, "tradesPerDay": 0.11, "aprPct": -15.63}, "train120d": {"trades": 15, "winratePct": 66.67, "returnPct": -3.98}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = -7.71,
    "backtestWinrate" = 63.16,
    "backtestTrades" = 19,
    "backtestDrawdown" = 12.88,
    "backtestProfitFactor" = 0.0,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'BNBUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 33.0, "chopMaxForEntry": 50.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 3.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 3, "wins": 3, "losses": 0, "winratePct": 100.0, "netPnlUsd": 16.54, "returnPct": 1.65, "maxDrawdownPct": 4.39, "feesUsd": 3.59, "tradesPerDay": 0.05}, "full180d": {"trades": 15, "wins": 13, "losses": 2, "winratePct": 86.67, "netPnlUsd": 139.83, "returnPct": 13.98, "maxDrawdownPct": 15.8, "feesUsd": 14.36, "tradesPerDay": 0.08, "aprPct": 28.35}, "train120d": {"trades": 12, "winratePct": 83.33, "returnPct": 12.33}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = 13.98,
    "backtestWinrate" = 86.67,
    "backtestTrades" = 15,
    "backtestDrawdown" = 15.8,
    "backtestProfitFactor" = 2.72,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'XRPUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 35.0, "chopMaxForEntry": 60.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 2.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 7, "wins": 5, "losses": 2, "winratePct": 71.43, "netPnlUsd": 185.31, "returnPct": 18.53, "maxDrawdownPct": 20.04, "feesUsd": 12.66, "tradesPerDay": 0.12}, "full180d": {"trades": 23, "wins": 11, "losses": 12, "winratePct": 47.83, "netPnlUsd": 368.96, "returnPct": 36.9, "maxDrawdownPct": 33.24, "feesUsd": 41.33, "tradesPerDay": 0.13, "aprPct": 74.83}, "train120d": {"trades": 16, "winratePct": 37.5, "returnPct": 18.36}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = 36.9,
    "backtestWinrate" = 47.83,
    "backtestTrades" = 23,
    "backtestDrawdown" = 33.24,
    "backtestProfitFactor" = 2.27,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'AVAXUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 35.0, "chopMaxForEntry": 50.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 2.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 2, "wins": 0, "losses": 2, "winratePct": 0.0, "netPnlUsd": -97.52, "returnPct": -9.75, "maxDrawdownPct": 9.87, "feesUsd": 3.55, "tradesPerDay": 0.03}, "full180d": {"trades": 10, "wins": 5, "losses": 5, "winratePct": 50.0, "netPnlUsd": -165.67, "returnPct": -16.57, "maxDrawdownPct": 19.54, "feesUsd": 12.52, "tradesPerDay": 0.06, "aprPct": -33.6}, "train120d": {"trades": 8, "winratePct": 62.5, "returnPct": -6.81}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = -16.57,
    "backtestWinrate" = 50.0,
    "backtestTrades" = 10,
    "backtestDrawdown" = 19.54,
    "backtestProfitFactor" = 0.0,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'ADAUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 35.0, "chopMaxForEntry": 42.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 3.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 5, "wins": 4, "losses": 1, "winratePct": 80.0, "netPnlUsd": 18.98, "returnPct": 1.9, "maxDrawdownPct": 4.92, "feesUsd": 2.72, "tradesPerDay": 0.08}, "full180d": {"trades": 23, "wins": 17, "losses": 6, "winratePct": 73.91, "netPnlUsd": 102.17, "returnPct": 10.22, "maxDrawdownPct": 11.9, "feesUsd": 10.83, "tradesPerDay": 0.13, "aprPct": 20.72}, "train120d": {"trades": 17, "winratePct": 64.71, "returnPct": 7.14}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = 10.22,
    "backtestWinrate" = 73.91,
    "backtestTrades" = 23,
    "backtestDrawdown" = 11.9,
    "backtestProfitFactor" = 2.98,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'DOGEUSDT';
  UPDATE bot_pair_configs SET
    enabled = true,
    "customSettings" = '{"adxThreshold": 35.0, "chopMaxForEntry": 50.0, "slAtrMultiplier": 1.0, "trailingAtrMultiplier": 3.5, "useChoppinessFilter": true, "useAtrSl": true, "useOppositeSignalExit": true, "oppositeSignalRequiresFullConfirm": true, "maxSignalAgeBars": 3, "minNetProfitFeeFactor": 1.0, "maxTradesPerHour": 4, "consecutiveLossesThreshold": 3, "consecutiveLossesCooldownBars": 20, "chopPeriod": 14, "adxPeriod": 14, "atrPeriod": 14, "supertrendPeriod": 10, "supertrendMultiplier": 2.0, "useHtfFilter": true, "htfEmaFast": 21, "htfEmaSlow": 50, "trailingActivationPercent": 1.0, "usePartialTp": true, "partialTp1Percent": 2.0, "partialTp1SizePct": 0.33, "partialTp2Percent": 5.0, "partialTp2SizePct": 0.33, "maxDrawdownPercent": 20.0, "cooldownBarsAfterSl": 5, "_v3WalkForward": {"version": "v3_all_pairs", "trainPeriod": "2025-11-05 -> 2026-03-05 (120 kun)", "testPeriod": "2026-03-05 -> 2026-05-04 (60 kun, OUT-OF-SAMPLE)", "fullPeriod": "2025-11-05 -> 2026-05-04 (180 kun)", "validatedAt": "2026-05-04"}, "detailedStats": {"test60d": {"trades": 0, "wins": 0, "losses": 0, "winratePct": 0.0, "netPnlUsd": 0.0, "returnPct": 0.0, "maxDrawdownPct": 0.0, "feesUsd": 0.0, "tradesPerDay": 0.0}, "full180d": {"trades": 8, "wins": 6, "losses": 2, "winratePct": 75.0, "netPnlUsd": 24.72, "returnPct": 2.47, "maxDrawdownPct": 7.57, "feesUsd": 7.24, "tradesPerDay": 0.04, "aprPct": 5.01}, "train120d": {"trades": 8, "winratePct": 75.0, "returnPct": 2.47}}}'::jsonb,
    leverage = 10,
    "backtestReturn" = 2.47,
    "backtestWinrate" = 75.0,
    "backtestTrades" = 8,
    "backtestDrawdown" = 7.57,
    "backtestProfitFactor" = 1.49,
    "backtestPeriodDays" = 180,
    "backtestUpdatedAt" = NOW(),
    "updatedAt" = NOW()
  WHERE "botId" = bot_id AND "tradingPair" = 'LINKUSDT';
  UPDATE bots SET "supportedPairs" = ARRAY['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT']::text[] WHERE id = bot_id;
END $$;

SELECT bpc."tradingPair", bpc.enabled, bpc."backtestReturn" AS bk_ret, bpc."backtestWinrate" AS bk_wr, bpc."backtestTrades" AS bk_t, bpc."backtestDrawdown" AS bk_dd, bpc."backtestPeriodDays" AS bk_days FROM bot_pair_configs bpc JOIN bots b ON b.id = bpc."botId" WHERE b.slug = 'trend-following-robot' ORDER BY bpc."backtestReturn" DESC NULLS LAST;
COMMIT;