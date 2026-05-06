-- Trend bot uchun mavjud pair configs
SELECT bpc."tradingPair", bpc."takeProfit", bpc."stopLoss", bpc.leverage, bpc.enabled,
  bpc."backtestReturn" AS bk_ret, bpc."backtestWinrate" AS bk_wr,
  bpc."backtestTrades" AS bk_t, bpc."backtestDrawdown" AS bk_dd,
  bpc."backtestProfitFactor" AS bk_pf, bpc."backtestPeriodDays" AS bk_days,
  CASE WHEN bpc."customSettings" IS NULL THEN 0 ELSE 1 END AS has_custom
FROM bot_pair_configs bpc
JOIN bots b ON b.id = bpc."botId"
WHERE b.slug = 'trend-following-robot'
ORDER BY bpc."displayOrder" DESC;

-- Boshqa bot'lar uchun pair_configs status
SELECT b.slug,
  COUNT(bpc.id) AS pair_count,
  SUM(CASE WHEN bpc.enabled THEN 1 ELSE 0 END) AS enabled,
  SUM(CASE WHEN bpc."customSettings" IS NOT NULL THEN 1 ELSE 0 END) AS has_custom_settings
FROM bots b
LEFT JOIN bot_pair_configs bpc ON bpc."botId" = b.id
GROUP BY b.slug
ORDER BY b.slug;
