-- HEDGE BOT 100% AUDIT — backup'dan restore qilingan ma'lumot bo'yicha

-- 1. UMUMIY STATISTIKA
SELECT '=== 1. HEDGE UMUMIY ===' AS s;
SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
  SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
  SUM(CASE WHEN pnl = 0 THEN 1 ELSE 0 END) AS zero,
  ROUND(SUM(pnl)::numeric, 2) AS net_pnl,
  ROUND(SUM(fee)::numeric, 2) AS fees,
  ROUND((SUM(pnl) - SUM(fee))::numeric, 2) AS net_after,
  ROUND(MIN(pnl)::numeric, 2) AS worst,
  ROUND(MAX(pnl)::numeric, 2) AS best,
  MIN("createdAt") AS first,
  MAX("createdAt") AS last
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot';

-- 2. PER USER_BOT
SELECT '=== 2. PER USER_BOT ===' AS s;
SELECT
  ub.id,
  ub."tradingPair",
  COUNT(t.id) AS trades,
  SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS wins,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fees,
  MIN(t."createdAt") AS first,
  MAX(t."createdAt") AS last
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot'
GROUP BY ub.id, ub."tradingPair"
ORDER BY trades DESC;

-- 3. SIDE DISTRIBUTION
SELECT '=== 3. SIDE DISTRIBUTION ===' AS s;
SELECT side, COUNT(*) AS trades, ROUND(SUM(pnl)::numeric, 2) AS pnl
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot'
GROUP BY side;

-- 4. PAIR DISTRIBUTION
SELECT '=== 4. PAIR ===' AS s;
SELECT pair, COUNT(*), ROUND(SUM(pnl)::numeric, 2) AS pnl, ROUND(SUM(fee)::numeric, 2) AS fees
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot'
GROUP BY pair ORDER BY 2 DESC;

-- 5. HOURLY PATTERN — loop topish
SELECT '=== 5. HOURLY (top 10) ===' AS s;
SELECT
  TO_CHAR(t."createdAt", 'YYYY-MM-DD HH24') AS hour,
  COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fees,
  ROUND(AVG(EXTRACT(EPOCH FROM (t."closedAt" - t."openedAt")))::numeric, 0) AS avg_duration_sec
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot'
GROUP BY 1 ORDER BY 2 DESC LIMIT 10;

-- 6. INTER-TRADE INTERVAL ANALIZI (loop indicator)
SELECT '=== 6. INTER-TRADE INTERVAL DAVRI ===' AS s;
WITH t_sorted AS (
  SELECT t.*, LAG(t."createdAt") OVER (PARTITION BY t."userBotId" ORDER BY t."createdAt") AS prev_ts
  FROM audit_trades_backup t
  JOIN user_bots ub ON ub.id = t."userBotId"
  JOIN bots b ON b.id = ub."botId"
  WHERE b.slug = 'hedging-grid-robot'
)
SELECT
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("createdAt" - prev_ts)) < 60) AS less_60s,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("createdAt" - prev_ts)) BETWEEN 60 AND 120) AS s60_120,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("createdAt" - prev_ts)) BETWEEN 120 AND 600) AS s2_10min,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("createdAt" - prev_ts)) BETWEEN 600 AND 3600) AS s10_60min,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("createdAt" - prev_ts)) > 3600) AS over_1h
FROM t_sorted
WHERE prev_ts IS NOT NULL;

-- 7. TRADE DURATION (qancha vaqt ochiq qoldi?)
SELECT '=== 7. TRADE DURATION (qancha vaqt ochiq) ===' AS s;
SELECT
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("closedAt" - "openedAt")) < 60) AS less_1min,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("closedAt" - "openedAt")) BETWEEN 60 AND 300) AS m1_5,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("closedAt" - "openedAt")) BETWEEN 300 AND 3600) AS m5_60,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("closedAt" - "openedAt")) BETWEEN 3600 AND 86400) AS h1_24,
  COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM ("closedAt" - "openedAt")) > 86400) AS over_1d
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot' AND t."closedAt" IS NOT NULL;

-- 8. PnL DISTRIBUTION
SELECT '=== 8. PnL DISTRIBUTION ===' AS s;
SELECT
  COUNT(*) FILTER (WHERE pnl > 100) AS huge_win,
  COUNT(*) FILTER (WHERE pnl BETWEEN 10 AND 100) AS big_win,
  COUNT(*) FILTER (WHERE pnl BETWEEN 0.01 AND 10) AS small_win,
  COUNT(*) FILTER (WHERE pnl BETWEEN -0.01 AND 0.01) AS zero,
  COUNT(*) FILTER (WHERE pnl BETWEEN -10 AND -0.01) AS small_loss,
  COUNT(*) FILTER (WHERE pnl BETWEEN -100 AND -10) AS big_loss,
  COUNT(*) FILTER (WHERE pnl < -100) AS huge_loss
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot';

-- 9. LOT SIZE DISTRIBUTION (martingale topish)
SELECT '=== 9. LOT SIZE DISTRIBUTION ===' AS s;
SELECT
  CASE
    WHEN amount <= 0.05 THEN 'tiny <0.05'
    WHEN amount <= 0.5 THEN 'small 0.05-0.5'
    WHEN amount <= 2 THEN 'medium 0.5-2'
    WHEN amount <= 10 THEN 'large 2-10'
    ELSE 'huge >10'
  END AS lot_size,
  COUNT(*) AS trades,
  ROUND(SUM(pnl)::numeric, 2) AS pnl,
  ROUND(MIN(amount)::numeric, 4) AS min_lot,
  ROUND(MAX(amount)::numeric, 4) AS max_lot
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot'
GROUP BY 1
ORDER BY MIN(amount);

-- 10. FEE/PNL RATIO — fee-bleed indicator
SELECT '=== 10. FEE-BLEED INDICATOR (fee/|pnl|) ===' AS s;
SELECT
  COUNT(*) FILTER (WHERE ABS(pnl) > 0.01 AND fee > ABS(pnl) * 0.5) AS fee_high,
  COUNT(*) FILTER (WHERE ABS(pnl) > 0.01 AND fee > ABS(pnl) * 0.9) AS fee_almost_pnl,
  COUNT(*) FILTER (WHERE ABS(pnl) > 0.01 AND fee >= ABS(pnl)) AS fee_eq_pnl,
  ROUND(AVG(fee / NULLIF(ABS(pnl), 0))::numeric, 3) AS avg_fee_to_pnl_ratio
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot' AND ABS(pnl) > 0.01;

-- 11. RECENT 30 TRADES (eng so'nggi pattern)
SELECT '=== 11. OXIRGI 30 TRADE ===' AS s;
SELECT
  TO_CHAR(t."createdAt", 'MM-DD HH24:MI:SS') AS dt,
  t.pair,
  t.side,
  ROUND(t.amount::numeric, 4) AS amt,
  ROUND(t.price::numeric, 2) AS px,
  ROUND(t."exitPrice"::numeric, 2) AS exit_p,
  ROUND(t.pnl::numeric, 2) AS pnl,
  ROUND(t.fee::numeric, 2) AS fee
FROM audit_trades_backup t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE b.slug = 'hedging-grid-robot'
ORDER BY t."createdAt" DESC LIMIT 30;
