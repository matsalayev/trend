-- 1) HEDGE LOOP (2026-05-04 04:00-05:30)
SELECT '1) HEDGE LOOP' AS cat,
  COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee
FROM trades t JOIN user_bots ub ON ub.id=t."userBotId" JOIN bots b ON b.id=ub."botId"
WHERE b.slug='hedging-grid-robot'
  AND t."createdAt" BETWEEN '2026-05-04 04:00' AND '2026-05-04 05:30';

-- 2) HEDGE MARTINGALE BIG LOSS (PnL < -30, after Apr 28)
SELECT '2) HEDGE MARTINGALE BIG LOSS' AS cat,
  COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee
FROM trades t JOIN user_bots ub ON ub.id=t."userBotId" JOIN bots b ON b.id=ub."botId"
WHERE b.slug='hedging-grid-robot'
  AND t.pnl < -30
  AND t."createdAt" >= '2026-04-28'
  AND NOT (t."createdAt" BETWEEN '2026-05-04 04:00' AND '2026-05-04 05:30');

-- 3) SUREFIRE ETHUSDT bad preset trades
SELECT '3) SUREFIRE ETHUSDT' AS cat,
  COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee
FROM trades t JOIN user_bots ub ON ub.id=t."userBotId" JOIN bots b ON b.id=ub."botId"
WHERE b.slug='sure-fire-hedging-robot' AND t.pair='ETHUSDT' AND t.pnl < 0;

-- 4) TREND May 2026 losses
SELECT '4) TREND MAY-2026 LOSS' AS cat,
  COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee
FROM trades t JOIN user_bots ub ON ub.id=t."userBotId" JOIN bots b ON b.id=ub."botId"
WHERE b.slug='trend-following-robot'
  AND t.pnl < 0
  AND t."createdAt" >= '2026-05-01';

-- 5) HEDGE losses last 7 days (other small bug trades)
SELECT '5) HEDGE last-7d losses' AS cat,
  COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee
FROM trades t JOIN user_bots ub ON ub.id=t."userBotId" JOIN bots b ON b.id=ub."botId"
WHERE b.slug='hedging-grid-robot'
  AND t.pnl < 0
  AND t."createdAt" >= NOW() - INTERVAL '7 days'
  AND NOT (t."createdAt" BETWEEN '2026-05-04 04:00' AND '2026-05-04 05:30');

-- BEFORE total
SELECT 'BEFORE TOTAL' AS cat, COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee
FROM trades t;
