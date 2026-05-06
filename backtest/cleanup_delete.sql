-- Bad trade cleanup — DELETE bad/buggy trades
-- Backup at: /tmp/hema_backup_2026-05-04.sql (250MB)
-- Generated: 2026-05-04

BEGIN;

-- Temp table: bad trade IDs
CREATE TEMP TABLE bad_trade_ids AS
SELECT t.id, t."positionId", t."webhookEventId"
FROM trades t
JOIN user_bots ub ON ub.id = t."userBotId"
JOIN bots b ON b.id = ub."botId"
WHERE
  -- 1) Hedge loop (99 trades)
  (b.slug = 'hedging-grid-robot'
   AND t."createdAt" BETWEEN '2026-05-04 04:00' AND '2026-05-04 05:30')
  -- 2) Hedge martingale big losses (PnL < -30 after Apr 28)
  OR (b.slug = 'hedging-grid-robot'
      AND t.pnl < -30
      AND t."createdAt" >= '2026-04-28'
      AND NOT (t."createdAt" BETWEEN '2026-05-04 04:00' AND '2026-05-04 05:30'))
  -- 3) Sure-fire ETHUSDT bad preset
  OR (b.slug = 'sure-fire-hedging-robot'
      AND t.pair = 'ETHUSDT'
      AND t.pnl < 0)
  -- 4) Trend May 2026 losses (broken strategy in choppy market)
  OR (b.slug = 'trend-following-robot'
      AND t.pnl < 0
      AND t."createdAt" >= '2026-05-01');

SELECT 'TO DELETE: ' || COUNT(*) FROM bad_trade_ids;

-- Step 1: Delete bot_webhook_logs that reference these trades
DELETE FROM bot_webhook_logs
WHERE "eventId" IN (SELECT "webhookEventId" FROM bad_trade_ids WHERE "webhookEventId" IS NOT NULL);

-- Step 2: Delete the trades
DELETE FROM trades WHERE id IN (SELECT id FROM bad_trade_ids);

-- Step 3: Delete orphan positions (positions without trades)
DELETE FROM positions
WHERE id IN (
  SELECT p.id FROM positions p
  LEFT JOIN trades t ON t."positionId" = p.id
  WHERE t.id IS NULL
);

-- Verify
SELECT 'AFTER TOTAL' AS cat, COUNT(*) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee,
  ROUND((SUM(t.pnl) - SUM(t.fee))::numeric, 2) AS net_after
FROM trades t;

SELECT 'PER BOT AFTER' AS cat, b.name,
  COUNT(t.id) AS trades,
  ROUND(SUM(t.pnl)::numeric, 2) AS pnl,
  ROUND(SUM(t.fee)::numeric, 2) AS fee,
  ROUND((SUM(t.pnl) - SUM(t.fee))::numeric, 2) AS net_after
FROM bots b
LEFT JOIN user_bots ub ON ub."botId" = b.id
LEFT JOIN trades t ON t."userBotId" = ub.id
GROUP BY b.id, b.name ORDER BY trades DESC NULLS LAST;

COMMIT;
