-- userBots aggregate maydonlarini qayta hisoblash
-- Cleanup'dan keyin trades soni o'zgardi, lekin aggregate cache'lar eski
-- (totalPnL, totalTrades, winningTrades, losingTrades, peakBalance, maxDrawdown)

BEGIN;

-- Recompute totalPnL, totalTrades, winningTrades, losingTrades from trades
UPDATE user_bots ub
SET
  "totalPnL" = COALESCE(t_agg.total_pnl, 0),
  "realizedPnL" = COALESCE(t_agg.total_pnl, 0),
  "totalTrades" = COALESCE(t_agg.total_trades, 0),
  "winningTrades" = COALESCE(t_agg.winning_trades, 0),
  "losingTrades" = COALESCE(t_agg.losing_trades, 0),
  "updatedAt" = NOW()
FROM (
  SELECT
    "userBotId",
    SUM(pnl)::numeric(18,8) AS total_pnl,
    COUNT(*) AS total_trades,
    COUNT(*) FILTER (WHERE pnl > 0) AS winning_trades,
    COUNT(*) FILTER (WHERE pnl < 0) AS losing_trades
  FROM trades
  WHERE status = 'CLOSED'
  GROUP BY "userBotId"
) t_agg
WHERE ub.id = t_agg."userBotId";

-- userBots without trades — reset to 0
UPDATE user_bots
SET
  "totalPnL" = 0,
  "realizedPnL" = 0,
  "totalTrades" = 0,
  "winningTrades" = 0,
  "losingTrades" = 0,
  "updatedAt" = NOW()
WHERE id NOT IN (SELECT DISTINCT "userBotId" FROM trades WHERE status='CLOSED');

-- Recompute peakBalance and maxDrawdown (dollar) from cumulative balance curve
-- Run for each user_bot via PL/pgSQL
DO $$
DECLARE
  ub RECORD;
  trade RECORD;
  initial_bal NUMERIC;
  cur_bal NUMERIC;
  peak NUMERIC;
  max_dd NUMERIC;
  max_dd_pct NUMERIC;
BEGIN
  FOR ub IN SELECT id, "allocatedBalance" FROM user_bots LOOP
    initial_bal := COALESCE(ub."allocatedBalance", 0);
    IF initial_bal <= 0 THEN
      CONTINUE;
    END IF;
    cur_bal := initial_bal;
    peak := initial_bal;
    max_dd := 0;
    FOR trade IN
      SELECT pnl FROM trades
      WHERE "userBotId" = ub.id AND status='CLOSED'
      ORDER BY "createdAt" ASC
    LOOP
      cur_bal := cur_bal + COALESCE(trade.pnl, 0);
      IF cur_bal > peak THEN
        peak := cur_bal;
      END IF;
      IF (peak - cur_bal) > max_dd THEN
        max_dd := peak - cur_bal;
      END IF;
    END LOOP;
    max_dd_pct := CASE WHEN peak > 0 THEN max_dd / peak * 100 ELSE 0 END;

    UPDATE user_bots
    SET
      "peakBalance" = peak,
      "maxDrawdown" = max_dd,
      "maxDrawdownPercent" = LEAST(99.99, max_dd_pct),
      "updatedAt" = NOW()
    WHERE id = ub.id;
  END LOOP;
END $$;

-- Verify
SELECT
  ub.id,
  b.name AS bot,
  ub."tradingPair",
  ub."allocatedBalance" AS allocated,
  ub."totalPnL",
  ub."totalTrades",
  ub."winningTrades",
  ub."peakBalance",
  ub."maxDrawdownPercent",
  ROUND((ub."totalPnL" / NULLIF(ub."allocatedBalance", 0) * 100)::numeric, 2) AS return_pct
FROM user_bots ub
JOIN bots b ON b.id = ub."botId"
WHERE ub."totalTrades" > 0
ORDER BY return_pct DESC NULLS LAST;

COMMIT;
