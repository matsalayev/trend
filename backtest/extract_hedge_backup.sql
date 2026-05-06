-- Backup'dan original Hedge bot trade'larini extract qilish
-- Backup data-only edi, jadval struktura'lari hozir mavjud DB'da
-- Strategy: vaqtinchalik schema yaratamiz, restore qilamiz, query qilamiz, drop qilamiz

DROP SCHEMA IF EXISTS hedge_audit CASCADE;
CREATE SCHEMA hedge_audit;

-- Faqat trades jadvalini restore qilamiz
CREATE TABLE hedge_audit.trades (LIKE public.trades INCLUDING ALL);
