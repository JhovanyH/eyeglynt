-- =====================================================================
-- Migration 006 (2026-11-10): the dwell time is picked again at Start
-- session (2, 3, 4 default, 5 or 6 seconds). 6 s was above the old limit
-- of 5 s, so the limit becomes 10 s.
--
-- Safe to run more than once. No data is deleted. On the Pi it is run for
-- you by setup_dashboard.sh. By hand:
--     sudo mariadb eyeglynt < ~/eyeglynt/dashboard/database/migrations/006_dwell_options.sql
-- In HeidiSQL: open this file in a Query tab and press F9.
-- =====================================================================
USE eyeglynt;

ALTER TABLE sessions DROP CONSTRAINT IF EXISTS chk_dwell;
ALTER TABLE sessions ADD CONSTRAINT chk_dwell CHECK (dwell_time_ms BETWEEN 500 AND 10000);
ALTER TABLE sessions ALTER COLUMN dwell_time_ms SET DEFAULT 4000;
