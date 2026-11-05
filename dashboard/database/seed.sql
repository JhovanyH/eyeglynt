-- =====================================================================
-- EyeGlynt — reference data (safe to re-run)
-- Run after schema.sql:  sudo mariadb eyeglynt < database/seed.sql
-- =====================================================================
USE eyeglynt;

INSERT IGNORE INTO device_status (device_id, tracking_state) VALUES (1, 'idle');

