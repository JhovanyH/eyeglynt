-- =====================================================================
-- EyeGlynt — database accounts (least privilege)
-- Change BOTH passwords before running, and put them in .env (not in Git).
--   eyeglynt_web    : Flask dashboard
--   eyeglynt_engine : Python gaze-tracking engine (writes events, reads commands)
-- =====================================================================
CREATE USER IF NOT EXISTS 'eyeglynt_web'@'localhost'    IDENTIFIED BY 'CHANGE_ME_web';
CREATE USER IF NOT EXISTS 'eyeglynt_engine'@'localhost' IDENTIFIED BY 'CHANGE_ME_engine';

-- Dashboard: full data access, no schema changes
GRANT SELECT, INSERT, UPDATE, DELETE ON eyeglynt.* TO 'eyeglynt_web'@'localhost';

-- Gaze engine: read config/commands, write events
GRANT SELECT ON eyeglynt.sessions             TO 'eyeglynt_engine'@'localhost';
GRANT SELECT ON eyeglynt.patients             TO 'eyeglynt_engine'@'localhost';
GRANT SELECT ON eyeglynt.test_blocks          TO 'eyeglynt_engine'@'localhost';
GRANT UPDATE (ended_at) ON eyeglynt.test_blocks TO 'eyeglynt_engine'@'localhost';  -- close block at N trials
GRANT SELECT, INSERT, UPDATE ON eyeglynt.calibration_profiles TO 'eyeglynt_engine'@'localhost';
GRANT SELECT, INSERT         ON eyeglynt.selections           TO 'eyeglynt_engine'@'localhost';
GRANT SELECT, INSERT         ON eyeglynt.system_alerts        TO 'eyeglynt_engine'@'localhost';
GRANT SELECT, INSERT, UPDATE ON eyeglynt.recalibrations       TO 'eyeglynt_engine'@'localhost';
GRANT SELECT, UPDATE         ON eyeglynt.device_commands      TO 'eyeglynt_engine'@'localhost';
GRANT SELECT, INSERT, UPDATE ON eyeglynt.device_status        TO 'eyeglynt_engine'@'localhost';

FLUSH PRIVILEGES;
