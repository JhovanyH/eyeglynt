-- =====================================================================
-- Migration 005 (2026-11-08): one session mode, fixed 4 s dwell time,
-- "Ignore" in the activity log, auto-end when the app closes, Stop
-- calibration, and passwords kept as plain text.
--
-- Safe to run more than once. No data is deleted. On the Pi it is run
-- for you by setup_dashboard.sh. By hand:
--     sudo mariadb eyeglynt < ~/eyeglynt/dashboard/database/migrations/005_one_mode_ignore.sql
-- In HeidiSQL: open this file in a Query tab and press F9.
-- Fresh installs from the current schema.sql already include all of this.
-- =====================================================================
USE eyeglynt;

-- 1. Passwords are stored as typed (the team's choice), so the column is
--    renamed. Accounts made before this keep their old scrambled (hashed)
--    password until that person logs in once; the dashboard then saves it
--    as plain text.
ALTER TABLE caregivers CHANGE COLUMN IF EXISTS password_hash password VARCHAR(255) NOT NULL;

-- 2. Every new session is a normal session with the device's 4 s dwell time.
ALTER TABLE sessions ALTER COLUMN mode SET DEFAULT 'normal';
ALTER TABLE sessions ALTER COLUMN dwell_time_ms SET DEFAULT 4000;

-- 3. "Ignore" in the activity log: an ignored selection is left out of
--    every metric (it is not counted as intended or unintended).
ALTER TABLE selections MODIFY intent_status
  ENUM('unreviewed','intended','unintended','ignored') NOT NULL DEFAULT 'unreviewed';

-- 4. The app tells the dashboard when it is closed, so the session ends.
ALTER TABLE device_status MODIFY tracking_state
  ENUM('idle','calibrating','tracking','lost','closed') NOT NULL DEFAULT 'idle';

-- 5. Stop calibration from the dashboard.
ALTER TABLE device_commands MODIFY command
  ENUM('start_session','end_session','recalibrate','set_dwell_time',
       'start_test_block','end_test_block','cancel_recalibration') NOT NULL;
ALTER TABLE system_alerts MODIFY alert_type
  ENUM('lighting_shift','gaze_lock_loss','gaze_lock_restored',
       'recalibration_started','recalibration_success','recalibration_failed',
       'recalibration_cancelled','calibration_required','session_auto_ended',
       'system') NOT NULL;

-- 6. Metrics leave out ignored selections.
CREATE OR REPLACE VIEW v_session_metrics AS
SELECT
  s.session_id,
  s.mode,
  COALESCE(sel.total, 0)                                   AS total_selections,
  COALESCE(sel.kb, 0)                                      AS keyboard_selections,
  COALESCE(sel.qf, 0)                                      AS quickfire_selections,
  CASE WHEN COALESCE(sel.total,0) = 0 THEN NULL
       WHEN sel.kb >= sel.qf THEN 'keyboard' ELSE 'quickfire' END AS most_used_interface,
  CASE
    WHEN s.mode = 'test' AND sel.trials > 0
      THEN ROUND(100 * sel.correct / sel.trials, 2)
    WHEN s.mode = 'normal' AND sel.total > 0
      THEN ROUND(100 * (sel.total - sel.unintended) / sel.total, 2)
  END                                                      AS gaze_accuracy_pct,
  CASE WHEN s.mode = 'test' THEN 'test_trials' ELSE 'caregiver_review' END AS accuracy_basis,
  COALESCE(sel.trials, 0)                                  AS accuracy_trials,
  CASE WHEN COALESCE(sel.total,0) = 0 THEN NULL
       ELSE ROUND(100 * sel.unintended / sel.total, 2) END AS false_positive_pct,
  COALESCE(sel.unintended, 0)                              AS unintended_selections,
  al.avg_latency_ms                                        AS avg_detection_latency_ms,
  COALESCE(al.latency_events, 0)                           AS latency_events,
  CASE WHEN COALESCE(rc.done,0) = 0 THEN NULL
       ELSE ROUND(100 * rc.ok / rc.done, 2) END            AS recalibration_success_pct,
  COALESCE(rc.done, 0)                                     AS recalibration_attempts
FROM sessions s
LEFT JOIN (
  SELECT x.session_id,
         COUNT(*)                                             AS total,
         SUM(x.interface = 'keyboard')                        AS kb,
         SUM(x.interface = 'quickfire')                       AS qf,
         COALESCE(SUM(COALESCE(b.block_type = 'accuracy', 0) AND x.target_element_id IS NOT NULL), 0) AS trials,
         COALESCE(SUM(COALESCE(b.block_type = 'accuracy', 0) AND x.is_correct = 1), 0)  AS correct,
         COALESCE(SUM(x.intent_status = 'unintended'
             OR COALESCE(b.block_type = 'free_look', 0)), 0)  AS unintended
  FROM selections x
  LEFT JOIN test_blocks b ON b.block_id = x.block_id
  WHERE x.intent_status <> 'ignored'
  GROUP BY x.session_id
) sel ON sel.session_id = s.session_id
LEFT JOIN (
  SELECT session_id,
         ROUND(AVG(detection_latency_ms)) AS avg_latency_ms,
         COUNT(*)                         AS latency_events
  FROM system_alerts
  WHERE alert_type IN ('lighting_shift','gaze_lock_loss') AND event_at IS NOT NULL
  GROUP BY session_id
) al ON al.session_id = s.session_id
LEFT JOIN (
  SELECT session_id,
         SUM(success IS NOT NULL) AS done,
         SUM(success = 1)         AS ok
  FROM recalibrations
  GROUP BY session_id
) rc ON rc.session_id = s.session_id;
