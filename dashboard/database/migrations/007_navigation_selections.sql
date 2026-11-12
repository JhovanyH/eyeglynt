-- =====================================================================
-- Migration 007 (2026-11-12): every button the patient selects is saved.
-- Navigation selections (opening a letter group or a Quick Fires category,
-- Back, 123/ABC, Shift, the tabs) get their own type. Only researcher
-- accounts see them, and they are left out of the metrics.
--
-- Safe to run more than once. No data is deleted. On the Pi it is run for
-- you by setup_dashboard.sh. By hand:
--     sudo mariadb eyeglynt < ~/eyeglynt/dashboard/database/migrations/007_navigation_selections.sql
-- In HeidiSQL: open this file in a Query tab and press F9.
-- =====================================================================
USE eyeglynt;

ALTER TABLE selections MODIFY selection_type
  ENUM('key','shorthand','phrase','speak','control','navigation') NOT NULL;

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
    AND x.selection_type <> 'navigation'
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
