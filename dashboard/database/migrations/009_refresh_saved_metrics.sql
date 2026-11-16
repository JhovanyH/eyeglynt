-- =====================================================================
-- Migration 009 (2026-11-16): recompute the saved results of ended
-- sessions with the current counting rules (every selection counts unless
-- it is marked Unintended). Ended sessions keep a saved copy of their
-- results, so without this they keep the old numbers.
--
-- Safe to run any time. On the Pi it is run by setup_dashboard.sh.
-- =====================================================================
USE eyeglynt;

REPLACE INTO session_metrics
  (session_id, total_selections, keyboard_selections, quickfire_selections,
   most_used_interface, gaze_accuracy_pct, accuracy_basis, accuracy_trials,
   false_positive_pct, unintended_selections, avg_detection_latency_ms,
   latency_events, recalibration_success_pct, recalibration_attempts)
SELECT v.session_id, v.total_selections, v.keyboard_selections, v.quickfire_selections,
       v.most_used_interface, v.gaze_accuracy_pct, v.accuracy_basis, v.accuracy_trials,
       v.false_positive_pct, v.unintended_selections, v.avg_detection_latency_ms,
       v.latency_events, v.recalibration_success_pct, v.recalibration_attempts
FROM v_session_metrics v
JOIN sessions s ON s.session_id = v.session_id
WHERE s.status = 'ended';
