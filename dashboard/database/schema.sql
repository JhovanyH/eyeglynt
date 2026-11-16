-- =====================================================================
-- EyeGlynt Caregiver Monitoring Dashboard — MariaDB schema
-- Target: MariaDB 10.11 (default on Raspberry Pi OS Bookworm)
-- Run:    sudo mariadb < database/schema.sql
-- =====================================================================

CREATE DATABASE IF NOT EXISTS eyeglynt
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE eyeglynt;

-- ---------------------------------------------------------------------
-- 1. ACCOUNTS  (Fig. 32 Registration, Fig. 33 Login)
-- ---------------------------------------------------------------------
CREATE TABLE caregivers (
  caregiver_id    INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  first_name      VARCHAR(60)  NOT NULL,
  last_name       VARCHAR(60)  NOT NULL,
  role            ENUM('Family Member','Personal Caregiver','Therapist',
                       'Researcher','Other') NOT NULL,
  contact_number  VARCHAR(20)  NOT NULL,
  username        VARCHAR(40)  NOT NULL,
  password        VARCHAR(255) NOT NULL,          -- stored as typed (team decision, 2026-11-08)
  account_type    ENUM('caregiver','researcher') NOT NULL DEFAULT 'caregiver',
                  -- caregiver: simple summary; researcher: full metrics summary
  is_active       TINYINT(1)   NOT NULL DEFAULT 1,
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_login_at   DATETIME     NULL,
  UNIQUE KEY uq_caregivers_username (username)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 2. PATIENTS (SSMI users) and their calibration profiles
-- ---------------------------------------------------------------------
CREATE TABLE patients (
  patient_id      INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  patient_code    VARCHAR(12)  NOT NULL,           -- e.g. P-001, used in exports/thesis tables (anonymised)
  full_name       VARCHAR(120) NOT NULL,
  care_setting    ENUM('Home','Hospital','Rehabilitation / Therapy Center',
                       'Long-term Care Facility','Other') NULL,   -- set by researchers at enrolment
  created_by      INT UNSIGNED NULL,
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_patients_code (patient_code),
  CONSTRAINT fk_patients_creator FOREIGN KEY (created_by)
    REFERENCES caregivers(caregiver_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Written by the gaze engine after the 5-point calibration (Fig. 38).
CREATE TABLE calibration_profiles (
  profile_id        INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  patient_id        INT UNSIGNED NOT NULL,
  created_at        DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  points_captured   TINYINT UNSIGNED NOT NULL DEFAULT 5,
  model_params      JSON         NOT NULL,          -- regression coefficients, screen size, etc.
  fit_accuracy_pct  DECIMAL(5,2) NOT NULL,          -- must be >= 90 to pass (thesis §1.2)
  passed            TINYINT(1)   AS (fit_accuracy_pct >= 90) PERSISTENT,
  is_active         TINYINT(1)   NOT NULL DEFAULT 0,
  KEY ix_profiles_patient (patient_id, is_active),
  CONSTRAINT fk_profiles_patient FOREIGN KEY (patient_id)
    REFERENCES patients(patient_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 3. SESSIONS  (Fig. 34 Start Session, Fig. 36 Summary)
-- ---------------------------------------------------------------------
CREATE TABLE sessions (
  session_id          INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  session_code        CHAR(14)     NOT NULL,        -- human-readable ID, e.g. S-20261001-03
  patient_id          INT UNSIGNED NOT NULL,
  caregiver_id        INT UNSIGNED NOT NULL,
  care_setting        ENUM('Home','Hospital','Rehabilitation / Therapy Center',
                           'Long-term Care Facility','Other') NOT NULL,
  mode                ENUM('normal','test') NOT NULL DEFAULT 'normal',
  dwell_time_ms       SMALLINT UNSIGNED NOT NULL DEFAULT 4000,   -- picked at Start: 2-6 s, 4 s default
  profile_id          INT UNSIGNED NULL,           -- calibration profile in use
  pre_session_notes   TEXT NULL,
  post_session_notes  TEXT NULL,
  status              ENUM('active','ended') NOT NULL DEFAULT 'active',
  started_at          DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  ended_at            DATETIME(3)  NULL,
  -- Only ONE active session per device: NULLs are ignored by UNIQUE.
  active_lock         TINYINT(1)   AS (IF(status = 'active', 1, NULL)) PERSISTENT,
  UNIQUE KEY uq_sessions_code (session_code),
  UNIQUE KEY uq_one_active_session (active_lock),
  KEY ix_sessions_patient (patient_id, started_at),
  CONSTRAINT chk_dwell CHECK (dwell_time_ms BETWEEN 500 AND 10000),
  CONSTRAINT chk_end   CHECK (ended_at IS NULL OR ended_at >= started_at),
  CONSTRAINT fk_sessions_patient   FOREIGN KEY (patient_id)   REFERENCES patients(patient_id),
  CONSTRAINT fk_sessions_caregiver FOREIGN KEY (caregiver_id) REFERENCES caregivers(caregiver_id),
  CONSTRAINT fk_sessions_profile   FOREIGN KEY (profile_id)
    REFERENCES calibration_profiles(profile_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Test Mode only (no longer used since 2026-11-08; kept for old sessions): groups selections into the thesis test procedures.
--   accuracy  = 30 highlighted-target trials (Gaze Selection Accuracy)
--   free_look = user looks around without intending to select (False Positives)
CREATE TABLE test_blocks (
  block_id      INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  session_id    INT UNSIGNED NOT NULL,
  block_type    ENUM('accuracy','free_look') NOT NULL,
  planned_trials SMALLINT UNSIGNED NULL,           -- e.g. 30 for accuracy
  planned_seconds SMALLINT UNSIGNED NULL,          -- e.g. 120 for free_look
  started_at    DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  ended_at      DATETIME(3) NULL,
  KEY ix_blocks_session (session_id),
  CONSTRAINT fk_blocks_session FOREIGN KEY (session_id)
    REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 4. ACTIVITY LOG  — one row per dwell-time activation (written by gaze engine)
-- ---------------------------------------------------------------------
CREATE TABLE selections (
  selection_id      BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  session_id        INT UNSIGNED NOT NULL,
  block_id          INT UNSIGNED NULL,             -- set only in Test Mode
  selected_at       DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  interface         ENUM('keyboard','quickfire') NOT NULL,
  selection_type    ENUM('key','shorthand','phrase','speak','control','navigation') NOT NULL,  -- navigation: researchers only, not in metrics
  element_id        VARCHAR(64)  NOT NULL,         -- e.g. 'kb:A', 'kb:speak', 'qf:pain:head'
  output_text       VARCHAR(500) NOT NULL,         -- character, word, or full spoken phrase
  category          VARCHAR(40)  NULL,             -- Quick-Fire tab name
  dwell_ms          SMALLINT UNSIGNED NULL,        -- actual dwell measured
  gaze_x            SMALLINT NULL,
  gaze_y            SMALLINT NULL,
  -- Test Mode ground truth (accuracy blocks)
  target_element_id VARCHAR(64)  NULL,
  is_correct        TINYINT(1)   AS (IF(target_element_id IS NULL, NULL,
                                      element_id = target_element_id)) PERSISTENT,
  -- Normal Mode ground truth: caregiver review from the Live log
  intent_status     ENUM('unreviewed','intended','unintended','ignored') NOT NULL DEFAULT 'unreviewed',
  reviewed_by       INT UNSIGNED NULL,
  reviewed_at       DATETIME NULL,
  KEY ix_sel_session_time (session_id, selection_id),   -- polling: WHERE session_id=? AND selection_id > ?
  KEY ix_sel_block (block_id),
  CONSTRAINT fk_sel_session  FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
  CONSTRAINT fk_sel_block    FOREIGN KEY (block_id)   REFERENCES test_blocks(block_id) ON DELETE SET NULL,
  CONSTRAINT fk_sel_reviewer FOREIGN KEY (reviewed_by) REFERENCES caregivers(caregiver_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 5. SYSTEM ALERTS  (Live screen alerts panel; Detection Latency source)
-- ---------------------------------------------------------------------
CREATE TABLE system_alerts (
  alert_id        BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  session_id      INT UNSIGNED NULL,               -- NULL if raised with no session running
  alert_type      ENUM('lighting_shift','gaze_lock_loss','gaze_lock_restored',
                       'recalibration_started','recalibration_success','recalibration_failed',
                       'recalibration_cancelled','calibration_required','session_auto_ended',
                       'system') NOT NULL,
  severity        ENUM('info','warning','critical') NOT NULL DEFAULT 'warning',
  message         VARCHAR(255) NOT NULL,
  event_at        DATETIME(3)  NULL,               -- when the condition began (first affected frame)
  detected_at     DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),  -- when the system raised it
  detection_latency_ms INT AS (TIMESTAMPDIFF(MICROSECOND, event_at, detected_at) DIV 1000) VIRTUAL,
  lux_reading     DECIMAL(7,1) NULL,
  acknowledged_by INT UNSIGNED NULL,
  acknowledged_at DATETIME NULL,
  KEY ix_alerts_session (session_id, alert_id),
  KEY ix_alerts_type (alert_type),
  CONSTRAINT fk_alerts_session FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
  CONSTRAINT fk_alerts_ack     FOREIGN KEY (acknowledged_by) REFERENCES caregivers(caregiver_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 6. RECALIBRATIONS  (Recalibration Success Rate source)
-- ---------------------------------------------------------------------
CREATE TABLE recalibrations (
  recal_id          INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  session_id        INT UNSIGNED NOT NULL,
  trigger_source    ENUM('auto_gaze_loss','auto_lighting','auto_drift',
                         'manual_caregiver','manual_user') NOT NULL,
  trigger_alert_id  BIGINT UNSIGNED NULL,
  started_at        DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  completed_at      DATETIME(3) NULL,
  success           TINYINT(1)  NULL,              -- NULL = still running
  verify_selection_id BIGINT UNSIGNED NULL,        -- first selection after recal (thesis success rule)
  new_profile_id    INT UNSIGNED NULL,
  KEY ix_recal_session (session_id),
  CONSTRAINT fk_recal_session FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
  CONSTRAINT fk_recal_alert   FOREIGN KEY (trigger_alert_id) REFERENCES system_alerts(alert_id) ON DELETE SET NULL,
  CONSTRAINT fk_recal_verify  FOREIGN KEY (verify_selection_id) REFERENCES selections(selection_id) ON DELETE SET NULL,
  CONSTRAINT fk_recal_profile FOREIGN KEY (new_profile_id) REFERENCES calibration_profiles(profile_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 7. DASHBOARD -> DEVICE commands (Session Controls: Recalibrate / End)
--    Gaze engine polls: SELECT ... WHERE status='pending' ORDER BY command_id
-- ---------------------------------------------------------------------
CREATE TABLE device_commands (
  command_id     INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  session_id     INT UNSIGNED NULL,
  command        ENUM('start_session','end_session','recalibrate','set_dwell_time',
                      'start_test_block','end_test_block','cancel_recalibration') NOT NULL,
  payload        JSON NULL,                        -- e.g. {"dwell_time_ms":1800}
  issued_by      INT UNSIGNED NULL,
  issued_at      DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  status         ENUM('pending','acknowledged','done','failed') NOT NULL DEFAULT 'pending',
  handled_at     DATETIME(3) NULL,
  KEY ix_cmd_pending (status, command_id),
  CONSTRAINT fk_cmd_session FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
  CONSTRAINT fk_cmd_issuer  FOREIGN KEY (issued_by)  REFERENCES caregivers(caregiver_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Single-row heartbeat so the dashboard can show "Device online / tracking".
CREATE TABLE device_status (
  device_id        TINYINT UNSIGNED PRIMARY KEY DEFAULT 1,
  last_heartbeat   DATETIME(3) NULL,
  tracking_state   ENUM('idle','calibrating','tracking','lost','closed') NOT NULL DEFAULT 'idle',
  fps              DECIMAL(5,1) NULL,
  current_lux      DECIMAL(7,1) NULL,
  active_profile_id INT UNSIGNED NULL,
  software_version VARCHAR(20) NULL,
  CONSTRAINT chk_single_device CHECK (device_id = 1)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 8. SESSION METRICS
--    v_session_metrics = live calculation (used by Live screen)
--    session_metrics   = frozen snapshot written at End Session (used by Summary + PDF)
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_session_metrics AS
SELECT
  s.session_id,
  s.mode,
  COALESCE(sel.total, 0)                                   AS total_selections,
  COALESCE(sel.kb, 0)                                      AS keyboard_selections,
  COALESCE(sel.qf, 0)                                      AS quickfire_selections,
  CASE WHEN COALESCE(sel.total,0) = 0 THEN NULL
       WHEN sel.kb >= sel.qf THEN 'keyboard' ELSE 'quickfire' END AS most_used_interface,

  -- Gaze Selection Accuracy (%)
  CASE
    WHEN s.mode = 'test' AND sel.trials > 0
      THEN ROUND(100 * sel.correct / sel.trials, 2)
    -- Normal Mode: selections are assumed intended unless the caregiver flags them
    WHEN s.mode = 'normal' AND sel.total > 0
      THEN ROUND(100 * (sel.total - sel.unintended) / sel.total, 2)
  END                                                      AS gaze_accuracy_pct,
  CASE WHEN s.mode = 'test' THEN 'test_trials' ELSE 'caregiver_review' END AS accuracy_basis,
  COALESCE(sel.trials, 0)                                  AS accuracy_trials,

  -- False Positive Rate (%) = unintended activations / all activations
  CASE WHEN COALESCE(sel.total,0) = 0 THEN NULL
       ELSE ROUND(100 * sel.unintended / sel.total, 2) END AS false_positive_pct,
  COALESCE(sel.unintended, 0)                              AS unintended_selections,

  -- Detection Latency (ms)
  al.avg_latency_ms                                        AS avg_detection_latency_ms,
  COALESCE(al.latency_events, 0)                           AS latency_events,

  -- Recalibration Success Rate (%)
  CASE WHEN COALESCE(rc.done,0) = 0 THEN NULL
       ELSE ROUND(100 * rc.ok / rc.done, 2) END            AS recalibration_success_pct,
  COALESCE(rc.done, 0)                                     AS recalibration_attempts
FROM sessions s
LEFT JOIN (
  SELECT x.session_id,
         COUNT(*)                                             AS total,
         SUM(x.interface = 'keyboard')                        AS kb,
         SUM(x.interface = 'quickfire')                       AS qf,
         -- COALESCE: rows outside a test block have b.block_type = NULL,
         -- and NULL inside SUM() would turn the whole metric into NULL.
         COALESCE(SUM(COALESCE(b.block_type = 'accuracy', 0) AND x.target_element_id IS NOT NULL), 0) AS trials,
         COALESCE(SUM(COALESCE(b.block_type = 'accuracy', 0) AND x.is_correct = 1), 0)  AS correct,
         COALESCE(SUM(x.intent_status = 'unintended'
             OR COALESCE(b.block_type = 'free_look', 0)), 0)  AS unintended
  FROM selections x
  LEFT JOIN test_blocks b ON b.block_id = x.block_id
  WHERE x.intent_status <> 'ignored'      -- "Unintended" in the activity log: left out of every metric
    AND x.selection_type <> 'navigation'  -- opening a group, Back, tabs: shown to researchers only
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

CREATE TABLE session_metrics (
  session_id                INT UNSIGNED PRIMARY KEY,
  total_selections          INT UNSIGNED NOT NULL,
  keyboard_selections       INT UNSIGNED NOT NULL,
  quickfire_selections      INT UNSIGNED NOT NULL,
  most_used_interface       ENUM('keyboard','quickfire') NULL,
  gaze_accuracy_pct         DECIMAL(5,2) NULL,
  accuracy_basis            ENUM('test_trials','caregiver_review') NOT NULL,
  accuracy_trials           INT UNSIGNED NOT NULL,
  false_positive_pct        DECIMAL(5,2) NULL,
  unintended_selections     INT UNSIGNED NOT NULL,
  avg_detection_latency_ms  INT NULL,
  latency_events            INT UNSIGNED NOT NULL,
  recalibration_success_pct DECIMAL(5,2) NULL,
  recalibration_attempts    INT UNSIGNED NOT NULL,
  computed_at               DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_metrics_session FOREIGN KEY (session_id)
    REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB;
