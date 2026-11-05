-- =====================================================================
-- Migration 003 (2026-11-02): the care setting belongs to the patient.
-- Researchers set it when adding the patient; caregivers can't change it.
-- Each session still stores the setting it was recorded with.
-- Safe to run more than once. Run it ONCE in HeidiSQL (open this file in a
-- Query tab and press F9), or:
--     mariadb -u root -p -P 3307 eyeglynt < database/migrations/003_patient_care_setting.sql
-- =====================================================================
USE eyeglynt;

ALTER TABLE patients
  ADD COLUMN IF NOT EXISTS care_setting
      ENUM('Home','Hospital','Rehabilitation / Therapy Center','Long-term Care Facility','Other')
      NULL AFTER full_name;

-- Existing patients: copy the setting from their most recent session (if any).
UPDATE patients p
   SET p.care_setting = (SELECT s.care_setting FROM sessions s
                          WHERE s.patient_id = p.patient_id
                          ORDER BY s.session_id DESC LIMIT 1)
 WHERE p.care_setting IS NULL;
