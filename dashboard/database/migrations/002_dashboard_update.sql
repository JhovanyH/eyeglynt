-- =====================================================================
-- Migration 002 (2026-11-01): brings an existing EyeGlynt database up to date.
-- Safe to run more than once. Run it ONCE in HeidiSQL (open this file in a
-- Query tab and press F9), or:
--     mariadb -u root -p -P 3307 eyeglynt < database/migrations/002_dashboard_update.sql
-- Fresh installs from the current schema.sql already include all of this.
-- =====================================================================
USE eyeglynt;

-- 1. Caregiver vs researcher accounts (same as migration 001)
ALTER TABLE caregivers
  ADD COLUMN IF NOT EXISTS account_type ENUM('caregiver','researcher')
      NOT NULL DEFAULT 'caregiver' AFTER password_hash;

-- Accounts registered with the Researcher role become researcher accounts
UPDATE caregivers SET account_type = 'researcher' WHERE role = 'Researcher';

-- 2. New role list (old names are mapped to the closest new one)
ALTER TABLE caregivers MODIFY role
  ENUM('Family Caregiver','Professional Caregiver','Speech-Language Therapist','Occupational Therapist',
       'Family Member','Personal Caregiver','Nurse','Therapist','Researcher','Other') NOT NULL;
UPDATE caregivers SET role = 'Family Member'      WHERE role = 'Family Caregiver';
UPDATE caregivers SET role = 'Personal Caregiver' WHERE role = 'Professional Caregiver';
UPDATE caregivers SET role = 'Therapist'          WHERE role IN ('Speech-Language Therapist','Occupational Therapist');
UPDATE caregivers SET role = 'Other'              WHERE role = 'Nurse';
ALTER TABLE caregivers MODIFY role
  ENUM('Family Member','Personal Caregiver','Therapist','Researcher','Other') NOT NULL;

-- 3. New care settings (old names are mapped to the closest new one)
ALTER TABLE sessions MODIFY care_setting
  ENUM('Home','Care Facility','Rehabilitation Center',
       'Hospital','Rehabilitation / Therapy Center','Long-term Care Facility','Community / PDAO Center','Other') NOT NULL;
UPDATE sessions SET care_setting = 'Long-term Care Facility'        WHERE care_setting = 'Care Facility';
UPDATE sessions SET care_setting = 'Rehabilitation / Therapy Center' WHERE care_setting = 'Rehabilitation Center';
UPDATE sessions SET care_setting = 'Other'                          WHERE care_setting = 'Community / PDAO Center';
ALTER TABLE sessions MODIFY care_setting
  ENUM('Home','Hospital','Rehabilitation / Therapy Center','Long-term Care Facility','Other') NOT NULL;

-- 4. Patient condition / diagnosis is no longer collected
ALTER TABLE patients DROP COLUMN IF EXISTS condition_type;
ALTER TABLE patients DROP COLUMN IF EXISTS condition_notes;
