-- =====================================================================
-- Migration 001 — caregiver vs researcher accounts
-- Run ONCE on a database created before this change (fresh installs from
-- the current schema.sql already include it):
--     mariadb -u root -p eyeglynt < database/migrations/001_account_types.sql
-- =====================================================================
USE eyeglynt;

ALTER TABLE caregivers
  ADD COLUMN IF NOT EXISTS account_type ENUM('caregiver','researcher')
      NOT NULL DEFAULT 'caregiver' AFTER password_hash;

-- Existing accounts become caregivers. To make your own account a researcher:
--   UPDATE caregivers SET account_type='researcher' WHERE username='your_username';
