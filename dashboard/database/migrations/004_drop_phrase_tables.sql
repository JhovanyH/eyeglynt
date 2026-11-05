-- =====================================================================
-- 004: remove the phrase library tables
-- The dashboard no longer manages phrases; the device's Quick-Fire phrases
-- (from the caregiver/therapist interviews) are written into the device code.
-- Safe to run twice. Run it in HeidiSQL (select the eyeglynt database, F9) or:
--     mariadb -u root -p -P 3307 eyeglynt < database/migrations/004_drop_phrase_tables.sql
-- =====================================================================
DROP TABLE IF EXISTS phrases;            -- drop first: it points to phrase_categories
DROP TABLE IF EXISTS phrase_categories;
