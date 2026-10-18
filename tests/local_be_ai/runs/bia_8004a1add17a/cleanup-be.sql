-- Only resources recorded for bia_8004a1add17a
USE lookddak_be_test;
DROP TRIGGER IF EXISTS local_fail_bia_8004a1add17a;
START TRANSACTION;
DELETE FROM notification WHERE target_type='FITTING_JOB' AND target_id IN (1) AND member_id IN (1);
DELETE FROM fitting_temp_result WHERE fitting_job_id IN (1);
DELETE FROM fitting_job_product WHERE fitting_job_id IN (1);
DELETE FROM outbox_event WHERE aggregate_id IN (1) AND event_type='FITTING_REQUEST';
DELETE FROM fitting_job WHERE id IN (1) AND member_id IN (1);
DELETE FROM fitting_candidate WHERE member_id IN (1) AND product_id IN (1);
DELETE FROM member_profile WHERE member_id IN (1);
DELETE FROM member WHERE id IN (1) AND email LIKE 'bia_8004a1add17a-%@local.invalid';
DELETE FROM product WHERE id IN (1) AND id NOT IN (SELECT product_id FROM fitting_candidate) AND id NOT IN (SELECT product_id FROM fitting_job_product);
COMMIT;
