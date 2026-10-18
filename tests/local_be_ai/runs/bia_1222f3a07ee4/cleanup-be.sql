-- Only resources recorded for bia_1222f3a07ee4
USE lookddak_be_test;
DROP TRIGGER IF EXISTS local_fail_bia_1222f3a07ee4;
START TRANSACTION;
DELETE FROM notification WHERE target_type='FITTING_JOB' AND target_id IN (2,3,4,5,6) AND member_id IN (2,3,4,5,6);
DELETE FROM fitting_temp_result WHERE fitting_job_id IN (2,3,4,5,6);
DELETE FROM fitting_job_product WHERE fitting_job_id IN (2,3,4,5,6);
DELETE FROM outbox_event WHERE aggregate_id IN (2,3,4,5,6) AND event_type='FITTING_REQUEST';
DELETE FROM fitting_job WHERE id IN (2,3,4,5,6) AND member_id IN (2,3,4,5,6);
DELETE FROM fitting_candidate WHERE member_id IN (2,3,4,5,6) AND product_id IN (2,3);
DELETE FROM member_profile WHERE member_id IN (2,3,4,5,6);
DELETE FROM member WHERE id IN (2,3,4,5,6) AND email LIKE 'bia_1222f3a07ee4-%@local.invalid';
DELETE FROM product WHERE id IN (2,3) AND id NOT IN (SELECT product_id FROM fitting_candidate) AND id NOT IN (SELECT product_id FROM fitting_job_product);
COMMIT;
