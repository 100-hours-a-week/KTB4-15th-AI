-- Only resources recorded for bia_d986e9c43329
USE lookddak_be_test;
DROP TRIGGER IF EXISTS local_fail_bia_d986e9c43329;
START TRANSACTION;
DELETE FROM notification WHERE target_type='FITTING_JOB' AND target_id IN (7,8,9,10,11,12,13) AND member_id IN (7,8,9,10,11,12,13);
DELETE FROM fitting_temp_result WHERE fitting_job_id IN (7,8,9,10,11,12,13);
DELETE FROM fitting_job_product WHERE fitting_job_id IN (7,8,9,10,11,12,13);
DELETE FROM outbox_event WHERE aggregate_id IN (7,8,9,10,11,12,13) AND event_type='FITTING_REQUEST';
DELETE FROM fitting_job WHERE id IN (7,8,9,10,11,12,13) AND member_id IN (7,8,9,10,11,12,13);
DELETE FROM fitting_candidate WHERE member_id IN (7,8,9,10,11,12,13) AND product_id IN (4,5);
DELETE FROM member_profile WHERE member_id IN (7,8,9,10,11,12,13);
DELETE FROM member WHERE id IN (7,8,9,10,11,12,13) AND email LIKE 'bia_d986e9c43329-%@local.invalid';
DELETE FROM product WHERE id IN (4,5) AND id NOT IN (SELECT product_id FROM fitting_candidate) AND id NOT IN (SELECT product_id FROM fitting_job_product);
COMMIT;
