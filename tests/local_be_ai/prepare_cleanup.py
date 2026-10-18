"""Write reviewable cleanup SQL for this harness's recorded runs; never execute SQL."""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def main():
    inventory = []
    for directory in sorted((HERE / "runs").iterdir()):
        run = directory.name
        if not re.fullmatch(r"bia_[0-9a-f]{12}", run):
            continue
        created = [json.loads(p.read_text()) for p in directory.glob("*-created.json")]
        jobs = sorted({int(d["created"]["fittingJobId"]) for d in created})
        members = sorted({int(d["member_id"]) for d in created})
        products = sorted({int(d["product_id"]) for d in created})
        lines = ["-- Only resources recorded for " + run, "USE lookddak_be_test;",
                 "DROP TRIGGER IF EXISTS local_fail_" + run + ";"]
        if jobs:
            ids = ",".join(map(str, jobs))
            mids = ",".join(map(str, members))
            pids = ",".join(map(str, products))
            lines += ["START TRANSACTION;",
                      f"DELETE FROM notification WHERE target_type='FITTING_JOB' AND target_id IN ({ids}) AND member_id IN ({mids});",
                      f"DELETE FROM fitting_temp_result WHERE fitting_job_id IN ({ids});",
                      f"DELETE FROM fitting_job_product WHERE fitting_job_id IN ({ids});",
                      f"DELETE FROM outbox_event WHERE aggregate_id IN ({ids}) AND event_type='FITTING_REQUEST';",
                      f"DELETE FROM fitting_job WHERE id IN ({ids}) AND member_id IN ({mids});",
                      f"DELETE FROM fitting_candidate WHERE member_id IN ({mids}) AND product_id IN ({pids});",
                      f"DELETE FROM member_profile WHERE member_id IN ({mids});",
                      f"DELETE FROM member WHERE id IN ({mids}) AND email LIKE '{run}-%@local.invalid';",
                      # Codes are unique per run, but require no remaining references before deletion.
                      f"DELETE FROM product WHERE id IN ({pids}) AND id NOT IN (SELECT product_id FROM fitting_candidate) AND id NOT IN (SELECT product_id FROM fitting_job_product);",
                      "COMMIT;"]
        (directory / "cleanup-be.sql").write_text("\n".join(lines) + "\n")
        (directory / "cleanup-ai.sql").write_text(
            "-- Connect ONLY to 127.0.0.1:15432/lookddak_ai_test\n"
            f'DROP SCHEMA IF EXISTS "{run}" CASCADE;\n')
        inventory.append({"run": run, "job_ids": jobs, "member_ids": members,
                          "product_ids": products, "ai_schema": run,
                          "status": json.loads((directory / "summary.json").read_text()).get("status")
                          if (directory / "summary.json").exists() else "running"})
    (HERE / "resource-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
