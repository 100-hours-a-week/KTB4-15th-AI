"""Run from AI root: .venv/bin/python tests/local_be_ai/run.py.

No dotenv, migration runner, real cloud credentials, commits, or existing-data cleanup.
Run under localhost.sb for OS-level network isolation of Python AND the Java child.
"""
import dataclasses
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# Explicitly replace the environment BEFORE importing application modules. Never load .env.
safe_env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
os.environ.clear()
os.environ.update(safe_env)
os.environ.update(AWS_EC2_METADATA_DISABLED="true", AWS_ACCESS_KEY_ID="local-only",
                  AWS_SECRET_ACCESS_KEY="local-only", AUTH_DISABLED="1",
                  RUNWARE_VTON_API_KEY="local-mock-only", RUNWARE_LLM_API_KEY="local-mock-only")

import boto3
import psycopg
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.config import settings
from tests.v2_integration_harness import (
    V2Env, QUEUE_URLS, REGION, BUCKET, WEBHOOK_TOKEN, WEBHOOK_URL,
    USER_IMAGE_KEY, PRODUCTS, RUNWARE_MODULES, Stack, close_env, post_webhook, COMMENT, TITLE,
    wait_until,
)

HERE = Path(__file__).resolve().parent
RUN = "bia_" + uuid.uuid4().hex[:12]
SCHEMA = RUN
ART = HERE / "runs" / RUN
ART.mkdir(parents=True)
DB = "postgresql://ai_test:local_ai_password@127.0.0.1:15432/lookddak_ai_test"
BE = Path("/Users/ijaein/Documents/lookddak-be-local")
mp = pytest.MonkeyPatch()
env = V2Env(mp, DB, SCHEMA)
original_urlopen = urlopen
checks = []
job_ids = []


def dump(name, data):
    (ART / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def check(name, condition, evidence=None):
    checks.append({"check": name, "passed": bool(condition), "evidence": evidence})
    dump("checks.json", checks)
    assert condition, name
    print("PASS", name, flush=True)


def rpc(port, operation, payload=None):
    req = Request(f"http://127.0.0.1:{port}/{operation}",
                  data=json.dumps(payload or {}).encode(), headers={"Content-Type": "application/json"})
    try:
        with original_urlopen(req, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        if port == 18762:
            raise RuntimeError(f"local BE {operation}: {error.read().decode()}") from error
        raise


class QueueHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        try:
            args = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = self.path.lstrip("/")
            assert method in {"send_message", "receive_message", "delete_message", "change_message_visibility"}
            result = getattr(env.sqs.client(), method)(**args)
            code = 200
        except ClientError as error:
            result, code = {"client_error": error.response}, 409
        except ReadTimeoutError:
            result, code = {"stored_timeout": True}, 408
        except Exception as error:
            result, code = {"error": type(error).__name__}, 500
        data = json.dumps(result).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


class RemoteSqsClient:
    """Injected boto3 boundary. BOTH services use the same HTTP queue server."""
    def __getattr__(self, operation):
        def call(**kwargs):
            try:
                return rpc(18761, operation, kwargs)
            except HTTPError as error:
                body = json.load(error)
                if "client_error" in body:
                    raise ClientError(body["client_error"], operation) from error
                if "stored_timeout" in body:
                    raise ReadTimeoutError(endpoint_url="http://127.0.0.1:18761") from error
                raise
        return call


def public_snapshot(connection):
    tables = [r[0] for r in connection.execute(
        "select tablename from pg_tables where schemaname='public' order by tablename")]
    return {t: sorted(r[0] for r in connection.execute(f'SELECT row_to_json(t)::text FROM public."{t}" t'))
            for t in tables}


def configure():
    values = {"AWS_REGION": REGION, "S3_BUCKET": BUCKET, "WEBHOOK_TOKEN": WEBHOOK_TOKEN,
              "VIRTUAL_FITTING_RUNWARE_WEBHOOK_URL": WEBHOOK_URL, "CHECKPOINT_DSN": "",
              "BE_FITTING_REQUEST_QUEUE_URL": QUEUE_URLS["be-fitting-request"],
              "AI_FITTING_RESPONSE_QUEUE_URL": QUEUE_URLS["ai-fitting-response"],
              "BE_FITTING_RESPONSE_QUEUE_URL": QUEUE_URLS["be-fitting-response"],
              "VIRTUAL_FITTING_PRUNA_MIN_SEND_INTERVAL_SECONDS": 0.0}
    for name, value in values.items():
        mp.setattr(settings, name, value)

    def boto(service, *args, **kwargs):
        if service == "sqs": return RemoteSqsClient()
        if service == "s3": return env.s3.client()
        raise AssertionError("unexpected AWS boundary: " + service)
    mp.setattr(boto3, "client", boto)
    for name in RUNWARE_MODULES:
        mp.setattr(importlib.import_module(name), "urlopen", env.runware)
    env.s3.store(USER_IMAGE_KEY, content_type="image/png")
    env.new_process_pool()


def be(operation, **kwargs):
    return rpc(18762, operation, kwargs)


def snapshot(label, job):
    result = {"be": be("snapshot", job_id=job), "ai": dataclasses.asdict(env.job(job)),
              "queue_live": {name: len(env.sqs.live(name)) for name in QUEUE_URLS}}
    dump(label + ".json", result)
    return result


def create(label, code):
    data = be("create", label=label, code=code)
    job = data["created"]["fittingJobId"]
    job_ids.append(job)
    dump(label + "-created.json", data)
    check(label + ": real BE transaction saved PENDING job + outbox",
          data["created"]["status"] == "PENDING" and data["outbox"]["status"] == "PENDING")
    return job, json.loads(data["outbox"]["payload"])


def submit(stack, job, payload):
    be("publish")
    check(f"{job}: BE outbox delivered exact payload to shared queue",
          payload in env.sqs.bodies("be-fitting-request"))
    stack.request_once()
    row = env.job(job)
    check(f"{job}: requested_at, fitting_job_id, product codes preserved",
          row.fitting_job_id == job and row.requested_at == datetime.fromisoformat(payload["requested_at"].replace("Z", "+00:00"))
          and list(row.products) == [p["product_code"] for p in payload["products"]])
    return row


def finish(stack, job, label, expected="COMPLETED"):
    stack.postprocess_once()
    stack.publish_once()
    be("consume")
    result = snapshot(label, job)
    b, a = result["be"], result["ai"]
    check(label + ": BE status query + durable result + single notification",
          b["query"]["status"] == expected and b["job"]["status"] == expected
          and len(b["notifications"]) == 1 and len(b["results"]) == (1 if expected == "COMPLETED" else 0), result)
    check(label + ": BE request counter increments once", b["member"]["fitting_request_count"] == 1)
    check(label + ": result contract matches",
          (b["results"][0]["result_image_key"] == a["result_payload"]["result_image_key"]
           and b["results"][0]["outfit_name"] == a["result_payload"]["llm_title"]
           and b["results"][0]["ai_comment"] == a["result_payload"]["llm_comment"])
          if expected == "COMPLETED" else
          (b["job"]["failure_code"] == a["result_payload"]["error_code"] and
           b["job"]["failure_message"] == a["result_payload"]["error_message"]))
    return result


def scenarios(client, codes):
    stack = Stack(env).mark_running()
    # Success, duplicate deliveries BEFORE and AFTER completion, real queue retention on DB error.
    job, payload = create("success", codes[0])
    row = submit(stack, job, payload)
    check("success: AI WAITING_PRUNA while BE remains PENDING",
          row.status.value == "WAITING_PRUNA" and be("snapshot", job_id=job)["query"]["status"] == "PENDING")
    rpc(18761, "send_message", {"QueueUrl": QUEUE_URLS["be-fitting-request"], "MessageBody": json.dumps(payload)})
    stack.request_once()
    check("duplicate request during generation: only one paid async submit", env.runware.count("async_submit") == 1)
    body = env.runware.complete(row.task_uuid)
    check("success: mock stores image and delivers actual webhook", post_webhook(client, body).status_code == 200)
    internal = env.sqs.bodies("ai-fitting-response")[-1]
    rpc(18761, "send_message", {"QueueUrl": QUEUE_URLS["ai-fitting-response"], "MessageBody": json.dumps(internal)})
    stack.postprocess_once()
    requests = env.runware.llm_requests
    check("generated image -> comment -> title using exact comment",
          len(requests) == 2 and COMMENT in requests[1]["messages"][1]["content"]
          and row.async_image_key in json.dumps(requests[0])
          and env.runware.count("llm_comment") == env.runware.count("llm_title") == 1)
    stack.publish_once()
    be("fail-save", job_id=job)
    be("consume")
    failed_save = snapshot("success-save-failure", job)
    live = env.sqs.live("be-fitting-response")
    check("BE actual MySQL INSERT failure rolls back and retains queue message",
          failed_save["be"]["job"]["status"] == "PENDING" and not failed_save["be"]["results"]
          and not failed_save["be"]["notifications"] and len(live) == 1 and live[0].receive_count == 1)
    be("restore-save", job_id=job)
    time.sleep(2.2)  # real visibility timeout, not a fake expiry
    final = finish(stack, job, "success")
    check("BE recovers on actual redelivery", not env.sqs.live("be-fitting-response") and live[0].receive_count == 2)
    baseline = dataclasses.asdict(env.job(job))
    result_payload = baseline["result_payload"]
    for name, message in (("be-fitting-request", payload), ("be-fitting-response", result_payload),
                          ("be-fitting-response", result_payload), ("ai-fitting-response", internal)):
        rpc(18761, "send_message", {"QueueUrl": QUEUE_URLS[name], "MessageBody": json.dumps(message)})
    stack.request_once(); stack.postprocess_once(); stack.publish_once(); be("consume")
    check("duplicate request/result/internal messages preserve AI counters and records",
          dataclasses.asdict(env.job(job)) == baseline)
    duplicate = snapshot("success-duplicates", job)
    check("duplicate results preserve one BE result + notification + member counter",
          len(duplicate["be"]["results"]) == len(duplicate["be"]["notifications"]) == 1
          and duplicate["be"]["member"]["fitting_request_count"] == 1)

    # Product exists in BE but is deliberately absent from AI catalog.
    job, payload = create("business-failure", codes[1])
    submit(stack, job, payload)
    failure = finish(stack, job, "business-failure", "FAILED")
    check("business failure error code is PRODUCT_NOT_FOUND", failure["be"]["job"]["failure_code"] == "PRODUCT_NOT_FOUND")

    # Confirmed retryable provider rejection routes to real fallback implementation.
    env.runware.async_modes.append(("errors", [{"code": "providerRateLimitExceeded", "message": "local"}]))
    job, payload = create("fallback", codes[0])
    submit(stack, job, payload)
    before = env.runware.count("sync_generate")
    final = finish(stack, job, "fallback")
    check("confirmed fallback generated once and BE stores fallback key",
          env.runware.count("sync_generate") == before + 1
          and final["ai"]["fallback_attempt_count"] == 1
          and final["ai"]["result_image_key"] == final["ai"]["fallback_image_key"]
          and final["ai"]["task_uuid"] != final["ai"]["fallback_task_uuid"])
    rpc(18761,"send_message",{"QueueUrl":QUEUE_URLS["be-fitting-request"],"MessageBody":json.dumps(payload)})
    before_fallback = dataclasses.asdict(env.job(job))
    stack.request_once(); Stack(env).mark_running().postprocess_once()
    check("fallback duplicate preserves generation count, task IDs and prior counters",
          env.runware.count("sync_generate") == before + 1 and dataclasses.asdict(env.job(job)) == before_fallback)

    # Failed outbox publish, failed internal publish then webhook retry, failed result publish then retry.
    job, payload = create("queue-retries", codes[0])
    env.sqs.fault("be-fitting-request", "send", "throttled")
    be("publish")
    failed_outbox = be("snapshot", job_id=job)
    dump("outbox-failure.json", failed_outbox)
    check("BE outbox publish failure retains event + retry counter",
          failed_outbox["outbox"]["status"] == "PENDING" and failed_outbox["outbox"]["retry_count"] == 1)
    time.sleep(2.2)
    row = submit(stack, job, payload)
    env.sqs.fault("ai-fitting-response", "send", "throttled", "stored_timeout")
    body = env.runware.complete(row.task_uuid)
    response = post_webhook(client, body)
    check("internal queue confirmed publish failure keeps durable next action and no queue message",
          response.status_code == 503 and env.job(job).next_action.value == "POSTPROCESS"
          and not env.sqs.live("ai-fitting-response"),response.json())
    response = post_webhook(client, body)
    check("internal queue ambiguous publish reports retryable 503", response.status_code == 503, response.json())
    check("webhook redelivery recovers internal publish", post_webhook(client, body).status_code == 200)
    calls = env.runware.count("llm_comment")
    stack.postprocess_once()
    check("duplicate internal messages after publish timeout run LLM once", env.runware.count("llm_comment") == calls + 1)
    env.sqs.fault("be-fitting-response", "send", "throttled")
    stack.publish_once()
    failed_publish = snapshot("result-publish-failure", job)
    check("result publish failure preserves saved result + PENDING + attempt 1",
          failed_publish["ai"]["result_message_status"] == "PENDING"
          and failed_publish["ai"]["result_message_attempt_count"] == 1
          and failed_publish["ai"]["result_payload"]["status"] == "COMPLETED")
    time.sleep(5.2)
    final = finish(stack, job, "queue-retries")
    check("result publish retry uses same payload and increments existing counter",
          final["ai"]["result_message_attempt_count"] == 2
          and final["ai"]["result_payload"] == failed_publish["ai"]["result_payload"])

    # Title failures cause 3 comment/title pairs, then retain image success with default text.
    job, payload = create("llm-limit", codes[0])
    row = submit(stack, job, payload)
    before_calls = (env.runware.count("llm_comment"),env.runware.count("llm_title"))
    env.runware.title_modes.extend([("http",500)] * 3)
    check("LLM limit webhook accepted",post_webhook(client,env.runware.complete(row.task_uuid)).status_code == 200)
    final = finish(stack,job,"llm-limit")
    check("LLM ceiling is 3 pairs (6 HTTP calls), with successful image preserved",
          final["ai"]["llm_generation_attempt_count"] == 3
          and env.runware.count("llm_comment")-before_calls[0] == 3
          and env.runware.count("llm_title")-before_calls[1] == 3
          and final["ai"]["result_image_key"] == row.async_image_key
          and final["ai"]["result_payload"]["llm_title"] != TITLE)
    current_calls = len(env.runware.llm_requests)
    internal = env.sqs.bodies("ai-fitting-response")[-1]
    rpc(18761,"send_message",{"QueueUrl":QUEUE_URLS["ai-fitting-response"],"MessageBody":json.dumps(internal)})
    env.new_process_pool()
    Stack(env).mark_running().postprocess_once()
    check("LLM exhausted counter survives pool/worker restart; no seventh call",
          len(env.runware.llm_requests)==current_calls and env.job(job).llm_generation_attempt_count==3)

    # Result publisher ceiling: no fourth send even on explicit subsequent cycles / reconstructed stack.
    job, payload = create("result-limit", codes[1])
    submit(stack, job, payload)
    env.sqs.fault("be-fitting-response", "send", "throttled", "throttled", "throttled")
    stack.publish_once(); time.sleep(5.2); stack.publish_once(); time.sleep(30.2); stack.publish_once()
    before = dataclasses.asdict(env.job(job))
    env.new_process_pool()
    restarted = Stack(env).mark_running()
    restarted.publish_once(); restarted.publish_once()
    after = snapshot("result-limit", job)
    check("result send ceiling 3 survives reconstructed workers without counter reset",
          before == dataclasses.asdict(env.job(job)) and after["ai"]["result_message_attempt_count"] == 3
          and after["ai"]["result_message_status"] == "RETRY_EXHAUSTED"
          and after["be"]["job"]["status"] == "PENDING")
    all_rows = env.sql("SELECT fitting_job_id,s3_check_attempt_count,llm_generation_attempt_count,fallback_attempt_count,result_message_attempt_count FROM virtual_fitting_async_jobs ORDER BY fitting_job_id")
    check("all jobs obey generation/LLM/result attempt ceilings", all(s3 <= 2 and llm <= 3 and fb <= 1 and send <= 3 for _, s3, llm, fb, send in all_rows), all_rows)


def automatic_scenario(codes):
    # Independent real BE timer + real AI lifespan tasks. No direct cycle calls here.
    for name in ("VIRTUAL_FITTING_REQUEST_WORKER_ENABLED","VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED",
                 "VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED","VIRTUAL_FITTING_WATCHDOG_ENABLED"):
        mp.setattr(settings,name,True)
    mp.setattr(settings,"VIRTUAL_FITTING_RESULT_PUBLISHER_POLL_SECONDS",.05)
    mp.setattr(settings,"VIRTUAL_FITTING_WATCHDOG_INTERVAL_SECONDS",.2)
    from app.main import app
    be("auto-start")
    try:
        with TestClient(app) as client:
            job,payload=create("automatic-lifespan",codes[0])
            row=wait_until(lambda:(r:=env.job(job)) and r.runware_accepted_at and r,timeout=30,what="automatic AI acceptance")
            check("automatic: exact BE metadata preserved",row.requested_at==datetime.fromisoformat(payload["requested_at"].replace("Z","+00:00")) and row.products==(codes[0],))
            body=env.runware.complete(row.task_uuid)
            check("automatic: mock completion webhook accepted",post_webhook(client,body).status_code==200)
            wait_until(lambda:be("snapshot",job_id=job)["query"]["status"]=="COMPLETED",timeout=30,what="automatic BE completion")
            result=snapshot("automatic-lifespan",job)
            check("automatic: Outbox scheduler -> AI lifespan workers -> BE consumer -> status query",
                  result["ai"]["result_message_status"]=="PUBLISHED"
                  and result["be"]["outbox"]["status"]=="PUBLISHED"
                  and len(result["be"]["results"])==len(result["be"]["notifications"])==1)
            check("automatic: exact result key/title/comment persisted in BE",
                  result["be"]["results"][0]["result_image_key"]==result["ai"]["result_image_key"]
                  and result["be"]["results"][0]["outfit_name"]==TITLE
                  and result["be"]["results"][0]["ai_comment"]==COMMENT)
    finally:
        be("auto-stop")
    costs=env.sql("SELECT task_uuid::text,fitting_job_id,cost FROM virtual_fitting_task_costs ORDER BY fitting_job_id")
    usage=env.sql("SELECT recorded_amount,legacy_amount,reconciled_amount,amount FROM usage_amount")
    from decimal import Decimal
    check("actual task-cost and usage repositories: unique tasks, exact costs, no duplicate billing",
          len(costs)==len({c[0] for c in costs}) and sum(c[2] for c in costs)==sum(u[0] for u in usage)
          and len(costs)==5 and sum(c[2] for c in costs)==Decimal("0.107"),{"costs":costs,"usage":usage})


def main():
    process = server = admin = None
    status = "failed"
    before = None
    try:
        # OS-level egress guard is mandatory, including libpq/native Java networking.
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=1).close()
        except PermissionError:
            pass
        else:
            raise RuntimeError("run under sandbox-exec localhost.sb; egress guard absent")
        check("OS sandbox blocks nonlocal network", True)
        admin = psycopg.connect(DB, autocommit=True)
        before = public_snapshot(admin)
        dump("public-before.json", before)
        admin.execute(f'CREATE SCHEMA "{SCHEMA}"')
        for table in before:
            admin.execute(f'CREATE TABLE "{SCHEMA}"."{table}" (LIKE public."{table}" INCLUDING ALL)')
        # Required singleton comes from existing prepared schema, no migration scripts are applied.
        admin.execute(f'INSERT INTO "{SCHEMA}".virtual_fitting_pruna_limiter SELECT * FROM public.virtual_fitting_pruna_limiter')
        configure()
        code = str(800000000000 + int(uuid.uuid4().hex[:7],16))
        codes = (code, str(int(code)+1))
        env.sql("INSERT INTO products (product_code,id,product_name,price,detail_url,image_url,main_category,sub_category,is_sold_out,crawled_at,description_summary) VALUES (%s,1,'로컬상의',10000,'http://127.0.0.1/product','http://127.0.0.1/product.jpg','상의','반소매 티셔츠',false,now(),'네이비 반소매 티셔츠')", (int(code),))
        server = ThreadingHTTPServer(("127.0.0.1",18761),QueueHandler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        manifest = {"run":RUN,"ai_schema":SCHEMA,"be_db":"lookddak_be_test","be_repo":str(BE),
                    "ai_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    "be_commit":subprocess.check_output(["git","-C",str(BE),"rev-parse","HEAD"],text=True).strip(),
                    "product_codes":codes,"ports":[18761,18762],"queue_urls":QUEUE_URLS,
                    "mock_boundaries":["SQS clients injected into both services; shared HTTP queue server", "S3 boto3 and BE presign", "Runware/LLM urlopen"],
                    "execution":"fault scenarios: controlled real cycles; automatic scenario: real AI lifespan and BE Outbox timer; real DB/provider codecs/webhook/business services"}
        dump("manifest.json",manifest)
        child = dict(os.environ, JAVA_HOME="/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home",
                     LOCAL_INTEGRATION_DIR=str(HERE), LOCAL_INTEGRATION_RUN=RUN,
                     JAVA_TOOL_OPTIONS="-Djava.net.preferIPv4Stack=true")
        child["PATH"] = child["JAVA_HOME"]+"/bin:"+child.get("PATH","")
        log = (ART/"be-gradle.log").open("w")
        process = subprocess.Popen(["./gradlew","--offline","--no-daemon","--console=plain","-I",str(HERE/"be-test.init.gradle"),"test"],cwd=BE,env=child,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError("BE failed to start; inspect be-gradle.log")
            try:
                if be("health").get("ready"): break
            except OSError: pass
            time.sleep(.5)
        else: raise RuntimeError("BE startup timeout")
        check("BE real Spring context connected to local MySQL",True)
        check("Java child OS sandbox blocks nonlocal network",be("health")["java_egress_blocked"])
        from app.virtual_fitting.webhook_router import router
        app=FastAPI(); app.include_router(router)
        with TestClient(app) as client:
            scenarios(client,codes)
        automatic_scenario(codes)
        status="passed"
    except Exception:
        (ART/"failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if process and process.poll() is None:
            try:
                # Restore only this run's failure-injection trigger if the scenario stopped early.
                if job_ids:
                    be("restore-save",job_id=job_ids[0])
                be("shutdown")
                code = process.wait(timeout=30)
                if code != 0:
                    status = "failed"
                    (ART / "be-exit-failure.txt").write_text(str(code))
            except Exception:
                process.terminate()
        if server: server.shutdown(); server.server_close()
        if env.pools: close_env(env)
        if admin:
            if before is not None:
                after=public_snapshot(admin)
                dump("public-after.json",after)
                check("existing AI public data including usage/counters unchanged",before==after)
            dump("ai-tables.json",{t:env.sql(f'SELECT row_to_json(t) FROM "{SCHEMA}"."{t}" t') for t in before} if before else {})
            admin.close()
        dump("queue.json", {name:[dataclasses.asdict(m) for m in env.sqs.queues[url]] for name,url in QUEUE_URLS.items()})
        dump("runware.json",{"calls":env.runware.calls,"llm_requests":env.runware.llm_requests,"objects":list(env.s3.objects)})
        dump("summary.json",{"status":status,"run":RUN,"job_ids":job_ids,"checks":len(checks),"passed":sum(c["passed"] for c in checks),"resources_retained":True})
        mp.undo()
        print("ARTIFACTS",ART,flush=True)
    return 0 if status=="passed" else 1


if __name__ == "__main__":
    sys.exit(main())
