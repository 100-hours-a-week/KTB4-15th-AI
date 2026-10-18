"""Real SQS, real BE/AI business code, local DBs, existing S3/Runware/LLM mocks.

Invoke with python -B. No shell evaluation of dotenv. No migration or existing file writes.
Requires an explicit isolation confirmation; preflight never sends or receives messages.
"""
import argparse
import contextlib
import dataclasses
from datetime import datetime, UTC
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BE = Path('/Users/ijaein/Documents/lookddak-be-local')
PROFILE = 'lookddak-sqs-local'
ROLES = ('BE_FITTING_REQUEST', 'AI_FITTING_RESPONSE', 'BE_FITTING_RESPONSE')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


class TargetIsolationError(RuntimeError):
    pass


def read_settings():
    values = {}
    allowed = {'AWS_PROFILE', 'AWS_REGION'} | {r + '_QUEUE_URL' for r in ROLES}
    for line in (ROOT / '.env.sqs-local').read_text().splitlines():
        parts = shlex.split(line, comments=True, posix=True)
        if not parts:
            continue
        if parts[0] == 'export':
            parts.pop(0)
        if len(parts) != 1 or '=' not in parts[0]:
            raise ValueError('unsupported dotenv syntax; nothing was evaluated')
        key, value = parts[0].split('=', 1)
        if key not in allowed or key in values:
            raise ValueError('unexpected/duplicate dotenv key')
        values[key] = value
    if set(values) != allowed or values['AWS_PROFILE'] != PROFILE:
        raise ValueError('required test profile/settings missing')
    accounts = set()
    for role in ROLES:
        match = re.fullmatch(r'https://sqs\.([a-z0-9-]+)\.amazonaws\.com/(\d{12})/([A-Za-z0-9_-]+)', values[role + '_QUEUE_URL'])
        if not match or match[1] != values['AWS_REGION']:
            raise ValueError('invalid/region-mismatched standard queue URL')
        accounts.add(match[2])
    if len(accounts) != 1:
        raise ValueError('queue accounts differ')
    return values


def clean_environment(values):
    keep = {k: os.environ[k] for k in ('HOME', 'PATH', 'TMPDIR', 'LANG') if k in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update(values)
    os.environ.update(AWS_EC2_METADATA_DISABLED='true', AWS_SDK_LOAD_CONFIG='true',
                      PYTHONDONTWRITEBYTECODE='1', AUTH_DISABLED='1',
                      RUNWARE_VTON_API_KEY='test-mock-only', RUNWARE_LLM_API_KEY='test-mock-only')


def redact(value):
    value = re.sub(r'jdbc:[^\s\]\),]+', '<local-db-redacted>', str(value))
    value = re.sub(r'postgres(?:ql)?://[^\s\]\),]+', '<local-db-redacted>', value)
    value = re.sub(r'https://sqs\.[^\s\"\'\]\)]+', '<queue-url-redacted>', value)
    value = re.sub(r'(?:arn:aws:[^\s\"\'\]\)]+)', '<aws-principal-redacted>', value)
    return value


class Run:
    def __init__(self, values):
        self.values = values
        self.run = 'sqs_' + uuid.uuid4().hex[:12]
        self.art = HERE / 'runs' / self.run
        self.art.mkdir(parents=True)
        self.urls = {role: values[role + '_QUEUE_URL'] for role in ROLES}
        self.jobs = set()
        self.sent = {}
        self.receipts = {}
        self.deleted = set()
        self.phase = 'preflight'
        self.checks = []
        self.lock = threading.RLock()
        self.process = None
        self.client = None
        self.entered = False
        self.pool = None
        self.mp = None
        self.sqs = None
        self.stage = 'preflight'
        self.be_created = None
        self.cleanup_errors = []
        self.cancel = False

    def dump(self, name, data):
        (self.art / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')

    def event(self, data):
        with self.lock:
            data = dict(data, time=datetime.now(UTC).isoformat(), phase=self.phase)
            with (self.art / 'python-sqs.jsonl').open('a') as stream:
                stream.write(json.dumps(data, ensure_ascii=False, default=str) + '\n')

    def check(self, name, passed, evidence=None):
        self.checks.append(dict(check=name, passed=bool(passed), evidence=evidence))
        self.dump('checks.json', self.checks)
        if not passed:
            if 'preexisting queued/inflight/delayed' in name:
                raise TargetIsolationError(name)
            raise AssertionError(name)
        print('PASS', name, flush=True)

    @staticmethod
    def body_job(body):
        try:
            return json.loads(body).get('fitting_job_id')
        except (ValueError, AttributeError):
            return None

    def role(self, url):
        for role, target in self.urls.items():
            if target == url:
                return role
        raise RuntimeError('queue outside test allowlist')

    def attach_observer(self, client):
        # Observe real botocore requests/responses; no response replacement or fake client.
        def before(params, context, model, **kwargs):
            url = params.get('QueueUrl')
            role = self.role(url) if url else None
            api = model.name
            if api in ('SendMessage', 'DeleteMessage', 'ChangeMessageVisibility'):
                if api == 'SendMessage':
                    if self.body_job(params.get('MessageBody', '')) not in self.jobs:
                        raise RuntimeError('foreign message send blocked')
                elif params.get('ReceiptHandle') not in self.receipts:
                    raise RuntimeError('untracked receipt mutation blocked')
            context['test_trace'] = {'api': api, 'queue': role,
                'body_job': self.body_job(params.get('MessageBody', '')),
                'receipt': params.get('ReceiptHandle'),
                'visibility_seconds': params.get('VisibilityTimeout')}

        def after(parsed, model, context, http_response, **kwargs):
            info = context.get('test_trace', {})
            api, role = info.get('api', model.name), info.get('queue')
            status = http_response.status_code
            data = {'api': api, 'queue': role, 'http_status': status}
            if parsed.get('Error'):
                data['error_code'] = parsed['Error'].get('Code')
            if api == 'SendMessage' and parsed.get('MessageId'):
                mid = parsed['MessageId']
                self.sent[mid] = role
                data.update(message_id=mid, job_id=info['body_job'])
            elif api == 'ReceiveMessage':
                data['messages'] = []
                for message in parsed.get('Messages', []):
                    mid, receipt = message['MessageId'], message['ReceiptHandle']
                    job = self.body_job(message.get('Body', ''))
                    own = job in self.jobs or mid in self.sent
                    # Handles stay in memory, never in report/log output.
                    self.receipts[receipt] = dict(message_id=mid, role=role, own=own, received=time.monotonic())
                    data['messages'].append(dict(message_id=mid, job_id=job, own=own))
                    if not own and self.phase != 'cleanup':
                        self.event(data)
                        raise RuntimeError('foreign message received; stopped before application handler')
            elif api in ('DeleteMessage', 'ChangeMessageVisibility'):
                receipt = self.receipts.get(info.get('receipt'), {})
                data['message_id'] = receipt.get('message_id')
                data['visibility_seconds'] = info.get('visibility_seconds')
                if status == 200 and api == 'DeleteMessage':
                    self.deleted.add(receipt['message_id'])
            self.event(data)

        def endpoint_guard(request, **kwargs):
            host = urlparse(request.url).hostname
            if host != 'sqs.' + self.values['AWS_REGION'] + '.amazonaws.com':
                raise RuntimeError('non-SQS endpoint blocked')
        client.meta.events.register('before-parameter-build.sqs', before)
        client.meta.events.register('after-call.sqs', after)
        client.meta.events.register('before-send.sqs', endpoint_guard)
        return client

    def preflight(self):
        import boto3
        from botocore.config import Config
        self.session = boto3.Session(profile_name=PROFILE, region_name=self.values['AWS_REGION'])
        identity = self.session.client('sts', config=Config(retries={'total_max_attempts': 1})).get_caller_identity()
        account = identity['Account']
        self.sqs = self.attach_observer(self.session.client('sqs', config=Config(retries={'total_max_attempts': 1},connect_timeout=3,read_timeout=25)))
        summary = {'profile': PROFILE, 'region': self.values['AWS_REGION'],
                   'principal_type': identity['Arn'].split(':')[5].split('/')[0],
                   'principal_sha256': hashlib.sha256(identity['Arn'].encode()).hexdigest(),
                   'account_suffix': account[-4:], 'queues': []}
        self.dump('preflight.json',summary)
        main = list(self.urls.items())
        for role, url in main:
            attrs = self.sqs.get_queue_attributes(QueueUrl=url, AttributeNames=['All'])['Attributes']
            arn = attrs['QueueArn'].split(':')
            self.check(role + ': actual target account/region matches STS', arn[4] == account and arn[3] == self.values['AWS_REGION'])
            counts = {k: int(attrs.get(k, 0)) for k in ('ApproximateNumberOfMessages','ApproximateNumberOfMessagesNotVisible','ApproximateNumberOfMessagesDelayed')}
            item = dict(role=role, name=url.rsplit('/',1)[-1], counts=counts,
                        visibility_seconds=attrs.get('VisibilityTimeout'))
            summary['queues'].append(item)
            self.dump('preflight.json',summary)
            self.check(role + ': no preexisting queued/inflight/delayed messages', not any(counts.values()),counts)
            if 'RedrivePolicy' in attrs:
                redrive = json.loads(attrs['RedrivePolicy'])
                pieces = redrive['deadLetterTargetArn'].split(':')
                if pieces[3] != self.values['AWS_REGION'] or pieces[4] != account:
                    raise RuntimeError('DLQ account/region mismatch')
                self.urls[role + '_DLQ'] = f'https://sqs.{pieces[3]}.amazonaws.com/{pieces[4]}/{pieces[5]}'
                item.update(dlq_name=pieces[5], max_receive_count=redrive['maxReceiveCount'])
            try:
                tags = self.sqs.list_queue_tags(QueueUrl=url).get('Tags', {})
                item['tag_keys'] = sorted(tags)
            except Exception as error:
                item['tags_error'] = self.error(error)
            self.dump('preflight.json',summary)
        self.dump('preflight.json',summary)
        return summary

    @staticmethod
    def error(error):
        response = getattr(error, 'response', {})
        code = response.get('Error', {}).get('Code')
        category = 'application_or_contract'
        if isinstance(error,TargetIsolationError):
            category = 'target_isolation'
        elif code in ('AccessDenied','AccessDeniedException','KmsAccessDenied'):
            category = 'permission'
        elif code in ('ExpiredToken','InvalidClientTokenId','InvalidSignatureException','SignatureDoesNotMatch') or 'Credential' in type(error).__name__:
            category = 'authentication'
        elif any(word in type(error).__name__ for word in ('Connection','Timeout','Endpoint')):
            category = 'network'
        return dict(category=category, error_type=type(error).__name__, error_code=code,
                    http_status=response.get('ResponseMetadata',{}).get('HTTPStatusCode'))

    def be(self, route, payload=None):
        request = Request('http://127.0.0.1:18772/'+route,
                          data=json.dumps(payload or {}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=35) as response:
            return json.load(response)

    def start_be(self):
        self.stage = 'BE startup'
        child = dict(os.environ, JAVA_HOME='/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home',
                     JAVA_TOOL_OPTIONS='-Djava.net.preferIPv4Stack=true', REAL_SQS_RUN=self.run,
                     REAL_SQS_TEST_DIR=str(HERE), REAL_SQS_ARTIFACT_DIR=str(self.art))
        child['PATH'] = child['JAVA_HOME']+'/bin:'+child.get('PATH','')
        command = ['./gradlew','--offline','--no-daemon','--console=plain','--project-cache-dir',str(self.art/'project-cache'),'-I',str(HERE/'be.init.gradle'),'test']
        self.process = subprocess.Popen(command,cwd=BE,env=child,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        def capture():
            with (self.art/'be-gradle.log').open('w') as stream:
                for line in self.process.stdout:
                    stream.write(redact(line));stream.flush()
        self.log_thread = threading.Thread(target=capture,daemon=True);self.log_thread.start()
        deadline = time.monotonic()+180
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeError('BE fixture startup failed; redacted log preserved')
            try:
                health=self.be('health')
                if health['ready']:
                    self.check('BE injects actual Java SDK SqsClient', 'DefaultSqsClient' in health['sqs_real_class'],health)
                    return
            except OSError:
                pass
            time.sleep(.5)
        raise TimeoutError('BE startup timeout')

    def sql(self, statement, params=None):
        import psycopg
        with psycopg.connect('postgresql://ai_test:local_ai_password@127.0.0.1:15432/lookddak_ai_test',autocommit=True) as connection:
            cursor=connection.execute(statement,params)
            return cursor.fetchall() if cursor.description else None

    def job(self, job):
        from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
        return AsyncJobRepository(self.pool).find_by_fitting_job_id(job)

    def configure_ai(self):
        import boto3
        import pytest
        from psycopg_pool import ConnectionPool
        from app.config import settings, database
        # Read and reuse the existing boundary mocks, not the old in-memory SQS tool.
        from tests.v2_integration_harness import FakeS3Backend, FakeRunware, RUNWARE_MODULES, BUCKET, REGION, WEBHOOK_TOKEN, WEBHOOK_URL
        self.mp = pytest.MonkeyPatch()
        self.storage = FakeS3Backend()
        self.mock = FakeRunware(self.storage)
        self.pool = ConnectionPool('postgresql://ai_test:local_ai_password@127.0.0.1:15432/lookddak_ai_test',min_size=1,max_size=10)
        self.mp.setattr(database,'_pool',self.pool)
        for key,value in dict(AWS_REGION=self.values['AWS_REGION'], S3_BUCKET=BUCKET,
                              WEBHOOK_TOKEN=WEBHOOK_TOKEN,VIRTUAL_FITTING_RUNWARE_WEBHOOK_URL=WEBHOOK_URL,
                              CHECKPOINT_DSN='',VIRTUAL_FITTING_REQUEST_WORKER_ENABLED=True,
                              VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED=True,
                              VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED=True,
                              VIRTUAL_FITTING_WATCHDOG_ENABLED=True).items():
            self.mp.setattr(settings,key,value)
        for role in ROLES:
            self.mp.setattr(settings,role+'_QUEUE_URL',self.urls[role])
        real_boto = boto3.client
        def factory(service,*args,**kwargs):
            if service=='s3':return self.storage.client()
            if service=='sqs':return self.attach_observer(real_boto(service,*args,**kwargs))
            raise RuntimeError('non-SQS AWS client blocked')
        self.mp.setattr(boto3,'client',factory)
        for name in RUNWARE_MODULES:
            self.mp.setattr(importlib.import_module(name),'urlopen',self.mock)
        from app.clients import sqs,s3
        for getter in (sqs.get_sqs_client,sqs.get_webhook_sqs_client,s3.get_s3_client,s3.get_s3_presign_client):getter.cache_clear()
        # Keep the existing 120s lease / 40s heartbeat policy. The Mock boundary waits 43s
        # so an actual worker keepalive has time to call real ChangeMessageVisibility.
        def slow_accept(submission):
            self.dump('mock-generation-start.json',{'task_uuid':submission.task_uuid,'wait_seconds':43})
            time.sleep(43)
            return 'accept'
        self.mock.async_modes.append(slow_accept)
        # Block accidental Python HTTP clients outside the actual regional SQS endpoint.
        # Native libpq receives only the explicit loopback DSN above. Java SDK requests are
        # separately guarded by the fixture's beforeTransmission interceptor.
        allowed_hosts={'localhost','127.0.0.1','::1','sqs.'+self.values['AWS_REGION']+'.amazonaws.com'}
        real_resolve=socket.getaddrinfo
        allowed_ips={'127.0.0.1','::1'}
        for answer in real_resolve('sqs.'+self.values['AWS_REGION']+'.amazonaws.com',443):
            allowed_ips.add(answer[4][0])
        def resolve(host,*args,**kwargs):
            name=host.decode() if isinstance(host,bytes) else host
            if name not in allowed_hosts and name not in allowed_ips:
                raise PermissionError('non-SQS hostname blocked in test process')
            result=real_resolve(host,*args,**kwargs)
            allowed_ips.update(answer[4][0] for answer in result)
            return result
        real_connect=socket.socket.connect
        def connect(sock,address):
            if sock.family in (socket.AF_INET,socket.AF_INET6) and address[0] not in allowed_ips:
                raise PermissionError('non-SQS socket blocked in test process')
            return real_connect(sock,address)
        self.mp.setattr(socket,'getaddrinfo',resolve)
        self.mp.setattr(socket.socket,'connect',connect)

    def journal(self, language):
        path=self.art/(language+'-sqs.jsonl')
        if not path.exists():return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def wait(self, predicate, timeout=120):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if self.cancel:raise InterruptedError('test interrupted')
            value=predicate()
            if value:return value
            time.sleep(.3)
        raise TimeoutError(self.stage)

    def execute(self):
        from fastapi.testclient import TestClient
        from tests.v2_integration_harness import post_webhook, COMMENT, TITLE
        self.stage='local fixture preparation'
        active=self.sql("SELECT fitting_job_id FROM virtual_fitting_async_jobs WHERE next_action IS NOT NULL OR status NOT IN ('COMPLETED','FAILED')")
        self.check('no unrelated active AI public jobs before starting workers',not active)
        code=str(900000000000+int(uuid.uuid4().hex[:7],16))
        self.image='local-tests/'+self.run+'/original.png'
        self.sql("INSERT INTO products(product_code,id,product_name,price,detail_url,image_url,main_category,sub_category,is_sold_out,crawled_at,description_summary) VALUES(%s,1,'실제 SQS 테스트 상의',10000,'http://127.0.0.1/product','http://127.0.0.1/product.jpg','상의','반소매 티셔츠',false,now(),'네이비 반소매 티셔츠')",(int(code),))
        self.start_be()
        created=self.be('create',{'code':code,'image_key':self.image});self.be_created=created
        self.dump('be-created.json',created)
        job=created['created']['fittingJobId'];self.jobs.add(job)
        self.check('real BE service created PENDING Job and Outbox',created['created']['status']=='PENDING' and created['outbox']['status']=='PENDING')
        self.check('no preexisting AI row collides with BE job ID',not self.sql('SELECT fitting_job_id FROM virtual_fitting_async_jobs WHERE fitting_job_id=%s',(job,)))
        payload=json.loads(created['outbox']['payload'])
        self.dump('resources.json',dict(run=self.run,job_id=job,member_id=created['member_id'],product_id=created['product_id'],product_code=code,original_image_key=self.image,ai_schema='public'))
        self.configure_ai();self.storage.store(self.image,content_type='image/png')
        from app.main import app
        self.client=TestClient(app);self.client.__enter__();self.entered=True
        self.phase='normal';self.stage='actual BE Outbox -> SQS -> AI request worker'
        self.be('start')
        row=self.wait(lambda:(r:=self.job(job)) and r.runware_accepted_at and r,timeout=100)
        self.check('BE ID, requested_at, product code preserved',row.fitting_job_id==job and row.products==(code,) and row.requested_at==datetime.fromisoformat(payload['requested_at'].replace('Z','+00:00')))
        self.check('AI WAITING_PRUNA and BE PENDING are distinct service states',row.status.value=='WAITING_PRUNA' and self.be('snapshot')['job']['status']=='PENDING')
        visibility=[e for e in self.journal('python') if e.get('api')=='ChangeMessageVisibility' and e.get('http_status')==200 and e.get('queue')=='BE_FITTING_REQUEST']
        self.check('actual worker heartbeat extended actual request SQS visibility',bool(visibility),visibility)
        self.stage='Mock image + actual webhook + internal SQS processing'
        response=post_webhook(self.client,self.mock.complete(row.task_uuid))
        self.check('Mock stores result image and actual webhook succeeds',response.status_code==200)
        self.stage='actual result SQS -> BE Consumer persistence/query'
        final=self.wait(lambda:(b:=self.be('snapshot'))['job']['status']=='COMPLETED' and b,timeout=90)
        ai=self.job(job)
        self.dump('normal-be.json',final);self.dump('normal-ai.json',dataclasses.asdict(ai))
        self.check('AI publication complete and BE persistence complete independently',ai.result_message_status.value=='PUBLISHED' and final['query']['status']=='COMPLETED' and final['outbox']['status']=='PUBLISHED')
        self.check('one result, one notification, one BE request count',len(final['results'])==len(final['notifications'])==final['member']['fitting_request_count']==1)
        result=final['results'][0]
        self.check('actual AI result payload matches BE persisted key/title/comment',result['result_image_key']==ai.result_payload['result_image_key'] and result['outfit_name']==ai.result_payload['llm_title'] and result['ai_comment']==ai.result_payload['llm_comment'])
        requests=self.mock.llm_requests
        self.check('image -> comment -> exact comment-based title',len(requests)==2 and row.async_image_key in json.dumps(requests[0]) and COMMENT in requests[1]['messages'][1]['content'] and result['outfit_name']==TITLE)
        # Wait for application DeleteMessage completion, not only its earlier DB commit.
        self.stage='normal DeleteMessage acknowledgements'
        def normal_deletes():
            records=self.journal('python')+self.journal('java')
            return {e.get('queue') for e in records if e.get('api')=='DeleteMessage' and e.get('http_status')==200 and e.get('phase')=='normal'}>=set(ROLES)
        self.wait(normal_deletes,timeout=30)
        self.check('all three queues acknowledged normal DeleteMessage',normal_deletes())
        self.stage='same result duplicate via real SQS'
        self.phase='duplicate'
        before=final
        duplicate=self.sqs.send_message(QueueUrl=self.urls['BE_FITTING_RESPONSE'],MessageBody=json.dumps(ai.result_payload,ensure_ascii=False))
        duplicate_id=duplicate['MessageId']
        self.dump('duplicate-message.json',{'message_id':duplicate_id,'job_id':job})
        self.wait(lambda:any(e.get('api')=='DeleteMessage' and e.get('message_id')==duplicate_id and e.get('http_status')==200 for e in self.journal('java')),timeout=60)
        after=self.be('snapshot');self.dump('duplicate-be.json',after)
        self.check('duplicate SQS result causes no added/changed BE results or notifications',after['results']==before['results'] and after['notifications']==before['notifications'] and after['job']==before['job'] and after['member']==before['member'])
        self.check('duplicate does not rerun paid generation/LLM',self.mock.count('async_submit')==1 and self.mock.count('sync_generate')==0 and len(self.mock.llm_requests)==2)
        self.dump('mock-summary.json',{'calls':self.mock.calls,'objects':list(self.storage.objects),'llm_pair_count':len(requests)//2})
        self.dump('costs.json',self.sql('SELECT fitting_job_id,task_uuid::text,cost FROM virtual_fitting_task_costs WHERE fitting_job_id=%s',(job,)))
        self.dump('message-correlation.json',self.journal('python')+self.journal('java'))

    def stop(self):
        results={}
        if self.process and self.process.poll() is None:
            try:
                results['be_timer']=self.be('stop')
                self.dump('be-before-shutdown.json',self.be('snapshot'))
            except Exception as error:results['be_stop_error']=self.error(error)
        if self.entered:
            try:
                self.client.__exit__(None,None,None)
                results['ai_lifespan']='stopped; request/postprocess/result publisher/watchdog shutdown awaited'
            except Exception as error:results['ai_stop_error']=self.error(error)
            self.entered=False
        if self.process and self.process.poll() is None:
            try:self.be('shutdown');self.process.wait(timeout=45)
            except Exception:
                self.process.terminate()
                try:self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                results['be_forced_termination']=True
        if self.process:
            results['be_exit_code']=self.process.returncode
            self.log_thread.join(timeout=5)
        if self.pool and not self.pool.closed:self.pool.close()
        self.dump('process-shutdown.json',results)
        return results

    def cleanup(self):
        self.phase='cleanup'
        # Called AFTER evidence capture, outcome decision, and worker/BE shutdown.
        summary={'forced_deletes':[],'remaining':[],'foreign_messages_untouched':[], 'queue_errors':[]}
        if not self.jobs:
            summary['skipped']='no job created/no message sends possible'
            self.dump('cleanup.json',summary);return summary
        java=self.journal('java')
        for e in java:
            if e.get('api')=='SendMessage' and e.get('message_id'):self.sent[e['message_id']]=e['queue']
            if e.get('api')=='DeleteMessage' and e.get('http_status')==200:self.deleted.add(e.get('message_id'))
        # Recover failed/inflight Python deliveries with their latest valid receipt.
        latest={}
        for receipt,info in self.receipts.items():
            if info['own'] and info['message_id'] not in self.deleted:latest[info['message_id']]=(receipt,info)
        for mid,(receipt,info) in latest.items():
            try:
                self.sqs.delete_message(QueueUrl=self.urls[info['role']],ReceiptHandle=receipt)
                summary['forced_deletes'].append({'queue':info['role'],'message_id':mid,'method':'retained_receipt'})
            except Exception as error:
                summary['queue_errors'].append(dict(queue=info['role'],api='DeleteMessage',message_id=mid,**self.error(error)))
        # Poll main queues and declared DLQs. Unknown IDs after uncertain send are identified by job ID.
        # No PurgeQueue, queue/attribute mutation, or deletion of foreign payloads.
        pending=set(self.urls)
        deadline=time.monotonic()+360
        empty_passes={r:0 for r in pending}
        while pending and time.monotonic()<deadline:
            for role in list(pending):
                try:
                    response=self.sqs.receive_message(QueueUrl=self.urls[role],MaxNumberOfMessages=10,WaitTimeSeconds=1,VisibilityTimeout=10,MessageSystemAttributeNames=['ApproximateReceiveCount'])
                    messages=response.get('Messages',[])
                    for message in messages:
                        mid,receipt=message['MessageId'],message['ReceiptHandle']
                        own=self.body_job(message.get('Body','')) in self.jobs or mid in self.sent
                        if own:
                            self.sqs.delete_message(QueueUrl=self.urls[role],ReceiptHandle=receipt)
                            summary['forced_deletes'].append(dict(queue=role,message_id=mid,method='cleanup_redelivery'))
                        else:
                            summary['foreign_messages_untouched'].append(dict(queue=role,message_id=mid))
                            self.sqs.change_message_visibility(QueueUrl=self.urls[role],ReceiptHandle=receipt,VisibilityTimeout=0)
                    attrs=self.sqs.get_queue_attributes(QueueUrl=self.urls[role],AttributeNames=['ApproximateNumberOfMessages','ApproximateNumberOfMessagesNotVisible','ApproximateNumberOfMessagesDelayed'])['Attributes']
                    counts=[int(v) for v in attrs.values()]
                    unknown=set(self.sent)-self.deleted
                    if not messages and not any(counts):empty_passes[role]+=1
                    else:empty_passes[role]=0
                    if empty_passes[role]>=2:pending.remove(role)
                    if summary['foreign_messages_untouched']:
                        # Do not churn unrelated messages. Own known inflight IDs still reported.
                        pending.remove(role)
                except Exception as error:
                    summary['queue_errors'].append(dict(queue=role,api=getattr(error,'operation_name','cleanup_receive_or_delete_or_attributes'),**self.error(error)))
                    pending.remove(role)
            if pending:time.sleep(1)
        summary['remaining']=[dict(queue=self.sent[mid],message_id=mid) for mid in set(self.sent)-self.deleted]
        summary['unverified_queues']=sorted(pending)
        summary['known_sent_ids']=len(self.sent)
        summary['normal_deleted_ids']=sorted({e.get('message_id') for e in self.journal('python')+java if e.get('api')=='DeleteMessage' and e.get('phase')!='cleanup' and e.get('http_status')==200})
        self.dump('cleanup.json',summary)
        return summary


def verify_baseline():
    baseline=json.loads((HERE/'baseline.json').read_text())
    result={}
    for label,state in baseline.items():
        root=Path(state['root']);changed=[];missing=[]
        for relative,digest in state['files'].items():
            path=root/relative
            if not path.exists():missing.append(relative)
            elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:changed.append(relative)
        result[label]={'existing_files_checked':len(state['files']),'changed':changed,'missing':missing,
                       'head_unchanged':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==state['head']}
        current_status=subprocess.check_output(['git','-C',str(root),'status','--porcelain=v1','--untracked-files=all'],text=True)
        current_status='\n'.join(line for line in current_status.splitlines() if not (label=='ai' and line.startswith('?? tests/real_sqs_local/')))
        result[label]['prior_staged_unstaged_untracked_status_unchanged']=current_status.rstrip()==state['status'].rstrip()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--exclusive-queues-confirmed',action='store_true',help='Only after the user confirms these three queues have no other consumers.')
    args=parser.parse_args()
    values=read_settings();clean_environment(values)
    run=Run(values)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(run,'cancel',True))
    status='failed';cleanup={};shutdown={}
    try:
        run.dump('manifest.json',{'run':run.run,'ai_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'be_commit':subprocess.check_output(['git','-C',str(BE),'rev-parse','HEAD'],text=True).strip(),
            'profile':PROFILE,'region':values['AWS_REGION'],'exclusive_queues_confirmed':args.exclusive_queues_confirmed,
            'queue_names':{r:u.rsplit('/',1)[-1] for r,u in run.urls.items()},
            'mocks':['existing FakeS3Backend','existing FakeRunware for generation/LLM','BE S3 presign provider'],
            'real':['Java SQS SDK','boto3 SQS','BE job/outbox/result/query services','AI workers/provider codecs/repositories','local MySQL/PostgreSQL']})
        run.preflight()
        if args.preflight_only:
            status='preflight_only'
        elif not args.exclusive_queues_confirmed:
            status='blocked_isolation_confirmation'
        else:
            run.execute();status='passed'
    except Exception as error:
        run.dump('failure.json',dict(stage=run.stage,**run.error(error)))
        (run.art/'failure-trace.txt').write_text(redact(''.join(traceback.format_tb(error.__traceback__)))+'\n'+type(error).__name__+'\n')
        print('TEST FAILED',run.stage,run.error(error),flush=True)
    finally:
        # Save evidence before terminating any worker or performing cleanup deletions.
        if run.pool and run.jobs:
            try:run.dump('ai-before-shutdown.json',[dataclasses.asdict(run.job(j)) for j in run.jobs if run.job(j)])
            except Exception as error:run.dump('snapshot-error.json',run.error(error))
        shutdown=run.stop()
        try:cleanup=run.cleanup()
        except Exception as error:
            cleanup={'cleanup_error':run.error(error),'remaining_known_ids':sorted(set(run.sent)-run.deleted)}
            run.dump('cleanup.json',cleanup)
        if run.mp:run.mp.undo()
        baseline=verify_baseline();run.dump('existing-files-verification.json',baseline)
        if any(s['changed'] or s['missing'] or not s['head_unchanged'] or not s['prior_staged_unstaged_untracked_status_unchanged'] for s in baseline.values()):status='failed_existing_files_changed'
        if status=='passed' and (cleanup.get('remaining') or cleanup.get('unverified_queues') or cleanup.get('queue_errors') or cleanup.get('cleanup_error') or shutdown.get('be_exit_code')!=0):status='passed_flow_cleanup_or_shutdown_incomplete'
        summary={'run':run.run,'status':status,'stage':run.stage,'checks':len(run.checks),'checks_passed':sum(c['passed'] for c in run.checks),'job_ids':sorted(run.jobs),'processes':shutdown,'cleanup':cleanup,'existing_files':baseline}
        run.dump('summary.json',summary)
        print('RESULT',status,'ARTIFACTS',run.art,flush=True)
    return 0 if status in ('passed','preflight_only') else 1


if __name__=='__main__':sys.exit(main())
