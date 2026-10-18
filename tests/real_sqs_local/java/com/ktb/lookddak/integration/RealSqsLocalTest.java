package com.ktb.lookddak.integration;

import com.ktb.lookddak.domain.fitting.dto.FittingJobCreateRequest;
import com.ktb.lookddak.domain.fitting.entity.FittingCandidate;
import com.ktb.lookddak.domain.fitting.repository.FittingCandidateRepository;
import com.ktb.lookddak.domain.fitting.service.FittingJobService;
import com.ktb.lookddak.domain.fitting.mq.FittingResultConsumer;
import com.ktb.lookddak.domain.fitting.mq.FittingResultConsumerProperties;
import com.ktb.lookddak.domain.fitting.mq.FittingResultMessageHandler;
import com.ktb.lookddak.domain.fitting.mq.outbox.*;
import com.ktb.lookddak.domain.member.entity.*;
import com.ktb.lookddak.domain.member.repository.*;
import com.ktb.lookddak.domain.product.entity.*;
import com.ktb.lookddak.domain.product.repository.ProductRepository;
import com.ktb.lookddak.global.aws.sqs.SqsProperties;
import com.ktb.lookddak.global.storage.s3.S3PresignedUrlProvider;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.bean.override.convention.TestBean;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.core.interceptor.*;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.*;
import tools.jackson.databind.ObjectMapper;
import java.math.BigDecimal;
import java.net.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.*;

/** New fixture reusing the real service/scheduler/consumer arrangement of local_be_ai. */
@SpringBootTest(properties={
 "spring.datasource.url=jdbc:mysql://127.0.0.1:13306/lookddak_be_test?allowPublicKeyRetrieval=true&useSSL=false&serverTimezone=UTC",
 "spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver",
 "spring.datasource.username=be_test", "spring.datasource.password=local_be_password",
 "spring.jpa.hibernate.ddl-auto=validate", "spring.jpa.open-in-view=false",
 "aws.sqs.region=${AWS_REGION}",
 "aws.sqs.fitting-request-queue-url=${BE_FITTING_REQUEST_QUEUE_URL}",
 "aws.sqs.fitting-response-queue-url=${BE_FITTING_RESPONSE_QUEUE_URL}",
 "fitting.outbox.enabled=false", "fitting.result-consumer.enabled=false",
 "fitting.outbox.poll-interval=24h", "fitting.result-consumer.poll-interval=24h",
 "ai.server.base-url=http://127.0.0.1:18772", "sentry.dsn=", "sentry.logging.enabled=false"
})
class RealSqsLocalTest {
 @Autowired FittingJobService jobs;
 @Autowired MemberRepository members;
 @Autowired MemberProfileRepository profiles;
 @Autowired ProductRepository products;
 @Autowired FittingCandidateRepository candidates;
 @Autowired OutboxEventClaimService claims;
 @Autowired OutboxEventPublisher publisher;
 @Autowired SqsProperties sqsProperties;
 @Autowired FittingResultMessageHandler resultHandler;
 @Autowired ApplicationEventPublisher events;
 @Autowired JdbcTemplate jdbc;
 @Autowired ObjectMapper mapper;
 @TestBean(methodName="realSqsClient") SqsClient sqs;
 @MockitoBean S3PresignedUrlProvider presign;
 static volatile Long ownJob;
 static final Map<String,String> receipts=new ConcurrentHashMap<>();
 static final ObjectMapper JSON=new ObjectMapper();
 final String run=System.getenv("REAL_SQS_RUN");
 final CountDownLatch done=new CountDownLatch(1);
 Long memberId, productId;
 ScheduledExecutorService timer;
 volatile Throwable timerError;

 static String role(String url) {
   for(String key:List.of("BE_FITTING_REQUEST_QUEUE_URL","AI_FITTING_RESPONSE_QUEUE_URL","BE_FITTING_RESPONSE_QUEUE_URL"))
     if(url.equals(System.getenv(key))) return key.replace("_QUEUE_URL","");
   throw new IllegalStateException("queue outside explicit allowlist");
 }
 static synchronized void journal(Map<String,Object> data) {
   try {
     data.put("time",java.time.Instant.now().toString());
     Files.writeString(Path.of(System.getenv("REAL_SQS_ARTIFACT_DIR"),"java-sqs.jsonl"),JSON.writeValueAsString(data)+"\n",StandardOpenOption.CREATE,StandardOpenOption.APPEND);
   } catch(Exception e) {throw new IllegalStateException("journal write failed");}
 }
 static long bodyJob(String body) {
   try {return JSON.readTree(body).path("fitting_job_id").asLong(-1);}
   catch(Exception e) {return -1;}
 }
 static SqsClient realSqsClient() {
   return SqsClient.builder().region(Region.of(System.getenv("AWS_REGION")))
    .credentialsProvider(ProfileCredentialsProvider.create("lookddak-sqs-local"))
    .overrideConfiguration(c->c.addExecutionInterceptor(new ExecutionInterceptor() {
      @Override public void beforeExecution(Context.BeforeExecution c,ExecutionAttributes a) {
        if(c.request() instanceof SendMessageRequest r) {
          role(r.queueUrl());
          if(ownJob==null || bodyJob(r.messageBody())!=ownJob) throw new IllegalStateException("foreign job send blocked");
        } else if(c.request() instanceof ReceiveMessageRequest r) {role(r.queueUrl());}
        else if(c.request() instanceof DeleteMessageRequest r) {
          role(r.queueUrl());
          if(!receipts.containsKey(r.receiptHandle()))throw new IllegalStateException("unknown receipt delete blocked");
        } else throw new IllegalStateException("unexpected SQS operation blocked");
      }
      @Override public void beforeTransmission(Context.BeforeTransmission c,ExecutionAttributes a) {
        String expected="sqs."+System.getenv("AWS_REGION")+".amazonaws.com";
        if(!c.httpRequest().host().equals(expected) || !c.httpRequest().protocol().equals("https"))throw new IllegalStateException("non-SQS endpoint blocked");
      }
      @Override public void afterExecution(Context.AfterExecution c,ExecutionAttributes a) {
        Map<String,Object> log=new LinkedHashMap<>();log.put("http_status",c.httpResponse().statusCode());log.put("phase","normal");
        if(c.request() instanceof SendMessageRequest r && c.response() instanceof SendMessageResponse s) {
          log.put("api","SendMessage");log.put("queue",role(r.queueUrl()));log.put("message_id",s.messageId());log.put("job_id",bodyJob(r.messageBody()));
        } else if(c.request() instanceof ReceiveMessageRequest r && c.response() instanceof ReceiveMessageResponse s) {
          log.put("api","ReceiveMessage");log.put("queue",role(r.queueUrl()));
          List<Object> ids=new ArrayList<>();
          for(Message m:s.messages()) {
            long job=bodyJob(m.body());
            if(ownJob==null || job!=ownJob) {log.put("foreign_message_detected",true);journal(log);throw new IllegalStateException("foreign message: consumer stopped before handler");}
            receipts.put(m.receiptHandle(),m.messageId());ids.add(Map.of("message_id",m.messageId(),"job_id",job));
          }
          log.put("messages",ids);
        } else if(c.request() instanceof DeleteMessageRequest r) {
          log.put("api","DeleteMessage");log.put("queue",role(r.queueUrl()));log.put("message_id",receipts.get(r.receiptHandle()));
        }
        journal(log);
      }
      @Override public void onExecutionFailure(Context.FailedExecution c,ExecutionAttributes a) {
        Map<String,Object> log=new LinkedHashMap<>();log.put("api",c.request().getClass().getSimpleName().replace("Request",""));log.put("phase","normal");
        log.put("error_type",c.exception().getClass().getSimpleName());
        if(c.exception() instanceof SqsException e) {log.put("error_code",e.awsErrorDetails().errorCode());log.put("http_status",e.statusCode());}
        journal(log);
      }
    })).build();
 }

 FittingResultConsumer consumer() {
   return new FittingResultConsumer(sqs,sqsProperties,new FittingResultConsumerProperties(true,1,1,20),mapper,resultHandler,events);
 }
 void publishOwned() {
   Long foreign=jdbc.queryForObject("select count(*) from outbox_event where status in ('PENDING','PROCESSING') and aggregate_id<>?",Long.class,ownJob);
   if(foreign!=0)throw new IllegalStateException("unrelated outbox: publication refused");
   new OutboxEventScheduler(new FittingOutboxProperties(true,run,1,Duration.ofSeconds(30)),claims,publisher).publishPendingEvents();
 }
 @Test void serve() throws Exception {
   when(presign.createGetUrl(anyString())).thenAnswer(i->"http://127.0.0.1:18772/mock-image/"+i.getArgument(0));
   var server=HttpServer.create(new InetSocketAddress("127.0.0.1",18772),0);
   server.createContext("/",exchange->{
     try {
       Map body=mapper.readValue(new String(exchange.getRequestBody().readAllBytes()),Map.class);
       byte[] out=mapper.writeValueAsBytes(route(exchange.getRequestURI().getPath(),body));
       exchange.sendResponseHeaders(200,out.length);exchange.getResponseBody().write(out);
     }catch(Throwable e){
       byte[] out=mapper.writeValueAsBytes(Map.of("error_type",e.getClass().getSimpleName()));
       exchange.sendResponseHeaders(500,out.length);exchange.getResponseBody().write(out);
     }finally{exchange.close();}
   });
   server.start();
   try {if(!done.await(15,TimeUnit.MINUTES))throw new AssertionError("orchestrator timeout");}
   finally{if(timer!=null){timer.shutdown();timer.awaitTermination(30,TimeUnit.SECONDS);}server.stop(0);}
 }
 Object route(String path,Map body) throws Exception {
   if(path.equals("/health"))return Map.of("ready",true,"sqs_real_class",sqs.getClass().getName(),"profile","lookddak-sqs-local");
   if(path.equals("/shutdown")){done.countDown();return Map.of("stopping",true);}
   if(path.equals("/create")) {
     if(ownJob!=null)throw new IllegalStateException("one normal job only");
     String code=(String)body.get("code"), imageKey=(String)body.get("image_key");
     Member m=members.saveAndFlush(Member.create(run+"@local.invalid","test-only-encoded-password"));memberId=m.getId();
     profiles.saveAndFlush(MemberProfile.create(m,"SQS테스트",29,new BigDecimal("175"),new BigDecimal("70"),imageKey));
     Product p=products.saveAndFlush(Product.create(code,"SQS 통합 상의","http://127.0.0.1/product.jpg",10000,"NAVY",ProductItemType.TOP,"http://127.0.0.1/product"));productId=p.getId();
     candidates.saveAndFlush(FittingCandidate.create(m,p));
     var created=jobs.createFittingJob(memberId,new FittingJobCreateRequest(productId,null));ownJob=created.getFittingJobId();
     return Map.of("created",created,"member_id",memberId,"product_id",productId,"outbox",jdbc.queryForMap("select * from outbox_event where aggregate_id=?",ownJob));
   }
   if(path.equals("/publish")){publishOwned();return Map.of("cycle",true);}
   if(path.equals("/consume")){consumer().consume();return Map.of("cycle",true);}
   if(path.equals("/start")) {
     timer=Executors.newSingleThreadScheduledExecutor();
     timer.scheduleWithFixedDelay(()->{try{publishOwned();consumer().consume();}catch(Throwable e){timerError=e;}},0,200,TimeUnit.MILLISECONDS);
     return Map.of("running",true);
   }
   if(path.equals("/stop")) {
     if(timer!=null){timer.shutdown();if(!timer.awaitTermination(30,TimeUnit.SECONDS))throw new IllegalStateException("timer did not stop");}
     return Map.of("stopped",true,"timer_error",timerError==null?"":timerError.getClass().getSimpleName());
   }
   if(path.equals("/snapshot")) {
     if(ownJob==null)return Map.of("created",false);
     Map<String,Object> out=new LinkedHashMap<>();out.put("query",jobs.getFittingJobStatus(memberId,ownJob));
     out.put("job",jdbc.queryForMap("select * from fitting_job where id=?",ownJob));
     out.put("outbox",jdbc.queryForMap("select * from outbox_event where aggregate_id=?",ownJob));
     out.put("results",jdbc.queryForList("select * from fitting_temp_result where fitting_job_id=?",ownJob));
     out.put("notifications",jdbc.queryForList("select * from notification where target_id=? and member_id=?",ownJob,memberId));
     out.put("member",jdbc.queryForMap("select id,fitting_request_count from member where id=?",memberId));return out;
   }
   throw new IllegalArgumentException("unknown test route");
 }
}
