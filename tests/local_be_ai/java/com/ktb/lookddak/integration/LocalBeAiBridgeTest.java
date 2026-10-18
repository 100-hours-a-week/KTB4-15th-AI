package com.ktb.lookddak.integration;

import com.ktb.lookddak.domain.fitting.dto.FittingJobCreateRequest;
import com.ktb.lookddak.domain.fitting.entity.FittingCandidate;
import com.ktb.lookddak.domain.fitting.repository.FittingCandidateRepository;
import com.ktb.lookddak.domain.fitting.service.FittingJobService;
import com.ktb.lookddak.domain.fitting.mq.FittingResultConsumer;
import com.ktb.lookddak.domain.fitting.mq.FittingResultConsumerProperties;
import com.ktb.lookddak.domain.fitting.mq.outbox.OutboxEventClaimService;
import com.ktb.lookddak.domain.fitting.mq.outbox.OutboxEventPublisher;
import com.ktb.lookddak.domain.fitting.mq.outbox.OutboxEventScheduler;
import com.ktb.lookddak.domain.fitting.mq.outbox.FittingOutboxProperties;
import com.ktb.lookddak.domain.member.entity.Member;
import com.ktb.lookddak.domain.member.entity.MemberProfile;
import com.ktb.lookddak.domain.member.repository.MemberRepository;
import com.ktb.lookddak.domain.member.repository.MemberProfileRepository;
import com.ktb.lookddak.domain.product.entity.Product;
import com.ktb.lookddak.domain.product.entity.ProductItemType;
import com.ktb.lookddak.domain.product.repository.ProductRepository;
import com.ktb.lookddak.global.storage.s3.S3PresignedUrlProvider;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.*;
import tools.jackson.databind.ObjectMapper;
import java.math.BigDecimal;
import java.net.*;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.*;

/** Local-only test control plane. Every job is created by the real transactional service. */
@SpringBootTest(properties = {
 "spring.datasource.url=jdbc:mysql://127.0.0.1:13306/lookddak_be_test?allowPublicKeyRetrieval=true&useSSL=false&serverTimezone=UTC",
 "spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver",
 "spring.datasource.username=be_test", "spring.datasource.password=local_be_password",
 "spring.jpa.hibernate.ddl-auto=update",
 "aws.sqs.fitting-request-queue-url=https://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-request",
 "aws.sqs.fitting-response-queue-url=https://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-response",
 "fitting.outbox.enabled=false", "fitting.result-consumer.enabled=false",
 "fitting.result-consumer.wait-time-seconds=0", "fitting.result-consumer.visibility-timeout-seconds=2",
 "fitting.result-consumer.poll-interval=24h", "fitting.outbox.poll-interval=24h",
 "sentry.dsn=", "sentry.logging.enabled=false"
})
class LocalBeAiBridgeTest {
 @Autowired FittingJobService jobs;
 @Autowired MemberRepository members;
 @Autowired MemberProfileRepository profiles;
 @Autowired ProductRepository products;
 @Autowired FittingCandidateRepository candidates;
 @Autowired OutboxEventClaimService claims;
 @Autowired OutboxEventPublisher publisher;
 @Autowired FittingResultConsumer consumer;
 @MockitoBean FittingResultConsumerProperties consumerProperties;
 @Autowired JdbcTemplate jdbc;
 @Autowired ObjectMapper mapper;
 @MockitoBean SqsClient sqs;
 @MockitoBean S3PresignedUrlProvider presign;
 final Map<Long,Long> owners = new LinkedHashMap<>();
 final String run = System.getenv("LOCAL_INTEGRATION_RUN");
 final HttpClient http = HttpClient.newHttpClient();
 final CountDownLatch done = new CountDownLatch(1);
 java.util.concurrent.ScheduledExecutorService automatic;
 volatile Throwable automaticError;

 void faultDdl(String sql) {
   // MySQL binary logging requires SUPER to CREATE TRIGGER. Use the Compose-local root
   // connection ONLY for this run-named trigger; application transactions remain be_test.
   try(var connection=java.sql.DriverManager.getConnection(
       "jdbc:mysql://127.0.0.1:13306/lookddak_be_test?allowPublicKeyRetrieval=true&useSSL=false",
       "root","local_root_password"); var statement=connection.createStatement()) {
     statement.execute(sql);
   } catch(java.sql.SQLException error) { throw new IllegalStateException(error); }
 }

 @SuppressWarnings("unchecked")
 Map<String,Object> rpc(String operation, Map<String,Object> args) throws Exception {
   var request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:18761/" + operation))
     .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(args))).build();
   var response = http.send(request, HttpResponse.BodyHandlers.ofString());
   if (response.statusCode() != 200) throw new IllegalStateException("local queue injected failure");
   return mapper.readValue(response.body(), Map.class);
 }

 @Test void serveRealBusinessCode() throws Exception {
   try(var probe=new java.net.Socket()) {
     probe.connect(new InetSocketAddress("1.1.1.1",443),1000);
     throw new AssertionError("Java external egress guard missing");
   } catch(java.io.IOException expected) {
     if(!expected.toString().contains("Operation not permitted")) throw expected;
   }
   when(presign.createGetUrl(anyString())).thenAnswer(i -> "http://127.0.0.1:18761/objects/" + i.getArgument(0));
   when(sqs.sendMessage(any(SendMessageRequest.class))).thenAnswer(i -> {
     SendMessageRequest r=i.getArgument(0);
     var out=rpc("send_message", Map.of("QueueUrl",r.queueUrl(),"MessageBody",r.messageBody()));
     return SendMessageResponse.builder().messageId((String)out.get("MessageId")).build();
   });
   when(sqs.receiveMessage(any(ReceiveMessageRequest.class))).thenAnswer(i -> {
     ReceiveMessageRequest r=i.getArgument(0);
     var out=rpc("receive_message", Map.of("QueueUrl",r.queueUrl(),"MaxNumberOfMessages",r.maxNumberOfMessages(),"VisibilityTimeout",r.visibilityTimeout()));
     List<Message> result=new ArrayList<>();
     for (Object value : (List<?>)out.getOrDefault("Messages",List.of())) {
       Map<?,?> m=(Map<?,?>)value;
       result.add(Message.builder().messageId((String)m.get("MessageId")).body((String)m.get("Body")).receiptHandle((String)m.get("ReceiptHandle")).build());
     }
     return ReceiveMessageResponse.builder().messages(result).build();
   });
   when(sqs.deleteMessage(any(DeleteMessageRequest.class))).thenAnswer(i -> {
     DeleteMessageRequest r=i.getArgument(0);
     rpc("delete_message",Map.of("QueueUrl",r.queueUrl(),"ReceiptHandle",r.receiptHandle()));
     return DeleteMessageResponse.builder().build();
   });
   // Start consumption only after the transport stubs exist; Spring's initial scheduled tick
   // safely sees disabled defaults while its application context is being created.
   when(consumerProperties.getMaxMessages()).thenReturn(10);
   when(consumerProperties.getWaitTimeSeconds()).thenReturn(0);
   when(consumerProperties.getVisibilityTimeoutSeconds()).thenReturn(2);
   when(consumerProperties.isEnabled()).thenReturn(true);
   var server=HttpServer.create(new InetSocketAddress("127.0.0.1",18762),0);
   server.createContext("/", exchange -> {
     try {
       Map body=mapper.readValue(new String(exchange.getRequestBody().readAllBytes()),Map.class);
       Object result=route(exchange.getRequestURI().getPath(),body);
       byte[] bytes=mapper.writeValueAsBytes(result);
       exchange.sendResponseHeaders(200,bytes.length); exchange.getResponseBody().write(bytes);
     } catch(Throwable error) {
       byte[] bytes=error.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
       exchange.sendResponseHeaders(500,bytes.length); exchange.getResponseBody().write(bytes);
     } finally { exchange.close(); }
   });
   server.start();
   try { if(!done.await(30,java.util.concurrent.TimeUnit.MINUTES)) throw new AssertionError("orchestrator timeout"); }
   finally { if(automatic!=null) automatic.shutdownNow(); server.stop(0); }
 }

 Object route(String path, Map body) {
   if(path.equals("/health")) return Map.of("ready",true,"run",run,"java_egress_blocked",true);
   if(path.equals("/shutdown")) { done.countDown(); return Map.of("stopping",true); }
   if(path.equals("/create")) {
     String label=(String)body.get("label"), code=(String)body.get("code");
     Member member=members.saveAndFlush(Member.create(run+"-"+label+"@local.invalid","local-test-only"));
     profiles.saveAndFlush(MemberProfile.create(member,"로컬테스트",29,new BigDecimal("175"),new BigDecimal("70"),"users/10/body.png"));
     Product product=products.findAll().stream().filter(p->p.getProductCode().equals(code)).findFirst().orElseGet(()->products.saveAndFlush(Product.create(code,"테스트 상의","http://127.0.0.1:18761/product.jpg",10000,"NAVY",ProductItemType.TOP,"http://127.0.0.1:18761/product")));
     candidates.saveAndFlush(FittingCandidate.create(member,product));
     var response=jobs.createFittingJob(member.getId(),new FittingJobCreateRequest(product.getId(),null));
     owners.put(response.getFittingJobId(),member.getId());
     return Map.of("created",response,"member_id",member.getId(),"product_id",product.getId(),"outbox",jdbc.queryForMap("select * from outbox_event where aggregate_id=?",response.getFittingJobId()));
   }
   if(path.equals("/publish")) {
     // Claim only events created by this test, so unrelated pre-existing jobs are never published.
     // Refuse to run if another claimable event is present (the actual claim query remains unchanged).
     Long foreign=jdbc.queryForObject("select count(*) from outbox_event where status in ('PENDING','PROCESSING') and aggregate_id not in ("+String.join(",",owners.keySet().stream().map(String::valueOf).toList())+")",Long.class);
     if(foreign!=0) throw new IllegalStateException("unrelated outbox present; refusing claim");
     new OutboxEventScheduler(new FittingOutboxProperties(true,run,20,Duration.ofSeconds(30)),claims,publisher).publishPendingEvents();
     return Map.of("published_cycle",true);
   }
   if(path.equals("/auto-start")) {
     automatic=java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
     automatic.scheduleWithFixedDelay(()-> {
       try { route("/publish",Map.of()); consumer.consume(); }
       catch(Throwable error) { automaticError=error; }
     },200,200,java.util.concurrent.TimeUnit.MILLISECONDS);
     return Map.of("automatic",true);
   }
   if(path.equals("/auto-stop")) {
     automatic.shutdownNow();
     if(automaticError!=null) throw new IllegalStateException(automaticError);
     return Map.of("automatic",false);
   }
   if(path.equals("/consume")) {
     consumer.consume();
     return Map.of("consumed_cycle",true);
   }
   long id=((Number)body.get("job_id")).longValue();
   if(!owners.containsKey(id)) throw new IllegalArgumentException("not this run's job");
   String trigger="local_fail_"+run;
   if(path.equals("/fail-save")) {
     faultDdl("CREATE TRIGGER "+trigger+" BEFORE INSERT ON fitting_temp_result FOR EACH ROW BEGIN IF NEW.fitting_job_id="+id+" THEN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='local integration injected persistence failure'; END IF; END");
     return Map.of("trigger",trigger);
   }
   if(path.equals("/restore-save")) { faultDdl("DROP TRIGGER IF EXISTS "+trigger); return Map.of("restored",true); }
   if(path.equals("/snapshot")) {
     Map<String,Object> out=new LinkedHashMap<>();
     out.put("query",jobs.getFittingJobStatus(owners.get(id),id));
     out.put("job",jdbc.queryForMap("select * from fitting_job where id=?",id));
     out.put("outbox",jdbc.queryForMap("select * from outbox_event where aggregate_id=?",id));
     out.put("results",jdbc.queryForList("select * from fitting_temp_result where fitting_job_id=?",id));
     out.put("notifications",jdbc.queryForList("select * from notification where target_id=? and member_id=?",id,owners.get(id)));
     out.put("member",jdbc.queryForMap("select id,fitting_request_count from member where id=?",owners.get(id)));
     return out;
   }
   throw new IllegalArgumentException(path);
 }
}
