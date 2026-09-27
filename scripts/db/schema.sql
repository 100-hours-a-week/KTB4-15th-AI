-- 상품 원본 테이블. AI 쪽 RDB가 원본이고, 백엔드 RDB는 추천된 상품만 사본으로 가진다.
-- 두 DB가 같은 상품을 가리키는 키는 29cm의 product_code 다.

CREATE EXTENSION IF NOT EXISTS vector;

-- LangGraph checkpointer 전용 스키마. CHECKPOINT_DSN 의 search_path 와 이름이 같아야 한다.
-- checkpointer.setup() 은 테이블만 만들고 스키마는 만들지 않는다.
CREATE SCHEMA IF NOT EXISTS langgraph;

-- sabu: 식별자 — product_code 가 PK가 된 지금, CSV의 id 컬럼은 이 테이블에서 무슨 역할을 하는가? 다시 적재해도 같은 값이 나오는가?
CREATE TABLE IF NOT EXISTS products (
    product_code  bigint       PRIMARY KEY,
    id            integer      NOT NULL,
    product_name  text         NOT NULL,
    price         integer      NOT NULL,
    color         text,
    detail_url    text         NOT NULL,
    image_url     text         NOT NULL,
    main_category text         NOT NULL,
    sub_category  text         NOT NULL,
    is_sold_out   boolean      NOT NULL,
    crawled_at    timestamptz  NOT NULL
);

-- 이미지 → 키워드 → LLM 설명 → 요약/임베딩 파이프라인의 산출물. 생성 전에는 NULL.
-- 이름은 docs/5단계.md §상품 데이터 구축을 따른다.
--   description_summary: 사용자에게 보여줄 문장. 글자 수 상한은 적재 전에 코드에서 자른다.
--   embedding: image_description 의 벡터 (text-embedding-3-small, 1536차원 — docs/6단계.md)
-- sabu: 파생 데이터 신선도 — 재크롤링으로 image_url 이 바뀐 상품의 이 컬럼들은 무엇을 설명하고 있는가? 그걸 어떻게 알아챌 것인가?
ALTER TABLE products
    ADD COLUMN IF NOT EXISTS image_attribute_keywords text[],
    ADD COLUMN IF NOT EXISTS colors                   text[],
    ADD COLUMN IF NOT EXISTS image_description        text,
    ADD COLUMN IF NOT EXISTS description_summary      text,
    ADD COLUMN IF NOT EXISTS embedding                vector(1536);

-- 검색 기반 추천의 색상 필터 (2026-09-23 결정).
--   color  : 메인 색상 한 개. "메인 일치"와 "로고 등만 겹침"을 가르는 데 쓴다.
--   colors : 이미지에서 뽑은 색상 전부. main_color 를 빼지 않고 그대로 담는다.
-- 필터는 colors 한 컬럼으로 건다: WHERE colors && ARRAY['화이트']
-- sabu: 같은 사실이 두 컬럼에 있다 — color 가 colors 안에 없는 행이 생기면 무엇이 맞는 값인가? 적재할 때 그걸 막는가, 읽을 때 알아채는가?
CREATE INDEX IF NOT EXISTS products_colors_gin ON products USING gin (colors);
