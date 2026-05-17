# ChromaDB 및 벡터 스토어(Vector Store) 구성 개요

본 문서는 `SocrAItes` 프로젝트에서 RAG(Retrieval-Augmented Generation) 파이프라인 구축을 위해 사용하는 **ChromaDB의 도입 목적, 핵심 동작 로직, 그리고 임베딩 모델(Embedding Model) 구성 방식**을 설명합니다.

관련 소스 코드: `src/rag/vectorstore.py`

---

## 1. ChromaDB 도입 목적

`SocrAItes` 에이전트는 강의 자료(PDF 등)를 기반으로 학생들에게 소크라테스 문답법을 통한 맞춤형 교육을 제공합니다. 이를 위해 외부 지식을 효율적으로 검색하고 LLM에 제공하는 **RAG 시스템**이 필수적이며, 경량 로컬 벡터 데이터베이스인 **ChromaDB**를 채택했습니다.

* **효율적인 지식 검색 (Semantic Search):** 텍스트의 의미적 유사도를 기반으로 질문과 가장 관련성이 높은 문서 청크(Chunk)를 신속하게 추출합니다.
* **로컬 환경 최적화 및 영구 저장 (Persistence):** 별도의 무거운 DB 서버를 구축할 필요 없이 `./chroma_db` 폴더에 파일 형태로 데이터를 영구 저장하므로, 설치가 간편하고 MVP 및 로컬 구동에 최적화되어 있습니다.
* **메타데이터 관리:** 텍스트뿐만 아니라 출처 파일명, 페이지 번호 등의 메타데이터를 함께 저장하여 인용 및 출처 추적에 용이합니다.

---

## 2. 핵심 동작 로직

### ① 클라이언트 및 컬렉션 초기화
* **Persistent Client 생성:** 
  환경 변수 `CHROMA_PERSIST_DIR`(기본값: `./chroma_db`) 경로를 기반으로 로컬 파일 시스템에 저장소를 구축합니다.
* **컬렉션(Collection) 연결:** 
  `"socratic_docs"`라는 이름의 컬렉션을 관리합니다. 컬렉션 로드 시 사전에 정의된 임베딩 함수(`embedding_function`)를 자동으로 매핑하여 데이터 삽입 및 검색 시 일관된 임베딩 처리가 이루어지도록 합니다.

### ② 문서 적재 (Ingestion & deduplication)
* **함수명:** `add_documents(docs, metadatas, ids)`
* **중복 방지 처리:** 
  기존 컬렉션에 적재된 문서들의 `ids`를 사전에 조회(`existing_ids`)하여, 전달받은 청크 중 이미 존재하는 ID는 제외하고 새로운 문서만 선별적으로 추가합니다.
* **자동 식별자 부여:** 
  명시적인 `ids`가 전달되지 않은 경우, 내부적으로 `uuid.uuid4()`를 사용하여 고유한 ID를 자동으로 할당합니다.

### ③ 유사도 검색 (Querying)
* **함수명:** `query(query_text, k=5)`
* **검색 흐름:** 
  사용자의 질문(`query_text`)을 인자로 받아 컬렉션에 조회를 요청합니다. ChromaDB가 자동으로 질문을 임베딩 벡터로 변환하고, 저장된 문서 벡터들과의 **거리(Distance)**를 계산하여 가장 유사도가 높은 상위 $k$개의 문서 청크와 해당 거리 값을 튜플 형태로 반환합니다.

---

## 3. 임베딩 모델(Embedding Model) 구성 방식

텍스트를 벡터로 변환하는 임베딩 함수는 유연성을 위해 **환경 변수(`OPENAI_API_KEY`)의 유무에 따라 조건부로 구성**됩니다.

```python
if os.getenv("OPENAI_API_KEY"):
    embeddings = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
else:
    # API 키가 없을 경우의 기본 폴백(Fallback)
    embeddings = embedding_functions.DefaultEmbeddingFunction()
```

### 1) OpenAI API Key가 존재하는 경우
* **사용 클래스:** `OpenAIEmbeddingFunction`
* **사용 모델:** **`text-embedding-3-small`**
* **특징:** 고성능이면서도 비용 효율적인 최신 소형 임베딩 모델을 사용하여, 고품질의 다국어 및 의미 검색 성능을 보장합니다.

### 2) OpenAI API Key가 없는 경우 (Fallback)
* **사용 클래스:** `DefaultEmbeddingFunction`
* **사용 모델:** **`all-MiniLM-L6-v2`** (ChromaDB 내장 기본 모델)
* **특징:** 외부 API 연결이나 과금 없이도 로컬에서 오프라인으로 텍스트 임베딩을 수행할 수 있도록 지원하여, 환경 설정이 완비되지 않은 상태에서도 시스템이 동작하도록 보장합니다.
