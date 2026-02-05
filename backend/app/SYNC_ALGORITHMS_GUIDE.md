# 동기화 알고리즘 모듈 사용 가이드

## 개요

이 모듈은 `auto_sync` 함수를 분리하여 여러 알고리즘을 실험하고 비교할 수 있도록 설계되었습니다.

## 파일 구조

```
app/
├── sync_algorithms/
│   ├── __init__.py           # 모듈 초기화 및 레지스트리
│   ├── base.py               # 기본 클래스 및 유틸리티
│   ├── exact_matching.py     # 1. 키워드 정확 매칭
│   ├── cosine_similarity.py  # 2. 코사인 유사도
│   ├── hybrid.py             # 3. 하이브리드 (1+2)
│   ├── llm_transcription.py  # 4. LLM 전사 비교
│   ├── structured_pdf.py     # 5. 구조화된 PDF (제목 기반)
│   └── evaluation.py         # 평가 도구 (F1, ROC, etc.)
└── sync_experiments.py       # 실험용 API 엔드포인트
```

## main.py에 통합하기

### 1. 임포트 추가

```python
# main.py 상단에 추가
from .sync_experiments import register_experiment_routes
```

### 2. 라우트 등록

```python
# app 인스턴스 생성 후 추가
app = FastAPI(title="AI Agent for Professors - MVP Backend")

# ... 기존 코드 ...

# 실험용 라우트 등록
register_experiment_routes(app, BASE_DIR)
```

## API 엔드포인트

### 알고리즘 목록 조회
```
GET /experiments/algorithms
```

### 단일 알고리즘 실행
```
POST /experiments/lectures/{lecture_id}/sync/{algorithm}

파라미터:
- algorithm: exact_matching, cosine_similarity, hybrid, structured_pdf
- grouping: none, duration, count (기본: none)
- group_duration: 10.0 (duration 그룹화 시)
- group_count: 5 (count 그룹화 시)
- confidence_threshold: 0.4
```

### 알고리즘 비교
```
POST /experiments/lectures/{lecture_id}/compare

Body:
{
    "algorithms": ["exact_matching", "cosine_similarity", "hybrid"],
    "grouping": "none",
    "group_duration": 10.0,
    "confidence_threshold": 0.4
}
```

### Ground Truth 설정
```
POST /experiments/lectures/{lecture_id}/ground_truth

Body:
[
    {"page": 1, "time": 0.0},
    {"page": 2, "time": 45.5},
    {"page": 3, "time": 120.0}
]
```

### 알고리즘 평가
```
POST /experiments/lectures/{lecture_id}/evaluate/{algorithm}?tolerance=5.0
```

### 그룹화 전략 테스트
```
GET /experiments/lectures/{lecture_id}/grouping_test
```

### 결과 적용
```
POST /experiments/lectures/{lecture_id}/apply_sync

Body:
{
    "algorithm": "hybrid"
}
```

## 실험 순서

### 1단계: 데이터 준비
```bash
# OCR 실행
POST /lectures/{lecture_id}/ocr_pdf

# 음성 인식 실행
POST /lectures/{lecture_id}/transcribe
```

### 2단계: 그룹화 테스트
```bash
# 그룹화 전략별 세그먼트 수 확인
GET /experiments/lectures/{lecture_id}/grouping_test
```

### 3단계: 알고리즘 비교
```bash
# 여러 알고리즘 동시 실행 및 비교
POST /experiments/lectures/{lecture_id}/compare
{
    "algorithms": ["exact_matching", "cosine_similarity", "hybrid", "structured_pdf"],
    "grouping": "none"  # 또는 "duration"
}
```

### 4단계: Ground Truth 설정 (선택)
```bash
# 정답 데이터 설정
POST /experiments/lectures/{lecture_id}/ground_truth
[
    {"page": 1, "time": 0.0},
    {"page": 2, "time": 45.5},
    ...
]
```

### 5단계: 평가
```bash
# F1-score, MAE, ROC 등 계산
POST /experiments/lectures/{lecture_id}/evaluate/hybrid?tolerance=5.0
```

### 6단계: 결과 적용
```bash
# 가장 좋은 알고리즘 결과를 실제 앵커로 적용
POST /experiments/lectures/{lecture_id}/apply_sync
{
    "algorithm": "hybrid"
}
```

## 알고리즘 설명

### 1. exact_matching (키워드 정확 매칭)
- 공통 키워드 개수 기반
- 빠른 계산, 명확한 해석
- 전문 용어가 많은 강의에 효과적

### 2. cosine_similarity (코사인 유사도)
- 임베딩 벡터 기반
- 동의어/유의어 처리 가능
- 의미적 유사성 포착

### 3. hybrid (하이브리드)
- 키워드 + 코사인 결합
- 가중치 조정 가능
- 두 방법의 장점 활용

### 4. structured_pdf (구조화된 PDF)
- 제목 기반 유사도
- 페이지 구조 활용
- 주제 전환 감지

### 5. llm_transcription (LLM 전사)
- GPT-4로 PDF 전사
- OCR보다 정확한 텍스트
- 비용 발생

## 평가 지표

- **MAE**: 평균 절대 오차 (초)
- **RMSE**: 제곱근 평균 제곱 오차
- **F1-score**: 정밀도와 재현율의 조화 평균
- **Precision**: 예측 중 정답 비율
- **Recall**: 정답 중 예측 비율
- **AUC**: ROC 곡선 아래 면적

## 그룹화 옵션

### none (그룹화 없음)
- 각 자막 세그먼트를 그대로 사용
- 세그먼트 수가 많아 세밀한 매칭 가능
- 노이즈에 취약할 수 있음

### duration (시간 기반)
- 일정 시간(초) 단위로 그룹화
- 예: 10초 단위 → 5분 영상은 30개 그룹

### count (개수 기반)
- 일정 개수의 세그먼트를 그룹화
- 예: 5개씩 → 100개 세그먼트는 20개 그룹

## 권장 설정

| 상황 | 알고리즘 | 그룹화 | 신뢰도 임계값 |
|------|---------|--------|--------------|
| 전문 용어 많음 | exact_matching | none | 0.3 |
| 일반 강의 | hybrid | duration 10s | 0.4 |
| 슬라이드 제목 명확 | structured_pdf | duration 10s | 0.45 |
| 고품질 필요 | cosine_similarity | duration 10s | 0.5 |

## 디버깅

### 유사도 행렬 확인
```bash
GET /lectures/{lecture_id}/similarity_matrix
```

### 동기화 디버그 정보
```bash
GET /lectures/{lecture_id}/sync_debug
```

### 결과 파일 위치
```
data/lectures/{lecture_id}/
├── sync_result_{algorithm}.json      # 알고리즘별 결과
├── similarity_matrix_{algorithm}.json # 유사도 행렬
├── ground_truth.json                 # 정답 데이터
└── sync_debug.json                   # 디버그 정보
```
