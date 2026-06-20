# 동기화 알고리즘 벤치마크 리포트

**생성 시간:** 20260327_025141

## 실험 설정

- **테스트 강의:** [1, 2, 3]
- **알고리즘:** exact_matching, cosine_similarity, hybrid, llm_transcription, structured_pdf, llm_semantic
- **그룹화:** none (30.0초)
- **신뢰도 임계값:** 0.4
- **tolerance (정답 허용오차):** 10.0초

## 전체 결과 요약

### 평균 성능 비교 (4가지 지표)

| 알고리즘 | F1 | Precision | Recall | ROC-AUC | 실행시간 |
|---------|----:|----------:|-------:|--------:|---------:|
| exact_matching | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.42s |
| cosine_similarity | - | - | - | - | - |
| hybrid | - | - | - | - | - |
| llm_transcription | - | - | - | - | - |
| structured_pdf | - | - | - | - | - |
| llm_semantic | - | - | - | - | - |

### 순위

- **F1 기준 최고:** exact_matching
- **ROC-AUC 기준 최고:** exact_matching

## 강의별 상세 결과

### Lecture 1

- 페이지 수: 23
- 세그먼트 수: 406
- Ground Truth: 23

| 알고리즘 | F1 | Precision | Recall | AUC | TP | FP | FN |
|---------|---:|----------:|-------:|----:|---:|---:|---:|
| exact_matching | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 23 |
| cosine_similarity | ERROR | - | - | - | - | - | - |
| hybrid | ERROR | - | - | - | - | - | - |
| llm_transcription | ERROR | - | - | - | - | - | - |
| structured_pdf | ERROR | - | - | - | - | - | - |
| llm_semantic | ERROR | - | - | - | - | - | - |

### Lecture 2

- 페이지 수: 26
- 세그먼트 수: 390
- Ground Truth: 26

| 알고리즘 | F1 | Precision | Recall | AUC | TP | FP | FN |
|---------|---:|----------:|-------:|----:|---:|---:|---:|
| exact_matching | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 26 |
| cosine_similarity | ERROR | - | - | - | - | - | - |
| hybrid | ERROR | - | - | - | - | - | - |
| llm_transcription | ERROR | - | - | - | - | - | - |
| structured_pdf | ERROR | - | - | - | - | - | - |
| llm_semantic | ERROR | - | - | - | - | - | - |

### Lecture 3

- 페이지 수: 43
- 세그먼트 수: 371
- Ground Truth: 40

| 알고리즘 | F1 | Precision | Recall | AUC | TP | FP | FN |
|---------|---:|----------:|-------:|----:|---:|---:|---:|
| exact_matching | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 40 |
| cosine_similarity | ERROR | - | - | - | - | - | - |
| hybrid | ERROR | - | - | - | - | - | - |
| llm_transcription | ERROR | - | - | - | - | - | - |
| structured_pdf | ERROR | - | - | - | - | - | - |
| llm_semantic | ERROR | - | - | - | - | - | - |
