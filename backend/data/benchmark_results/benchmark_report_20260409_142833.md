# 동기화 알고리즘 벤치마크 리포트

**생성 시간:** 20260409_142833

## 실험 설정

- **테스트 강의:** [1, 2, 3]
- **알고리즘:** exact_matching, cosine_similarity, hybrid, structured_pdf
- **그룹화:** duration (30.0초)
- **신뢰도 임계값:** 0.05
- **tolerance (정답 허용오차):** 10.0초

## 전체 결과 요약

### 평균 성능 비교 (4가지 지표)

| 알고리즘 | F1 | Precision | Recall | ROC-AUC | 실행시간 |
|---------|----:|----------:|-------:|--------:|---------:|
| exact_matching | 0.155±0.028 | 0.155±0.028 | 0.155±0.028 | 0.546±0.092 | 0.33s |
| cosine_similarity | - | - | - | - | - |
| hybrid | - | - | - | - | - |
| structured_pdf | - | - | - | - | - |

### 순위

- **F1 기준 최고:** exact_matching
- **ROC-AUC 기준 최고:** exact_matching

## 강의별 상세 결과

### Lecture 1

- 페이지 수: 23
- 세그먼트 수: 68
- Ground Truth: 23

| 알고리즘 | F1 | Precision | Recall | AUC | TP | FP | FN |
|---------|---:|----------:|-------:|----:|---:|---:|---:|
| exact_matching | 0.174 | 0.174 | 0.174 | 0.421 | 4 | 19 | 19 |
| cosine_similarity | ERROR | - | - | - | - | - | - |
| hybrid | ERROR | - | - | - | - | - | - |
| structured_pdf | ERROR | - | - | - | - | - | - |

### Lecture 2

- 페이지 수: 26
- 세그먼트 수: 59
- Ground Truth: 26

| 알고리즘 | F1 | Precision | Recall | AUC | TP | FP | FN |
|---------|---:|----------:|-------:|----:|---:|---:|---:|
| exact_matching | 0.115 | 0.115 | 0.115 | 0.580 | 3 | 23 | 23 |
| cosine_similarity | ERROR | - | - | - | - | - | - |
| hybrid | ERROR | - | - | - | - | - | - |
| structured_pdf | ERROR | - | - | - | - | - | - |

### Lecture 3

- 페이지 수: 43
- 세그먼트 수: 72
- Ground Truth: 40

| 알고리즘 | F1 | Precision | Recall | AUC | TP | FP | FN |
|---------|---:|----------:|-------:|----:|---:|---:|---:|
| exact_matching | 0.175 | 0.175 | 0.175 | 0.639 | 7 | 33 | 33 |
| cosine_similarity | ERROR | - | - | - | - | - | - |
| hybrid | ERROR | - | - | - | - | - | - |
| structured_pdf | ERROR | - | - | - | - | - | - |
