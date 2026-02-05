"""
동기화 알고리즘 모듈

PDF 페이지와 음성 자막을 동기화하기 위한 다양한 알고리즘을 제공합니다.

알고리즘 목록:
1. exact_matching - 키워드 정확 매칭
2. cosine_similarity - 코사인 유사도 기반
3. hybrid - 하이브리드 (키워드 + 코사인)
4. llm_transcription - LLM 전사 비교
5. structured_pdf - 구조화된 PDF (제목 기반)
6. llm_semantic - LLM 의미 기반 매칭 (NEW)

사용 예시:
```python
from app.sync_algorithms import (
    ExactMatchingAlgorithm,
    CosineSimilarityAlgorithm,
    HybridAlgorithm,
    StructuredPDFAlgorithm,
    LLMSemanticAlgorithm,
    SyncEvaluator,
    PageData,
    TranscriptSegment,
    ALGORITHMS,
    get_algorithm,
    run_benchmark,
)

# 단일 알고리즘 실행
algo = get_algorithm("hybrid")
result = algo.run(pages, segments)

# 전체 벤치마크 실행
from app.sync_algorithms.benchmark import run_full_benchmark
results = run_full_benchmark(lecture_ids=[1, 2, 3])
```
"""

# 기본 클래스 및 유틸리티
from .base import (
    PageData,
    TranscriptSegment,
    SegmentGroup,
    SyncAnchor,
    SyncResult,
    TextProcessor,
    SimilarityCalculator,
    BaseSyncAlgorithm,
    SegmentGrouper,
)

# 알고리즘 1: 키워드 정확 매칭
from .exact_matching import (
    ExactMatchingAlgorithm,
    exact_matching_sync,
)

# 알고리즘 2: 코사인 유사도
from .cosine_similarity import (
    CosineSimilarityAlgorithm,
    cosine_similarity_sync,
)

# 알고리즘 3: 하이브리드
from .hybrid import (
    HybridAlgorithm,
    hybrid_sync,
)

# 알고리즘 4: LLM 전사 비교
from .llm_transcription import (
    LLMTranscriptionAlgorithm,
    llm_transcription_sync,
)

# 알고리즘 5: 구조화된 PDF
from .structured_pdf import (
    StructuredPDFAlgorithm,
    structured_pdf_sync,
)

# 알고리즘 6: LLM 의미 기반 매칭
from .llm_semantic import (
    LLMSemanticAlgorithm,
    llm_semantic_sync,
)

# 평가 도구
from .evaluation import (
    GroundTruth,
    EvaluationResult,
    SyncEvaluator,
    GroundTruthManager,
    calculate_metrics,
    generate_evaluation_report,
)


__all__ = [
    # 기본 클래스
    "PageData",
    "TranscriptSegment",
    "SegmentGroup",
    "SyncAnchor",
    "SyncResult",
    "TextProcessor",
    "SimilarityCalculator",
    "BaseSyncAlgorithm",
    "SegmentGrouper",
    
    # 알고리즘
    "ExactMatchingAlgorithm",
    "exact_matching_sync",
    "CosineSimilarityAlgorithm",
    "cosine_similarity_sync",
    "HybridAlgorithm",
    "hybrid_sync",
    "LLMTranscriptionAlgorithm",
    "llm_transcription_sync",
    "StructuredPDFAlgorithm",
    "structured_pdf_sync",
    "LLMSemanticAlgorithm",
    "llm_semantic_sync",
    
    # 평가
    "GroundTruth",
    "EvaluationResult",
    "SyncEvaluator",
    "GroundTruthManager",
    "calculate_metrics",
    "generate_evaluation_report",
    
    # 유틸리티
    "ALGORITHMS",
    "get_algorithm",
    "list_algorithms",
]


# 알고리즘 레지스트리
ALGORITHMS = {
    "exact_matching": {
        "class": ExactMatchingAlgorithm,
        "name": "Exact Matching",
        "description": "키워드 정확 매칭 - 공통 키워드 기반",
        "requires_embedding": False,
        "requires_llm": False,
    },
    "cosine_similarity": {
        "class": CosineSimilarityAlgorithm,
        "name": "Cosine Similarity",
        "description": "코사인 유사도 - 임베딩 벡터 기반",
        "requires_embedding": True,
        "requires_llm": False,
    },
    "hybrid": {
        "class": HybridAlgorithm,
        "name": "Hybrid",
        "description": "하이브리드 - 키워드 + 코사인 결합",
        "requires_embedding": True,
        "requires_llm": False,
    },
    "llm_transcription": {
        "class": LLMTranscriptionAlgorithm,
        "name": "LLM Transcription",
        "description": "LLM 전사 - GPT-4로 PDF 전사 후 비교",
        "requires_embedding": True,
        "requires_llm": True,
    },
    "structured_pdf": {
        "class": StructuredPDFAlgorithm,
        "name": "Structured PDF",
        "description": "구조화 PDF - 제목 기반 유사도",
        "requires_embedding": True,
        "requires_llm": False,
    },
    "llm_semantic": {
        "class": LLMSemanticAlgorithm,
        "name": "LLM Semantic",
        "description": "LLM 의미 분석 - LLM으로 직접 매칭 판단",
        "requires_embedding": False,
        "requires_llm": True,
    },
}


def get_algorithm(name: str, embedding_fn=None, llm_fn=None, **kwargs):
    """알고리즘 인스턴스 생성
    
    Args:
        name: 알고리즘 이름
        embedding_fn: 임베딩 함수 (필요한 경우)
        llm_fn: LLM 함수 (필요한 경우)
        **kwargs: 알고리즘 초기화 파라미터
        
    Returns:
        알고리즘 인스턴스
    """
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}")
    
    algo_info = ALGORITHMS[name]
    algo_class = algo_info["class"]
    
    # 의존성 주입
    init_kwargs = kwargs.copy()
    
    if algo_info["requires_embedding"] and embedding_fn:
        init_kwargs["embedding_fn"] = embedding_fn
    
    if algo_info["requires_llm"] and llm_fn:
        init_kwargs["llm_fn"] = llm_fn
    
    return algo_class(**init_kwargs)


def list_algorithms() -> list:
    """사용 가능한 알고리즘 목록 반환"""
    return list(ALGORITHMS.keys())


def get_algorithm_info(name: str = None) -> dict:
    """알고리즘 정보 반환
    
    Args:
        name: 알고리즘 이름 (None이면 전체)
        
    Returns:
        알고리즘 정보 딕셔너리
    """
    if name:
        if name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {name}")
        info = ALGORITHMS[name].copy()
        info.pop("class")  # 클래스 객체 제외
        return info
    
    return {
        k: {key: val for key, val in v.items() if key != "class"}
        for k, v in ALGORITHMS.items()
    }
