"""
알고리즘 3: 하이브리드 (Hybrid)

이 알고리즘은 키워드 정확 매칭과 코사인 유사도를 결합하여
두 방법의 장점을 활용합니다.

장점:
- 키워드 매칭의 정확성 + 임베딩의 의미적 유사성
- 전문 용어와 일반 텍스트 모두 처리 가능
- 조정 가능한 가중치로 유연한 튜닝

단점:
- 두 알고리즘 모두 실행해야 하므로 계산 비용 증가
- 가중치 조정이 필요할 수 있음
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np

from .base import (
    BaseSyncAlgorithm,
    PageData,
    TranscriptSegment,
    TextProcessor,
    SimilarityCalculator,
    SyncResult,
    SyncAnchor,
)
from .exact_matching import ExactMatchingAlgorithm
from .cosine_similarity import CosineSimilarityAlgorithm


class HybridAlgorithm(BaseSyncAlgorithm):
    """하이브리드 동기화 알고리즘 (키워드 + 코사인 유사도)"""
    
    def __init__(
        self,
        embedding_fn: Optional[Callable] = None,
        keyword_weight: float = 0.3,
        cosine_weight: float = 0.7
    ):
        """
        Args:
            embedding_fn: 텍스트를 임베딩으로 변환하는 함수
            keyword_weight: 키워드 매칭 가중치 (0 ~ 1)
            cosine_weight: 코사인 유사도 가중치 (0 ~ 1)
        """
        super().__init__(
            name="hybrid",
            description="하이브리드 동기화 (키워드 + 코사인 유사도)"
        )
        self.exact_algo = ExactMatchingAlgorithm()
        self.cosine_algo = CosineSimilarityAlgorithm(embedding_fn=embedding_fn)
        self.keyword_weight = keyword_weight
        self.cosine_weight = cosine_weight
    
    def set_embedding_function(self, embedding_fn: Callable):
        """임베딩 함수 설정"""
        self.cosine_algo.set_embedding_function(embedding_fn)
    
    def set_weights(self, keyword_weight: float, cosine_weight: float):
        """가중치 설정
        
        Args:
            keyword_weight: 키워드 매칭 가중치
            cosine_weight: 코사인 유사도 가중치
        """
        self.keyword_weight = keyword_weight
        self.cosine_weight = cosine_weight
    
    def compute_similarity(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        keyword_scoring: str = "overlap_ratio",
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """하이브리드 유사도 행렬 계산
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            keyword_scoring: 키워드 점수 계산 방법
            normalize: 정규화 여부
            
        Returns:
            유사도 행렬 (num_pages x num_segments)
        """
        # 키워드 매칭 유사도
        keyword_matrix = self.exact_algo.compute_similarity(
            pages, segments,
            scoring_method=keyword_scoring
        )
        
        # 코사인 유사도
        cosine_matrix = self.cosine_algo.compute_similarity(
            pages, segments, **kwargs
        )
        
        # 정규화
        if normalize:
            # Min-Max 정규화
            if keyword_matrix.max() > keyword_matrix.min():
                keyword_matrix = (keyword_matrix - keyword_matrix.min()) / (keyword_matrix.max() - keyword_matrix.min())
            
            # 코사인 유사도는 이미 -1 ~ 1 범위이므로 0 ~ 1로 변환
            cosine_matrix = (cosine_matrix + 1) / 2
        
        # 가중 평균
        hybrid_matrix = (
            self.keyword_weight * keyword_matrix +
            self.cosine_weight * cosine_matrix
        )
        
        return hybrid_matrix
    
    def compute_similarity_detailed(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """상세 유사도 행렬 반환 (개별 행렬 포함)
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            
        Returns:
            개별 행렬과 결합 행렬
        """
        keyword_matrix = self.exact_algo.compute_similarity(
            pages, segments, scoring_method="overlap_ratio"
        )
        
        cosine_matrix = self.cosine_algo.compute_similarity(
            pages, segments, **kwargs
        )
        
        # 정규화
        keyword_normalized = keyword_matrix.copy()
        if keyword_normalized.max() > keyword_normalized.min():
            keyword_normalized = (keyword_normalized - keyword_normalized.min()) / (keyword_normalized.max() - keyword_normalized.min())
        
        cosine_normalized = (cosine_matrix + 1) / 2
        
        hybrid_matrix = (
            self.keyword_weight * keyword_normalized +
            self.cosine_weight * cosine_normalized
        )
        
        return {
            "keyword_raw": keyword_matrix,
            "keyword_normalized": keyword_normalized,
            "cosine_raw": cosine_matrix,
            "cosine_normalized": cosine_normalized,
            "hybrid": hybrid_matrix
        }
    
    def find_optimal_weights(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        ground_truth: List[SyncAnchor],
        weight_range: List[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """최적 가중치 탐색 (그라운드 트루스 필요)
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            ground_truth: 정답 앵커 리스트
            weight_range: 탐색할 가중치 범위
            
        Returns:
            최적 가중치 및 성능 지표
        """
        if weight_range is None:
            weight_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # 개별 행렬 계산 (한 번만)
        keyword_matrix = self.exact_algo.compute_similarity(
            pages, segments, scoring_method="overlap_ratio"
        )
        cosine_matrix = self.cosine_algo.compute_similarity(
            pages, segments, **kwargs
        )
        
        # 정규화
        keyword_norm = keyword_matrix.copy()
        if keyword_norm.max() > keyword_norm.min():
            keyword_norm = (keyword_norm - keyword_norm.min()) / (keyword_norm.max() - keyword_norm.min())
        cosine_norm = (cosine_matrix + 1) / 2
        
        # 그라운드 트루스를 딕셔너리로 변환
        gt_dict = {a.page: a.time for a in ground_truth}
        
        results = []
        for kw in weight_range:
            cw = 1.0 - kw
            
            # 하이브리드 행렬
            hybrid_matrix = kw * keyword_norm + cw * cosine_norm
            
            # 최적 경로 탐색
            path = self.find_optimal_path(hybrid_matrix)
            
            # 성능 평가 (MAE: Mean Absolute Error)
            errors = []
            for page_idx, seg_idx, score in path:
                page = page_idx + 1
                pred_time = segments[seg_idx].start
                
                if page in gt_dict:
                    gt_time = gt_dict[page]
                    errors.append(abs(pred_time - gt_time))
            
            mae = np.mean(errors) if errors else float('inf')
            
            results.append({
                "keyword_weight": kw,
                "cosine_weight": cw,
                "mae": mae,
                "matched_pages": len(errors)
            })
        
        # 최적 가중치 찾기
        best = min(results, key=lambda x: x["mae"])
        
        return {
            "best_weights": {
                "keyword": best["keyword_weight"],
                "cosine": best["cosine_weight"]
            },
            "best_mae": best["mae"],
            "all_results": results
        }
    
    def run_with_analysis(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.4,
        **kwargs
    ) -> SyncResult:
        """분석 정보와 함께 동기화 실행
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            동기화 결과 (분석 정보 포함)
        """
        # 상세 행렬 계산
        matrices = self.compute_similarity_detailed(pages, segments, **kwargs)
        
        # 최적 경로 탐색
        path = self.find_optimal_path(matrices["hybrid"])
        
        # 신뢰도 높은 앵커 선별
        reliable_anchors = []
        for page_idx, seg_idx, score in path:
            if score >= confidence_threshold:
                reliable_anchors.append(SyncAnchor(
                    page=page_idx + 1,
                    time=segments[seg_idx].start,
                    confidence=score,
                    method=self.name
                ))
        
        # 보간
        total_duration = segments[-1].end if segments else 0
        final_anchors = self.interpolate_anchors(
            reliable_anchors, len(pages), total_duration
        )
        
        # 결과 구성
        result = SyncResult(
            anchors=final_anchors,
            similarity_matrix=matrices["hybrid"],
            debug_info={
                "algorithm": self.name,
                "weights": {
                    "keyword": self.keyword_weight,
                    "cosine": self.cosine_weight
                },
                "num_pages": len(pages),
                "num_segments": len(segments),
                "matched_path": [(p+1, s, sc) for p, s, sc in path],
                "reliable_count": len(reliable_anchors),
                "confidence_threshold": confidence_threshold,
                "individual_matrices": {
                    "keyword_stats": {
                        "min": float(matrices["keyword_raw"].min()),
                        "max": float(matrices["keyword_raw"].max()),
                        "mean": float(matrices["keyword_raw"].mean())
                    },
                    "cosine_stats": {
                        "min": float(matrices["cosine_raw"].min()),
                        "max": float(matrices["cosine_raw"].max()),
                        "mean": float(matrices["cosine_raw"].mean())
                    }
                }
            }
        )
        
        return result


# 편의 함수
def hybrid_sync(
    pages: List[PageData],
    segments: List[TranscriptSegment],
    embedding_fn: Callable,
    keyword_weight: float = 0.3,
    cosine_weight: float = 0.7,
    confidence_threshold: float = 0.4,
    **kwargs
) -> SyncResult:
    """하이브리드 동기화 실행
    
    Args:
        pages: 페이지 데이터 리스트
        segments: 자막 세그먼트 리스트
        embedding_fn: 임베딩 함수
        keyword_weight: 키워드 가중치
        cosine_weight: 코사인 유사도 가중치
        confidence_threshold: 신뢰도 임계값
        
    Returns:
        동기화 결과
    """
    algorithm = HybridAlgorithm(
        embedding_fn=embedding_fn,
        keyword_weight=keyword_weight,
        cosine_weight=cosine_weight
    )
    return algorithm.run_with_analysis(
        pages, segments,
        confidence_threshold=confidence_threshold,
        **kwargs
    )
