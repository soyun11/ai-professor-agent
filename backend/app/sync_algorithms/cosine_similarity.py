"""
알고리즘 2: 코사인 유사도 (Cosine Similarity)

이 알고리즘은 텍스트를 임베딩 벡터로 변환한 후,
코사인 유사도를 기반으로 유사도를 계산합니다.

장점:
- 의미적 유사성 포착 가능
- 동의어/유의어 처리 가능
- 문맥 이해 가능

단점:
- 임베딩 생성에 시간/비용 소요
- 전문 용어에 약할 수 있음
- 짧은 텍스트에서 노이즈 발생 가능
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np

from .base import (
    BaseSyncAlgorithm,
    PageData,
    TranscriptSegment,
    SegmentGroup,
    TextProcessor,
    SimilarityCalculator,
    SyncResult,
    SyncAnchor,
)


class CosineSimilarityAlgorithm(BaseSyncAlgorithm):
    """코사인 유사도 기반 동기화 알고리즘"""
    
    def __init__(self, embedding_fn: Optional[Callable] = None):
        """
        Args:
            embedding_fn: 텍스트를 임베딩으로 변환하는 함수
                         signature: (List[str]) -> List[List[float]]
        """
        super().__init__(
            name="cosine_similarity",
            description="코사인 유사도 기반 동기화"
        )
        self.embedding_fn = embedding_fn
    
    def set_embedding_function(self, embedding_fn: Callable):
        """임베딩 함수 설정
        
        Args:
            embedding_fn: 텍스트를 임베딩으로 변환하는 함수
        """
        self.embedding_fn = embedding_fn
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            임베딩 리스트
        """
        if self.embedding_fn is None:
            raise ValueError("임베딩 함수가 설정되지 않았습니다. set_embedding_function()을 호출하세요.")
        
        # 빈 텍스트 처리
        processed_texts = []
        for t in texts:
            clean_t = t.strip() if t else ""
            processed_texts.append(clean_t if clean_t else "빈 텍스트")
        
        return self.embedding_fn(processed_texts)
    
    def compute_similarity(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        use_cached_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """코사인 유사도 행렬 계산
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            use_cached_embeddings: 캐시된 임베딩 사용 여부
            
        Returns:
            유사도 행렬 (num_pages x num_segments)
        """
        # 텍스트 준비
        page_texts = []
        for page in pages:
            text = page.text.strip() if page.text else ""
            page_texts.append(text[:2000] if text else f"페이지 {page.page_num}")
        
        segment_texts = []
        for seg in segments:
            text = seg.text.strip() if seg.text else ""
            segment_texts.append(text[:2000] if text else "빈 자막")
        
        # 임베딩 생성 또는 캐시 사용
        if use_cached_embeddings:
            # 캐시된 임베딩 확인
            need_page_emb = any(p.embedding is None for p in pages)
            need_seg_emb = any(s.embedding is None for s in segments)
            
            if need_page_emb or need_seg_emb:
                all_texts = page_texts + segment_texts
                all_embeddings = self._get_embeddings(all_texts)
                
                # 임베딩 분리 및 캐싱
                for i, page in enumerate(pages):
                    page.embedding = all_embeddings[i]
                for j, seg in enumerate(segments):
                    seg.embedding = all_embeddings[len(pages) + j]
            
            page_embeddings = [p.embedding for p in pages]
            segment_embeddings = [s.embedding for s in segments]
        else:
            # 캐시 무시하고 새로 생성
            all_texts = page_texts + segment_texts
            all_embeddings = self._get_embeddings(all_texts)
            
            page_embeddings = all_embeddings[:len(pages)]
            segment_embeddings = all_embeddings[len(pages):]
        
        # numpy 배열로 변환
        page_vecs = np.array(page_embeddings, dtype=np.float32)
        seg_vecs = np.array(segment_embeddings, dtype=np.float32)
        
        # 디버깅을 위한 코드 추가
        # 유사도 행렬 계산
        cosine_matrix = SimilarityCalculator.compute_similarity_matrix(page_vecs, seg_vecs)

        # 디버깅 출력 (범위 확인)
        print(
            "COSINE RANGE:",
            float(cosine_matrix.min()),
            float(cosine_matrix.max()),
            float(cosine_matrix.mean()),
        )

        return cosine_matrix

    
    def compute_pairwise_similarity(
        self,
        text_a: str,
        text_b: str
    ) -> float:
        """두 텍스트 간의 코사인 유사도 계산
        
        Args:
            text_a: 텍스트 A
            text_b: 텍스트 B
            
        Returns:
            코사인 유사도 (-1 ~ 1)
        """
        embeddings = self._get_embeddings([text_a, text_b])
        vec_a = np.array(embeddings[0], dtype=np.float32)
        vec_b = np.array(embeddings[1], dtype=np.float32)
        
        return SimilarityCalculator.cosine_similarity(vec_a, vec_b)
    
    def get_similarity_distribution(
        self,
        similarity_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """유사도 분포 분석
        
        Args:
            similarity_matrix: 유사도 행렬
            
        Returns:
            분포 통계
        """
        flat = similarity_matrix.flatten()
        
        return {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "median": float(np.median(flat)),
            "percentiles": {
                "10": float(np.percentile(flat, 10)),
                "25": float(np.percentile(flat, 25)),
                "50": float(np.percentile(flat, 50)),
                "75": float(np.percentile(flat, 75)),
                "90": float(np.percentile(flat, 90)),
            }
        }
    
    def run_with_analysis(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.5,
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
        # 기본 실행
        result = self.run(
            pages, segments,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
        # 분포 분석 추가
        if result.similarity_matrix is not None:
            result.debug_info["similarity_distribution"] = self.get_similarity_distribution(
                result.similarity_matrix
            )
        
        # 페이지별 최고 유사도 정보
        if result.similarity_matrix is not None:
            page_best_matches = []
            for i in range(result.similarity_matrix.shape[0]):
                best_j = np.argmax(result.similarity_matrix[i])
                best_score = result.similarity_matrix[i, best_j]
                page_best_matches.append({
                    "page": i + 1,
                    "best_segment": int(best_j),
                    "best_score": float(best_score),
                    "best_time": segments[best_j].start if best_j < len(segments) else 0
                })
            result.debug_info["page_best_matches"] = page_best_matches
        
        return result


# 편의 함수
def cosine_similarity_sync(
    pages: List[PageData],
    segments: List[TranscriptSegment],
    embedding_fn: Callable,
    confidence_threshold: float = 0.5,
    **kwargs
) -> SyncResult:
    """코사인 유사도 동기화 실행
    
    Args:
        pages: 페이지 데이터 리스트
        segments: 자막 세그먼트 리스트
        embedding_fn: 임베딩 함수
        confidence_threshold: 신뢰도 임계값
        
    Returns:
        동기화 결과
    """
    algorithm = CosineSimilarityAlgorithm(embedding_fn=embedding_fn)
    return algorithm.run_with_analysis(
        pages, segments,
        confidence_threshold=confidence_threshold,
        **kwargs
    )
