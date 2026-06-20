"""
알고리즘 1: 키워드 정확 매칭 (Exact Keyword Matching)

이 알고리즘은 PDF 페이지와 자막 세그먼트 간의 공통 키워드 개수를
기반으로 유사도를 계산합니다.

장점:
- 빠른 계산 속도
- 명확한 해석 가능성
- 전문 용어가 많은 강의에 효과적

단점:
- 동의어/유의어 처리 불가
- 문맥 이해 불가
- 텍스트가 짧으면 정확도 저하
"""

from typing import List, Dict, Any, Optional
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


class ExactMatchingAlgorithm(BaseSyncAlgorithm):
    """키워드 정확 매칭 알고리즘"""
    
    def __init__(self):
        super().__init__(
            name="exact_matching",
            description="키워드 정확 매칭 기반 동기화"
        )
    
    def compute_similarity(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        **kwargs
    ) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        num_pages = len(pages)
        num_segments = len(segments)
        
        # 전체 텍스트 수집
        page_texts = [p.text for p in pages]
        seg_texts = [s.text for s in segments]
        all_texts = page_texts + seg_texts
        
        # TF-IDF 벡터화 (페이지+세그먼트 합쳐서 학습)
        vectorizer = TfidfVectorizer(
            tokenizer=lambda t: list(TextProcessor.extract_keywords(t)),
            lowercase=False,
            min_df=1,
            sublinear_tf=True  # 빈도수 log 스케일링
        )
    
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        page_vecs = tfidf_matrix[:num_pages]
        seg_vecs = tfidf_matrix[num_pages:]
        
        # 코사인 유사도 계산
        similarity_matrix = (page_vecs @ seg_vecs.T).toarray()
        
        return similarity_matrix
    
    def get_matching_keywords(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
    ) -> Dict[str, Any]:
        """매칭된 키워드 상세 정보 반환 (디버깅용)
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            
        Returns:
            페이지별, 세그먼트별 키워드 매칭 정보
        """
        result = {
            "page_keywords": [],
            "segment_keywords": [],
            "matches": []
        }
        
        # 키워드 추출
        for i, page in enumerate(pages):
            kw = TextProcessor.extract_keywords(page.text)
            result["page_keywords"].append({
                "page": i + 1,
                "keywords": list(kw),
                "count": len(kw)
            })
        
        for j, seg in enumerate(segments):
            kw = TextProcessor.extract_keywords(seg.text)
            result["segment_keywords"].append({
                "segment": j,
                "start": seg.start,
                "end": seg.end,
                "keywords": list(kw),
                "count": len(kw)
            })
        
        # 매칭 정보
        for i, page in enumerate(pages):
            p_kw = TextProcessor.extract_keywords(page.text)
            for j, seg in enumerate(segments):
                s_kw = TextProcessor.extract_keywords(seg.text)
                common = p_kw & s_kw
                if common:
                    result["matches"].append({
                        "page": i + 1,
                        "segment": j,
                        "time": seg.start,
                        "common_keywords": list(common),
                        "count": len(common)
                    })
        
        return result
    
    def run_with_analysis(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.3,
        scoring_method: str = "overlap_ratio",
        **kwargs
    ) -> SyncResult:
        """분석 정보와 함께 동기화 실행
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            confidence_threshold: 신뢰도 임계값
            scoring_method: 점수 계산 방법
            
        Returns:
            동기화 결과 (분석 정보 포함)
        """
        # 기본 실행
        result = self.run(
            pages, segments,
            confidence_threshold=confidence_threshold,
            scoring_method=scoring_method,
            **kwargs
        )
        
        # 분석 정보 추가
        matching_info = self.get_matching_keywords(pages, segments)
        result.debug_info["keyword_analysis"] = matching_info
        
        # 통계 정보 추가
        total_page_keywords = sum(len(p.keywords or TextProcessor.extract_keywords(p.text)) for p in pages)
        total_segment_keywords = sum(len(s.keywords or TextProcessor.extract_keywords(s.text)) for s in segments)
        total_matches = sum(m["count"] for m in matching_info["matches"])
        
        result.debug_info["statistics"] = {
            "total_page_keywords": total_page_keywords,
            "total_segment_keywords": total_segment_keywords,
            "total_matches": total_matches,
            "avg_keywords_per_page": total_page_keywords / len(pages) if pages else 0,
            "avg_keywords_per_segment": total_segment_keywords / len(segments) if segments else 0,
        }
        
        return result


# 편의 함수
def exact_matching_sync(
    pages: List[PageData],
    segments: List[TranscriptSegment],
    scoring_method: str = "overlap_ratio",
    confidence_threshold: float = 0.3,
    **kwargs
) -> SyncResult:
    """키워드 정확 매칭 동기화 실행
    
    Args:
        pages: 페이지 데이터 리스트
        segments: 자막 세그먼트 리스트
        scoring_method: 점수 계산 방법
        confidence_threshold: 신뢰도 임계값
        
    Returns:
        동기화 결과
    """
    algorithm = ExactMatchingAlgorithm()
    return algorithm.run_with_analysis(
        pages, segments,
        confidence_threshold=confidence_threshold,
        scoring_method=scoring_method,
        **kwargs
    )
