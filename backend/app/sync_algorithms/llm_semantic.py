"""
알고리즘 6: LLM 의미 기반 매칭 (LLM Semantic Matching)

이 알고리즘은 LLM을 직접 사용하여 PDF 페이지와 자막 세그먼트 간의
의미적 연관성을 판단합니다.

장점:
- 가장 높은 정확도 가능
- 문맥 이해 가능
- 동의어/유의어 자동 처리

단점:
- API 비용 발생
- 속도가 느림
- Rate limiting 고려 필요
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import json
import time

from .base import (
    BaseSyncAlgorithm,
    PageData,
    TranscriptSegment,
    SyncResult,
    SyncAnchor,
    TextProcessor,
)


class LLMSemanticAlgorithm(BaseSyncAlgorithm):
    """LLM 의미 기반 매칭 알고리즘"""
    
    def __init__(self, llm_fn: Callable = None):
        """
        Args:
            llm_fn: LLM 호출 함수 (prompt -> response)
        """
        super().__init__(
            name="llm_semantic",
            description="LLM 의미 분석 기반 동기화"
        )
        self.llm_fn = llm_fn
        self._cache = {}  # 캐시
    
    def set_llm_function(self, llm_fn: Callable):
        """LLM 함수 설정"""
        self.llm_fn = llm_fn
    
    def _call_llm(self, prompt: str) -> str:
        """LLM 호출 (캐싱 지원)"""
        if not self.llm_fn:
            raise ValueError("LLM 함수가 설정되지 않았습니다. set_llm_function()을 먼저 호출하세요.")
        
        # 캐시 확인
        cache_key = hash(prompt)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # LLM 호출
        result = self.llm_fn(prompt)
        self._cache[cache_key] = result
        
        return result
    
    def judge_page_segment_match(
        self,
        page_text: str,
        segment_text: str,
        page_num: int,
    ) -> Dict[str, Any]:
        """LLM으로 페이지-세그먼트 매칭 판단
        
        Args:
            page_text: 페이지 텍스트 (처음 500자)
            segment_text: 세그먼트 텍스트
            page_num: 페이지 번호
            
        Returns:
            {"score": 0.0~1.0, "reason": "...", "confidence": 0.0~1.0}
        """
        prompt = f"""다음 PDF 슬라이드 내용과 강의 음성 자막이 얼마나 관련있는지 판단해주세요.

[슬라이드 {page_num} 내용]
{page_text[:500]}

[음성 자막]
{segment_text}

다음 JSON 형식으로만 응답하세요:
{{"score": 0.0에서 1.0 사이의 관련성 점수, "reason": "판단 이유 (한 문장)", "confidence": 0.0에서 1.0 사이의 확신도}}

점수 기준:
- 0.9~1.0: 이 자막이 이 슬라이드를 설명하고 있음이 확실함
- 0.7~0.9: 높은 관련성 (같은 주제, 유사한 키워드)
- 0.4~0.7: 중간 관련성 (부분적으로 관련)
- 0.1~0.4: 낮은 관련성
- 0.0~0.1: 관련 없음

JSON만 출력하세요:"""

        try:
            response = self._call_llm(prompt)
            
            # JSON 파싱
            # 응답에서 JSON 부분만 추출
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            result = json.loads(response.strip())
            return {
                "score": float(result.get("score", 0)),
                "reason": result.get("reason", ""),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except Exception as e:
            return {"score": 0.0, "reason": f"Error: {e}", "confidence": 0.0}
    
    def compute_similarity(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        sample_rate: float = 0.3,
        **kwargs
    ) -> np.ndarray:
        """LLM 기반 유사도 행렬 계산
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            sample_rate: 샘플링 비율 (비용 절감용)
            
        Returns:
            유사도 행렬 (num_pages x num_segments)
        """
        num_pages = len(pages)
        num_segments = len(segments)
        
        similarity_matrix = np.zeros((num_pages, num_segments))
        
        # 전체 비교는 비용이 많이 들므로 핵심 조합만 비교
        # 각 페이지에 대해 일부 세그먼트만 샘플링
        sample_count = max(1, int(num_segments * sample_rate))
        
        for i, page in enumerate(pages):
            # 균등 샘플링
            sample_indices = np.linspace(0, num_segments - 1, sample_count, dtype=int)
            
            for j in sample_indices:
                segment = segments[j]
                
                result = self.judge_page_segment_match(
                    page.text,
                    segment.text,
                    page.page_num,
                )
                
                similarity_matrix[i, j] = result["score"]
                
                # Rate limiting
                time.sleep(0.1)
        
        # 샘플링 안된 부분은 보간
        for i in range(num_pages):
            for j in range(num_segments):
                if similarity_matrix[i, j] == 0:
                    # 주변 값으로 보간
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < num_pages and 0 <= nj < num_segments:
                                if similarity_matrix[ni, nj] > 0:
                                    neighbors.append(similarity_matrix[ni, nj])
                    
                    if neighbors:
                        similarity_matrix[i, j] = np.mean(neighbors) * 0.8
        
        return similarity_matrix
    
    def find_best_matches_direct(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        window_size: int = 10,
    ) -> List[SyncAnchor]:
        """LLM으로 직접 최적 매칭 찾기 (더 효율적)
        
        페이지 순서대로 탐색하며 각 페이지의 시작 시점 찾기
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            window_size: 각 페이지당 탐색할 세그먼트 수
            
        Returns:
            앵커 리스트
        """
        anchors = []
        current_segment_idx = 0
        
        for page in pages:
            best_score = 0
            best_segment_idx = current_segment_idx
            best_confidence = 0
            
            # 현재 위치부터 window_size만큼 탐색
            search_end = min(current_segment_idx + window_size, len(segments))
            
            for j in range(current_segment_idx, search_end):
                segment = segments[j]
                
                result = self.judge_page_segment_match(
                    page.text,
                    segment.text,
                    page.page_num,
                )
                
                if result["score"] > best_score:
                    best_score = result["score"]
                    best_segment_idx = j
                    best_confidence = result["confidence"]
                
                # 높은 점수를 찾으면 조기 종료
                if best_score >= 0.85:
                    break
                
                time.sleep(0.05)
            
            # 앵커 추가
            if best_segment_idx < len(segments):
                anchors.append(SyncAnchor(
                    page=page.page_num,
                    time=segments[best_segment_idx].start,
                    confidence=best_score * best_confidence,
                    method="llm_semantic"
                ))
                
                # 다음 페이지는 현재 매칭 이후부터 탐색
                current_segment_idx = best_segment_idx + 1
        
        return anchors
    
    def run_with_analysis(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.4,
        use_direct_search: bool = True,
        **kwargs
    ) -> SyncResult:
        """분석 정보와 함께 동기화 실행
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            confidence_threshold: 신뢰도 임계값
            use_direct_search: 직접 탐색 방식 사용 (더 효율적)
            
        Returns:
            동기화 결과
        """
        if use_direct_search:
            # 직접 탐색 방식 (효율적)
            anchors = self.find_best_matches_direct(pages, segments)
            
            return SyncResult(
                anchors=anchors,
                similarity_matrix=None,
                debug_info={
                    "method": "direct_search",
                    "reliable_count": len([a for a in anchors if a.confidence >= confidence_threshold]),
                    "total_anchors": len(anchors),
                }
            )
        else:
            # 행렬 기반 방식 (정밀하지만 비용 높음)
            return self.run(
                pages, segments,
                confidence_threshold=confidence_threshold,
                **kwargs
            )


def llm_semantic_sync(
    pages: List[PageData],
    segments: List[TranscriptSegment],
    llm_fn: Callable,
    confidence_threshold: float = 0.4,
    **kwargs
) -> SyncResult:
    """LLM 의미 기반 동기화 실행
    
    Args:
        pages: 페이지 데이터 리스트
        segments: 자막 세그먼트 리스트
        llm_fn: LLM 호출 함수
        confidence_threshold: 신뢰도 임계값
        
    Returns:
        동기화 결과
    """
    algorithm = LLMSemanticAlgorithm(llm_fn=llm_fn)
    return algorithm.run_with_analysis(
        pages, segments,
        confidence_threshold=confidence_threshold,
        **kwargs
    )
