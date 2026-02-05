"""
동기화 알고리즘 기본 클래스 및 공통 유틸리티

이 모듈은 모든 동기화 알고리즘이 상속받는 기본 클래스와
공통으로 사용하는 유틸리티 함수들을 제공합니다.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np


@dataclass
class PageData:
    """PDF 페이지 데이터"""
    page_num: int           # 페이지 번호 (1-based)
    text: str               # OCR 추출 텍스트
    title: str = ""         # 페이지 제목 (첫 번째 줄)
    keywords: set = field(default_factory=set)  # 추출된 키워드
    embedding: Optional[List[float]] = None     # 임베딩 벡터


@dataclass
class TranscriptSegment:
    """음성 자막 세그먼트"""
    start: float            # 시작 시간 (초)
    end: float              # 종료 시간 (초)
    text: str               # 자막 텍스트
    keywords: set = field(default_factory=set)  # 추출된 키워드
    embedding: Optional[List[float]] = None     # 임베딩 벡터


@dataclass
class SegmentGroup:
    """자막 그룹 (여러 세그먼트를 묶은 것)"""
    start: float
    end: float
    text: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    keywords: set = field(default_factory=set)
    embedding: Optional[List[float]] = None


@dataclass
class SyncAnchor:
    """동기화 앵커 포인트"""
    page: int               # 페이지 번호
    time: float             # 시간 (초)
    confidence: float = 0.0 # 신뢰도 점수
    method: str = ""        # 매칭 방법 (exact, cosine, hybrid, etc.)


@dataclass
class SyncResult:
    """동기화 결과"""
    anchors: List[SyncAnchor]
    similarity_matrix: Optional[np.ndarray] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)


class TextProcessor:
    """텍스트 전처리 유틸리티"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """텍스트 정규화
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정규화된 텍스트 (소문자, 공백 정리, 특수문자 제거)
        """
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^0-9a-z가-힣 ]+", "", text)
        return text
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 2) -> set:
        """키워드 추출
        
        Args:
            text: 원본 텍스트
            min_length: 최소 키워드 길이
            
        Returns:
            키워드 집합
        """
        normalized = TextProcessor.normalize(text)
        return set(w for w in normalized.split() if len(w) >= min_length)
    
    @staticmethod
    def extract_title(text: str, max_length: int = 50) -> str:
        """텍스트에서 제목 추출 (첫 번째 줄)
        
        Args:
            text: 원본 텍스트
            max_length: 최대 제목 길이
            
        Returns:
            추출된 제목
        """
        text = (text or "").strip()
        if not text:
            return ""
        
        # 줄바꿈으로 분리하여 첫 번째 줄 추출
        lines = text.split("\n")
        first_line = lines[0].strip() if lines else ""
        
        # 마침표로 분리하여 첫 문장 추출
        if not first_line:
            sentences = text.split(".")
            first_line = sentences[0].strip() if sentences else ""
        
        # 길이 제한
        if len(first_line) > max_length:
            first_line = first_line[:max_length] + "..."
            
        return first_line
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
        """긴 텍스트를 청크로 분할
        
        Args:
            text: 원본 텍스트
            max_chars: 청크당 최대 문자 수
            overlap: 청크 간 겹침 문자 수
            
        Returns:
            청크 리스트
        """
        text = (text or "").strip()
        if not text:
            return []
        
        chunks = []
        i = 0
        n = len(text)
        
        while i < n:
            end = min(n, i + max_chars)
            chunk = text[i:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            i = max(0, end - overlap)
            
        return chunks


class SimilarityCalculator:
    """유사도 계산 유틸리티"""
    
    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """코사인 유사도 계산
        
        Args:
            vec_a: 벡터 A
            vec_b: 벡터 B
            
        Returns:
            코사인 유사도 (-1 ~ 1)
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    
    @staticmethod
    def jaccard_similarity(set_a: set, set_b: set) -> float:
        """자카드 유사도 계산 (집합 기반)
        
        Args:
            set_a: 집합 A
            set_b: 집합 B
            
        Returns:
            자카드 유사도 (0 ~ 1)
        """
        if not set_a or not set_b:
            return 0.0
            
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def keyword_overlap_count(set_a: set, set_b: set) -> int:
        """키워드 겹침 개수
        
        Args:
            set_a: 키워드 집합 A
            set_b: 키워드 집합 B
            
        Returns:
            겹치는 키워드 개수
        """
        if not set_a or not set_b:
            return 0
        return len(set_a & set_b)
    
    @staticmethod
    def compute_similarity_matrix(
        page_embeddings: np.ndarray,
        segment_embeddings: np.ndarray
    ) -> np.ndarray:
        """유사도 행렬 계산 (벡터 기반)
        
        Args:
            page_embeddings: 페이지 임베딩 행렬 (num_pages x embedding_dim)
            segment_embeddings: 세그먼트 임베딩 행렬 (num_segments x embedding_dim)
            
        Returns:
            유사도 행렬 (num_pages x num_segments)
        """
        # 벡터 정규화
        page_norms = np.linalg.norm(page_embeddings, axis=1, keepdims=True)
        seg_norms = np.linalg.norm(segment_embeddings, axis=1, keepdims=True)
        
        page_normalized = page_embeddings / np.where(page_norms > 0, page_norms, 1)
        seg_normalized = segment_embeddings / np.where(seg_norms > 0, seg_norms, 1)
        
        # 행렬 곱으로 코사인 유사도 계산
        return np.dot(page_normalized, seg_normalized.T)


class BaseSyncAlgorithm(ABC):
    """동기화 알고리즘 기본 클래스
    
    모든 동기화 알고리즘은 이 클래스를 상속받아 구현합니다.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.text_processor = TextProcessor()
        self.similarity_calculator = SimilarityCalculator()
    
    @abstractmethod
    def compute_similarity(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        **kwargs
    ) -> np.ndarray:
        """유사도 행렬 계산
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            **kwargs: 추가 파라미터
            
        Returns:
            유사도 행렬 (num_pages x num_segments)
        """
        pass
    
    def find_optimal_path(
        self,
        similarity_matrix: np.ndarray,
        enforce_monotonic: bool = True
    ) -> List[Tuple[int, int, float]]:
        """동적 프로그래밍으로 최적 매칭 경로 탐색
        
        Args:
            similarity_matrix: 유사도 행렬
            enforce_monotonic: 페이지 순서 강제 여부
            
        Returns:
            매칭 경로 리스트 [(page_idx, segment_idx, score), ...]
        """
        num_pages, num_segments = similarity_matrix.shape
        
        if not enforce_monotonic:
            # 단순히 각 페이지에서 최고 점수 세그먼트 선택
            path = []
            for i in range(num_pages):
                j = np.argmax(similarity_matrix[i])
                score = similarity_matrix[i, j]
                path.append((i, j, float(score)))
            return path
        
        # DP 테이블 초기화
        dp = np.full((num_pages + 1, num_segments + 1), -np.inf)
        dp[0, 0] = 0
        
        # 경로 역추적 정보
        parent = {}
        
        # DP 계산
        for i in range(1, num_pages + 1):
            for j in range(1, num_segments + 1):
                for prev_j in range(j):
                    score = dp[i-1, prev_j] + similarity_matrix[i-1, j-1]
                    if score > dp[i, j]:
                        dp[i, j] = score
                        parent[(i, j)] = (i-1, prev_j, j-1)
        
        # 최고 점수 찾기
        best_j = np.argmax(dp[num_pages, 1:]) + 1
        
        # 경로 역추적
        path = []
        i, j = num_pages, best_j
        while i > 0 and (i, j) in parent:
            prev_i, prev_j, matched_seg = parent[(i, j)]
            score = similarity_matrix[i-1, matched_seg]
            path.append((i - 1, matched_seg, float(score)))
            i, j = prev_i, prev_j
        
        path.reverse()
        return path
    
    def interpolate_anchors(
        self,
        reliable_anchors: List[SyncAnchor],
        num_pages: int,
        total_duration: float
    ) -> List[SyncAnchor]:
        """신뢰도 높은 앵커 사이를 선형 보간
        
        Args:
            reliable_anchors: 신뢰도 높은 앵커 리스트
            num_pages: 총 페이지 수
            total_duration: 총 오디오 길이 (초)
            
        Returns:
            보간된 전체 앵커 리스트
        """
        if not reliable_anchors:
            # 앵커가 없으면 균등 분할
            return [
                SyncAnchor(
                    page=i + 1,
                    time=(i / max(1, num_pages)) * total_duration,
                    confidence=0.0,
                    method="uniform"
                )
                for i in range(num_pages)
            ]
        
        # 시작점과 끝점 보장
        anchors = list(reliable_anchors)
        if anchors[0].page != 1:
            anchors.insert(0, SyncAnchor(page=1, time=0.0, confidence=0.5, method="boundary"))
        if anchors[-1].page < num_pages:
            anchors.append(SyncAnchor(page=num_pages, time=total_duration, confidence=0.5, method="boundary"))
        
        # 보간
        result = []
        for k in range(len(anchors) - 1):
            curr = anchors[k]
            next_anchor = anchors[k + 1]
            
            result.append(curr)
            
            # 중간 페이지 보간
            page_gap = next_anchor.page - curr.page
            if page_gap > 1:
                time_gap = next_anchor.time - curr.time
                time_per_page = time_gap / page_gap
                
                for step in range(1, page_gap):
                    interp_page = curr.page + step
                    interp_time = curr.time + (time_per_page * step)
                    result.append(SyncAnchor(
                        page=interp_page,
                        time=interp_time,
                        confidence=0.3,
                        method="interpolation"
                    ))
        
        # 마지막 앵커 추가
        if anchors:
            result.append(anchors[-1])
        
        # 중복 제거 및 정렬
        seen_pages = set()
        unique_result = []
        for anchor in sorted(result, key=lambda x: x.page):
            if anchor.page not in seen_pages:
                seen_pages.add(anchor.page)
                unique_result.append(anchor)
        
        return unique_result
    
    def run(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> SyncResult:
        """동기화 실행
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            confidence_threshold: 신뢰도 임계값
            **kwargs: 추가 파라미터
            
        Returns:
            동기화 결과
        """
        # 1. 유사도 계산
        similarity_matrix = self.compute_similarity(pages, segments, **kwargs)
        
        # 2. 최적 경로 탐색
        path = self.find_optimal_path(similarity_matrix)
        
        # 3. 신뢰도 높은 앵커 선별
        reliable_anchors = []
        for page_idx, seg_idx, score in path:
            if score >= confidence_threshold:
                reliable_anchors.append(SyncAnchor(
                    page=page_idx + 1,
                    time=segments[seg_idx].start,
                    confidence=score,
                    method=self.name
                ))
        
        # 4. 보간
        total_duration = segments[-1].end if segments else 0
        final_anchors = self.interpolate_anchors(
            reliable_anchors, len(pages), total_duration
        )
        
        return SyncResult(
            anchors=final_anchors,
            similarity_matrix=similarity_matrix,
            debug_info={
                "algorithm": self.name,
                "num_pages": len(pages),
                "num_segments": len(segments),
                "matched_path": [(p+1, s, sc) for p, s, sc in path],
                "reliable_count": len(reliable_anchors),
                "confidence_threshold": confidence_threshold,
            }
        )


# 그룹화 유틸리티
class SegmentGrouper:
    """자막 세그먼트 그룹화 유틸리티"""
    
    @staticmethod
    def group_by_duration(
        segments: List[TranscriptSegment],
        duration: float = 10.0
    ) -> List[SegmentGroup]:
        """시간 단위로 그룹화
        
        Args:
            segments: 자막 세그먼트 리스트
            duration: 그룹 길이 (초)
            
        Returns:
            그룹 리스트
        """
        if not segments:
            return []
        
        groups = []
        current_group = SegmentGroup(start=0, end=0, text="", segments=[])
        
        for seg in segments:
            if not seg.text.strip():
                continue
            
            if not current_group.segments:
                current_group.start = seg.start
            
            current_group.segments.append(seg)
            current_group.end = seg.end
            
            if seg.end - current_group.start >= duration:
                current_group.text = " ".join(s.text for s in current_group.segments)
                current_group.keywords = TextProcessor.extract_keywords(current_group.text)
                groups.append(current_group)
                current_group = SegmentGroup(start=0, end=0, text="", segments=[])
        
        # 마지막 그룹
        if current_group.segments:
            current_group.text = " ".join(s.text for s in current_group.segments)
            current_group.keywords = TextProcessor.extract_keywords(current_group.text)
            groups.append(current_group)
        
        return groups
    
    @staticmethod
    def group_by_count(
        segments: List[TranscriptSegment],
        count: int = 5
    ) -> List[SegmentGroup]:
        """개수 단위로 그룹화
        
        Args:
            segments: 자막 세그먼트 리스트
            count: 그룹당 세그먼트 개수
            
        Returns:
            그룹 리스트
        """
        if not segments:
            return []
        
        groups = []
        
        for i in range(0, len(segments), count):
            batch = segments[i:i + count]
            batch = [s for s in batch if s.text.strip()]
            
            if not batch:
                continue
            
            group = SegmentGroup(
                start=batch[0].start,
                end=batch[-1].end,
                text=" ".join(s.text for s in batch),
                segments=batch
            )
            group.keywords = TextProcessor.extract_keywords(group.text)
            groups.append(group)
        
        return groups
    
    @staticmethod
    def no_grouping(segments: List[TranscriptSegment]) -> List[SegmentGroup]:
        """그룹화 없이 각 세그먼트를 그대로 사용
        
        Args:
            segments: 자막 세그먼트 리스트
            
        Returns:
            각 세그먼트가 하나의 그룹인 리스트
        """
        groups = []
        for seg in segments:
            if not seg.text.strip():
                continue
            group = SegmentGroup(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                segments=[seg]
            )
            group.keywords = TextProcessor.extract_keywords(group.text)
            groups.append(group)
        return groups
