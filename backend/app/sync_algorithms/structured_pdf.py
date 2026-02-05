"""
알고리즘 5: 구조화된 PDF 유사도 (Structured PDF Similarity)

이 알고리즘은 PDF 페이지의 구조적 정보(제목, 핵심 키워드)를
활용하여 음성 자막과 비교합니다.

주요 기법:
- 페이지 제목 추출 (첫 번째 줄 기반)
- LLM을 활용한 주제 적합도 판단
- 구조적 계층 정보 활용

장점:
- 슬라이드 구조를 활용한 정확한 매칭
- 제목/핵심 내용 기반으로 노이즈 감소
- 주제 전환 지점을 잘 포착

단점:
- 제목이 없거나 명확하지 않은 슬라이드에 취약
- LLM 사용 시 비용 발생
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import re

from .base import (
    BaseSyncAlgorithm,
    PageData,
    TranscriptSegment,
    TextProcessor,
    SimilarityCalculator,
    SyncResult,
    SyncAnchor,
)
from .cosine_similarity import CosineSimilarityAlgorithm


class StructuredPDFAlgorithm(BaseSyncAlgorithm):
    """구조화된 PDF 기반 동기화 알고리즘"""
    
    def __init__(
        self,
        embedding_fn: Optional[Callable] = None,
        llm_fn: Optional[Callable] = None,
        title_weight: float = 0.6,
        content_weight: float = 0.4
    ):
        """
        Args:
            embedding_fn: 텍스트를 임베딩으로 변환하는 함수
            llm_fn: 주제 적합도 판단용 LLM 함수 (선택)
            title_weight: 제목 유사도 가중치
            content_weight: 본문 유사도 가중치
        """
        super().__init__(
            name="structured_pdf",
            description="구조화된 PDF 기반 동기화"
        )
        self.cosine_algo = CosineSimilarityAlgorithm(embedding_fn=embedding_fn)
        self.llm_fn = llm_fn
        self.title_weight = title_weight
        self.content_weight = content_weight
    
    def set_embedding_function(self, embedding_fn: Callable):
        """임베딩 함수 설정"""
        self.cosine_algo.set_embedding_function(embedding_fn)
    
    def set_llm_function(self, llm_fn: Callable):
        """LLM 함수 설정"""
        self.llm_fn = llm_fn
    
    def extract_page_title(self, text: str) -> str:
        """페이지에서 제목 추출
        
        Args:
            text: 페이지 전체 텍스트
            
        Returns:
            추출된 제목
        """
        text = (text or "").strip()
        if not text:
            return ""
        
        # 줄바꿈으로 분리
        lines = text.split("\n")
        
        # 첫 번째 비어있지 않은 줄 찾기
        for line in lines:
            line = line.strip()
            if len(line) >= 3:  # 최소 3글자
                # 불필요한 문자 제거
                title = re.sub(r'^[\s\-\*\#\•]+', '', line)
                title = title.strip()
                if len(title) >= 3:
                    return title[:100]  # 최대 100자
        
        return ""
    
    def extract_structured_info(self, page: PageData) -> Dict[str, Any]:
        """페이지에서 구조적 정보 추출
        
        Args:
            page: 페이지 데이터
            
        Returns:
            구조적 정보 딕셔너리
        """
        text = page.text or ""
        
        # 제목 추출
        title = self.extract_page_title(text)
        if not title:
            title = page.title or f"페이지 {page.page_num}"
        
        # 본문 (제목 제외)
        content = text.replace(title, "", 1).strip() if title else text
        
        # 키워드 추출
        title_keywords = TextProcessor.extract_keywords(title, min_length=2)
        content_keywords = TextProcessor.extract_keywords(content, min_length=2)
        
        # 구분자 기반 항목 추출
        bullet_items = []
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r'^[\-\*\•\d+\.\)]\s*', line):
                item = re.sub(r'^[\-\*\•\d+\.\)]\s*', '', line).strip()
                if item:
                    bullet_items.append(item)
        
        return {
            "page_num": page.page_num,
            "title": title,
            "content": content,
            "title_keywords": title_keywords,
            "content_keywords": content_keywords,
            "all_keywords": title_keywords | content_keywords,
            "bullet_items": bullet_items,
            "text_length": len(text)
        }
    
    def compute_similarity(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        use_title_similarity: bool = True,
        use_content_similarity: bool = True,
        **kwargs
    ) -> np.ndarray:
        """구조화된 유사도 행렬 계산
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            use_title_similarity: 제목 유사도 사용 여부
            use_content_similarity: 본문 유사도 사용 여부
            
        Returns:
            유사도 행렬 (num_pages x num_segments)
        """
        num_pages = len(pages)
        num_segments = len(segments)
        
        # 구조적 정보 추출
        page_structures = [self.extract_structured_info(p) for p in pages]
        
        # 제목 텍스트 리스트
        titles = [ps["title"] for ps in page_structures]
        
        # 본문 텍스트 리스트
        contents = [ps["content"][:1500] for ps in page_structures]
        
        # 자막 텍스트 리스트
        seg_texts = [s.text for s in segments]
        
        # 임베딩 생성을 위한 텍스트 통합
        all_texts = titles + contents + seg_texts
        all_texts = [t if t.strip() else "빈 텍스트" for t in all_texts]
        
        # 임베딩 생성
        all_embeddings = self.cosine_algo._get_embeddings(all_texts)
        
        # 임베딩 분리
        title_embeddings = all_embeddings[:num_pages]
        content_embeddings = all_embeddings[num_pages:2*num_pages]
        segment_embeddings = all_embeddings[2*num_pages:]
        
        # 제목 유사도 행렬
        if use_title_similarity:
            title_vecs = np.array(title_embeddings, dtype=np.float32)
            seg_vecs = np.array(segment_embeddings, dtype=np.float32)
            title_matrix = SimilarityCalculator.compute_similarity_matrix(title_vecs, seg_vecs)
        else:
            title_matrix = np.zeros((num_pages, num_segments))
        
        # 본문 유사도 행렬
        if use_content_similarity:
            content_vecs = np.array(content_embeddings, dtype=np.float32)
            seg_vecs = np.array(segment_embeddings, dtype=np.float32)
            content_matrix = SimilarityCalculator.compute_similarity_matrix(content_vecs, seg_vecs)
        else:
            content_matrix = np.zeros((num_pages, num_segments))
        
        # 가중 결합
        combined_matrix = (
            self.title_weight * title_matrix +
            self.content_weight * content_matrix
        )
        
        return combined_matrix
    
    def compute_similarity_detailed(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        **kwargs
    ) -> Dict[str, Any]:
        """상세 유사도 정보 반환
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            
        Returns:
            상세 유사도 정보
        """
        num_pages = len(pages)
        num_segments = len(segments)
        
        # 구조적 정보 추출
        page_structures = [self.extract_structured_info(p) for p in pages]
        
        # 텍스트 준비
        titles = [ps["title"] for ps in page_structures]
        contents = [ps["content"][:1500] for ps in page_structures]
        seg_texts = [s.text for s in segments]
        
        all_texts = titles + contents + seg_texts
        all_texts = [t if t.strip() else "빈 텍스트" for t in all_texts]
        
        # 임베딩 생성
        all_embeddings = self.cosine_algo._get_embeddings(all_texts)
        
        title_embeddings = all_embeddings[:num_pages]
        content_embeddings = all_embeddings[num_pages:2*num_pages]
        segment_embeddings = all_embeddings[2*num_pages:]
        
        # 행렬 계산
        title_vecs = np.array(title_embeddings, dtype=np.float32)
        content_vecs = np.array(content_embeddings, dtype=np.float32)
        seg_vecs = np.array(segment_embeddings, dtype=np.float32)
        
        title_matrix = SimilarityCalculator.compute_similarity_matrix(title_vecs, seg_vecs)
        content_matrix = SimilarityCalculator.compute_similarity_matrix(content_vecs, seg_vecs)
        combined_matrix = self.title_weight * title_matrix + self.content_weight * content_matrix
        
        return {
            "title_matrix": title_matrix,
            "content_matrix": content_matrix,
            "combined_matrix": combined_matrix,
            "page_structures": page_structures,
            "weights": {
                "title": self.title_weight,
                "content": self.content_weight
            }
        }
    
    def judge_topic_relevance_with_llm(
        self,
        page_structure: Dict[str, Any],
        segment_text: str
    ) -> Tuple[float, str]:
        """LLM을 사용하여 주제 적합도 판단
        
        Args:
            page_structure: 페이지 구조 정보
            segment_text: 자막 텍스트
            
        Returns:
            (적합도 점수 0~1, 판단 이유)
        """
        if self.llm_fn is None:
            return 0.5, "LLM 함수가 설정되지 않음"
        
        prompt = f"""다음 슬라이드 내용과 강의 자막이 같은 주제를 다루고 있는지 판단해주세요.

슬라이드 제목: {page_structure['title']}
슬라이드 키워드: {', '.join(list(page_structure['all_keywords'])[:10])}
슬라이드 내용 미리보기: {page_structure['content'][:200]}

자막 내용: {segment_text[:300]}

다음 형식으로만 응답해주세요:
적합도: [0.0 ~ 1.0 사이의 숫자]
이유: [한 줄 설명]"""
        
        try:
            response = self.llm_fn(prompt)
            
            # 응답 파싱
            lines = response.strip().split("\n")
            score = 0.5
            reason = "파싱 실패"
            
            for line in lines:
                if "적합도:" in line:
                    try:
                        score_str = line.split(":")[-1].strip()
                        score = float(score_str)
                        score = max(0.0, min(1.0, score))
                    except:
                        pass
                elif "이유:" in line:
                    reason = line.split(":", 1)[-1].strip()
            
            return score, reason
        except Exception as e:
            return 0.5, f"LLM 오류: {str(e)}"
    
    def detect_topic_changes(
        self,
        pages: List[PageData]
    ) -> List[Dict[str, Any]]:
        """페이지 간 주제 변경 지점 감지
        
        Args:
            pages: 페이지 데이터 리스트
            
        Returns:
            주제 변경 정보 리스트
        """
        if len(pages) < 2:
            return []
        
        # 구조적 정보 추출
        structures = [self.extract_structured_info(p) for p in pages]
        
        # 제목 임베딩 생성
        titles = [s["title"] for s in structures]
        titles = [t if t.strip() else "빈 제목" for t in titles]
        
        title_embeddings = self.cosine_algo._get_embeddings(titles)
        title_vecs = np.array(title_embeddings, dtype=np.float32)
        
        # 연속 페이지 간 유사도 계산
        changes = []
        for i in range(len(pages) - 1):
            sim = SimilarityCalculator.cosine_similarity(
                title_vecs[i], title_vecs[i+1]
            )
            
            # 키워드 겹침
            kw_overlap = len(
                structures[i]["all_keywords"] & structures[i+1]["all_keywords"]
            )
            
            # 주제 변경 판단 (유사도가 낮고 키워드 겹침이 적으면)
            is_change = sim < 0.5 and kw_overlap < 3
            
            changes.append({
                "from_page": i + 1,
                "to_page": i + 2,
                "title_similarity": float(sim),
                "keyword_overlap": kw_overlap,
                "is_topic_change": is_change,
                "from_title": structures[i]["title"],
                "to_title": structures[i+1]["title"]
            })
        
        return changes
    
    def run_with_analysis(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.45,
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
        # 상세 유사도 계산
        detailed = self.compute_similarity_detailed(pages, segments, **kwargs)
        
        # 최적 경로 탐색
        path = self.find_optimal_path(detailed["combined_matrix"])
        
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
        
        # 주제 변경 감지
        topic_changes = self.detect_topic_changes(pages)
        
        # 결과 구성
        result = SyncResult(
            anchors=final_anchors,
            similarity_matrix=detailed["combined_matrix"],
            debug_info={
                "algorithm": self.name,
                "weights": detailed["weights"],
                "num_pages": len(pages),
                "num_segments": len(segments),
                "matched_path": [(p+1, s, sc) for p, s, sc in path],
                "reliable_count": len(reliable_anchors),
                "confidence_threshold": confidence_threshold,
                "page_titles": [ps["title"] for ps in detailed["page_structures"]],
                "topic_changes": topic_changes,
                "matrix_stats": {
                    "title": {
                        "min": float(detailed["title_matrix"].min()),
                        "max": float(detailed["title_matrix"].max()),
                        "mean": float(detailed["title_matrix"].mean())
                    },
                    "content": {
                        "min": float(detailed["content_matrix"].min()),
                        "max": float(detailed["content_matrix"].max()),
                        "mean": float(detailed["content_matrix"].mean())
                    }
                }
            }
        )
        
        return result


# 편의 함수
def structured_pdf_sync(
    pages: List[PageData],
    segments: List[TranscriptSegment],
    embedding_fn: Callable,
    title_weight: float = 0.6,
    content_weight: float = 0.4,
    confidence_threshold: float = 0.45,
    **kwargs
) -> SyncResult:
    """구조화된 PDF 기반 동기화 실행
    
    Args:
        pages: 페이지 데이터 리스트
        segments: 자막 세그먼트 리스트
        embedding_fn: 임베딩 함수
        title_weight: 제목 가중치
        content_weight: 본문 가중치
        confidence_threshold: 신뢰도 임계값
        
    Returns:
        동기화 결과
    """
    algorithm = StructuredPDFAlgorithm(
        embedding_fn=embedding_fn,
        title_weight=title_weight,
        content_weight=content_weight
    )
    return algorithm.run_with_analysis(
        pages, segments,
        confidence_threshold=confidence_threshold,
        **kwargs
    )
