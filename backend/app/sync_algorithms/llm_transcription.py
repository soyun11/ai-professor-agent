"""
알고리즘 4: LLM 전사 비교 (LLM Transcription Comparison)

이 알고리즘은 PDF 이미지를 LLM(GPT-4o-mini)으로 전사한 후,
음성 자막과 코사인 유사도로 비교합니다.

장점:
- OCR보다 정확한 텍스트 추출 가능
- 이미지 내 도표/그래프 설명 가능
- 문맥 이해 기반 전사

단점:
- API 비용 발생
- 처리 시간 증가
- 환각(hallucination) 가능성
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import base64
from pathlib import Path

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


class LLMTranscriptionAlgorithm(BaseSyncAlgorithm):
    """LLM 전사 기반 동기화 알고리즘"""
    
    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        embedding_fn: Optional[Callable] = None
    ):
        """
        Args:
            llm_fn: PDF 페이지 이미지를 텍스트로 전사하는 LLM 함수
                   signature: (image_base64: str, prompt: str) -> str
            embedding_fn: 텍스트를 임베딩으로 변환하는 함수
        """
        super().__init__(
            name="llm_transcription",
            description="LLM 전사 기반 동기화"
        )
        self.llm_fn = llm_fn
        self.cosine_algo = CosineSimilarityAlgorithm(embedding_fn=embedding_fn)
        self.transcription_cache: Dict[int, str] = {}  # 페이지 번호 -> 전사 텍스트
    
    def set_llm_function(self, llm_fn: Callable):
        """LLM 함수 설정"""
        self.llm_fn = llm_fn
    
    def set_embedding_function(self, embedding_fn: Callable):
        """임베딩 함수 설정"""
        self.cosine_algo.set_embedding_function(embedding_fn)
    
    def transcribe_page_image(
        self,
        image_path: str,
        page_num: int,
        use_cache: bool = True
    ) -> str:
        """PDF 페이지 이미지를 LLM으로 전사
        
        Args:
            image_path: 이미지 파일 경로
            page_num: 페이지 번호
            use_cache: 캐시 사용 여부
            
        Returns:
            전사된 텍스트
        """
        # 캐시 확인
        if use_cache and page_num in self.transcription_cache:
            return self.transcription_cache[page_num]
        
        if self.llm_fn is None:
            raise ValueError("LLM 함수가 설정되지 않았습니다. set_llm_function()을 호출하세요.")
        
        # 이미지 로드 및 base64 인코딩
        image_path = Path(image_path)
        if not image_path.exists():
            return ""
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # LLM 호출
        prompt = """이 강의 슬라이드 이미지의 내용을 텍스트로 전사해주세요.
        
요구사항:
1. 슬라이드에 있는 모든 텍스트를 추출해주세요.
2. 도표, 그래프, 다이어그램이 있다면 그 내용을 설명해주세요.
3. 수식이 있다면 텍스트로 표현해주세요.
4. 불필요한 설명 없이 내용만 간결하게 작성해주세요.

출력 형식:
- 제목이 있다면 첫 줄에 작성
- 내용은 자연스러운 문장으로 작성
- 핵심 키워드가 잘 드러나도록 작성"""
        
        try:
            transcribed_text = self.llm_fn(image_data, prompt)
            
            # 캐시 저장
            if use_cache:
                self.transcription_cache[page_num] = transcribed_text
            
            return transcribed_text
        except Exception as e:
            print(f"LLM 전사 실패 (페이지 {page_num}): {e}")
            return ""
    
    def transcribe_pages_batch(
        self,
        image_paths: List[str],
        use_cache: bool = True
    ) -> List[str]:
        """여러 페이지 일괄 전사
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            use_cache: 캐시 사용 여부
            
        Returns:
            전사된 텍스트 리스트
        """
        transcriptions = []
        for i, path in enumerate(image_paths):
            text = self.transcribe_page_image(path, i + 1, use_cache)
            transcriptions.append(text)
        return transcriptions
    
    def compute_similarity(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        use_llm_transcription: bool = True,
        image_paths: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """LLM 전사 기반 유사도 행렬 계산
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            use_llm_transcription: LLM 전사 사용 여부
            image_paths: 페이지 이미지 경로 리스트 (LLM 전사 시 필요)
            
        Returns:
            유사도 행렬 (num_pages x num_segments)
        """
        # LLM 전사 사용 시
        if use_llm_transcription and image_paths:
            transcriptions = self.transcribe_pages_batch(image_paths)
            
            # PageData 업데이트
            for i, page in enumerate(pages):
                if i < len(transcriptions) and transcriptions[i]:
                    page.text = transcriptions[i]
        
        # 코사인 유사도 계산
        return self.cosine_algo.compute_similarity(pages, segments, **kwargs)
    
    def compare_ocr_vs_llm(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        image_paths: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """OCR vs LLM 전사 비교 분석
        
        Args:
            pages: 페이지 데이터 리스트 (OCR 텍스트 포함)
            segments: 자막 세그먼트 리스트
            image_paths: 페이지 이미지 경로 리스트
            
        Returns:
            비교 분석 결과
        """
        # OCR 텍스트 백업
        ocr_texts = [p.text for p in pages]
        
        # OCR 기반 유사도
        ocr_matrix = self.cosine_algo.compute_similarity(pages, segments, **kwargs)
        
        # LLM 전사
        llm_transcriptions = self.transcribe_pages_batch(image_paths)
        
        # LLM 전사로 PageData 업데이트
        for i, page in enumerate(pages):
            if i < len(llm_transcriptions) and llm_transcriptions[i]:
                page.text = llm_transcriptions[i]
        
        # LLM 기반 유사도
        llm_matrix = self.cosine_algo.compute_similarity(
            pages, segments, use_cached_embeddings=False, **kwargs
        )
        
        # 원본 텍스트 복원
        for i, page in enumerate(pages):
            page.text = ocr_texts[i]
        
        # 비교 분석
        comparison = {
            "ocr_matrix_stats": {
                "min": float(ocr_matrix.min()),
                "max": float(ocr_matrix.max()),
                "mean": float(ocr_matrix.mean())
            },
            "llm_matrix_stats": {
                "min": float(llm_matrix.min()),
                "max": float(llm_matrix.max()),
                "mean": float(llm_matrix.mean())
            },
            "improvement": {
                "mean_diff": float(llm_matrix.mean() - ocr_matrix.mean()),
                "max_diff": float(llm_matrix.max() - ocr_matrix.max())
            },
            "page_comparison": []
        }
        
        # 페이지별 비교
        for i in range(len(pages)):
            ocr_best = np.argmax(ocr_matrix[i])
            llm_best = np.argmax(llm_matrix[i])
            
            comparison["page_comparison"].append({
                "page": i + 1,
                "ocr_text_length": len(ocr_texts[i]),
                "llm_text_length": len(llm_transcriptions[i]) if i < len(llm_transcriptions) else 0,
                "ocr_best_segment": int(ocr_best),
                "ocr_best_score": float(ocr_matrix[i, ocr_best]),
                "llm_best_segment": int(llm_best),
                "llm_best_score": float(llm_matrix[i, llm_best]),
                "same_match": ocr_best == llm_best
            })
        
        return {
            "ocr_matrix": ocr_matrix,
            "llm_matrix": llm_matrix,
            "comparison": comparison,
            "llm_transcriptions": llm_transcriptions
        }
    
    def run_with_analysis(
        self,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.5,
        image_paths: Optional[List[str]] = None,
        **kwargs
    ) -> SyncResult:
        """분석 정보와 함께 동기화 실행
        
        Args:
            pages: 페이지 데이터 리스트
            segments: 자막 세그먼트 리스트
            confidence_threshold: 신뢰도 임계값
            image_paths: 페이지 이미지 경로 리스트
            
        Returns:
            동기화 결과 (분석 정보 포함)
        """
        # LLM 전사 (이미지 경로가 제공된 경우)
        llm_used = False
        if image_paths:
            transcriptions = self.transcribe_pages_batch(image_paths)
            for i, page in enumerate(pages):
                if i < len(transcriptions) and transcriptions[i]:
                    page.text = transcriptions[i]
            llm_used = True
        
        # 기본 실행
        result = self.run(
            pages, segments,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
        # 추가 정보
        result.debug_info["llm_transcription_used"] = llm_used
        if llm_used:
            result.debug_info["transcription_lengths"] = [
                len(p.text) for p in pages
            ]
        
        return result


# 편의 함수
def llm_transcription_sync(
    pages: List[PageData],
    segments: List[TranscriptSegment],
    llm_fn: Callable,
    embedding_fn: Callable,
    image_paths: List[str],
    confidence_threshold: float = 0.5,
    **kwargs
) -> SyncResult:
    """LLM 전사 기반 동기화 실행
    
    Args:
        pages: 페이지 데이터 리스트
        segments: 자막 세그먼트 리스트
        llm_fn: LLM 전사 함수
        embedding_fn: 임베딩 함수
        image_paths: 페이지 이미지 경로 리스트
        confidence_threshold: 신뢰도 임계값
        
    Returns:
        동기화 결과
    """
    algorithm = LLMTranscriptionAlgorithm(
        llm_fn=llm_fn,
        embedding_fn=embedding_fn
    )
    return algorithm.run_with_analysis(
        pages, segments,
        confidence_threshold=confidence_threshold,
        image_paths=image_paths,
        **kwargs
    )
