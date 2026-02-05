"""
실험용 동기화 API 엔드포인트

다양한 알고리즘을 실험하고 비교할 수 있는 엔드포인트를 제공합니다.

주요 기능:
- 개별 알고리즘 테스트
- 알고리즘 비교
- Ground Truth 관리
- 평가 지표 계산

main.py에 import하여 사용:
```python
from .sync_experiments import register_experiment_routes
register_experiment_routes(app)
```
"""

from fastapi import FastAPI, Depends, HTTPException, Body
from sqlmodel import Session, select
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# 동기화 알고리즘 모듈
from .sync_algorithms import (
    PageData,
    TranscriptSegment,
    SegmentGroup,
    SyncAnchor,
    SegmentGrouper,
    ExactMatchingAlgorithm,
    CosineSimilarityAlgorithm,
    HybridAlgorithm,
    StructuredPDFAlgorithm,
    SyncEvaluator,
    GroundTruth,
    GroundTruthManager,
    calculate_metrics,
    generate_evaluation_report,
    list_algorithms,
    get_algorithm,
)

# DB 모델 및 세션
from .db import get_session
from .models import TranscriptSegment as DBTranscriptSegment, PageAnchor

# 임베딩 함수 (GPU 서버)
from .embedding_runner import get_embeddings_from_gpu


def register_experiment_routes(app: FastAPI, base_dir: Path):
    """실험용 라우트 등록
    
    Args:
        app: FastAPI 앱 인스턴스
        base_dir: 데이터 저장 기본 디렉토리
    """
    
    # base_dir을 Path로 변환 (문자열로 들어올 경우 대비)
    base_dir = Path(base_dir)
    
    # Ground Truth 매니저 (앱 수준에서 유지)
    gt_manager = GroundTruthManager()
    
    def _lecture_data_dir(lecture_id: int) -> Path:
        """강의 데이터 디렉토리 경로"""
        d = (base_dir / "lectures" / str(lecture_id)).resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def _load_json(path: Path) -> Any:
        """JSON 파일 로드"""
        return json.loads(path.read_text(encoding="utf-8"))
    
    def _save_json(path: Path, obj: Any) -> None:
        """JSON 파일 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def _load_pages(lecture_id: int) -> List[PageData]:
        """pages.json에서 PageData 로드"""
        out_dir = _lecture_data_dir(lecture_id)
        pages_json_path = out_dir / "pages.json"
        
        if not pages_json_path.exists():
            raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")
        
        pages_obj = _load_json(pages_json_path)
        pages = []
        
        for p in pages_obj.get("pages", []):
            text = (p.get("text") or "").strip()
            if len(text) < 50 and p.get("words"):
                text = " ".join([w.get("t", "") for w in p.get("words", [])])
            
            page_data = PageData(
                page_num=int(p.get("page", 0)),
                text=text[:2000] if text else f"페이지 {p.get('page', 0)}",
                title=""
            )
            pages.append(page_data)
        
        return pages
    
    def _load_transcript_segments(
        lecture_id: int,
        session: Session,
        grouping: str = "none",
        group_duration: float = 10.0,
        group_count: int = 5
    ) -> List[TranscriptSegment]:
        """DB에서 자막 세그먼트 로드 및 그룹화
        
        Args:
            lecture_id: 강의 ID
            session: DB 세션
            grouping: 그룹화 방식 ("none", "duration", "count")
            group_duration: duration 그룹화 시 시간 (초)
            group_count: count 그룹화 시 개수
        """
        rows = session.exec(
            select(DBTranscriptSegment)
            .where(DBTranscriptSegment.lecture_id == lecture_id)
            .order_by(DBTranscriptSegment.start)
        ).all()
        
        if not rows:
            raise HTTPException(status_code=404, detail="자막이 없습니다. 먼저 /transcribe를 실행하세요.")
        
        # 기본 세그먼트 생성
        segments = [
            TranscriptSegment(
                start=float(r.start),
                end=float(r.end),
                text=str(r.text or "")
            )
            for r in rows
        ]
        
        # 그룹화
        if grouping == "duration":
            groups = SegmentGrouper.group_by_duration(segments, duration=group_duration)
            return [
                TranscriptSegment(
                    start=g.start,
                    end=g.end,
                    text=g.text,
                    keywords=g.keywords
                )
                for g in groups
            ]
        elif grouping == "count":
            groups = SegmentGrouper.group_by_count(segments, count=group_count)
            return [
                TranscriptSegment(
                    start=g.start,
                    end=g.end,
                    text=g.text,
                    keywords=g.keywords
                )
                for g in groups
            ]
        else:
            return segments
    
    # =========================================================================
    # 엔드포인트 정의
    # =========================================================================
    
    @app.get("/experiments/algorithms")
    def list_available_algorithms():
        """사용 가능한 알고리즘 목록 조회"""
        return {
            "algorithms": list_algorithms(),
            "descriptions": {
                "exact_matching": "키워드 정확 매칭 - 공통 키워드 기반",
                "cosine_similarity": "코사인 유사도 - 임베딩 벡터 기반",
                "hybrid": "하이브리드 - 키워드 + 코사인 결합",
                "llm_transcription": "LLM 전사 - GPT-4로 PDF 전사 후 비교",
                "structured_pdf": "구조화 PDF - 제목 기반 유사도",
            }
        }
    
    @app.post("/experiments/lectures/{lecture_id}/sync/{algorithm}")
    def run_single_algorithm(
        lecture_id: int,
        algorithm: str,
        grouping: str = "none",
        group_duration: float = 10.0,
        group_count: int = 5,
        confidence_threshold: float = 0.4,
        session: Session = Depends(get_session)
    ):
        """단일 알고리즘으로 동기화 실행
        
        Args:
            lecture_id: 강의 ID
            algorithm: 알고리즘 이름
            grouping: 그룹화 방식 (none, duration, count)
            group_duration: duration 그룹화 시간 (초)
            group_count: count 그룹화 개수
            confidence_threshold: 신뢰도 임계값
        """
        # 데이터 로드
        pages = _load_pages(lecture_id)
        segments = _load_transcript_segments(
            lecture_id, session, grouping, group_duration, group_count
        )
        
        if not pages:
            raise HTTPException(status_code=400, detail="페이지 데이터가 없습니다.")
        if not segments:
            raise HTTPException(status_code=400, detail="자막 데이터가 없습니다.")
        
        # 알고리즘 선택 및 실행
        try:
            if algorithm == "exact_matching":
                algo = ExactMatchingAlgorithm()
                result = algo.run_with_analysis(
                    pages, segments,
                    confidence_threshold=confidence_threshold
                )
            
            elif algorithm == "cosine_similarity":
                algo = CosineSimilarityAlgorithm(embedding_fn=get_embeddings_from_gpu)
                result = algo.run_with_analysis(
                    pages, segments,
                    confidence_threshold=confidence_threshold
                )
            
            elif algorithm == "hybrid":
                algo = HybridAlgorithm(embedding_fn=get_embeddings_from_gpu)
                result = algo.run_with_analysis(
                    pages, segments,
                    confidence_threshold=confidence_threshold
                )
            
            elif algorithm == "structured_pdf":
                algo = StructuredPDFAlgorithm(embedding_fn=get_embeddings_from_gpu)
                result = algo.run_with_analysis(
                    pages, segments,
                    confidence_threshold=confidence_threshold
                )
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"지원하지 않는 알고리즘: {algorithm}. 사용 가능: {list_algorithms()}"
                )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"알고리즘 실행 실패: {str(e)}")
        
        # 결과 저장
        out_dir = _lecture_data_dir(lecture_id)
        result_file = out_dir / f"sync_result_{algorithm}.json"
        
        result_obj = {
            "lecture_id": lecture_id,
            "algorithm": algorithm,
            "params": {
                "grouping": grouping,
                "group_duration": group_duration,
                "group_count": group_count,
                "confidence_threshold": confidence_threshold,
            },
            "anchors": [
                {"page": a.page, "time": a.time, "confidence": a.confidence, "method": a.method}
                for a in result.anchors
            ],
            "debug_info": result.debug_info,
            "similarity_matrix_shape": list(result.similarity_matrix.shape) if result.similarity_matrix is not None else None,
        }
        _save_json(result_file, result_obj)
        
        # 유사도 행렬도 별도 저장
        if result.similarity_matrix is not None:
            matrix_file = out_dir / f"similarity_matrix_{algorithm}.json"
            _save_json(matrix_file, {
                "algorithm": algorithm,
                "matrix": result.similarity_matrix.tolist(),
                "num_pages": len(pages),
                "num_segments": len(segments),
            })
        
        return {
            "ok": True,
            "algorithm": algorithm,
            "lecture_id": lecture_id,
            "anchors_count": len(result.anchors),
            "anchors": result_obj["anchors"],
            "debug_info": result.debug_info,
        }
    
    @app.post("/experiments/lectures/{lecture_id}/compare")
    def compare_algorithms(
        lecture_id: int,
        algorithms: List[str] = Body(default=["exact_matching", "cosine_similarity", "hybrid"]),
        grouping: str = Body(default="none"),
        group_duration: float = Body(default=10.0),
        confidence_threshold: float = Body(default=0.4),
        session: Session = Depends(get_session)
    ):
        """여러 알고리즘 비교
        
        Args:
            lecture_id: 강의 ID
            algorithms: 비교할 알고리즘 목록
            grouping: 그룹화 방식
            group_duration: 그룹화 시간
            confidence_threshold: 신뢰도 임계값
        """
        # 데이터 로드
        pages = _load_pages(lecture_id)
        segments = _load_transcript_segments(
            lecture_id, session, grouping, group_duration, 5
        )
        
        results = {}
        errors = {}
        
        for algo_name in algorithms:
            try:
                if algo_name == "exact_matching":
                    algo = ExactMatchingAlgorithm()
                elif algo_name == "cosine_similarity":
                    algo = CosineSimilarityAlgorithm(embedding_fn=get_embeddings_from_gpu)
                elif algo_name == "hybrid":
                    algo = HybridAlgorithm(embedding_fn=get_embeddings_from_gpu)
                elif algo_name == "structured_pdf":
                    algo = StructuredPDFAlgorithm(embedding_fn=get_embeddings_from_gpu)
                else:
                    errors[algo_name] = f"지원하지 않는 알고리즘"
                    continue
                
                result = algo.run_with_analysis(
                    pages, segments,
                    confidence_threshold=confidence_threshold
                )
                results[algo_name] = result
                
            except Exception as e:
                errors[algo_name] = str(e)
        
        # Ground Truth 확인
        gt = gt_manager.get_ground_truth(lecture_id)
        
        comparison = {}
        if gt:
            evaluator = SyncEvaluator(tolerance=5.0)
            comparison = evaluator.compare_algorithms(results, gt)
        
        # 결과 정리
        algorithm_results = {}
        for algo_name, result in results.items():
            algorithm_results[algo_name] = {
                "anchors_count": len(result.anchors),
                "anchors": [
                    {"page": a.page, "time": a.time, "confidence": a.confidence}
                    for a in result.anchors
                ],
                "debug_summary": {
                    "reliable_count": result.debug_info.get("reliable_count", 0),
                    "matched_pairs_count": len(result.debug_info.get("matched_path", [])),
                }
            }
        
        return {
            "ok": True,
            "lecture_id": lecture_id,
            "algorithms_tested": list(results.keys()),
            "errors": errors,
            "results": algorithm_results,
            "comparison": comparison if comparison else "Ground Truth가 없어 비교 불가",
            "params": {
                "grouping": grouping,
                "group_duration": group_duration,
                "confidence_threshold": confidence_threshold,
            }
        }
    
    @app.post("/experiments/lectures/{lecture_id}/ground_truth")
    def set_ground_truth(
        lecture_id: int,
        data: List[Dict[str, Any]] = Body(...)
    ):
        """Ground Truth 설정
        
        요청 본문 예시:
        [
            {"page": 1, "time": 0.0},
            {"page": 2, "time": 45.5},
            {"page": 3, "time": 120.0},
            ...
        ]
        """
        gt_manager.set_ground_truth_from_list(lecture_id, data)
        
        # 파일로도 저장
        out_dir = _lecture_data_dir(lecture_id)
        gt_file = out_dir / "ground_truth.json"
        _save_json(gt_file, data)
        
        return {
            "ok": True,
            "lecture_id": lecture_id,
            "ground_truth_count": len(data),
            "saved_to": str(gt_file)
        }
    
    @app.get("/experiments/lectures/{lecture_id}/ground_truth")
    def get_ground_truth(lecture_id: int):
        """Ground Truth 조회"""
        # 파일에서 로드 시도
        out_dir = _lecture_data_dir(lecture_id)
        gt_file = out_dir / "ground_truth.json"
        
        if gt_file.exists():
            data = _load_json(gt_file)
            gt_manager.set_ground_truth_from_list(lecture_id, data)
        
        gt = gt_manager.get_ground_truth(lecture_id)
        
        return {
            "lecture_id": lecture_id,
            "ground_truth": [
                {"page": g.page, "time": g.time, "tolerance": g.tolerance}
                for g in gt
            ],
            "count": len(gt)
        }
    
    @app.post("/experiments/lectures/{lecture_id}/evaluate/{algorithm}")
    def evaluate_algorithm(
        lecture_id: int,
        algorithm: str,
        tolerance: float = 5.0,
        session: Session = Depends(get_session)
    ):
        """알고리즘 평가 (Ground Truth 필요)
        
        Args:
            lecture_id: 강의 ID
            algorithm: 평가할 알고리즘
            tolerance: 허용 오차 (초)
        """
        # Ground Truth 로드
        out_dir = _lecture_data_dir(lecture_id)
        gt_file = out_dir / "ground_truth.json"
        
        if not gt_file.exists():
            raise HTTPException(
                status_code=400,
                detail="Ground Truth가 없습니다. 먼저 /ground_truth 엔드포인트로 설정하세요."
            )
        
        gt_data = _load_json(gt_file)
        gt_manager.set_ground_truth_from_list(lecture_id, gt_data)
        gt = gt_manager.get_ground_truth(lecture_id)
        
        # 알고리즘 결과 로드 또는 실행
        result_file = out_dir / f"sync_result_{algorithm}.json"
        
        if not result_file.exists():
            raise HTTPException(
                status_code=400,
                detail=f"{algorithm} 결과가 없습니다. 먼저 /sync/{algorithm} 엔드포인트를 실행하세요."
            )
        
        result_obj = _load_json(result_file)
        anchors = [
            SyncAnchor(
                page=a["page"],
                time=a["time"],
                confidence=a.get("confidence", 0.5),
                method=a.get("method", algorithm)
            )
            for a in result_obj["anchors"]
        ]
        
        # 평가
        evaluator = SyncEvaluator(tolerance=tolerance)
        eval_result = evaluator.evaluate(anchors, gt, compute_curves=True)
        
        return {
            "ok": True,
            "lecture_id": lecture_id,
            "algorithm": algorithm,
            "tolerance": tolerance,
            "metrics": {
                "mae": eval_result.mae,
                "rmse": eval_result.rmse,
                "f1_score": eval_result.f1_score,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "accuracy": eval_result.accuracy,
            },
            "confusion_matrix": eval_result.confusion_matrix,
            "matched_count": eval_result.matched_count,
            "total_pages": eval_result.total_pages,
            "page_errors": eval_result.page_errors,
            "roc_auc": eval_result.roc_data.get("auc", 0) if eval_result.roc_data else 0,
        }
    
    @app.get("/experiments/lectures/{lecture_id}/grouping_test")
    def test_grouping_strategies(
        lecture_id: int,
        session: Session = Depends(get_session)
    ):
        """그룹화 전략별 세그먼트 수 비교
        
        그룹화 방식에 따른 세그먼트 개수 변화를 확인합니다.
        """
        base_segments = _load_transcript_segments(lecture_id, session, "none", 10, 5)
        
        strategies = {
            "none": len(base_segments),
            "duration_5s": len(_load_transcript_segments(lecture_id, session, "duration", 5.0, 5)),
            "duration_10s": len(_load_transcript_segments(lecture_id, session, "duration", 10.0, 5)),
            "duration_15s": len(_load_transcript_segments(lecture_id, session, "duration", 15.0, 5)),
            "duration_30s": len(_load_transcript_segments(lecture_id, session, "duration", 30.0, 5)),
            "count_3": len(_load_transcript_segments(lecture_id, session, "count", 10, 3)),
            "count_5": len(_load_transcript_segments(lecture_id, session, "count", 10, 5)),
            "count_10": len(_load_transcript_segments(lecture_id, session, "count", 10, 10)),
        }
        
        return {
            "lecture_id": lecture_id,
            "original_segment_count": len(base_segments),
            "strategies": strategies,
            "recommendation": "세그먼트 수가 페이지 수와 비슷하거나 2~3배 정도가 적당합니다."
        }
    
    @app.post("/experiments/lectures/{lecture_id}/apply_sync")
    def apply_sync_result(
        lecture_id: int,
        algorithm: str = Body(..., embed=True),
        session: Session = Depends(get_session)
    ):
        """알고리즘 결과를 실제 PageAnchor에 적용
        
        실험 결과 중 가장 좋은 알고리즘을 선택하여 실제 앵커로 저장합니다.
        """
        out_dir = _lecture_data_dir(lecture_id)
        result_file = out_dir / f"sync_result_{algorithm}.json"
        
        if not result_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"{algorithm} 결과가 없습니다. 먼저 /sync/{algorithm}을 실행하세요."
            )
        
        result_obj = _load_json(result_file)
        
        # 기존 앵커 삭제
        old_anchors = session.exec(
            select(PageAnchor).where(PageAnchor.lecture_id == lecture_id)
        ).all()
        for anchor in old_anchors:
            session.delete(anchor)
        
        # 새 앵커 추가
        for a in result_obj["anchors"]:
            anchor = PageAnchor(
                lecture_id=lecture_id,
                page=a["page"],
                time=float(a["time"])
            )
            session.add(anchor)
        
        session.commit()
        
        return {
            "ok": True,
            "lecture_id": lecture_id,
            "algorithm": algorithm,
            "applied_anchors_count": len(result_obj["anchors"]),
            "message": f"{algorithm} 결과가 PageAnchor에 적용되었습니다."
        }
    
    return gt_manager  # 필요시 외부에서 접근 가능
