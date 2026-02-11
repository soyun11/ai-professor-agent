"""
벤치마크 모듈

모든 동기화 알고리즘을 비교 평가하고 결과를 리포트로 생성합니다.

사용법:
    # 커맨드라인에서 실행
    python -m app.sync_algorithms.benchmark --lectures 1,2,3
    
    # 또는 코드에서
    from app.sync_algorithms.benchmark import run_full_benchmark
    results = run_full_benchmark(lecture_ids=[1, 2, 3])
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import PageData, TranscriptSegment, SegmentGrouper
from .evaluation import SyncEvaluator, GroundTruth, EvaluationResult
from . import ALGORITHMS, get_algorithm


class BenchmarkRunner:
    """벤치마크 실행기"""
    
    def __init__(
        self,
        base_dir: str,
        embedding_fn=None,
        llm_fn=None,
        tolerance: float = 10.0,
    ):
        """
        Args:
            base_dir: 데이터 저장 디렉토리 (예: ./data)
            embedding_fn: 임베딩 함수
            llm_fn: LLM 함수
            tolerance: 정답 판정 허용 오차 (초)
        """
        self.base_dir = Path(base_dir)
        self.embedding_fn = embedding_fn
        self.llm_fn = llm_fn
        self.tolerance = tolerance
        self.evaluator = SyncEvaluator(tolerance=tolerance)
        
        # 결과 저장 디렉토리
        self.results_dir = self.base_dir / "benchmark_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _lecture_data_dir(self, lecture_id: int) -> Path:
        """강의 원본 데이터 디렉토리 (ground_truth.json 위치)"""
        return self.base_dir / "lectures" / str(lecture_id)
    
    def _lecture_processed_dir(self, lecture_id: int) -> Path:
        """강의 처리 결과 디렉토리 (pages.json, transcript.json 위치)"""
        return self.base_dir.parent / "lectures" / str(lecture_id)
    
    def load_pages(self, lecture_id: int) -> List[PageData]:
        """pages.json에서 페이지 로드"""
        pages_path = self._lecture_processed_dir(lecture_id) / "pages.json"
        
        if not pages_path.exists():
            raise FileNotFoundError(f"pages.json not found: {pages_path}")
        
        with open(pages_path, "r", encoding="utf-8") as f:
            pages_obj = json.load(f)
        
        pages = []
        for p in pages_obj.get("pages", []):
            text = (p.get("text") or "").strip()
            if len(text) < 50 and p.get("words"):
                text = " ".join([w.get("t", "") for w in p.get("words", [])])
            
            pages.append(PageData(
                page_num=int(p.get("page", 0)),
                text=text[:2000] if text else f"페이지 {p.get('page', 0)}",
                title=""
            ))
        
        # 디버깅을 위한 추가
        pages.sort(key=lambda x: x.page_num) # pages.json 로드 후 무조건 정렬 추가
        print("[DEBUG] pages page_num head:", [p.page_num for p in pages[:10]])
        print("[DEBUG] pages page_num tail:", [p.page_num for p in pages[-10:]])

        return pages
    
    def load_segments(
        self,
        lecture_id: int,
        grouping: str = "none",
        group_duration: float = 10.0,
    ) -> List[TranscriptSegment]:
        """transcript.json에서 자막 세그먼트 로드"""
        transcript_path = self._lecture_processed_dir(lecture_id) / "transcript.json"
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"transcript.json not found: {transcript_path}")
        
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        
        # transcript.json 형식에 따라 파싱
        if isinstance(transcript_data, dict):
            raw_segments = transcript_data.get("segments", transcript_data.get("results", []))
        else:
            raw_segments = transcript_data
        
        if not raw_segments:
            raise ValueError(f"No segments found in transcript.json for lecture {lecture_id}")
        
        segments = [
            TranscriptSegment(
                start=float(s.get("start", 0)),
                end=float(s.get("end", s.get("start", 0) + 1)),
                text=str(s.get("text", ""))
            )
            for s in raw_segments
        ]
        
        # 그룹화
        if grouping == "duration":
            groups = SegmentGrouper.group_by_duration(segments, duration=group_duration)
            return [
                TranscriptSegment(start=g.start, end=g.end, text=g.text, keywords=g.keywords)
                for g in groups
            ]
        elif grouping == "count":
            groups = SegmentGrouper.group_by_count(segments, count=int(group_duration))
            return [
                TranscriptSegment(start=g.start, end=g.end, text=g.text, keywords=g.keywords)
                for g in groups
            ]
        
        return segments
    
    def load_ground_truth(self, lecture_id: int) -> List[GroundTruth]:
        """Ground Truth 로드 (skip 필드 처리)"""
        gt_path = self._lecture_data_dir(lecture_id) / "ground_truth.json"
        
        if not gt_path.exists():
            raise FileNotFoundError(f"ground_truth.json not found: {gt_path}")
        
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        
        ground_truths = []
        skipped_pages = []
        
        for d in gt_data:
            # skip 필드가 있으면 건너뛰기
            if d.get("skip", False):
                skipped_pages.append(d["page"])
                continue
            
            # time이 없거나 숫자가 아니면 건너뛰기
            time_val = d.get("time")
            if time_val is None or isinstance(time_val, str):
                skipped_pages.append(d["page"])
                continue
            
            try:
                time_float = float(time_val)
                ground_truths.append(
                    GroundTruth(
                        page=d["page"],
                        time=time_float,
                        tolerance=d.get("tolerance", self.tolerance)
                    )
                )
            except (ValueError, TypeError):
                skipped_pages.append(d["page"])
                continue
        
        if skipped_pages:
            print(f"    (Skipped pages: {skipped_pages})")
        
        return ground_truths
    
    def run_algorithm(
        self,
        algorithm_name: str,
        pages: List[PageData],
        segments: List[TranscriptSegment],
        confidence_threshold: float = 0.4,
    ) -> Dict[str, Any]:
        """단일 알고리즘 실행
        
        Returns:
            {"anchors": [...], "time_elapsed": float, "debug_info": {...}, "error": str or None}
        """
        start_time = time.time()
        
        try:
            algo = get_algorithm(
                algorithm_name,
                embedding_fn=self.embedding_fn,
                llm_fn=self.llm_fn,
            )
            
            result = algo.run(
                pages, segments,
                confidence_threshold=confidence_threshold,
            )
            
            elapsed = time.time() - start_time
            
            return {
                "anchors": result.anchors,
                "time_elapsed": elapsed,
                "debug_info": result.debug_info,
                "error": None,
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            import traceback
            return {
                "anchors": [],
                "time_elapsed": elapsed,
                "debug_info": {"error_traceback": traceback.format_exc()},
                "error": str(e),
            }
    
    def evaluate_result(
        self,
        anchors,
        ground_truth: List[GroundTruth],
        confidence_threshold: float = 0.4,
    ) -> Dict[str, Any]:
        """알고리즘 결과 평가 (4가지 지표)
        
        Returns:
            {
                "f1_score", "precision", "recall", "roc_auc",
                "confusion_matrix": {"TP", "FP", "FN"},
                "roc_data": {...}
            }
        """
        if not anchors:
            return {
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "roc_auc": 0.0,
                "confusion_matrix": {"TP": 0, "FP": 0, "FN": len(ground_truth)},
                "roc_data": None,
                "page_errors": [],
            }
        
        eval_result = self.evaluator.evaluate(
            anchors, 
            ground_truth, 
            confidence_threshold=confidence_threshold,
            compute_curves=True
        )
        
        return {
            "f1_score": eval_result.f1_score,
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "roc_auc": eval_result.roc_auc,
            "confusion_matrix": {
                "TP": eval_result.tp,
                "FP": eval_result.fp,
                "FN": eval_result.fn,
            },
            "roc_data": eval_result.roc_data,
            "page_errors": eval_result.page_errors,
        }
    
    def run_benchmark_single_lecture(
        self,
        lecture_id: int,
        algorithms: List[str],
        grouping: str = "none",
        group_duration: float = 30.0,
        confidence_threshold: float = 0.4,
    ) -> Dict[str, Any]:
        """단일 강의에 대해 벤치마크 실행
        
        Args:
            lecture_id: 강의 ID
            algorithms: 테스트할 알고리즘 목록
            grouping: 그룹화 방식 ("none", "duration", "count")
            group_duration: 그룹화 시간/개수
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            강의별 벤치마크 결과
        """
        print(f"\n{'='*60}")
        print(f"Benchmark: Lecture {lecture_id}")
        print(f"{'='*60}")
        
        # 데이터 로드
        try:
            pages = self.load_pages(lecture_id)
            segments = self.load_segments(lecture_id, grouping, group_duration)
            ground_truth = self.load_ground_truth(lecture_id)
        except Exception as e:
            print(f"  ❌ 데이터 로드 실패: {e}")
            return {
                "lecture_id": lecture_id,
                "error": str(e),
                "results": {},
            }
        
        print(f"  Pages: {len(pages)}, Segments: {len(segments)}, GT: {len(ground_truth)}")
        
        results = {}
        
        for algo_name in algorithms:
            print(f"\n  Running {algo_name}...", end=" ", flush=True)
            
            # 알고리즘 의존성 체크
            algo_info = ALGORITHMS.get(algo_name, {})
            
            if algo_info.get("requires_llm") and not self.llm_fn:
                print("SKIPPED (no LLM)")
                results[algo_name] = {"error": "LLM function not provided", "metrics": None}
                continue
            
            if algo_info.get("requires_embedding") and not self.embedding_fn:
                print("SKIPPED (no embedding)")
                results[algo_name] = {"error": "Embedding function not provided", "metrics": None}
                continue
            
            # 알고리즘 실행
            run_result = self.run_algorithm(
                algo_name, pages, segments, confidence_threshold
            )
            
            if run_result["error"]:
                print(f"ERROR: {run_result['error']}")
                results[algo_name] = {
                    "error": run_result["error"],
                    "metrics": None,
                    "time_elapsed": run_result["time_elapsed"],
                }
                continue
            
            # 평가
            metrics = self.evaluate_result(
                run_result["anchors"], 
                ground_truth,
                confidence_threshold
            )
            
            print(f"F1={metrics['f1_score']:.3f}, P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, AUC={metrics['roc_auc']:.3f} "
                  f"({run_result['time_elapsed']:.1f}s)")
            
            results[algo_name] = {
                "metrics": metrics,
                "time_elapsed": run_result["time_elapsed"],
                "anchors_count": len(run_result["anchors"]),
                "anchors": [
                    {"page": a.page, "time": a.time, "confidence": a.confidence}
                    for a in run_result["anchors"]
                ],
                "error": None,
            }
        
        return {
            "lecture_id": lecture_id,
            "num_pages": len(pages),
            "num_segments": len(segments),
            "num_ground_truth": len(ground_truth),
            "results": results,
            "error": None,
        }
    
    def run_full_benchmark(
        self,
        lecture_ids: List[int],
        algorithms: List[str] = None,
        grouping: str = "none",
        group_duration: float = 30.0,
        confidence_threshold: float = 0.4,
    ) -> Dict[str, Any]:
        """전체 벤치마크 실행
        
        Args:
            lecture_ids: 테스트할 강의 ID 목록
            algorithms: 테스트할 알고리즘 목록 (None이면 전체)
            grouping: 그룹화 방식
            group_duration: 그룹화 시간
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            전체 벤치마크 결과
        """
        if algorithms is None:
            algorithms = list(ALGORITHMS.keys())
        
        print("\n" + "="*60)
        print("FULL BENCHMARK")
        print("="*60)
        print(f"Lectures: {lecture_ids}")
        print(f"Algorithms: {algorithms}")
        print(f"Grouping: {grouping} ({group_duration}s)")
        
        all_results = []
        
        for lecture_id in lecture_ids:
            result = self.run_benchmark_single_lecture(
                lecture_id,
                algorithms=algorithms,
                grouping=grouping,
                group_duration=group_duration,
                confidence_threshold=confidence_threshold,
            )
            all_results.append(result)
        
        # 전체 집계
        summary = self._compute_summary(all_results, algorithms)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"benchmark_{timestamp}.json"
        
        final_result = {
            "timestamp": timestamp,
            "params": {
                "lecture_ids": lecture_ids,
                "algorithms": algorithms,
                "grouping": grouping,
                "group_duration": group_duration,
                "confidence_threshold": confidence_threshold,
                "tolerance": self.tolerance,
            },
            "lecture_results": all_results,
            "summary": summary,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        
        # 리포트 생성
        report_path = self.generate_report(final_result, timestamp)
        print(f"Report saved to: {report_path}")
        
        return final_result
    
    def _compute_summary(
        self,
        all_results: List[Dict],
        algorithms: List[str],
    ) -> Dict[str, Any]:
        """전체 결과 집계 (4가지 지표)"""
        
        # 알고리즘별 지표 수집
        per_algo = {algo: {
            "f1_scores": [], 
            "precisions": [], 
            "recalls": [], 
            "aucs": [], 
            "times": []
        } for algo in algorithms}
        
        for result in all_results:
            if result.get("error"):
                continue
            
            for algo_name, algo_result in result.get("results", {}).items():
                if algo_result.get("error") or not algo_result.get("metrics"):
                    continue
                
                m = algo_result["metrics"]
                per_algo[algo_name]["f1_scores"].append(m["f1_score"])
                per_algo[algo_name]["precisions"].append(m["precision"])
                per_algo[algo_name]["recalls"].append(m["recall"])
                per_algo[algo_name]["aucs"].append(m["roc_auc"])
                per_algo[algo_name]["times"].append(algo_result["time_elapsed"])
        
        # 평균 및 표준편차 계산
        evaluations = {}
        for algo_name, data in per_algo.items():
            if data["f1_scores"]:
                evaluations[algo_name] = {
                    "avg_f1": float(np.mean(data["f1_scores"])),
                    "std_f1": float(np.std(data["f1_scores"])),
                    "avg_precision": float(np.mean(data["precisions"])),
                    "std_precision": float(np.std(data["precisions"])),
                    "avg_recall": float(np.mean(data["recalls"])),
                    "std_recall": float(np.std(data["recalls"])),
                    "avg_roc_auc": float(np.mean(data["aucs"])),
                    "std_roc_auc": float(np.std(data["aucs"])),
                    "avg_time": float(np.mean(data["times"])),
                    "num_lectures": len(data["f1_scores"]),
                }
            else:
                evaluations[algo_name] = {
                    "error": "No valid results",
                    "num_lectures": 0,
                }
        
        # 랭킹 (유효한 결과만)
        valid_algos = [a for a in evaluations if "avg_f1" in evaluations[a]]
        
        rankings = {
            "by_f1": sorted(valid_algos, key=lambda a: evaluations[a]["avg_f1"], reverse=True),
            "by_auc": sorted(valid_algos, key=lambda a: evaluations[a]["avg_roc_auc"], reverse=True),
        }
        
        return {
            "evaluations": evaluations,
            "rankings": rankings,
            "best_algorithm": {
                "by_f1": rankings["by_f1"][0] if rankings["by_f1"] else None,
                "by_auc": rankings["by_auc"][0] if rankings["by_auc"] else None,
            }
        }
    
    def generate_report(self, results: Dict, timestamp: str) -> Path:
        """마크다운 리포트 생성"""
        report_path = self.results_dir / f"benchmark_report_{timestamp}.md"
        
        lines = [
            "# 동기화 알고리즘 벤치마크 리포트",
            "",
            f"**생성 시간:** {timestamp}",
            "",
            "## 실험 설정",
            "",
            f"- **테스트 강의:** {results['params']['lecture_ids']}",
            f"- **알고리즘:** {', '.join(results['params']['algorithms'])}",
            f"- **그룹화:** {results['params']['grouping']} ({results['params']['group_duration']}초)",
            f"- **신뢰도 임계값:** {results['params']['confidence_threshold']}",
            f"- **tolerance (정답 허용오차):** {results['params'].get('tolerance', 5.0)}초",
            "",
            "## 전체 결과 요약",
            "",
            "### 평균 성능 비교 (4가지 지표)",
            "",
            "| 알고리즘 | F1 | Precision | Recall | ROC-AUC | 실행시간 |",
            "|---------|----:|----------:|-------:|--------:|---------:|",
        ]
        
        evals = results.get("summary", {}).get("evaluations", {})
        
        for algo_name, data in evals.items():
            if "avg_f1" in data:
                lines.append(
                    f"| {algo_name} | "
                    f"{data['avg_f1']:.3f}±{data['std_f1']:.3f} | "
                    f"{data['avg_precision']:.3f}±{data['std_precision']:.3f} | "
                    f"{data['avg_recall']:.3f}±{data['std_recall']:.3f} | "
                    f"{data['avg_roc_auc']:.3f}±{data['std_roc_auc']:.3f} | "
                    f"{data['avg_time']:.2f}s |"
                )
            else:
                lines.append(f"| {algo_name} | - | - | - | - | - |")
        
        lines.extend([
            "",
            "### 순위",
            "",
        ])
        
        best = results.get("summary", {}).get("best_algorithm", {})
        lines.append(f"- **F1 기준 최고:** {best.get('by_f1', 'N/A')}")
        lines.append(f"- **ROC-AUC 기준 최고:** {best.get('by_auc', 'N/A')}")
        
        # 강의별 상세 결과
        lines.extend([
            "",
            "## 강의별 상세 결과",
            "",
        ])
        
        for lecture_result in results.get("lecture_results", []):
            lecture_id = lecture_result.get("lecture_id")
            
            if lecture_result.get("error"):
                lines.append(f"### Lecture {lecture_id}: ERROR - {lecture_result['error']}")
                lines.append("")
                continue
            
            lines.extend([
                f"### Lecture {lecture_id}",
                "",
                f"- 페이지 수: {lecture_result.get('num_pages', 0)}",
                f"- 세그먼트 수: {lecture_result.get('num_segments', 0)}",
                f"- Ground Truth: {lecture_result.get('num_ground_truth', 0)}",
                "",
                "| 알고리즘 | F1 | Precision | Recall | AUC | TP | FP | FN |",
                "|---------|---:|----------:|-------:|----:|---:|---:|---:|",
            ])
            
            for algo_name, algo_result in lecture_result.get("results", {}).items():
                if algo_result.get("error"):
                    lines.append(f"| {algo_name} | ERROR | - | - | - | - | - | - |")
                elif algo_result.get("metrics"):
                    m = algo_result["metrics"]
                    cm = m.get("confusion_matrix", {})
                    lines.append(
                        f"| {algo_name} | "
                        f"{m['f1_score']:.3f} | "
                        f"{m['precision']:.3f} | "
                        f"{m['recall']:.3f} | "
                        f"{m['roc_auc']:.3f} | "
                        f"{cm.get('TP', 0)} | "
                        f"{cm.get('FP', 0)} | "
                        f"{cm.get('FN', 0)} |"
                    )
            
            lines.append("")
        
        # 파일 저장
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return report_path


def run_full_benchmark(
    base_dir: str,
    lecture_ids: List[int],
    algorithms: List[str] = None,
    embedding_fn=None,
    llm_fn=None,
    grouping: str = "none",
    group_duration: float = 30.0,
    confidence_threshold: float = 0.4,
) -> Dict[str, Any]:
    """벤치마크 실행 편의 함수"""
    runner = BenchmarkRunner(
        base_dir=base_dir,
        embedding_fn=embedding_fn,
        llm_fn=llm_fn,
    )
    
    return runner.run_full_benchmark(
        lecture_ids=lecture_ids,
        algorithms=algorithms,
        grouping=grouping,
        group_duration=group_duration,
        confidence_threshold=confidence_threshold,
    )


# CLI 인터페이스
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="동기화 알고리즘 벤치마크")
    parser.add_argument("--lectures", type=str, required=True, help="강의 ID (쉼표로 구분)")
    parser.add_argument("--algorithms", type=str, default=None, help="알고리즘 (쉼표로 구분)")
    parser.add_argument("--grouping", type=str, default="none", help="그룹화 방식")
    parser.add_argument("--group-duration", type=float, default=30.0, help="그룹화 시간(초)")
    parser.add_argument("--base-dir", type=str, default="./data", help="데이터 디렉토리")
    
    args = parser.parse_args()
    
    lecture_ids = [int(x.strip()) for x in args.lectures.split(",")]
    algorithms = [x.strip() for x in args.algorithms.split(",")] if args.algorithms else None
    
    results = run_full_benchmark(
        base_dir=args.base_dir,
        lecture_ids=lecture_ids,
        algorithms=algorithms,
        grouping=args.grouping,
        group_duration=args.group_duration,
    )
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)