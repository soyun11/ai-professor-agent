"""
평가 도구 (Evaluation Tools)

동기화 알고리즘의 성능을 평가하기 위한 도구들입니다.

핵심 목표
- “이 알고리즘이 PDF page ↔ audio time을 얼마나 잘 맞추는가?”를
  **한 개의 강의(lecture)** 또는 **여러 강의 묶음**에서 비교 가능하게 만드는 것.

이번 버전에서 쓰는 지표 (요청 반영)
- Precision / Recall / F1-score
- ROC Curve & AUC
- (옵션) Precision-Recall Curve & AP

설명(아주 쉽게)
- 각 페이지(page)에 대해 모델이 “여기가 이 페이지 시작이야”라고 시간(time)을 찍습니다.
- 정답(ground_truth)의 해당 페이지 시간과 비교해서, 오차가 tolerance(초) 안이면 “정답”으로 봅니다.
- 그리고 confidence_threshold 이상인 예측만 “채택(positive)”했다고 보고 TP/FP/FN을 셉니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import SyncAnchor, SyncResult


@dataclass
class GroundTruth:
    """정답 데이터 (Ground Truth)"""
    page: int
    time: float
    tolerance: float = 5.0  # 허용 오차 (초)


@dataclass
class EvaluationResult:
    """평가 결과"""
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

    tp: int
    fp: int
    fn: int

    total_pages: int
    confidence_threshold: float
    tolerance: float

    # 디버깅/시각화용
    page_errors: List[Dict[str, Any]]
    roc_data: Optional[Dict[str, Any]] = None
    pr_data: Optional[Dict[str, Any]] = None


class SyncEvaluator:
    """동기화 알고리즘 평가기"""

    def __init__(self, tolerance: float = 5.0, confidence_threshold: float = 0.0):
        """
        Args:
            tolerance: 정답으로 판정할 시간 허용 오차(초)
            confidence_threshold: 이 값 이상인 예측만 “채택”했다고 봄
        """
        self.tolerance = float(tolerance)
        self.confidence_threshold = float(confidence_threshold)

    def evaluate(
        self,
        predictions: List[SyncAnchor],
        ground_truth: List[GroundTruth],
        *,
        confidence_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
        compute_curves: bool = True,
    ) -> EvaluationResult:
        """예측 결과 평가

        핵심 아이디어(페이지 단위):
        - 각 gt page에 대해 pred가 있으면 score=pred.confidence, 없으면 score=0
        - label=1은 “그 페이지가 tolerance 안으로 맞았을 때”만 1, 그 외는 0
        - threshold 이상 score인 것만 “채택” → TP/FP 계산
        - FN은 “정답 페이지 개수 - TP” (정답 페이지 중 맞춘 개수만 true positive)

        Args:
            predictions: 예측된 앵커 리스트 (page, time, confidence 포함)
            ground_truth: 정답 앵커 리스트
            confidence_threshold: (옵션) 이번 평가에서만 임계값 override
            tolerance: (옵션) 이번 평가에서만 tolerance override
            compute_curves: ROC/PR 곡선 계산 여부

        Returns:
            EvaluationResult
        """

        thr = float(self.confidence_threshold if confidence_threshold is None else confidence_threshold)
        tol = float(self.tolerance if tolerance is None else tolerance)

        pred_dict: Dict[int, SyncAnchor] = {a.page: a for a in predictions}
        gt_dict: Dict[int, GroundTruth] = {gt.page: gt for gt in ground_truth}

        pages = sorted(gt_dict.keys())
        total_pages = len(pages)

        # 페이지별 score/label 만들기
        scores: List[float] = []
        labels: List[bool] = []
        page_errors: List[Dict[str, Any]] = []

        for page in pages:
            gt_time = gt_dict[page].time
            pred = pred_dict.get(page)
            # 디버깅을 위한 코드 추가
            page_tol = float(getattr(gt_dict[page],"tolerance", tol) or tol)

            if pred is None:
                score = 0.0
                is_correct = False
                pred_time = None
                error = None
            else:
                score = float(getattr(pred, "confidence", 0.0) or 0.0)
                pred_time = pred.time
                error = abs(float(pred_time) - float(gt_time))
                is_correct = error <= page_tol # 페이지별 tolerance 적용

            scores.append(score)
            labels.append(is_correct)

            page_errors.append(
                {
                    "page": page,
                    "predicted_time": pred_time,
                    "ground_truth_time": gt_time,
                    "error": error,
                    "confidence": score,
                    "is_correct": bool(is_correct),
                }
            )

        # threshold 적용하여 TP/FP/FN 계산
        tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y)
        fp = sum(1 for s, y in zip(scores, labels) if s >= thr and (not y))
        fn = total_pages - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_pages if total_pages > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        roc_data = None
        pr_data = None
        roc_auc = 0.0

        if compute_curves and total_pages > 0:
            roc_data = self._compute_roc_data_from_scores(scores, labels)
            pr_data = self._compute_pr_data_from_scores(scores, labels)
            roc_auc = float(roc_data.get("auc", 0.0)) if roc_data else 0.0

        return EvaluationResult(
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            roc_auc=float(roc_auc),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            total_pages=int(total_pages),
            confidence_threshold=float(thr),
            tolerance=float(tol),
            page_errors=page_errors,
            roc_data=roc_data,
            pr_data=pr_data,
        )

    @staticmethod
    def _compute_roc_data_from_scores(scores: List[float], labels: List[bool]) -> Dict[str, Any]:
        """ROC curve & AUC (페이지 단위 score/label 기반)"""
        if not scores:
            return {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0}

        # thresholds: unique scores (내림차순)
        thresholds = sorted(set(float(s) for s in scores), reverse=True)

        total_pos = sum(1 for y in labels if y)
        total_neg = len(labels) - total_pos

        tpr_list: List[float] = []
        fpr_list: List[float] = []

        for th in thresholds:
            tp = sum(1 for s, y in zip(scores, labels) if s >= th and y)
            fp = sum(1 for s, y in zip(scores, labels) if s >= th and (not y))

            tpr = tp / total_pos if total_pos > 0 else 0.0
            fpr = fp / total_neg if total_neg > 0 else 0.0

            tpr_list.append(float(tpr))
            fpr_list.append(float(fpr))

        # AUC (trapezoid): fpr가 증가하도록 정렬
        pairs = sorted(zip(fpr_list, tpr_list), key=lambda x: x[0])
        auc = 0.0
        for i in range(1, len(pairs)):
            x1, y1 = pairs[i - 1]
            x2, y2 = pairs[i]
            auc += (x2 - x1) * (y1 + y2) / 2.0

        return {
            "fpr": [p[0] for p in pairs],
            "tpr": [p[1] for p in pairs],
            "thresholds": thresholds,
            "auc": abs(float(auc)),
        }

    @staticmethod
    def _compute_pr_data_from_scores(scores: List[float], labels: List[bool]) -> Dict[str, Any]:
        """Precision-Recall curve & AP (페이지 단위 score/label 기반)"""
        if not scores:
            return {"precision": [], "recall": [], "thresholds": [], "ap": 0.0}

        thresholds = sorted(set(float(s) for s in scores), reverse=True)
        total_pos = sum(1 for y in labels if y)

        precision_list: List[float] = []
        recall_list: List[float] = []

        for th in thresholds:
            selected = [(s, y) for s, y in zip(scores, labels) if s >= th]
            if not selected:
                continue
            tp = sum(1 for _, y in selected if y)
            precision = tp / len(selected)
            recall = tp / total_pos if total_pos > 0 else 0.0

            precision_list.append(float(precision))
            recall_list.append(float(recall))

        # Average Precision (simple area under PR)
        ap = 0.0
        for i in range(1, len(recall_list)):
            ap += (recall_list[i] - recall_list[i - 1]) * precision_list[i]

        return {
            "precision": precision_list,
            "recall": recall_list,
            "thresholds": thresholds[: len(precision_list)],
            "ap": abs(float(ap)),
        }

    def compare_algorithms(
        self,
        results: Dict[str, SyncResult],
        ground_truth: List[GroundTruth],
        *,
        confidence_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """여러 알고리즘 비교 (동일한 threshold/tolerance로 비교)"""
        evaluations: Dict[str, Any] = {}

        for algo_name, sync_result in results.items():
            eval_result = self.evaluate(
                sync_result.anchors,
                ground_truth,
                confidence_threshold=confidence_threshold,
                tolerance=tolerance,
                compute_curves=True,
            )
            evaluations[algo_name] = {
                "f1_score": eval_result.f1_score,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "roc_auc": eval_result.roc_auc,
                "tp": eval_result.tp,
                "fp": eval_result.fp,
                "fn": eval_result.fn,
            }

        rankings = {
            "by_f1": sorted(evaluations.items(), key=lambda x: x[1]["f1_score"], reverse=True),
            "by_auc": sorted(evaluations.items(), key=lambda x: x[1]["roc_auc"], reverse=True),
        }

        return {
            "evaluations": evaluations,
            "rankings": {k: [item[0] for item in v] for k, v in rankings.items()},
            "best_algorithm": {
                "by_f1": rankings["by_f1"][0][0] if rankings["by_f1"] else None,
                "by_auc": rankings["by_auc"][0][0] if rankings["by_auc"] else None,
            },
        }


class GroundTruthManager:
    """정답 데이터 관리"""

    def __init__(self):
        self.ground_truths: Dict[int, List[GroundTruth]] = {}  # lecture_id -> ground_truth

    def add_ground_truth(self, lecture_id: int, page: int, time: float, tolerance: float = 5.0) -> None:
        if lecture_id not in self.ground_truths:
            self.ground_truths[lecture_id] = []
        self.ground_truths[lecture_id].append(GroundTruth(page=page, time=float(time), tolerance=float(tolerance)))

    def set_ground_truth_from_list(self, lecture_id: int, data: List[Dict[str, Any]]) -> None:
        """리스트에서 정답 데이터 설정

        ground_truth.json 포맷은 **그대로 유지**해도 됩니다. ✅
        - {"page": N, "time": 123.4}  → 사용
        - {"page": N, "time": null, "skip": true} → 제외
        """
        self.ground_truths[lecture_id] = []

        for d in data:
            if d.get("skip", False):
                continue
            time_val = d.get("time")
            if time_val is None or isinstance(time_val, str):
                continue
            try:
                time_float = float(time_val)
            except (ValueError, TypeError):
                continue

            self.ground_truths[lecture_id].append(
                GroundTruth(
                    page=int(d["page"]),
                    time=time_float,
                    tolerance=float(d.get("tolerance", 5.0)),
                )
            )

    def get_ground_truth(self, lecture_id: int) -> List[GroundTruth]:
        return self.ground_truths.get(lecture_id, [])

    def export_to_dict(self, lecture_id: int) -> List[Dict[str, Any]]:
        return [{"page": gt.page, "time": gt.time, "tolerance": gt.tolerance} for gt in self.get_ground_truth(lecture_id)]


def calculate_metrics(
    predictions: List[SyncAnchor],
    ground_truth: List[GroundTruth],
    *,
    tolerance: float = 5.0,
    confidence_threshold: float = 0.0,
) -> Dict[str, float]:
    """간편 지표 계산 함수 (4가지 지표만)"""
    evaluator = SyncEvaluator(tolerance=tolerance, confidence_threshold=confidence_threshold)
    result = evaluator.evaluate(predictions, ground_truth, compute_curves=True)

    return {
        "f1_score": result.f1_score,
        "precision": result.precision,
        "recall": result.recall,
        "roc_auc": result.roc_auc,
    }


def generate_evaluation_report(
    results: Dict[str, SyncResult],
    ground_truth: List[GroundTruth],
    *,
    tolerance: float = 5.0,
    confidence_threshold: float = 0.0,
) -> str:
    """평가 보고서 생성 (마크다운)"""
    evaluator = SyncEvaluator(tolerance=tolerance, confidence_threshold=confidence_threshold)
    comparison = evaluator.compare_algorithms(results, ground_truth, confidence_threshold=confidence_threshold, tolerance=tolerance)

    report = "# 동기화 알고리즘 평가 보고서\n\n"
    report += "## 평가 설정\n"
    report += f"- tolerance: {tolerance}초\n"
    report += f"- confidence_threshold: {confidence_threshold}\n"
    report += f"- 정답 페이지 수: {len(ground_truth)}개\n"
    report += f"- 비교 알고리즘 수: {len(results)}개\n\n"

    report += "## 알고리즘별 성능\n\n"
    report += "| 알고리즘 | F1 | Precision | Recall | ROC-AUC | TP | FP | FN |\n"
    report += "|---|---:|---:|---:|---:|---:|---:|---:|\n"

    for algo_name, m in comparison["evaluations"].items():
        report += (
            f"| {algo_name} | {m['f1_score']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['roc_auc']:.3f} "
            f"| {m['tp']} | {m['fp']} | {m['fn']} |\n"
        )

    report += "\n## 순위\n\n"
    report += f"- F1 기준 최고: **{comparison['best_algorithm']['by_f1']}**\n"
    report += f"- ROC-AUC 기준 최고: **{comparison['best_algorithm']['by_auc']}**\n"
    return report
