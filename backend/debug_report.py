import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from app.sync_algorithms.evaluation import SyncEvaluator, GroundTruth
from app.sync_algorithms.base import SyncAnchor, TextProcessor

# 사용법:
#   python debug_report.py 1,2,3 30 exact_matching
#   python debug_report.py 1,2,3 30 kiwi_auto_sync
#   python debug_report.py 1,2,3 30 cosine_similarity

AUTOSYNC_ALGOS = {'kiwi_auto_sync', 'original_auto_sync'}

def sec_to_mmss(sec):
    if sec is None: return "  None "
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"

def get_fail_reason(e):
    if e['is_correct']: return ""
    conf = e['confidence']
    err = e['error'] or 0
    if conf == 0.3: return "보간값 (키워드 매칭 실패 → 선형 추정)"
    elif conf == 0.5: return "경계값 (시작/끝 고정값)"
    elif err > 120: return f"오차 너무 큼 ({err:.0f}s) → 잘못된 세그먼트 매칭"
    elif err > 30: return f"오차 초과 ({err:.0f}s > 30s) → 키워드 부족"
    else: return f"오차 {err:.0f}s (threshold 미만)"

def load_anchors(lecture_id: int, algo_name: str):
    """알고리즘에 따라 적절한 JSON에서 앵커 로드"""
    results_dir = Path("data/benchmark_results")

    if algo_name in AUTOSYNC_ALGOS:
        # autosync_eval_*.json 에서 로드
        candidates = sorted(results_dir.glob("autosync_eval_*.json"))
        if not candidates:
            raise FileNotFoundError("autosync_eval_*.json 파일이 없습니다. run_autosync_eval.py를 먼저 실행하세요.")
        latest = candidates[-1]
        with open(latest) as f:
            r = json.load(f)
        lecture = next(l for l in r['lecture_results'] if l['lecture_id'] == lecture_id)
        if algo_name not in lecture['results']:
            raise KeyError(f"{algo_name} 결과가 없습니다.")
        anchors_raw = lecture['results'][algo_name]['anchors']
        # autosync는 confidence 필드가 없으므로 0.5로 채움
        for a in anchors_raw:
            a.setdefault('confidence', 0.5)
        return anchors_raw, latest.name

    else:
        # benchmark_*.json 에서 로드
        candidates = sorted(results_dir.glob("benchmark_2*.json"))
        if not candidates:
            raise FileNotFoundError("benchmark_*.json 파일이 없습니다.")
        latest = candidates[-1]
        with open(latest) as f:
            r = json.load(f)
        lecture = next(l for l in r['lecture_results'] if l['lecture_id'] == lecture_id)
        if algo_name not in lecture['results']:
            available = list(lecture['results'].keys())
            raise KeyError(f"{algo_name} 결과가 없습니다. 사용 가능: {available}")
        anchors_raw = lecture['results'][algo_name]['anchors']
        return anchors_raw, latest.name


def print_report(lecture_id: int, tolerance: float = 30.0, algo_name: str = 'exact_matching', save: bool = True):
    anchors_raw, source_file = load_anchors(lecture_id, algo_name)

    with open(f'data/lectures/{lecture_id}/ground_truth.json') as f:
        gt_raw = json.load(f)

    with open(f'lectures/{lecture_id}/pages.json') as f:
        pages_raw = json.load(f)
    pages_text = {p['page']: p.get('text', '') for p in pages_raw.get('pages', [])}

    with open(f'lectures/{lecture_id}/transcript.json') as f:
        transcript_raw = json.load(f)
    segments_raw = transcript_raw.get('segments', transcript_raw) if isinstance(transcript_raw, dict) else transcript_raw

    anchors = [SyncAnchor(page=a['page'], time=a['time'], confidence=a['confidence']) for a in anchors_raw]
    gt = [GroundTruth(page=g['page'], time=float(g['time']), tolerance=tolerance)
          for g in gt_raw if not g.get('skip') and g.get('time') is not None]
    gt_dict = {g.page: g.time for g in gt}

    evaluator = SyncEvaluator(tolerance=tolerance)
    result = evaluator.evaluate(anchors, gt, confidence_threshold=0.05)

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  Lecture {lecture_id} | tolerance=±{int(tolerance)}초 | algo={algo_name} | {source_file}")
    lines.append(f"{'='*80}")
    lines.append(f"  {'Page':>4} | {'예측':>7} | {'정답':>7} | {'오차':>6} | {'신뢰도':>6} | 결과 | 실패 이유")
    lines.append(f"  {'-'*75}")

    fail_details = []

    for e in result.page_errors:
        status = "✅" if e['is_correct'] else "❌"
        err_str = f"{e['error']:.0f}s" if e['error'] is not None else "  -"
        reason = get_fail_reason(e)
        lines.append(
            f"  {e['page']:>4} | {sec_to_mmss(e['predicted_time']):>7} | "
            f"{sec_to_mmss(e['ground_truth_time']):>7} | {err_str:>6} | "
            f"{e['confidence']:>6.3f} | {status}  | {reason}"
        )

        if not e['is_correct']:
            page_num = e['page']
            pdf_text = pages_text.get(page_num, "")
            pdf_kw = TextProcessor.extract_keywords(pdf_text)

            gt_time = gt_dict.get(page_num, 0)
            nearby_segs = [s for s in segments_raw
                          if abs(float(s.get('start', 0)) - gt_time) <= 60]
            nearby_text = " ".join(s.get('text', '') for s in nearby_segs)
            nearby_kw = TextProcessor.extract_keywords(nearby_text)

            pred_time = e['predicted_time'] or 0
            pred_segs = [s for s in segments_raw
                        if abs(float(s.get('start', 0)) - pred_time) <= 60]
            pred_text = " ".join(s.get('text', '') for s in pred_segs)
            pred_kw = TextProcessor.extract_keywords(pred_text)

            common_nearby = pdf_kw & nearby_kw
            common_pred = pdf_kw & pred_kw

            fail_details.append({
                "page": page_num,
                "reason": reason,
                "pdf_keywords": sorted(pdf_kw)[:15],
                "gt_time": gt_time,
                "nearby_keywords": sorted(nearby_kw)[:15],
                "common_with_gt": sorted(common_nearby),
                "pred_time": pred_time,
                "pred_keywords": sorted(pred_kw)[:15],
                "common_with_pred": sorted(common_pred),
            })

    lines.append(f"{'='*80}")
    lines.append(f"  F1={result.f1_score:.3f} | P={result.precision:.3f} | R={result.recall:.3f} | AUC={result.roc_auc:.3f}")
    lines.append(f"  TP={result.tp} | FP={result.fp} | FN={result.fn} | 총 {result.total_pages}페이지")
    lines.append(f"{'='*80}")

    lines.append(f"\n\n{'#'*80}")
    lines.append(f"  실패 페이지 상세 분석 (Lecture {lecture_id} / {algo_name})")
    lines.append(f"{'#'*80}")

    for d in fail_details:
        lines.append(f"\n  ── Page {d['page']} ──────────────────────────────────────────")
        lines.append(f"  원인: {d['reason']}")
        lines.append(f"  PDF  키워드 ({len(d['pdf_keywords'])}개): {', '.join(d['pdf_keywords']) or '없음'}")
        lines.append(f"  정답 근처 음성 키워드 ({sec_to_mmss(d['gt_time'])} ±60s): {', '.join(d['nearby_keywords']) or '없음'}")
        lines.append(f"  → 공통 키워드: {', '.join(d['common_with_gt']) if d['common_with_gt'] else '❌ 없음 (매칭 불가)'}")
        lines.append(f"  예측 근처 음성 키워드 ({sec_to_mmss(d['pred_time'])} ±60s): {', '.join(d['pred_keywords']) or '없음'}")
        lines.append(f"  → 공통 키워드: {', '.join(d['common_with_pred']) if d['common_with_pred'] else '❌ 없음'}")

    lines.append(f"\n{'#'*80}\n")

    output = "\n".join(lines)
    print(output)

    if save:
        save_path = Path(f"data/debug_report_lecture{lecture_id}_{algo_name}.txt")
        save_path.write_text(output, encoding="utf-8")
        print(f"  💾 저장됨: {save_path}")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 3]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    algo = sys.argv[3] if len(sys.argv) > 3 else 'exact_matching'
    for lid in ids:
        print_report(lid, tol, algo)