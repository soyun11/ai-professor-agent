#!/bin/bash
#
# 동기화 알고리즘 벤치마크 실행 스크립트
#
# 사용법:
#   ./benchmark.sh                      # 기본 실행 (강의 1,2,3)
#   ./benchmark.sh 1,2,3,4,5           # 특정 강의들
#   ./benchmark.sh 1,2,3 --include-llm # LLM 알고리즘 포함
#
# 결과 파일:
#   data/benchmark_results/benchmark_YYYYMMDD_HHMMSS.json
#   data/benchmark_results/benchmark_report_YYYYMMDD_HHMMSS.md
#

set -e

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 활성화 (있는 경우)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 기본 설정
LECTURES="${1:-1,2,3}"
shift 2>/dev/null || true

echo "========================================"
echo "동기화 알고리즘 벤치마크"
echo "========================================"
echo "강의 ID: $LECTURES"
echo "추가 옵션: $@"
echo ""

# Ground Truth 확인
echo "Ground Truth 확인 중..."
for lid in $(echo $LECTURES | tr ',' ' '); do
    GT_FILE="data/lectures/$lid/ground_truth.json"
    if [ -f "$GT_FILE" ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('$GT_FILE'))))")
        echo "  ✓ Lecture $lid: $COUNT 페이지"
    else
        echo "  ✗ Lecture $lid: ground_truth.json 없음!"
        echo ""
        echo "정답 데이터를 먼저 설정하세요:"
        echo "  curl -X POST http://localhost:8000/experiments/lectures/$lid/ground_truth \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '[{\"page\": 1, \"time\": 0.0}, ...]'"
        exit 1
    fi
done
echo ""

# 벤치마크 실행
echo "벤치마크 실행 중..."
python3 run_benchmark.py --lectures "$LECTURES" "$@"

# 결과 파일 확인
LATEST_RESULT=$(ls -t data/benchmark_results/benchmark_*.json 2>/dev/null | head -1)
LATEST_REPORT=$(ls -t data/benchmark_results/benchmark_report_*.md 2>/dev/null | head -1)

if [ -n "$LATEST_RESULT" ]; then
    echo ""
    echo "========================================"
    echo "결과 파일"
    echo "========================================"
    echo "JSON: $LATEST_RESULT"
    echo "Report: $LATEST_REPORT"
    echo ""
    echo "리포트 미리보기:"
    echo "----------------------------------------"
    head -50 "$LATEST_REPORT"
    echo "..."
    echo "----------------------------------------"
fi

echo ""
echo "✅ 벤치마크 완료!"
