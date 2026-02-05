#!/usr/bin/env python3
"""
벤치마크 실행 스크립트

사용법:
    python run_benchmark.py --lectures 1,2,3
    python run_benchmark.py --lectures 1,2,3 --algorithms hybrid,cosine_similarity
    python run_benchmark.py --lectures 1,2,3 --grouping duration --group-duration 30

결과:
    - data/benchmark_results/benchmark_YYYYMMDD_HHMMSS.json
    - data/benchmark_results/benchmark_report_YYYYMMDD_HHMMSS.md
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    print("✓ .env 파일 로드됨")
except ImportError:
    print("⚠ python-dotenv 없음. 환경변수에서 직접 읽습니다.")

from app.sync_algorithms.benchmark import BenchmarkRunner
from app.sync_algorithms import ALGORITHMS


def main():
    parser = argparse.ArgumentParser(
        description="동기화 알고리즘 벤치마크 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 기본 실행 (3개 강의, 모든 알고리즘)
    python run_benchmark.py --lectures 1,2,3
    
    # 특정 알고리즘만 테스트
    python run_benchmark.py --lectures 1,2,3 --algorithms hybrid,cosine_similarity
    
    # 그룹화 옵션 변경
    python run_benchmark.py --lectures 1,2,3 --grouping duration --group-duration 15
    
    # LLM 기반 알고리즘 포함 (API 키 필요)
    python run_benchmark.py --lectures 1 --include-llm
        """
    )
    
    parser.add_argument(
        "--lectures", "-l",
        type=str,
        required=True,
        help="테스트할 강의 ID (쉼표로 구분, 예: 1,2,3)"
    )
    
    parser.add_argument(
        "--algorithms", "-a",
        type=str,
        default=None,
        help=f"테스트할 알고리즘 (쉼표로 구분, 기본: 전체)\n사용 가능: {', '.join(ALGORITHMS.keys())}"
    )
    
    parser.add_argument(
        "--grouping", "-g",
        type=str,
        default="none",
        choices=["none", "duration", "count"],
        help="자막 그룹화 방식 (기본: none)"
    )
    
    parser.add_argument(
        "--group-duration",
        type=float,
        default=30.0,
        help="그룹화 시간(초), grouping=duration일 때 사용 (기본: 30)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="신뢰도 임계값 (기본: 0.4)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./data",
        help="데이터 디렉토리 (기본: ./data)"
    )
    
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="LLM 기반 알고리즘 포함 (OPENAI_API_KEY 환경변수 필요)"
    )
    
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="임베딩 기반 알고리즘 스킵 (GPU 서버 없을 때)"
    )
    
    args = parser.parse_args()
    
    # 파라미터 파싱
    lecture_ids = [int(x.strip()) for x in args.lectures.split(",")]
    
    if args.algorithms:
        algorithms = [x.strip() for x in args.algorithms.split(",")]
        # 유효성 검사
        for algo in algorithms:
            if algo not in ALGORITHMS:
                print(f"Error: Unknown algorithm '{algo}'")
                print(f"Available: {', '.join(ALGORITHMS.keys())}")
                sys.exit(1)
    else:
        algorithms = list(ALGORITHMS.keys())
    
    # 임베딩 함수 설정
    embedding_fn = None
    if not args.skip_embedding:
        try:
            from app.embedding_runner import get_embeddings_from_gpu
            embedding_fn = get_embeddings_from_gpu
            print("✓ Embedding function loaded (GPU server)")
        except ImportError as e:
            print(f"⚠ Embedding function not available: {e}")
            print("  임베딩 기반 알고리즘은 스킵됩니다.")
    
    # LLM 함수 설정
    llm_fn = None
    if args.include_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                
                def call_llm(prompt: str) -> str:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )
                    return response.choices[0].message.content
                
                llm_fn = call_llm
                print("✓ LLM function loaded (OpenAI API)")
            except Exception as e:
                print(f"⚠ LLM function error: {e}")
        else:
            print("⚠ OPENAI_API_KEY not set. LLM 알고리즘은 스킵됩니다.")
    
    # 알고리즘 필터링
    if args.skip_embedding:
        algorithms = [a for a in algorithms if not ALGORITHMS[a]["requires_embedding"]]
    
    if not args.include_llm:
        algorithms = [a for a in algorithms if not ALGORITHMS[a]["requires_llm"]]
    
    if not algorithms:
        print("Error: 실행할 알고리즘이 없습니다.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("벤치마크 설정")
    print("="*60)
    print(f"강의 ID: {lecture_ids}")
    print(f"알고리즘: {algorithms}")
    print(f"그룹화: {args.grouping} ({args.group_duration}초)")
    print(f"신뢰도 임계값: {args.confidence_threshold}")
    print(f"데이터 디렉토리: {args.base_dir}")
    
    # Ground Truth 확인
    print("\nGround Truth 확인:")
    for lid in lecture_ids:
        gt_path = Path(args.base_dir) / "lectures" / str(lid) / "ground_truth.json"
        if gt_path.exists():
            print(f"  ✓ Lecture {lid}: {gt_path}")
        else:
            print(f"  ✗ Lecture {lid}: ground_truth.json 없음!")
            print(f"    먼저 정답 데이터를 설정하세요:")
            print(f"    POST /experiments/lectures/{lid}/ground_truth")
    
    # 벤치마크 실행
    runner = BenchmarkRunner(
        base_dir=args.base_dir,
        embedding_fn=embedding_fn,
        llm_fn=llm_fn,
    )
    
    results = runner.run_full_benchmark(
        lecture_ids=lecture_ids,
        algorithms=algorithms,
        grouping=args.grouping,
        group_duration=args.group_duration,
        confidence_threshold=args.confidence_threshold,
    )
    
    # 결과 요약 출력
    print("\n" + "="*70)
    print("📊 평가 결과")
    print("="*70)
    
    summary = results.get("summary", {}).get("evaluations", {})
    
    # 알고리즘 이름 매핑
    algo_display_names = {
        "exact_matching": "1. Exact",
        "cosine_similarity": "2. Cosine", 
        "hybrid": "3. Hybrid(E&C)",
        "llm_transcription": "4. LLM-text",
        "structured_pdf": "5. Title-weighted",
        "llm_semantic": "6. LLM",
    }
    
    # 알고리즘 순서 정의
    algo_order = ["exact_matching", "cosine_similarity", "hybrid", 
                  "llm_transcription", "structured_pdf", "llm_semantic"]
    
    print("\n┌" + "─"*20 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*12 + "┐")
    print(f"│ {'방법':<18} │ {'F1':^10} │ {'Precision':^10} │ {'Recall':^10} │ {'ROC-AUC':^10} │")
    print("├" + "─"*20 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┤")
    
    for algo_name in algo_order:
        if algo_name not in summary:
            continue
        data = summary[algo_name]
        display_name = algo_display_names.get(algo_name, algo_name)
        
        if "avg_f1" in data:
            print(f"│ {display_name:<18} │ {data['avg_f1']:^10.4f} │ {data['avg_precision']:^10.4f} │ "
                  f"{data['avg_recall']:^10.4f} │ {data['avg_roc_auc']:^10.4f} │")
        else:
            print(f"│ {display_name:<18} │ {'N/A':^10} │ {'N/A':^10} │ {'N/A':^10} │ {'N/A':^10} │")
    
    print("└" + "─"*20 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*12 + "┘")
    
    # 최고 성능 알고리즘
    best = results.get("summary", {}).get("best", {})
    print(f"\n🏆 최고 성능:")
    print(f"   F1 기준: {algo_display_names.get(best.get('by_f1'), 'N/A')}")
    print(f"   ROC-AUC 기준: {algo_display_names.get(best.get('by_auc'), 'N/A')}")
    
    # 시각화 생성
    print("\n" + "="*70)
    print("📈 시각화 생성 중...")
    print("="*70)
    
    try:
        from app.sync_algorithms.visualize import generate_full_visualization
        
        # 가장 최근 결과 파일 찾기
        results_dir = Path(args.base_dir) / "benchmark_results"
        latest_json = sorted(results_dir.glob("benchmark_*.json"))[-1]
        
        viz_files = generate_full_visualization(str(latest_json), str(results_dir))
        
        print(f"\n생성된 파일:")
        for name, path in viz_files.items():
            print(f"  - {name}: {path}")
    except Exception as e:
        print(f"⚠ 시각화 생성 실패: {e}")
    
    print("\n✅ 벤치마크 완료!")


if __name__ == "__main__":
    main()
