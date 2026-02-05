"""
벤치마크 결과 시각화

벤치마크 결과를 그래프와 표로 시각화합니다.

생성되는 파일:
1. benchmark_charts_*.html - 전체 결과 차트 (F1, Precision, Recall, ROC-AUC)
2. roc_curves_*.html - ROC Curve 그래프
3. confusion_matrix_*.html - Threshold별 혼동행렬

사용법:
    python -m app.sync_algorithms.visualize --result benchmark_20240131_120000.json
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import argparse


# 알고리즘 표시 이름
ALGO_DISPLAY_NAMES = {
    "exact_matching": "1. Exact",
    "cosine_similarity": "2. Cosine", 
    "hybrid": "3. Hybrid(E&C)",
    "llm_transcription": "4. LLM-text",
    "structured_pdf": "5. Title-weighted",
    "llm_semantic": "6. LLM",
}

ALGO_ORDER = ["exact_matching", "cosine_similarity", "hybrid", 
              "llm_transcription", "structured_pdf", "llm_semantic"]


def generate_main_chart_html(results: Dict[str, Any]) -> str:
    """메인 결과 차트 HTML 생성"""
    
    summary = results.get("summary", {}).get("evaluations", {})
    
    # 데이터 추출 (순서대로)
    algorithms = []
    f1_scores = []
    precisions = []
    recalls = []
    aucs = []
    
    for algo in ALGO_ORDER:
        if algo in summary and "avg_f1" in summary[algo]:
            algorithms.append(ALGO_DISPLAY_NAMES.get(algo, algo))
            data = summary[algo]
            f1_scores.append(data["avg_f1"])
            precisions.append(data["avg_precision"])
            recalls.append(data["avg_recall"])
            aucs.append(data["avg_roc_auc"])
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>동기화 알고리즘 평가 결과</title>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #333; margin-bottom: 30px; }}
        h2 {{ color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
        
        .result-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .result-table th {{
            background: #4a90d9;
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: 600;
        }}
        .result-table td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }}
        .result-table td:first-child {{ text-align: left; font-weight: 500; }}
        .result-table tr:hover {{ background: #f5f9ff; }}
        .best {{ background: #e8f5e9 !important; font-weight: bold; }}
        
        .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 30px; }}
        .chart-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-box h3 {{ margin-top: 0; color: #555; }}
        
        .info-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 동기화 알고리즘 평가 결과</h1>
        
        <div class="info-box">
            <strong>실험 설정:</strong>
            강의 {results['params']['lecture_ids']} | 
            그룹화: {results['params']['grouping']} ({results['params']['group_duration']}초) |
            Threshold: {results['params']['confidence_threshold']}
        </div>
        
        <h2>📋 성능 비교표</h2>
        <table class="result-table">
            <thead>
                <tr>
                    <th style="width:25%">방법</th>
                    <th>F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>ROC-AUC</th>
                </tr>
            </thead>
            <tbody>
                {"".join([
                    f'<tr><td>{algorithms[i]}</td><td>{f1_scores[i]:.4f}</td><td>{precisions[i]:.4f}</td><td>{recalls[i]:.4f}</td><td>{aucs[i]:.4f}</td></tr>'
                    for i in range(len(algorithms))
                ])}
            </tbody>
        </table>
        
        <h2>📈 성능 그래프</h2>
        <div class="charts">
            <div class="chart-box">
                <h3>F1 / Precision / Recall 비교</h3>
                <canvas id="metricsChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>ROC-AUC 비교</h3>
                <canvas id="aucChart"></canvas>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart-box" style="grid-column: span 2;">
                <h3>🎯 종합 레이더 차트</h3>
                <canvas id="radarChart" style="max-height: 400px;"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        const algorithms = {json.dumps(algorithms)};
        const f1Scores = {json.dumps(f1_scores)};
        const precisions = {json.dumps(precisions)};
        const recalls = {json.dumps(recalls)};
        const aucs = {json.dumps(aucs)};
        
        const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
        
        // 1. 메트릭스 비교 차트
        new Chart(document.getElementById('metricsChart'), {{
            type: 'bar',
            data: {{
                labels: algorithms,
                datasets: [
                    {{ label: 'F1', data: f1Scores, backgroundColor: 'rgba(54, 162, 235, 0.7)' }},
                    {{ label: 'Precision', data: precisions, backgroundColor: 'rgba(255, 99, 132, 0.7)' }},
                    {{ label: 'Recall', data: recalls, backgroundColor: 'rgba(75, 192, 192, 0.7)' }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true, max: 1 }} }}
            }}
        }});
        
        // 2. AUC 차트
        new Chart(document.getElementById('aucChart'), {{
            type: 'bar',
            data: {{
                labels: algorithms,
                datasets: [{{ label: 'ROC-AUC', data: aucs, backgroundColor: colors }}]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true, max: 1 }} }}
            }}
        }});
        
        // 3. 레이더 차트
        new Chart(document.getElementById('radarChart'), {{
            type: 'radar',
            data: {{
                labels: ['F1', 'Precision', 'Recall', 'ROC-AUC'],
                datasets: algorithms.map((algo, i) => ({{
                    label: algo,
                    data: [f1Scores[i], precisions[i], recalls[i], aucs[i]],
                    borderColor: colors[i],
                    backgroundColor: colors[i].replace(')', ', 0.2)').replace('rgb', 'rgba').replace('#', 'rgba('),
                    fill: true
                }}))
            }},
            options: {{
                responsive: true,
                scales: {{ r: {{ beginAtZero: true, max: 1 }} }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def generate_roc_curves_html(results: Dict[str, Any]) -> str:
    """ROC Curve 그래프 HTML 생성"""
    
    # 각 알고리즘의 ROC 데이터 수집
    roc_data_all = {}
    
    for lecture_result in results.get("lecture_results", []):
        if lecture_result.get("error"):
            continue
        
        for algo_name, algo_result in lecture_result.get("results", {}).items():
            if algo_result.get("roc_data"):
                if algo_name not in roc_data_all:
                    roc_data_all[algo_name] = []
                roc_data_all[algo_name].append(algo_result["roc_data"])
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>ROC Curves</title>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f8f9fa;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { text-align: center; color: #333; }
        .chart-container {
            background: white; padding: 30px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-top: 20px;
        }
        .legend-info {
            background: #fff3e0; padding: 15px; border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 ROC Curves</h1>
        
        <div class="legend-info">
            <strong>ROC Curve 해석:</strong><br>
            - 좌상단에 가까울수록 좋은 성능<br>
            - 대각선(점선)은 랜덤 분류기 (AUC = 0.5)<br>
            - AUC가 1에 가까울수록 좋음
        </div>
        
        <div class="chart-container">
            <canvas id="rocChart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('rocChart').getContext('2d');
        const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
        
        const datasets = [
            // 대각선 (랜덤 분류기)
            {
                label: 'Random (AUC=0.5)',
                data: [{x: 0, y: 0}, {x: 1, y: 1}],
                borderColor: '#ccc',
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            }
        ];
        
        // TODO: 실제 ROC 데이터 추가
        // 현재는 샘플 데이터
        
        new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
                    y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
                },
                plugins: {
                    title: { display: true, text: 'ROC Curve - All Algorithms' }
                }
            }
        });
    </script>
</body>
</html>
"""
    return html


def generate_confusion_matrix_html(results: Dict[str, Any]) -> str:
    """Threshold별 혼동행렬 HTML 생성"""
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Confusion Matrix by Threshold</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f8f9fa;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; color: #333; }
        h2 { color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }
        
        .threshold-section {
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;
        }
        
        .cm-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .cm-box { text-align: center; }
        .cm-box h4 { margin: 0 0 10px 0; color: #555; }
        
        .confusion-matrix {
            display: grid;
            grid-template-columns: 80px 80px;
            gap: 2px;
            justify-content: center;
        }
        .cm-cell {
            width: 80px; height: 60px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 18px;
            border-radius: 4px;
        }
        .tp { background: #c8e6c9; color: #2e7d32; }
        .tn { background: #c8e6c9; color: #2e7d32; }
        .fp { background: #ffcdd2; color: #c62828; }
        .fn { background: #ffcdd2; color: #c62828; }
        
        .cm-labels {
            display: grid;
            grid-template-columns: 80px 80px;
            gap: 2px;
            justify-content: center;
            margin-top: 5px;
            font-size: 12px; color: #666;
        }
        
        .info-box {
            background: #e3f2fd; padding: 15px; border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Threshold별 혼동행렬 (Confusion Matrix)</h1>
        
        <div class="info-box">
            <strong>혼동행렬 해석:</strong><br>
            - <span style="color:#2e7d32">TP (True Positive)</span>: 정확하게 매칭됨 (예측 O, 실제 O)<br>
            - <span style="color:#2e7d32">TN (True Negative)</span>: 정확하게 제외됨 (예측 X, 실제 X)<br>
            - <span style="color:#c62828">FP (False Positive)</span>: 잘못 매칭됨 (예측 O, 실제 X)<br>
            - <span style="color:#c62828">FN (False Negative)</span>: 놓침 (예측 X, 실제 O)
        </div>
"""
    
    # 각 threshold에 대한 섹션 추가
    for thresh in thresholds:
        html += f"""
        <div class="threshold-section">
            <h2>Threshold = {thresh}</h2>
            <p style="color:#666">confidence ≥ {thresh} 인 경우만 매칭으로 판정</p>
            <div class="cm-grid">
"""
        
        # 각 알고리즘에 대한 혼동행렬
        for algo in ALGO_ORDER[:4]:  # 처음 4개만 (나머지는 LLM 필요)
            display_name = ALGO_DISPLAY_NAMES.get(algo, algo)
            # 샘플 데이터 (실제로는 results에서 가져와야 함)
            html += f"""
                <div class="cm-box">
                    <h4>{display_name}</h4>
                    <div class="confusion-matrix">
                        <div class="cm-cell tp">TP<br><small>-</small></div>
                        <div class="cm-cell fp">FP<br><small>-</small></div>
                        <div class="cm-cell fn">FN<br><small>-</small></div>
                        <div class="cm-cell tn">TN<br><small>-</small></div>
                    </div>
                    <div class="cm-labels">
                        <span>Pred: Yes</span>
                        <span>Pred: No</span>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    return html


def generate_full_visualization(results_path: str, output_dir: str = None) -> Dict[str, str]:
    """전체 시각화 파일 생성
    
    Args:
        results_path: benchmark_*.json 파일 경로
        output_dir: 출력 디렉토리
        
    Returns:
        생성된 파일 경로 딕셔너리
    """
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = results_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = results.get("timestamp", "unknown")
    generated_files = {}
    
    # 1. 메인 차트
    main_html = generate_main_chart_html(results)
    main_path = output_dir / f"benchmark_charts_{timestamp}.html"
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_html)
    generated_files["메인 차트"] = str(main_path)
    
    # 2. ROC Curves
    roc_html = generate_roc_curves_html(results)
    roc_path = output_dir / f"roc_curves_{timestamp}.html"
    with open(roc_path, "w", encoding="utf-8") as f:
        f.write(roc_html)
    generated_files["ROC Curves"] = str(roc_path)
    
    # 3. Confusion Matrix
    cm_html = generate_confusion_matrix_html(results)
    cm_path = output_dir / f"confusion_matrix_{timestamp}.html"
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(cm_html)
    generated_files["혼동행렬"] = str(cm_path)
    
    return generated_files


def generate_visualization(results_path: str, output_dir: str = None):
    """기존 호환성 유지용 함수"""
    files = generate_full_visualization(results_path, output_dir)
    return Path(list(files.values())[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="벤치마크 결과 시각화")
    parser.add_argument("--result", "-r", type=str, required=True, help="benchmark_*.json 파일 경로")
    parser.add_argument("--output", "-o", type=str, default=None, help="출력 디렉토리")
    
    args = parser.parse_args()
    
    files = generate_full_visualization(args.result, args.output)
    
    print("생성된 파일:")
    for name, path in files.items():
        print(f"  - {name}: {path}")
