import json
import subprocess
from pathlib import Path
from typing import List
import tempfile


def get_embeddings_from_gpu(texts: List[str]) -> List[List[float]]:
    """
    GPU 서버에서 한국어 임베딩 생성
    
    Args:
        texts: 임베딩할 텍스트 리스트
    
    Returns:
        임베딩 벡터 리스트
    """
    SSH_HOST = "test223"
    REMOTE_DIR = "/mnt/home_dnlab/sypark/whisper-gpu"
    
    # 임시 파일 경로
    local_input = Path(tempfile.gettempdir()) / "embed_input.json"
    local_output = Path(tempfile.gettempdir()) / "embed_output.json"
    remote_input = f"{REMOTE_DIR}/embed_input.json"
    remote_output = f"{REMOTE_DIR}/embed_output.json"
    
    # 1. 입력 JSON 저장
    with open(local_input, 'w', encoding='utf-8') as f:
        json.dump({'texts': texts}, f, ensure_ascii=False)
    
    print(f'[Embedding] {len(texts)}개 텍스트 업로드 중...')
    
    # 2. GPU 서버로 전송
    subprocess.run(['scp', str(local_input), f'{SSH_HOST}:{remote_input}'], check=True)
    
    # 3. GPU 서버에서 임베딩 실행
    print('[Embedding] GPU 서버에서 임베딩 생성 중...')
    ssh_cmd = f'cd {REMOTE_DIR} && source venv/bin/activate && python3 embedding_server.py {remote_input} {remote_output}'
    subprocess.run(['ssh', SSH_HOST, ssh_cmd], check=True)
    
    # 4. 결과 가져오기
    subprocess.run(['scp', f'{SSH_HOST}:{remote_output}', str(local_output)], check=True)
    
    # 5. 결과 읽기
    with open(local_output, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    print(f'[Embedding] 완료!')
    return result['embeddings']