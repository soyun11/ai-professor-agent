import json
import subprocess
from pathlib import Path
import tempfile


def ocr_pdf_with_gpu(pdf_path: str, dpi: int = 200) -> dict:
    SSH_HOST = "test223"
    REMOTE_DIR = "/mnt/home_dnlab/sypark/qwen-ocr"
    
    pdf_path = Path(pdf_path).resolve()
    local_output = Path(tempfile.gettempdir()) / "ocr_output.json"
    remote_pdf = f"{REMOTE_DIR}/temp_input.pdf"
    remote_output = f"{REMOTE_DIR}/ocr_output.json"
    
    # 1. PDF 업로드
    print(f'[OCR] PDF 업로드 중... ({pdf_path.name})')
    subprocess.run(['scp', str(pdf_path), f'{SSH_HOST}:{remote_pdf}'], check=True)
    
    # 2. GPU 서버에서 OCR 실행 (venv python 직접 호출)
    print('[OCR] GPU 서버에서 OCR 실행 중...')
    ssh_cmd = f'{REMOTE_DIR}/venv/bin/python {REMOTE_DIR}/ocr_script.py {remote_pdf} {remote_output} --dpi {dpi}'
    subprocess.run(['ssh', SSH_HOST, ssh_cmd], check=True)
    
    # 3. 결과 가져오기
    subprocess.run(['scp', f'{SSH_HOST}:{remote_output}', str(local_output)], check=True)
    
    # 4. 결과 읽기
    with open(local_output, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    # 5. 임시 파일 삭제
    subprocess.run(['ssh', SSH_HOST, f'rm -f {remote_pdf} {remote_output}'], check=False)
    
    print(f'[OCR] 완료! {result["num_pages"]}페이지')
    return result