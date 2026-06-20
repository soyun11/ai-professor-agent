from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def transcribe_to_json_with_progress(
    audio_path: Path,
    out_json_path: Path,
    *,
    lecture_id: Optional[int] = None,
    model_name: str = "large-v3",
    language: str = "ko",
    device: str = "cuda",
    compute_type: str = "float16",
    vad_filter: bool = True,
) -> Dict[str, Any]:
    """
    GPU 서버(test223)에서 Whisper 실행
    """
    
    SSH_HOST = "test223"
    REMOTE_DIR = "/mnt/home_dnlab/sypark/whisper-gpu"
    
    audio_path = Path(audio_path)
    remote_audio = f"{REMOTE_DIR}/{audio_path.name}"
    remote_json = f"{REMOTE_DIR}/result.json"
    
    # 1. 오디오 파일 업로드
    print(f"[STT] 오디오 파일 업로드 중: {audio_path.name}")
    subprocess.run(["scp", str(audio_path), f"{SSH_HOST}:{remote_audio}"], check=True)
    print("[STT] 업로드 완료")
    
    # 2. GPU 서버에서 Whisper 실행 (쉘 스크립트 호출)
    print(f"[STT] GPU 서버에서 변환 중... (model: {model_name})")
    ssh_cmd = f"{REMOTE_DIR}/run_whisper.sh {remote_audio} {remote_json} {model_name} {language}"
    subprocess.run(["ssh", SSH_HOST, ssh_cmd], check=True)
    
    # 3. 결과 다운로드
    print("[STT] 결과 다운로드 중...")
    subprocess.run(["scp", f"{SSH_HOST}:{remote_json}", str(out_json_path)], check=True)
    
    # 4. 결과 읽기
    with open(out_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    # lecture_id 추가
    payload["lecture_id"] = lecture_id
    
    # 5. 임시 파일 삭제
    subprocess.run(["ssh", SSH_HOST, f"rm -f {remote_audio} {remote_json}"], check=False)
    
    print(f"[STT] 완료! segments: {payload.get('num_segments', 0)}개")
    return payload