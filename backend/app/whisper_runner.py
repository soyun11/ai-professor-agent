from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

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
    
    # SSH 설정
    SSH_HOST = "test223"
    REMOTE_DIR = "/mnt/home_dnlab/sypark/whisper-gpu"
    
    audio_path = Path(audio_path)
    remote_audio = f"{REMOTE_DIR}/{audio_path.name}"
    remote_json = f"{REMOTE_DIR}/result.json"
    
    print(f"[STT] 오디오 파일 업로드 중: {audio_path.name}")
    
    # 1. 오디오 파일을 GPU 서버로 전송
    scp_cmd = ["scp", str(audio_path), f"{SSH_HOST}:{remote_audio}"]
    subprocess.run(scp_cmd, check=True)
    print(f"[STT] 업로드 완료")
    
    # 2. GPU 서버에서 Whisper 실행
    print(f"[STT] GPU 서버에서 변환 중... (model: {model_name})")
    
    ssh_script = f'''
cd {REMOTE_DIR}
source venv/bin/activate
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0] + '/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas; print(nvidia.cublas.__path__[0] + '/lib')"):$LD_LIBRARY_PATH

python3 << 'PYTHON_EOF'
from faster_whisper import WhisperModel
import json

model = WhisperModel("{model_name}", device="{device}", compute_type="{compute_type}")
segments_iter, info = model.transcribe(
    "{remote_audio}",
    language="{language}",
    beam_size=5,
    vad_filter={vad_filter},
    condition_on_previous_text=False,
    temperature=0.0,
)

segments = []
for s in segments_iter:
    text = (s.text or "").strip()
    if text and len(text) > 1:
        segments.append({{"start": float(s.start), "end": float(s.end), "text": text}})
        print(f"[{{s.end:.1f}}s] {{text}}")

payload = {{
    "lecture_id": {lecture_id},
    "model": "{model_name}",
    "language": "{language}",
    "duration_sec": getattr(info, "duration", 0),
    "num_segments": len(segments),
    "segments": segments,
}}

with open("{remote_json}", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("완료!")
PYTHON_EOF
'''
    
    ssh_cmd = ["ssh", SSH_HOST, ssh_script]
    subprocess.run(ssh_cmd, check=True)
    
    # 3. 결과 JSON 가져오기
    print(f"[STT] 결과 다운로드 중...")
    scp_result = ["scp", f"{SSH_HOST}:{remote_json}", str(out_json_path)]
    subprocess.run(scp_result, check=True)
    
    # 4. 결과 읽어서 반환
    with open(out_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    print(f"[STT] 완료! segments: {payload.get('num_segments', 0)}개")
    return payload