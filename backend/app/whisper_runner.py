from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import whisper

_model = None

def transcribe_with_whisper_api(audio_path: Path, model_name: str = "tiny") -> List[Dict[str, Any]]:
    global _model
    if _model is None:
        _model = whisper.load_model(model_name)

    result = _model.transcribe(
        str(audio_path),
        language="ko",
        fp16=False,   # CPU면 False
        verbose=False # 콘솔 출력 최소화
    )

    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text
        })
    return segments
