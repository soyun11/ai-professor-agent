from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import whisper

_model = None

def get_model():
    global _model
    if _model is None:
        # tiny / base / small / medium / large 중 선택
        # MVP는 base 추천
        _model = whisper.load_model("base")
    return _model


def transcribe_audio(audio_abs_path: Path) -> List[Dict[str, Any]]:
    model = get_model()

    result = model.transcribe(
        str(audio_abs_path),
        language="ko",  # 영어면 "en" 또는 None
        fp16=False,     # CPU면 False 권장(글에서도 CPU일 때 FP16 warning 언급) :contentReference[oaicite:6]{index=6}
    )

    # result["segments"] = [{start, end, text, ...}, ...]
    out = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {"start": float(seg["start"]), "end": float(seg["end"]), "text": text}
        )
    return out
