from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

BASE_DIR = Path(__file__).resolve().parent.parent # backend
DATA_DIR = BASE_DIR / "data"  # backend/data
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_PDF = {".pdf"}
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def _safe_ext(filename: str) -> str:
    return Path(filename).suffix.lower().strip()


def validate_file_extensions(pdf: UploadFile, audio: UploadFile) -> None:
    pdf_ext = _safe_ext(pdf.filename or "")
    audio_ext = _safe_ext(audio.filename or "")

    if pdf_ext not in ALLOWED_PDF:
        raise ValueError(f"PDF 확장자만 허용됩니다: {ALLOWED_PDF}")
    if audio_ext not in ALLOWED_AUDIO:
        raise ValueError(f"오디오 확장자는 다음만 허용됩니다: {ALLOWED_AUDIO}")


def lecture_dir(lecture_id: int) -> Path:
    d = DATA_DIR / "lectures" / str(lecture_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def save_uploads(lecture_id: int, pdf: UploadFile, audio: UploadFile) -> Tuple[str, str]:
    """파일을 서버에 저장하고 경로 반환"""
    validate_file_extensions(pdf, audio)

    d = lecture_dir(lecture_id)
    pdf_path = d / "source.pdf"
    audio_path = d / f"audio{_safe_ext(audio.filename or 'audio.mp3')}"

    # FastAPI UploadFile: async read
    pdf_bytes = await pdf.read()
    audio_bytes = await audio.read()

    pdf_path.write_bytes(pdf_bytes)
    audio_path.write_bytes(audio_bytes)

    # 상대경로 저장(배포시 유연)
    rel_pdf = os.path.relpath(pdf_path, BASE_DIR)
    rel_audio = os.path.relpath(audio_path, BASE_DIR)
    return rel_pdf, rel_audio
