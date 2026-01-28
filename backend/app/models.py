from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint


class Lecture(SQLModel, table=True):
    """강의 테이블"""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Asset(SQLModel, table=True):
    """파일 정보 테이블"""
    id: Optional[int] = Field(default=None, primary_key=True)
    lecture_id: int = Field(index=True)

    pdf_path: Optional[str] = None
    audio_path: Optional[str] = None

    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

# STT 실행해서 SRT 파싱할 DB 모델 추가
class TranscriptSegment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    lecture_id: int = Field(index=True)

    start: float
    end: float
    text: str

# DB에 페이지 앵커 저장 테이블 만들기(백엔드)
class PageAnchor(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("lecture_id", "page", name="uq_page_anchor_lecture_page"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    lecture_id: int = Field(index=True)
    page: int = Field(index=True)  # 1-based
    time: float  # seconds