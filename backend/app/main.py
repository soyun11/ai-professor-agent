from __future__ import annotations # 타입 힌트에서 자기 자신을 참조할 수 있게 함.
from contextlib import asynccontextmanager # 앱이 시작/종료될 때 자동으로 실행되는 코드를 만들 때 사용
from pathlib import Path # 파일 경로를 객체지향적으로 다루기 위한 라이브러리
import select # SQL 쿼리을 위한 모듈

# FastAPI 핵심 컨포넌트
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException # Depends: 함수가 필요한 것들을 자동으로 가져다줌.
from fastapi.middleware.cors import CORSMiddleware # 다른 도메인에서 접속할 수 있게 해주는 설정
from sqlmodel import Session, select # 데이터베이스와 대화하는 통로
# 데이터베이스 초기화 및 세션 관리
from .db import init_db, get_session 

# 데이터 모델 정의 (Lecture, Asset, TranscriptSegment 등)
from .models import Lecture, Asset, TranscriptSegment

# 파일 업로드 저장 로직 + 기본 저장 경로
from .storage import save_uploads, BASE_DIR

# 정적 파일(PDF, 오디오 등) 서빙을 위한 미들웨어
from fastapi.staticfiles import StaticFiles

# 페이지-시간 동기화 앵커 모델
from .models import PageAnchor

# Request Body 파싱용
from fastapi import Body 

# 외부 라이브러리 임포트(OCR, AI, 데이터 처리)
import os, json # 환경변수, JSON 파싱
from typing import Set, List, Dict, Any, Optional # 타입 힌팅
import numpy as np # 수치 연산 (임베딩 벡터 계산)
from openai import OpenAI # OpneAI API 클라이언트

from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output
import requests

import re # 정규표현식 (텍스트 정규화)

import shutil # 디렉터리 삭제용
from fastapi import status # HTTP 상태 코드 상수

# transcribe_to_json_with_progress: 오디오 파일 -> 텍스트 변환 (음성인식)
# GPU 서버에서 Whisper STT 수행
from .whisper_runner import transcribe_to_json_with_progress

# get_embeddings_from_gpu: 텍스트 -> 숫자 벡터 변환 (임베딩)
# GPU 서버에서 텍스트 임베딩 생성
from .embedding_runner import get_embeddings_from_gpu

import httpx
from typing import List, Dict
# ocr_pdf_with_gpu: PDF 이미지 -> 텍스트 추출 (OCR)
# GPU 서버에서 Qwen-VL OCR 수행
from .ocr_runner import ocr_pdf_with_gpu

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="AI Agent for Professors - MVP Backend")

# 정적 파일 제공: /files 경로로 BASE_DIR 폴더 내용 서빙
app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")

# OpenAI api key 키를 환경변수에서 가져와 클라인트 초기화
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# CORS 설정: 모든 출처에서의 요청 허용 (개발 편의성)
# 프로덕션에서는 allow_origins를 특정 도메인으로 제한해야 함
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 도메인 허용
    allow_credentials=False, # 쿠키 전송 비활성화
    allow_methods=["*"], # 모든 HTTP 메서드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 컨텍스트 매니저"""
    # 시작 시: DB 초기화
    init_db()
    print("Database initialized.")
    
    yield # 앱 실행 중

@app.get("/health")
def health():
    """서버 상태 확인용 엔드포인트
    - 로드밸런서나 모니터링 도구에서 사용
    """
    return {"ok": True}


@app.post("/lectures")
def create_lecture(payload: dict, session: Session = Depends(get_session)):
    """
    새 강의 생성
    
    설계 의도: 
    1. 제목(title)은 필수, 설명(description)은 선택
    2. Lecture 레코드 생성 후 연결된 Asset 레코드도 자동 생성
    3. Asset은 PDF/오디오 경로를 저장하는 용도
    """
    # 1. 요청 데이터에서 title과 description 추출
    title = (payload.get("title") or "").strip() # strip(): 앞뒤 공백 제거
    description = (payload.get("description") or "").strip()

    # 2. 제목 검증: 제목이 비어있으면 에러 발생
    if not title:
        raise HTTPException(status_code=400, detail="title은 필수입니다.") # HTTPExceiption: 사용자에게 에러 메시지 보내기

    # 3. Lecture 레코드 생성 (아직 DB에는 저장 안 함)
    lec = Lecture(title=title, description=description)
    
    # 4. DB에 추가(staged 상태)
    session.add(lec)
    
    # 5. DB에 실제로 저장(INSERT 쿼리 실행)->이때 자동으로 ID가 생성됨(auto increment)
    session.commit() 
    
    # 6. DB에서 최신 정보 다시 가져옴.(DB가 자동 생성한 ID를 가져옴)
    session.refresh(lec) 

    # 7. 연결된 Assest 레코드 자동 생성
    # Assest: PDF 경로, 오디오 경로를 저장할 테이블
    # 처음엔 빈 상태로 만들고, 나중에 /upload 엔드포인트에서 경로 업데이트
    asset = Asset(lecture_id=lec.id)
    session.add(asset)
    session.commit()

    # 8. 결과 반환
    return {"lecture_id": lec.id, "title": lec.title, "description": lec.description}

# 엔드포인트: 강의 목록 조회
@app.get("/lectures")
def list_lectures(session: Session = Depends(get_session)):
    """
    모든 강의 목록 조회
    
     프로세스:
    1. Lecture 테이블에서 모든 레코드 조회
    2. 최신 강의가 위에 오도록 ID 내림차순 정렬
    3. 각 강의의 id, title, description만 추출해서 반환
    
    사용 예:
    - 프론트엔드 메인 페이지에서 강의 목록 표시
    - 사용자가 특정 강의를 선택할 수 있게
    
    요청 예시:
    GET /lectures
    """
    # 1. DB 쿼리 실행
    # Lecture 테이블 선택해서, ID 내림차순 정렬(최신강의가 위로)하고, 모든 결과를 리스트로 반환
    rows = session.exec(select(Lecture).order_by(Lecture.id.desc())).all()
    
    # 2. 결과 정리(딕셔너리로)
    return {
        "lectures": [
            {"id": r.id, "title": r.title, "description": r.description}
            for r in rows # rows의 각 강의(r)에 대해 반복
        ]
    }


@app.get("/lectures/{lecture_id}")
def get_lecture(lecture_id: int, session: Session = Depends(get_session)):
    """
    특정 강의의 상세 정보 조회
    
    설계 의도:
    1. lecture_id로 DB에서 강의 찾기
    2. 강의가 없으면 404 에러 반환
    3. 연결된 Asset(파일 정보)도 함께 조회
    4. 두 정보를 합쳐서 반환"""
    
    # session.get(모델, ID): Primary Key로 레코드 조회
    # SQL: SELECT * FROM lecture WHERE id = lecture_id
    lec = session.get(Lecture, lecture_id) # 1. DB에서 강의 찾기->반환: 강의 객체 또는 None
    
    # 2. 강의 존재 여부 확인
    # 강의가 없으면(lec이 None이면) 404 에러 발생
    if not lec:
        raise HTTPException(status_code=404, detail="Lecture not found")

    # 3. 연결된 Asset 찾기
    # select(Asset): Asset 테이블 선택
    # .where(조건): lecture_id가 일치하는 레코드만 필터링
    # .first(): 첫 번째 결과만 가져옴 (없으면 None)
    # SQL: SELECT * FROM asset WHERE lecture_id = lecture_id LIMIT 1
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    
    # 4. 결과 반환
    return {
        "lecture": lec.model_dump(), # Lecture 객체 → 딕셔너리 변환 (JSON 직렬화를 위해)
        "asset": asset.model_dump() if asset else None,# asset이 있으면 변환, 없으면 None
    # asset이 None인 경우: 강의는 만들었지만 아직 파일 업로드 전
    }


@app.post("/lectures/{lecture_id}/upload")
async def upload_assets(
    lecture_id: int,
    pdf: UploadFile = File(...), 
    audio: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """
    강의에 PDF와 오디오 파일 업로드
    
    설계 의도:
    1. lecture_id로 강의가 존재하는지 확인
    2. 파일들을 서버 디스크에 저장
    3. 저장된 경로를 DB의 Asset 테이블에 기록
    4. 나중에 이 경로로 파일 접근 가능
    
    요청 예시:
    POST /lectures/1/upload
    Content-Type: multipart/form-data
    pdf: [파일 바이너리]
    audio: [파일 바이너리]
    
    응답 예시:
    {
        "lecture_id": 1,
        "pdf_path": "data/lectures/1/slide.pdf",
        "audio_path": "data/lectures/1/audio.mp3"
    }
    """
    # 1. 강의 존재 여부 확인
    lec = session.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(status_code=404, detail="Lecture not found") # 404: 강의를 찾을 수 없음.

    # 2. 파일을 서버 디스크에 저장
    try:
        pdf_path, audio_path = await save_uploads(lecture_id, pdf, audio) 
        # await: 파일 저장은 시간이 걸리는 작업이므로 비동기 처리
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3. DB에서 Asset 레코드 찾기 
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    
    # 4. Asset이 없으면 새로 생성
    if not asset:
        asset = Asset(lecture_id=lecture_id)

    # 5. Asset에 파일 경로 저장
    # 아직 DB에는 반영 안 됨(메모리에만 존재)
    asset.pdf_path = pdf_path
    asset.audio_path = audio_path

    # 6. DB에 저장
    session.add(asset) # asset을 세션에 추가
    session.commit() # DB에 실제로 저장(UPDATE 쿼리 실행)
    session.refresh(asset) # DB에서 최신 상태 다시 가져오기

    # 7. 결과 반환
    return {
        "lecture_id": lecture_id,
        "pdf_path": asset.pdf_path, # 저장된 PDF 경로
        "audio_path": asset.audio_path, # 저장된 오디오 경로
    }
    
# transcribe 변경
@app.post("/lectures/{lecture_id}/transcribe")
def transcribe(lecture_id: int, session: Session = Depends(get_session)):
    """오디오 파일을 음성인식(STT)하여 텍스트로 변환
    
    설계 의도:
    1. 업로드된 오디오 파일 경로를 DB에서 조회
    2. GPU 서버에서 Whisper 모델 실행 (faster-whisper)
    3. 결과를 transcript.json 파일로 저장
    4. 결과를 DB의 TranscriptSegment 테이블에도 저장
    5. 각 세그먼트: [시작시간, 종료시간, 텍스트]
    
    """
    # 1. DB에서 Asset 조회 (오디오 파일 경로 가져오기)
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    
    # 2. Asset 존재 및 오디오 경로 확인
    if not asset or not asset.audio_path: # 파일 업로드 전이거나 오디오만 안 올리는 경우
        raise HTTPException(status_code=400, detail="오디오가 업로드되지 않았습니다.")

    # 3. 오디오 파일의 절대 경로 생성
    # 기본 디렉토리에 상대 경로를 추가하고 절대 경로로 변환.
    audio_abs = (Path(BASE_DIR) / asset.audio_path).resolve()
    
    # 4. 파일 실제 존재 여부 확인
    if not audio_abs.exists(): # 파일이 디스크에 실제로 없으면
        raise HTTPException(status_code=404, detail=f"오디오 파일을 찾을 수 없습니다: {asset.audio_path}")

    # 5. 기본 transcript 데이터 삭제 (재실행 시 중복 방지)
    old = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.lecture_id == lecture_id)
    ).all()
    
    # 6. 기존 데이터 하나씩 삭제
    for row in old:
        session.delete(row) # 세션에 삭제 예정 표시
    session.commit() # DB에 실제로 삭제 (DELETE 쿼리 실행)

    # 7. transcript.json 저장 위치 결정
    out_dir = _lecture_data_dir(lecture_id)        
    out_json = out_dir / "transcript.json"

    # 8. GPU 서버에서 Whisper STT 실행
    # transcribe_to_json_with_progress(): whisper_runner.py에 정의된 함수
    # 역할:
    # - 오디오 파일을 GPU 서버로 전송
    # - faster-whisper 모델 실행 (large-v3)
    # - 결과를 JSON 파일로 저장
    # - 세그먼트 리스트 반환
    try:
        payload = transcribe_to_json_with_progress(
            audio_path=audio_abs,
            out_json_path=out_json,
            lecture_id=lecture_id,
            model_name="large-v3",   # Whisper 모델 크기(정확도 가장 높음)
            device="cuda",           # GPU 사용
            compute_type="float16",  # 연산 정밀도 (float 16: 빠르고 메모리 적게 씀)
        )
    except Exception as e: # STT 중 오류 발생(예: GPU 메모리 부족, 오디오 형식 오류)
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}") # 서버 문제

    # 9. 결과를 DB에 저장
    # payload에서 segments 추출(없으면 빈 리스트)
    segments = payload.get("segments", [])
    
    # 10. 각 세그먼트를 DB에 추가
    for s in segments:
        # TranscriptSegment 객체 생성
        session.add(
            TranscriptSegment(
                lecture_id=lecture_id, # 어느 강의의 자막인지
                start=float(s["start"]), # 시작 시간 (초 단위)
                end=float(s["end"]), # 종료 시간 (초 단위)
                text=str(s["text"]), # 인식된 텍스트
            )
        )
    # 11. DB에 실제로 저장
    session.commit() # 세그먼트 리스트를 DB에 추가 (INSERT 쿼리 실행)

    # 12. 결과 반환
    return {
        "ok": True, # 성공 여부
        "lecture_id": lecture_id,
        "segments_count": len(segments), # 총 세그먼트 개수
        "transcript_json": str(out_json), # JSON 파일 경로
    }

@app.get("/lectures/{lecture_id}/transcript")
def get_transcript(lecture_id: int, session: Session = Depends(get_session)):
    """
    특정 강의의 음성인식(STT) 결과 조회
    
    설계 의도:
    1. DB에 저장된 TranscriptSegment 레코드들을 조회
    2. 시작 시간 순서대로 정렬
    3. 각 세그먼트의 [시작시간, 종료시간, 텍스트]를 반환
    
    사용 예:
    - 프론트엔드에서의 자막 표시
    - 특정 시간대 자막 검색
    - 전체 스크립트 확인
    """
    # 1. DB에서 자막 세그먼트 조회
    rows = session.exec(
        select(TranscriptSegment)
        .where(TranscriptSegment.lecture_id == lecture_id)
        .order_by(TranscriptSegment.start)
    ).all() # 모든 결과를 리스트로 변환

    # 2. 결과를 json 형식으로 변환
    return {
        "lecture_id": lecture_id,
        "segments": [{"start": r.start, "end": r.end, "text": r.text} for r in rows],
    }


@app.get("/lectures/{lecture_id}/anchors")
def get_anchors(lecture_id: int, session: Session = Depends(get_session)):
    """
    특정 강의의 페이지-시간 동기화 앵커 포인트 조회
    
    설계 의도:
    1. PDF 페이지 번호와 오디오 시간을 매칭한 앵커 포인트 조회
    2. 페이지 번호 순서대로 정렬
    3. 프론트엔드에서 "3페이지는 45초부터 시작" 같은 정보 활용
    
    앵커란?
    - 특정 DPF 페이지가 오디오의 몇 초부터 시작하는지 표시하는 기준점
    - 예: {"page": 3, "time": 45.0}->3페이지는 45초부터
    - 프론트엔드에서 PDF와 오디오를 동기화하는 데 사용
    
    사용 예:
    - 사용자가 PDF 3페이지 클릭 시 오디오 45초로 이동
    - 오디오 50초 재생 중일때, PDF 3페이지 하이라이트
    - 슬라이드 넘어갈 때마다 오디오 시간 자동 매칭
    """
    # 1. DB에서 앵커 포인트 조회
    rows = session.exec(
        select(PageAnchor)
        .where(PageAnchor.lecture_id == lecture_id)
        .order_by(PageAnchor.page)
    ).all()
    # 2. 결과를 JSON 형식으로 변환
    return {"lecture_id": lecture_id, "anchors": [{"page": r.page, "time": r.time} for r in rows]}


@app.post("/lectures/{lecture_id}/anchors")
def upsert_anchor(lecture_id: int, payload: dict, session: Session = Depends(get_session)):
    """
    페이지-시간 앵커 추가 또는 수정(Upsert)
    
    설계 의도:
    1. 특정 페이지의 시작 시간을 설정
    2. 이미 앵커가 있으면 업데이트(Update)
    3. 앵커가 없으면 새로 생성 (Insert)
    4. Upsert = Update + Insert
    
    사용 예:
    - 교수님이 수동으로 "3페이지는 45초부터"라고 설정
    - 자동 동기화 결과를 수정
    - 잘못된 앵커 시간 보정"""
    
    # 1. 요청 데이터 추출 및 타입 변환
    page = int(payload.get("page") or 0)
    time = float(payload.get("time") or 0)

    # 2. 입력 값 검증(Validation)
    if page < 1:
        raise HTTPException(status_code=400, detail="page는 1 이상이어야 합니다.")
    if time < 0:
        raise HTTPException(status_code=400, detail="time은 0 이상이어야 합니다.")

    # 3. 기존 앵커 존재 여부 확인
    row = session.exec(
        select(PageAnchor).where(PageAnchor.lecture_id == lecture_id, PageAnchor.page == page)
    ).first()

    # 4. Upsert 로직(Update 또는 Insert)
    if row:
        # update: 기존 앵커의 시간만 수정
        row.time = time # 기존 앵커 객체의 time 속성을 새로 받은 시간 값으로 변경
    else: # 기존 앵커가 없으면
        # insert: 새 앵커 생성
        row = PageAnchor(lecture_id=lecture_id, page=page, time=time)
        session.add(row) # 세션에 추가

    # 5. DB에 실제로 저장
    session.commit() 
    # 6. 결과 반환
    return {"ok": True, "lecture_id": lecture_id, "page": page, "time": time}

def page_to_time_with_anchors(
    page: int, # 시간을 알고 싶은 페이지 번호
    num_pages: int, # PDF 총 페이지 수
    duration: float, # 오디오 총 길이 (초)
    anchors: list[tuple[int, float]] # 앵커 리스트 [(페이지, 시간),...]
    ) -> float: # 반환: 해당 페이지의 시작 시간 (초)
    """
    앵커 기반 페이지 시작 시간 계산 (보간법 사용)
    
    설계 의도:
    1. 앵커가 있는 페이지는 정확한 시간 반환
    2. 앵커가 없느 페이지는 주변 앵커를 기준으로 선형 보간
    3. 앵커가 전혀 없으면 균등 분할
    
    보간법(Interpolation):
    - 알려진 두 점 사이의 값을 추정하는 방법
    - 예: 페이지 3은 30초, 페이지 7은 90초
    - 페이지 5는? (30+90)/2 = 60초 (선형 보간)"""
    
    # 1. 첫 페이지 특별 처리
    if page <= 1:
        # 1페이지에 앵커가 있는지 확인
        for p, t in anchors:
            if p == 1:
                return t # 1페이지 앵커의 시간 반환
        # 1페이지 앵커가 없으면 0초 반환(기본값)
        return 0.0

    # 2. 입력 검증
    if num_pages <= 0 or duration <= 0:
        return 0.0

    # 3. 앵커가 없는 경우(균등 분할)
    if not anchors:
        return (page - 1) / max(1, num_pages) * duration

    # 4. 앵커 정렬
    anchors = sorted(anchors, key=lambda x: x[0])
    
    # 5. 정확한 앵커가 있는지 확인
    for p, t in anchors:
        if p == page:
            return t
        
    # 6. 정확한 앵커가 없는 경우(보간법 사용)
    # 좌우 앵커 찾기
    left = None
    right = None
    for p, t in anchors:
        if p < page:
            left = (p, t)
        elif p > page and right is None:
            right = (p, t)
            break

    # 7. 케이스별 보간
    if left and right:
        lp, lt = left
        rp, rt = right
        ratio = (page - lp) / (rp - lp)
        return lt + ratio * (rt - lt)

    if left and not right:
        lp, lt = left
        ratio = (page - lp) / max(1, (num_pages - lp))
        return lt + ratio * (duration - lt)

    if right and not left:
        rp, rt = right
        ratio = (page - 1) / max(1, (rp - 1))
        return 0.0 + ratio * (rt - 0.0)

    # 8. 예외 케이스(도달하면 안됨)
    # 모든 케이스를 다 처리했는데도 여기 온 경우
    return 0.0


@app.get("/lectures/{lecture_id}/page_time")
def get_page_time(
    lecture_id: int,
    page: int,
    num_pages: int,
    duration: float,
    session: Session = Depends(get_session),
):
    """
    특정 페이지의 시작 시간 계산(앵커 + 보간법)
    
    설계 의도:
    1. DB에서 저장된 앵커 포인트들을 조회
    2. page_to_time_with_anchors 함수로 시간 계산
    3. 앵커가 있으면 정확한 시간, 없으면 보간으로 추정"""
    rows = session.exec(
        select(PageAnchor).where(PageAnchor.lecture_id == lecture_id).order_by(PageAnchor.page)
    ).all()
    anchors = [(r.page, r.time) for r in rows]
    t = page_to_time_with_anchors(page, num_pages, duration, anchors)
    return {"lecture_id": lecture_id, "page": page, "time": t, "anchors": anchors}


# =============================================================================
# 유틸 함수
# =============================================================================

def _lecture_data_dir(lecture_id: int) -> Path:
    """
    특정 강의의 데이터 저장 디렉토리 생성 및 반환
    
    목적:
    - 강의별로 별도의 폴더 생성(격리)
    - JSON 파일, 임시 파일 등을 저장할 공간 제공
    
    사용 예:
    - transcript.json 저장
    - pages.json 저장
    - index.json 저장
    - summary.txt 저장
    """
    d = (Path(BASE_DIR) / "lectures" / str(lecture_id)).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_json(path: Path, obj: Any) -> None:
    """Python 객체를 JSON 파일로 저장
    
    목적:
    - 딕셔너리, 리스트 등을 JSON 형식으로 파일 저장
    - 한글 깨짐 방지
    - 읽기 쉬운 포맷
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_json(path: Path) -> Any:
    """JSON 파일을 읽어서 Python 객체로 반환
    
    목적:
    - 저장된 JSON 파일을 다시 메모리로 로드
    - 딕셔너리, 리스트 등으로 변환"""
    return json.loads(path.read_text(encoding="utf-8"))

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """ 긴 텍스트를 작은 청크(덩어리)로 분할
    
    목적:
    - 임베딩 모델의 토큰 제한 대응
    - 문맥 유지를 위해 청크 간 일부 겹침
    - RAG 인덱싱 시 검색 단위 생성"""
    text = (text or "").strip()
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        ch = text[i:end].strip()
        if ch:
            out.append(ch)
        if end == n:
            break
        i = max(0, end - overlap)
    return out

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """ 두 벡터 간의 코사인 유사도 계산
    
    목적:
    - 임베딩 벡터 간 유사도 측정
    - 값 범위: -1(완전 반대)~0(무관)~1(완전 동일)"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _norm_text(s: str) -> str:
    """ 텍스트 정규화
    목적:
    - 키워드 매칭의 정확도 향상
    - 표기 차이 무시(대소문자, 공백, 특수문자)"""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9a-z가-힣 ]+", "", s)
    return s

# =============================================================================
# [수정됨] OCR: PDF -> GPU 서버로 전송 -> pages.json 저장
# =============================================================================

# ========== 설정 ==========
# GPU_OCR_SERVER = "http://localhost:8003"  # GPU 서버 주소 (필요시 수정)


# # ========== OCR 함수 (Qwen-VL) ==========
# async def ocr_pdf_with_qwen(pdf_path: str) -> List[Dict]:
#     """
#     GPU 서버의 Qwen-VL을 사용해 PDF OCR 수행
#     """
#     async with httpx.AsyncClient(timeout=600.0) as client:  # 타임아웃 늘림
#         with open(pdf_path, "rb") as f:
#             files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            
#             response = await client.post(
#                 f"{GPU_OCR_SERVER}/ocr/pdf",
#                 files=files,
#                 data={"dpi": 200, "max_tokens": 2048}
#             )
        
#         if response.status_code != 200:
#             raise Exception(f"OCR 서버 오류: {response.text}")
        
#         result = response.json()
#         return result["pages"]


# # ========== OCR 엔드포인트 ==========
# @app.post("/lectures/{lecture_id}/ocr_pdf")
# async def ocr_pdf(lecture_id: int, session: Session = Depends(get_session)):
#     """PDF OCR 수행 (Qwen-VL 사용)"""
    
#     # Asset에서 PDF 경로 가져오기
#     asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
#     if not asset or not asset.pdf_path:
#         raise HTTPException(status_code=404, detail="PDF가 업로드되지 않았습니다.")
    
#     # PDF 절대 경로
#     pdf_abs = (Path(BASE_DIR) / asset.pdf_path).resolve()
#     if not pdf_abs.exists():
#         raise HTTPException(status_code=404, detail=f"PDF 파일을 찾을 수 없습니다: {asset.pdf_path}")
    
#     try:
#         # GPU 서버에 OCR 요청
#         pages_result = await ocr_pdf_with_qwen(str(pdf_abs))
        
#         # 결과를 pages.json 형태로 변환
#         pages_info = []
#         for page in pages_result:
#             pages_info.append({
#                 "page": page["page"],
#                 "text": page["text"],
#             })
        
#         # pages.json 파일로 저장
#         out_dir = _lecture_data_dir(lecture_id)
#         pages_obj = {
#             "lecture_id": lecture_id,
#             "num_pages": len(pages_info),
#             "pages": pages_info,
#         }
#         _save_json(out_dir / "pages.json", pages_obj)
        
#         return {
#             "status": "success",
#             "lecture_id": lecture_id,
#             "total_pages": len(pages_info),
#             "pages": pages_info,
#         }
    
#     except httpx.ConnectError:
#         raise HTTPException(
#             status_code=503, 
#             detail="GPU OCR 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
#         )
#     except httpx.ReadTimeout:
#         raise HTTPException(
#             status_code=504,
#             detail="OCR 처리 시간이 초과되었습니다. 페이지 수가 많으면 시간이 더 걸릴 수 있습니다."
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# def ocr_pdf_with_qwen_sync(pdf_path: str) -> List[Dict]:
#     """동기 버전 OCR 함수"""
#     with open(pdf_path, "rb") as f:
#         files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
#         response = requests.post(
#             f"{GPU_OCR_SERVER}/ocr/pdf",
#             files=files,
#             data={"dpi": 200, "max_tokens": 2048},
#             timeout=600
#         )
    
#     if response.status_code != 200:
#         raise Exception(f"OCR 서버 오류: {response.text}")
    
#     return response.json()["pages"]
"""
main.py OCR 부분 수정

변경 전: httpx로 GPU 서버 API 호출
변경 후: SSH로 GPU 서버 스크립트 호출 (embedding과 동일한 방식)
"""

# ========== 엔드포인트 수정 (async → sync) ==========
@app.post("/lectures/{lecture_id}/ocr_pdf")
def ocr_pdf(lecture_id: int, session: Session = Depends(get_session)):
    """PDF OCR 수행 (Qwen-VL, SSH 호출)"""
    
    # Asset에서 PDF 경로 가져오기
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    if not asset or not asset.pdf_path:
        raise HTTPException(status_code=404, detail="PDF가 업로드되지 않았습니다.")
    
    # PDF 절대 경로
    pdf_abs = (Path(BASE_DIR) / asset.pdf_path).resolve()
    if not pdf_abs.exists():
        raise HTTPException(status_code=404, detail=f"PDF 파일을 찾을 수 없습니다: {asset.pdf_path}")
    
    try:
        # GPU 서버에서 OCR 실행 (SSH 호출)
        ocr_result = ocr_pdf_with_gpu(str(pdf_abs), dpi=200)
        
        # pages.json 파일로 저장
        out_dir = _lecture_data_dir(lecture_id)
        pages_obj = {
            "lecture_id": lecture_id,
            "num_pages": ocr_result["num_pages"],
            "pages": ocr_result["pages"],
        }
        _save_json(out_dir / "pages.json", pages_obj)
        
        return {
            "status": "success",
            "lecture_id": lecture_id,
            "total_pages": ocr_result["num_pages"],
            "pages": ocr_result["pages"],
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 실패: {str(e)}")


# =============================================================================
# RAG Index
# =============================================================================

@app.post("/lectures/{lecture_id}/rag_index")
def rag_index(lecture_id: int, session: Session = Depends(get_session)):
    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"
    if not pages_json_path.exists():
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")

    obj = _load_json(pages_json_path)
    pages = obj.get("pages", [])

    records = []
    for p in pages:
        page_no = int(p.get("page"))
        chunks = _chunk_text(p.get("text", ""))
        for ci, ch in enumerate(chunks):
            records.append({
                "lecture_id": lecture_id,
                "page": page_no,
                "chunkId": f"p{page_no}-c{ci}",
                "text": ch,
            })

    if not records:
        raise HTTPException(status_code=400, detail="OCR 텍스트가 비어 있어 인덱스를 만들 수 없습니다.")

    # PDF OCR 텍스트를 임베딩
    try:
        vectors = get_embeddings_from_gpu([r["text"] for r in records])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding 실패: {e}")
    
    index_obj = {
        "lecture_id": lecture_id,
        "embedding_model": "text-embedding-3-small",
        "items": [{**records[i], "embedding": vectors[i]} for i in range(len(records))],
    }

    index_json_path = out_dir / "index.json"
    _save_json(index_json_path, index_obj)

    return {"ok": True, "lecture_id": lecture_id, "chunks": len(records), "index_json": str(index_json_path)}


# =============================================================================
# RAG Ask
# =============================================================================

@app.post("/lectures/{lecture_id}/rag_ask")
def rag_ask(
    lecture_id: int,
    question: str = Body(..., embed=True),
    topK: int = Body(5, embed=True),
):
    out_dir = _lecture_data_dir(lecture_id)
    index_json_path = out_dir / "index.json"
    if not index_json_path.exists():
        raise HTTPException(status_code=404, detail="index.json이 없습니다. 먼저 /rag_index를 실행하세요.")

    index_obj = _load_json(index_json_path)
    items = index_obj.get("items", [])
    if not items:
        raise HTTPException(status_code=400, detail="index items가 비어 있습니다.")

    try:
            qemb = get_embeddings_from_gpu([question])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question embedding 실패: {e}")

    qv = np.array(qemb, dtype=np.float32)

    scored = []
    for it in items:
        v = np.array(it["embedding"], dtype=np.float32)
        s = _cosine_sim(qv, v)
        scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)

    k = max(1, min(int(topK), 15))
    top = scored[:k]

    citations = []
    ctx_blocks = []
    for rank, (score, it) in enumerate(top, start=1):
        snippet = it["text"][:280].replace("\n", " ").strip()
        citations.append({
            "page": it["page"],
            "chunkId": it["chunkId"],
            "score": float(score),
            "snippet": snippet,
        })
        ctx_blocks.append(f"[{rank}] (page {it['page']}) {it['text']}")

    context = "\n\n".join(ctx_blocks)
    prompt = f"""너는 강의자료(PDF OCR) 기반 Q&A 조교야.
아래 CONTEXT에 있는 내용만 근거로 답해.
모르면 "자료에서 확인되지 않습니다"라고 말해.
마지막에 참고 페이지를 bullet로 적어.

QUESTION:
{question}

CONTEXT:
{context}
"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        answer = resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 답변 생성 실패: {e}")

    return {"lecture_id": lecture_id, "answer": answer, "citations": citations}


# =============================================================================
# Summary (GET/POST)
# =============================================================================

@app.get("/lectures/{lecture_id}/summary")
def get_summary(lecture_id: int):
    out_dir = _lecture_data_dir(lecture_id)
    p = out_dir / "summary.txt"
    if not p.exists():
        return {"lecture_id": lecture_id, "summary": ""}
    return {"lecture_id": lecture_id, "summary": p.read_text(encoding="utf-8")}


@app.post("/lectures/{lecture_id}/summary")
def make_summary(lecture_id: int):
    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"
    if not pages_json_path.exists():
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")

    obj = _load_json(pages_json_path)
    pages = obj.get("pages", [])
    joined = "\n\n".join([f"[p{p.get('page')}]\n{p.get('text','')}" for p in pages])[:12000]

    prompt = f"""너는 강의 조교야. 아래 OCR 텍스트만 근거로 요약을 작성해.
모르면 추측하지 말고 "자료에서 확인되지 않습니다"라고 말해.

형식:
## 📚 강의 주제 및 목표
(2~3줄)

## 🎯 핵심 개념
1) ...
2) ...
3) ...

## 📖 주요 내용
(3~4문단)

## 💡 학습 포인트
- ...
- ...
- ...

OCR:
{joined}
"""

    try:
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        summary = resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 생성 실패: {e}")

    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return {"lecture_id": lecture_id, "summary": summary}


# =============================================================================
# Quiz (GET/POST)
# =============================================================================

@app.get("/lectures/{lecture_id}/quiz")
def get_quiz(lecture_id: int):
    out_dir = _lecture_data_dir(lecture_id)
    p = out_dir / "quiz.json"
    if not p.exists():
        return {"lecture_id": lecture_id, "quiz": []}
    return {"lecture_id": lecture_id, "quiz": _load_json(p)}


@app.post("/lectures/{lecture_id}/quiz")
def make_quiz(lecture_id: int):
    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"
    if not pages_json_path.exists():
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")

    obj = _load_json(pages_json_path)
    pages = obj.get("pages", [])
    joined = "\n\n".join([p.get("text", "") for p in pages])[:12000]

    prompt = f"""다음 OCR 내용을 바탕으로 5개 객관식 퀴즈를 만들어.
반드시 "JSON 배열"만 출력해. 다른 텍스트/코드블록 금지.

[{{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}}]
OCR:
{joined}
"""

    try:
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        raw = (resp.output_text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 실패: {e}")

    try:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        quiz = json.loads(cleaned)
        if not isinstance(quiz, list):
            quiz = []
    except Exception:
        return {"lecture_id": lecture_id, "quiz": [], "raw": raw}

    _save_json(out_dir / "quiz.json", quiz)
    return {"lecture_id": lecture_id, "quiz": quiz}


# =============================================================================
# Chat
# =============================================================================

def _chat_path(lecture_id: int) -> Path:
    return _lecture_data_dir(lecture_id) / "chat.json"

def _append_chat(lecture_id: int, role: str, content: str) -> None:
    p = _chat_path(lecture_id)
    arr = []
    if p.exists():
        arr = _load_json(p)
        if not isinstance(arr, list):
            arr = []
    arr.append({"role": role, "content": content})
    _save_json(p, arr)

@app.get("/lectures/{lecture_id}/chat")
def get_chat(lecture_id: int):
    p = _chat_path(lecture_id)
    if not p.exists():
        return {"lecture_id": lecture_id, "chat": []}
    return {"lecture_id": lecture_id, "chat": _load_json(p)}


# =============================================================================
# PDF Info
# =============================================================================

from PyPDF2 import PdfReader

@app.get("/lectures/{lecture_id}/pdf_info")
def get_pdf_info(lecture_id: int, session: Session = Depends(get_session)):
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    if not asset or not asset.pdf_path:
        raise HTTPException(status_code=404, detail="PDF not uploaded")

    pdf_abs = (Path(BASE_DIR) / asset.pdf_path).resolve()
    if not pdf_abs.exists():
        raise HTTPException(status_code=404, detail=f"PDF file not found: {asset.pdf_path}")

    try:
        with open(pdf_abs, "rb") as f:
            reader = PdfReader(f)
            num_pages = len(reader.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")

    return {"lecture_id": lecture_id, "num_pages": num_pages, "pdf_path": asset.pdf_path}


# =============================================================================
# Delete APIs
# =============================================================================

def _lecture_upload_dir(lecture_id: int) -> Path:
    return (Path(BASE_DIR) / "data" / "lectures" / str(lecture_id)).resolve()

def _lecture_artifacts_dir(lecture_id: int) -> Path:
    return (Path(BASE_DIR) / "lectures" / str(lecture_id)).resolve()

def _safe_rmtree(p: Path) -> None:
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


@app.delete("/lectures/{lecture_id}", status_code=status.HTTP_200_OK)
def delete_lecture(lecture_id: int, session: Session = Depends(get_session)):
    lec = session.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(status_code=404, detail="Lecture not found")

    ts = session.exec(select(TranscriptSegment).where(TranscriptSegment.lecture_id == lecture_id)).all()
    for r in ts:
        session.delete(r)

    an = session.exec(select(PageAnchor).where(PageAnchor.lecture_id == lecture_id)).all()
    for r in an:
        session.delete(r)

    assets = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).all()
    for r in assets:
        session.delete(r)

    session.delete(lec)
    session.commit()

    _safe_rmtree(_lecture_upload_dir(lecture_id))
    _safe_rmtree(_lecture_artifacts_dir(lecture_id))

    return {
        "ok": True,
        "deleted_lecture_id": lecture_id,
        "deleted_dirs": [
            str(_lecture_upload_dir(lecture_id)),
            str(_lecture_artifacts_dir(lecture_id)),
        ],
    }


@app.delete("/lectures", status_code=status.HTTP_200_OK)
def delete_all_lectures(session: Session = Depends(get_session)):
    ids = session.exec(select(Lecture.id)).all()
    ids = [int(x) for x in ids]

    for r in session.exec(select(TranscriptSegment)).all():
        session.delete(r)
    for r in session.exec(select(PageAnchor)).all():
        session.delete(r)
    for r in session.exec(select(Asset)).all():
        session.delete(r)
    for r in session.exec(select(Lecture)).all():
        session.delete(r)

    session.commit()

    _safe_rmtree((Path(BASE_DIR) / "data" / "lectures").resolve())
    _safe_rmtree((Path(BASE_DIR) / "lectures").resolve())

    return {
        "ok": True,
        "deleted_count": len(ids),
        "deleted_ids": ids,
        "deleted_dirs": [
            str((Path(BASE_DIR) / "data" / "lectures").resolve()),
            str((Path(BASE_DIR) / "lectures").resolve()),
        ],
    }


# =============================================================================
# Pages Info (OCR 페이지 정보)
# =============================================================================

@app.get("/lectures/{lecture_id}/pages_info")
def get_pages_info(lecture_id: int):
    """
    OCR 결과에서 페이지 수와 기본 정보를 반환합니다.
    pages.json이 없으면 빈 응답 반환 (에러 대신)
    """
    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"
    
    if not pages_json_path.exists():
        return {"lecture_id": lecture_id, "num_pages": 0, "pages": []}
    
    pages_obj = _load_json(pages_json_path)
    pages = pages_obj.get("pages", [])
    
    pages_summary = []
    for p in pages:
        text = (p.get("text") or "").strip()
        text = (p.get("text") or "").strip().lower()
        tokens = re.findall(r"[0-9a-z가-힣]{3,}", text)
        word_freq = {}
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1

        
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [k[0] for k in keywords]
        
        first_sentence = text.split(".")[0].strip() if text else ""
        title = first_sentence[:50] if len(first_sentence) > 5 else f"페이지 {p.get('page', 0)}"
        
        pages_summary.append({
            "page": p.get("page"),
            "title": title,
            "keywords": keywords,
            "text_length": len(text),
        })
    
    return {
        "lecture_id": lecture_id,
        "num_pages": len(pages),
        "pages": pages_summary,
    }


# =============================================================================
# Auto Sync (임베딩 기반 페이지-오디오 매칭)
# =============================================================================
@app.post("/lectures/{lecture_id}/auto_sync")
def auto_sync(lecture_id: int, session: Session = Depends(get_session)):
    """
    [최종] 임베딩(Embedding) + 키워드 스포팅(Keyword Spotting) + 신뢰도 기반 보간법(DWT) 적용
    """
    # 1. 내부 유틸 함수 정의
    def clean_text_local(s:str) -> str:
        import re
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^0-9a-z가-힣 ]+", "", s)
        return s
    
    def get_keywords_local(text: str) -> set:
        normalized = clean_text_local(text)
        return set([w for w in normalized.split() if len(w) >= 2])
    
    # 2. 데이터 로드 및 전처리
    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"
    if not pages_json_path.exists():
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")
    
    pages_obj = _load_json(pages_json_path)
    pages = pages_obj.get("pages", [])
    if not pages:
        raise HTTPException(status_code=400, detail="OCR 페이지가 비어 있습니다.")
    
    transcript_rows = session.exec(
        select(TranscriptSegment)
        .where(TranscriptSegment.lecture_id == lecture_id)
        .order_by(TranscriptSegment.start)
    ).all()
    
    if not transcript_rows:
        raise HTTPException(status_code=400, detail="Transcript가 없습니다. 먼저 /transcribe를 실행하세요.")
    
    # 페이지 텍스트 준비
    page_texts = []
    for p in pages:
        text = (p.get("text") or "").strip()
        if len(text) < 50 and p.get("words"):
            text = " ".join([w.get("t", "") for w in p.get("words", [])])
        text = text.strip()
        if not text or len(text) < 5:
            text = f"페이지 {p.get('page', 0)} 내용"
        page_texts.append(text[:2000])
    
    # 자막 그룹화 (10초 단위)
    segment_groups = []
    current_group = {"start": 0, "end": 0, "texts": []}
    group_duration = 10 
    
    for seg in transcript_rows:
        seg_text = (seg.text or "").strip()
        if not seg_text: continue
            
        if not current_group["texts"]:
            current_group["start"] = seg.start
        
        current_group["texts"].append(seg_text)
        current_group["end"] = seg.end
        
        if seg.end - current_group["start"] >= group_duration:
            group_text = " ".join(current_group["texts"]).strip()
            if group_text:
                segment_groups.append({
                    "start": current_group["start"],
                    "end": current_group["end"],
                    "text": group_text
                })
            current_group = {"start": 0, "end": 0, "texts": []}
    
    if current_group["texts"]:
        group_text = " ".join(current_group["texts"]).strip()
        if group_text:
            segment_groups.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "text": group_text
            })
    
    if not segment_groups:
        raise HTTPException(status_code=400, detail="자막 그룹을 생성할 수 없습니다.")
    
    # 3. 임베딩 생성
    all_texts = []
    for t in page_texts:
        clean_t = t.strip()
        all_texts.append(clean_t if clean_t else "빈 페이지")
    
    for g in segment_groups:
        clean_t = g["text"][:2000].strip()
        all_texts.append(clean_t if clean_t else "빈 구간")
    
    all_texts = [t if t.strip() else "placeholder" for t in all_texts]
    
    try:
        all_embeddings = get_embeddings_from_gpu(all_texts) # 페이지 글이랑 자막 글을 숫자로 바꾼다.
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding 생성 실패: {e}")

    num_pages = len(page_texts)
    page_embeddings = all_embeddings[:num_pages]
    segment_embeddings = all_embeddings[num_pages:]
    
    # 4. 유사도 매트릭스 계산
    page_vecs = np.array(page_embeddings, dtype=np.float32)
    seg_vecs = np.array(segment_embeddings, dtype=np.float32)
    
    page_norms = np.linalg.norm(page_vecs, axis=1, keepdims=True)
    seg_norms = np.linalg.norm(seg_vecs, axis=1, keepdims=True)
    
    page_vecs_norm = page_vecs / np.where(page_norms > 0, page_norms, 1)
    seg_vecs_norm = seg_vecs / np.where(seg_norms > 0, seg_norms, 1)
    
    similarity_matrix = np.dot(page_vecs_norm, seg_vecs_norm.T) # 유사도 행렬 만드는 코드: 페이지랑 자막이 얼마나 비슷한지 점수표를 만든다.
    
    # 5. Keyword Spotting (키워드 가산점)
    page_keywords = [get_keywords_local(t) for t in page_texts]
    segment_keywords = [get_keywords_local(g["text"]) for g in segment_groups]
    
    num_segments = len(segment_groups)
    keyword_matrix = np.zeros((num_pages, num_segments))
    
    for i in range(num_pages):
        p_set = page_keywords[i]
        if not p_set: continue
        for j in range(num_segments):
            s_set = segment_keywords[j]
            if not s_set: continue
            common_words = p_set & s_set
            if common_words:
                score_boost = len(common_words) * 0.5
                keyword_matrix[i][j] = score_boost
                
    similarity_matrix += keyword_matrix
    
    # 6. DP 알고리즘 (최적 경로 탐색)
    dp = np.full((num_pages + 1, num_segments + 1), -np.inf)
    dp[0][0] = 0
    parent = {}
    
    for i in range(1, num_pages + 1): # 페이지 순서를 지키면서 가장 자연스러운 매칭을 고른다.
        for j in range(1, num_segments + 1):
            for prev_j in range(j):
                score = dp[i-1][prev_j] + similarity_matrix[i-1][j-1]
                if score > dp[i][j]:
                    dp[i][j] = score
                    parent[(i, j)] = (i-1, prev_j, j-1)
    
    best_j = np.argmax(dp[num_pages, 1:]) + 1
    
    path = []
    i, j = num_pages, best_j
    while i > 0 and (i, j) in parent:
        prev_i, prev_j, matched_seg = parent[(i, j)]
        path.append((i - 1, matched_seg))
        i, j = prev_i, prev_j
    
    path.reverse()
    
    # 7. [핵심] 신뢰도 기반 보간법 (Interpolation)
    reliable_anchors = []
    reliable_anchors.append({"page": 0, "time": 0.0}) # 시작점 고정
    
    for page_idx, seg_idx in path:
        score = similarity_matrix[page_idx][seg_idx]
        THRESHOLD = 1.3 # 임계값 설정
        
        if score >= THRESHOLD:
            time = segment_groups[seg_idx]["start"]
            reliable_anchors.append({"page": page_idx + 1, "time": time})
            
    # 마지막 페이지 처리
    last_transcript_time = transcript_rows[-1].end if transcript_rows else 0
    if reliable_anchors[-1]["page"] < num_pages:
        reliable_anchors.append({"page": num_pages + 1, "time": last_transcript_time})

    # 빈 구간 채우기
    final_anchors = []
    for k in range(len(reliable_anchors) - 1):
        curr_anchor = reliable_anchors[k]
        next_anchor = reliable_anchors[k+1]
        
        start_page = int(curr_anchor["page"])
        end_page = int(next_anchor["page"])
        start_time = curr_anchor["time"]
        end_time = next_anchor["time"]
        
        # 현재 앵커 추가
        if start_page > 0 and start_page <= num_pages:
             final_anchors.append({"page": start_page, "time": start_time})
        
        # 중간 페이지 보간
        page_gap = end_page - start_page
        if page_gap > 1:
            time_gap = end_time - start_time
            time_per_page = time_gap / page_gap
            for step in range(1, page_gap):
                interp_page = start_page + step
                interp_time = start_time + (time_per_page * step)
                if interp_page <= num_pages:
                    final_anchors.append({"page": interp_page, "time": interp_time})

    final_anchors.sort(key=lambda x: x["page"])
    
    # Sliding Window 미세 조정 (Fine-tuning)
    # 목표: 10초 단위로 뭉뚱그려진 시간을 원본 자막(3~5초) 단위로 정밀 보정
    
    refined_anchors = []
    
    for anchor in final_anchors:
        page_idx = anchor["page"] - 1 # 0-based index
        if page_idx < 0 or page_idx >= len(page_texts):
            refined_anchors.append(anchor)
            continue
            
        coarse_time = anchor["time"]
        
        # 1. 탐색 범위 설정 (현재 시간 ±15초)
        search_start = max(0, coarse_time - 15)
        search_end = coarse_time + 15
        
        # 2. 범위 내의 "원본 자막 세그먼트" 찾기
        candidates = [
            seg for seg in transcript_rows 
            if seg.end >= search_start and seg.start <= search_end
        ]
        
        if not candidates:
            refined_anchors.append(anchor)
            continue
            
        # 3. 페이지 키워드 가져오기
        p_keywords = page_keywords[page_idx]
        if not p_keywords:
            refined_anchors.append(anchor)
            continue
            
        # 4. 세그먼트별 매칭 점수 계산 (Sliding)
        best_time = coarse_time
        max_score = 0
        found_better = False
        
        for seg in candidates:
            # 자막 세그먼트에서 키워드 추출
            s_keywords = get_keywords_local(seg.text)
            
            # 교집합 개수 확인
            match_count = len(p_keywords & s_keywords)
            
            # 키워드가 발견되면 그 세그먼트의 시작 시간이 더 정확할 확률이 높음
            if match_count > 0:
                # 단순히 개수만 보는게 아니라, 원래 시간과 얼마나 가까운지도 가중치로 고려 가능
                # 여기서는 "키워드가 가장 많이 겹치는 가장 빠른 구간"을 선호하도록 로직 구성
                if match_count > max_score:
                    max_score = match_count
                    best_time = seg.start
                    found_better = True
                elif match_count == max_score and match_count > 0:
                    # 점수가 같다면, 원래 예측 시간(coarse_time)과 더 가까운 쪽 선택
                    if abs(seg.start - coarse_time) < abs(best_time - coarse_time):
                        best_time = seg.start
        
        # 5. 더 나은 시간이 발견되었고, 오차가 너무 크지 않다면 업데이트
        if found_better:
            refined_anchors.append({"page": anchor["page"], "time": best_time})
        else:
            refined_anchors.append(anchor)

    # 최종 결과를 refined_anchors로 교체
    final_anchors = refined_anchors
    
    # 미세 조정 완료
    
    # 8. 결과 저장
    old_anchors = session.exec(
        select(PageAnchor).where(PageAnchor.lecture_id == lecture_id)
    ).all()
    for anchor in old_anchors:
        session.delete(anchor)
    session.commit()
    
    anchors_created = []
    for item in final_anchors:
        anchor = PageAnchor(lecture_id=lecture_id, page=item["page"], time=float(item["time"]))
        session.add(anchor)
        anchors_created.append(item)
    
    session.commit()
    
    # 9. 디버그 및 리턴
    debug_info = {
        "lecture_id": lecture_id,
        "num_pages": num_pages,
        "num_segment_groups": num_segments,
        "matched_pairs": [(p+1, s, float(similarity_matrix[p][s])) for p, s in path],
        "anchors": anchors_created,
    }
    _save_json(out_dir / "sync_debug.json", debug_info)
    
    similarity_data = {
        "lecture_id": lecture_id,
        "num_pages": num_pages,
        "num_segments": num_segments,
        "segment_times": [{"start": sg["start"], "end": sg["end"]} for sg in segment_groups],
        "matrix": similarity_matrix.tolist(),
        "matched_path": [(p+1, s) for p, s in path],
    }
    _save_json(out_dir / "similarity_matrix.json", similarity_data)
    
    return {
        "ok": True,
        "lecture_id": lecture_id,
        "anchors_count": len(anchors_created),
        "anchors": anchors_created,
        "debug": {
            "num_pages": num_pages,
            "num_segment_groups": num_segments,
            "matched_pairs_count": len(path),
        }
    }


# =============================================================================
# Sync Debug
# =============================================================================

@app.get("/lectures/{lecture_id}/sync_debug")
def get_sync_debug(lecture_id: int):
    out_dir = _lecture_data_dir(lecture_id)
    debug_path = out_dir / "sync_debug.json"
    
    if not debug_path.exists():
        raise HTTPException(status_code=404, detail="sync_debug.json이 없습니다. 먼저 /auto_sync를 실행하세요.")
    
    return _load_json(debug_path)


# =============================================================================
# Similarity Matrix
# =============================================================================

@app.get("/lectures/{lecture_id}/similarity_matrix")
def get_similarity_matrix(lecture_id: int, session: Session = Depends(get_session)):
    out_dir = _lecture_data_dir(lecture_id)
    
    similarity_json_path = out_dir / "similarity_matrix.json"
    if similarity_json_path.exists():
        return _load_json(similarity_json_path)
    
    pages_json_path = out_dir / "pages.json"
    
    if not pages_json_path.exists():
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")
    
    pages_obj = _load_json(pages_json_path)
    pages = pages_obj.get("pages", [])
    
    transcript_rows = session.exec(
        select(TranscriptSegment)
        .where(TranscriptSegment.lecture_id == lecture_id)
        .order_by(TranscriptSegment.start)
    ).all()
    
    if not pages or not transcript_rows:
        raise HTTPException(status_code=400, detail="페이지 또는 자막 데이터가 없습니다.")
    
    def word_overlap(text1: str, text2: str) -> float:
        words1 = set(_norm_text(text1).split())
        words2 = set(_norm_text(text2).split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    segment_groups = []
    current_group = {"start": 0, "end": 0, "texts": []}
    
    for seg in transcript_rows:
        if not current_group["texts"]:
            current_group["start"] = seg.start
        current_group["texts"].append(seg.text)
        current_group["end"] = seg.end
        
        if seg.end - current_group["start"] >= 30:
            segment_groups.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "text": " ".join(current_group["texts"])
            })
            current_group = {"start": 0, "end": 0, "texts": []}
    
    if current_group["texts"]:
        segment_groups.append({
            "start": current_group["start"],
            "end": current_group["end"],
            "text": " ".join(current_group["texts"])
        })
    
    matrix = []
    for p in pages:
        row = []
        page_text = p.get("text", "")
        for sg in segment_groups:
            sim = word_overlap(page_text, sg["text"])
            row.append(round(sim, 3))
        matrix.append(row)
    
    return {
        "lecture_id": lecture_id,
        "num_pages": len(pages),
        "num_segments": len(segment_groups),
        "segment_times": [{"start": sg["start"], "end": sg["end"]} for sg in segment_groups],
        "matrix": matrix,
        "matched_path": [],
        "is_embedding_based": False,
    }
    
# [추가] 각 단계별 파일이 서버에 존재하는지 확인하여 미리 있는 파일이면, 불러오도록 하는
@app.get("/lectures/{lecture_id}/status")
def get_lecture_status(lecture_id: int):
    """
    각 단계별 파일이 서버에 존재하는지 확인하여 상태 반환
    """
    d = _lecture_data_dir(lecture_id)
    return {
        "transcript": (d / "transcript.json").exists(),
        "ocr": (d / "pages.json").exists(),
        "index": (d / "index.json").exists(),       # RAG Index 존재 여부
        "summary": (d / "summary.txt").exists(),    # 요약 존재 여부
        "quiz": (d / "quiz.json").exists(),         # 퀴즈 존재 여부
    }
    