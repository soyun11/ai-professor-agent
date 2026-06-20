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
from dotenv import load_dotenv 
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

load_dotenv()
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

from .sync_experiments import register_experiment_routes 

from .sync_algorithms.base import TextProcessor

register_experiment_routes(app, BASE_DIR) # 실험용 라우트 등록

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

"""
main.py OCR 부분 수정

변경 전: httpx로 GPU 서버 API 호출
변경 후: SSH로 GPU 서버 스크립트 호출 (embedding과 동일한 방식)
"""

# ========== 엔드포인트 수정 (async → sync) ==========
@app.post("/lectures/{lecture_id}/ocr_pdf")
def ocr_pdf(lecture_id: int, session: Session = Depends(get_session)):
    """PDF OCR 수행 (Qwen-VL, SSH 호출)
    
    1. PDF 파일이 업로드되었는지 확인하고 절대 경로 검증
    2. GPU 서버에 SSH로 OCR 작업 요청
    3. OCR 결과를 pages.json으로 저장 및 반환
    """
    
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
    """RAG 검색용 임베딩 인덱스 생성 
    
    1. pages.json에서 OCR 추출 텍스트 로드
    2. 텍스트를 청크 단위로 분할하여 검색 가능한 단위로 준비
    3. 각 청크를 임베딩(벡터)으로 변환 (GPU 서버 활용)
    4. 임베딩 벡터와 메타데이터를 index.json으로 저장 (검색 시 사용)
    """
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
        "embedding_model": "text-embedding-3-small",# OCR로 추출된 텍스트를 벡터로 변환하는 작업
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
    """
    RAG 기반 강의자료 Q&A
    
    1. index.json에서 임베딩된 청크 데이터 로드
    2. 사용자 질문을 임베딩으로 변환
    3. 코사인 유사도로 질문과 유사한 상위 k개 청크 검색
    4. 검색된 청크를 컨텍스트로 구성하여 LLM에 전달
    5. LLM이 컨텍스트 기반으로 답변 생성 및 인용 정보 반환
    """
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
    """
    PDF 파일 정보 조회
    
    1. 데이터베이스에서 해당 강의의 PDF 경로 조회
    2. PDF 파일의 존재 여부 검증
    3. PyPDF2로 PDF를 열어 총 페이지 수 계산
    4. 페이지 수와 경로 정보 반환

    Args:
        lecture_id (int): 조회할 강의의 고유 ID
        session (Session, optional): 데이터베이스 세션(의존성 주입). Defaults to Depends(get_session).

    Raises:
        HTTPException: 404 - 해당 강의의 PDF가 업로드되지 않았을 경우
        HTTPException: 404 - PDF 파일이 서버에 존재하지 않을 경우
        HTTPException: 500 - PDF 파일을 읽는 중 오류 발생

    Returns:
        dict:{
            "lecture_id": int, # 강의 ID
            "num_pages": int, # PDF 총 페이지 수
            "pdf_path": str # 상대 경로
        }
    """
    
    # Asset 테이블에서 lecture_id와 일치하는 레코드 조회
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    if not asset or not asset.pdf_path:
        raise HTTPException(status_code=404, detail="PDF not uploaded")

    # 상대 경로를 절대 경로로 변환
    pdf_abs = (Path(BASE_DIR) / asset.pdf_path).resolve()
    if not pdf_abs.exists():
        raise HTTPException(status_code=404, detail=f"PDF file not found: {asset.pdf_path}")

    try:
        # 바이너리 모드로 PDF 파일 열기
        with open(pdf_abs, "rb") as f:
            # PyPDF2 객체로 PDF 파싱
            reader = PdfReader(f)
            # 페이지 객체 리스트의 길이로 총 페이지 수 계산
            num_pages = len(reader.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")

    return {"lecture_id": lecture_id, "num_pages": num_pages, "pdf_path": asset.pdf_path}


# =============================================================================
# Delete APIs
# =============================================================================

def _lecture_upload_dir(lecture_id: int) -> Path:
    """
    강의 업로드 파일 저장 디렉토리 경로 반환
    base_dir/data/lectures/{lecture_id} 형태의 절대 경로 생성 """
    return (Path(BASE_DIR) / "data" / "lectures" / str(lecture_id)).resolve()

def _lecture_artifacts_dir(lecture_id: int) -> Path:
    """
    강의 처리 결과물 저장 디렉토리 경로 반환
    base_dir/letures/{lecture_id} 형태의 절대 경로 생성"""
    return (Path(BASE_DIR) / "lectures" / str(lecture_id)).resolve()

def _safe_rmtree(p: Path) -> None:
    """디렉토리를 안전하게 삭제
    경로가 존재하지 않거나 삭제 중 오류 발생해도 무시하고 진행
    """
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


@app.delete("/lectures/{lecture_id}", status_code=status.HTTP_200_OK)
def delete_lecture(lecture_id: int, session: Session = Depends(get_session)):
    """
    특정 강의 및 관련 데이터 완전 삭제
    
    1. 데이터베이스에서 해당 강의 조회
    2. 강의와 연관된 모든 레코드 삭제 (TranscriptSegment, PageAnchor, Assset, Lecture)
    3. 변경사항 커밋
    4. 강의 업로드 파일 디렉토리 삭제
    5. 강의 처리 결과물 디렉토리 삭제

    Args:
        lecture_id (int): 삭제할 강의의 고유 ID
        session (Session, optional): 데이터베이스 세션

    Raises:
        HTTPException: 404 - 해당 강의가 존재하지 않을 경우

    Returns:
        dict: {
            "ok": bool, # 삭제 성공 여부
            "deleted_lecture_id":int, # 삭제된 강의 ID
            "deleted_dirs: list # 삭제된 디렉토리 경로 목록"
        }
    """
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
    """
    모든 강의 및 관련 데이터 완전 삭제

    1. 데이터베이스의 모든 강의 ID 조회
    2. 모든 관련 레코드 일괄 삭제 (TranscriptSegment, PageAnchor, Asset, Lecture)
    3. 변경사항 커밋
    4. 모든 강의 업로드 파일 디렉토리 삭제
    5. 모든 강의 처리 결과물 디렉토리 삭제

    Args:
        session (Session, optional): 데이터베이스 세션

    Returns:
        dict: {
            "ok": bool,           # 삭제 성공 여부
            "deleted_count": int, # 삭제된 강의 개수
            "deleted_ids": list,  # 삭제된 강의 ID 목록
            "deleted_dirs": list  # 삭제된 디렉토리 경로 목록
        }
    """
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
    OCR 결과 페이지 정보 조회 및 요약

    1. pages.json 파일 존재 여부 확인 (없으면 빈 응답 반환)
    2. 각 페이지의 OCR 텍스트에서 주요 키워드 추출
    3. 각 페이지의 첫 문장을 제목으로 생성
    4. 페이지별 요약 정보 (제목, 키워드, 텍스트 길이) 반환

    Args:
        lecture_id (int): 조회할 강의의 고유 ID

    Returns:
        dict: {
            "lecture_id": int,     # 강의 ID
            "num_pages": int,      # 총 페이지 수
            "pages": list[dict]    # 페이지별 요약 정보
                - page: int        # 페이지 번호
                - title: str       # 첫 문장 또는 자동 생성 제목
                - keywords: list   # 빈도 상위 5개 키워드
                - text_length: int # 텍스트 길이
        }
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
        text = (p.get("text") or "").strip().lower() # 텍스트 소문자 변환 및 공백 제거
        tokens = re.findall(r"[0-9a-z가-힣]{3,}", text) # 3글자 이상의 단어/숫자 추출(영문, 한글 모두 포함)
        word_freq = {} # 각 단어의 출현 빈도 계산
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1

        # 빈도 기준 상위 5개 선택
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [k[0] for k in keywords]
        
        # 첫 문장 추출(마침표 기준)
        first_sentence = text.split(".")[0].strip() if text else ""
        # 제목 설정(길이 5글자 이상이면 사용,
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
# Auto_Sync 알고리즘의 목표
# 문제: 강의 PDF 페이지와 음성 자막이 어느 시점에 일치하는지 알고 싶다.

# 예: "페이지 5는 음성에서 2분 30초일 때 나타난다"는 정보를 찾는 것

# 입력(Input):

# PDF에서 추출한 각 페이지의 텍스트 (pages.json)
# 음성을 텍스트로 변환한 자막 (TranscriptSegment) - 시작시간, 끝시간, 텍스트 포함

# 출력(Output):

# 각 페이지가 음성에서 어느 시간에 나타나는지 (PageAnchor) - 페이지 번호, 시간
# =============================================================================
@app.post("/lectures/{lecture_id}/auto_sync")
def auto_sync(lecture_id: int, session: Session = Depends(get_session)):
    # 디버깅을 위한 코드 추가
    print("=== SYNC DEBUG SNAPSHOT ===")
    print("BASE_DIR:", BASE_DIR)
    print("CWD:", os.getcwd())
    print("pages_json_path:", pages_json_path, "exists:", pages_json_path.exists())
    print("pages_count:", len(pages), "first_page:", pages[0].get("page") if pages else None)
    print("transcript_count:", len(transcript_rows), "first_start:", float(transcript_rows[0].start) if transcript_rows else None)
    print("===========================")

    """
    임베딩 + 키워드 스포팅 + 신뢰도 기반 보간법을 활용한 PDF 페이지-음성 자막 자동 동기화

    **알고리즘 개요:**
    1. PDF 각 페이지의 텍스트와 음성 자막을 텍스트-임베딩-3-small 모델로 숫자(벡터)로 변환
    2. 각 페이지와 자막 그룹 간의 의미적 유사도를 코사인 유사도로 계산
    3. 공통 키워드가 있으면 추가 점수 부여 (키워드 스포팅)
    4. 동적 프로그래밍으로 페이지 순서를 지키면서 최고 점수의 매칭 경로 탐색
    5. 점수 임계값(1.3) 이상의 신뢰도 높은 앵커만 선택
    6. 신뢰도 낮은 구간은 선형 보간으로 채우기
    7. 원본 자막(3~5초 단위)으로 최종 미세 조정 (Sliding Window)
    8. 결과를 PageAnchor 테이블에 저장

    **입력 데이터:**
    - pages.json: OCR로 추출한 PDF 각 페이지의 텍스트
    - TranscriptSegment: 음성을 텍스트로 변환한 자막 (시작시간, 끝시간, 텍스트)

    **출력 데이터:**
    - PageAnchor: 각 PDF 페이지가 음성에서 나타나는 시간
    - sync_debug.json: 매칭 경로와 유사도 점수 (디버깅용)
    - similarity_matrix.json: 전체 유사도 행렬 (시각화용)

    Args:
        lecture_id (int): 동기화할 강의의 고유 ID
        session (Session, optional): 데이터베이스 세션. Defaults to Depends(get_session).

    Raises:
        HTTPException: 404 - pages.json 파일이 없을 경우 (먼저 /ocr_pdf 실행 필요)
        HTTPException: 400 - OCR 페이지가 비어 있을 경우
        HTTPException: 400 - Transcript 자막이 없을 경우 (먼저 /transcribe 실행 필요)
        HTTPException: 400 - 자막 그룹 생성 실패할 경우
        HTTPException: 500 - 임베딩 생성 중 오류 발생 시

    Returns:
        dict: {
            "ok": bool,                      # 동기화 성공 여부
            "lecture_id": int,               # 강의 ID
            "anchors_count": int,            # 생성된 앵커 개수
            "anchors": list[dict],           # 페이지별 앵커 정보
                - "page": int                # 페이지 번호
                - "time": float              # 음성에서의 시간(초)
            "debug": {
                "num_pages": int,            # 총 페이지 수
                "num_segment_groups": int,   # 10초 단위로 묶은 자막 그룹 개수
                "matched_pairs_count": int   # DP로 찾은 매칭 쌍의 개수
            }
        }"""
    
    # 1. 내부 유틸 함수 정의
    def clean_text_local(s:str) -> str:
        """
        텍스트를 정규화(normalize)하여 불필요한 문자를 제거하고 표준화하는 함수
        
        Args:
            s (str): 정규화할 텍스트
                    예: "안녕하세요!!! 저는 A학생입니다..."
        
        Returns:
            str: 정규화된 텍스트
                예: "안녕하세요 저는 a학생입니다"
        
        처리 단계:
        1. None/빈 문자열 처리
        2. 대문자를 소문자로 변환
        3. 연속된 공백을 단일 공백으로 정리
        4. 숫자/한글/영문/공백 외의 모든 특수문자 제거
        """
        import re
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^0-9a-z가-힣 ]+", "", s)
        return s
    
    def get_keywords_local(text: str) -> set:
        """
        텍스트에서 의미 있는 키워드(단어)를 추출하는 함수
        
        Args:
            text (str): 키워드를 추출할 텍스트
                    예: "안녕하세요 저는 1 1 2 입니다"
        
        Returns:
            set: 2글자 이상의 단어들의 집합 (순서 없음, 중복 없음)
                예: {"안녕", "하세요", "입니다"}
        
        처리 단계:
        1. 텍스트를 clean_text_local()로 정규화
        2. 정규화된 텍스트를 공백으로 분할하여 단어 리스트 만들기
        3. 2글자 이상의 단어만 필터링
        4. set(집합)으로 변환하여 중복 제거
        """
        normalized = clean_text_local(text)
        return set([w for w in normalized.split() if len(w) >= 2])
    
    # 2. 데이터 로드 및 전처리
    out_dir = _lecture_data_dir(lecture_id) # 강의 데이터가 저장된 디렉토리 경로 반환
    pages_json_path = out_dir / "pages.json" # 그 디렉토리 안의 pages.json 파일 경로(나중에 저장한 OCR결과를 찾기 위해)
    if not pages_json_path.exists(): 
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.") # 경로의 파일이 실제로 존재하지 않으면 에러
    
    pages_obj = _load_json(pages_json_path) # JSON 파일 전체를 파이썬 딕셔너리로 변환, pages_obj는 이 전체 딕셔너리.
    pages = pages_obj.get("pages", []) # pages_obj 딕셔너리에서 pages 배열만 추출
    if not pages: # pages 리스트가 비어있으면 HTTP 400 에러 반환
        raise HTTPException(status_code=400, detail="OCR 페이지가 비어 있습니다.")
    
    transcript_rows = session.exec( # 데이터베이스 쿼리 실행해서 음성 자막 데이터 조회
        select(TranscriptSegment) # TranscriptSegment 테이블에서 조회
        .where(TranscriptSegment.lecture_id == lecture_id) # 해당 강의의 레코드만
        .order_by(TranscriptSegment.start) # 시작 시간 순서대로 정렬
    ).all() # 모든 결과를 리스트로 반환
    
    if not transcript_rows: # 자막이 없으면 HTTP 400에러
        raise HTTPException(status_code=400, detail="Transcript가 없습니다. 먼저 /transcribe를 실행하세요.")
    
    # ===== [STEP 2] 페이지 텍스트 준비 =====
    page_texts = [] # 각 페이지의 텍스트를 저장할 빈 리스트 생성
    for p in pages: # 각 페이지를 순회
        
        # ===== 상황별 처리: 1단계 - 기본 텍스트 추출 =====
        text = (p.get("text") or "").strip() # pages 딕셔너리에서 text 키의 값을 추출해서 없으면 빈 문자열 사용하고 앞뒤 공백 제거
        
        # ===== 상황별 처리: 2단계 - OCR 실패 대비 (words로 복구) =====
        if len(text) < 50 and p.get("words"): # 텍스트가 50자 미만이면, 페이지의 "words"필드가 있으면,
            text = " ".join([w.get("t", "") for w in p.get("words", [])]) # 각 단어를 합쳐서 텍스트 복구
        text = text.strip() # 다시 공백 제거
        
        # ===== 상황별 처리: 3단계 - 완전히 빈 페이지 또는 너무 짧은 페이지 =====
        if not text or len(text) < 5:# 텍스트가 빈 문자열이거나 5글자 미만이면
            text = f"페이지 {p.get('page', 0)} 내용" # 플레이스홀더 텍스트 생성
        
        # ===== 상황별 처리: 4단계 - 최종 저장 (길이 제한 적용) =====
        page_texts.append(text[:2000]) # 텍스트의 처음 2000글자만 선택, pages_texts 리스트에 추가
    
    # ===== [STEP3] 자막 그룹화 (10초 단위) =====
    segment_groups = [] # 완성될 그룹들을 저장할 리스트
    current_group = {"start": 0, "end": 0, "texts": []} # 현재 만들고 있는 그룹(딕셔너리)
    group_duration = 10 # 한 그룹의 목표 길이(10초)
    
    # 각 자막 세그먼트를 순회
    for seg in transcript_rows:
        seg_text = (seg.text or "").strip() # 자막 텍스트 추출, None이면 "", 공백 제거
        if not seg_text: continue # 텍스트가 비어있으면 스킵
        
        # 새 그룹 시작    
        if not current_group["texts"]: # 현재 그룹이 비어있으면(처음 시작 또는 방금 그룹 완성)
            current_group["start"] = seg.start # 이 자막의 시작시간을 그룹의 시작시간으로 설정
        
        # 현재 자막 텍스트를 그룹에 추가
        current_group["texts"].append(seg_text)
        # 끝시간을 현재 자막의 끝시간으로 업데이트
        current_group["end"] = seg.end
        
        # 그룹이 10초 이상이면 저장
        if seg.end - current_group["start"] >= group_duration: # 현재 그룹의 길이(초)가 10초 이상이면
            group_text = " ".join(current_group["texts"]).strip()# 텍스트들을 공백으로 합치기
            if group_text:
                segment_groups.append({# 완성된 그룹을 리스트에 추가
                    "start": current_group["start"],
                    "end": current_group["end"],
                    "text": group_text
                })
            current_group = {"start": 0, "end": 0, "texts": []} # 새로운 그룹 시작을 위해 초기화
    
    # 마지막 그룹 처리(루프가 끝난 후 남은 자막이 있을 수 있음. 그것들도 하나의 그룹으로 저장함. 10초 미만이어도.)
    if current_group["texts"]:
        group_text = " ".join(current_group["texts"]).strip()
        if group_text:
            segment_groups.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "text": group_text
            })
    
    # 그룹이 생성되었는지 확인
    if not segment_groups:
        raise HTTPException(status_code=400, detail="자막 그룹을 생성할 수 없습니다.")
    
    # ===== [STEP 4] 임베딩 생성 =====
    all_texts = [] # 모든 텍스트를 저장할 리스트 생성
    for t in page_texts: # 페이지 텍스트 추가(공백 정리 및 플레이스홀더)=> 빈 텍스트를 임베딩 모델에 주면 의미 있는 벡터가 생성되지 않음. 플레이스홀더를 주면 최소한의 의미 있는 벡터 생성됨.
        clean_t = t.strip()
        all_texts.append(clean_t if clean_t else "빈 페이지")
    
    # 자막 그룹 텍스트 추가
    for g in segment_groups: # 각 자막 그룹을 순회
        clean_t = g["text"][:2000].strip() # 최대 2000자만(너무 길면 잘라내기), 공백 제거
        all_texts.append(clean_t if clean_t else "빈 구간") # 빈 텍스트면 플레이스홀더 사용
    
    # 최종 점검: 혹시 모를 빈 텍스트가 남아있으면 placeholder로 교체
    all_texts = [t if t.strip() else "placeholder" for t in all_texts]
    
    # GPU에서 임베딩 생성
    try:
        # text-embedding-3-small 모델을 사용해 각 텍스트를 벡터로 변환
        # 각 텍스트는 300차원의 벡터가 됨.
        all_embeddings = get_embeddings_from_gpu(all_texts) # 페이지 글이랑 자막 글을 숫자로 바꾼다.
    except Exception as e: # GPU 서버 연결 실패, 모델 로깅 실패 등이 발생할 수 있음. 
        raise HTTPException(status_code=500, detail=f"Embedding 생성 실패: {e}")

    # 페이지와 자막 임베딩 분리
    num_pages = len(page_texts) # 페이지 개수(예를 들어 3개라고 하면)
    page_embeddings = all_embeddings[:num_pages] # 처음 페이지 3개의 벡터(페이지용)
    segment_embeddings = all_embeddings[num_pages:]# 나머지 자막 그룹의 벡터들(자막용)
    
    # ===== [STEP 5] 유사도 매트릭스 계산 =====
    # NumPy 배열로 변환(행렬 연산이 가능하도록)
    # page_embeddings = [
    #     [0.1, 0.5, 0.3],
    #     [0.2, 0.3, 0.1],
    #     [0.15, 0.45, 0.25]
    # ]

    # page_vecs = np.array(page_embeddings, dtype=np.float32)
    # 결과: 3x3 행렬 (실제로는 300x300)
    # [
    #   [0.1,  0.5,  0.3],
    #   [0.2,  0.3,  0.1],
    #   [0.15, 0.45, 0.25]
    # ]
    page_vecs = np.array(page_embeddings, dtype=np.float32)
    seg_vecs = np.array(segment_embeddings, dtype=np.float32)
    
    # 벡터의 Norm(원점에서 해당 벡터의 끝점까지의 거리) 계산. 벡터 [a, b, c]의 노름 = √(a² + b² + c²)
    # axis = 1의 의미: 행별 계산
    # keepdims=True의 의미: 차원(형태) 유지(벡터 정규화를 위해)
    page_norms = np.linalg.norm(page_vecs, axis=1, keepdims=True)
    seg_norms = np.linalg.norm(seg_vecs, axis=1, keepdims=True)
    
    # 벡터 정규화(코사인 유사도 계산을 위해)
    page_vecs_norm = page_vecs / np.where(page_norms > 0, page_norms, 1)
    seg_vecs_norm = seg_vecs / np.where(seg_norms > 0, seg_norms, 1)
    
    # 정규화된 벡터 = 원래 벡터 / ||벡터||

    # 벡터 v = [0.1, 0.5, 0.3], ||v|| ≈ 0.592
    # 정규화 v = [0.1/0.592, 0.5/0.592, 0.3/0.592]
    #        = [0.169, 0.844, 0.507]
    
    # 왜 정규화? 코사인 유사도를 계산하기 위해. 정규화 후 두 벡터의 내적 = 코사인 유사도.
    # page_vecs = [
    #     [0.1, 0.5, 0.3],
    #     [0.2, 0.3, 0.1]
    # ]
    # page_norms = [[0.592], [0.374]]

    # page_vecs_norm = [
    #     [0.1/0.592, 0.5/0.592, 0.3/0.592],    # [0.169, 0.844, 0.507]
    #     [0.2/0.374, 0.3/0.374, 0.1/0.374]     # [0.535, 0.802, 0.267]
    # ]
    
    # 코사인 유사도 계산(행렬 곱)
    # 결과: (페이지 수) x (자막 그룹 수) 크기의 행렬
    similarity_matrix = np.dot(page_vecs_norm, seg_vecs_norm.T) # 유사도 행렬 만드는 코드: 페이지랑 자막이 얼마나 비슷한지 점수표를 만든다.
    
    # similarity_matrix = [
    #     [0.95, 0.20],    # 페이지1: 자막1과 0.95 유사, 자막2와 0.20 유사
    #     [0.30, 0.92],    # 페이지2: 자막1과 0.30 유사, 자막2와 0.92 유사
    #     [0.10, 0.88]     # 페이지3: 자막1과 0.10 유사, 자막2와 0.88 유사
    # ]

    # 해석:
    # 페이지1은 자막1과 가장 비슷 (0.95)
    # 페이지2는 자막2와 가장 비슷 (0.92)
    # 페이지3은 자막2와 가장 비슷 (0.88)
    
    # ===== [STEP 6] Keyword Spotting (키워드 가산점) =====
    # # 각 페이지의 키워드 추출
    # page_keywords = [get_keywords_local(t) for t in page_texts]
    # # 각 자막 그룹의 키워드 추출
    # segment_keywords = [get_keywords_local(g["text"]) for g in segment_groups]
    # 각 페이지의 키워드 추출 (kiwipiepy 형태소 분석)
    page_keywords = [TextProcessor.extract_keywords(t) for t in page_texts]
    # 각 자막 그룹의 키워드 추출
    segment_keywords = [TextProcessor.extract_keywords(g["text"]) for g in segment_groups]
        
    # 키워드 점수 행렬 생성
    num_segments = len(segment_groups) # 자막 그룹 개수
    keyword_matrix = np.zeros((num_pages, num_segments)) # 모두 0으로 채운 행렬 생성
    # 각 칸은 "이 페이지와 이 자막이 공통 키워드로 인한 추가 점수"
    
    # 이중 루프 시작(모든 페이지-자막 쌍 확인)
    for i in range(num_pages): # i는 페이지 인덱스(0부터 num_pages-1까지)
        p_set = page_keywords[i] # p_set = i번 페이지의 키워드 집합
        if not p_set: continue # 키워드가 없으면 다음 페이지로 넘어감
        
        # 각 자막 그룹과 비교
        for j in range(num_segments):  # j는 자막 그룹 인덱스(0부터 num_segments-1까지)
            s_set = segment_keywords[j] # s_set = j번 자막의 키워드 집합
            if not s_set: continue # 키워드가 없으면 다음 자막으로 넘어감
            
            # 공통 키워드 찾기 및 점수 계산
            common_words = p_set & s_set # 두 집합의 교집합인 공통 키워드
            if common_words: # 공통 키워드가 있으면
                # score_boost = len(common_words) * 0.5 # 공통 키워드 개수 x 0.5
                score_boost = len(common_words) * 0.2
                keyword_matrix[i][j] = score_boost # 그 점수를 행렬에 저장
                
                # 오차가 큰 부분을 keyword_matrix로 출력해보기
    
    # 유사도 행렬에 키워드 점수 추가            
    similarity_matrix += keyword_matrix
    
    # ===== [STEP 7]: DP 알고리즘(최적 경로 탐색)
    # DP 테이블 초기화
    # 크기: (페이지 수 + 1) x (자막 그룹 수 + 1)
    dp = np.full((num_pages + 1, num_segments + 1), -np.inf) # 모든 초기값을 마이너스 무한대로 채움.
    dp[0][0] = 0 # 시작점만 0으로 설정
    
    # dp 초기 상태:
    #     자막0  자막1  자막2
    # 페이지0  0    -∞    -∞
    # 페이지1  -∞   -∞    -∞
    # 페이지2  -∞   -∞    -∞
    # 페이지3  -∞   -∞    -∞

    # 각 칸의 의미:
    # dp[i][j] = "페이지 0~i를 자막 0~j로 매칭했을 때의 최고 점수"
    
    # 경로 역추적 정보 저장 딕셔너리: 나중에 최적 경로를 찾기 위해 이전 상태를 기록하는 딕셔너리
    parent = {}
    
    # DP 계산(3중 루프)
    for i in range(1, num_pages + 1): # 페이지 순서를 지키면서 가장 자연스러운 매칭을 고른다.
        for j in range(1, num_segments + 1):
            for prev_j in range(j):
                score = dp[i-1][prev_j] + similarity_matrix[i-1][j-1]
                if score > dp[i][j]:
                    dp[i][j] = score
                    parent[(i, j)] = (i-1, prev_j, j-1)
    
    # 최고 점수 찾기: 마지막 페이지의 점수들(첫번째 제외)에서 최고 점수의 인덱스를 찾고 인덱스 조정(슬라이싱 때문에)
    best_j = np.argmax(dp[num_pages, 1:]) + 1
    
    # 경로 역추적
    path = []
    i, j = num_pages, best_j
    while i > 0 and (i, j) in parent:
        prev_i, prev_j, matched_seg = parent[(i, j)]
        path.append((i - 1, matched_seg))
        i, j = prev_i, prev_j
    
    path.reverse()
    
    # ===== [STEP 8] [핵심] 신뢰도 기반 보간법: 신뢰할 수 있는 앵커들 사이의 빈 부분을 채우기 =====
    # 신뢰도 높은 앵커 리스트 시작
    reliable_anchors = [] # DP로 찾은 경로 중 신뢰도 높은 것만 저장
    reliable_anchors.append({"page": 0, "time": 0.0}) # 시작점(페이지 0, 시간 0.0)을 반드시 포함
    
    # DP 결과에서 신뢰도 높은 매칭만 선택
    for page_idx, seg_idx in path: # path의 각 (페이지, 자막) 쌍을 순회
        score = similarity_matrix[page_idx][seg_idx] # 그 쌍의 유사도 점수
        THRESHOLD = 1.3 # 신뢰도 임계값(이 이상만 신뢰)
        
        # 점수가 높으면 reliable_anchors에 추가
        if score >= THRESHOLD: 
            time = segment_groups[seg_idx]["start"]
            reliable_anchors.append({"page": page_idx + 1, "time": time})
            
    # 마지막 페이지 처리
    last_transcript_time = transcript_rows[-1].end if transcript_rows else 0 
    if reliable_anchors[-1]["page"] < num_pages: # 아직 마지막 페이지까지 도달하지 못했다면
        reliable_anchors.append({"page": num_pages + 1, "time": last_transcript_time}) # 마지막 페이지를 추가

    # 빈 구간 보간
    final_anchors = []
    
    # 신뢰도 높은 앵커들 사이를 순회
    for k in range(len(reliable_anchors) - 1):
        curr_anchor = reliable_anchors[k]
        next_anchor = reliable_anchors[k+1]
        
        # 앵커의 시작과 끝 추출
        start_page = int(curr_anchor["page"])
        end_page = int(next_anchor["page"])
        start_time = curr_anchor["time"]
        end_time = next_anchor["time"]
        
        # 현재 앵커를 최종 결과에 추가
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

    # 최종 정렬: 페이지 번호 순서대로 정렬
    final_anchors.sort(key=lambda x: x["page"])
    
    # ===== [STEP 9] Sliding Window 미세 조정 (Fine-tuning) =====
    # 목표: 10초 단위로 뭉뚱그려진 시간을 원본 자막(3~5초) 단위로 정밀 보정
    
    # 최종 앵커를 저장할 리스트: 미세 조정된 앵커들을 저장할 새로운 리스트
    refined_anchors = []
    
    # 각 앵커를 순회하고 페이지 인덱스 확인
    for anchor in final_anchors: # final_anchors의 각 앵커를 하나씩 처리
        page_idx = anchor["page"] - 1 # 페이지 번호를 0-based index로 변환
        if page_idx < 0 or page_idx >= len(page_texts):
            refined_anchors.append(anchor) # 유효하지 않은 페이지면 그대로 추가하고 
            continue # 계속
        
        # 보간으로 구한 대략적인 시간 저장 
        # 목표: STEP 8 에서 보간으로 구한 시간을 미세하게 조정하는 것
        # 이 시간이 정확한 지 확인하고 더 정확한 시간이 있으면 그걸 사용
        coarse_time = anchor["time"]
        
        # 1. 탐색 범위 설정 (현재 시간 ±15초)
        search_start = max(0, coarse_time - 15)
        search_end = coarse_time + 15
        
        # 2. 범위 내의 "원본 자막 세그먼트" 찾기
        candidates = [
            seg for seg in transcript_rows # transcript_rows는 원보 자막으로 3~5초 단위
            if seg.end >= search_start and seg.start <= search_end # 범위 [search_start, search_end]와 겹치는 자막만 선택
        ]
        
        # 후보가 없으면 원래 시간 사용
        if not candidates:
            refined_anchors.append(anchor)
            continue
            
        # 3. 페이지 키워드 가져오기
        p_keywords = page_keywords[page_idx] # page_keywords[page_index]는 STEP6에서 구한 페이지의 키우더드
        if not p_keywords: # 키워드가 없으면 원래 시간 사용
            refined_anchors.append(anchor)
            continue
            
        # 4. 세그먼트별 매칭 점수 계산 (Sliding)
        best_time = coarse_time # 현재까지 찾은 최고의 시간(초기값: 원래 시간)
        max_score = 0 # 현재까지의 최고 키워드 매칭 개수
        found_better = False # 더 나은 시간을 찾았는지 여부
        
        # 각 자막 후보를 확인하고 점수 계산
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
        if found_better: # 키워드 매칭으로 더 나은 시간을 찾음
            refined_anchors.append({"page": anchor["page"], "time": best_time})
        else:
            refined_anchors.append(anchor)

    # 최종 앵커 결과를 refined_anchors로 교체: 미세 조정된 앵커들로 최종 결과 업데이트
    final_anchors = refined_anchors
    
    # 미세 조정 완료
    
    # ===== [STEP 10] 결과 저장 =====
    old_anchors = session.exec( # 데이터베이스에서 같은 강의의 기존 PageAnchor 레코드 모두 조회
        select(PageAnchor).where(PageAnchor.lecture_id == lecture_id)
    ).all()
    for anchor in old_anchors:
        session.delete(anchor) # 그것들을 하나씩 삭제
    session.commit() # 변경사항 저장
    
    # 새로 생성할 앵커를 저장할 리스트
    anchors_created = []
    # 새로운 앵커 생성 및 추가
    for item in final_anchors:
        anchor = PageAnchor(lecture_id=lecture_id, page=item["page"], time=float(item["time"]))
        session.add(anchor)
        anchors_created.append(item)
    # 모든 변경사항 커밋
    session.commit()
    
    # ===== [STEP 11] 디버그 및 리턴 =====
    # 디버그 정보 저장
    debug_info = {
        "lecture_id": lecture_id,
        "num_pages": num_pages,
        "num_segment_groups": num_segments,
        "matched_pairs": [(p+1, s, float(similarity_matrix[p][s])) for p, s in path],
        "anchors": anchors_created,
    }
    _save_json(out_dir / "sync_debug.json", debug_info)
    
    # 유사도 행렬 저장
    similarity_data = {
        "lecture_id": lecture_id,
        "num_pages": num_pages,
        "num_segments": num_segments,
        "segment_times": [{"start": sg["start"], "end": sg["end"]} for sg in segment_groups],
        "matrix": similarity_matrix.tolist(),
        "matched_path": [(p+1, s) for p, s in path],
    }
    _save_json(out_dir / "similarity_matrix.json", similarity_data)
    
    # 최종 응답 반환
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
    