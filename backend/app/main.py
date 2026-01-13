from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from .db import init_db, get_session
from .models import Lecture, Asset, TranscriptSegment
from .storage import save_uploads, BASE_DIR
# from .whisper_runner import transcribe_with_whisper_api
from fastapi.staticfiles import StaticFiles

from .models import PageAnchor

from fastapi import Body 

import os, json
from typing import Any
import numpy as np
from openai import OpenAI
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output
import requests
import re


import shutil
from fastapi import status
from .whisper_runner import transcribe_to_json_with_progress
from .embedding_runner import get_embeddings_from_gpu

import httpx
from typing import List, Dict

# ========== 상단에 import 추가 ==========
from .ocr_runner import ocr_pdf_with_gpu

app = FastAPI(title="AI Agent for Professors - MVP Backend")
app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")

# OpenAI api key 설정
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/lectures")
def create_lecture(payload: dict, session: Session = Depends(get_session)):
    title = (payload.get("title") or "").strip()
    description = (payload.get("description") or "").strip()

    if not title:
        raise HTTPException(status_code=400, detail="title은 필수입니다.")

    lec = Lecture(title=title, description=description)
    session.add(lec)
    session.commit()
    session.refresh(lec)

    asset = Asset(lecture_id=lec.id)
    session.add(asset)
    session.commit()

    return {"lecture_id": lec.id, "title": lec.title, "description": lec.description}


@app.get("/lectures")
def list_lectures(session: Session = Depends(get_session)):
    rows = session.exec(select(Lecture).order_by(Lecture.id.desc())).all()
    return {
        "lectures": [
            {"id": r.id, "title": r.title, "description": r.description}
            for r in rows
        ]
    }


@app.get("/lectures/{lecture_id}")
def get_lecture(lecture_id: int, session: Session = Depends(get_session)):
    lec = session.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(status_code=404, detail="Lecture not found")

    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    return {
        "lecture": lec.model_dump(),
        "asset": asset.model_dump() if asset else None,
    }


@app.post("/lectures/{lecture_id}/upload")
async def upload_assets(
    lecture_id: int,
    pdf: UploadFile = File(...),
    audio: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    lec = session.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(status_code=404, detail="Lecture not found")

    try:
        pdf_path, audio_path = await save_uploads(lecture_id, pdf, audio)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    if not asset:
        asset = Asset(lecture_id=lecture_id)

    asset.pdf_path = pdf_path
    asset.audio_path = audio_path

    session.add(asset)
    session.commit()
    session.refresh(asset)

    return {
        "lecture_id": lecture_id,
        "pdf_path": asset.pdf_path,
        "audio_path": asset.audio_path,
    }
    
# transcribe 변경
@app.post("/lectures/{lecture_id}/transcribe")
def transcribe(lecture_id: int, session: Session = Depends(get_session)):
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    if not asset or not asset.audio_path:
        raise HTTPException(status_code=400, detail="오디오가 업로드되지 않았습니다.")

    audio_abs = (Path(BASE_DIR) / asset.audio_path).resolve()
    if not audio_abs.exists():
        raise HTTPException(status_code=404, detail=f"오디오 파일을 찾을 수 없습니다: {asset.audio_path}")

    # (A) 기존 transcript DB 비우기 (원하면 유지)
    old = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.lecture_id == lecture_id)
    ).all()
    for row in old:
        session.delete(row)
    session.commit()

    # (B) transcript.json 저장 위치 (너 프로젝트 디렉토리 규칙에 맞춤)
    out_dir = _lecture_data_dir(lecture_id)          # 이미 아래에 정의돼 있지? (lectures/{id})
    out_json = out_dir / "transcript.json"

    # (C) faster-whisper + JSON 저장 실행
    try:
        payload = transcribe_to_json_with_progress(
            audio_path=audio_abs,
            out_json_path=out_json,
            lecture_id=lecture_id,
            model_name="large-v3",   # ← 변경
            device="cuda",           # ← 변경
            compute_type="float16",  # ← 변경
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")

    # (D) (선택) DB에도 넣고 싶으면 payload["segments"]를 다시 저장
    segments = payload.get("segments", [])
    for s in segments:
        session.add(
            TranscriptSegment(
                lecture_id=lecture_id,
                start=float(s["start"]),
                end=float(s["end"]),
                text=str(s["text"]),
            )
        )
    session.commit()

    return {
        "ok": True,
        "lecture_id": lecture_id,
        "segments_count": len(segments),
        "transcript_json": str(out_json),
    }

@app.get("/lectures/{lecture_id}/transcript")
def get_transcript(lecture_id: int, session: Session = Depends(get_session)):
    rows = session.exec(
        select(TranscriptSegment)
        .where(TranscriptSegment.lecture_id == lecture_id)
        .order_by(TranscriptSegment.start)
    ).all()

    return {
        "lecture_id": lecture_id,
        "segments": [{"start": r.start, "end": r.end, "text": r.text} for r in rows],
    }


@app.get("/lectures/{lecture_id}/anchors")
def get_anchors(lecture_id: int, session: Session = Depends(get_session)):
    rows = session.exec(
        select(PageAnchor)
        .where(PageAnchor.lecture_id == lecture_id)
        .order_by(PageAnchor.page)
    ).all()
    return {"lecture_id": lecture_id, "anchors": [{"page": r.page, "time": r.time} for r in rows]}


@app.post("/lectures/{lecture_id}/anchors")
def upsert_anchor(lecture_id: int, payload: dict, session: Session = Depends(get_session)):
    page = int(payload.get("page") or 0)
    time = float(payload.get("time") or 0)

    if page < 1:
        raise HTTPException(status_code=400, detail="page는 1 이상이어야 합니다.")
    if time < 0:
        raise HTTPException(status_code=400, detail="time은 0 이상이어야 합니다.")

    row = session.exec(
        select(PageAnchor).where(PageAnchor.lecture_id == lecture_id, PageAnchor.page == page)
    ).first()

    if row:
        row.time = time
    else:
        row = PageAnchor(lecture_id=lecture_id, page=page, time=time)
        session.add(row)

    session.commit()
    return {"ok": True, "lecture_id": lecture_id, "page": page, "time": time}
def page_to_time_with_anchors(page: int, num_pages: int, duration: float, anchors: list[tuple[int, float]]) -> float:
    if page <= 1:
        for p, t in anchors:
            if p == 1:
                return t
        return 0.0

    if num_pages <= 0 or duration <= 0:
        return 0.0

    if not anchors:
        return (page - 1) / max(1, num_pages) * duration

    anchors = sorted(anchors, key=lambda x: x[0])

    for p, t in anchors:
        if p == page:
            return t

    left = None
    right = None
    for p, t in anchors:
        if p < page:
            left = (p, t)
        elif p > page and right is None:
            right = (p, t)
            break

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

    return 0.0


@app.get("/lectures/{lecture_id}/page_time")
def get_page_time(
    lecture_id: int,
    page: int,
    num_pages: int,
    duration: float,
    session: Session = Depends(get_session),
):
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
    d = (Path(BASE_DIR) / "lectures" / str(lecture_id)).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
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
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _norm_text(s: str) -> str:
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
    임베딩 기반으로 PDF 페이지와 오디오 자막을 자동 매칭하여 앵커를 생성합니다.
    """
    
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
    
    # 페이지별 텍스트 (빈 텍스트는 placeholder로 대체)
    page_texts = []
    for p in pages:
        text = (p.get("text") or "").strip()
        if len(text) < 50 and p.get("words"):
            text = " ".join([w.get("t", "") for w in p.get("words", [])])
        text = text.strip()
        # ✅ 빈 텍스트 방지: 최소한의 placeholder
        if not text or len(text) < 5:
            text = f"페이지 {p.get('page', 0)} 내용"
        page_texts.append(text[:2000])
    
    # 30초 단위로 자막 그룹화
    segment_groups = []
    current_group = {"start": 0, "end": 0, "texts": []}
    group_duration = 30
    
    for seg in transcript_rows:
        seg_text = (seg.text or "").strip()
        if not seg_text:
            continue  # ✅ 빈 자막 스킵
            
        if not current_group["texts"]:
            current_group["start"] = seg.start
        
        current_group["texts"].append(seg_text)
        current_group["end"] = seg.end
        
        if seg.end - current_group["start"] >= group_duration:
            group_text = " ".join(current_group["texts"]).strip()
            if group_text:  # ✅ 빈 그룹 방지
                segment_groups.append({
                    "start": current_group["start"],
                    "end": current_group["end"],
                    "text": group_text
                })
            current_group = {"start": 0, "end": 0, "texts": []}
    
    if current_group["texts"]:
        group_text = " ".join(current_group["texts"]).strip()
        if group_text:  # ✅ 빈 그룹 방지
            segment_groups.append({
                "start": current_group["start"],
                "end": current_group["end"],
                "text": group_text
            })
    
    if not segment_groups:
        raise HTTPException(status_code=400, detail="자막 그룹을 생성할 수 없습니다.")
    
    # 임베딩 생성 - 빈 텍스트 필터링
    all_texts = []
    for t in page_texts:
        clean_t = t.strip()
        all_texts.append(clean_t if clean_t else "빈 페이지")
    
    for g in segment_groups:
        clean_t = g["text"][:2000].strip()
        all_texts.append(clean_t if clean_t else "빈 구간")
    
    # ✅ 최종 검증: 모든 텍스트가 비어있지 않은지 확인
    all_texts = [t if t.strip() else "placeholder" for t in all_texts]
    
    # 페이지 텍스트 + 자막 텍스트 모두 임베딩
    try:
        all_embeddings = get_embeddings_from_gpu(all_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding 생성 실패: {e}")
    num_pages = len(page_texts)
    page_embeddings = all_embeddings[:num_pages]
    segment_embeddings = all_embeddings[num_pages:]
    
    # 코사인 유사도 행렬
    page_vecs = np.array(page_embeddings, dtype=np.float32)
    seg_vecs = np.array(segment_embeddings, dtype=np.float32)
    
    page_norms = np.linalg.norm(page_vecs, axis=1, keepdims=True)
    seg_norms = np.linalg.norm(seg_vecs, axis=1, keepdims=True)
    
    page_vecs_norm = page_vecs / np.where(page_norms > 0, page_norms, 1)
    seg_vecs_norm = seg_vecs / np.where(seg_norms > 0, seg_norms, 1)
    
    similarity_matrix = np.dot(page_vecs_norm, seg_vecs_norm.T)
    
    # DP로 순차 매칭
    num_segments = len(segment_groups)
    
    dp = np.full((num_pages + 1, num_segments + 1), -np.inf)
    dp[0][0] = 0
    parent = {}
    
    for i in range(1, num_pages + 1):
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
    
    page_to_segment = {}
    for page_idx, seg_idx in path:
        page_to_segment[page_idx] = seg_idx
    
    # 기존 앵커 삭제
    old_anchors = session.exec(
        select(PageAnchor).where(PageAnchor.lecture_id == lecture_id)
    ).all()
    for anchor in old_anchors:
        session.delete(anchor)
    session.commit()
    
    # 새 앵커 생성
    anchors_created = []
    matched_pages = sorted(page_to_segment.keys())
    
    for page_idx in range(num_pages):
        page_num = page_idx + 1
        
        if page_idx in page_to_segment:
            seg_idx = page_to_segment[page_idx]
            time = segment_groups[seg_idx]["start"]
        else:
            prev_matched = None
            next_matched = None
            
            for mp in matched_pages:
                if mp < page_idx:
                    prev_matched = mp
                elif mp > page_idx and next_matched is None:
                    next_matched = mp
                    break
            
            if prev_matched is not None and next_matched is not None:
                prev_time = segment_groups[page_to_segment[prev_matched]]["start"]
                next_time = segment_groups[page_to_segment[next_matched]]["start"]
                ratio = (page_idx - prev_matched) / (next_matched - prev_matched)
                time = prev_time + ratio * (next_time - prev_time)
            elif prev_matched is not None:
                prev_time = segment_groups[page_to_segment[prev_matched]]["start"]
                prev_end = segment_groups[page_to_segment[prev_matched]]["end"]
                time = prev_end
            elif next_matched is not None:
                next_time = segment_groups[page_to_segment[next_matched]]["start"]
                ratio = page_idx / next_matched if next_matched > 0 else 0
                time = ratio * next_time
            else:
                if transcript_rows:
                    total_duration = transcript_rows[-1].end
                    time = (page_idx / num_pages) * total_duration
                else:
                    time = 0
        
        anchor = PageAnchor(lecture_id=lecture_id, page=page_num, time=float(time))
        session.add(anchor)
        anchors_created.append({"page": page_num, "time": float(time)})
    
    session.commit()
    
    # 디버깅 정보 저장
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