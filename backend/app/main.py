from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from .db import init_db, get_session
from .models import Lecture, Asset, TranscriptSegment
from .storage import save_uploads, BASE_DIR
from .whisper_runner import transcribe_with_whisper_api
from fastapi.staticfiles import StaticFiles

from .models import PageAnchor # 앵커 목록 조회 등에 사용 예정

from fastapi import Body 

import os, json
from typing import Any
import numpy as np
from openai import OpenAI
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output
import re
from fastapi import Query
import shutil
from fastapi import status


app = FastAPI(title="AI Agent for Professors - MVP Backend")
app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")

# OpenAI api key 설정
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# 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

# 프론트(Next.js)에서 호출할 거라 CORS 열어둠 (MVP용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 개발용: 전체 허용
    allow_credentials=False, # allow_origins=["*"]일 때는 True 불가
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

    # Asset 레코드도 미리 생성
    asset = Asset(lecture_id=lec.id)
    session.add(asset)
    session.commit()

    return {"lecture_id": lec.id, "title": lec.title, "description": lec.description}

# Get lecture  추가
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
    
    
@app.post("/lectures/{lecture_id}/transcribe")
def transcribe(lecture_id: int, session: Session = Depends(get_session)):
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    if not asset or not asset.audio_path:
        raise HTTPException(status_code=400, detail="오디오가 업로드되지 않았습니다.")

    audio_abs = (Path(BASE_DIR) / asset.audio_path).resolve()
    if not audio_abs.exists():
        raise HTTPException(status_code=404, detail=f"오디오 파일을 찾을 수 없습니다: {asset.audio_path}")

    # 기존 transcript 삭제 (재실행 대비)
    old = session.exec(
        select(TranscriptSegment).where(TranscriptSegment.lecture_id == lecture_id)
    ).all()
    for row in old:
        session.delete(row)
    session.commit()

    # ✅ whisper CLI가 아니라 Python API로 실행 (인코딩 문제/파일생성 문제 없음)
    try:
        segments = transcribe_with_whisper_api(audio_abs, model_name="tiny")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper STT 실패: {str(e)}")

    # DB 저장
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

    return {"lecture_id": lecture_id, "segments_count": len(segments)}


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


# 페이지 앵커 목록 조회 API 추가
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
    # anchors: sorted list of (page, time)
    if page <= 1:
        # 1페이지 앵커가 있으면 그걸 우선
        for p, t in anchors:
            if p == 1:
                return t
        return 0.0

    if num_pages <= 0 or duration <= 0:
        return 0.0

    if not anchors:
        # fallback: 균등 분배(현재 네 방식)
        return (page - 1) / max(1, num_pages) * duration

    # 범위 밖(가장 앞/뒤)은 가장 가까운 앵커 기반으로 선형 외삽
    # 중간은 구간별 선형 보간
    anchors = sorted(anchors, key=lambda x: x[0])

    # exact match
    for p, t in anchors:
        if p == page:
            return t

    # find neighbors
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
        # 뒤쪽 외삽: 마지막 앵커~끝(duration)으로 비율
        lp, lt = left
        ratio = (page - lp) / max(1, (num_pages - lp))
        return lt + ratio * (duration - lt)

    if right and not left:
        # 앞쪽 외삽: 0~첫 앵커
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



# 유틸 함수 추가
def _lecture_data_dir(lecture_id: int) -> Path:
    # BASE_DIR 밑에 lecture별 산출물 저장
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


# OCR: PDF-> 페이지별 텍스트(pages.json)
@app.post("/lectures/{lecture_id}/ocr_pdf")
def ocr_pdf(lecture_id: int, session: Session = Depends(get_session)):
    asset = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).first()
    if not asset or not asset.pdf_path:
        raise HTTPException(status_code=400, detail="PDF가 업로드되지 않았습니다.")

    pdf_abs = (Path(BASE_DIR) / asset.pdf_path).resolve()
    if not pdf_abs.exists():
        raise HTTPException(status_code=404, detail=f"PDF 파일을 찾을 수 없습니다: {asset.pdf_path}")

    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"

    # PDF -> images
    try:
        images = convert_from_path(str(pdf_abs), dpi=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 렌더링 실패(poppler 확인): {e}")

    pages = []
    for idx, img in enumerate(images, start=1):
        w, h = img.size

        # 1) 전체 텍스트
        try:
            text = pytesseract.image_to_string(img, lang="kor+eng")
        except Exception:
            text = ""

        # 2) 단어 좌표(box) 추출
        words = []
        try:
            data = pytesseract.image_to_data(img, lang="kor+eng", output_type=Output.DICT)
            n = len(data.get("text", []))
            for i in range(n):
                t = (data["text"][i] or "").strip()
                conf = float(data.get("conf", [0])[i] or 0)

                if not t:
                    continue
                if conf < 30:
                    continue

                x = int(data["left"][i]); y = int(data["top"][i])
                ww = int(data["width"][i]); hh = int(data["height"][i])

                words.append({
                    "t": t,                 # 원문 단어
                    "n": _norm_text(t),      # 정규화 단어(매칭용)
                    "x": x / w, "y": y / h,  # ⭐️ 0~1 정규화 좌표
                    "w": ww / w, "h": hh / h
                })
        except Exception:
            words = []

        pages.append({
            "page": idx,
            "text": text,
            "words": words,     # ⭐️ 추가된 핵심 필드
            "img_w": w, "img_h": h
        })


    _save_json(pages_json_path, {"lecture_id": lecture_id, "pages": pages})
    return {"ok": True, "lecture_id": lecture_id, "pages": len(pages), "pages_json": str(pages_json_path)}

# Index: pages.json → index.json (chunk + embedding)
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

    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=[r["text"] for r in records],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding 실패(OPENAI_API_KEY 확인): {e}")

    vectors = [d.embedding for d in emb.data]
    index_obj = {
        "lecture_id": lecture_id,
        "embedding_model": "text-embedding-3-small",
        "items": [{**records[i], "embedding": vectors[i]} for i in range(len(records))],
    }

    index_json_path = out_dir / "index.json"
    _save_json(index_json_path, index_obj)

    return {"ok": True, "lecture_id": lecture_id, "chunks": len(records), "index_json": str(index_json_path)}

# Ask: 질문 → topK 검색 → OpenAI 답변(+근거)
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

    # question embedding
    try:
        qemb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question,
        ).data[0].embedding
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

@app.post("/lectures/{lecture_id}/summary")
def make_summary(lecture_id: int):
    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"
    if not pages_json_path.exists():
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")

    obj = _load_json(pages_json_path)
    pages = obj.get("pages", [])

    # 너무 길면 비용/속도 문제 → 적당히 자름
    joined = "\n\n".join([f"[p{p.get('page')}]\n{p.get('text','')}" for p in pages])
    joined = joined[:12000]

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
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        summary = resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 생성 실패: {e}")

    return {"lecture_id": lecture_id, "summary": summary}
@app.post("/lectures/{lecture_id}/quiz")
def make_quiz(lecture_id: int):
    out_dir = _lecture_data_dir(lecture_id)
    pages_json_path = out_dir / "pages.json"
    if not pages_json_path.exists():
        raise HTTPException(status_code=404, detail="pages.json이 없습니다. 먼저 /ocr_pdf를 실행하세요.")

    obj = _load_json(pages_json_path)
    pages = obj.get("pages", [])
    joined = "\n\n".join([p.get("text", "") for p in pages])
    joined = joined[:12000]

    prompt = f"""다음 OCR 내용을 바탕으로 5개 객관식 퀴즈를 만들어.
반드시 "JSON 배열"만 출력해. 다른 텍스트/코드블록 금지.

형식:
[
  {{
    "question": "문제 내용",
    "options": ["선택지 A", "선택지 B", "선택지 C", "선택지 D"],
    "answer": 0,
    "explanation": "정답 해설"
  }}
]

OCR:
{joined}
"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        raw = (resp.output_text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 실패: {e}")

    # 서버에서 JSON 파싱 시도 (프론트 편하게)
    quiz = []
    try:
        # 혹시 ```json ``` 같은 게 섞일 경우 제거
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        quiz = json.loads(cleaned)
        if not isinstance(quiz, list):
            quiz = []
    except Exception:
        # 파싱 실패하면 raw도 같이 줘서 디버깅 가능하게
        return {"lecture_id": lecture_id, "quiz": [], "raw": raw}

    return {"lecture_id": lecture_id, "quiz": quiz}


# =========================
# Delete APIs (Lecture cleanup)
# =========================

def _lecture_upload_dir(lecture_id: int) -> Path:
    """
    업로드 저장 위치:
    - save_uploads()가 보통 data/lectures/<id>/... 로 저장함
    """
    return (Path(BASE_DIR) / "data" / "lectures" / str(lecture_id)).resolve()

def _lecture_artifacts_dir(lecture_id: int) -> Path:
    """
    OCR/RAG 산출물 저장 위치:
    - 네 코드의 _lecture_data_dir()가 BASE_DIR/lectures/<id> 를 쓰고 있음
    """
    return (Path(BASE_DIR) / "lectures" / str(lecture_id)).resolve()

def _safe_rmtree(p: Path) -> None:
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


@app.delete("/lectures/{lecture_id}", status_code=status.HTTP_200_OK)
def delete_lecture(lecture_id: int, session: Session = Depends(get_session)):
    # 1) lecture 존재 확인
    lec = session.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(status_code=404, detail="Lecture not found")

    # 2) DB 관련 row 삭제 (FK cascade 설정이 없어도 안전하게 수동 삭제)
    # Transcript
    ts = session.exec(select(TranscriptSegment).where(TranscriptSegment.lecture_id == lecture_id)).all()
    for r in ts:
        session.delete(r)

    # Anchors
    an = session.exec(select(PageAnchor).where(PageAnchor.lecture_id == lecture_id)).all()
    for r in an:
        session.delete(r)

    # Asset
    assets = session.exec(select(Asset).where(Asset.lecture_id == lecture_id)).all()
    for r in assets:
        session.delete(r)

    # Lecture
    session.delete(lec)
    session.commit()

    # 3) 파일/산출물 폴더 삭제 (data/lectures/<id>, lectures/<id> 둘 다)
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
    전체 강의 초기화(리셋):
    - Lecture/Asset/Transcript/PageAnchor 전부 삭제
    - data/lectures/*, lectures/* 폴더도 전부 정리
    """
    # 1) 모든 lecture id 수집
    ids = session.exec(select(Lecture.id)).all()
    ids = [int(x) for x in ids]

    # 2) 관련 테이블 싹 삭제 (순서 중요)
    for r in session.exec(select(TranscriptSegment)).all():
        session.delete(r)
    for r in session.exec(select(PageAnchor)).all():
        session.delete(r)
    for r in session.exec(select(Asset)).all():
        session.delete(r)
    for r in session.exec(select(Lecture)).all():
        session.delete(r)

    session.commit()

    # 3) 폴더 정리
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
(형식은 자유롭게 너가 쓰던 걸 써도 됨)

OCR:
{joined}
"""

    try:
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        summary = resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 생성 실패: {e}")

    # ✅ 저장
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    return {"lecture_id": lecture_id, "summary": summary}

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

    # ✅ 저장
    _save_json(out_dir / "quiz.json", quiz)

    return {"lecture_id": lecture_id, "quiz": quiz}

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
