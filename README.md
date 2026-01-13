# 🎓 AI Agent for Professors

교수용 강의 관리 AI 시스템 - PDF 강의자료와 오디오를 업로드하면 자동으로 STT, OCR, 페이지-오디오 싱크, 요약, 퀴즈를 생성합니다.

---

## 📋 목차

- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [설치 방법](#-설치-방법)
- [실행 방법](#-실행-방법)
- [API 엔드포인트](#-api-엔드포인트)
- [데이터 저장 구조](#-데이터-저장-구조)

---

## ✨ 주요 기능

| 기능 | 설명 | 모델/기술 |
|------|------|----------|
| 🎤 **STT** | 강의 오디오를 텍스트로 변환 | faster-whisper (large-v3) |
| 📄 **OCR** | PDF 슬라이드에서 텍스트 추출 | Qwen2.5-VL-7B |
| 🧠 **임베딩** | 한국어 텍스트 벡터화 | jhgan/ko-sroberta-multitask |
| 🔗 **Auto Sync** | 페이지↔오디오 자동 매칭 | 임베딩 기반 DP 알고리즘 |
| 💬 **RAG Q&A** | 강의 내용 기반 질의응답 | GPT-4.1-mini + 벡터 검색 |
| 📝 **요약** | 강의 내용 자동 요약 | GPT-4.1-mini |
| ❓ **퀴즈** | 객관식 문제 자동 생성 | GPT-4.1-mini |

### 학습 모드 (학생용)

- **스캔 모드**: 전체 페이지를 카드로 탐색, 키워드/강조도 표시
- **집중학습 모드**: PDF + 실시간 자막 싱크
- **시험 모드**: 컨닝페이퍼 빌더, 예상문제 확인
- **질의응답 모드**: RAG 기반 Q&A

---

## 🏗 시스템 아키텍처

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│   Frontend      │     │    Backend      │     │      GPU Server         │
│   (Next.js)     │────▶│   (FastAPI)     │────▶│   (RTX A6000 x2)        │
│   :3000         │     │   :8000         │ SSH │                         │
└─────────────────┘     └─────────────────┘     │  ┌───────────────────┐  │
                                                │  │ faster-whisper    │  │
                                                │  │ (STT)             │  │
                                                │  └───────────────────┘  │
                                                │  ┌───────────────────┐  │
                                                │  │ Qwen2.5-VL-7B     │  │
                                                │  │ (OCR)             │  │
                                                │  └───────────────────┘  │
                                                │  ┌───────────────────┐  │
                                                │  │ ko-sroberta       │  │
                                                │  │ (Embedding)       │  │
                                                │  └───────────────────┘  │
                                                └─────────────────────────┘
```

### 데이터 흐름

```
1. 업로드     : PDF + 오디오 → 백엔드 저장
2. STT        : 오디오 → GPU (faster-whisper) → transcript.json
3. OCR        : PDF → GPU (Qwen-VL) → pages.json
4. RAG Index  : pages.json → GPU (임베딩) → index.json
5. Auto Sync  : pages + transcript → GPU (임베딩) → anchors (DB)
6. Summary    : pages.json → GPT-4 → summary.txt
7. Quiz       : pages.json → GPT-4 → quiz.json
```

---

## 🛠 기술 스택

### Frontend
- **Next.js 14** (App Router)
- **TypeScript**
- **Tailwind CSS**
- **shadcn/ui** (컴포넌트)
- **react-pdf** (PDF 뷰어)

### Backend
- **FastAPI** (Python)
- **SQLModel** (ORM)
- **SQLite** (DB)

### GPU Server
- **faster-whisper** (STT) - large-v3 모델
- **Qwen2.5-VL-7B-Instruct** (OCR) - Vision-Language 모델
- **sentence-transformers** (임베딩) - jhgan/ko-sroberta-multitask

### External API
- **OpenAI GPT-4.1-mini** (요약, 퀴즈, RAG 답변)

---

## 📁 프로젝트 구조

```
ai-agent-professors/
├── frontend/                    # Next.js 프론트엔드
│   ├── app/
│   │   ├── page.tsx            # 메인 페이지
│   │   └── lectures/[id]/
│   │       └── page.tsx        # 강의 상세 페이지
│   ├── components/
│   │   ├── TopNav.tsx
│   │   ├── PdfViewer.tsx
│   │   └── ui/                 # shadcn 컴포넌트
│   └── package.json
│
├── backend/                     # FastAPI 백엔드
│   ├── app/
│   │   ├── main.py             # API 엔드포인트
│   │   ├── models.py           # DB 모델
│   │   ├── db.py               # DB 설정
│   │   ├── storage.py          # 파일 저장
│   │   ├── whisper_runner.py   # STT 실행 (SSH)
│   │   ├── embedding_runner.py # 임베딩 실행 (SSH)
│   │   └── ocr_runner.py       # OCR 실행 (SSH)
│   ├── data/                   # 업로드/생성 파일
│   │   └── lectures/{id}/
│   └── requirements.txt
│
└── GPU Server (별도 서버)
    ├── whisper-gpu/             # STT + 임베딩
    │   ├── venv/
    │   ├── whisper_server.py
    │   └── embedding_server.py
    │
    └── qwen-ocr/                # OCR
        ├── venv/
        └── ocr_script.py
```

---

## 🚀 설치 방법

### 1. 프론트엔드

```bash
cd frontend
npm install
```

### 2. 백엔드

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. GPU 서버 - STT & 임베딩

```bash
# whisper-gpu 환경
cd ~/whisper-gpu
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper
pip install sentence-transformers
pip install transformers==4.44.0  # 호환성 위해 버전 고정
```

### 4. GPU 서버 - OCR (Qwen-VL)

```bash
# qwen-ocr 환경 (별도 가상환경)
cd ~/qwen-ocr
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45.0 accelerate qwen-vl-utils
pip install pillow pdf2image

# poppler 설치 (PDF 변환용)
sudo apt install poppler-utils
```

### 5. 환경 변수

```bash
# 백엔드 (.env 또는 환경변수)
export OPENAI_API_KEY="sk-..."
```

---

## ▶️ 실행 방법

### 1. 백엔드 실행

```bash
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 실행

```bash
cd frontend
npm run dev
```

### 3. 접속

- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:8000
- API 문서: http://localhost:8000/docs

---

## 📡 API 엔드포인트

### 강의 관리

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/lectures` | 강의 생성 |
| GET | `/lectures` | 강의 목록 |
| GET | `/lectures/{id}` | 강의 상세 |
| DELETE | `/lectures/{id}` | 강의 삭제 |
| POST | `/lectures/{id}/upload` | PDF/오디오 업로드 |

### AI 파이프라인

| Method | Endpoint | 설명 | 결과 파일 |
|--------|----------|------|----------|
| POST | `/lectures/{id}/transcribe` | STT 실행 | transcript.json |
| POST | `/lectures/{id}/ocr_pdf` | OCR 실행 | pages.json |
| POST | `/lectures/{id}/rag_index` | 임베딩 인덱스 생성 | index.json |
| POST | `/lectures/{id}/auto_sync` | 페이지-오디오 자동 싱크 | anchors (DB) |
| POST | `/lectures/{id}/summary` | 요약 생성 | summary.txt |
| POST | `/lectures/{id}/quiz` | 퀴즈 생성 | quiz.json |

### 조회

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/lectures/{id}/transcript` | 자막 조회 |
| GET | `/lectures/{id}/anchors` | 앵커 조회 |
| GET | `/lectures/{id}/summary` | 요약 조회 |
| GET | `/lectures/{id}/quiz` | 퀴즈 조회 |
| GET | `/lectures/{id}/similarity_matrix` | 유사도 히트맵 |
| POST | `/lectures/{id}/rag_ask` | RAG 질의응답 |

---

## 💾 데이터 저장 구조

```
backend/data/lectures/{lecture_id}/
├── source.pdf              # 업로드된 PDF
├── audio.mp3               # 업로드된 오디오
├── transcript.json         # STT 결과 (자막)
├── pages.json              # OCR 결과 (페이지별 텍스트)
├── index.json              # RAG 임베딩 인덱스
├── sync_debug.json         # Auto Sync 디버그 정보
├── similarity_matrix.json  # 페이지-자막 유사도 행렬
├── summary.txt             # GPT 생성 요약
├── quiz.json               # GPT 생성 퀴즈
└── chat.json               # 채팅 기록
```

### 파일 포맷 예시

#### transcript.json
```json
{
  "segments": [
    {"start": 0.0, "end": 3.5, "text": "안녕하세요, 오늘 강의를 시작하겠습니다."},
    {"start": 3.5, "end": 7.2, "text": "오늘은 TCP 프로토콜에 대해 알아보겠습니다."}
  ]
}
```

#### pages.json
```json
{
  "lecture_id": 7,
  "num_pages": 10,
  "pages": [
    {"page": 1, "text": "TCP/IP 프로토콜 개요..."},
    {"page": 2, "text": "3-way handshake..."}
  ]
}
```

---

## 🔧 SSH 설정

GPU 서버와 백엔드 간 SSH 키 인증 설정이 필요합니다:

```bash
# Windows 백엔드에서
ssh-keygen -t rsa -b 4096
ssh-copy-id user@gpu-server

# ~/.ssh/config 설정
Host test223
    HostName gpu-server-ip
    User sypark
    IdentityFile ~/.ssh/id_rsa
```

---

## 📊 성능 참고

| 작업 | 소요 시간 (예상) |
|------|-----------------|
| STT (30분 오디오) | ~3-5분 |
| OCR (10페이지 PDF) | ~2-3분 |
| RAG Index (30청크) | ~10초 |
| Auto Sync | ~15초 |
| Summary | ~5초 |
| Quiz | ~5초 |

---

## 📝 라이선스

MIT License

---

## 🙏 크레딧

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - STT
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - OCR
- [sentence-transformers](https://www.sbert.net/) - 임베딩
- [OpenAI](https://openai.com/) - GPT API