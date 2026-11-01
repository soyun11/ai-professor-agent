"""RAG 검색 품질 미니 평가

서비스(main.py rag_index/rag_ask)와 같은 방식으로 검색만 재현해서,
질문별 정답 페이지가 top-k 청크 안에 들어오는지 측정한다.

- 청킹: main.py _chunk_text(max_chars=900, overlap=150) 그대로
- 검색: 강의별 인덱스, 코사인 유사도 top-k (청크 단위)
- 지표: Hit@1, Hit@5, MRR (정답 페이지가 처음 나온 청크 순위 기준)
- 임베딩: GPU 서버 대신 sentence-transformers로 같은 모델을 로컬 실행

실행 (backend/ 에서):
    .venv/bin/python eval/rag/eval_rag.py
입력: lectures/{1,2,3}/pages.json (git 히스토리 609ded6^ 에서 추출, git 제외 경로)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent
MODELS = {"bge-m3": "BAAI/bge-m3", "ko-sroberta": "jhgan/ko-sroberta-multitask"}


def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """main.py _chunk_text 복사본 (main.py는 서버 의존성이 많아 import하지 않음)"""
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


def first_hit_rank(ranked_pages: list[int], answer_pages: set[int]) -> int | None:
    """정답 페이지가 처음 등장한 순위(1부터). 없으면 None"""
    for r, p in enumerate(ranked_pages, start=1):
        if p in answer_pages:
            return r
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "hit@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "hit@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
    }


def _selftest() -> None:
    assert first_hit_rank([3, 5, 2], {2}) == 3
    assert first_hit_rank([3, 5], {2}) is None
    s = summarize([1, 3, None, 6])
    assert s["hit@1"] == 0.25 and s["hit@5"] == 0.5
    assert abs(s["mrr"] - (1 + 1 / 3 + 1 / 6) / 4) < 1e-9
    assert _chunk_text("a" * 1000) == ["a" * 900, "a" * 250]


def build_index(lecture: int) -> list[dict]:
    pages = json.loads((BACKEND / "lectures" / str(lecture) / "pages.json").read_text(encoding="utf-8"))["pages"]
    return [
        {"page": int(p["page"]), "text": ch}
        for p in pages
        for ch in _chunk_text(p.get("text", ""))
    ]


def run(model_key: str, questions: list[dict], indexes: dict[int, list[dict]]) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODELS[model_key])
    enc = lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    chunk_vecs = {lec: enc([c["text"] for c in idx]) for lec, idx in indexes.items()}

    rows = []
    for q in questions:
        idx = indexes[q["lecture"]]
        scores = chunk_vecs[q["lecture"]] @ enc([q["question"]])[0]  # 정규화 벡터 내적 = 코사인
        order = np.argsort(-scores)
        ranked_pages = [idx[i]["page"] for i in order]
        rows.append({
            "id": q["id"],
            "rank": first_hit_rank(ranked_pages, set(q["answer_pages"])),
            "top5_pages": ranked_pages[:5],
        })
    return rows


def main() -> None:
    _selftest()
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    indexes = {lec: build_index(lec) for lec in sorted({q["lecture"] for q in questions})}
    qmap = {q["id"]: q for q in questions}

    groups = {
        "전체(음성전용 제외)": lambda q: q["type"] != "spoken",
        "강의3 영어 슬라이드(음성전용 제외)": lambda q: q["lecture"] == 3 and q["type"] != "spoken",
        "강의1+2 한/영 혼합 슬라이드": lambda q: q["lecture"] in (1, 2),
        "약어 질문": lambda q: q["type"] == "acronym",
        "음성에서만 설명한 내용": lambda q: q["type"] == "spoken",
    }

    results = {"chunks_per_lecture": {k: len(v) for k, v in indexes.items()}, "models": {}}
    for mk in MODELS:
        rows = run(mk, questions, indexes)
        results["models"][mk] = {
            "summary": {g: summarize([r["rank"] for r in rows if f(qmap[r["id"]])]) for g, f in groups.items()},
            "per_question": rows,
        }

    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["| 그룹 | n | " + " | ".join(f"{m} Hit@1 / Hit@5 / MRR" for m in MODELS) + " |",
             "|---|---|" + "---|" * len(MODELS)]
    for g in groups:
        cells = []
        for mk in MODELS:
            s = results["models"][mk]["summary"][g]
            cells.append(f"{s['hit@1']:.2f} / {s['hit@5']:.2f} / {s['mrr']:.2f}")
        lines.append(f"| {g} | {results['models']['bge-m3']['summary'][g]['n']} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (HERE / "results.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
