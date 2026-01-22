// frontend/app/lectures/[id]/page.tsx
"use client";

import React, { useEffect, useRef, useState, useMemo } from "react";
import { useParams } from "next/navigation";
import TopNav from "@/components/TopNav";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  Upload,
  PlayCircle,
  FileText,
  Sparkles,
  Grid3X3,
  BookOpen,
  Clock,
  Volume2,
  ChevronRight,
  Zap,
  FileDown,
  Printer,
  Star,
  Flame,
  Search,
  Filter,
  MessageCircle,
  GraduationCap,
  Settings,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Copy,
  RefreshCw,
  PenLine,
  ChevronDown,
  ChevronUp,
  Bookmark,
  Target,
  Brain,
  ListChecks,
  FileQuestion,
  ArrowRight,
} from "lucide-react";
import dynamic from "next/dynamic";
import { getStoredRole, type UserRole } from "@/lib/role";

const PdfViewer = dynamic(() => import("@/components/PdfViewer"), { ssr: false });

const API_BASE = "http://127.0.0.1:8000";

type Anchor = { page: number; time: number };
type TranscriptSeg = { start: number; end: number; text: string };
type Citation = { chunkId: string; page: number; snippet: string };
type QuizItem = { question: string; options: string[]; answer: number; explanation?: string };

// 페이지 카드 데이터 타입
type PageCard = {
  page: number;
  title: string; // 자동 생성된 제목
  keywords: string[];
  emphasisScore: number; // 0~1, 교수님이 얼마나 강조했는지
  duration: number; // 해당 페이지 설명 시간 (초)
  transcript: string; // 해당 페이지의 자막
  startTime: number; // 시작 시간
  endTime: number; // 종료 시간
  isStarred: boolean; // 학생이 ⭐ 표시했는지
};

// 학습 모드 타입
type StudyMode = "scan" | "focus" | "exam" | "chat";

// 컨닝페이퍼 설정 타입
type CheatSheetConfig = {
  pageCount: number; // 1, 2, 3 페이지
  includeConceptSummary: boolean;
  includeFormulas: boolean;
  includeEmphasis: boolean;
  includeKeywords: boolean;
  includeQuestions: boolean;
};

export default function LectureDetailPage() {


  const getPageStartTime = (page: number) => {
  // 1) anchors가 있으면 anchors 기반이 가장 정확함
  const a = anchors.find((x) => x.page === page);
  if (a) return a.time;

  // 2) pageCards가 있으면 pageCards 기반(너가 계산해둔 값)
  const c = pageCards.find((x) => x.page === page);
  if (c) return c.startTime;

  // 3) 마지막 fallback: 균등 분할
  if (audioDuration > 0 && numPages > 0) {
    return (page - 1) * (audioDuration / numPages);
  }
  return 0;
};

const goToPageAndPlay = (page: number) => {
  const p = Math.max(1, Math.min(numPages || 1, page));
  setViewPage(p);

  // 오디오 메타가 로드된 이후에만 점프
  const t = getPageStartTime(p);
  if (audioDuration > 0) {
    setIsAutoSync(true);   // 오디오->페이지 자동동기화는 켜두는게 UX 좋음
    jumpTo(t);             // 여기서 play까지 됨
  }
};

  const params = useParams<{ id: string }>();
  const lectureId = Number(params?.id);

  if (!params?.id || Number.isNaN(lectureId)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900">
        <div className="text-sm text-slate-400">잘못된 강의 ID</div>
      </div>
    );
  }

  const [role, setRole] = useState<UserRole>("professor");
  const [lectureTitle, setLectureTitle] = useState<string>("Lecture");
  const [lectureDesc, setLectureDesc] = useState<string>("");

  const [pdf, setPdf] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);

  const [pdfUrl, setPdfUrl] = useState("");
  const [audioUrl, setAudioUrl] = useState("");

  const [status, setStatus] = useState("강의 로드 준비 ✨");
  const [isProcessing, setIsProcessing] = useState(false);
  const [agentThinking, setAgentThinking] = useState("");
  const [summary, setSummary] = useState("");
  const [quiz, setQuiz] = useState<QuizItem[]>([]);
  const [chatHistory, setChatHistory] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [currentMessage, setCurrentMessage] = useState("");

  const [transcript, setTranscript] = useState<TranscriptSeg[]>([]);
  const [anchors, setAnchors] = useState<Anchor[]>([]);
  const [anchorPage, setAnchorPage] = useState(1);

  const [ocrReady, setOcrReady] = useState(false);
  const [indexReady, setIndexReady] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const [numPages, setNumPages] = useState(0);
  const [pdfBlobUrl, setPdfBlobUrl] = useState("");
  const prevPdfBlobUrlRef = useRef("");

  const [viewPage, setViewPage] = useState(1);
  const canPageJump = audioDuration > 0 && numPages > 0;

  const [dragActive, setDragActive] = useState({ pdf: false, audio: false });

  // ✅ 학습 모드 상태
  const [studyMode, setStudyMode] = useState<StudyMode>("scan");
  
  // ✅ 페이지 카드 데이터
  const [pageCards, setPageCards] = useState<PageCard[]>([]);
  
  // ✅ 검색/필터 상태
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<"page" | "emphasis" | "starred">("page");
  
  // ✅ 컨닝페이퍼 설정
  const [cheatSheetConfig, setCheatSheetConfig] = useState<CheatSheetConfig>({
    pageCount: 1,
    includeConceptSummary: true,
    includeFormulas: true,
    includeEmphasis: true,
    includeKeywords: true,
    includeQuestions: true,
  });
  const [generatedCheatSheet, setGeneratedCheatSheet] = useState<string>("");
  const [isGeneratingCheatSheet, setIsGeneratingCheatSheet] = useState(false);

  // ✅ 교수 모드 탭 (업로드/싱크)
  const [showProfessorTools, setShowProfessorTools] = useState(false);

  // ✅ 히트맵 데이터
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapData, setHeatmapData] = useState<{
    matrix: number[][];
    segment_times: { start: number; end: number }[];
    matched_path: [number, number][];
    num_pages: number;
    num_segments: number;
  } | null>(null);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState(false);

  const lsKey = (name: string) => `lecture:${lectureId}:${name}`;

  // 로컬 스토리지 저장/로드
  const saveLectureCache = () => {
    try {
      localStorage.setItem(lsKey("summary"), summary || "");
      localStorage.setItem(lsKey("quiz"), JSON.stringify(quiz || []));
      localStorage.setItem(lsKey("chat"), JSON.stringify(chatHistory || []));
      localStorage.setItem(lsKey("ocrReady"), JSON.stringify(!!ocrReady));
      localStorage.setItem(lsKey("indexReady"), JSON.stringify(!!indexReady));
      localStorage.setItem(lsKey("starredPages"), JSON.stringify(
        pageCards.filter(c => c.isStarred).map(c => c.page)
      ));
    } catch {}
  };

  const loadLectureCache = () => {
    try {
      const cachedSummary = localStorage.getItem(lsKey("summary"));
      const cachedQuiz = localStorage.getItem(lsKey("quiz"));
      const cachedChat = localStorage.getItem(lsKey("chat"));
      const cachedOcrReady = localStorage.getItem(lsKey("ocrReady"));
      const cachedIndexReady = localStorage.getItem(lsKey("indexReady"));

      setSummary(cachedSummary ?? "");
      setQuiz(cachedQuiz ? JSON.parse(cachedQuiz) : []);
      setChatHistory(cachedChat ? JSON.parse(cachedChat) : []);
      setOcrReady(cachedOcrReady ? JSON.parse(cachedOcrReady) : false);
      setIndexReady(cachedIndexReady ? JSON.parse(cachedIndexReady) : false);
    } catch {
      setSummary("");
      setQuiz([]);
      setChatHistory([]);
      setOcrReady(false);
      setIndexReady(false);
    }
  };

  // ✅ 페이지 카드 데이터 생성
  const generatePageCards = useMemo(() => {
    if (numPages === 0) return [];

    const sortedAnchors = [...anchors].sort((a, b) => a.page - b.page);
    const cards: PageCard[] = [];
    
    // 로컬 스토리지에서 ⭐ 표시된 페이지 로드
    let starredPages: number[] = [];
    try {
      const saved = localStorage.getItem(lsKey("starredPages"));
      starredPages = saved ? JSON.parse(saved) : [];
    } catch {}

    for (let p = 1; p <= numPages; p++) {
      const anchor = sortedAnchors.find((a) => a.page === p);
      const nextAnchor = sortedAnchors.find((a) => a.page > p);

      const startTime = anchor?.time ?? (p - 1) * (audioDuration / numPages);
      const endTime = nextAnchor?.time ?? p * (audioDuration / numPages);
      const duration = endTime - startTime;

      // 해당 시간 범위의 자막 찾기
      const pageTranscripts = transcript.filter(
        (t) => t.start >= startTime && t.start < endTime
      );
      const transcriptText = pageTranscripts.map((t) => t.text).join(" ");

      // [수정] 조사/어미 제거 및 불용어 처리 강화
      const stopWords = new Set([
        "있습니다", "합니다", "하는", "할", "수", "것입니다", "그리고", "그래서", "하지만", 
        "저는", "제가", "그", "저", "이", "여러분", "오늘", "이번", "시간", "대해서", "대해",
        "보시면", "보면", "다음", "이제", "여기", "저기", "그럼", "자", "네", "아", "음"
      ]);

      const words = transcriptText
        .replace(/[.,!?'"()\[\]]/g, " ") // 특수문자 제거
        .split(/\s+/)
        .map(w => {
            // 1. 기본적인 조사/어미 제거 (간이 형태소 분석)
            let clean = w.trim();
            if (clean.endsWith("은") || clean.endsWith("는") || clean.endsWith("이") || clean.endsWith("가") || 
                clean.endsWith("을") || clean.endsWith("를") || clean.endsWith("에") || clean.endsWith("의") || 
                clean.endsWith("로") || clean.endsWith("도")) {
                clean = clean.slice(0, -1);
            }
            if (clean.endsWith("에서") || clean.endsWith("으로")) {
                clean = clean.slice(0, -2);
            }
            return clean;
        })
        .filter(w => w.length >= 2) // 2글자 이상만
        .filter(w => !stopWords.has(w)); // 불용어 제거

      const wordFreq: Record<string, number> = {};
      words.forEach((w) => {
        wordFreq[w] = (wordFreq[w] || 0) + 1;
      });
      
      const keywords = Object.entries(wordFreq)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([word]) => word);

      // 강조도 계산 (설명 시간 + 강조 표현 빈도)
      const avgDuration = audioDuration / numPages;
      const emphasisWords = ["중요", "핵심", "시험", "꼭", "반드시", "기억", "주의"];
      const emphasisCount = emphasisWords.reduce((acc, word) => 
        acc + (transcriptText.match(new RegExp(word, "gi"))?.length || 0), 0
      );
      const timeScore = Math.min(1, duration / (avgDuration * 1.5));
      const emphasisScore = Math.min(1, (timeScore * 0.6) + (emphasisCount * 0.1));

      // 자동 제목 생성 (첫 번째 문장 또는 키워드 기반)
      const firstSentence = transcriptText.split(/[.!?]/)[0]?.trim() || "";
      const title = firstSentence.length > 5 && firstSentence.length < 50 
        ? firstSentence 
        : keywords.slice(0, 3).join(", ") || `페이지 ${p}`;

      cards.push({
        page: p,
        title,
        keywords: keywords.length > 0 ? keywords : [`페이지 ${p}`],
        emphasisScore,
        duration,
        transcript: transcriptText,
        startTime,
        endTime,
        isStarred: starredPages.includes(p),
      });
    }

    return cards;
  }, [numPages, anchors, transcript, audioDuration, lectureId]);

  useEffect(() => {
    setPageCards(generatePageCards);
  }, [generatePageCards]);

  // ⭐ 토글 핸들러
  const toggleStar = (page: number) => {
    setPageCards(prev => prev.map(card => 
      card.page === page ? { ...card, isStarred: !card.isStarred } : card
    ));
  };

  // [추가] 오디오 시간에 맞춰 페이지를 자동으로 넘길지 여부 (기본값: ture)
  const [isAutoSync, setIsAutoSync] = useState(true);


  // 필터링 및 정렬된 카드
  const filteredCards = useMemo(() => {
    let result = [...pageCards];
    
    // 검색 필터
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(card => 
        card.title.toLowerCase().includes(query) ||
        card.keywords.some(k => k.toLowerCase().includes(query)) ||
        card.transcript.toLowerCase().includes(query)
      );
    }
    
    // 정렬
    switch (sortBy) {
      case "emphasis":
        result.sort((a, b) => b.emphasisScore - a.emphasisScore);
        break;
      case "starred":
        result.sort((a, b) => (b.isStarred ? 1 : 0) - (a.isStarred ? 1 : 0));
        break;
      default:
        result.sort((a, b) => a.page - b.page);
    }
    
    return result;
  }, [pageCards, searchQuery, sortBy]);

  // 현재 페이지의 카드 정보
  const currentPageCard = useMemo(() => {
    return pageCards.find(c => c.page === viewPage);
  }, [pageCards, viewPage]);

  useEffect(() => {
    setRole(getStoredRole());
    const onRoleChange = () => setRole(getStoredRole());
    window.addEventListener("aiagent:rolechange", onRoleChange);
    return () => window.removeEventListener("aiagent:rolechange", onRoleChange);
  }, []);

  // 강의 메타/에셋 로드 함수 수정
  const loadLecture = async () => {
    setStatus("강의 불러오는 중...");
    setCitations([]);
    setPdf(null);
    setAudio(null);

    // 1. 로컬 스토리지 캐시 먼저 로드(빠른 화면 표시용)
    loadLectureCache();

    try {
      const res = await fetch(`${API_BASE}/lectures/${lectureId}`);
      if (res.ok) {
        const data = await res.json();
        setLectureTitle(data?.lecture?.title || data?.title || `Lecture ${lectureId}`);
        setLectureDesc(data?.lecture?.description || data?.description || "");
        const asset = data?.asset;
        const nextPdfUrl = asset?.pdf_path ? `${API_BASE}/files/${asset.pdf_path}` : "";
        const nextAudioUrl = asset?.audio_path ? `${API_BASE}/files/${asset.audio_path}` : "";
        setPdfUrl(nextPdfUrl);
        setAudioUrl(nextAudioUrl);
      }
    } catch {
      setPdfUrl("");
      setAudioUrl("");
    }

    // 3. [핵심] 서버의 작업 상태 확인 (파일 존재 여부 체크)
    try {
      const statusRes = await fetch(`${API_BASE}/lectures/${lectureId}/status`);
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        
        // (A) OCR 및 Index 상태 동기화
        if (statusData.ocr) setOcrReady(true);
        if (statusData.index) setIndexReady(true);

        // (B) 각 데이터가 존재하면 내용물 가져오기 (이미 있으면 안 가져와도 되지만 확실하게 하기 위해)
        
        // Transcript (STT)
        if (statusData.transcript) {
           const tRes = await fetch(`${API_BASE}/lectures/${lectureId}/transcript`);
           if (tRes.ok) {
             const tJson = await tRes.json();
             setTranscript(tJson.segments || []);
           }
        }

        // Summary (요약)
        if (statusData.summary) {
          const sRes = await fetch(`${API_BASE}/lectures/${lectureId}/summary`);
          if (sRes.ok) {
            const sJson = await sRes.json();
            setSummary(sJson.summary || "");
          }
        }

        // Quiz (퀴즈)
        if (statusData.quiz) {
          const qRes = await fetch(`${API_BASE}/lectures/${lectureId}/quiz`);
          if (qRes.ok) {
            const qJson = await qRes.json();
            setQuiz(Array.isArray(qJson.quiz) ? qJson.quiz : []);
          }
        }
      }
    } catch (e) {
      console.error("상태 확인 실패:", e);
    }


    // 4. Anchors 및 Pages Info 로드(기존 로직 유지)
    try {
      const aRes = await fetch(`${API_BASE}/lectures/${lectureId}/anchors`);
      if (aRes.ok) {
        const aJson = await aRes.json();
        setAnchors(aJson.anchors || []);
        if (aJson.anchors && aJson.anchors.length > 0) {
          const maxPage = Math.max(...aJson.anchors.map((a: Anchor) => a.page));
          if (maxPage > 0) setNumPages(maxPage);
        }
      }
    } catch {}

    // 페이지 수 확인 (OCR 완료 여부 재확인 용도)
    try {
      const pagesRes = await fetch(`${API_BASE}/lectures/${lectureId}/pages_info`);
      if (pagesRes.ok) {
        const pagesData = await pagesRes.json();
        if (pagesData.num_pages > 0) {
          setNumPages(pagesData.num_pages);
          setOcrReady(true); // 혹시 status API가 실패했어도 여기서 복구
        }
      }
    } catch {}

    setStatus("✅ 강의 로드 완료");
  };

  useEffect(() => {
    if (!Number.isFinite(lectureId)) return;
    loadLecture();
  }, [lectureId]);

  useEffect(() => {
    if (!Number.isFinite(lectureId)) return;
    saveLectureCache();
  }, [summary, quiz, chatHistory, ocrReady, indexReady, pageCards]);

  // PDF Blob URL 생성
  useEffect(() => {
    const run = async () => {
      if (!pdfUrl) {
        if (prevPdfBlobUrlRef.current) {
          URL.revokeObjectURL(prevPdfBlobUrlRef.current);
          prevPdfBlobUrlRef.current = "";
        }
        setPdfBlobUrl("");
        return;
      }

      try {
        const res = await fetch(pdfUrl);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        if (prevPdfBlobUrlRef.current) URL.revokeObjectURL(prevPdfBlobUrlRef.current);
        prevPdfBlobUrlRef.current = url;
        setPdfBlobUrl(url);
      } catch {
        setPdfBlobUrl("");
      }
    };

    run();
    return () => {
      if (prevPdfBlobUrlRef.current) {
        URL.revokeObjectURL(prevPdfBlobUrlRef.current);
        prevPdfBlobUrlRef.current = "";
      }
    };
  }, [pdfUrl]);


  // [추가1] 현재 재생 중인 자막 인덱스 계산
  const activeTranscriptIndex = useMemo(() => {
    return transcript.findIndex(
      (seg) => currentTime >= seg.start && currentTime < seg.end
    );
  }, [currentTime, transcript]);

  // [추가2] 자막 자동 스크롤(재생 위치에 맞춰 스크롤 이동)
  useEffect(() => {
    if (activeTranscriptIndex !== -1 && studyMode === "focus") {
      const element = document.getElementById(`transcript-seg-${activeTranscriptIndex}`);
      if (element) {
        element.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }
    }
  }, [activeTranscriptIndex, studyMode]);
  useEffect(() => {
    // anchors가 없거나, 사용자가 수동 조작 중(!isAutoSync)이면 자동 넘김 방지
    if (anchors.length === 0 || !isAutoSync) return;

    const sorted = [...anchors].sort((a, b) => a.time - b.time);
    let target = 1;
    
    for (const a of sorted) {
      if (currentTime >= a.time) target = a.page;
      else break;
    }
    
    if (target !== viewPage) setViewPage(target);
  }, [currentTime, anchors, viewPage, isAutoSync]);
  // Drag & Drop
  const handleDrag = (e: React.DragEvent, type: "pdf" | "audio") => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive((prev) => ({ ...prev, [type]: true }));
    } else if (e.type === "dragleave") {
      setDragActive((prev) => ({ ...prev, [type]: false }));
    }
  };

  const handleDrop = (e: React.DragEvent, type: "pdf" | "audio") => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive((prev) => ({ ...prev, [type]: false }));

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (type === "pdf" && file.type === "application/pdf") setPdf(file);
      if (type === "audio" && file.type.startsWith("audio/")) setAudio(file);
    }
  };

  const uploadFiles = async () => {
    if (!lectureId) return;
    if (!pdf || !audio) {
      setStatus("⚠️ PDF와 오디오를 모두 선택해주세요");
      return;
    }
    setIsProcessing(true);
    setStatus("파일 업로드 중...");
    try {
      const formData = new FormData();
      formData.append("pdf", pdf);
      formData.append("audio", audio);

      const res = await fetch(`${API_BASE}/lectures/${lectureId}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setStatus(`에러: ${data?.detail ?? "업로드 실패"}`);
        return;
      }

      setAudioUrl(`${API_BASE}/files/${data.audio_path}`);
      setPdfUrl(`${API_BASE}/files/${data.pdf_path}`);
      setStatus("✅ 파일 업로드 완료");
      setPdf(null);
      setAudio(null);
    } catch (e: any) {
      setStatus(`업로드 실패: ${e?.message || "오류"}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const jumpTo = (time: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, time);
    audioRef.current.play().catch(() => {});
    setIsPlaying(true);
  };

  const jumpToPage = async (page: number) => {
    if (!lectureId || !canPageJump) {
      setStatus("페이지 점프 불가");
      return;
    }
    try {
      const res = await fetch(
        `${API_BASE}/lectures/${lectureId}/page_time?page=${page}&num_pages=${numPages}&duration=${audioDuration}`
      );
      if (!res.ok) throw new Error("page_time 실패");
      const json = await res.json();
      jumpTo(Number(json.time || 0));
      setViewPage(page);
      setStatus(`페이지 ${page} → ${Number(json.time || 0).toFixed(2)}s 점프`);
    } catch (e: any) {
      setStatus(`페이지 점프 실패: ${e?.message || "오류"}`);
    }
  };

// [수정] 
  const handleCardClick = async (card: PageCard) => {
    // 1. 먼저 싱크를 켭니다 (해당 시간으로 갈 거니까)
    setIsAutoSync(true);
    // 2. 뷰 페이지 설정
    setViewPage(card.page);
    // 3. 모드 변경
    setStudyMode("focus");
    // 4. 오디오 점프 (약간의 딜레이를 주어 상태 반영 확보)
    setTimeout(() => jumpTo(card.startTime), 50);
  };

  const loadAnchors = async () => {
    if (!lectureId) return;
    try {
      const res = await fetch(`${API_BASE}/lectures/${lectureId}/anchors`);
      const json = await res.json();
      setAnchors(json.anchors || []);
    } catch {}
  };

  // 히트맵 데이터 로드
  const loadHeatmapData = async () => {
    if (!lectureId) return;
    setIsLoadingHeatmap(true);
    try {
      const res = await fetch(`${API_BASE}/lectures/${lectureId}/similarity_matrix`);
      if (res.ok) {
        const data = await res.json();
        setHeatmapData({
          matrix: data.matrix || [],
          segment_times: data.segment_times || [],
          matched_path: data.matched_path || [],
          num_pages: data.num_pages || 0,
          num_segments: data.num_segments || 0,
        });
      }
    } catch (e) {
      console.error("히트맵 데이터 로드 실패:", e);
    } finally {
      setIsLoadingHeatmap(false);
    }
  };

  const saveAnchor = async () => {
    if (!lectureId) return;
    try {
      const payload = { page: anchorPage, time: currentTime };
      const res = await fetch(`${API_BASE}/lectures/${lectureId}/anchors`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("앵커 저장 실패");
      setStatus(`앵커 저장 완료: page=${anchorPage}, time=${currentTime.toFixed(2)}s`);
      loadAnchors();
    } catch (e: any) {
      setStatus(`앵커 저장 실패: ${e?.message || "오류"}`);
    }
  };

  const runTranscribe = async () => {
    if (!lectureId) return;
    const res = await fetch(`${API_BASE}/lectures/${lectureId}/transcribe`, { method: "POST" });
    if (!res.ok) throw new Error("STT 실행 실패");
    const tRes = await fetch(`${API_BASE}/lectures/${lectureId}/transcript`);
    const tJson = await tRes.json();
    setTranscript(tJson.segments || []);
  };

  const runOcr = async () => {
    if (!lectureId) return;
    const res = await fetch(`${API_BASE}/lectures/${lectureId}/ocr_pdf`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || "OCR 실패");
    setOcrReady(true);
    setIndexReady(false);
  };

  const buildRagIndex = async () => {
    if (!lectureId) return;
    const res = await fetch(`${API_BASE}/lectures/${lectureId}/rag_index`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || "Index 실패");
    setIndexReady(true);
  };

  const generateSummary = async () => {
    if (!lectureId) return;
    const res = await fetch(`${API_BASE}/lectures/${lectureId}/summary`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || "요약 실패");
    setSummary(data.summary || "");
  };

  const generateQuiz = async () => {
    if (!lectureId) return;
    const res = await fetch(`${API_BASE}/lectures/${lectureId}/quiz`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || "퀴즈 실패");
    setQuiz(Array.isArray(data.quiz) ? data.quiz : []);
  };

  // ✅ 임베딩 기반 자동 싱크 (새로운 방식)
  const autoGenerateAnchors = async () => {
    if (!lectureId) return;

    try {
      // 임베딩 기반 자동 싱크 API 호출
      const res = await fetch(`${API_BASE}/lectures/${lectureId}/auto_sync`, {
        method: "POST",
      });
      
      const data = await res.json().catch(() => ({}));
      
      if (!res.ok) {
        // 실패 시 fallback: 균등 분할 방식
        console.warn("auto_sync 실패, fallback 사용:", data?.detail);
        await fallbackGenerateAnchors();
        return;
      }
      
      // 성공 시 앵커 새로고침
      await loadAnchors();
      setStatus(`✅ 임베딩 기반 싱크 완료 (${data.anchors_count}개 앵커 생성)`);
      
    } catch (e: any) {
      console.error("auto_sync 에러:", e);
      // 에러 시 fallback
      await fallbackGenerateAnchors();
    }
  };

  // Fallback: 균등 분할 방식 (기존 방식)
  const fallbackGenerateAnchors = async () => {
    if (!lectureId || !numPages || !audioDuration) return;

    const newAnchors: Anchor[] = [];
    const interval = audioDuration / numPages;
    for (let i = 1; i <= numPages; i++) {
      newAnchors.push({ page: i, time: (i - 1) * interval });
    }

    for (const a of newAnchors) {
      await fetch(`${API_BASE}/lectures/${lectureId}/anchors`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(a),
      });
    }
    await loadAnchors();
    setStatus("⚠️ Fallback 싱크 사용 (균등 분할)");
  };

  const runAgentPipeline = async () => {
    if (!lectureId) {
      setStatus("⚠️ 강의가 올바르지 않습니다");
      return;
    }
    setIsProcessing(true);
    try {
      setAgentThinking("🎤 음성을 텍스트로 변환 중 (Whisper STT)...");
      setStatus("AI Agent 작동 중...");
      await runTranscribe();

      setAgentThinking("📄 PDF에서 텍스트 추출 중 (OCR)...");
      await runOcr();

      setAgentThinking("🧠 지식 베이스 구축 중 (RAG Index)...");
      await buildRagIndex();

      setAgentThinking("🔗 정밀 싱크 생성 중...");
      await autoGenerateAnchors();

      setAgentThinking("📝 강의 요약 생성 중...");
      await generateSummary();

      setAgentThinking("❓ 퀴즈 문제 생성 중...");
      await generateQuiz();

      setAgentThinking("");
      setStatus("✅ AI Agent가 모든 작업을 완료했습니다!");
    } catch (e: any) {
      setStatus(`❌ 오류 발생: ${e?.message || "오류"}`);
      setAgentThinking("");
    } finally {
      setIsProcessing(false);
    }
  };

  const askRag = async () => {
    if (!lectureId || !currentMessage.trim()) {
      setStatus("질문을 입력하세요.");
      return;
    }
    const userMsg = { role: "user" as const, content: currentMessage };
    setChatHistory((prev) => [...prev, userMsg]);
    const question = currentMessage;
    setCurrentMessage("");
    setIsProcessing(true);

    try {
      const res = await fetch(`${API_BASE}/lectures/${lectureId}/rag_ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, topK: 5 }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Ask 실패");
      setCitations(data.citations || []);

      setChatHistory((prev) => [...prev, { role: "assistant", content: data.answer || "" }]);
      setStatus("답변 생성 완료");
    } catch {
      setChatHistory((prev) => [...prev, { role: "assistant", content: "답변 생성 중 오류가 발생했습니다." }]);
    } finally {
      setIsProcessing(false);
    }
  };

  // ✅ 컨닝페이퍼 생성
  const generateCheatSheet = async () => {
    setIsGeneratingCheatSheet(true);
    try {
      // 선택된 페이지들 (⭐ 표시 + 🔥 높은 페이지)
      const selectedPages = pageCards
        .filter(c => c.isStarred || c.emphasisScore > 0.5)
        .sort((a, b) => b.emphasisScore - a.emphasisScore);

      // TODO: 실제 API 호출로 대체
      // 임시로 로컬에서 생성
      let content = `# ${lectureTitle} - 컨닝페이퍼\n\n`;
      
      if (cheatSheetConfig.includeConceptSummary && summary) {
        content += `## 📝 핵심 개념 요약\n${summary}\n\n`;
      }
      
      if (cheatSheetConfig.includeEmphasis) {
        content += `## 🔥 교수 강조 포인트\n`;
        selectedPages.slice(0, 5).forEach(p => {
          content += `- **p.${p.page}** ${p.title}\n`;
        });
        content += "\n";
      }
      
      if (cheatSheetConfig.includeKeywords) {
        const allKeywords = [...new Set(selectedPages.flatMap(p => p.keywords))];
        content += `## 🏷️ 핵심 키워드\n${allKeywords.slice(0, 15).join(", ")}\n\n`;
      }
      
      if (cheatSheetConfig.includeQuestions && quiz.length > 0) {
        content += `## ❓ 예상 문제\n`;
        quiz.slice(0, 3).forEach((q, i) => {
          content += `${i + 1}. ${q.question}\n`;
          content += `   정답: ${String.fromCharCode(65 + q.answer)}\n\n`;
        });
      }

      setGeneratedCheatSheet(content);
    } catch (e: any) {
      setStatus(`컨닝페이퍼 생성 실패: ${e?.message || "오류"}`);
    } finally {
      setIsGeneratingCheatSheet(false);
    }
  };

  // 질문을 시험문제로 변환
  const convertToQuiz = async (question: string) => {
    // TODO: API 호출로 구현
    setStatus("질문을 시험문제로 변환 중...");
  };

  // 강조도에 따른 배지
  const getEmphasisBadge = (score: number) => {
    if (score > 0.7) return { label: "🔥 핵심", color: "bg-rose-500/20 text-rose-300 border-rose-500/50" };
    if (score > 0.4) return { label: "⚡ 중요", color: "bg-amber-500/20 text-amber-300 border-amber-500/50" };
    return null;
  };

  // 시간 포맷팅
  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // 오디오 컨트롤
  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(() => {});
    }
    setIsPlaying(!isPlaying);
  };

  const skipTime = (delta: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Math.min(audioDuration, currentTime + delta));
  };

  // ========== 렌더링 ==========

  // 업로드 UI (PDF/오디오 없을 때)
  if (!pdfBlobUrl || !audioUrl) {
    return (
      <div className="h-screen w-full flex flex-col bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900">
        <TopNav showBack title={lectureTitle} subtitle={lectureDesc} />
        
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-3xl space-y-6">
            <div className="text-center mb-8">
              <div className="bg-gradient-to-br from-indigo-600 to-violet-600 w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-2xl">
                <Upload className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-white mb-2">강의 자료 업로드</h2>
              <p className="text-slate-400">PDF 강의자료와 오디오 파일을 업로드하세요</p>
            </div>

            {/* PDF drop */}
            <div
              className={`relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 ${
                dragActive.pdf
                  ? "border-indigo-400 bg-indigo-950/50 scale-105"
                  : pdf
                  ? "border-emerald-500 bg-emerald-950/30"
                  : "border-slate-600 bg-slate-800/50 hover:border-indigo-500"
              }`}
              onDragEnter={(e) => handleDrag(e, "pdf")}
              onDragLeave={(e) => handleDrag(e, "pdf")}
              onDragOver={(e) => handleDrag(e, "pdf")}
              onDrop={(e) => handleDrop(e, "pdf")}
            >
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setPdf(e.target.files?.[0] || null)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={role === "student"}
              />
              <div className="text-center pointer-events-none">
                <FileText className={`w-16 h-16 mx-auto mb-4 ${pdf ? "text-emerald-400" : "text-slate-500"}`} />
                <div className="text-xl font-bold text-white mb-2">
                  {pdf ? "✓ PDF 선택됨" : "📄 PDF 강의자료"}
                </div>
                <div className="text-sm">
                  {pdf ? <span className="text-emerald-400">{pdf.name}</span> : <span className="text-slate-400">드래그 또는 클릭</span>}
                </div>
              </div>
            </div>

            {/* Audio drop */}
            <div
              className={`relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 ${
                dragActive.audio
                  ? "border-indigo-400 bg-indigo-950/50 scale-105"
                  : audio
                  ? "border-emerald-500 bg-emerald-950/30"
                  : "border-slate-600 bg-slate-800/50 hover:border-indigo-500"
              }`}
              onDragEnter={(e) => handleDrag(e, "audio")}
              onDragLeave={(e) => handleDrag(e, "audio")}
              onDragOver={(e) => handleDrag(e, "audio")}
              onDrop={(e) => handleDrop(e, "audio")}
            >
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setAudio(e.target.files?.[0] || null)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={role === "student"}
              />
              <div className="text-center pointer-events-none">
                <Volume2 className={`w-16 h-16 mx-auto mb-4 ${audio ? "text-emerald-400" : "text-slate-500"}`} />
                <div className="text-xl font-bold text-white mb-2">
                  {audio ? "✓ 오디오 선택됨" : "🎵 강의 오디오"}
                </div>
                <div className="text-sm">
                  {audio ? <span className="text-emerald-400">{audio.name}</span> : <span className="text-slate-400">드래그 또는 클릭</span>}
                </div>
              </div>
            </div>

            {role === "professor" && (pdf || audio) && (
              <Button
                onClick={uploadFiles}
                disabled={isProcessing || !pdf || !audio}
                className="w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white py-6 text-lg font-bold"
                size="lg"
              >
                {isProcessing ? (
                  <><Loader2 className="w-5 h-5 animate-spin mr-2" /> 업로드 중...</>
                ) : (
                  <><Upload className="w-5 h-5 mr-2" /> 파일 업로드하기</>
                )}
              </Button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // 메인 UI
  return (
    <div className="h-screen w-full flex flex-col bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900">
      <TopNav showBack title={lectureTitle} subtitle={lectureDesc} />

      {/* 상태 바 + 학습 모드 선택 */}
      <div className="px-4 pt-3 flex items-center gap-3">
        {/* 상태 표시 */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl px-4 py-2 flex items-center gap-2 text-sm text-slate-200">
          {isProcessing && <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />}
          {agentThinking && <Sparkles className="w-4 h-4 animate-pulse text-amber-400" />}
          <span className="font-medium truncate max-w-[200px]">{agentThinking || status}</span>
        </div>

        {/* 학습 모드 선택 */}
        <div className="flex-1 flex items-center justify-center gap-1 bg-slate-800/30 rounded-xl p-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setStudyMode("scan")}
            className={`h-9 px-4 rounded-lg transition-all ${
              studyMode === "scan"
                ? "bg-indigo-600 text-white shadow-lg"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
          >
            <Grid3X3 className="w-4 h-4 mr-2" />
            스캔
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setStudyMode("focus")}
            className={`h-9 px-4 rounded-lg transition-all ${
              studyMode === "focus"
                ? "bg-indigo-600 text-white shadow-lg"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
          >
            <BookOpen className="w-4 h-4 mr-2" />
            집중학습
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setStudyMode("exam")}
            className={`h-9 px-4 rounded-lg transition-all ${
              studyMode === "exam"
                ? "bg-indigo-600 text-white shadow-lg"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
          >
            <GraduationCap className="w-4 h-4 mr-2" />
            시험모드
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setStudyMode("chat")}
            className={`h-9 px-4 rounded-lg transition-all ${
              studyMode === "chat"
                ? "bg-indigo-600 text-white shadow-lg"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
          >
            <MessageCircle className="w-4 h-4 mr-2" />
            질의응답
          </Button>
        </div>

        {/* 교수 도구 토글 */}
        {role === "professor" && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => setShowProfessorTools(!showProfessorTools)}
            className={`h-9 border-slate-600 ${showProfessorTools ? "bg-slate-700 text-white" : "text-slate-400"}`}
          >
            <Settings className="w-4 h-4 mr-1" />
            관리
          </Button>
        )}
      </div>

      {/* 메인 레이아웃 */}
      <div className="flex-1 flex min-h-0 mt-3">
        {/* ========== 스캔 모드 ========== */}
        {studyMode === "scan" && (
          <div className="flex-1 flex flex-col p-4">
            {/* 검색/필터 바 */}
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <Input
                  placeholder="키워드로 검색..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500"
                />
              </div>
              <div className="flex items-center gap-1 bg-slate-800/50 rounded-lg p-1">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSortBy("page")}
                  className={`h-8 px-3 ${sortBy === "page" ? "bg-slate-700 text-white" : "text-slate-400"}`}
                >
                  페이지순
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSortBy("emphasis")}
                  className={`h-8 px-3 ${sortBy === "emphasis" ? "bg-slate-700 text-white" : "text-slate-400"}`}
                >
                  <Flame className="w-3 h-3 mr-1" />
                  강조순
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSortBy("starred")}
                  className={`h-8 px-3 ${sortBy === "starred" ? "bg-slate-700 text-white" : "text-slate-400"}`}
                >
                  <Star className="w-3 h-3 mr-1" />
                  내 표시
                </Button>
              </div>
            </div>

            {/* 카드 리스트 */}
            <ScrollArea className="flex-1">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 pr-4">
                {filteredCards.map((card) => {
                  const badge = getEmphasisBadge(card.emphasisScore);
                  const isActive = viewPage === card.page;

                  return (
                    <Card
                      key={card.page}
                      className={`cursor-pointer transition-all duration-200 hover:scale-[1.02] ${
                        isActive
                          ? "ring-2 ring-indigo-500 bg-slate-800"
                          : "bg-slate-800/50 hover:bg-slate-800 border-slate-700"
                      }`}
                    >
                      <CardContent className="p-4">
                        {/* 헤더: 페이지 + 배지 + ⭐ */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-lg font-bold text-white">p.{card.page}</span>
                            {badge && (
                              <span className={`text-xs px-2 py-0.5 rounded-full border ${badge.color}`}>
                                {badge.label}
                              </span>
                            )}
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleStar(card.page);
                            }}
                            className={`h-8 w-8 p-0 ${card.isStarred ? "text-amber-400" : "text-slate-500 hover:text-amber-400"}`}
                          >
                            <Star className={`w-5 h-5 ${card.isStarred ? "fill-current" : ""}`} />
                          </Button>
                        </div>

                        {/* 제목 */}
                        <h3 className="text-sm font-medium text-white mb-2 line-clamp-2">{card.title}</h3>

                        {/* 키워드 */}
                        <div className="flex flex-wrap gap-1 mb-3">
                          {card.keywords.slice(0, 4).map((kw, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 bg-slate-700 text-slate-300 rounded">
                              {kw}
                            </span>
                          ))}
                        </div>

                        {/* 재생 버튼 */}
                        <div
                          onClick={() => handleCardClick(card)}
                          className="flex items-center justify-between p-2 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 transition-colors"
                        >
                          <div className="flex items-center gap-2 text-indigo-300">
                            <PlayCircle className="w-5 h-5" />
                            <span className="text-sm font-medium">
                              {formatDuration(card.startTime)} ~ {formatDuration(card.endTime)}
                            </span>
                          </div>
                          <ChevronRight className="w-4 h-4 text-indigo-400" />
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </ScrollArea>
          </div>
        )}

        {/* ========== 집중학습 모드 ========== */}
        {studyMode === "focus" && (
          <div className="flex-1 flex min-h-0">
            {/* 왼쪽: PDF 영역 */}
            <div className="flex-1 flex flex-col min-h-0">
              {/* PDF 뷰어 */}
              <div className="flex-1 overflow-auto p-4">
                <div className="flex justify-center">
                  <div className="bg-white shadow-2xl rounded-lg overflow-hidden">
                    <PdfViewer file={pdfBlobUrl} page={viewPage} onLoad={(n) => setNumPages(n)} width={650} />
                  </div>
                </div>
              </div>

              {/* 페이지 네비게이션 */}
              <div className="border-t border-slate-700 bg-slate-800/50 p-3 flex items-center justify-center gap-3">
                {/* 이전 버튼 */}
                <Button
                  size="sm"
                  onClick={() => goToPageAndPlay(viewPage - 1)}
                  disabled={viewPage <= 1}
                  className="bg-slate-700 hover:bg-slate-600"
                >
                  ←
                </Button>


                {/* 페이지 입력 + 재동기화 */}
                <div className="flex items-center gap-2 bg-slate-700/50 px-3 py-1.5 rounded-lg relative">
                  <Input
                    type="number"
                    min={1}
                    max={numPages}
                    value={viewPage}
                    onChange={(e) => {
                      const p = Number(e.target.value);
                      if (p >= 1 && p <= numPages) setViewPage(p);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        goToPageAndPlay(viewPage);
                      }
                    }}
                    className="w-16 text-center border-0 bg-transparent text-white font-semibold"
                  />

                  <span className="text-slate-400">/ {numPages}</span>

                  {!isAutoSync && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="absolute -right-8 top-1/2 -translate-y-1/2 w-6 h-6 p-0 text-slate-400 hover:text-emerald-400 animate-pulse"
                      onClick={() => setIsAutoSync(true)}
                      title="오디오 싱크 다시 켜기"
                    >
                      <RefreshCw className="w-3 h-3" />
                    </Button>
                  )}
                </div>

                {/* 다음 버튼 */}
                <Button
                  size="sm"
                  onClick={() => goToPageAndPlay(viewPage + 1)}
                  disabled={viewPage >= numPages}
                  className="bg-slate-700 hover:bg-slate-600"
                >
                  →
                </Button>


                {/* 이 페이지 재생 */}
                <Button
                  size="sm"
                  onClick={() => goToPageAndPlay(viewPage)}
                  disabled={!canPageJump}
                  className="bg-indigo-600 hover:bg-indigo-700 ml-2"
                >
                  <Play className="w-4 h-4 mr-1" />
                  이 페이지 재생
                </Button>

              </div>
            </div>

            {/* 오른쪽: 페이지 정보 패널 */}
            <div className="w-[380px] border-l border-slate-700 bg-slate-800/30 flex flex-col min-h-0">
              <ScrollArea className="flex-1 p-4">
                {currentPageCard && (
                  <div className="space-y-4">
                    {/* 페이지 헤더 */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xl font-bold text-white">p.{currentPageCard.page}</span>
                        {getEmphasisBadge(currentPageCard.emphasisScore) && (
                          <span
                            className={`text-xs px-2 py-0.5 rounded-full border ${
                              getEmphasisBadge(currentPageCard.emphasisScore)!.color
                            }`}
                          >
                            {getEmphasisBadge(currentPageCard.emphasisScore)!.label}
                          </span>
                        )}
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => toggleStar(currentPageCard.page)}
                        className={currentPageCard.isStarred ? "text-amber-400" : "text-slate-500"}
                      >
                        <Star className={`w-5 h-5 ${currentPageCard.isStarred ? "fill-current" : ""}`} />
                      </Button>
                    </div>

                    {/* 전체 강의 스크립트 (Live Sync) */}
                    <Card className="bg-slate-800/50 border-slate-700 flex flex-col h-[400px]">
                      <CardHeader className="p-3 pb-2 shrink-0">
                        <div className="text-sm font-semibold text-white flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <MessageCircle className="w-4 h-4 text-emerald-400" />
                            전체 강의 스크립트
                          </div>
                          <Badge
                            variant="outline"
                            className="text-[10px] border-emerald-500/50 text-emerald-400 bg-emerald-500/10"
                          >
                            Live Sync
                          </Badge>
                        </div>
                      </CardHeader>

                      <CardContent className="p-0 flex-1 min-h-0 relative">
                        <ScrollArea className="h-full w-full">
                          <div className="p-3 space-y-2">
                            {transcript.length > 0 ? (
                              transcript.map((seg, i) => {
                                const isActive = activeTranscriptIndex === i;
                                return (
                                  <div
                                    key={i}
                                    id={`transcript-seg-${i}`}
                                    onClick={() => jumpTo(seg.start)}
                                    className={`p-3 rounded-xl transition-all duration-300 cursor-pointer border ${
                                      isActive
                                        ? "bg-indigo-600/30 border-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.2)]"
                                        : "bg-slate-700/30 border-transparent hover:bg-slate-700/50"
                                    }`}
                                  >
                                    <div className="flex items-center justify-between mb-1">
                                      <span className={`text-[10px] font-mono ${isActive ? "text-indigo-300" : "text-slate-500"}`}>
                                        ⏱️ {formatDuration(seg.start)}
                                      </span>
                                      {isActive && (
                                        <div className="flex items-center gap-1">
                                          <span className="text-[10px] text-indigo-300 animate-pulse">Now Playing</span>
                                          <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-ping" />
                                        </div>
                                      )}
                                    </div>
                                    <p className={`text-sm leading-relaxed ${isActive ? "text-white font-medium" : "text-slate-300"}`}>
                                      {seg.text}
                                    </p>
                                  </div>
                                );
                              })
                            ) : (
                              <div className="text-center text-slate-500 py-10 text-sm">
                                스크립트가 없습니다. <br /> AI Agent를 실행하여 STT를 진행해주세요.
                              </div>
                            )}
                          </div>
                        </ScrollArea>
                      </CardContent>
                    </Card>

                    {/* 핵심 키워드 */}
                    <Card className="bg-slate-800/50 border-slate-700">
                      <CardHeader className="p-3 pb-2">
                        <div className="text-sm font-semibold text-white flex items-center gap-2">
                          <Brain className="w-4 h-4 text-amber-400" />
                          핵심 키워드
                        </div>
                      </CardHeader>
                      <CardContent className="p-3 pt-0">
                        <div className="flex flex-wrap gap-2">
                          {currentPageCard.keywords.map((kw, i) => (
                            <span key={i} className="text-sm px-3 py-1 bg-slate-700 text-slate-200 rounded-full">
                              {kw}
                            </span>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </ScrollArea>
            </div>
          </div>
        )}

        {/* ========== 시험 모드 ========== */}
        {studyMode === "exam" && (
          <div className="flex-1 flex min-h-0 p-4 gap-4">
            {/* 설정 패널 */}
            <div className="w-[320px] space-y-4">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="p-4 pb-2">
                  <div className="font-bold text-white flex items-center gap-2">
                    <GraduationCap className="w-5 h-5 text-indigo-400" />
                    컨닝페이퍼 빌더
                  </div>
                </CardHeader>
                <CardContent className="p-4 pt-2 space-y-4">
                  {/* 압축 강도 */}
                  <div>
                    <div className="text-sm text-slate-300 mb-2">압축 강도</div>
                    <div className="flex gap-2">
                      {[1, 2, 3].map(n => (
                        <Button
                          key={n}
                          size="sm"
                          variant={cheatSheetConfig.pageCount === n ? "default" : "outline"}
                          onClick={() => setCheatSheetConfig(c => ({ ...c, pageCount: n }))}
                          className={cheatSheetConfig.pageCount === n 
                            ? "bg-indigo-600 flex-1" 
                            : "border-slate-600 text-slate-300 flex-1"}
                        >
                          {n}페이지
                        </Button>
                      ))}
                    </div>
                  </div>

                  {/* 포함 요소 */}
                  <div className="space-y-3">
                    <div className="text-sm text-slate-300">포함 요소</div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">핵심 개념 요약</span>
                      <Switch
                        checked={cheatSheetConfig.includeConceptSummary}
                        onCheckedChange={(v) => setCheatSheetConfig(c => ({ ...c, includeConceptSummary: v }))}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">공식/정의</span>
                      <Switch
                        checked={cheatSheetConfig.includeFormulas}
                        onCheckedChange={(v) => setCheatSheetConfig(c => ({ ...c, includeFormulas: v }))}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">교수 강조 포인트</span>
                      <Switch
                        checked={cheatSheetConfig.includeEmphasis}
                        onCheckedChange={(v) => setCheatSheetConfig(c => ({ ...c, includeEmphasis: v }))}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">핵심 키워드</span>
                      <Switch
                        checked={cheatSheetConfig.includeKeywords}
                        onCheckedChange={(v) => setCheatSheetConfig(c => ({ ...c, includeKeywords: v }))}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">예상 문제</span>
                      <Switch
                        checked={cheatSheetConfig.includeQuestions}
                        onCheckedChange={(v) => setCheatSheetConfig(c => ({ ...c, includeQuestions: v }))}
                      />
                    </div>
                  </div>

                  {/* 선택된 페이지 표시 */}
                  <div className="p-3 rounded-lg bg-slate-700/50">
                    <div className="text-xs text-slate-400 mb-2">선택된 범위</div>
                    <div className="text-sm text-white">
                      ⭐ 내가 표시한 페이지: {pageCards.filter(c => c.isStarred).length}개
                    </div>
                    <div className="text-sm text-white">
                      🔥 교수 강조 페이지: {pageCards.filter(c => c.emphasisScore > 0.5).length}개
                    </div>
                  </div>

                  {/* 생성 버튼 */}
                  <Button
                    onClick={generateCheatSheet}
                    disabled={isGeneratingCheatSheet}
                    className="w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700"
                  >
                    {isGeneratingCheatSheet ? (
                      <><Loader2 className="w-4 h-4 animate-spin mr-2" /> 생성 중...</>
                    ) : (
                      <><Sparkles className="w-4 h-4 mr-2" /> 컨닝페이퍼 생성</>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* 예상문제 */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="p-4 pb-2">
                  <div className="font-bold text-white flex items-center gap-2">
                    <ListChecks className="w-5 h-5 text-amber-400" />
                    예상 문제 ({quiz.length}개)
                  </div>
                </CardHeader>
                <CardContent className="p-4 pt-2">
                  {quiz.length > 0 ? (
                    <ScrollArea className="h-[200px]">
                      <div className="space-y-2">
                        {quiz.map((q, i) => (
                          <div key={i} className="p-2 rounded-lg bg-slate-700/50 text-sm text-slate-300">
                            Q{i + 1}. {q.question.slice(0, 50)}...
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <div className="text-sm text-slate-500 text-center py-4">
                      AI Agent를 실행하면 예상문제가 생성됩니다
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* 컨닝페이퍼 결과 */}
            <div className="flex-1">
              <Card className="h-full bg-slate-800/50 border-slate-700 flex flex-col">
                <CardHeader className="p-4 pb-2 flex-row items-center justify-between">
                  <div className="font-bold text-white flex items-center gap-2">
                    <FileText className="w-5 h-5 text-emerald-400" />
                    컨닝페이퍼 결과
                  </div>
                  {generatedCheatSheet && (
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-slate-600 text-slate-300 hover:bg-slate-700"
                        onClick={() => {
                          navigator.clipboard.writeText(generatedCheatSheet);
                          setStatus("클립보드에 복사됨!");
                        }}
                      >
                        <Copy className="w-4 h-4 mr-1" />
                        복사
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-slate-600 text-slate-300 hover:bg-slate-700"
                        onClick={() => {
                          const blob = new Blob([generatedCheatSheet], { type: "text/markdown" });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement("a");
                          a.href = url;
                          a.download = `${lectureTitle}_컨닝페이퍼.md`;
                          a.click();
                          URL.revokeObjectURL(url);
                        }}
                      >
                        <FileDown className="w-4 h-4 mr-1" />
                        저장
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-slate-600 text-slate-300 hover:bg-slate-700"
                        onClick={() => window.print()}
                      >
                        <Printer className="w-4 h-4 mr-1" />
                        프린트
                      </Button>
                    </div>
                  )}
                </CardHeader>
                <CardContent className="flex-1 p-4 pt-2 overflow-hidden">
                  {generatedCheatSheet ? (
                    <ScrollArea className="h-full">
                      <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
                        {generatedCheatSheet}
                      </div>
                    </ScrollArea>
                  ) : (
                    <div className="h-full flex items-center justify-center text-slate-500">
                      <div className="text-center">
                        <FileText className="w-16 h-16 mx-auto mb-4 opacity-30" />
                        <p>설정을 선택하고 "컨닝페이퍼 생성" 버튼을 누르세요</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* ========== 질의응답 모드 ========== */}
        {studyMode === "chat" && (
          <div className="flex-1 flex min-h-0 p-4 gap-4">
            {/* 채팅 영역 */}
            <div className="flex-1 flex flex-col">
              <Card className="flex-1 bg-slate-800/50 border-slate-700 flex flex-col min-h-0">
                <CardHeader className="p-4 pb-2">
                  <div className="font-bold text-white flex items-center gap-2">
                    <MessageCircle className="w-5 h-5 text-indigo-400" />
                    강의 내용 질의응답
                  </div>
                </CardHeader>
                <CardContent className="flex-1 p-4 pt-2 flex flex-col min-h-0">
                  <ScrollArea className="flex-1 mb-4">
                    <div className="space-y-3">
                      {chatHistory.length === 0 && (
                        <div className="text-center text-slate-500 py-8">
                          강의 내용에 대해 질문해보세요.<br />
                          근거 페이지와 함께 답변해드립니다.
                        </div>
                      )}
                      {chatHistory.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                          <div
                            className={`max-w-[80%] p-3 rounded-xl ${
                              msg.role === "user"
                                ? "bg-indigo-600 text-white"
                                : "bg-slate-700 text-slate-200"
                            }`}
                          >
                            <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                            
                            {/* 질문을 시험문제로 변환 버튼 (사용자 메시지에만) */}
                            {msg.role === "user" && (
                              <Button
                                size="sm"
                                variant="ghost"
                                className="mt-2 h-7 text-xs text-indigo-300 hover:text-white hover:bg-indigo-700"
                                onClick={() => convertToQuiz(msg.content)}
                              >
                                <FileQuestion className="w-3 h-3 mr-1" />
                                시험문제로 변환
                              </Button>
                            )}
                          </div>
                        </div>
                      ))}

                      {/* 근거 카드 */}
                      {citations.length > 0 && (
                        <div className="space-y-2 mt-4">
                          <div className="text-xs text-slate-400 font-semibold">📚 근거 페이지</div>
                          {citations.map((c) => (
                            <div key={c.chunkId} className="p-3 rounded-lg bg-slate-700/50 border border-slate-600">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-sm font-bold text-white">📄 p.{c.page}</span>
                                <Button
                                  size="sm"
                                  onClick={() => {
                                    setViewPage(Number(c.page));
                                    setStudyMode("focus");
                                    jumpToPage(Number(c.page));
                                  }}
                                  className="h-7 text-xs bg-indigo-600 hover:bg-indigo-700"
                                >
                                  이동 <ArrowRight className="w-3 h-3 ml-1" />
                                </Button>
                              </div>
                              <p className="text-xs text-slate-400 line-clamp-2">{c.snippet}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </ScrollArea>

                  {/* 입력 */}
                  <div className="space-y-2">
                    <Textarea
                      value={currentMessage}
                      onChange={(e) => setCurrentMessage(e.target.value)}
                      placeholder="강의 내용에 대해 질문하세요..."
                      rows={3}
                      className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 resize-none"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          askRag();
                        }
                      }}
                    />
                    <Button
                      onClick={askRag}
                      disabled={isProcessing || !currentMessage.trim() || !indexReady}
                      className="w-full bg-indigo-600 hover:bg-indigo-700"
                    >
                      {isProcessing ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <MessageCircle className="w-4 h-4 mr-2" />}
                      질문하기
                    </Button>
                    {!indexReady && (
                      <p className="text-xs text-amber-400 text-center">
                        AI Agent를 먼저 실행해주세요 (RAG 인덱스 필요)
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* ========== 교수 도구 사이드바 (단계별 실행) ========== */}
        {showProfessorTools && role === "professor" && (
          <div className="w-[350px] border-l border-slate-700 bg-slate-800/30 p-4 space-y-4 overflow-auto">
            <div className="font-bold text-white flex items-center gap-2 mb-4">
              <Settings className="w-5 h-5" />
              관리 도구
            </div>

            {/* 파이프라인 단계별 실행 */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader className="p-3 pb-2">
                <div className="text-sm font-semibold text-white flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  AI 파이프라인
                </div>
              </CardHeader>
              <CardContent className="p-3 pt-0 space-y-2">
                {/* 1. STT */}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      setIsProcessing(true);
                      setAgentThinking("🎤 음성 → 텍스트 변환 중...");
                      try {
                        await runTranscribe();
                        setStatus("✅ STT 완료");
                      } catch (e: any) {
                        setStatus(`❌ STT 실패: ${e?.message}`);
                      } finally {
                        setIsProcessing(false);
                        setAgentThinking("");
                      }
                    }}
                    disabled={isProcessing || !audioUrl}
                    size="sm"
                    className="flex-1 bg-slate-700 hover:bg-slate-600 justify-start"
                  >
                    <span className="w-5 h-5 rounded-full bg-indigo-600 text-xs flex items-center justify-center mr-2">1</span>
                    🎤 STT (음성→텍스트)
                  </Button>
                  {transcript.length > 0 && (
                    <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/50">✓</Badge>
                  )}
                </div>

                {/* 2. OCR */}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      setIsProcessing(true);
                      setAgentThinking("📄 PDF 텍스트 추출 중...");
                      try {
                        await runOcr();
                        setStatus("✅ OCR 완료");
                      } catch (e: any) {
                        setStatus(`❌ OCR 실패: ${e?.message}`);
                      } finally {
                        setIsProcessing(false);
                        setAgentThinking("");
                      }
                    }}
                    disabled={isProcessing || !pdfUrl}
                    size="sm"
                    className="flex-1 bg-slate-700 hover:bg-slate-600 justify-start"
                  >
                    <span className="w-5 h-5 rounded-full bg-indigo-600 text-xs flex items-center justify-center mr-2">2</span>
                    📄 OCR (PDF→텍스트)
                  </Button>
                  {ocrReady && (
                    <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/50">✓</Badge>
                  )}
                </div>

                {/* 3. RAG Index */}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      setIsProcessing(true);
                      setAgentThinking("🧠 지식 베이스 구축 중...");
                      try {
                        await buildRagIndex();
                        setStatus("✅ RAG Index 완료");
                      } catch (e: any) {
                        setStatus(`❌ RAG Index 실패: ${e?.message}`);
                      } finally {
                        setIsProcessing(false);
                        setAgentThinking("");
                      }
                    }}
                    disabled={isProcessing || !ocrReady}
                    size="sm"
                    className="flex-1 bg-slate-700 hover:bg-slate-600 justify-start"
                  >
                    <span className="w-5 h-5 rounded-full bg-indigo-600 text-xs flex items-center justify-center mr-2">3</span>
                    🧠 RAG Index (임베딩)
                  </Button>
                  {indexReady && (
                    <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/50">✓</Badge>
                  )}
                </div>

                {/* 4. Auto Sync */}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      setIsProcessing(true);
                      setAgentThinking("🔗 페이지-오디오 싱크 중...");
                      try {
                        await autoGenerateAnchors();
                        setStatus("✅ 자동 싱크 완료");
                      } catch (e: any) {
                        setStatus(`❌ 자동 싱크 실패: ${e?.message}`);
                      } finally {
                        setIsProcessing(false);
                        setAgentThinking("");
                      }
                    }}
                    disabled={isProcessing || !ocrReady || transcript.length === 0}
                    size="sm"
                    className="flex-1 bg-slate-700 hover:bg-slate-600 justify-start"
                  >
                    <span className="w-5 h-5 rounded-full bg-indigo-600 text-xs flex items-center justify-center mr-2">4</span>
                    🔗 Auto Sync (페이지↔오디오)
                  </Button>
                  {anchors.length > 0 && (
                    <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/50">✓ {anchors.length}</Badge>
                  )}
                </div>

                {/* 5. Summary */}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      setIsProcessing(true);
                      setAgentThinking("📝 요약 생성 중...");
                      try {
                        await generateSummary();
                        setStatus("✅ 요약 완료");
                      } catch (e: any) {
                        setStatus(`❌ 요약 실패: ${e?.message}`);
                      } finally {
                        setIsProcessing(false);
                        setAgentThinking("");
                      }
                    }}
                    disabled={isProcessing || !ocrReady}
                    size="sm"
                    className="flex-1 bg-slate-700 hover:bg-slate-600 justify-start"
                  >
                    <span className="w-5 h-5 rounded-full bg-indigo-600 text-xs flex items-center justify-center mr-2">5</span>
                    📝 Summary (요약)
                  </Button>
                  {summary && (
                    <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/50">✓</Badge>
                  )}
                </div>

                {/* 6. Quiz */}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      setIsProcessing(true);
                      setAgentThinking("❓ 퀴즈 생성 중...");
                      try {
                        await generateQuiz();
                        setStatus("✅ 퀴즈 완료");
                      } catch (e: any) {
                        setStatus(`❌ 퀴즈 실패: ${e?.message}`);
                      } finally {
                        setIsProcessing(false);
                        setAgentThinking("");
                      }
                    }}
                    disabled={isProcessing || !ocrReady}
                    size="sm"
                    className="flex-1 bg-slate-700 hover:bg-slate-600 justify-start"
                  >
                    <span className="w-5 h-5 rounded-full bg-indigo-600 text-xs flex items-center justify-center mr-2">6</span>
                    ❓ Quiz (퀴즈 생성)
                  </Button>
                  {quiz.length > 0 && (
                    <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/50">✓ {quiz.length}</Badge>
                  )}
                </div>

                {/* 구분선 */}
                <div className="border-t border-slate-600 my-3" />

                {/* 전체 실행 버튼 */}
                <Button
                  onClick={runAgentPipeline}
                  disabled={isProcessing}
                  className="w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700"
                >
                  {isProcessing ? (
                    <><Loader2 className="w-4 h-4 animate-spin mr-2" /> 작동 중...</>
                  ) : (
                    <><PlayCircle className="w-4 h-4 mr-2" /> 전체 파이프라인 실행</>
                  )}
                </Button>

                {/* 상태 표시 */}
                {agentThinking && (
                  <div className="text-xs text-amber-400 text-center animate-pulse">
                    {agentThinking}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 앵커 관리 */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader className="p-3 pb-2">
                <div className="text-sm font-semibold text-white">🎯 싱크 앵커</div>
              </CardHeader>
              <CardContent className="p-3 pt-0 space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <div className="text-xs text-slate-400 mb-1">페이지</div>
                    <Input
                      type="number"
                      min={1}
                      max={numPages}
                      value={anchorPage}
                      onChange={(e) => setAnchorPage(Number(e.target.value) || 1)}
                      className="h-8 bg-slate-700/50 border-slate-600 text-white"
                    />
                  </div>
                  <div>
                    <div className="text-xs text-slate-400 mb-1">현재 시간</div>
                    <div className="h-8 px-2 flex items-center bg-slate-700/50 rounded text-sm text-white font-mono">
                      {currentTime.toFixed(1)}s
                    </div>
                  </div>
                </div>
                <Button onClick={saveAnchor} disabled={!canPageJump} className="w-full bg-slate-700 hover:bg-slate-600" size="sm">
                  ⚓ 앵커 저장
                </Button>
                
                {/* 저장된 앵커 목록 */}
                {anchors.length > 0 && (
                  <div className="mt-2 space-y-1 max-h-32 overflow-auto">
                    {anchors.sort((a, b) => a.page - b.page).map((a, i) => (
                      <div key={i} className="flex items-center justify-between text-xs p-1.5 rounded bg-slate-700/30">
                        <span className="text-slate-300">p.{a.page} → {a.time.toFixed(1)}s</span>
                        <Button size="sm" variant="ghost" onClick={() => jumpTo(a.time)} className="h-6 px-2 text-xs">
                          ▶
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* ✅ 유사도 히트맵 */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader className="p-3 pb-2">
                <div className="text-sm font-semibold text-white flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    📊 유사도 히트맵
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      if (!heatmapData) loadHeatmapData();
                      setShowHeatmap(!showHeatmap);
                    }}
                    className="h-6 px-2 text-xs"
                  >
                    {showHeatmap ? "숨기기" : "보기"}
                  </Button>
                </div>
              </CardHeader>
              {showHeatmap && (
                <CardContent className="p-3 pt-0">
                  {isLoadingHeatmap ? (
                    <div className="flex items-center justify-center py-4">
                      <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
                    </div>
                  ) : heatmapData && heatmapData.matrix.length > 0 ? (
                    <div className="space-y-2">
                      <div className="text-xs text-slate-400">
                        {heatmapData.num_pages}페이지 × {heatmapData.num_segments}구간
                      </div>
                      
                      {/* 히트맵 그리드 */}
                      <div className="overflow-auto max-h-64">
                        <div className="inline-block">
                          {/* 헤더: 시간 구간 */}
                          <div className="flex">
                            <div className="w-8 h-6 shrink-0" /> {/* 코너 */}
                            {heatmapData.segment_times.map((seg, i) => (
                              <div
                                key={i}
                                className="w-6 h-6 shrink-0 text-[8px] text-slate-500 flex items-center justify-center"
                                title={`${Math.floor(seg.start / 60)}:${Math.floor(seg.start % 60).toString().padStart(2, '0')}`}
                              >
                                {i + 1}
                              </div>
                            ))}
                          </div>
                          
                          {/* 행: 페이지별 유사도 */}
                          {heatmapData.matrix.map((row, pageIdx) => (
                            <div key={pageIdx} className="flex">
                              {/* 페이지 번호 */}
                              <div className="w-8 h-6 shrink-0 text-[10px] text-slate-400 flex items-center justify-center">
                                p.{pageIdx + 1}
                              </div>
                              
                              {/* 유사도 셀 */}
                              {row.map((sim, segIdx) => {
                                const isMatched = heatmapData.matched_path.some(
                                  ([p, s]) => p === pageIdx + 1 && s === segIdx
                                );
                                const intensity = Math.min(1, sim);
                                const bgColor = isMatched
                                  ? `rgba(34, 197, 94, ${0.3 + intensity * 0.7})` // 초록 (매칭됨)
                                  : `rgba(99, 102, 241, ${intensity})`; // 인디고 (유사도)
                                
                                return (
                                  <div
                                    key={segIdx}
                                    className={`w-6 h-6 shrink-0 border border-slate-700/50 cursor-pointer hover:ring-1 hover:ring-white/50 ${
                                      isMatched ? "ring-1 ring-emerald-400" : ""
                                    }`}
                                    style={{ backgroundColor: bgColor }}
                                    title={`p.${pageIdx + 1} → 구간${segIdx + 1}: ${(sim * 100).toFixed(1)}%${isMatched ? " ✓매칭" : ""}`}
                                    onClick={() => {
                                      // 클릭 시 해당 페이지와 시간으로 이동
                                      setViewPage(pageIdx + 1);
                                      jumpTo(heatmapData.segment_times[segIdx].start);
                                    }}
                                  />
                                );
                              })}
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      {/* 범례 */}
                      <div className="flex items-center gap-3 text-[10px] text-slate-400 mt-2">
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 bg-emerald-500 rounded-sm ring-1 ring-emerald-400" />
                          <span>매칭됨</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 bg-indigo-500 rounded-sm" />
                          <span>높은 유사도</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 bg-indigo-900 rounded-sm" />
                          <span>낮은 유사도</span>
                        </div>
                      </div>
                      
                      {/* 새로고침 버튼 */}
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={loadHeatmapData}
                        className="w-full mt-2 border-slate-600 text-slate-300 hover:bg-slate-700"
                      >
                        <RefreshCw className="w-3 h-3 mr-1" />
                        새로고침
                      </Button>
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500 text-center py-4">
                      Auto Sync를 실행하면 유사도 데이터가 생성됩니다
                    </div>
                  )}
                </CardContent>
              )}
            </Card>
          </div>
        )}
      </div>

      {/* ========== 하단 오디오 플레이어 ========== */}
      {audioUrl && (
        <div className="border-t border-slate-700 bg-slate-800/80 backdrop-blur-sm px-4 py-3">
          <div className="flex items-center gap-4">
            {/* 컨트롤 */}
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" onClick={() => skipTime(-10)} className="text-slate-400 hover:text-white">
                <SkipBack className="w-4 h-4" />
              </Button>
              <Button
                size="sm"
                onClick={togglePlay}
                className="w-10 h-10 rounded-full bg-indigo-600 hover:bg-indigo-700"
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => skipTime(10)} className="text-slate-400 hover:text-white">
                <SkipForward className="w-4 h-4" />
              </Button>
            </div>

            {/* 시간 */}
            <span className="text-sm text-slate-400 font-mono w-20">
              {formatDuration(currentTime)}
            </span>

            {/* 프로그레스 바 */}
            <div className="flex-1">
              <Slider
                value={[currentTime]}
                max={audioDuration || 100}
                step={0.1}
                onValueChange={([v]) => {
                  if (audioRef.current) audioRef.current.currentTime = v;
                }}
                className="cursor-pointer"
              />
            </div>

            {/* 총 시간 */}
            <span className="text-sm text-slate-400 font-mono w-20 text-right">
              {formatDuration(audioDuration)}
            </span>

            {/* 숨겨진 오디오 요소 */}
            <audio
              ref={audioRef}
              src={audioUrl}
              onLoadedMetadata={(e) => setAudioDuration((e.target as HTMLAudioElement).duration || 0)}
              onTimeUpdate={(e) => setCurrentTime((e.target as HTMLAudioElement).currentTime)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              className="hidden"
            />
          </div>
        </div>
      )}
    </div>
  )
};