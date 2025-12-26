"use client";

import React, { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Upload, BookOpen, MessageSquare, Brain, PlayCircle, FileText, Sparkles, GraduationCap, CheckCircle } from "lucide-react";
import dynamic from "next/dynamic";

const PdfViewer = dynamic(() => import("../components/PdfViewer"), { ssr: false });

const API_BASE = "http://127.0.0.1:8000";

export default function Page() {
  const [userRole, setUserRole] = useState("professor");
  const [lectures, setLectures] = useState([]);
  const [selectedLecture, setSelectedLecture] = useState(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [pdf, setPdf] = useState(null);
  const [audio, setAudio] = useState(null);
  const [pdfUrl, setPdfUrl] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [status, setStatus] = useState("AI Agent 준비 완료 ✨");
  const [isProcessing, setIsProcessing] = useState(false);
  const [agentThinking, setAgentThinking] = useState("");
  const [summary, setSummary] = useState("");
  const [quiz, setQuiz] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [transcript, setTranscript] = useState([]);
  const audioRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [numPages, setNumPages] = useState(0);
  const [anchors, setAnchors] = useState([]);
  const [anchorPage, setAnchorPage] = useState(1);
  const [ocrReady, setOcrReady] = useState(false);
  const [indexReady, setIndexReady] = useState(false);
  const [citations, setCitations] = useState([]);
  const [pdfBlobUrl, setPdfBlobUrl] = useState("");
  const [viewPage, setViewPage] = useState(1);

  const canPageJump = audioDuration > 0 && numPages > 0;
  const prevPdfBlobUrlRef = useRef("");

  // Drag and drop states
  const [dragActive, setDragActive] = useState({ pdf: false, audio: false });

  const handleDrag = (e: React.DragEvent, type: 'pdf' | 'audio') => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(prev => ({ ...prev, [type]: true }));
    } else if (e.type === "dragleave") {
      setDragActive(prev => ({ ...prev, [type]: false }));
    }
  };

  const handleDrop = (e: React.DragEvent, type: 'pdf' | 'audio') => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(prev => ({ ...prev, [type]: false }));
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (type === 'pdf' && file.type === 'application/pdf') {
        setPdf(file);
      } else if (type === 'audio' && file.type.startsWith('audio/')) {
        setAudio(file);
      }
    }
  };

  useEffect(() => {
    loadLectures();
  }, []);
  // ============================
  // Lecture-scoped UI state cache (localStorage)
  //  - 백엔드에 summary/quiz/status GET이 아직 없어서
  //    (1) 단계에서는 프론트에서 강의별로 상태를 복구하도록 처리
  // ============================
  const lsKey = (lectureId: number | string, name: string) => `lecture:${lectureId}:${name}`;

  const saveLectureCache = (lectureId: number) => {
    try {
      localStorage.setItem(lsKey(lectureId, "summary"), summary || "");
      localStorage.setItem(lsKey(lectureId, "quiz"), JSON.stringify(quiz || []));
      localStorage.setItem(lsKey(lectureId, "chat"), JSON.stringify(chatHistory || []));
      localStorage.setItem(lsKey(lectureId, "ocrReady"), JSON.stringify(!!ocrReady));
      localStorage.setItem(lsKey(lectureId, "indexReady"), JSON.stringify(!!indexReady));
    } catch (e) {
      console.warn("localStorage save failed", e);
    }
  };

  const loadLectureCache = (lectureId: number) => {
    try {
      const cachedSummary = localStorage.getItem(lsKey(lectureId, "summary"));
      const cachedQuiz = localStorage.getItem(lsKey(lectureId, "quiz"));
      const cachedChat = localStorage.getItem(lsKey(lectureId, "chat"));
      const cachedOcrReady = localStorage.getItem(lsKey(lectureId, "ocrReady"));
      const cachedIndexReady = localStorage.getItem(lsKey(lectureId, "indexReady"));

      setSummary(cachedSummary ?? "");
      setQuiz(cachedQuiz ? JSON.parse(cachedQuiz) : []);
      setChatHistory(cachedChat ? JSON.parse(cachedChat) : []);
      setOcrReady(cachedOcrReady ? JSON.parse(cachedOcrReady) : false);
      setIndexReady(cachedIndexReady ? JSON.parse(cachedIndexReady) : false);
    } catch (e) {
      console.warn("localStorage load failed", e);
      setSummary("");
      setQuiz([]);
      setChatHistory([]);
      setOcrReady(false);
      setIndexReady(false);
    }
  };

  // 강의 선택: 저장된 PDF/오디오/산출물(요약/퀴즈/채팅/상태)을 복구
  const selectLecture = async (lectureId: number) => {
    setSelectedLecture(lectureId);
    setStatus("강의 불러오는 중...");

    // 업로드용 파일 선택은 강의 전환 시 초기화
    setPdf(null);
    setAudio(null);
    setCitations?.([]); // citations state 있으면

    // 1) 먼저 프론트 캐시 복구 (요약/퀴즈/질의응답/상태)
    loadLectureCache(lectureId);

    // 2) DB에 저장된 asset(pdf/audio) 로드
    try {
      const res = await fetch(`${API_BASE}/lectures/${lectureId}`);
      if (res.ok) {
        const data = await res.json();
        const asset = data?.asset;
        const nextPdfUrl = asset?.pdf_path ? `${API_BASE}/files/${asset.pdf_path}` : "";
        const nextAudioUrl = asset?.audio_path ? `${API_BASE}/files/${asset.audio_path}` : "";
        setPdfUrl(nextPdfUrl);
        setAudioUrl(nextAudioUrl);
      } else {
        setPdfUrl("");
        setAudioUrl("");
      }
    } catch (e) {
      console.error("Failed to load lecture asset", e);
      setPdfUrl("");
      setAudioUrl("");
    }

    // 3) transcript/anchors 로드 (있으면 UI가 바로 채워짐)
    try {
      const tRes = await fetch(`${API_BASE}/lectures/${lectureId}/transcript`);
      if (tRes.ok) {
        const tJson = await tRes.json();
        setTranscript(tJson.segments || []);
      } else {
        setTranscript([]);
      }
    } catch {
      setTranscript([]);
    }

    try {
      const aRes = await fetch(`${API_BASE}/lectures/${lectureId}/anchors`);
      if (aRes.ok) {
        const aJson = await aRes.json();
        setAnchors(aJson.anchors || []);
      } else {
        setAnchors([]);
      }
    } catch {
      setAnchors([]);
    }

    setStatus("✅ 강의 로드 완료");
  };

  useEffect(() => {
  if (!selectedLecture) return;
  saveLectureCache(selectedLecture);
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [selectedLecture, summary, quiz, chatHistory, ocrReady, indexReady]);



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

        if (prevPdfBlobUrlRef.current) {
          URL.revokeObjectURL(prevPdfBlobUrlRef.current);
        }
        prevPdfBlobUrlRef.current = url;
        setPdfBlobUrl(url);
      } catch (e) {
        console.error("PDF blob fetch failed:", e);
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

  // 자동 페이지 동기화 (GoodNotes 스타일)
  useEffect(() => {
    if (anchors.length === 0) return;
    
    const sortedAnchors = [...anchors].sort((a, b) => a.time - b.time);
    let targetPage = 1;
    
    for (const anchor of sortedAnchors) {
      if (currentTime >= anchor.time) {
        targetPage = anchor.page;
      } else {
        break;
      }
    }
    
    if (targetPage !== viewPage) {
      setViewPage(targetPage);
    }
  }, [currentTime, anchors]);

  const loadLectures = async () => {
    try {
      const res = await fetch(`${API_BASE}/lectures`);
      if (res.ok) {
        const data = await res.json();
        setLectures(data.lectures || []);
      }
    } catch (e) {
      console.error("Failed to load lectures", e);
    }
  };

  const deleteLecture = async (lectureId) => {
    if (!confirm("이 강의를 삭제할까요?")) return;

    try {
      const res = await fetch(`${API_BASE}/lectures/${lectureId}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "강의 삭제 실패");
        return;
      }

      if (selectedLecture === lectureId) {
        setSelectedLecture(null);
        setPdfUrl("");
        setAudioUrl("");
        setTranscript([]);
        setAnchors([]);
      }

      loadLectures();
    } catch (e) {
      alert("강의 삭제 중 오류 발생");
    }
  };

  const createLecture = async () => {
    if (!title.trim()) {
      setStatus("⚠️ 강의명을 입력해주세요");
      return;
    }
    setIsProcessing(true);
    setStatus("강의 생성 중...");
    try {
      const res = await fetch(`${API_BASE}/lectures`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, description }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatus(`에러: ${err.detail ?? "강의 생성 실패"}`);
        return;
      }
      const data = await res.json();
      setSummary("");
      setQuiz([]);
      setChatHistory([]);
      setCitations([]);
      setAnchors([]);
      setTranscript([]);
      setOcrReady(false);
      setIndexReady(false);

      await selectLecture(data.lecture_id);

      setStatus(`✅ 강의 생성 완료: ${title}`);
      setShowCreateForm(false);
      setTitle("");
      setDescription("");
      loadLectures();
    } catch (e) {
      setStatus(`강의 생성 실패: ${e.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const uploadFiles = async () => {
    if (!selectedLecture) {
      setStatus("⚠️ 강의를 먼저 선택해주세요");
      return;
    }
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
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatus(`에러: ${err.detail ?? "업로드 실패"}`);
        return;
      }
      const data = await res.json();
      setAudioUrl(`${API_BASE}/files/${data.audio_path}`);
      setPdfUrl(`${API_BASE}/files/${data.pdf_path}`);
      setStatus(`✅ 파일 업로드 완료`);
      setPdf(null);
      setAudio(null);
    } catch (e) {
      setStatus(`업로드 실패: ${e.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const runAgentPipeline = async () => {
    if (!selectedLecture) {
      setStatus("⚠️ 강의를 선택해주세요");
      return;
    }
    setIsProcessing(true);
    try {
      setAgentThinking("📁 강의 자료 확인 중...");
      setStatus("AI Agent 작동 중...");
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAgentThinking("🎤 음성을 텍스트로 변환 중 (Whisper STT)...");
      await runTranscribe();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAgentThinking("📄 PDF에서 텍스트 추출 중 (OCR)...");
      await runOcr();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAgentThinking("🧠 지식 베이스 구축 중 (RAG Index)...");
      await buildRagIndex();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAgentThinking("🔗 정밀 싱크 생성 중...");
      await autoGenerateAnchors();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAgentThinking("📝 강의 요약 생성 중...");
      await generateSummary();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAgentThinking("❓ 퀴즈 문제 생성 중...");
      await generateQuiz();
      
      setAgentThinking("");
      setStatus("✅ AI Agent가 모든 작업을 완료했습니다!");
    } catch (e) {
      setStatus(`❌ 오류 발생: ${e.message}`);
      setAgentThinking("");
    } finally {
      setIsProcessing(false);
    }
  };

  const autoGenerateAnchors = async () => {
    if (!selectedLecture || !numPages || !audioDuration) {
      if (!numPages && pdfBlobUrl) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      return;
    }

    const newAnchors = [];
    const interval = audioDuration / numPages;
    
    for (let i = 1; i <= numPages; i++) {
      newAnchors.push({
        page: i,
        time: (i - 1) * interval
      });
    }

    try {
      for (const anchor of newAnchors) {
        await fetch(`${API_BASE}/lectures/${selectedLecture}/anchors`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(anchor),
        });
      }
      await loadAnchors();
    } catch (e) {
      console.error("Auto anchor generation failed:", e);
    }
  };

  const runTranscribe = async () => {
    if (!selectedLecture) return;
    try {
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/transcribe`, { method: "POST" });
      if (!res.ok) throw new Error("STT 실행 실패");
      const tRes = await fetch(`${API_BASE}/lectures/${selectedLecture}/transcript`);
      const tJson = await tRes.json();
      setTranscript(tJson.segments || []);
    } catch (e) {
      console.error("STT 오류:", e);
    }
  };

  const runOcr = async () => {
    if (!selectedLecture) return;
    try {
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/ocr_pdf`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "OCR 실패");
      setOcrReady(true);
      setIndexReady(false);
    } catch (e) {
      console.error("OCR 오류:", e);
      throw e;
    }
  };

  const buildRagIndex = async () => {
    if (!selectedLecture) return;
    try {
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/rag_index`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Index 실패");
      setIndexReady(true);
    } catch (e) {
      console.error("Index 오류:", e);
      throw e;
    }
  };

  const generateSummary = async () => {
    if (!selectedLecture) return;
    try {
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/summary`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "요약 실패");
      setSummary(data.summary || "");
    } catch (e) {
      setSummary("요약 생성 중 오류가 발생했습니다.");
    }
  };

  const generateQuiz = async () => {
    if (!selectedLecture) return;
    try {
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/quiz`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "퀴즈 실패");
      setQuiz(Array.isArray(data.quiz) ? data.quiz : []);
    } catch (e) {
      setQuiz([]);
    }
  };

  const askRag = async () => {
    if (!selectedLecture || !currentMessage.trim()) {
      setStatus("질문을 입력하세요.");
      return;
    }
    const userMsg = { role: "user", content: currentMessage };
    setChatHistory(prev => [...prev, userMsg]);
    const question = currentMessage;
    setCurrentMessage("");
    setIsProcessing(true);
    try {
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/rag_ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, topK: 5 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Ask 실패");
      const answer = data.answer || "";
      setCitations(data.citations || []);
      
      const aiMsg = { role: "assistant", content: answer };
      setChatHistory(prev => [...prev, aiMsg]);
      setStatus("답변 생성 완료");
    } catch (e) {
      const errorMsg = { role: "assistant", content: "답변 생성 중 오류가 발생했습니다." };
      setChatHistory(prev => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const jumpTo = (time) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, time);
    audioRef.current.play().catch(() => {});
  };

  const jumpToPage = async (page) => {
    if (!selectedLecture || !canPageJump) {
      setStatus("페이지 점프 불가");
      return;
    }
    try {
      const res = await fetch(
        `${API_BASE}/lectures/${selectedLecture}/page_time?page=${page}&num_pages=${numPages}&duration=${audioDuration}`
      );
      if (!res.ok) throw new Error("page_time 실패");
      const json = await res.json();
      jumpTo(Number(json.time || 0));
      setViewPage(page);
      setStatus(`페이지 ${page} → ${Number(json.time || 0).toFixed(2)}s 점프`);
    } catch (e) {
      setStatus(`페이지 점프 실패: ${e.message}`);
    }
  };

  const loadAnchors = async () => {
    if (!selectedLecture) return;
    try {
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/anchors`);
      const json = await res.json();
      setAnchors(json.anchors || []);
    } catch (e) {
      console.error("앵커 로드 실패", e);
    }
  };

  const saveAnchor = async () => {
    if (!selectedLecture) return;
    try {
      const payload = { page: anchorPage, time: currentTime };
      const res = await fetch(`${API_BASE}/lectures/${selectedLecture}/anchors`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("앵커 저장 실패");
      setStatus(`앵커 저장 완료: page=${anchorPage}, time=${currentTime.toFixed(2)}s`);
      loadAnchors();
    } catch (e) {
      setStatus(`앵커 저장 실패: ${e.message}`);
    }
  };

  return (
    <div className="h-screen w-full flex flex-col bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 shadow-xl">
        <div className="py-4 px-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-white/20 p-2 rounded-xl backdrop-blur-sm">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <div className="text-xl font-bold text-white tracking-tight">AI Agent for Professors</div>
                <div className="text-xs text-white/90">GoodNotes 스타일 강의 도우미</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                className={userRole === "professor" ? "bg-white text-indigo-700 font-semibold border-0" : "bg-white/20 text-white border-white/30 hover:bg-white/30"}
                onClick={() => setUserRole("professor")}
              >
                <GraduationCap className="w-4 h-4 mr-1.5" />
                교수
              </Button>
              <Button
                variant="outline"
                size="sm"
                className={userRole === "student" ? "bg-white text-indigo-700 font-semibold border-0" : "bg-white/20 text-white border-white/30 hover:bg-white/30"}
                onClick={() => setUserRole("student")}
              >
                <BookOpen className="w-4 h-4 mr-1.5" />
                학생
              </Button>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-2 text-white/95 text-sm bg-white/10 px-3 py-2 rounded-lg backdrop-blur-sm">
            {isProcessing && <Loader2 className="w-4 h-4 animate-spin" />}
            {agentThinking && <Sparkles className="w-4 h-4 animate-pulse" />}
            <span className="font-medium">{agentThinking || status}</span>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex-1 grid grid-cols-[260px_1fr_460px] gap-0 min-h-0">
        {/* 왼쪽: 강의 목록 */}
        <div className="bg-white/80 backdrop-blur-sm border-r border-indigo-100 shadow-lg flex flex-col">
          <div className="p-4 border-b border-indigo-100 bg-gradient-to-br from-indigo-50 to-blue-50">
            <div className="font-bold text-indigo-900 text-base mb-3">📚 강의 목록</div>
            {userRole === "professor" && (
              <Button
                onClick={() => setShowCreateForm(!showCreateForm)}
                className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white shadow-md"
                size="sm"
              >
                + 새 강의 만들기
              </Button>
            )}
          </div>
          
          <ScrollArea className="flex-1 p-3">
            {showCreateForm && userRole === "professor" && (
              <Card className="mb-3 border-indigo-200 shadow-md bg-white">
                <CardContent className="p-3 space-y-2">
                  <Input
                    placeholder="강의명"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="border-indigo-200 focus:border-indigo-400"
                  />
                  <Input
                    placeholder="설명"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    className="border-indigo-200 focus:border-indigo-400"
                  />
                  <div className="flex gap-2">
                    <Button onClick={createLecture} disabled={isProcessing} className="flex-1 bg-indigo-600 hover:bg-indigo-700" size="sm">
                      생성
                    </Button>
                    <Button onClick={() => setShowCreateForm(false)} variant="outline" size="sm" className="border-indigo-200">
                      취소
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
            
            <div className="space-y-2">
              {lectures.map((lecture) => (
                <Card
                  key={lecture.id}
                  className={`cursor-pointer transition-all duration-200 ${
                    selectedLecture === lecture.id
                      ? "border-indigo-500 bg-gradient-to-r from-indigo-100 to-blue-100 shadow-md scale-[1.02]"
                      : "border-indigo-100 hover:bg-indigo-50 hover:border-indigo-300"
                  }`}
                  onClick={() => selectLecture(lecture.id)}
                >
                  <CardContent className="p-3 flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="font-semibold text-indigo-900 text-sm flex items-center gap-2">
                        {selectedLecture === lecture.id && <CheckCircle className="w-4 h-4 text-indigo-600 shrink-0" />}
                        <span className="truncate">{lecture.title}</span>
                      </div>
                      <div className="text-xs text-indigo-600 mt-1 truncate">{lecture.description}</div>
                    </div>
                    {userRole === "professor" && (
                      <Button
                        size="icon"
                        variant="ghost"
                        className="text-red-500 hover:bg-red-50 h-7 w-7 shrink-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteLecture(lecture.id);
                        }}
                      >
                        <span className="text-sm">🗑️</span>
                      </Button>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* 가운데: PDF 뷰어 또는 업로드 영역 */}
        <div className="bg-gradient-to-br from-gray-50 to-slate-100 flex flex-col">
          {!selectedLecture ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="bg-gradient-to-br from-indigo-100 to-blue-100 w-24 h-24 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <BookOpen className="w-12 h-12 text-indigo-600" />
                </div>
                <div className="text-2xl font-bold text-indigo-900 mb-2">강의를 선택해주세요</div>
                <div className="text-sm text-indigo-600">왼쪽에서 강의를 선택하거나 새로 만들어보세요</div>
              </div>
            </div>
          ) : !pdfBlobUrl || !audioUrl ? (
            // 🎯 대형 드래그 앤 드롭 업로드 영역
            <div className="h-full flex items-center justify-center p-8">
              <div className="w-full max-w-3xl space-y-6">
                <div className="text-center mb-8">
                  <div className="bg-gradient-to-br from-indigo-100 to-blue-100 w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                    <Upload className="w-10 h-10 text-indigo-600" />
                  </div>
                  <h2 className="text-3xl font-bold text-indigo-900 mb-2">강의 자료 업로드</h2>
                  <p className="text-indigo-600">PDF 강의자료와 오디오 파일을 드래그하거나 클릭하여 업로드하세요</p>
                </div>

                {/* PDF 드롭존 */}
                <div
                  className={`relative border-3 border-dashed rounded-2xl p-12 transition-all duration-300 ${
                    dragActive.pdf
                      ? "border-indigo-500 bg-indigo-50 scale-105 shadow-2xl"
                      : pdf
                      ? "border-green-400 bg-green-50"
                      : "border-indigo-300 bg-white hover:border-indigo-400 hover:bg-indigo-50"
                  }`}
                  onDragEnter={(e) => handleDrag(e, 'pdf')}
                  onDragLeave={(e) => handleDrag(e, 'pdf')}
                  onDragOver={(e) => handleDrag(e, 'pdf')}
                  onDrop={(e) => handleDrop(e, 'pdf')}
                >
                  <input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => setPdf(e.target.files?.[0] || null)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    disabled={userRole === "student"}
                  />
                  <div className="text-center pointer-events-none">
                    <FileText className={`w-16 h-16 mx-auto mb-4 ${pdf ? "text-green-600" : "text-indigo-400"}`} />
                    <div className="text-xl font-bold text-indigo-900 mb-2">
                      {pdf ? "✓ PDF 업로드 완료" : "📄 PDF 강의자료"}
                    </div>
                    {pdf ? (
                      <div className="text-sm text-green-700 font-medium">{pdf.name}</div>
                    ) : (
                      <div className="text-sm text-indigo-600">
                        PDF 파일을 드래그하거나 클릭하여 선택하세요
                      </div>
                    )}
                  </div>
                </div>

                {/* 오디오 드롭존 */}
                <div
                  className={`relative border-3 border-dashed rounded-2xl p-12 transition-all duration-300 ${
                    dragActive.audio
                      ? "border-indigo-500 bg-indigo-50 scale-105 shadow-2xl"
                      : audio
                      ? "border-green-400 bg-green-50"
                      : "border-indigo-300 bg-white hover:border-indigo-400 hover:bg-indigo-50"
                  }`}
                  onDragEnter={(e) => handleDrag(e, 'audio')}
                  onDragLeave={(e) => handleDrag(e, 'audio')}
                  onDragOver={(e) => handleDrag(e, 'audio')}
                  onDrop={(e) => handleDrop(e, 'audio')}
                >
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setAudio(e.target.files?.[0] || null)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    disabled={userRole === "student"}
                  />
                  <div className="text-center pointer-events-none">
                    <div className={`w-16 h-16 mx-auto mb-4 ${audio ? "text-green-600" : "text-indigo-400"}`}>
                      🎤
                    </div>
                    <div className="text-xl font-bold text-indigo-900 mb-2">
                      {audio ? "✓ 오디오 업로드 완료" : "🎵 강의 오디오"}
                    </div>
                    {audio ? (
                      <div className="text-sm text-green-700 font-medium">{audio.name}</div>
                    ) : (
                      <div className="text-sm text-indigo-600">
                        오디오 파일을 드래그하거나 클릭하여 선택하세요
                      </div>
                    )}
                  </div>
                </div>

                {/* 업로드 버튼 */}
                {userRole === "professor" && (pdf || audio) && (
                  <Button
                    onClick={uploadFiles}
                    disabled={isProcessing || (!pdf && !audio)}
                    className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white py-6 text-lg font-bold shadow-lg"
                    size="lg"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin mr-2" />
                        업로드 중...
                      </>
                    ) : (
                      <>
                        <Upload className="w-5 h-5 mr-2" />
                        파일 업로드하기
                      </>
                    )}
                  </Button>
                )}
              </div>
            </div>
          ) : (
            <>
              {/* PDF 뷰어 */}
              <div className="flex-1 overflow-auto p-6 bg-gradient-to-br from-gray-50 to-slate-100">
                <div className="flex justify-center">
                  <div className="bg-white shadow-2xl rounded-lg overflow-hidden">
                    <PdfViewer
                      file={pdfBlobUrl}
                      page={viewPage}
                      onLoad={(n) => setNumPages(n)}
                      width={750}
                    />
                  </div>
                </div>
              </div>

              {/* 페이지 네비게이션 */}
              <div className="border-t border-indigo-200 bg-white/90 backdrop-blur-sm p-4 flex items-center justify-center gap-4 shadow-lg">
                <Button
                  size="sm"
                  onClick={() => setViewPage(Math.max(1, viewPage - 1))}
                  disabled={viewPage <= 1}
                  className="bg-indigo-600 hover:bg-indigo-700 px-6 shadow-md"
                >
                  ← 이전
                </Button>
                
                <div className="flex items-center gap-3 bg-indigo-50 px-4 py-2 rounded-lg">
                  <Input
                    type="number"
                    min={1}
                    max={numPages}
                    value={viewPage}
                    onChange={(e) => {
                      const p = Number(e.target.value);
                      if (p >= 1 && p <= numPages) setViewPage(p);
                    }}
                    className="w-20 text-center border-indigo-300 font-semibold"
                  />
                  <span className="text-sm font-semibold text-indigo-900">/ {numPages}</span>
                </div>
                
                <Button
                  size="sm"
                  onClick={() => setViewPage(Math.min(numPages, viewPage + 1))}
                  disabled={viewPage >= numPages}
                  className="bg-indigo-600 hover:bg-indigo-700 px-6 shadow-md"
                >
                  다음 →
                </Button>

                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => jumpToPage(viewPage)}
                  disabled={!canPageJump}
                  className="border-indigo-300 text-indigo-700 hover:bg-indigo-50 font-semibold"
                >
                  🎵 이 페이지 재생
                </Button>
              </div>

              {/* 오디오 플레이어 */}
              {audioUrl && (
                <div className="border-t border-indigo-200 bg-gradient-to-r from-indigo-50 to-blue-50 p-4 shadow-inner">
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    controls
                    className="w-full"
                    onLoadedMetadata={(e) => setAudioDuration((e.target as HTMLAudioElement).duration || 0)}
                    onTimeUpdate={(e) => setCurrentTime((e.target as HTMLAudioElement).currentTime)}
                  />
                  <div className="text-sm text-indigo-700 mt-2 text-center font-medium">
                    ⏱️ {currentTime.toFixed(1)}s / {audioDuration.toFixed(1)}s
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* 오른쪽: 컨트롤 패널 */}
        <div className="bg-white/90 backdrop-blur-sm border-l border-indigo-100 shadow-xl flex flex-col min-h-0 h-full">
          <Tabs defaultValue="upload" className="flex-1 flex flex-col min-h-0 h-full">
            <TabsList className="shrink-0 grid grid-cols-5 bg-gradient-to-r from-indigo-100 to-blue-100 mx-3 mt-3 mb-1 p-1 rounded-lg">
              <TabsTrigger value="upload" className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:shadow-md">업로드</TabsTrigger>
              <TabsTrigger value="summary" className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:shadow-md">요약</TabsTrigger>
              <TabsTrigger value="quiz" className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:shadow-md">퀴즈</TabsTrigger>
              <TabsTrigger value="chat" className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:shadow-md">질의응답</TabsTrigger>
              <TabsTrigger value="sync" className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:shadow-md">싱크</TabsTrigger>
            </TabsList>

            {/* 업로드 탭 */}
            <TabsContent value="upload" className="flex-1 p-4 space-y-4 overflow-auto">
              {userRole === "professor" && (
                <Card className="border-indigo-300 bg-gradient-to-br from-indigo-50 to-blue-50 shadow-lg">
                  <CardHeader className="p-4">
                    <div className="font-bold text-indigo-900 flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-yellow-500" />
                      AI Agent 자동 처리
                    </div>
                  </CardHeader>
                  <CardContent className="p-4 space-y-3">
                    <div className="text-xs text-indigo-700 bg-white/60 p-3 rounded-lg leading-relaxed">
                      <strong>자동 처리 과정:</strong><br/>
                      STT (음성→텍스트) → OCR (PDF→텍스트) → RAG 인덱스 → 정밀 싱크 → 요약 생성 → 퀴즈 생성
                    </div>
                    <Button 
                      onClick={runAgentPipeline} 
                      disabled={isProcessing || !pdfUrl || !audioUrl} 
                      className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white font-bold py-6 shadow-lg"
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin mr-2" />
                          작동 중...
                        </>
                      ) : (
                        <>
                          <PlayCircle className="w-5 h-5 mr-2" />
                          AI Agent 실행
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* 요약 탭 */}
            <TabsContent value="summary" className="!mt-0 flex-1 min-h-0 flex flex-col overflow-hidden px-4 pb-4 pt-2">
              <Card className="flex-1 min-h-0 flex flex-col border-indigo-200 shadow-md">
                <CardHeader className="p-4 bg-gradient-to-r from-indigo-50 to-blue-50 shrink-0">
                  <div className="font-bold text-indigo-900">📝 강의 요약</div>
                </CardHeader>

                <CardContent className="p-4 flex-1 min-h-0 overflow-hidden">
                  <ScrollArea className="h-full">
                    {summary ? (
                      <div className="text-sm text-indigo-900 whitespace-pre-wrap leading-relaxed bg-white p-3 rounded-lg">
                        {summary}
                      </div>
                    ) : (
                      <div className="text-sm text-indigo-500 text-center py-8">
                        요약이 아직 생성되지 않았습니다
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            {/* 퀴즈 탭 */}
            <TabsContent
              value="quiz"
              className="px-4 pt-2 overflow-hidden"
            >
              <Card className="h-full flex flex-col overflow-hidden border-indigo-200 shadow-md">
                <CardHeader className="shrink-0 p-4 bg-gradient-to-r from-indigo-50 to-blue-50">
                  <div className="font-bold text-indigo-900">❓ 퀴즈 ({quiz.length}개)</div>
                </CardHeader>

                <CardContent className="flex-1 min-h-0 overflow-hidden p-4">
                  {quiz.length > 0 ? (
                    <ScrollArea className="h-full">
                      <div className="space-y-4 pr-2">
                        {quiz.map((q, idx) => (
                          <div
                            key={idx}
                            className="border border-indigo-200 rounded-lg p-3 bg-white shadow-sm"
                          >
                            <div className="text-sm font-semibold text-indigo-900 mb-3">
                              Q{idx + 1}. {q.question}
                            </div>

                            <div className="space-y-2">
                              {q.options.map((opt, optIdx) => (
                                <div
                                  key={optIdx}
                                  className={`text-xs p-2 rounded-lg ${
                                    optIdx === q.answer
                                      ? "bg-green-100 border-2 border-green-500 font-semibold"
                                      : "bg-gray-50 border border-gray-200"
                                  }`}
                                >
                                  {String.fromCharCode(65 + optIdx)}. {opt}
                                </div>
                              ))}
                            </div>

                            {q.explanation && (
                              <div className="mt-3 text-xs text-indigo-700 bg-indigo-50 border border-indigo-200 rounded-lg p-2">
                                <span className="font-semibold">해설:</span> {q.explanation}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <div className="text-sm text-indigo-500 text-center py-8">
                      퀴즈가 아직 생성되지 않았습니다
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>





            {/* 질의응답 탭 */}
            <TabsContent value="chat" className="!mt-0 flex-1 flex flex-col px-4 pb-4 pt-2">
              <ScrollArea className="flex-1 mb-4">
                <div className="space-y-3">
                  {chatHistory.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                      <div className={`max-w-[85%] p-3 rounded-xl shadow-md ${
                        msg.role === "user" 
                          ? "bg-gradient-to-r from-indigo-600 to-blue-600 text-white" 
                          : "bg-white text-indigo-900 border border-indigo-200"
                      }`}>
                        <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                      </div>
                    </div>
                  ))}
                  {citations.length > 0 && (
                    <div className="space-y-2 mt-3">
                      {citations.map((c) => (
                        <div key={c.chunkId} className="p-3 rounded-lg border border-indigo-200 bg-white shadow-sm">
                          <div className="flex justify-between items-start gap-2 mb-2">
                            <span className="text-sm font-bold text-indigo-900">📄 p.{c.page}</span>
                            <Button 
                              size="sm" 
                              variant="outline" 
                              onClick={() => { setViewPage(Number(c.page)); jumpToPage(Number(c.page)); }} 
                              className="h-7 text-xs px-3 border-indigo-300 hover:bg-indigo-50"
                            >
                              이동 →
                            </Button>
                          </div>
                          <div className="text-xs text-indigo-700 leading-relaxed">{c.snippet}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </ScrollArea>
              <div className="space-y-3">
                <Textarea
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  placeholder="강의 내용에 대해 질문하세요..."
                  rows={3}
                  className="border-indigo-300 focus:border-indigo-500 resize-none"
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
                  className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 shadow-md font-semibold"
                >
                  {isProcessing ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : "💬"} 질문하기
                </Button>
              </div>
            </TabsContent>

            {/* 싱크 탭 */}
            <TabsContent value="sync" className="!mt-0 flex-1 overflow-hidden">
              <ScrollArea className="h-full px-4 pb-4 pt-2">

                <div className="space-y-4">
                  <Card className="border-indigo-300 shadow-md">
                    <CardHeader className="p-4 bg-gradient-to-r from-indigo-50 to-blue-50">
                      <div className="font-bold text-indigo-900">🎯 정밀 싱크 (앵커)</div>
                    </CardHeader>
                    <CardContent className="p-4 space-y-3">
                      <div className="text-xs text-indigo-700 bg-indigo-50 p-3 rounded-lg leading-relaxed">
                        현재 재생 중인 시간을 특정 페이지와 연결하여 정확한 페이지-시간 싱크를 만들 수 있습니다.
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <div className="text-xs text-indigo-700 mb-2 font-semibold">페이지 번호</div>
                          <Input
                            type="number"
                            min={1}
                            max={numPages}
                            value={anchorPage}
                            onChange={(e) => setAnchorPage(Number(e.target.value) || 1)}
                            className="border-indigo-300 font-semibold"
                          />
                        </div>
                        <div>
                          <div className="text-xs text-indigo-700 mb-2 font-semibold">현재 시간</div>
                          <div className="text-sm text-indigo-900 py-2.5 px-3 bg-indigo-50 rounded-lg font-mono font-bold">
                            {currentTime.toFixed(2)}s
                          </div>
                        </div>
                      </div>
                      <Button 
                        onClick={saveAnchor} 
                        disabled={!canPageJump} 
                        className="w-full bg-indigo-600 hover:bg-indigo-700 font-semibold shadow-md"
                      >
                        ⚓ 앵커 저장
                      </Button>
                    </CardContent>
                  </Card>

                  <Card className="border-indigo-200 shadow-md">
                    <CardHeader className="p-4 bg-gradient-to-r from-indigo-50 to-blue-50">
                      <div className="font-bold text-indigo-900">📍 저장된 앵커</div>
                    </CardHeader>
                    <CardContent className="p-4">
                      {anchors.length === 0 ? (
                        <div className="text-sm text-indigo-500 text-center py-6">
                          아직 앵커가 없습니다.<br/>
                          <span className="text-xs">원하는 페이지에서 앵커를 저장하세요</span>
                        </div>
                      ) : (
                        <div className="space-y-2 max-h-52 overflow-y-auto">
                          {anchors.slice().sort((a, b) => a.page - b.page).map((a, idx) => (
                            <div 
                              key={`${a.page}-${idx}`} 
                              className="flex items-center justify-between gap-2 p-2.5 rounded-lg bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-200"
                            >
                              <span className="text-sm text-indigo-900 font-semibold">
                                📄 p.{a.page} → {Number(a.time).toFixed(1)}s
                              </span>
                              <Button 
                                size="sm" 
                                variant="outline" 
                                onClick={() => jumpTo(Number(a.time))} 
                                className="h-7 text-xs px-3 border-indigo-300 hover:bg-white"
                              >
                                ▶️ 이동
                              </Button>
                            </div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  <Card className="border-indigo-200 shadow-md">
                    <CardHeader className="p-4 bg-gradient-to-r from-indigo-50 to-blue-50">
                      <div className="font-bold text-indigo-900">📝 Transcript (자막)</div>
                    </CardHeader>
                    <CardContent className="p-4">
                      {transcript.length === 0 ? (
                        <div className="text-sm text-indigo-500 text-center py-6">
                          STT를 실행하면 자막이 표시됩니다
                        </div>
                      ) : (
                        <div className="space-y-2 max-h-64 overflow-y-auto">
                          {transcript.map((s, idx) => {
                            const start = Number(s.start);
                            const end = Number(s.end);
                            const active = currentTime >= start && currentTime < end;
                            return (
                              <div
                                key={idx}
                                onClick={() => jumpTo(start)}
                                className={`p-3 rounded-lg cursor-pointer border transition-all ${
                                  active 
                                    ? "border-indigo-500 bg-gradient-to-r from-indigo-100 to-blue-100 shadow-md scale-[1.02]" 
                                    : "border-indigo-200 hover:bg-indigo-50 bg-white"
                                }`}
                              >
                                <div className="text-xs text-indigo-600 mb-1 font-semibold">
                                  ⏱️ {start.toFixed(1)}s ~ {end.toFixed(1)}s
                                </div>
                                <div className="text-sm text-indigo-900 leading-relaxed">{s.text}</div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}