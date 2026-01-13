// frontend/app/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import TopNav from "@/components/TopNav";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, CheckCircle, Plus, BookOpen, Sparkles } from "lucide-react";
import { getStoredRole, type UserRole } from "@/lib/role";

const API_BASE = "http://127.0.0.1:8000";

type Lecture = {
  id: number;
  title: string;
  description?: string;
};

export default function Page() {
  const router = useRouter();

  const [role, setRole] = useState<UserRole>("professor");

  const [lectures, setLectures] = useState<Lecture[]>([]);
  const [loading, setLoading] = useState(false);

  const [showCreate, setShowCreate] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  const [status, setStatus] = useState("준비 완료 ✨");

  useEffect(() => {
    setRole(getStoredRole());

    const onRoleChange = () => setRole(getStoredRole());
    window.addEventListener("aiagent:rolechange", onRoleChange);
    return () => window.removeEventListener("aiagent:rolechange", onRoleChange);
  }, []);

  const loadLectures = async () => {
    try {
      const res = await fetch(`${API_BASE}/lectures`);
      if (res.ok) {
        const data = await res.json();
        setLectures(data.lectures || []);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadLectures();
  }, []);

  const createLecture = async () => {
    if (!title.trim()) {
      setStatus("⚠️ 강의명을 입력해주세요");
      return;
    }
    setLoading(true);
    setStatus("강의 생성 중...");
    try {
      const res = await fetch(`${API_BASE}/lectures`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, description }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setStatus(data?.detail || "강의 생성 실패");
        return;
      }

      setStatus("✅ 강의 생성 완료");
      setTitle("");
      setDescription("");
      setShowCreate(false);
      await loadLectures();

      // 생성 직후 상세로 이동 - ✅ 경로 수정: /lecture → /lectures
      if (data?.lecture_id) router.push(`/lectures/${data.lecture_id}`);
    } catch (e: any) {
      setStatus(e?.message || "강의 생성 실패");
    } finally {
      setLoading(false);
    }
  };

  const deleteLecture = async (lectureId: number) => {
    if (!confirm("이 강의를 삭제할까요?")) return;

    try {
      const res = await fetch(`${API_BASE}/lectures/${lectureId}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "강의 삭제 실패");
        return;
      }
      setStatus("🗑️ 강의 삭제 완료");
      loadLectures();
    } catch {
      alert("강의 삭제 중 오류 발생");
    }
  };

  return (
    <div className="h-screen w-full flex flex-col bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900">
      <TopNav />

      {/* 상태 바 */}
      <div className="px-6 pt-4">
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl shadow-lg px-4 py-3 flex items-center gap-2 text-sm text-slate-200">
          {loading && <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />}
          <span className="font-medium">{status}</span>
        </div>
      </div>

      <div className="flex-1 min-h-0 p-6">
        <div className="max-w-6xl mx-auto h-full flex flex-col min-h-0">
          {/* 헤더 */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <div className="text-2xl font-bold text-white flex items-center gap-3">
                <div className="bg-gradient-to-br from-indigo-600 to-violet-600 w-10 h-10 rounded-xl flex items-center justify-center shadow-lg">
                  <BookOpen className="w-5 h-5 text-white" />
                </div>
                강의 목록
              </div>
              <div className="text-sm text-slate-400 mt-2">강의를 선택하면 상세 화면으로 이동합니다</div>
            </div>

            {role === "professor" && (
              <Button
                onClick={() => setShowCreate((v) => !v)}
                className="bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white shadow-lg"
              >
                <Plus className="w-4 h-4 mr-2" />
                새 강의
              </Button>
            )}
          </div>

          {/* 강의 생성 폼 */}
          {showCreate && role === "professor" && (
            <Card className="mb-6 border-slate-700 shadow-xl bg-slate-800/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <div className="font-bold text-white flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-amber-400" />
                  새 강의 만들기
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <Input
                  placeholder="강의명"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="border-slate-600 bg-slate-900/50 text-white placeholder:text-slate-500 focus:border-indigo-500"
                />
                <Input
                  placeholder="설명"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="border-slate-600 bg-slate-900/50 text-white placeholder:text-slate-500 focus:border-indigo-500"
                />
                <div className="flex gap-2">
                  <Button
                    onClick={createLecture}
                    disabled={loading}
                    className="bg-indigo-600 hover:bg-indigo-700"
                  >
                    생성
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setShowCreate(false)}
                    className="border-slate-600 text-slate-300 hover:bg-slate-700"
                  >
                    취소
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* 강의 목록 */}
          <Card className="flex-1 min-h-0 border-slate-700 bg-slate-800/30 backdrop-blur-sm shadow-xl">
            <CardContent className="p-4 h-full min-h-0">
              <ScrollArea className="h-full">
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 pr-2">
                  {lectures.map((lec) => (
                    <Card
                      key={lec.id}
                      className="cursor-pointer border-slate-700 bg-slate-800/50 hover:border-indigo-500 hover:bg-slate-800 transition-all duration-300 hover:scale-[1.02]"
                      onClick={() => router.push(`/lectures/${lec.id}`)}  // ✅ 경로 수정
                    >
                      <CardContent className="p-4">
                        <div className="font-semibold text-white flex items-center gap-2">
                          <CheckCircle className="w-4 h-4 text-indigo-400" />
                          <span className="truncate">{lec.title}</span>
                        </div>
                        <div className="text-xs text-slate-400 mt-2 line-clamp-2">
                          {lec.description || "설명 없음"}
                        </div>

                        {role === "professor" && (
                          <div className="mt-3 flex justify-end">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="text-rose-400 hover:bg-rose-950/50 hover:text-rose-300"
                              onClick={(e) => {
                                e.stopPropagation();
                                deleteLecture(lec.id);
                              }}
                            >
                              🗑️ 삭제
                            </Button>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}

                  {lectures.length === 0 && (
                    <div className="col-span-full text-center py-16 text-slate-500">
                      아직 강의가 없습니다. (교수 모드에서 새 강의를 만들어보세요)
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}