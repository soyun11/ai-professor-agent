// frontend/components/TopNav.tsx
"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Brain, GraduationCap, BookOpen, ArrowLeft } from "lucide-react";
import { getStoredRole, setStoredRole, type UserRole } from "@/lib/role";

type Props = {
  title?: string;
  subtitle?: string;
  showBack?: boolean;
};

export default function TopNav({
  title = "AI Agent for Professors",
  subtitle = "GoodNotes 스타일 강의 도우미",
  showBack = false,
}: Props) {
  const [role, setRole] = useState<UserRole>("professor");

  useEffect(() => {
    setRole(getStoredRole());
  }, []);

  const setAndSave = (next: UserRole) => {
    setRole(next);
    setStoredRole(next);
    window.dispatchEvent(new Event("aiagent:rolechange"));
  };

  return (
    <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 shadow-xl">
      <div className="py-4 px-6">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            {showBack && (
              <Link href="/" className="shrink-0">
                <Button
                  variant="outline"
                  size="icon"
                  className="bg-white/20 text-white border-white/30 hover:bg-white/30"
                  aria-label="Back"
                >
                  <ArrowLeft className="w-4 h-4" />
                </Button>
              </Link>
            )}

            <div className="bg-white/20 p-2 rounded-xl backdrop-blur-sm shrink-0">
              <Brain className="w-7 h-7 text-white" />
            </div>

            <div className="min-w-0">
              <div className="text-xl font-bold text-white tracking-tight truncate">
                {title}
              </div>
              <div className="text-xs text-white/90 truncate">{subtitle}</div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <Button
              variant="outline"
              size="sm"
              className={
                role === "professor"
                  ? "bg-white text-indigo-700 font-semibold border-0"
                  : "bg-white/20 text-white border-white/30 hover:bg-white/30"
              }
              onClick={() => setAndSave("professor")}
            >
              <GraduationCap className="w-4 h-4 mr-1.5" />
              교수
            </Button>
            <Button
              variant="outline"
              size="sm"
              className={
                role === "student"
                  ? "bg-white text-indigo-700 font-semibold border-0"
                  : "bg-white/20 text-white border-white/30 hover:bg-white/30"
              }
              onClick={() => setAndSave("student")}
            >
              <BookOpen className="w-4 h-4 mr-1.5" />
              학생
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
