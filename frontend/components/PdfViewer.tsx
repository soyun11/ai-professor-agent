"use client";

import { useEffect, useState } from "react";

type Props = {
  file: string;
  page: number;
  width?: number;
  onLoad?: (numPages: number) => void;
  lectureId?: number;  // 백엔드에서 페이지 수 가져오기 위해
};

const API_BASE = "http://127.0.0.1:8000";

export default function PdfViewer({ file, page, width = 760, onLoad, lectureId }: Props) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    // 백엔드에서 PDF 페이지 수 가져오기
    const fetchPageCount = async () => {
      if (!lectureId || !onLoad) return;

      try {
        // 백엔드 API 호출 (필요시 엔드포인트 추가해야 함)
        const res = await fetch(`${API_BASE}/lectures/${lectureId}/pdf_info`);
        if (res.ok) {
          const data = await res.json();
          onLoad(data.num_pages || 100);
        } else {
          // API가 없으면 기본값 사용
          onLoad(100);
        }
      } catch (e) {
        console.log("페이지 수 가져오기 실패, 기본값 사용");
        onLoad(100);
      } finally {
        setLoading(false);
      }
    };

    fetchPageCount();
  }, [file, onLoad, lectureId]);

  const handleLoad = () => {
    setLoading(false);
    setError(false);
  };

  const handleError = () => {
    setLoading(false);
    setError(true);
  };

  // PDF URL에 페이지 번호 추가
  const pdfUrl = `${file}#page=${page}`;

  if (error) {
    return (
      <div className="flex items-center justify-center p-12 bg-red-50 rounded-lg">
        <div className="text-red-600">PDF 로딩 실패</div>
      </div>
    );
  }

  return (
    <div className="relative" style={{ width }}>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-blue-50 rounded-lg z-10">
          <div className="text-blue-600">PDF 로딩중...</div>
        </div>
      )}
      <iframe
        src={pdfUrl}
        onLoad={handleLoad}
        onError={handleError}
        className="w-full rounded-lg border border-gray-200 shadow-lg"
        style={{ 
          height: '800px',
          display: loading ? 'none' : 'block'
        }}
        title="PDF Viewer"
      />
    </div>
  );
}