from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any

_TIME_LINE = re.compile(r"(\d+:\d+:\d+,\d+)\s-->\s(\d+:\d+:\d+,\d+)")

def _to_sec(t: str) -> float:
    # 00:08:05,080
    hh, mm, ss_ms = t.split(":")
    ss, ms = ss_ms.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000

def parse_srt(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[Dict[str, Any]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if "-->" in line:
            m = _TIME_LINE.search(line)
            if not m:
                i += 1
                continue

            start = _to_sec(m.group(1))
            end = _to_sec(m.group(2))

            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            text = " ".join(text_lines).strip()
            if text:
                out.append({"start": start, "end": end, "text": text})

        i += 1

    return out
