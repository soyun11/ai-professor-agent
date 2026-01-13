import yt_dlp
from pathlib import Path
import re


def clean_filename(title: str) -> str:
    """파일 이름에 쓸 수 없는 문자 제거"""
    # 특수문자 제거
    clean = re.sub(r'[\\/*?:"<>|]', '', title)
    # 공백 정리
    clean = re.sub(r'\s+', ' ', clean).strip()
    # 너무 길면 자르기
    if len(clean) > 100:
        clean = clean[:100]
    return clean


def download_youtube_to_mp3(url: str, output_dir: Path, filename: str = None) -> Path:
    """
    유튜브 URL → MP3 파일로 변환
    
    Args:
        url: 유튜브 링크
        output_dir: 저장할 폴더
        filename: 파일 이름 (None이면 유튜브 제목 사용)
    
    Returns:
        MP3 파일 경로
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[YouTube] 다운로드 중: {url}")
    
    # 먼저 제목 가져오기
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'Unknown')
        print(f"[YouTube] 제목: {title}")
    
    # 파일 이름 결정
    if filename is None:
        filename = clean_filename(title)
    
    mp3_path = output_dir / f"{filename}.mp3"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / f"{filename}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"[YouTube] 완료: {mp3_path}")
    return mp3_path


def download_multiple(urls: list[str], output_dir: Path) -> list[Path]:
    """
    여러 유튜브 링크 한번에 다운로드
    
    Args:
        urls: 유튜브 링크 리스트
        output_dir: 저장할 폴더
    
    Returns:
        MP3 파일 경로 리스트
    """
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] 처리 중...")
        try:
            mp3_path = download_youtube_to_mp3(url, output_dir)
            results.append(mp3_path)
        except Exception as e:
            print(f"[오류] {url}: {e}")
    
    print(f"\n총 {len(results)}/{len(urls)}개 다운로드 완료!")
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  단일: python youtube_downloader.py [유튜브URL]")
        print("  여러개: python -m youtube_downloader.py [URL1] [URL2] [URL3] ...")
        sys.exit(1)
    
    output = Path("./downloads")
    
    if len(sys.argv) == 2:
        # 단일 URL
        result = download_youtube_to_mp3(sys.argv[1], output)
        print(f"저장됨: {result}")
    else:
        # 여러 URL
        urls = sys.argv[1:]
        results = download_multiple(urls, output)
        print("\n저장된 파일들:")
        for r in results:
            print(f"  - {r}")