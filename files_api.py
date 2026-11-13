"""Secure downloads for generated daily intelligence outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "output"
REPORTS = OUTPUT / "reports"
VIDEOS = OUTPUT / "videos"
files_router = APIRouter()
_DATE_FILE = re.compile(r"^AI_Daily_Intelligence_\d{4}-\d{2}-\d{2}(?:_Edition_\d+)?\.pdf$")
_VIDEO_FOLDER = re.compile(r"^AI_Daily_Intelligence_\d{4}-\d{2}-\d{2}_(one_minute|three_minute)$")
_VIDEO_FILE = re.compile(r"^AI_Daily_Intelligence_\d{4}-\d{2}-\d{2}\.mp4$")


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(directory.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _latest_pdf() -> Path | None:
    if not REPORTS.is_dir():
        return None
    files = [p for p in REPORTS.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
             and _DATE_FILE.fullmatch(p.name) and _inside(p, REPORTS)]
    return max(files, key=lambda p: (p.stat().st_mtime_ns, p.name)) if files else None


def _latest_video(video_type: str) -> Path | None:
    if not VIDEOS.is_dir():
        return None
    files = []
    for folder in VIDEOS.iterdir():
        match = _VIDEO_FOLDER.fullmatch(folder.name)
        if not match or match.group(1) != video_type or not folder.is_dir() or not _inside(folder, VIDEOS):
            continue
        path = folder / f"{folder.name[:-len(video_type)-1]}.mp4"
        if path.is_file() and path.suffix.lower() == ".mp4" and _VIDEO_FILE.fullmatch(path.name) and _inside(path, VIDEOS):
            files.append(path)
    return max(files, key=lambda p: (p.stat().st_mtime_ns, p.name)) if files else None


def _metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    stat = path.stat()
    return {"filename": path.name, "size": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()}


def _download(path: Path | None, directory: Path, media_type: str) -> FileResponse:
    if path is None or not _inside(path, directory):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type=media_type, filename=path.name)


@files_router.get("/files/manifest")
def manifest() -> dict[str, Any]:
    programme = OUTPUT / "daily_programme.json"
    programme_file = programme if programme.is_file() and _inside(programme, OUTPUT) else None
    return {"daily_programme": _metadata(programme_file), "pdf": _metadata(_latest_pdf()),
            "videos": {"one_minute": _metadata(_latest_video("one_minute")),
                       "three_minute": _metadata(_latest_video("three_minute"))}}


@files_router.get("/files/pdf")
def download_pdf() -> FileResponse:
    return _download(_latest_pdf(), REPORTS, "application/pdf")


@files_router.get("/files/video/one_minute")
def download_one_minute() -> FileResponse:
    return _download(_latest_video("one_minute"), VIDEOS, "video/mp4")


@files_router.get("/files/video/three_minute")
def download_three_minute() -> FileResponse:
    return _download(_latest_video("three_minute"), VIDEOS, "video/mp4")
