"""Run the daily collection, research, PDF and video pipeline for n8n."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
PROGRAMME = OUTPUT / "daily_programme.json"
REPORTS = OUTPUT / "reports"
VIDEOS = OUTPUT / "videos"
LOGS = OUTPUT / "pipeline_logs"
LOCK = OUTPUT / ".pipeline.lock"
TIMEOUT_SECONDS = 2400
LOCK_MAX_AGE_SECONDS = 3 * 60 * 60
VIDEO_TYPES = ("one_minute", "three_minute")


def _today() -> str:
    # India uses UTC+05:30 year-round; match the n8n schedule timezone even
    # if the Windows host itself is configured to another timezone.
    india = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(india).date().isoformat()


def _read_programme() -> dict[str, Any] | None:
    try:
        with PROGRAMME.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _find_pdf(day: str) -> Path | None:
    if not REPORTS.is_dir():
        return None
    candidates = list(REPORTS.glob(f"AI_Daily_Intelligence_{day}*.pdf"))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _video_path(day: str, video_type: str) -> Path:
    folder = VIDEOS / f"AI_Daily_Intelligence_{day}_{video_type}"
    return folder / f"AI_Daily_Intelligence_{day}.mp4"


def _normalise_narration() -> bool:
    data = _read_programme()
    if not data:
        raise ValueError("daily_programme.json is missing or invalid before video generation")
    programmes = data.get("programmes")
    if not isinstance(programmes, dict):
        raise ValueError("daily_programme.json has no programmes object")
    changed = False
    for video_type in VIDEO_TYPES:
        entry = programmes.get(video_type)
        if not isinstance(entry, dict) or not isinstance(entry.get("narration"), str):
            raise ValueError(f"daily_programme.json has no {video_type} narration")
        normalised = " ".join(entry["narration"].split())
        if normalised != entry["narration"]:
            entry["narration"] = normalised
            changed = True
    if changed:
        temporary = PROGRAMME.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, PROGRAMME)
    return changed


def _redact(value: str) -> str:
    # Avoid logging environment secrets and common API-token formats.
    safe = value
    secret_values = [
        value for name, value in os.environ.items()
        if any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        and len(value) >= 6
    ]
    for secret in secret_values:
        safe = safe.replace(secret, "[REDACTED]")
    safe = re.sub(r"(?i)(api[_ -]?key|access[_ -]?token|authorization|password)(\s*[:=]\s*)\S+",
                  r"\1\2[REDACTED]", safe)
    safe = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", safe)
    safe = re.sub(r"\b(?:gsk_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\b",
                  "[REDACTED]", safe)
    return safe


def _safe_tail(value: str, lines: int = 12, chars: int = 2400) -> str:
    return "\n".join(_redact(value).splitlines()[-lines:])[-chars:]


def _run_step(name: str, command: list[str], log, timeout: int = TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    log.write(f"\n===== {name} =====\n$ {' '.join(command)}\n")
    log.flush()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        out = error.stdout or ""
        err = error.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        log.write(_safe_tail(out + "\n" + err) + f"\n{ name } timed out after {timeout}s\n")
        log.flush()
        raise RuntimeError(f"{name} timed out after {timeout} seconds") from None
    log.write("[stdout]\n" + _redact(result.stdout or "") + "\n[stderr]\n" +
              _redact(result.stderr or "") + "\n")
    log.flush()
    return result


def _result(day: str, log_path: Path, *, ok: bool, stage: str, error: str = "",
            stories: int = 0, headline: str = "", pdf: str = "",
            videos: dict[str, str] | None = None, skipped_research: bool = False,
            duration_seconds: int = 0) -> dict[str, Any]:
    return {
        "ok": ok, "date": day, "stage": stage, "error": error,
        "stories": stories, "headline": headline, "pdf": pdf,
        "videos": videos or {"one_minute": "", "three_minute": ""},
        "skipped_research": skipped_research,
        "duration_seconds": duration_seconds, "log": str(log_path.resolve()),
    }


def _acquire_lock() -> bool:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            descriptor = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "created": time.time()}))
            return True
        except FileExistsError:
            try:
                age = time.time() - LOCK.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= LOCK_MAX_AGE_SECONDS:
                return False
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass
    try:
        descriptor = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "created": time.time()}))
        return True
    except FileExistsError:
        return False


def run_pipeline(*, force: bool = False, skip_video: bool = False,
                 only_type: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    day = _today()
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = LOGS / f"pipeline_{stamp}.log"
    result = _result(day, log_path, ok=False, stage="lock")
    lock_owned = False

    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        try:
            lock_owned = _acquire_lock()
            if not lock_owned:
                result["error"] = "already in progress"
                return result

            current = _read_programme()
            current_success = bool(
                current
                and current.get("programme_status") == "SUCCESS"
                and current.get("programme_date") == day
            )
            pdf_path = _find_pdf(day) if current_success else None
            skipped_research = current_success and not force

            if not skipped_research:
                before_mtime = PROGRAMME.stat().st_mtime_ns if PROGRAMME.exists() else None
                result["stage"] = "research"
                step = _run_step("collect + research + PDF", [sys.executable, "-m", "src.run_daily_programme"], log)
                if step.returncode != 0:
                    raise RuntimeError(f"daily programme command exited {step.returncode}: {_safe_tail(step.stderr or step.stdout)}")
                after_mtime = PROGRAMME.stat().st_mtime_ns if PROGRAMME.exists() else None
                programme = _read_programme()
                if after_mtime is None or after_mtime == before_mtime:
                    raise RuntimeError("daily_programme.json was not rewritten by today's research run")
                if not programme or programme.get("programme_status") != "SUCCESS" or programme.get("programme_date") != day:
                    raise RuntimeError("research did not produce a SUCCESS programme dated today")
                pdf_path = _find_pdf(day)
                if not pdf_path:
                    raise RuntimeError("research completed but no dated PDF was found in output/reports")
            elif not pdf_path:
                raise RuntimeError("today's successful programme exists, but its dated PDF is missing")

            programme = _read_programme()
            if not programme or programme.get("programme_status") != "SUCCESS" or programme.get("programme_date") != day:
                raise RuntimeError("daily programme is not SUCCESS for today")
            result.update({
                "date": day, "stories": len(programme.get("stories", [])),
                "headline": str(programme.get("overall_headline", "")),
                "pdf": str(pdf_path.resolve()) if pdf_path else "",
                "skipped_research": skipped_research,
                "stage": "video",
            })

            if not skip_video:
                changed = _normalise_narration()
                if changed:
                    log.write("Normalized narration whitespace in daily_programme.json.\n")
                programme = _read_programme() or programme
                selected = (only_type,) if only_type else VIDEO_TYPES
                for video_type in selected:
                    command = [sys.executable, "-m", "src.generate_video", "--type", video_type,
                               "--tts", "auto", "--music", "generated"]
                    step = _run_step(f"video {video_type}", command, log)
                    mp4 = _video_path(day, video_type)
                    if not mp4.is_file():
                        hint = " (FFmpeg may be missing or rendering may have failed)" if step.returncode == 0 else ""
                        raise RuntimeError(f"expected MP4 was not created: {mp4}{hint}")
                    if step.returncode != 0:
                        raise RuntimeError(f"video {video_type} command exited {step.returncode}: {_safe_tail(step.stderr or step.stdout)}")
                    result["videos"][video_type] = str(mp4.resolve())
            result["ok"] = True
            result["stage"] = "complete"
        except Exception as error:
            detail = str(error).strip() or type(error).__name__
            result["error"] = _safe_tail(detail, lines=8, chars=1200)
            log.write(f"\nPIPELINE FAILURE [{result['stage']}]: {result['error']}\n")
            log.flush()
        finally:
            result["duration_seconds"] = int(time.monotonic() - started)
            if lock_owned:
                try:
                    LOCK.unlink()
                except FileNotFoundError:
                    pass
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the daily news pipeline for n8n.")
    parser.add_argument("--force", action="store_true", help="Run collection and research even if today's programme already exists.")
    parser.add_argument("--skip-video", action="store_true", help="Run collection/research and PDF only; do not render videos.")
    parser.add_argument("--only-type", choices=VIDEO_TYPES, help="Render only this video type.")
    args = parser.parse_args(argv)
    try:
        payload = run_pipeline(force=args.force, skip_video=args.skip_video, only_type=args.only_type)
    except BaseException as error:
        day = _today()
        LOGS.mkdir(parents=True, exist_ok=True)
        log_path = LOGS / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
        payload = _result(day, log_path, ok=False, stage="startup", error=str(error) or type(error).__name__)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(_redact(traceback.format_exc()))
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
