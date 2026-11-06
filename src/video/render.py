"""FFmpeg video rendering, muxing, MP4 validation and atomic replacement."""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import HEIGHT, WIDTH, VideoConfig


class RenderError(RuntimeError):
    pass


def build_video_command(ffmpeg: str, out_path: Path, fps: int = 30,
                        width: int = WIDTH, height: int = HEIGHT) -> List[str]:
    """Raw RGB frames on stdin -> H.264 (CFR, BT.709, yuv420p), no audio."""
    return [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-",
        "-an",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-r", str(fps), "-fps_mode", "cfr",
        "-movflags", "+faststart",
        str(out_path),
    ]


def build_mux_command(ffmpeg: str, video: Path, audio: Path, out_path: Path,
                      seconds: float) -> List[str]:
    """Copy the video stream, encode the mixed audio to AAC, trim to the exact length."""
    return [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", f"{seconds:.3f}",
        "-movflags", "+faststart",
        str(out_path),
    ]


def frame_count(seconds: float, fps: int) -> int:
    return max(int(round(seconds * fps)), 1)


def render_silent_video(ffmpeg: str, frame_at: Callable[[float], Any], seconds: float,
                        out_path: Path, fps: int = 30,
                        progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Pipe every frame to FFmpeg. ``frame_at(t)`` returns a 1920x1080 RGB PIL image."""
    total = frame_count(seconds, fps)
    cmd = build_video_command(ffmpeg, out_path, fps)
    err_file = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=err_file)
    except OSError as exc:
        err_file.close()
        raise RenderError(f"Could not start FFmpeg: {exc}") from exc
    try:
        for i in range(total):
            frame = frame_at(i / float(fps))
            if frame.size != (WIDTH, HEIGHT):
                raise RenderError(f"Frame {i} is {frame.size}, expected {(WIDTH, HEIGHT)}")
            proc.stdin.write(frame.tobytes())
            if progress and (i % fps == 0 or i == total - 1):
                progress(i + 1, total)
        proc.stdin.close()
        code = proc.wait(timeout=600)
    except BrokenPipeError:
        proc.wait(timeout=30)
        err_file.seek(0)
        raise RenderError("FFmpeg stopped while receiving frames: "
                          + err_file.read().decode("utf-8", "replace")[-400:]) from None
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
    if code != 0 or not out_path.exists():
        err_file.seek(0)
        msg = err_file.read().decode("utf-8", "replace")[-400:]
        err_file.close()
        raise RenderError(f"FFmpeg video encode failed (exit {code}): {msg}")
    err_file.close()


def run_ffmpeg(cmd: List[str], what: str, timeout: int = 300) -> None:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderError(f"{what} could not run: {exc}") from exc
    if res.returncode != 0:
        raise RenderError(f"{what} failed: {res.stderr.strip()[-400:]}")


# --------------------------------------------------------------- validation
def probe(ffprobe: str, path: Path) -> Dict[str, Any]:
    cmd = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RenderError(f"ffprobe could not read the file: {res.stderr.strip()[-200:]}")
    return json.loads(res.stdout or "{}")


def audio_max_volume_db(ffmpeg: str, path: Path) -> Optional[float]:
    res = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", str(path), "-vn",
                          "-af", "volumedetect", "-f", "null", os.devnull],
                         capture_output=True, text=True, timeout=120)
    m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", res.stderr)
    if not m:
        return None
    return float("-inf") if m.group(1) == "-inf" else float(m.group(1))


def check_probe(info: Dict[str, Any], expected_seconds: float, fps: int = 30,
                tolerance: float = 0.6) -> List[str]:
    """Problems found in an ffprobe result (empty list = valid MP4)."""
    problems: List[str] = []
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video:
        problems.append("no video stream")
    else:
        if video.get("codec_name") != "h264":
            problems.append(f"video codec is {video.get('codec_name')}, expected h264")
        if (video.get("width"), video.get("height")) != (WIDTH, HEIGHT):
            problems.append(f"resolution is {video.get('width')}x{video.get('height')}, "
                            f"expected {WIDTH}x{HEIGHT}")
        rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
        try:
            num, den = rate.split("/")
            actual = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            actual = 0.0
        if abs(actual - fps) > 0.05:
            problems.append(f"frame rate is {actual:.2f}, expected {fps}")
        if video.get("pix_fmt") not in (None, "yuv420p"):
            problems.append(f"pixel format is {video.get('pix_fmt')}, expected yuv420p")
    if not audio:
        problems.append("no audio stream")
    elif audio.get("codec_name") != "aac":
        problems.append(f"audio codec is {audio.get('codec_name')}, expected aac")
    try:
        duration = float((info.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    if abs(duration - expected_seconds) > tolerance:
        problems.append(f"duration is {duration:.2f}s, expected about {expected_seconds:.2f}s")
    return problems


def validate_mp4(path: Path, expected_seconds: float, fps: int, ffmpeg: str,
                 ffprobe: Optional[str]) -> List[str]:
    """Full validation of a rendered MP4; returns a list of problems."""
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1024:
        return ["file is missing or empty"]
    if not ffprobe:
        return ["ffprobe was not found, so the MP4 could not be verified"]
    try:
        problems = check_probe(probe(ffprobe, path), expected_seconds, fps)
    except (RenderError, ValueError, OSError, subprocess.SubprocessError) as exc:
        return [f"could not inspect the MP4: {exc}"]
    if not problems:
        peak = audio_max_volume_db(ffmpeg, path)
        if peak is None or peak < -45.0:
            problems.append(f"audio is silent (peak {peak} dB)")
    return problems


def replace_atomically(temp: Path, final: Path) -> None:
    """Swap ``temp`` into place. The old file is only replaced by a complete new one."""
    os.replace(str(temp), str(final))