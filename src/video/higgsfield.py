"""Opt-in Higgsfield B-roll generation with a local, content-addressed cache.

The official SDK handles Higgsfield's queued request, polling, and result
lifecycle. Generation stays disabled unless HIGGSFIELD_ENABLED is explicitly
set; tests inject a client and never submit paid requests.
"""
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

MODEL_ID = "kling-video/v3.0/4k/text-to-video"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "output" / "video_assets" / "higgsfield"


def credentials_configured(config) -> bool:
    return bool(config.higgsfield_api_key_id and config.higgsfield_api_key_secret)


def build_story_prompt(scene: Dict[str, Any]) -> str:
    """Ask for illustrative B-roll using only the article's collected wording."""
    headline = str(scene.get("headline") or scene.get("title") or "").strip()
    evidence = " ".join(str(item) for item in scene.get("key_points", []) if item).strip()
    subject = "; ".join(part for part in (headline, evidence) if part)
    if not subject:
        return ""
    return (
        "Create realistic, restrained editorial B-roll illustrating this AI news topic: "
        f"{subject[:900]}. 16:9 newsroom-quality footage, neutral professional lighting. "
        "Illustrative imagery only: no text, logos, named people, screenshots, "
        "company announcements, or depictions of specific real events or products."
    )


def _video_url(result: Any) -> str:
    """Read the documented completed `video` result from the official SDK."""
    if not isinstance(result, dict):
        return ""
    video = result.get("video")
    if isinstance(video, str) and video.startswith(("https://", "http://")):
        return video
    if isinstance(video, dict):
        url = video.get("url")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            return url
    return ""


def _cache_path(scene: Dict[str, Any], prompt: str, config, cache_dir: Path) -> Path:
    identity = "\n".join((
        MODEL_ID,
        prompt,
        str(config.higgsfield_clip_seconds),
        config.higgsfield_aspect_ratio,
    ))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.mp4"


def generate_story_clip(
    scene: Dict[str, Any],
    config,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    client_factory: Optional[Callable[..., Any]] = None,
    downloader: Optional[Callable[..., Any]] = None,
) -> Tuple[Optional[Path], str]:
    """Return a cached/generated clip; failures leave the local renderer usable."""
    prompt = build_story_prompt(scene)
    if not prompt:
        return None, "No article evidence for a story-specific visual prompt."
    cache_dir = Path(cache_dir)
    destination = _cache_path(scene, prompt, config, cache_dir)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination, ""
    if not credentials_configured(config):
        return None, "Higgsfield credentials are not configured."

    try:
        if client_factory is None:
            import higgsfield_client
            client_factory = higgsfield_client.SyncClient
        if downloader is None:
            import requests
            downloader = requests.get

        credentials = f"{config.higgsfield_api_key_id}:{config.higgsfield_api_key_secret}"
        client = client_factory(api_key=credentials, timeout=config.higgsfield_timeout)
        result = client.subscribe(
            MODEL_ID,
            arguments={
                "prompt": prompt,
                "sound": "off",
                "duration": config.higgsfield_clip_seconds,
                "aspect_ratio": config.higgsfield_aspect_ratio,
                "multi_shots": False,
            },
        )
        url = _video_url(result)
        if not url:
            return None, "Higgsfield completed without a usable video result."
        response = downloader(url, stream=True, timeout=config.higgsfield_timeout)
        response.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.mp4")
        try:
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
            if temporary.stat().st_size == 0:
                return None, "Higgsfield returned an empty video file."
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination, ""
    except Exception as exc:
        # SDK/HTTP exceptions may contain request details; never surface them.
        return None, f"Higgsfield unavailable ({type(exc).__name__}); using local visuals."


def generate_story_clips(
    plan: Dict[str, Any],
    config,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    client_factory: Optional[Callable[..., Any]] = None,
    downloader: Optional[Callable[..., Any]] = None,
) -> Tuple[List[Optional[Path]], List[str]]:
    """Generate at most the configured number of three-minute story clips."""
    scenes = plan.get("scenes", [])
    clips: List[Optional[Path]] = [None] * len(scenes)
    warnings: List[str] = []
    if plan.get("programme_type") != "three_minute" or not config.higgsfield_enabled:
        return clips, warnings
    if not config.higgsfield_allow_paid:
        return clips, ["Higgsfield generation is opt-in but paid requests are not approved; using local visuals."]
    if not credentials_configured(config):
        return clips, ["Higgsfield is enabled but its API credentials are not configured; using local visuals."]

    remaining = config.higgsfield_max_clips
    for index, scene in enumerate(scenes):
        if remaining <= 0:
            break
        if scene.get("kind") != "story":
            continue
        clip, warning = generate_story_clip(
            scene, config, cache_dir, client_factory=client_factory,
            downloader=downloader,
        )
        if clip:
            clips[index] = clip
            remaining -= 1
        elif warning:
            warnings.append(warning)
    return clips, list(dict.fromkeys(warnings))


def extract_clip_frames(ffmpeg: str, clip: Path, output_dir: Path, fps: int) -> Optional[Path]:
    """Decode clip frames to disk; rendering opens only the frame needed at time t."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip),
             "-vf", f"fps={int(fps)},scale=960:540:force_original_aspect_ratio=increase,crop=960:540",
             "-q:v", "3", str(pattern)],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    frames = sorted(output_dir.glob("frame_*.jpg"))
    return output_dir if result.returncode == 0 and frames else None
