"""Video pipeline settings, read from environment variables (never hard-coded keys)."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

TTS_PROVIDERS = ("auto", "elevenlabs", "windows")
MUSIC_MODES = ("generated", "file", "off")

# A premade ElevenLabs voice (used as the documented example voice). Override
# with ELEVENLABS_VOICE_ID. Whether a given voice works on a free account is
# decided by ElevenLabs, not by this project.
DEFAULT_ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"

WIDTH, HEIGHT = 1920, 1080


def _flag(value: Optional[str], default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class VideoConfig:
    tts_provider: str = "auto"
    elevenlabs_api_key: str = field(default="", repr=False)  # never printed
    elevenlabs_voice_id: str = DEFAULT_ELEVENLABS_VOICE_ID
    elevenlabs_model: str = DEFAULT_ELEVENLABS_MODEL
    elevenlabs_timeout: float = 90.0
    # auto mode: fall back to Windows speech when ElevenLabs fails
    fallback_to_windows: bool = True
    music_mode: str = "generated"
    music_file: str = ""
    music_volume: float = 0.16
    image_dir: str = ""
    fps: int = 30
    scene_lead_seconds: float = 0.35
    scene_tail_seconds: float = 0.45
    bumper_seconds: float = 2.6
    closing_seconds: float = 3.4

    @property
    def has_elevenlabs_key(self) -> bool:
        return bool(self.elevenlabs_api_key.strip())

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, **overrides) -> "VideoConfig":
        if env is None:
            try:  # the project already uses python-dotenv for .env
                from dotenv import load_dotenv
                load_dotenv()
            except Exception:
                pass
            env = os.environ
        provider = (env.get("VIDEO_TTS_PROVIDER") or "auto").strip().lower()
        music = (env.get("VIDEO_MUSIC") or "generated").strip().lower()
        values = dict(
            tts_provider=provider if provider in TTS_PROVIDERS else "auto",
            elevenlabs_api_key=(env.get("ELEVENLABS_API_KEY") or "").strip(),
            elevenlabs_voice_id=(env.get("ELEVENLABS_VOICE_ID") or DEFAULT_ELEVENLABS_VOICE_ID).strip(),
            elevenlabs_model=(env.get("ELEVENLABS_MODEL") or DEFAULT_ELEVENLABS_MODEL).strip(),
            fallback_to_windows=_flag(env.get("VIDEO_TTS_FALLBACK"), True),
            music_mode=music if music in MUSIC_MODES else "generated",
            music_file=(env.get("VIDEO_MUSIC_FILE") or "").strip(),
            music_volume=min(max(_float(env.get("VIDEO_MUSIC_VOLUME"), 0.16), 0.0), 1.0),
            image_dir=(env.get("VIDEO_IMAGE_DIR") or "").strip(),
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def default_image_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "output" / "video_assets" / "images"