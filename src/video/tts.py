"""Narration audio: ElevenLabs (optional) and Windows speech (free fallback).

Rules this module enforces:
* The narration text is sent exactly as given (only typographic characters
  such as long dashes are normalised, the same as for on-screen text).
* A failure is reported with its real cause. Placeholder tones are never
  substituted for speech.
* The API key is only ever sent as a request header and never appears in
  messages, logs or exceptions.
"""
import base64
import hashlib
import json
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from . import voiceover
from .config import VideoConfig
from .planner import clean_display_text
from .timing import (WordTiming, align_to_text, estimate_word_timings, shift_words,
                     words_from_alignment)

SAMPLE_RATE = 48000
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
# 128 kbps MP3 is the documented default; higher-quality formats need paid tiers.
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"


class TTSError(RuntimeError):
    """Speech could not be produced. ``kind`` says why (auth, quota, plan, rate_limit,
    network, server, bad_request, unavailable, ffmpeg)."""

    def __init__(self, message: str, kind: str = "unknown", provider: str = ""):
        super().__init__(message)
        self.kind = kind
        self.provider = provider


@dataclass
class SceneAudio:
    path: Path                      # 48 kHz mono WAV: lead silence + speech + tail silence
    duration: float                 # total seconds, equals the scene length
    words: List[WordTiming]         # times relative to the scene start
    provider: str
    estimated_timing: bool = False  # True when word times are estimated, not measured


# ---------------------------------------------------------------- audio helpers
def convert_to_wav(ffmpeg: str, src: Path, dst: Path) -> Path:
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TTSError(f"FFmpeg could not convert audio: {exc}", "ffmpeg") from exc
    if res.returncode != 0 or not dst.exists():
        raise TTSError(f"FFmpeg could not convert audio: {res.stderr.strip()[:300]}", "ffmpeg")
    return dst


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def write_silence(path: Path, seconds: float) -> Path:
    frames = int(round(max(seconds, 0.0) * SAMPLE_RATE))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * frames)
    return path


def join_wavs(paths: Sequence[Path], out: Path) -> Path:
    """Concatenate 48 kHz mono 16-bit WAVs (all produced by this module)."""
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(SAMPLE_RATE)
        for p in paths:
            with wave.open(str(p), "rb") as src:
                if (src.getnchannels(), src.getsampwidth(), src.getframerate()) != (1, 2, SAMPLE_RATE):
                    raise TTSError(f"Unexpected audio format in {p.name}", "ffmpeg")
                dst.writeframes(src.readframes(src.getnframes()))
    return out


def pad_speech(speech_wav: Path, out: Path, lead: float, tail: float) -> Path:
    tmp = out.parent / (out.stem + "_pad")
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        a = write_silence(tmp / "lead.wav", lead)
        b = write_silence(tmp / "tail.wav", tail)
        return join_wavs([a, speech_wav, b], out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ providers
class ElevenLabsProvider:
    name = "ElevenLabs"

    def __init__(self, config: VideoConfig, ffmpeg: str, cache_dir: Path,
                 http_post: Optional[Callable[..., Any]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not config.has_elevenlabs_key:
            raise TTSError("ELEVENLABS_API_KEY is not set.", "auth", self.name)
        self.config = config
        self.ffmpeg = ffmpeg
        self.cache_dir = Path(cache_dir)
        self._post = http_post or requests.post
        self._sleep = sleep
        self.requests_made = 0
        self.cache_hits = 0

    # -- request ----------------------------------------------------------
    def _body(self, text: str, previous_text: str, next_text: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "text": text,
            "model_id": self.config.elevenlabs_model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0,
                               "use_speaker_boost": True},
        }
        if previous_text:
            body["previous_text"] = previous_text
        if next_text:
            body["next_text"] = next_text
        return body

    @staticmethod
    def error_from_response(status: int, payload: Any) -> TTSError:
        """Translate an HTTP error into a clear TTSError (never includes the key)."""
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        code, message = "", ""
        if isinstance(detail, dict):
            code = str(detail.get("status") or detail.get("code") or "")
            message = str(detail.get("message") or "")
        elif detail:
            message = str(detail)
        message = message.strip()[:300]
        low = f"{code} {message}".lower()
        if "quota" in low or ("credit" in low and ("exceed" in low or "insufficient" in low)):
            kind = "quota"
            hint = "ElevenLabs credits for this period are used up."
        elif status == 401:
            kind, hint = "auth", "ElevenLabs rejected the API key (check ELEVENLABS_API_KEY)."
        elif status == 402 or "paid_plan" in low or "payment" in low:
            kind, hint = "plan", "This voice or feature needs a paid ElevenLabs plan."
        elif status == 429:
            kind, hint = "rate_limit", "ElevenLabs rate limit / concurrency limit reached."
        elif status >= 500:
            kind, hint = "server", "ElevenLabs had a server problem."
        else:
            kind, hint = "bad_request", "ElevenLabs refused the request."
        return TTSError(f"{hint} (HTTP {status}{', ' + code if code else ''}) {message}".strip(),
                        kind, "ElevenLabs")

    def _cache_key(self, text: str, previous_text: str, next_text: str) -> str:
        raw = json.dumps([self.config.elevenlabs_voice_id, self.config.elevenlabs_model,
                          ELEVENLABS_OUTPUT_FORMAT, text, previous_text, next_text],
                         ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def _fetch(self, text: str, previous_text: str, next_text: str) -> Dict[str, Any]:
        url = ELEVENLABS_URL.format(voice_id=self.config.elevenlabs_voice_id)
        headers = {"xi-api-key": self.config.elevenlabs_api_key,
                   "Content-Type": "application/json", "Accept": "application/json"}
        last: Optional[TTSError] = None
        for attempt in range(3):
            try:
                resp = self._post(url, headers=headers,
                                  params={"output_format": ELEVENLABS_OUTPUT_FORMAT},
                                  json=self._body(text, previous_text, next_text),
                                  timeout=self.config.elevenlabs_timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = TTSError(f"Could not reach ElevenLabs ({type(exc).__name__}).",
                                "network", self.name)
            except requests.RequestException as exc:
                raise TTSError(f"ElevenLabs request failed ({type(exc).__name__}).",
                               "network", self.name) from exc
            else:
                self.requests_made += 1
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise TTSError("ElevenLabs returned a response that is not JSON.",
                                       "server", self.name) from exc
                try:
                    payload = resp.json()
                except ValueError:
                    payload = {"detail": (getattr(resp, "text", "") or "")[:200]}
                err = self.error_from_response(resp.status_code, payload)
                if err.kind not in ("rate_limit", "server"):
                    raise err
                last = err
            if attempt < 2:
                self._sleep(1.5 * (attempt + 1))
        assert last is not None
        raise last

    # -- public -----------------------------------------------------------
    def synthesize(self, text: str, out_wav: Path, lead: float, tail: float,
                   previous_text: str = "", next_text: str = "") -> SceneAudio:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(text, previous_text, next_text)
        mp3 = self.cache_dir / f"eleven_{key}.mp3"
        meta = self.cache_dir / f"eleven_{key}.json"
        alignment: Optional[Dict[str, Any]] = None
        if mp3.exists() and meta.exists():  # no credits spent twice for the same text
            try:
                alignment = json.loads(meta.read_text(encoding="utf-8"))
                self.cache_hits += 1
            except (OSError, ValueError):
                alignment = None
                mp3.unlink(missing_ok=True)
        if alignment is None:
            data = self._fetch(text, previous_text, next_text)
            try:
                audio = base64.b64decode(data["audio_base64"])
            except (KeyError, ValueError, TypeError) as exc:
                raise TTSError("ElevenLabs response had no usable audio.", "server",
                               self.name) from exc
            if not audio:
                raise TTSError("ElevenLabs returned empty audio.", "server", self.name)
            alignment = data.get("alignment") or {}
            mp3.write_bytes(audio)
            meta.write_text(json.dumps(alignment), encoding="utf-8")

        raw = out_wav.parent / (out_wav.stem + "_speech.wav")
        convert_to_wav(self.ffmpeg, mp3, raw)
        speech = wav_seconds(raw)
        measured = words_from_alignment(alignment.get("characters", []),
                                        alignment.get("character_start_times_seconds", []),
                                        alignment.get("character_end_times_seconds", []))
        words = align_to_text(measured, text, speech)
        estimated = [w.word for w in measured] != text.split()
        pad_speech(raw, out_wav, lead, tail)
        raw.unlink(missing_ok=True)
        return SceneAudio(out_wav, lead + speech + tail, shift_words(words, lead),
                          self.name, estimated)


class WindowsProvider:
    name = "Windows speech"

    def __init__(self, ffmpeg: str, speak: Optional[Callable[[str, Path], bool]] = None):
        self.ffmpeg = ffmpeg
        self._speak = speak or (lambda text, path: voiceover.try_native_speech(text, path))

    def synthesize(self, text: str, out_wav: Path, lead: float, tail: float,
                   previous_text: str = "", next_text: str = "") -> SceneAudio:
        native = out_wav.parent / (out_wav.stem + "_native.wav")
        try:
            ok = self._speak(text, native)
        except Exception as exc:  # engine crashed
            raise TTSError(f"Windows speech failed: {type(exc).__name__}: {exc}",
                           "unavailable", self.name) from exc
        if not ok or not native.exists():
            raise TTSError(
                "Windows speech produced no audible voice. Run "
                "'python -m src.video.voiceover' to test it.", "unavailable", self.name)
        raw = out_wav.parent / (out_wav.stem + "_speech.wav")
        convert_to_wav(self.ffmpeg, native, raw)
        speech = wav_seconds(raw)
        words = estimate_word_timings(text, speech)
        pad_speech(raw, out_wav, lead, tail)
        native.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        return SceneAudio(out_wav, lead + speech + tail, shift_words(words, lead),
                          self.name, True)


# ----------------------------------------------------------------- orchestration
def provider_order(config: VideoConfig) -> List[str]:
    """Which providers to try, in order."""
    if config.tts_provider == "windows":
        return ["windows"]
    if config.tts_provider == "elevenlabs":
        return ["elevenlabs"]
    order = ["elevenlabs"] if config.has_elevenlabs_key else []
    if config.fallback_to_windows or not order:
        order.append("windows")
    return order


def synthesize_scenes(
    narrations: Sequence[str],
    config: VideoConfig,
    work_dir: Path,
    ffmpeg: str,
    providers: Optional[Dict[str, Any]] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Speak every scene with ONE provider (consistent voice).

    Returns {"audio": [SceneAudio...], "provider": str, "warnings": [str...]}.
    Raises TTSError carrying every real failure when no provider works.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    texts = [clean_display_text(t) for t in narrations]
    warnings: List[str] = []
    failures: List[str] = []
    order = provider_order(config)
    last_error: Optional[TTSError] = None

    for name in order:
        try:
            if providers and name in providers:
                provider = providers[name]
            elif name == "elevenlabs":
                provider = ElevenLabsProvider(config, ffmpeg, work_dir.parent / "tts_cache")
            else:
                provider = WindowsProvider(ffmpeg)
            results: List[SceneAudio] = []
            for i, text in enumerate(texts):
                out = work_dir / f"scene_{i + 1:02d}.wav"
                results.append(provider.synthesize(
                    text, out, config.scene_lead_seconds, config.scene_tail_seconds,
                    previous_text=texts[i - 1] if i else "",
                    next_text=texts[i + 1] if i + 1 < len(texts) else ""))
            return {"audio": results, "provider": getattr(provider, "name", name),
                    "warnings": warnings}
        except TTSError as exc:
            last_error = exc
            failures.append(f"{name}: [{exc.kind}] {exc}")
            if name != order[-1]:
                msg = (f"{exc.provider or name} speech failed ({exc.kind}): {exc} "
                       f"- falling back to the next voice.")
                warnings.append(msg)
                log("WARNING: " + msg)

    detail = "; ".join(failures) or "no speech provider is available"
    raise TTSError(f"No narration audio could be produced - {detail}",
                   last_error.kind if last_error else "unavailable")