import array
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional


def synthesize_voiceover(
    text: str,
    output_path: Path,
    voice_rate: int = 0
) -> Path:
    """
    Synthesize narration text into a WAV voiceover audio file.

    Uses Windows built-in SAPI / PowerShell speech synthesis when available,
    falling back to standard audio tone synthesis if no native TTS is present.
    Never alters the narration text.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Cannot synthesize voiceover for empty narration.")

    if try_native_speech(clean_text, output_path, voice_rate):
        return output_path

    # Fallback to pure-Python wave generation if OS TTS is unavailable
    _generate_placeholder_audio(output_path, duration_seconds=max(5, len(clean_text.split()) // 3))
    return output_path


_PS_QUOTE_CHARS = ("'", "\u2018", "\u2019", "\u201a", "\u201b")


def _ps_quote(value: str) -> str:
    """A PowerShell single-quoted literal. PowerShell treats the curly quotes
    as quote characters too, so every one of them must be doubled."""
    for char in _PS_QUOTE_CHARS:
        value = value.replace(char, "''")
    return f"'{value}'"


def _run_powershell_speech(script_body: str, text: str, output_path: Path,
                           voice_rate: int, timeout: int) -> bool:
    """Run a PowerShell speech script. The narration travels in a UTF-8 text file,
    never inside the script, so apostrophes, quotes and symbols cannot break it."""
    pid = os.getpid()
    temp_script = output_path.parent / f".tmp_speech_{pid}.ps1"
    temp_text = output_path.parent / f".tmp_speech_{pid}.txt"
    header = (
        "$ErrorActionPreference = 'Stop'\n"
        f"$text = [System.IO.File]::ReadAllText({_ps_quote(str(temp_text.resolve()))}, "
        "[System.Text.Encoding]::UTF8)\n"
        f"$dest = {_ps_quote(str(output_path.resolve()))}\n"
        f"$rate = {int(voice_rate)}\n"
    )
    try:
        temp_text.write_text(text, encoding="utf-8")
        temp_script.write_text(header + script_body, encoding="utf-8-sig")
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(temp_script)],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        for path in (temp_script, temp_text):
            try:
                path.unlink()
            except OSError:
                pass


def _synthesize_windows_sapi(text: str, output_path: Path, voice_rate: int = 0) -> bool:
    """Windows SAPI COM speech exported to a WAV file."""
    body = (
        "$voice = New-Object -ComObject SAPI.SpVoice\n"
        "$voice.Rate = $rate\n"
        "$stream = New-Object -ComObject SAPI.SpFileStream\n"
        "$stream.Open($dest, 3, $false)\n"
        "$voice.AudioOutputStream = $stream\n"
        "$null = $voice.Speak($text)\n"
        "$stream.Close()\n"
    )
    return _run_powershell_speech(body, text, Path(output_path), voice_rate, 120)


def _generate_placeholder_audio(output_path: Path, duration_seconds: int = 10):
    """Generate a valid PCM WAV audio file with soft ambient tone."""
    import math
    import struct

    sample_rate = 16000
    total_samples = sample_rate * duration_seconds

    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Soft 440Hz tone with low amplitude for testing / fallback
        for i in range(total_samples):
            value = int(1000 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
            data = struct.pack("<h", value)
            wav_file.writeframes(data)



LAST_SPEECH_ENGINE = ""


def wav_has_audible_speech(path: Path, min_peak: int = 500) -> bool:
    """False for missing, empty or silent WAV files (a silent file has a header and a length)."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            if not frames:
                return False
            if handle.getsampwidth() != 2:
                return True  # cannot measure other formats; trust the engine
        samples = array.array("h")
        samples.frombytes(frames[: len(frames) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        return bool(samples) and max(abs(x) for x in samples) >= min_peak
    except (wave.Error, OSError, EOFError):
        return False


def _synthesize_system_speech(text: str, output_path: Path, voice_rate: int = 0) -> bool:
    """Windows .NET System.Speech to a WAV file (most reliable Windows route)."""
    body = (
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$s.Rate = $rate\n"
        "$s.SetOutputToWaveFile($dest)\n"
        "$s.Speak($text)\n"
        "$s.Dispose()\n"
    )
    return _run_powershell_speech(body, text, Path(output_path), voice_rate, 180)


def _speech_engines():
    if os.name != "nt":
        return []
    return [
        ("System.Speech", _synthesize_system_speech),
        ("SAPI", _synthesize_windows_sapi),
    ]


def try_native_speech(text: str, output_path: Path, voice_rate: int = 0) -> bool:
    """Synthesize real, audible speech with an OS engine. False when none works.

    Each engine's output is checked: a file that exists but is silent counts
    as a failure and the next engine is tried. Never falls back to a tone, so
    callers can tell real speech from a placeholder.
    """
    global LAST_SPEECH_ENGINE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_text = text.strip()
    if not clean_text:
        return False
    for name, engine in _speech_engines():
        try:
            ok = engine(clean_text, output_path, voice_rate)
        except Exception:
            ok = False
        if ok and output_path.exists() and output_path.stat().st_size > 1000 \
                and wav_has_audible_speech(output_path):
            LAST_SPEECH_ENGINE = name
            return True
        try:
            output_path.unlink()
        except OSError:
            pass
    return False


def list_installed_voices():
    """Names of Windows voices (empty list when none or not on Windows)."""
    if os.name != "nt":
        return []
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
        "ForEach-Object { $_.VoiceInfo.Name }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=60,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def concat_wavs(paths, output_path: Path, pad_seconds: float = 0.0) -> Path:
    """Join same-format PCM WAV files, adding ``pad_seconds`` of silence after each."""
    output_path = Path(output_path)
    params = None
    with wave.open(str(output_path), "wb") as out:
        for source in paths:
            with wave.open(str(source), "rb") as handle:
                fmt = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
                if params is None:
                    params = fmt
                    out.setnchannels(fmt[0])
                    out.setsampwidth(fmt[1])
                    out.setframerate(fmt[2])
                elif fmt != params:
                    raise ValueError("Cannot join WAV files with different formats.")
                out.writeframes(handle.readframes(handle.getnframes()))
            silence_frames = int(round(params[2] * pad_seconds))
            out.writeframes(b"\x00" * (silence_frames * params[0] * params[1]))
    return output_path


def _main() -> int:
    """Voice self-test:  python -m src.video.voiceover"""
    print(f"Operating system: {os.name}")
    voices = list_installed_voices()
    print("Installed voices:", ", ".join(voices) if voices else "NONE FOUND")
    out = Path("voice_test.wav")
    ok = try_native_speech("Testing the daily briefing voice. OpenAI\u2019s \u201cquoted\u201d text, 7,600\u00d7 faster.", out)
    if ok:
        print(f"OK: engine={LAST_SPEECH_ENGINE}, {wav_duration_seconds(out):.1f}s -> {out.resolve()}")
        print("Play voice_test.wav. If you hear nothing, check the PC volume and output device.")
        return 0
    print("FAILED: no engine produced audible speech.")
    if os.name == "nt" and not voices:
        print("Windows has no speech voice installed: Settings > Time & Language > Speech > Add voices.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())