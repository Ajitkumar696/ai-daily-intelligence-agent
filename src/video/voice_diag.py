import os
import subprocess
import tempfile
import wave
from pathlib import Path

SPEECH_TIMEOUT_SECONDS = 180
# A real spoken WAV is far larger than this; guards against empty output.
MIN_WAV_BYTES = 1000


def _is_windows():
    return os.name == "nt"


def _ps_quote(value):
    """Quote a value as a PowerShell single-quoted string literal.

    Only used for file paths, never for narration text. Curly quotes are
    treated as quote characters by PowerShell, so they are doubled too.
    """
    text = str(value)
    for quote in ("'", "\u2018", "\u2019", "\u201a", "\u201b"):
        text = text.replace(quote, quote * 2)
    return "'" + text + "'"


def synthesize_voiceover(text, output_path, voice_rate=0):
    """Synthesize narration text into a WAV file using Windows SAPI.

    The narration is written to a UTF-8 text file and read by PowerShell, so
    apostrophes, curly quotes, dashes and other special characters can never
    break the generated script. The narration text itself is never altered.

    Raises RuntimeError with the real cause if speech cannot be produced.
    No placeholder or fake audio is ever substituted.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_text = text.strip() if isinstance(text, str) else ""
    if not clean_text:
        raise ValueError("Cannot synthesize voiceover for empty narration.")
    if not isinstance(voice_rate, int) or not -10 <= voice_rate <= 10:
        raise ValueError("voice_rate must be an integer from -10 to 10.")
    if not _is_windows():
        raise RuntimeError(
            "No speech engine is available on this system; Windows SAPI is "
            "required. Refusing to substitute placeholder audio."
        )

    output_path.unlink(missing_ok=True)
    error = _synthesize_windows_sapi(clean_text, output_path, voice_rate)
    if error is None:
        error = _check_wav(output_path)
    if error is not None:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Speech synthesis failed; refusing to substitute placeholder "
            f"audio. Cause: {error}"
        )
    return output_path


def _synthesize_windows_sapi(text, output_path, voice_rate):
    """Return None on success, or a short description of the failure."""
    with tempfile.TemporaryDirectory(prefix="voiceover_") as tmp:
        tmp_dir = Path(tmp)
        text_file = tmp_dir / "narration.txt"
        script_file = tmp_dir / "synth.ps1"
        text_file.write_text(text, encoding="utf-8")
        script = "\n".join([
            "$ErrorActionPreference = 'Stop'",
            f"$text = [System.IO.File]::ReadAllText({_ps_quote(text_file)}, "
            "[System.Text.Encoding]::UTF8)",
            "$voice = New-Object -ComObject SAPI.SpVoice",
            f"$voice.Rate = {int(voice_rate)}",
            "$stream = New-Object -ComObject SAPI.SpFileStream",
            f"$stream.Open({_ps_quote(output_path.resolve())}, 3, $false)",
            "$voice.AudioOutputStream = $stream",
            "$null = $voice.Speak($text)",
            "$stream.Close()",
            "",
        ])
        # utf-8-sig (BOM) makes Windows PowerShell 5.1 read non-ASCII paths.
        script_file.write_text(script, encoding="utf-8-sig")
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(script_file)],
                capture_output=True,
                text=True,
                timeout=SPEECH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"PowerShell timed out after {SPEECH_TIMEOUT_SECONDS}s."
        except OSError as error:
            return f"PowerShell could not be started ({error})."
    if result.returncode != 0:
        detail = " ".join((result.stderr or result.stdout or "").split())
        return f"PowerShell exit code {result.returncode}: {detail[:400]}"
    return None


def _check_wav(output_path):
    """Return None if the file is a non-empty, readable WAV."""
    if not output_path.exists() or output_path.stat().st_size < MIN_WAV_BYTES:
        return "no audio file was produced (SAPI wrote nothing)."
    try:
        with wave.open(str(output_path), "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                return "the audio file contains no samples."
    except (wave.Error, EOFError) as error:
        return f"the audio file is not a valid WAV ({error})."
    return None