"""Diagnose why speech synthesis fails on the real narration.

Run from the project root:  python voice_diag.py
Makes no network or Groq calls. Writes test WAV files under output/.
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def run_ps(script_text, label):
    script = OUT / f".diag_{label}.ps1"
    # utf-8-sig (BOM) makes Windows PowerShell 5.1 read the file as UTF-8.
    script.write_text(script_text, encoding="utf-8-sig")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script)],
            capture_output=True, text=True, timeout=180,
        )
    finally:
        script.unlink(missing_ok=True)
    return result


def q(value):
    return str(value).replace("'", "''")


def old_style(text, dest):
    """Text embedded inside the PowerShell script (how the original worked)."""
    return (
        "$voice = New-Object -ComObject SAPI.SpVoice\n"
        "$stream = New-Object -ComObject SAPI.SpFileStream\n"
        f"$stream.Open('{q(dest)}', 3, $false)\n"
        "$voice.AudioOutputStream = $stream\n"
        f"$voice.Speak('{q(text)}')\n"
        "$stream.Close()\n"
    )


def robust(txt_path, dest):
    """Text read from a UTF-8 file, never embedded in the script."""
    return (
        "$ErrorActionPreference = 'Stop'\n"
        f"$text = [System.IO.File]::ReadAllText('{q(txt_path)}', "
        "[System.Text.Encoding]::UTF8)\n"
        "$voice = New-Object -ComObject SAPI.SpVoice\n"
        "$stream = New-Object -ComObject SAPI.SpFileStream\n"
        f"$stream.Open('{q(dest)}', 3, $false)\n"
        "$voice.AudioOutputStream = $stream\n"
        "$null = $voice.Speak($text)\n"
        "$stream.Close()\n"
    )


def report(label, result, dest):
    size = dest.stat().st_size if dest.exists() else 0
    print(f"[{label}] returncode={result.returncode} wav_bytes={size}")
    if result.stderr.strip():
        print(f"[{label}] stderr: {result.stderr.strip()[:600]}")
    if result.stdout.strip():
        print(f"[{label}] stdout: {result.stdout.strip()[:300]}")


if os.name != "nt":
    raise SystemExit("This diagnostic is for Windows (SAPI).")

programme = json.loads(
    (OUT / "daily_programme.json").read_text(encoding="utf-8")
)
text = programme["programmes"]["one_minute"]["narration"]
print(f"narration length: {len(text)} characters")
odd = sorted({c for c in text if ord(c) > 127})
print("non-ASCII characters:", [(c, hex(ord(c))) for c in odd] or "none")

# A) original approach, text embedded in the script
dest_a = OUT / "diag_old_style.wav"
dest_a.unlink(missing_ok=True)
report("A old-style", run_ps(old_style(text, dest_a.resolve()), "a"), dest_a)

# B) robust approach, text passed through a UTF-8 file
txt = OUT / "diag_text.txt"
txt.write_text(text, encoding="utf-8")
dest_b = OUT / "diag_robust.wav"
dest_b.unlink(missing_ok=True)
report("B robust", run_ps(robust(txt.resolve(), dest_b.resolve()), "b"), dest_b)
txt.unlink(missing_ok=True)

print("\nListen to output/diag_robust.wav. If B works and A fails, the fix is "
      "to pass the text through a UTF-8 file instead of embedding it.")