"""Export a saved session as text or audio. Everything stays on this computer.

text      transcript.txt as saved: [hh:mm:ss] Speaker N: text
srt       subtitles (one cue per Whisper segment) to play alongside the recording
recording the lecture audio as .mp3 or .wav
spoken    the transcript read aloud by the offline Windows voice, as .wav or .mp3
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf


def _require(folder: Path, name: str) -> Path:
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return path


def segments(folder: Path) -> list[dict]:
    rows = []
    for raw in _require(folder, "segments.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if "chunk" in row:
            rows.extend(row["segments"])
    return rows


def export_text(folder: Path, dest: Path):
    shutil.copyfile(_require(folder, "transcript.txt"), dest)


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def export_srt(folder: Path, dest: Path):
    cues = []
    for i, s in enumerate(segments(folder), 1):
        who = f"Speaker {s['speaker']}:" if s.get("speaker") is not None else ""
        cues.append(f"{i}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n{who}{s['text']}\n")
    dest.write_text("\n".join(cues), encoding="utf-8")


def _write_audio(src_wav_or_flac: Path, dest: Path):
    data, sr = sf.read(str(src_wav_or_flac), dtype="float32")
    suffix = dest.suffix.lower()
    if suffix == ".mp3":
        sf.write(str(dest), data, sr, format="MP3", subtype="MPEG_LAYER_III")
    elif suffix == ".wav":
        sf.write(str(dest), data, sr, format="WAV", subtype="PCM_16")
    else:
        raise ValueError(f"Audio can be exported as .mp3 or .wav, not {suffix or 'no extension'}")


def export_recording(folder: Path, dest: Path):
    src = folder / "audio.flac"
    if not src.is_file():
        raise FileNotFoundError(f"This session has no recording ({src} is missing). "
                                "Recording is saved only while save_audio is true in config.yaml.")
    _write_audio(src, dest)


def spoken_text(folder: Path) -> str:
    """The words of the lecture, without timestamps, for reading aloud."""
    return "".join(s["text"] for s in segments(folder)).strip()


def export_spoken(folder: Path, dest: Path):
    text = spoken_text(folder)
    if not text:
        raise ValueError("The transcript is empty, so there is nothing to read aloud.")
    with tempfile.TemporaryDirectory() as tmp:
        txt = Path(tmp) / "text.txt"
        wav = Path(tmp) / "spoken.wav"
        txt.write_text(text, encoding="utf-8")
        q = lambda p: "'" + str(p).replace("'", "''") + "'"  # PowerShell single-quoted string
        script = ("$ErrorActionPreference = 'Stop'; Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.SetOutputToWaveFile({q(wav)}); "
                  f"$s.Speak([IO.File]::ReadAllText({q(txt)}, [Text.Encoding]::UTF8)); $s.Dispose()")
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                           capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode != 0 or not wav.is_file():
            raise RuntimeError(f"Windows speech synthesis failed (exit {r.returncode}):\n{r.stderr or r.stdout}")
        if dest.suffix.lower() == ".wav":
            shutil.copyfile(wav, dest)
        else:
            _write_audio(wav, dest)


KINDS = {
    "text": ("Transcript (text)", "Text (*.txt)", ".txt", export_text),
    "srt": ("Subtitles with timestamps", "Subtitles (*.srt)", ".srt", export_srt),
    "recording": ("Lecture recording (audio)", "MP3 audio (*.mp3);;WAV audio (*.wav)", ".mp3", export_recording),
    "spoken": ("Transcript read aloud (audio)", "WAV audio (*.wav);;MP3 audio (*.mp3)", ".wav", export_spoken),
}
