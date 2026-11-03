"""The live pipeline: microphone -> signal chain -> VAD -> segmenter -> Whisper.

Threads
  PortAudio callback   copies device blocks into Capture.blocks
  "dsp"                signal chain, VAD, level meter, recording, segmentation
  "asr"                owns Whisper. Finished chunks run first, in order, and are never
                       dropped; when none are waiting it runs the newest preview request.

Events go out through `emit(kind, payload)`, from the dsp and asr threads. Any exception
in either thread ends the session and is reported as an "error" event with the full
traceback; nothing is retried or replaced.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from .audio import FRAME, SR, Capture, InputDevice, dbfs
from .config import Config
from .engine import Transcription, WhisperEngine
from .modules import Module
from .segmenter import Chunk, ContinuousSegmenter, VadSegmenter
from .speakers import SpeakerTracker
from .vad import SileroVad

CONTEXT_CHARS = 400      # transcript tail given to Whisper as its prompt (it keeps <= 224 tokens)

# GPU work per lecture, measured on a 26-minute lecture with large-v3-turbo on the Radeon 780M:
# with previews every 2 s the GPU was busy the whole time, and 70% of that was previews.
#   final_greedy    finished chunks decoded greedily (27% less GPU, ~1 word in 13 differs)
#   preview_greedy  previews decoded greedily (12-37% less GPU per preview)
#   preview_every   seconds between the end of one preview and the start of the next (None = no previews)
PROFILES = {
    "accuracy": dict(final_greedy=False, preview_greedy=False, preview_every=0.0),
    "balanced": dict(final_greedy=False, preview_greedy=True, preview_every=4.0),
    "saver": dict(final_greedy=True, preview_greedy=True, preview_every=None),
}
MIN_WHISPER_S = 1.0      # shorter chunks are padded with silence; whisper.cpp ignores input under 100 ms
HISTORY_S = 10.0         # processed audio kept for "hear what the app hears"
LEVEL_EVERY_S = 0.05


@dataclass
class Level:
    rms_db: float
    peak_db: float
    speech_prob: float
    clipped: int


@dataclass
class Partial:
    chunk_id: int
    text: str


@dataclass
class Run:
    start: float          # seconds from session start
    speaker: int | None   # None when speaker labels are off
    text: str             # Whisper's segment texts of this run, joined unchanged


@dataclass
class Final:
    chunk_id: int
    start: float          # seconds from session start
    end: float
    text: str
    language: str
    inference_s: float
    gain_db: float
    runs: list[Run]       # consecutive segments of the same speaker


def speaker_runs(t0: float, tr: Transcription, speakers: list[int | None]) -> list[Run]:
    runs: list[Run] = []
    for seg, spk in zip(tr.segments, speakers):
        if runs and runs[-1].speaker == spk:
            runs[-1].text += seg.text
        else:
            runs.append(Run(t0 + seg.start, spk, seg.text))
    return runs or [Run(t0, None, tr.text)]


def line(run: Run) -> str:
    """One transcript line: [hh:mm:ss] Speaker N: text (the text exactly as Whisper wrote it)."""
    who = f" Speaker {run.speaker}:" if run.speaker is not None else ""
    sep = "" if run.text[:1].isspace() else " "
    return f"[{hms(run.start)}]{who}{sep}{run.text}"


@dataclass
class Status:
    backlog_s: float      # audio waiting for Whisper
    last_rtf: float | None  # inference time / audio time of the last finished chunk
    overflows: int
    chunks: int


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


@dataclass
class Session:
    cfg: Config
    module: str
    folder: Path
    device: InputDevice
    raw_capture: bool
    gpu: str
    segmenter: VadSegmenter | ContinuousSegmenter = field(init=False)
    text: str = ""
    chunks: int = 0
    pending: int = 0
    overflows: int = 0
    stopping: bool = False
    closed: bool = False
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        self._lock = threading.Lock()   # file writes come from the dsp, asr and UI threads
        c = self.cfg
        if c.segmentation == "vad":
            self.segmenter = VadSegmenter(c.vad_threshold, c.vad_min_silence_ms, c.vad_speech_pad_ms, c.max_chunk_s)
        else:
            self.segmenter = ContinuousSegmenter(c.continuous_chunk_s)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.txt = open(self.folder / "transcript.txt", "w", encoding="utf-8", buffering=1)
        self.jsonl = open(self.folder / "segments.jsonl", "w", encoding="utf-8", buffering=1)
        self.log = open(self.folder / "log.txt", "w", encoding="utf-8", buffering=1)
        self.flac = (sf.SoundFile(str(self.folder / "audio.flac"), "w", SR, 1, "PCM_16", format="FLAC")
                     if c.save_audio else None)
        self._write_meta(None)

    def _write_meta(self, error: str | None):
        meta = {
            "module": self.module,
            "started": self.started,
            "ended": datetime.now().isoformat(timespec="seconds") if self.closed else None,
            "device": self.device.name,
            "device_samplerate": self.device.samplerate,
            "raw_capture": self.raw_capture,
            "gpu": self.gpu,
            "chunks": self.chunks,
            "input_overflows": self.overflows,
            "error": error,
            "config": asdict(self.cfg),
        }
        (self.folder / "session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_log(self, line: str):
        with self._lock:
            if not self.closed:
                self.log.write(line + "\n")

    def write_event(self, event: dict):
        with self._lock:
            if not self.closed:
                self.jsonl.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_final(self, chunk: Chunk, tr: Transcription, gain_db: float, speakers: list[int | None]) -> Final:
        t0 = chunk.start / SR
        runs = speaker_runs(t0, tr, speakers)
        with self._lock:
            for run in runs:
                self.txt.write(line(run) + "\n")
            self.jsonl.write(json.dumps({
                "chunk": chunk.id,
                "start": round(t0, 3),
                "end": round(t0 + chunk.seconds, 3),
                "language": tr.language,
                "gain_db": round(gain_db, 2),
                "inference_s": round(tr.inference_s, 3),
                "segments": [{"start": round(t0 + s.start, 2), "end": round(t0 + s.end, 2), "speaker": spk,
                              "text": s.text} for s, spk in zip(tr.segments, speakers)],
            }, ensure_ascii=False) + "\n")
            self.text += tr.text
            self.chunks += 1
        return Final(chunk.id, t0, t0 + chunk.seconds, tr.text, tr.language, tr.inference_s, gain_db, runs)

    def close(self, error: str | None = None):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if self.flac is not None:
                self.flac.close()
            self.txt.close()
            self.jsonl.close()
            if error:
                self.log.write(error + "\n")
            self.log.close()
            self._write_meta(error)


class Live:
    def __init__(self, cfg: Config, emit: Callable[[str, object], None]):
        self.cfg = cfg
        self.emit = emit
        self.engine: WhisperEngine | None = None
        self.tracker: SpeakerTracker | None = None
        self.vad = SileroVad()
        self.capture: Capture | None = None
        self.session: Session | None = None
        self._session_lock = threading.Lock()
        self._dsp_thread: threading.Thread | None = None
        self._dsp_stop = threading.Event()
        self._history: deque = deque(maxlen=int(HISTORY_S * SR / FRAME))
        self._history_lock = threading.Lock()
        # live settings, changed from the UI while running
        self.gain_db = cfg.input_gain_db
        self.agc = cfg.agc
        self.language = cfg.lang
        self.prompt = cfg.initial_prompt
        self.use_context = cfg.use_context
        self.speakers = cfg.speakers
        self.profile = PROFILES[cfg.compute_profile]
        self._preview_busy = False
        self._preview_done_at = 0.0
        # Whisper worker
        self._cv = threading.Condition()
        self._finals: deque[tuple[Session, Chunk]] = deque()
        self._partial: tuple[Session, Chunk] | None = None
        self._last_rtf: float | None = None
        self._asr = threading.Thread(target=self._asr_loop, name="asr", daemon=True)
        self._asr.start()

    # ------------------------------------------------------------------ logging
    def log(self, line: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        s = self.session
        if s is not None:
            s.write_log(f"{stamp} {line}")
        self.emit("log", f"{stamp} {line}")

    def _fatal(self, where: str):
        tb = traceback.format_exc()
        self.log(f"ERROR in {where}:\n{tb}")
        with self._session_lock:
            s, self.session = self.session, None
        with self._cv:
            self._finals = deque(j for j in self._finals if j[0] is not s)
            self._partial = None
        if s is not None:
            s.close(error=f"{where}: {tb}")
        self.emit("error", (where, tb, s.folder if s else None))

    # ------------------------------------------------------------------ model
    def load_engine(self):
        """Blocking; call from a worker thread. Raises on any problem."""
        path = self.cfg.speaker_model_path
        if path.is_file():
            self.tracker = SpeakerTracker(str(path), self.cfg.speaker_threshold)
            self.log(f"Speaker model loaded: {path.name}")
        elif self.cfg.speakers:
            raise FileNotFoundError(f"Speaker model not found: {path}\n"
                                    "Run setup.ps1 to download it, or set speakers: false in config.yaml.")
        engine = WhisperEngine(self.cfg, self.log)
        self.log(f"Whisper {engine.model_name} loaded on {engine.gpu}")
        self.log(f"whisper.cpp system info: {engine.system_info}")
        # The first decode compiles the GPU shaders (several seconds). Do it on silence now, so
        # the first chunk of the lecture is not delayed; this result is not part of any transcript.
        warm = engine.transcribe(np.zeros(SR, dtype=np.float32), "en", "")
        self.log(f"GPU warm-up decode took {warm.inference_s:.1f} s")
        self.engine = engine

    # ------------------------------------------------------------------ input
    def open_input(self, device: InputDevice, raw: bool):
        self.close_input()
        cap = Capture(device, raw, self.cfg.highpass_hz)
        cap.start()
        self.capture = cap
        self.vad.reset()
        self._dsp_stop.clear()
        self._dsp_thread = threading.Thread(target=self._dsp_loop, args=(cap,), name="dsp", daemon=True)
        self._dsp_thread.start()
        self.log(f"Microphone open: {device.name} ({device.channels} ch, {device.samplerate} Hz, "
                 f"raw capture {'on' if raw else 'off'})")

    def close_input(self):
        if self._dsp_thread is not None:
            self._dsp_stop.set()
            if threading.current_thread() is not self._dsp_thread:
                self._dsp_thread.join()
            self._dsp_thread = None
        if self.capture is not None:
            self.capture.close()
            self.capture = None

    def play_last(self, seconds: float):
        with self._history_lock:
            frames = list(self._history)[-int(seconds * SR / FRAME):]
        if not frames:
            raise RuntimeError("Nothing has been captured yet.")
        sd.play(np.concatenate(frames), SR)

    def _dsp_loop(self, cap: Capture):
        try:
            acc_sq, acc_n, acc_peak, acc_prob, acc_clip = 0.0, 0, 0.0, 0.0, 0
            last_emit = time.monotonic()
            while not self._dsp_stop.is_set():
                try:
                    block, overflow = cap.blocks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if overflow:
                    self._on_overflow()
                gain = 10 ** (self.gain_db / 20)
                for frame, clipped in cap.process(block, gain):
                    prob = self.vad(frame)
                    with self._history_lock:
                        self._history.append(frame)
                    acc_sq += float(np.sum(np.square(frame, dtype=np.float64)))
                    acc_n += len(frame)
                    acc_peak = max(acc_peak, float(np.max(np.abs(frame))))
                    acc_prob = max(acc_prob, prob)
                    acc_clip += clipped
                    with self._session_lock:
                        s = self.session
                        if s is not None:
                            if s.flac is not None:
                                s.flac.write(frame)
                            for chunk in s.segmenter.push(frame, prob):
                                self._submit_final(s, chunk)
                now = time.monotonic()
                if now - last_emit >= LEVEL_EVERY_S and acc_n:
                    rms = math.sqrt(acc_sq / acc_n)
                    self.emit("level", Level(20 * math.log10(rms) if rms > 0 else -math.inf,
                                             20 * math.log10(acc_peak) if acc_peak > 0 else -math.inf,
                                             acc_prob, acc_clip))
                    acc_sq, acc_n, acc_peak, acc_prob, acc_clip = 0.0, 0, 0.0, 0.0, 0
                    last_emit = now
                self._maybe_preview(now)
        except Exception:
            self._fatal("audio")
            self.close_input()

    def _on_overflow(self):
        s = self.session
        t = hms(s.segmenter.pos / SR) if s else "--:--:--"
        self.log(f"Audio input overflow at {t}: the audio device dropped samples.")
        if s is not None:
            s.overflows += 1
            s.write_event({"event": "input_overflow", "at": round(s.segmenter.pos / SR, 3)})
        self.emit("warning", f"Audio input overflow at {t}: some audio was lost.")
        self._emit_status()

    # ------------------------------------------------------------------ session
    def start_session(self, module: Module) -> Path:
        if self.engine is None:
            raise RuntimeError("The Whisper model is not loaded.")
        if self.capture is None:
            raise RuntimeError("No microphone is open.")
        folder = module.path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        s = Session(self.cfg, module.name, folder, self.capture.device, self.cfg.raw_capture, self.engine.gpu)
        if self.tracker is not None:
            self.tracker.reset()  # speaker numbers start again for every lecture
        self._last_rtf = None
        with self._session_lock:
            self.session = s
        self.log(f"Session started ({self.cfg.segmentation} mode): {folder}")
        self._emit_status()
        return folder

    def stop_session(self):
        """Stop taking audio; Whisper finishes the chunks already cut, then "finished" is emitted."""
        with self._session_lock:
            s, self.session = self.session, None
            if s is None:
                return
            s.stopping = True
            for chunk in s.segmenter.flush():
                self._submit_final(s, chunk)
        self.log("Stopping: finishing the remaining audio.")
        with self._cv:
            done = s.pending == 0
        if done:
            self._finish(s)

    def _finish(self, s: Session):
        s.close()
        self.log(f"Session saved: {s.folder}")
        self.emit("finished", s.folder)

    # ------------------------------------------------------------------ whisper worker
    def _submit_final(self, s: Session, chunk: Chunk):
        with self._cv:
            s.pending += 1
            self._finals.append((s, chunk))
            self._cv.notify()
        self._emit_status()

    def _maybe_preview(self, now: float):
        s = self.session
        every = self.profile["preview_every"]
        if s is None or every is None or self._preview_busy or now - self._preview_done_at < every:
            return
        with self._session_lock:
            chunk = s.segmenter.current()
        if chunk is None or chunk.seconds < MIN_WHISPER_S:
            return
        with self._cv:
            self._preview_busy = True
            self._partial = (s, chunk)
            self._cv.notify()

    def _emit_status(self):
        with self._cv:
            backlog = sum(c.seconds for _, c in self._finals)
        s = self.session
        self.emit("status", Status(backlog, self._last_rtf, s.overflows if s else 0, s.chunks if s else 0))

    def _prompt(self, s: Session) -> str:
        parts = [self.prompt] if self.prompt else []
        if self.use_context and s.text:
            tail = s.text[-CONTEXT_CHARS:]
            if len(s.text) > CONTEXT_CHARS and " " in tail:
                tail = tail[tail.index(" "):]  # start the prompt on a word boundary
            parts.append(tail)
        return " ".join(parts)

    def _agc_gain_db(self, chunk: Chunk) -> float:
        if not self.agc:
            return 0.0
        frames = chunk.audio.reshape(-1, FRAME)
        speech = chunk.probs >= self.cfg.vad_threshold
        level = dbfs((frames[speech] if speech.any() else frames).ravel())
        if level == -math.inf:
            return self.cfg.agc_max_gain_db
        return min(self.cfg.agc_target_dbfs - level, self.cfg.agc_max_gain_db)

    def _whisper(self, s: Session, chunk: Chunk, preview: bool) -> tuple[Transcription | None, float]:
        gain_db = self._agc_gain_db(chunk)
        x = chunk.audio * np.float32(10 ** (gain_db / 20))
        if len(x) < MIN_WHISPER_S * SR:
            x = np.concatenate([x, np.zeros(int(MIN_WHISPER_S * SR) - len(x), dtype=np.float32)])
        # a preview gives way as soon as a finished chunk is waiting
        abort = (lambda: bool(self._finals)) if preview else None
        greedy = self.profile["preview_greedy" if preview else "final_greedy"]
        return self.engine.transcribe(x, self.language, self._prompt(s), abort, greedy=greedy), gain_db

    def _asr_loop(self):
        while True:
            with self._cv:
                while not self._finals and self._partial is None:
                    self._cv.wait()
                if self._finals:
                    kind, (s, chunk) = "final", self._finals.popleft()
                else:
                    kind, (s, chunk) = "partial", self._partial
                    self._partial = None
            try:
                self._process(kind, s, chunk)
            finally:
                if kind == "partial":
                    self._preview_done_at = time.monotonic()
                    self._preview_busy = False

    def _process(self, kind: str, s: Session, chunk: Chunk):
        if s.closed or (kind == "partial" and s.stopping):
            return
        try:
            tr, gain_db = self._whisper(s, chunk, preview=kind == "partial")
        except Exception:
            self._fatal("whisper")
            return
        if kind == "partial":
            if tr is None:
                self.log("Preview stopped early so a finished chunk could go first "
                         "(the whisper.cpp 'failed to encode/decode' line above is that stop).")
            else:
                self.emit("partial", Partial(chunk.id, tr.text))
            return
        speakers: list[int | None] = [None] * len(tr.segments)
        if self.speakers and self.tracker is not None and tr.segments:
            try:
                speakers = self.tracker.label_segments(chunk.audio, tr.segments)
            except Exception:
                self._fatal("speakers")
                return
        final = s.write_final(chunk, tr, gain_db, speakers)
        self._last_rtf = tr.inference_s / chunk.seconds
        self.emit("final", final)
        with self._cv:
            s.pending -= 1
            done = s.stopping and s.pending == 0
        self._emit_status()
        if done:
            self._finish(s)

    def shutdown(self):
        self.close_input()
        if self.engine is not None:
            self.engine.close()
            self.engine = None
