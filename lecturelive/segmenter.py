"""Cut the stream of 32 ms frames into chunks for Whisper.

VadSegmenter    -- a chunk starts when Silero VAD hears speech (with a little audio
                   from before it) and ends after a pause. Silence between chunks is
                   not sent to Whisper.
ContinuousSegmenter -- every frame goes into a chunk; chunks are back to back, so
                   every second of audio reaches Whisper.

A chunk that grows too long is cut at its quietest frame near the end, and the rest
starts the next chunk, so no audio is lost at a cut. Each frame is sent in exactly one
chunk.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import FRAME, SR

FRAME_S = FRAME / SR


@dataclass
class Chunk:
    id: int
    start: int            # first sample, counted from the start of the session (16 kHz)
    audio: np.ndarray
    probs: np.ndarray     # VAD speech probability for every frame of the chunk

    @property
    def seconds(self) -> float:
        return len(self.audio) / SR


class _Segmenter:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.probs: list[float] = []
        self.start = 0
        self.pos = 0          # sample index of the next frame to arrive
        self.next_id = 0

    def _emit(self, n: int) -> Chunk:
        chunk = Chunk(self.next_id, self.start, np.concatenate(self.frames[:n]),
                      np.asarray(self.probs[:n], dtype=np.float32))
        self.next_id += 1
        self.frames = self.frames[n:]
        self.probs = self.probs[n:]
        self.start += n * FRAME
        return chunk

    def _quietest_cut(self, search_frames: int) -> int:
        """Number of frames to emit: up to and including the quietest frame among the last
        `search_frames` (lowest VAD probability, then lowest energy)."""
        lo = max(1, len(self.frames) - search_frames)
        probs = np.asarray(self.probs[lo:])
        energy = np.array([float(np.mean(np.square(f))) for f in self.frames[lo:]])
        quiet = probs <= probs.min() + 0.05
        idx = int(np.argmin(np.where(quiet, energy, np.inf)))
        return lo + idx + 1

    def current(self) -> Chunk | None:
        """The unfinished chunk (for the live preview), or None."""
        if not self.frames:
            return None
        return Chunk(self.next_id, self.start, np.concatenate(self.frames),
                     np.asarray(self.probs, dtype=np.float32))

    def flush(self) -> list[Chunk]:
        return [self._emit(len(self.frames))] if self.frames else []


class VadSegmenter(_Segmenter):
    def __init__(self, threshold: float, min_silence_ms: int, speech_pad_ms: int, max_chunk_s: float):
        super().__init__()
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)  # hysteresis, as in Silero's VADIterator
        self.min_silence = max(1, math.ceil(min_silence_ms / 1000 / FRAME_S))
        self.max_frames = max(2, int(max_chunk_s / FRAME_S))
        self.preroll: deque = deque(maxlen=max(0, math.ceil(speech_pad_ms / 1000 / FRAME_S)))
        self.silence = 0

    def push(self, frame: np.ndarray, prob: float) -> list[Chunk]:
        out = []
        if self.frames:
            self.frames.append(frame)
            self.probs.append(prob)
            if prob < self.neg_threshold:
                self.silence += 1
            elif prob >= self.threshold:
                self.silence = 0
            if self.silence >= self.min_silence:
                out.append(self._emit(len(self.frames)))
                self.silence = 0
            elif len(self.frames) >= self.max_frames:
                out.append(self._emit(self._quietest_cut(int(5 / FRAME_S))))
                self.silence = 0
        elif prob >= self.threshold:
            self.start = self.pos - len(self.preroll) * FRAME
            self.frames = [f for f, _ in self.preroll] + [frame]
            self.probs = [p for _, p in self.preroll] + [prob]
            self.preroll.clear()
            self.silence = 0
        elif self.preroll.maxlen:
            self.preroll.append((frame, prob))
        self.pos += FRAME
        return out


class ContinuousSegmenter(_Segmenter):
    def __init__(self, chunk_s: float):
        super().__init__()
        self.chunk_frames = max(2, int(chunk_s / FRAME_S))

    def push(self, frame: np.ndarray, prob: float) -> list[Chunk]:
        if not self.frames:
            self.start = self.pos
        self.frames.append(frame)
        self.probs.append(prob)
        self.pos += FRAME
        if len(self.frames) >= self.chunk_frames:
            return [self._emit(self._quietest_cut(int(2 / FRAME_S)))]
        return []
