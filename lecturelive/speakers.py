"""Speaker labels: WeSpeaker ResNet34 voice embeddings (ONNX, CPU) and online clustering.

Every Whisper segment gets an embedding; it joins the closest known voice if the cosine
similarity reaches `threshold`, otherwise it starts a new voice ("Speaker 1", "Speaker 2",
...). This labels the text, it never changes it. With one far-away microphone it is a best
effort: a lecturer is usually tracked well, short questions from the room less so.

Features are Kaldi filterbanks computed exactly like torchaudio.compliance.kaldi.fbank
(80 bins, 25 ms window, 10 ms shift, povey window, no dither), on int16-scaled samples
with the mean subtracted, which is what WeSpeaker models are trained on.
"""
from __future__ import annotations

import math

import numpy as np
import onnxruntime as ort

from .audio import SR

MIN_EMBED_S = 1.5        # shorter segments are widened (within the chunk) to this before embedding
WIN, HOP, NFFT, NMEL = 400, 160, 512, 80


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_banks() -> np.ndarray:
    low, high = _mel(20.0), _mel(SR / 2)
    delta = (high - low) / (NMEL + 1)
    mel_f = _mel(np.arange(NFFT // 2) * (SR / NFFT))
    left = low + np.arange(NMEL)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    up = (mel_f - left) / (center - left)
    down = (right - mel_f) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)  # zero weight for the Nyquist bin


_BANKS = _mel_banks()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / (WIN - 1))) ** 0.85
_EPS = np.finfo(np.float32).eps


def fbank(wave: np.ndarray) -> np.ndarray:
    """(samples,) float in [-1, 1] at 16 kHz -> (frames, 80) log mel filterbank, mean-normalized."""
    x = wave.astype(np.float64) * 32768.0
    n = 1 + (len(x) - WIN) // HOP
    if n < 1:
        raise ValueError(f"audio too short for a speaker embedding: {len(x) / SR:.2f} s")
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP][:n].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] -= 0.97 * frames[:, 0]
    frames *= _WINDOW
    power = np.abs(np.fft.rfft(frames, n=NFFT)) ** 2
    feats = np.log(np.maximum(power.astype(np.float32) @ _BANKS.T, _EPS))
    return feats - feats.mean(axis=0, keepdims=True)


class SpeakerTracker:
    def __init__(self, model_path: str, threshold: float):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.threshold = threshold
        self.centroids: list[np.ndarray] = []

    def reset(self):
        self.centroids = []

    def embed(self, audio: np.ndarray) -> np.ndarray:
        e = self.session.run(None, {self.input: fbank(audio)[None].astype(np.float32)})[0][0]
        norm = float(np.linalg.norm(e))
        if not math.isfinite(norm) or norm == 0:
            raise RuntimeError("the speaker model returned an empty embedding")
        return e / norm

    def assign(self, audio: np.ndarray, learn: bool) -> tuple[int, float]:
        """Speaker number (1-based) and its similarity. `learn` updates the voice's profile."""
        e = self.embed(audio)
        if self.centroids:
            sims = [float(e @ (c / np.linalg.norm(c))) for c in self.centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                if learn:
                    self.centroids[best] = self.centroids[best] + e
                return best + 1, sims[best]
        self.centroids.append(e.copy())
        return len(self.centroids), 1.0

    def label_segments(self, chunk_audio: np.ndarray, segments) -> list[int]:
        """One speaker number per Whisper segment of a chunk (times in seconds from chunk start)."""
        out = []
        total = len(chunk_audio)
        need = int(MIN_EMBED_S * SR)
        for s in segments:
            a = max(0, min(total, int(s.start * SR)))
            b = max(a, min(total, int(s.end * SR)))
            learn = b - a >= need
            if b - a < need:  # widen around the middle, staying inside the chunk
                mid = (a + b) // 2
                a = max(0, min(mid - need // 2, total - need))
                b = min(total, a + need)
            out.append(self.assign(chunk_audio[a:b], learn)[0])
        return out
