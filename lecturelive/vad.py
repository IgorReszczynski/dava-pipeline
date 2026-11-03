"""Silero VAD (the ONNX model shipped in the silero-vad package), run with onnxruntime.

The silero_vad Python module imports torch, so it is never imported: only its packaged
model file is used. The model takes 512-sample frames at 16 kHz with the previous 64
samples prepended as context, and returns a speech probability per frame.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio import FRAME, SR

CONTEXT = 64


def model_path() -> Path:
    if getattr(sys, "frozen", False):   # packaged app: bundled by packaging/build.ps1
        path = Path(sys._MEIPASS) / "silero_vad" / "data" / "silero_vad.onnx"
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD model missing from the app bundle: {path}")
        return path
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("The silero-vad package is not installed (pip install --no-deps silero-vad).")
    path = Path(spec.submodule_search_locations[0]) / "data" / "silero_vad.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Silero VAD model not found at {path}")
    return path


class SileroVad:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path()), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self._sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME,):
            raise ValueError(f"VAD frame must be {FRAME} samples, got {frame.shape}")
        x = np.concatenate([self._context, frame[None, :].astype(np.float32)], axis=1)
        out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT:]
        return float(out[0, 0])
