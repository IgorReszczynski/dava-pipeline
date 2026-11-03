"""Microphone capture and the signal chain in front of VAD and Whisper.

Capture goes through WASAPI at the device's own sample rate. With raw capture on, the
stream asks Windows for AUDCLNT_STREAMOPTIONS_RAW, which bypasses the driver's mic
effects (array beamforming, noise suppression). Those effects are tuned for the person
at the keyboard and can treat a lecturer at the front of the room as noise.

Chain, per block: downmix to mono -> resample to 16 kHz -> high-pass -> manual gain ->
clip to [-1, 1] (counted, shown on the meter).
"""
from __future__ import annotations

import math
import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soxr

SR = 16000          # Whisper and Silero VAD both take 16 kHz mono
FRAME = 512         # 32 ms at 16 kHz, the Silero VAD frame size


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    samplerate: int


def _wasapi_index() -> int:
    for i, api in enumerate(sd.query_hostapis()):
        if api["name"] == "Windows WASAPI":
            return i
    raise RuntimeError("The Windows WASAPI audio API is not available, so no microphone can be opened.")


def rescan():
    """PortAudio only enumerates devices when it starts; restart it to see newly plugged mics.
    Every stream must be closed first."""
    sd._terminate()
    sd._initialize()


def list_input_devices() -> tuple[list[InputDevice], int]:
    """WASAPI capture devices, and the index of the Windows default input."""
    api = _wasapi_index()
    devices = [
        InputDevice(i, d["name"], d["max_input_channels"], int(d["default_samplerate"]))
        for i, d in enumerate(sd.query_devices())
        if d["hostapi"] == api and d["max_input_channels"] > 0
    ]
    if not devices:
        raise RuntimeError("No microphone found. Plug one in or enable it in Windows sound settings.")
    return devices, sd.query_hostapis(api)["default_input_device"]


class _WasapiRaw(sd.WasapiSettings):
    """WasapiSettings with eStreamOptionRaw set. PortAudio requires struct version 1 and
    applies streamOption through IAudioClient2::SetClientProperties (Windows 8.1+)."""

    def __init__(self):
        super().__init__()
        self._streaminfo.streamOption = sd._lib.eStreamOptionRaw


class Biquad:
    """2nd-order Butterworth high-pass, transposed direct form II, state kept across blocks."""

    def __init__(self, cutoff_hz: float, sr: int = SR):
        w0 = 2 * math.pi * cutoff_hz / sr
        alpha = math.sin(w0) / math.sqrt(2)
        cos_w0 = math.cos(w0)
        a0 = 1 + alpha
        self.b0 = (1 + cos_w0) / 2 / a0
        self.b1 = -(1 + cos_w0) / a0
        self.b2 = self.b0
        self.a1 = -2 * cos_w0 / a0
        self.a2 = (1 - alpha) / a0
        self.z1 = 0.0
        self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        out = []
        append = out.append
        for v in x.tolist():
            y = b0 * v + z1
            z1 = b1 * v - a1 * y + z2
            z2 = b2 * v - a2 * y
            append(y)
        self.z1, self.z2 = z1, z2
        return np.asarray(out, dtype=np.float32)


class Capture:
    """An open microphone. Device blocks land in `blocks`; `process()` turns them into 16 kHz frames."""

    def __init__(self, device: InputDevice, raw: bool, highpass_hz: float):
        self.device = device
        self.blocks: queue.Queue = queue.Queue()
        self.overflows = 0
        self._resampler = soxr.ResampleStream(device.samplerate, SR, 1, dtype="float32", quality="HQ")
        self._hp = Biquad(highpass_hz) if highpass_hz > 0 else None
        self._pending = np.zeros(0, dtype=np.float32)
        self.stream = sd.InputStream(
            device=device.index,
            samplerate=device.samplerate,
            channels=device.channels,
            dtype="float32",
            latency="high",
            extra_settings=_WasapiRaw() if raw else sd.WasapiSettings(),
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            self.overflows += 1
        self.blocks.put((indata.copy(), bool(status.input_overflow)))

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()

    def process(self, block: np.ndarray, gain: float) -> list[tuple[np.ndarray, int]]:
        """One device block -> list of (512-sample frame, clipped sample count)."""
        mono = block.mean(axis=1, dtype=np.float32)
        x = self._resampler.resample_chunk(mono)
        if self._hp is not None:
            x = self._hp.process(x)
        x = np.concatenate([self._pending, x * np.float32(gain)])
        n = len(x) // FRAME * FRAME
        self._pending = x[n:]
        frames = []
        for i in range(0, n, FRAME):
            f = x[i:i + FRAME]
            clipped = int(np.count_nonzero(np.abs(f) > 1.0))
            if clipped:
                f = np.clip(f, -1.0, 1.0)
            frames.append((f, clipped))
        return frames


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * math.log10(rms) if rms > 0 else -math.inf
